//! Constant-time modular exponentiation for secret operands (`Nat::modpow_sec`,
//! `Int::modpow_sec`).
//!
//! The sequence of operations and memory accesses depends only on the *sizes*
//! (limb counts) of the base, exponent and modulus, never on their values:
//!
//! - `R mod m` and `R² mod m` come from a fixed number of masked doublings
//!   and Montgomery squarings (counts derived from the limb count), and the
//!   base enters Montgomery form through Montgomery multiplications by `R²`;
//! - Montgomery multiplication and squaring (separated product + REDC, see
//!   `crate::mont`) run fixed loops and end with a masked conditional
//!   subtraction;
//! - the exponent is consumed in fixed `w`-bit windows over all of its limbs
//!   (leading zero bits included), always squaring `w` times and always
//!   multiplying by a table entry fetched by scanning the whole table with
//!   masks, so neither the window value nor a zero window is observable.
//!
//! Masks go through [`core::hint::black_box`] to discourage the optimizer from
//! turning selects back into branches. This is best-effort hardening in
//! portable safe Rust, not a formally verified guarantee.

use alloc::vec;
use alloc::vec::Vec;

use crate::int::Int;
use crate::limb::{Limb, adc, sbb};
use crate::mont::{self, mask_from_bit};
use crate::nat::Nat;

/// All-ones when `a == b`, zero otherwise, without a data-dependent branch.
#[inline(always)]
fn mask_eq(a: Limb, b: Limb) -> Limb {
    let x = a ^ b;
    // (x | −x) has its top bit set iff x ≠ 0.
    mask_from_bit(((x | x.wrapping_neg()) >> 63) ^ 1)
}

/// Constant-flow Montgomery arithmetic modulo an odd `m` (`n = m.len()` limbs,
/// non-zero top limb).
struct Ctx<'a> {
    m: &'a [Limb],
    mont: mont::Mont<'a>,
    /// `R mod m`, the Montgomery form of 1 (`R = 2^(64n)`).
    one: Vec<Limb>,
    /// `R² mod m`, converts into Montgomery form.
    r2: Vec<Limb>,
}

impl<'a> Ctx<'a> {
    fn new(m: &'a [Limb]) -> Ctx<'a> {
        let n = m.len();
        debug_assert!(n > 0 && m[0] & 1 == 1);
        let mut ctx = Ctx {
            m,
            mont: mont::Mont::new(m),
            one: vec![0; n],
            r2: vec![0; n],
        };
        // R mod m: start from 2^(64(n−1)) ≤ m, reduce once, then double 64
        // times (each doubling reduced by one masked subtraction).
        let mut x = vec![0 as Limb; n];
        let mut d = vec![0 as Limb; n];
        x[n - 1] = 1;
        mont::final_sub(&mut d, &x, 0, m);
        x.copy_from_slice(&d);
        for _ in 0..64 {
            ctx.double(&mut x, &mut d);
        }
        ctx.one.copy_from_slice(&x);
        // R² mod m is the Montgomery form of 2^(64n) = 2^(k·2^j) with k odd:
        // k doublings of R mod m give the Montgomery form of 2^k, and j
        // Montgomery squarings raise it to 2^(k·2^j). Both counts depend on
        // `n` only.
        let bits = 64 * n;
        let j = bits.trailing_zeros();
        let k = bits >> j;
        for _ in 0..k {
            ctx.double(&mut x, &mut d);
        }
        for _ in 0..j {
            ctx.sqr(&mut d, &x);
            x.copy_from_slice(&d);
        }
        ctx.r2.copy_from_slice(&x);
        ctx
    }

    /// `x = 2x mod m` for `x < m`; `d` is `n` limbs of scratch.
    fn double(&self, x: &mut [Limb], d: &mut [Limb]) {
        let mut carry = 0;
        for l in x.iter_mut() {
            let next = *l >> 63;
            *l = (*l << 1) | carry;
            carry = next;
        }
        mont::final_sub(d, x, carry, self.m);
        x.copy_from_slice(d);
    }

    /// `out = a·b·R⁻¹ mod m` (`a·b < m·R`).
    #[inline]
    fn mul(&mut self, out: &mut [Limb], a: &[Limb], b: &[Limb]) {
        self.mont.mul(out, a, b);
    }

    /// `out = a²·R⁻¹ mod m` (`a < m`).
    #[inline]
    fn sqr(&mut self, out: &mut [Limb], a: &[Limb]) {
        self.mont.sqr(out, a);
    }

    /// Montgomery form `x·R mod m` of an arbitrary-length `x`, processed in
    /// `n`-limb chunks from the top (Horner in the Montgomery domain: each
    /// step multiplies the running value by `R` and adds the next chunk).
    fn enter(&mut self, x: &[Limb]) -> Vec<Limb> {
        let n = self.m.len();
        let chunks = x.len().div_ceil(n).max(1);
        let r2 = self.r2.clone();
        let mut acc = vec![0 as Limb; n];
        let mut chunk = vec![0 as Limb; n];
        let mut cm = vec![0 as Limb; n];
        let mut tmp = vec![0 as Limb; n];
        for c in (0..chunks).rev() {
            let lo = (c * n).min(x.len());
            let hi = ((c + 1) * n).min(x.len());
            chunk.fill(0);
            chunk[..hi - lo].copy_from_slice(&x[lo..hi]);
            // chunk < R and r2 < m, so the product is < m·R.
            self.mul(&mut cm, &chunk, &r2);
            if c + 1 < chunks {
                tmp.copy_from_slice(&acc);
                self.mul(&mut acc, &tmp, &r2);
            }
            // acc + cm < 2m.
            let mut carry = 0;
            for (t, (&a, &b)) in tmp.iter_mut().zip(acc.iter().zip(&cm)) {
                let (s, c) = adc(a, b, carry);
                *t = s;
                carry = c;
            }
            mont::final_sub(&mut acc, &tmp, carry, self.m);
        }
        acc
    }

    /// Leaves Montgomery form: `x·R⁻¹ mod m`.
    fn leave(&mut self, x: &[Limb]) -> Vec<Limb> {
        let mut out = vec![0 as Limb; self.m.len()];
        self.mont.leave(&mut out, x);
        out
    }
}

/// Window width for an exponent of `bits` bits (derived from its limb count).
fn window_bits(bits: usize) -> usize {
    match bits {
        0..=512 => 4,
        513..=1024 => 5,
        _ => 6,
    }
}

/// Constant-time table lookup: `sel = Σᵢ table[i]·[masks[i] = all-ones]`
/// over all `masks.len()` entries of `sel.len()` limbs each. Every entry is
/// read; limbs are gathered four at a time so the partial results stay in
/// registers across the scan.
fn ct_select(sel: &mut [Limb], table: &[Limb], masks: &[Limb]) {
    let n = sel.len();
    debug_assert_eq!(table.len(), n * masks.len());
    let mut j = 0;
    while j + 4 <= n {
        let mut s = [0 as Limb; 4];
        for (entry, &mk) in table.chunks_exact(n).zip(masks) {
            let e = &entry[j..j + 4];
            for k in 0..4 {
                s[k] |= e[k] & mk;
            }
        }
        sel[j..j + 4].copy_from_slice(&s);
        j += 4;
    }
    for (jj, out) in sel.iter_mut().enumerate().skip(j) {
        let mut s = 0;
        for (entry, &mk) in table.chunks_exact(n).zip(masks) {
            s |= entry[jj] & mk;
        }
        *out = s;
    }
}

/// `b^exp` in Montgomery form, for `b` already in Montgomery form, in
/// constant time with respect to the values (see the module docs).
fn pow_mont(ctx: &mut Ctx<'_>, b: &[Limb], exp: &[Limb]) -> Vec<Limb> {
    let n = ctx.m.len();
    let bits = exp.len() * 64;
    if bits == 0 {
        return ctx.one.clone();
    }
    let w = window_bits(bits);
    let size = 1usize << w;
    // Table of b^i (Montgomery form), i in 0..2^w: even entries by squaring.
    let mut table = vec![0 as Limb; size * n];
    table[..n].copy_from_slice(&ctx.one);
    table[n..2 * n].copy_from_slice(b);
    for i in 2..size {
        let (done, rest) = table.split_at_mut(i * n);
        if i % 2 == 0 {
            ctx.sqr(&mut rest[..n], &done[(i / 2) * n..(i / 2 + 1) * n]);
        } else {
            ctx.mul(&mut rest[..n], &done[(i - 1) * n..], b);
        }
    }

    let mut acc = vec![0 as Limb; n];
    let mut tmp = vec![0 as Limb; n];
    let mut sel = vec![0 as Limb; n];
    let mut masks = vec![0 as Limb; size];
    // Windows from the top; the first one takes the leftover high bits.
    let mut pos = bits;
    let mut width = if bits.is_multiple_of(w) { w } else { bits % w };
    let mut first = true;
    while pos > 0 {
        let lo = pos - width;
        // Window value: `width` bits of exp starting at `lo` (fixed shifts).
        let mut idx: Limb = 0;
        for k in (lo..pos).rev() {
            idx = (idx << 1) | ((exp[k / 64] >> (k % 64)) & 1);
        }
        for (i, mk) in masks.iter_mut().enumerate() {
            *mk = mask_eq(i as Limb, idx);
        }
        ct_select(&mut sel, &table, &masks);
        if first {
            // The accumulator starts at 1, so the first window needs neither
            // squarings nor a multiply (a fixed, size-derived shortcut).
            acc.copy_from_slice(&sel);
            first = false;
        } else {
            for _ in 0..width {
                ctx.sqr(&mut tmp, &acc);
                core::mem::swap(&mut acc, &mut tmp);
            }
            ctx.mul(&mut tmp, &acc, &sel);
            core::mem::swap(&mut acc, &mut tmp);
        }
        pos = lo;
        width = w;
    }
    acc
}

/// `base^exp mod m` in constant time with respect to the values (see the
/// module docs). `m` must be odd with a non-zero top limb; `base` and `exp`
/// are little-endian limbs of any length. Returns `n` limbs (not normalized).
fn modpow_sec_limbs(base: &[Limb], exp: &[Limb], m: &[Limb]) -> Vec<Limb> {
    let mut ctx = Ctx::new(m);
    let b = ctx.enter(base);
    let acc = pow_mont(&mut ctx, &b, exp);
    ctx.leave(&acc)
}

/// `(m − x) mod m` for `x < m`, branch-free (maps `0` to `0`).
fn ct_neg_mod(x: &[Limb], m: &[Limb]) -> Vec<Limb> {
    let n = m.len();
    let mut out = vec![0 as Limb; n];
    let mut borrow = 0;
    let mut nz = 0;
    for j in 0..n {
        let (v, b) = sbb(m[j], x[j], borrow);
        out[j] = v;
        borrow = b;
        nz |= x[j];
    }
    // x == 0 gives m, which must become 0.
    let keep = !mask_eq(nz, 0);
    for o in out.iter_mut() {
        *o &= keep;
    }
    out
}

impl Nat {
    /// Returns `self^exp mod modulus`, in time and with memory accesses that
    /// depend only on the limb counts of `self`, `exp` and `modulus`, not on
    /// their values — for secret exponents (RSA, Diffie–Hellman), like GMP's
    /// `mpz_powm_sec`. The modulus must be odd. `exp = 0` gives `1 mod modulus`.
    ///
    /// Slower than [`Nat::modpow`] (every window multiplies, and the table is
    /// scanned in full on every lookup). The returned `Nat` is normalized, so
    /// its own limb count is not hidden.
    ///
    /// # Panics
    ///
    /// If `modulus` is even or zero.
    pub fn modpow_sec(&self, exp: &Nat, modulus: &Nat) -> Nat {
        assert!(!modulus.is_even(), "modpow_sec: modulus must be odd");
        Nat::from_limb_vec(modpow_sec_limbs(
            self.as_limbs(),
            exp.as_limbs(),
            modulus.as_limbs(),
        ))
    }
}

impl Int {
    /// Returns `self^exp mod |modulus|` in `[0, |modulus|)`, in constant time
    /// with respect to the values of the base, exponent and modulus (only
    /// their limb counts, and the base's sign, are not hidden). See
    /// [`Nat::modpow_sec`].
    ///
    /// # Panics
    ///
    /// If `exp` is negative or `modulus` is even or zero.
    pub fn modpow_sec(&self, exp: &Int, modulus: &Int) -> Int {
        assert!(!exp.is_negative(), "modpow_sec: negative exponent");
        let m = modulus.magnitude_ref();
        assert!(!m.is_even(), "modpow_sec: modulus must be odd");
        let m = m.as_limbs();
        let mut ctx = Ctx::new(m);
        // ±|self| in Montgomery form (negation is linear, so it commutes).
        let pos = ctx.enter(self.magnitude_ref().as_limbs());
        let neg = ct_neg_mod(&pos, m);
        let mask = mask_from_bit(self.is_negative() as Limb);
        let base: Vec<Limb> = pos
            .iter()
            .zip(&neg)
            .map(|(&p, &q)| (q & mask) | (p & !mask))
            .collect();
        let acc = pow_mont(&mut ctx, &base, exp.magnitude_ref().as_limbs());
        Int::from(Nat::from_limb_vec(ctx.leave(&acc)))
    }
}
