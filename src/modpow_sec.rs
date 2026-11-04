//! Constant-time modular exponentiation for secret operands (`Nat::modpow_sec`,
//! `Int::modpow_sec`).
//!
//! The sequence of operations and memory accesses depends only on the *sizes*
//! (limb counts) of the base, exponent and modulus, never on their values:
//!
//! - every reduction (`x mod m`, `R mod m`, `x·R mod m`) is a bit-serial
//!   shift-and-subtract over a fixed number of bits, with the subtraction
//!   selected by a mask rather than a branch;
//! - Montgomery multiplication (CIOS) runs a fixed loop and ends with a masked
//!   conditional subtraction;
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
use core::hint::black_box;

use crate::int::Int;
use crate::limb::{Limb, adc, mac, sbb};
use crate::nat::Nat;

/// All-ones when `bit == 1`, zero when `bit == 0` (`bit` must be 0 or 1).
#[inline(always)]
fn mask_from_bit(bit: Limb) -> Limb {
    black_box(bit).wrapping_neg()
}

/// All-ones when `a == b`, zero otherwise, without a data-dependent branch.
#[inline(always)]
fn mask_eq(a: Limb, b: Limb) -> Limb {
    let x = a ^ b;
    // (x | −x) has its top bit set iff x ≠ 0.
    mask_from_bit(((x | x.wrapping_neg()) >> 63) ^ 1)
}

/// `−m⁻¹ mod 2^64` for odd `m0` (fixed Newton iteration count).
fn neg_inv_mod_2_64(m0: Limb) -> Limb {
    let mut y = m0.wrapping_mul(3) ^ 2;
    for _ in 0..4 {
        y = y.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(y)));
    }
    y.wrapping_neg()
}

/// `x mod m` (`m` odd, `n = m.len()` limbs, top limb may be anything non-zero
/// overall) by bit-serial shift-and-subtract over all `64·x.len()` bits.
/// Returns `n` limbs.
fn ct_mod(x: &[Limb], m: &[Limb]) -> Vec<Limb> {
    let n = m.len();
    // r < 2m < 2^(64n+1): one extra limb.
    let mut r = vec![0 as Limb; n + 1];
    let mut d = vec![0 as Limb; n + 1];
    for i in (0..x.len() * 64).rev() {
        let bit = (x[i / 64] >> (i % 64)) & 1;
        // r = 2r + bit.
        let mut carry = bit;
        for l in r.iter_mut() {
            let next = *l >> 63;
            *l = (*l << 1) | carry;
            carry = next;
        }
        // d = r − m; keep it iff no borrow.
        let mut borrow = 0;
        for j in 0..=n {
            let mj = if j < n { m[j] } else { 0 };
            let (v, b) = sbb(r[j], mj, borrow);
            d[j] = v;
            borrow = b;
        }
        let keep = mask_from_bit(borrow ^ 1);
        for (rl, &dl) in r.iter_mut().zip(&d) {
            *rl = (dl & keep) | (*rl & !keep);
        }
    }
    r.truncate(n);
    r
}

/// Montgomery product `a·b·R⁻¹ mod m` (`R = 2^(64n)`), CIOS, with `a, b < m`;
/// writes `n` limbs to `out`. `t` is `n + 2` limbs of scratch.
fn mont_mul(out: &mut [Limb], a: &[Limb], b: &[Limb], m: &[Limb], minv: Limb, t: &mut [Limb]) {
    let n = m.len();
    t.fill(0);
    for &bi in b.iter().take(n) {
        let mut carry = 0;
        for j in 0..n {
            let (lo, hi) = mac(t[j], a[j], bi, carry);
            t[j] = lo;
            carry = hi;
        }
        let (s, c) = adc(t[n], carry, 0);
        t[n] = s;
        t[n + 1] = c;
        let q = t[0].wrapping_mul(minv);
        let (_, mut carry) = mac(t[0], m[0], q, 0);
        for j in 1..n {
            let (lo, hi) = mac(t[j], m[j], q, carry);
            t[j - 1] = lo;
            carry = hi;
        }
        let (s, c) = adc(t[n], carry, 0);
        t[n - 1] = s;
        t[n] = t[n + 1] + c;
    }
    // T = t[n]·R + t[..n] < 2m; subtract m when T ≥ m.
    let mut borrow = 0;
    for j in 0..n {
        let (v, b) = sbb(t[j], m[j], borrow);
        out[j] = v;
        borrow = b;
    }
    let take_diff = mask_from_bit((t[n] | (borrow ^ 1)) & 1);
    for j in 0..n {
        out[j] = (out[j] & take_diff) | (t[j] & !take_diff);
    }
}

/// `base^exp mod m` in constant time with respect to the values (see the
/// module docs). `m` must be odd with a non-zero top limb; `base` and `exp`
/// are little-endian limbs of any length. Returns `n` limbs (not normalized).
fn modpow_sec_limbs(base: &[Limb], exp: &[Limb], m: &[Limb]) -> Vec<Limb> {
    let n = m.len();
    debug_assert!(n > 0 && m[0] & 1 == 1);
    let minv = neg_inv_mod_2_64(m[0]);

    // one = R mod m; b = base·R mod m (both via the constant-time reduction).
    let mut r_limbs = vec![0 as Limb; n + 1];
    r_limbs[n] = 1;
    let one = ct_mod(&r_limbs, m);
    let mut shifted = vec![0 as Limb; n + base.len()];
    shifted[n..].copy_from_slice(base);
    let b = ct_mod(&shifted, m);

    // Window width and table of b^i (Montgomery form), i in 0..2^w.
    let bits = exp.len() * 64;
    let w = if bits > 256 { 5 } else { 4 };
    let size = 1usize << w;
    let mut t = vec![0 as Limb; n + 2];
    let mut table = vec![0 as Limb; size * n];
    table[..n].copy_from_slice(&one);
    table[n..2 * n].copy_from_slice(&b);
    for i in 2..size {
        let (done, rest) = table.split_at_mut(i * n);
        mont_mul(&mut rest[..n], &done[(i - 1) * n..], &b, m, minv, &mut t);
    }

    let mut acc = one.clone();
    let mut tmp = vec![0 as Limb; n];
    let mut sel = vec![0 as Limb; n];
    // Windows from the top; the first one takes the leftover high bits.
    let mut pos = bits;
    let first = if bits.is_multiple_of(w) { w } else { bits % w };
    let mut width = first;
    while pos > 0 {
        let lo = pos - width;
        // Window value: `width` bits of exp starting at `lo` (fixed shifts).
        let mut idx: Limb = 0;
        for k in (lo..pos).rev() {
            idx = (idx << 1) | ((exp[k / 64] >> (k % 64)) & 1);
        }
        for _ in 0..width {
            tmp.copy_from_slice(&acc);
            mont_mul(&mut acc, &tmp, &tmp, m, minv, &mut t);
        }
        // Constant-time table lookup.
        sel.fill(0);
        for i in 0..size {
            let mask = mask_eq(i as Limb, idx);
            for (s, &e) in sel.iter_mut().zip(&table[i * n..(i + 1) * n]) {
                *s |= e & mask;
            }
        }
        tmp.copy_from_slice(&acc);
        mont_mul(&mut acc, &tmp, &sel, m, minv, &mut t);
        pos = lo;
        width = w;
    }

    // Leave Montgomery form: acc·1·R⁻¹.
    let mut unit = vec![0 as Limb; n];
    unit[0] = 1;
    tmp.copy_from_slice(&acc);
    mont_mul(&mut acc, &tmp, &unit, m, minv, &mut t);
    acc
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
        let reduced = ct_mod(self.magnitude_ref().as_limbs(), m);
        let neg = ct_neg_mod(&reduced, m);
        let mask = mask_from_bit(self.is_negative() as Limb);
        let base: Vec<Limb> = reduced
            .iter()
            .zip(&neg)
            .map(|(&p, &q)| (q & mask) | (p & !mask))
            .collect();
        Int::from(Nat::from_limb_vec(modpow_sec_limbs(
            &base,
            exp.magnitude_ref().as_limbs(),
            m,
        )))
    }
}
