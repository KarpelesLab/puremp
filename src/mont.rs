//! Montgomery-multiplication kernels on fixed-length limb slices.
//!
//! Every routine here has a *constant flow*: loop trip counts and memory
//! accesses depend only on slice lengths, never on limb values (no early
//! carry-ripple exits, no data-dependent comparisons). The same kernels serve
//! the fast variable-time [`Nat::modpow`](crate::nat::Nat::modpow) ladder and
//! the constant-time `modpow_sec`.
//!
//! The Montgomery product (Montgomery 1985) is computed by *finely integrated
//! product scanning* (FIPS; Koç–Acar–Kaliski, "Analyzing and comparing
//! Montgomery multiplication algorithms", 1996): output column `i` sums every
//! `aⱼ·bᵢ₋ⱼ` and `qⱼ·mᵢ₋ⱼ` landing on it into a three-limb accumulator, the
//! quotient digit `qᵢ` is chosen to clear the column (for `i < n`), and the
//! accumulator shifts down one limb. Each product costs one `u128` add plus a
//! carry count, with short independent carry chains — markedly faster in
//! portable Rust than row-wise `addmul` loops, whose single carry threads
//! through every product. Squaring sums each cross product `aⱼ·aₖ` (`j < k`)
//! once and doubles it.
//!
//! Larger operands use contiguous forward dot products: the second factor
//! (and the modulus) are stored reversed so that `bᵢ₋ⱼ` walks forward with
//! `j`, and each dot product runs two interleaved carry chains.

use alloc::vec;
use alloc::vec::Vec;

use crate::limb::{DLimb, LIMB_BITS, Limb, sbb};

/// Below this many limbs, products use the simple column loops.
const DOT_MUL_THRESHOLD: usize = 12;
/// Below this many limbs, squarings use the simple column loops.
const DOT_SQR_THRESHOLD: usize = 32;

/// `−m⁻¹ mod 2^64` for odd `m0`, by Newton's iteration (fixed step count).
#[inline]
pub(crate) fn neg_inv(m0: Limb) -> Limb {
    debug_assert!(m0 & 1 == 1, "Montgomery needs an odd modulus");
    // Seed correct to 5 bits, then 5 → 10 → 20 → 40 → 80 (≥ 64) bits.
    let mut y = m0.wrapping_mul(3) ^ 2;
    for _ in 0..4 {
        y = y.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(y)));
    }
    y.wrapping_neg()
}

/// All-ones when `bit == 1`, zero when `bit == 0`; laundered through
/// [`core::hint::black_box`] so the optimizer cannot turn selects on it back
/// into branches.
#[inline(always)]
pub(crate) fn mask_from_bit(bit: Limb) -> Limb {
    core::hint::black_box(bit).wrapping_neg()
}

/// Writes `x mod m` to `out` for `x = cy·R + hi < 2m` (`cy ∈ {0, 1}`): one
/// subtraction of `m`, kept by mask when `x ≥ m`.
#[inline]
pub(crate) fn final_sub(out: &mut [Limb], hi: &[Limb], cy: Limb, m: &[Limb]) {
    let mut borrow = 0;
    for ((o, &h), &mj) in out.iter_mut().zip(hi).zip(m) {
        let (v, b) = sbb(h, mj, borrow);
        *o = v;
        borrow = b;
    }
    // x ≥ m iff the top bit is set or the subtraction did not borrow.
    let take = mask_from_bit((cy | (borrow ^ 1)) & 1);
    for (o, &h) in out.iter_mut().zip(hi) {
        *o = (*o & take) | (h & !take);
    }
}

/// Three-limb column accumulator: `hi·2^128 + lo`.
#[derive(Clone, Copy)]
struct Acc {
    lo: DLimb,
    hi: Limb,
}

impl Acc {
    const ZERO: Acc = Acc { lo: 0, hi: 0 };

    #[inline(always)]
    fn add(&mut self, p: DLimb) {
        let (s, c) = self.lo.overflowing_add(p);
        self.lo = s;
        self.hi = self.hi.wrapping_add(c as Limb);
    }

    #[inline(always)]
    fn merge(&mut self, o: Acc) {
        let (s, c) = self.lo.overflowing_add(o.lo);
        self.lo = s;
        self.hi = self.hi.wrapping_add(o.hi).wrapping_add(c as Limb);
    }

    /// `2·self` (the column sums never come near 2^191).
    #[inline(always)]
    fn doubled(self) -> Acc {
        Acc {
            lo: self.lo << 1,
            hi: (self.hi << 1) | (self.lo >> (2 * LIMB_BITS - 1)) as Limb,
        }
    }

    /// Emits the low limb and shifts the accumulator down one limb.
    #[inline(always)]
    fn shift(&mut self) -> Limb {
        let out = self.lo as Limb;
        self.lo = (self.lo >> LIMB_BITS) | ((self.hi as DLimb) << LIMB_BITS);
        self.hi = 0;
        out
    }
}

#[inline(always)]
fn prod(a: Limb, b: Limb) -> DLimb {
    a as DLimb * b as DLimb
}

/// `Σ x[k]·y[k]` over `x.len()` terms, with two interleaved carry chains.
#[inline(always)]
fn dot(x: &[Limb], y: &[Limb]) -> Acc {
    let len = x.len();
    let y = &y[..len];
    let mut e = Acc::ZERO;
    let mut o = Acc::ZERO;
    let (x2, xr) = x.split_at(len & !1);
    let (y2, yr) = y.split_at(len & !1);
    for (xp, yp) in x2.as_chunks::<2>().0.iter().zip(y2.as_chunks::<2>().0) {
        e.add(prod(xp[0], yp[0]));
        o.add(prod(xp[1], yp[1]));
    }
    if let (Some(&xa), Some(&ya)) = (xr.first(), yr.first()) {
        e.add(prod(xa, ya));
    }
    e.merge(o);
    e
}

/// `(Σ x[k]·y[k], Σ u[k]·v[k])` over `x.len()` terms in one loop.
#[inline(always)]
fn dot2(x: &[Limb], y: &[Limb], u: &[Limb], v: &[Limb]) -> (Acc, Acc) {
    let len = x.len();
    let (y, u, v) = (&y[..len], &u[..len], &v[..len]);
    let mut s = Acc::ZERO;
    let mut t = Acc::ZERO;
    for k in 0..len {
        s.add(prod(x[k], y[k]));
        t.add(prod(u[k], v[k]));
    }
    (s, t)
}

/// Small-operand FIPS product: `out = a·b·R⁻¹ mod m`, `q` is `n` limbs of
/// scratch.
fn mul_cols(out: &mut [Limb], a: &[Limb], b: &[Limb], m: &[Limb], n0inv: Limb, q: &mut [Limb]) {
    let n = m.len();
    let (a, b, q, out) = (&a[..n], &b[..n], &mut q[..n], &mut out[..n]);
    let mut acc = Acc::ZERO;
    for i in 0..n {
        let mut acc2 = Acc::ZERO;
        for (&aj, &bk) in a[..i].iter().zip(b[1..=i].iter().rev()) {
            acc.add(prod(aj, bk));
        }
        for (&qj, &mk) in q[..i].iter().zip(m[1..=i].iter().rev()) {
            acc2.add(prod(qj, mk));
        }
        acc.add(prod(a[i], b[0]));
        acc.merge(acc2);
        let qi = (acc.lo as Limb).wrapping_mul(n0inv);
        q[i] = qi;
        acc.add(prod(qi, m[0]));
        acc.shift();
    }
    for i in n..2 * n {
        let lo = i - n + 1;
        let mut acc2 = Acc::ZERO;
        for (&aj, &bk) in a[lo..].iter().zip(b[lo..].iter().rev()) {
            acc.add(prod(aj, bk));
        }
        for (&qj, &mk) in q[lo..].iter().zip(m[lo..].iter().rev()) {
            acc2.add(prod(qj, mk));
        }
        acc.merge(acc2);
        // q[i − n] is no longer read: it holds result limb i − n.
        q[i - n] = acc.shift();
    }
    final_sub(out, q, acc.lo as Limb, m);
}

/// Small-operand FIPS squaring: `out = a²·R⁻¹ mod m`.
fn sqr_cols(out: &mut [Limb], a: &[Limb], m: &[Limb], n0inv: Limb, q: &mut [Limb]) {
    let n = m.len();
    let (a, q, out) = (&a[..n], &mut q[..n], &mut out[..n]);
    let mut acc = Acc::ZERO;
    for i in 0..n {
        // Cross pairs (j, i − j) with j < i − j.
        let h = i.div_ceil(2);
        let mut cross = Acc::ZERO;
        for (&aj, &ak) in a[..h].iter().zip(a[i + 1 - h..=i].iter().rev()) {
            cross.add(prod(aj, ak));
        }
        let mut red = Acc::ZERO;
        for (&qj, &mk) in q[..i].iter().zip(m[1..=i].iter().rev()) {
            red.add(prod(qj, mk));
        }
        acc.merge(cross.doubled());
        if i % 2 == 0 {
            acc.add(prod(a[i / 2], a[i / 2]));
        }
        acc.merge(red);
        let qi = (acc.lo as Limb).wrapping_mul(n0inv);
        q[i] = qi;
        acc.add(prod(qi, m[0]));
        acc.shift();
    }
    for i in n..2 * n {
        let lo = i - n + 1;
        // Cross pairs (j, i − j) with lo ≤ j < i − j ≤ n − 1.
        let h = i.div_ceil(2);
        let mut cross = Acc::ZERO;
        for (&aj, &ak) in a[lo..h].iter().zip(a[i + 1 - h..].iter().rev()) {
            cross.add(prod(aj, ak));
        }
        let mut red = Acc::ZERO;
        for (&qj, &mk) in q[lo..].iter().zip(m[lo..].iter().rev()) {
            red.add(prod(qj, mk));
        }
        acc.merge(cross.doubled());
        if i % 2 == 0 {
            acc.add(prod(a[i / 2], a[i / 2]));
        }
        acc.merge(red);
        q[i - n] = acc.shift();
    }
    final_sub(out, q, acc.lo as Limb, m);
}

/// Dot-product FIPS product. `rev` receives `b` reversed, `mrev` is `m`
/// reversed (so `bᵢ₋ⱼ = rev[n − 1 − i + j]`), `q` is `n` limbs of scratch.
#[allow(clippy::too_many_arguments)]
fn mul_dot(
    out: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    m: &[Limb],
    mrev: &[Limb],
    n0inv: Limb,
    q: &mut [Limb],
    rev: &mut [Limb],
) {
    let n = m.len();
    let (a, b, mrev) = (&a[..n], &b[..n], &mrev[..n]);
    let (q, rev, out) = (&mut q[..n], &mut rev[..n], &mut out[..n]);
    for (r, &x) in rev.iter_mut().zip(b.iter().rev()) {
        *r = x;
    }
    let mut acc = Acc::ZERO;
    for i in 0..n {
        let (s, t) = dot2(&a[..i], &rev[n - 1 - i..], &q[..i], &mrev[n - 1 - i..]);
        acc.merge(s);
        acc.merge(t);
        acc.add(prod(a[i], b[0]));
        let qi = (acc.lo as Limb).wrapping_mul(n0inv);
        q[i] = qi;
        acc.add(prod(qi, m[0]));
        acc.shift();
    }
    for i in n..2 * n {
        let lo = i - n + 1;
        let (s, t) = dot2(&a[lo..], rev, &q[lo..], mrev);
        acc.merge(s);
        acc.merge(t);
        q[i - n] = acc.shift();
    }
    final_sub(out, q, acc.lo as Limb, m);
}

/// Dot-product FIPS squaring (see [`mul_dot`]).
fn sqr_dot(
    out: &mut [Limb],
    a: &[Limb],
    m: &[Limb],
    mrev: &[Limb],
    n0inv: Limb,
    q: &mut [Limb],
    rev: &mut [Limb],
) {
    let n = m.len();
    let (a, mrev) = (&a[..n], &mrev[..n]);
    let (q, rev, out) = (&mut q[..n], &mut rev[..n], &mut out[..n]);
    for (r, &x) in rev.iter_mut().zip(a.iter().rev()) {
        *r = x;
    }
    let mut acc = Acc::ZERO;
    for i in 0..2 * n {
        let lo = (i + 1).saturating_sub(n);
        // Column i pairs j with i − j, i.e. with rev[n − 1 − i + j].
        let off = n - 1 + lo - i;
        let h = i.div_ceil(2).max(lo);
        let cross = dot(&a[lo..h], &rev[off..]);
        acc.merge(cross.doubled());
        if i % 2 == 0 {
            acc.add(prod(a[i / 2], a[i / 2]));
        }
        if i < n {
            acc.merge(dot(&q[..i], &mrev[off..]));
            let qi = (acc.lo as Limb).wrapping_mul(n0inv);
            q[i] = qi;
            acc.add(prod(qi, m[0]));
            acc.shift();
        } else {
            acc.merge(dot(&q[lo..], mrev));
            q[i - n] = acc.shift();
        }
    }
    final_sub(out, q, acc.lo as Limb, m);
}

/// Montgomery arithmetic modulo an odd `m` (`n = m.len()` limbs, top limb
/// non-zero), with its scratch, so repeated products allocate nothing.
/// Operands and results are `n`-limb slices; every operation has a constant
/// flow (see the module docs).
pub(crate) struct Mont<'a> {
    m: &'a [Limb],
    /// `m` reversed (dot-product kernels).
    mrev: Vec<Limb>,
    n0inv: Limb,
    /// `2n` limbs: quotient digits / result, then a reversed operand.
    t: Vec<Limb>,
}

impl<'a> Mont<'a> {
    pub(crate) fn new(m: &'a [Limb]) -> Mont<'a> {
        let n = m.len();
        debug_assert!(n > 0 && m[0] & 1 == 1, "Montgomery needs an odd modulus");
        Mont {
            m,
            mrev: m.iter().rev().copied().collect(),
            n0inv: neg_inv(m[0]),
            t: vec![0; 2 * n],
        }
    }

    /// `out = a·b·R⁻¹ mod m` (`R = 2^(64n)`) for `a·b < m·R` (e.g. `a < R`,
    /// `b < m`).
    pub(crate) fn mul(&mut self, out: &mut [Limb], a: &[Limb], b: &[Limb]) {
        let n = self.m.len();
        let (q, rev) = self.t.split_at_mut(n);
        let (m, n0inv) = (self.m, self.n0inv);
        if n < DOT_MUL_THRESHOLD {
            mul_cols(out, a, b, m, n0inv, q);
        } else {
            mul_dot(out, a, b, m, &self.mrev, n0inv, q, rev);
        }
    }

    /// `out = a²·R⁻¹ mod m` for `a < m`.
    pub(crate) fn sqr(&mut self, out: &mut [Limb], a: &[Limb]) {
        let n = self.m.len();
        let (q, rev) = self.t.split_at_mut(n);
        let (m, n0inv) = (self.m, self.n0inv);
        if n < DOT_SQR_THRESHOLD {
            sqr_cols(out, a, m, n0inv, q);
        } else {
            sqr_dot(out, a, m, &self.mrev, n0inv, q, rev);
        }
    }

    /// `out = x·R⁻¹ mod m` (leaves Montgomery form) for `x < R`.
    pub(crate) fn leave(&mut self, out: &mut [Limb], x: &[Limb]) {
        let mut unit = vec![0 as Limb; self.m.len()];
        unit[0] = 1;
        self.mul(out, x, &unit);
    }
}
