//! Arbitrary-precision natural numbers (unsigned integers).
//!
//! [`Nat`] stores magnitude as a little-endian `Vec` of 64-bit limbs,
//! kept in a canonical form with no trailing zero limbs (so the value zero is
//! the empty vector). That canonical form makes equality and ordering cheap and
//! lets the derived [`PartialEq`]/[`Eq`] be correct.
//!
//! This is the layer that carries the heavy limb-level algorithms: addition,
//! subtraction, multiplication (schoolbook → Karatsuba → Toom-3 → NTT),
//! squaring, division (single-limb, Knuth Algorithm D, and Burnikel–Ziegler),
//! shifts, GCD (binary → Lehmer), roots, and sub-quadratic radix I/O.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::limb::{LIMB_BITS, Limb, adc, mac, sbb};

/// Operands with fewer than this many limbs use schoolbook multiplication;
/// larger ones recurse via Karatsuba. Chosen conservatively; a tuned crossover
/// is a later milestone (see `ROADMAP.md`).
const KARATSUBA_THRESHOLD: usize = 32;

/// Operands with at least this many limbs use Toom-3 (above Karatsuba). Chosen
/// conservatively; a tuned crossover is a later milestone.
const TOOM3_THRESHOLD: usize = 128;

/// GCD switches from Stein's binary algorithm to Lehmer's above this many limbs.
const LEHMER_THRESHOLD: usize = 16;

/// Operands with at least this many limbs use NTT multiplication (above Toom-3),
/// provided the transform length stays within [`NTT_MAX_LEN`].
const NTT_THRESHOLD: usize = 1600;

/// Largest NTT transform length used; beyond this the convolution coefficients
/// could exceed the modulus, so multiplication falls back to Toom-3.
const NTT_MAX_LEN: usize = 1 << 28;

// --- Number-theoretic transform over the Goldilocks field 2^64 − 2^32 + 1 ---
//
// This prime has `p − 1 = 2^32·(2^32 − 1)`, so it supports NTTs of any power-of-
// two length up to 2^32, and 7 is a primitive root. Modular reduction uses the
// portable `u128 % p` (correct, if not the fastest possible).

/// The Goldilocks prime `2^64 − 2^32 + 1`.
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;
/// A primitive root of the Goldilocks multiplicative group.
const GOLDILOCKS_ROOT: u64 = 7;

#[inline]
fn gf_mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % GOLDILOCKS as u128) as u64
}

#[inline]
fn gf_add(a: u64, b: u64) -> u64 {
    let s = a as u128 + b as u128;
    (if s >= GOLDILOCKS as u128 {
        s - GOLDILOCKS as u128
    } else {
        s
    }) as u64
}

#[inline]
fn gf_sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        (a as u128 + GOLDILOCKS as u128 - b as u128) as u64
    }
}

fn gf_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut r = 1u64;
    base %= GOLDILOCKS;
    while exp > 0 {
        if exp & 1 == 1 {
            r = gf_mul(r, base);
        }
        base = gf_mul(base, base);
        exp >>= 1;
    }
    r
}

/// In-place iterative NTT (or its inverse) over the Goldilocks field.
fn ntt(a: &mut [u64], inverse: bool) {
    let n = a.len();
    // Bit-reversal permutation.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let mut wlen = gf_pow(GOLDILOCKS_ROOT, (GOLDILOCKS - 1) / len as u64);
        if inverse {
            wlen = gf_pow(wlen, GOLDILOCKS - 2);
        }
        let mut i = 0;
        while i < n {
            let mut w = 1u64;
            for k in 0..len / 2 {
                let u = a[i + k];
                let v = gf_mul(a[i + k + len / 2], w);
                a[i + k] = gf_add(u, v);
                a[i + k + len / 2] = gf_sub(u, v);
                w = gf_mul(w, wlen);
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let n_inv = gf_pow(n as u64, GOLDILOCKS - 2);
        for x in a.iter_mut() {
            *x = gf_mul(*x, n_inv);
        }
    }
}

/// Splits `x` into little-endian 16-bit digits (at least one).
fn to_digits16(x: &Nat) -> Vec<u64> {
    let bytes = x.to_bytes_le();
    let mut d = Vec::with_capacity(bytes.len() / 2 + 1);
    let mut i = 0;
    while i < bytes.len() {
        let lo = bytes[i] as u64;
        let hi = if i + 1 < bytes.len() {
            bytes[i + 1] as u64
        } else {
            0
        };
        d.push(lo | (hi << 8));
        i += 2;
    }
    if d.is_empty() {
        d.push(0);
    }
    d
}

/// NTT-based multiplication (falls back to Toom-3 if the transform would be too
/// long for a single Goldilocks prime).
fn mul_ntt(a: &Nat, b: &Nat) -> Nat {
    let da = to_digits16(a);
    let db = to_digits16(b);
    let need = da.len() + db.len();
    let mut n = 1usize;
    while n < need {
        n <<= 1;
    }
    if n > NTT_MAX_LEN {
        return a.mul_toom3(b);
    }
    let mut fa = alloc::vec![0u64; n];
    let mut fb = alloc::vec![0u64; n];
    fa[..da.len()].copy_from_slice(&da);
    fb[..db.len()].copy_from_slice(&db);
    ntt(&mut fa, false);
    ntt(&mut fb, false);
    for (x, y) in fa.iter_mut().zip(&fb) {
        *x = gf_mul(*x, *y);
    }
    ntt(&mut fa, true);
    // Carry-propagate the convolution coefficients in base 2^16.
    let mut bytes: Vec<u8> = Vec::with_capacity(2 * n + 8);
    let mut carry: u128 = 0;
    for &coef in &fa {
        carry += coef as u128;
        bytes.push((carry & 0xFF) as u8);
        bytes.push(((carry >> 8) & 0xFF) as u8);
        carry >>= 16;
    }
    while carry != 0 {
        bytes.push((carry & 0xFF) as u8);
        bytes.push(((carry >> 8) & 0xFF) as u8);
        carry >>= 16;
    }
    Nat::from_bytes_le(&bytes)
}

/// Divisors with at least this many limbs use Burnikel–Ziegler recursive
/// division; smaller ones use Knuth Algorithm D directly.
const BZ_THRESHOLD: usize = 64;

/// Recursion base case (in half-block limbs) for Burnikel–Ziegler.
const BZ_BASE: usize = 32;

/// Extracts block `i` (limbs `[i·n, (i+1)·n)`) of `x` as a [`Nat`].
fn bz_block(x: &Nat, i: usize, n: usize) -> Nat {
    let lo = i * n;
    let l = x.limbs.len();
    if lo >= l {
        Nat::zero()
    } else {
        Nat::from_limbs(&x.limbs[lo..(lo + n).min(l)])
    }
}

/// Burnikel–Ziegler top level: normalize the divisor, then process the dividend
/// in `n`-limb blocks from the top, dividing each `≤ 2n`-limb window. Requires
/// `a > b` and `b.limbs.len() >= 2`.
fn bz_div_rem(a: &Nat, b: &Nat) -> (Nat, Nat) {
    let n = b.limbs.len();
    let s = b.limbs[n - 1].leading_zeros() as u64;
    let bn = b.shl(s);
    let an = a.shl(s);
    let nbits = n as u64 * LIMB_BITS as u64;
    let t = an.limbs.len().div_ceil(n).max(2);

    let mut r = Nat::zero();
    let mut parts: Vec<Nat> = Vec::with_capacity(t);
    for i in (0..t).rev() {
        let cur = r.shl(nbits).add(&bz_block(&an, i, n));
        let (qi, ri) = bz_div_2n_1n(&cur, &bn, n);
        parts.push(qi);
        r = ri;
    }
    let mut q = Nat::zero();
    for (j, part) in parts.into_iter().enumerate() {
        q = q.add(&part.shl((t - 1 - j) as u64 * nbits));
    }
    (q, r.shr(s))
}

/// Divide a `≤ 2n`-limb value by the `n`-limb normalized divisor `b`
/// (`quotient < 2^(64n)`).
fn bz_div_2n_1n(a: &Nat, b: &Nat, n: usize) -> (Nat, Nat) {
    if a.cmp_ref(b) == Ordering::Less {
        return (Nat::zero(), a.clone());
    }
    if n < BZ_BASE || n % 2 == 1 {
        if a.cmp_ref(b) == Ordering::Equal {
            return (Nat::one(), Nat::zero());
        }
        if b.limbs.len() == 1 {
            let (q, rr) = a.divmod_small(b.limbs[0]);
            return (q, Nat::from_u64(rr));
        }
        return a.div_rem_knuth(b);
    }
    let half = n / 2;
    let hbits = half as u64 * LIMB_BITS as u64;
    let (q1, r1) = bz_div_3n_2n(&a.shr(hbits), b, half);
    let (q2, r2) = bz_div_3n_2n(&r1.shl(hbits).add(&a.low_bits(hbits)), b, half);
    (q1.shl(hbits).add(&q2), r2)
}

/// Divide a `≤ 3·half`-limb value by the `2·half`-limb normalized divisor `b`.
fn bz_div_3n_2n(a: &Nat, b: &Nat, half: usize) -> (Nat, Nat) {
    use crate::int::Int;
    let hbits = half as u64 * LIMB_BITS as u64;
    let b1 = b.shr(hbits);
    let b2 = b.low_bits(hbits);
    let a12 = a.shr(hbits);
    let a3 = a.low_bits(hbits);

    let (q_nat, r_pre): (Nat, Int) = if a12.shr(hbits).cmp_ref(&b1) == Ordering::Less {
        let (q, r) = bz_div_2n_1n(&a12, &b1, half);
        (q, Int::from(r))
    } else {
        // q = 2^(64·half) − 1; R = A12 − q·B1.
        let q = Nat::one()
            .shl(hbits)
            .checked_sub(&Nat::one())
            .expect("2^k >= 1");
        let r = Int::from(a12).sub(&Int::from(q.mul(&b1)));
        (q, r)
    };

    // R = R·2^(64·half) + A3 − q·B2, corrected to be non-negative.
    let mut r_int = r_pre
        .mul_2k(hbits as u32)
        .add(&Int::from(a3))
        .sub(&Int::from(q_nat.mul(&b2)));
    let mut q_int = Int::from(q_nat);
    let b_int = Int::from(b.clone());
    while r_int.is_negative() {
        q_int = q_int.sub(&Int::ONE);
        r_int = r_int.add(&b_int);
    }
    (q_int.magnitude(), r_int.magnitude())
}

/// `s·a + t·b` as an [`Int`], for the Lehmer cofactor combination.
fn lincomb(s: i128, a: &crate::int::Int, t: i128, b: &crate::int::Int) -> crate::int::Int {
    use crate::int::Int;
    Int::from_i128(s).mul(a).add(&Int::from_i128(t).mul(b))
}

/// An arbitrary-precision natural number (a non-negative integer).
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Nat {
    /// Little-endian limbs, normalized so the most-significant limb is non-zero.
    /// The value zero is represented by an empty vector.
    limbs: Vec<Limb>,
}

impl Nat {
    /// Returns the natural number zero.
    #[inline]
    pub fn zero() -> Self {
        Nat { limbs: Vec::new() }
    }

    /// Returns the natural number one.
    #[inline]
    pub fn one() -> Self {
        Nat::from_u64(1)
    }

    /// Builds a [`Nat`] from a `u64`.
    #[inline]
    pub fn from_u64(v: u64) -> Self {
        let mut n = Nat {
            limbs: if v == 0 { Vec::new() } else { alloc::vec![v] },
        };
        n.normalize();
        n
    }

    /// Builds a [`Nat`] from a `u128`.
    pub fn from_u128(v: u128) -> Self {
        let lo = v as Limb;
        let hi = (v >> LIMB_BITS) as Limb;
        let mut n = Nat {
            limbs: alloc::vec![lo, hi],
        };
        n.normalize();
        n
    }

    /// Returns `true` if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Returns `true` if this value is even (including zero).
    #[inline]
    pub fn is_even(&self) -> bool {
        self.limbs.first().is_none_or(|&l| l & 1 == 0)
    }

    /// Returns the number of significant bits (the position of the highest set
    /// bit plus one); zero has a bit length of zero.
    pub fn bit_len(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(&top) => {
                (self.limbs.len() as u64 - 1) * LIMB_BITS as u64
                    + (LIMB_BITS - top.leading_zeros()) as u64
            }
        }
    }

    /// Returns the number of trailing zero bits, i.e. the largest `k` such that
    /// `2^k` divides this value. Returns zero for the value zero.
    pub fn trailing_zeros(&self) -> u64 {
        for (i, &l) in self.limbs.iter().enumerate() {
            if l != 0 {
                return i as u64 * LIMB_BITS as u64 + l.trailing_zeros() as u64;
            }
        }
        0
    }

    /// Drops any trailing zero limbs, restoring the canonical form.
    fn normalize(&mut self) {
        while matches!(self.limbs.last(), Some(&0)) {
            self.limbs.pop();
        }
    }

    /// Compares two naturals.
    fn cmp_ref(&self, other: &Nat) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
        for (a, b) in self.limbs.iter().rev().zip(other.limbs.iter().rev()) {
            match a.cmp(b) {
                Ordering::Equal => continue,
                non_eq => return non_eq,
            }
        }
        Ordering::Equal
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Nat) -> Nat {
        let (long, short) = if self.limbs.len() >= rhs.limbs.len() {
            (self, rhs)
        } else {
            (rhs, self)
        };
        let mut out = Vec::with_capacity(long.limbs.len() + 1);
        let mut carry = 0;
        for (i, &a) in long.limbs.iter().enumerate() {
            let b = short.limbs.get(i).copied().unwrap_or(0);
            let (s, c) = adc(a, b, carry);
            out.push(s);
            carry = c;
        }
        if carry != 0 {
            out.push(carry);
        }
        // Already normalized: the top limb is non-zero (or both inputs were zero).
        Nat { limbs: out }
    }

    /// Returns `self - rhs`, or `None` if `rhs > self` (naturals cannot be
    /// negative).
    pub fn checked_sub(&self, rhs: &Nat) -> Option<Nat> {
        if self.cmp_ref(rhs) == Ordering::Less {
            return None;
        }
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0;
        for (i, &a) in self.limbs.iter().enumerate() {
            let b = rhs.limbs.get(i).copied().unwrap_or(0);
            let (d, bb) = sbb(a, b, borrow);
            out.push(d);
            borrow = bb;
        }
        debug_assert_eq!(borrow, 0, "checked_sub borrow escaped after a >= b check");
        let mut n = Nat { limbs: out };
        n.normalize();
        Some(n)
    }

    /// Returns `self · rhs`, dispatching to schoolbook or Karatsuba by size.
    ///
    /// Toom-Cook and FFT/NTT multiplication for very large operands are later
    /// milestones; see `ROADMAP.md`.
    pub fn mul(&self, rhs: &Nat) -> Nat {
        if self.is_zero() || rhs.is_zero() {
            return Nat::zero();
        }
        if self.limbs == rhs.limbs {
            return self.square();
        }
        let min_len = self.limbs.len().min(rhs.limbs.len());
        if min_len < KARATSUBA_THRESHOLD {
            self.mul_schoolbook(rhs)
        } else if min_len < TOOM3_THRESHOLD {
            self.mul_karatsuba(rhs)
        } else if min_len < NTT_THRESHOLD {
            self.mul_toom3(rhs)
        } else {
            mul_ntt(self, rhs)
        }
    }

    /// Toom-3 multiplication: five half-third-size products, evaluated at the
    /// points {0, 1, −1, 2, ∞} and interpolated (signed intermediates use
    /// [`Int`]). Asymptotically `O(n^1.465)`.
    fn mul_toom3(&self, rhs: &Nat) -> Nat {
        use crate::int::Int;

        let n = self.limbs.len().max(rhs.limbs.len());
        let k = n.div_ceil(3);
        let bshift = k as u64 * LIMB_BITS as u64;

        // Split a value into its base-2^(64k) digits a0 + a1·B + a2·B², as Int.
        let part = |x: &Nat, lo: usize, hi: usize| -> Int {
            let l = x.limbs.len();
            if lo >= l {
                Int::ZERO
            } else {
                Int::from(Nat::from_limbs(&x.limbs[lo..hi.min(l)]))
            }
        };
        let (a0, a1, a2) = (
            part(self, 0, k),
            part(self, k, 2 * k),
            part(self, 2 * k, 3 * k),
        );
        let (b0, b1, b2) = (
            part(rhs, 0, k),
            part(rhs, k, 2 * k),
            part(rhs, 2 * k, 3 * k),
        );

        // Evaluate a(x), b(x) at 1, −1, 2 (0 and ∞ are a0/a2 directly).
        let pa = a0.add(&a2);
        let (pm1, p1) = (pa.sub(&a1), pa.add(&a1));
        let p2 = p1.add(&a2).mul_2k(1).sub(&a0);
        let qb = b0.add(&b2);
        let (qm1, q1) = (qb.sub(&b1), qb.add(&b1));
        let q2 = q1.add(&b2).mul_2k(1).sub(&b0);

        // Pointwise products (these recurse through the dispatcher).
        let r0 = a0.mul(&b0);
        let r1 = p1.mul(&q1);
        let rm1 = pm1.mul(&qm1);
        let r2 = p2.mul(&q2);
        let rinf = a2.mul(&b2);

        // Interpolate the coefficients c0..c4 (exact divisions by 2 and 6).
        let two = Int::from_i64(2);
        let c0 = r0;
        let c4 = rinf;
        let c2 = r1.add(&rm1).div_exact(&two).sub(&c0).sub(&c4);
        let s = r1.sub(&rm1).div_exact(&two);
        let t = r2
            .sub(&c0)
            .sub(&c2.mul(&Int::from_i64(4)))
            .sub(&c4.mul(&Int::from_i64(16)))
            .sub(&s.mul(&two));
        let c3 = t.div_exact(&Int::from_i64(6));
        let c1 = s.sub(&c3);

        let result = c0
            .add(&c1.mul_2k(bshift as u32))
            .add(&c2.mul_2k((2 * bshift) as u32))
            .add(&c3.mul_2k((3 * bshift) as u32))
            .add(&c4.mul_2k((4 * bshift) as u32));
        debug_assert!(!result.is_negative(), "toom3 produced a negative result");
        result.magnitude()
    }

    /// Quadratic schoolbook (long) multiplication.
    fn mul_schoolbook(&self, rhs: &Nat) -> Nat {
        let mut out = alloc::vec![0 as Limb; self.limbs.len() + rhs.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0;
            for (j, &b) in rhs.limbs.iter().enumerate() {
                let (lo, hi) = mac(out[i + j], a, b, carry);
                out[i + j] = lo;
                carry = hi;
            }
            out[i + rhs.limbs.len()] = carry;
        }
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }

    /// Returns `self²`, using a symmetric schoolbook or Karatsuba squaring
    /// (roughly half the limb multiplications of the general `mul`).
    pub fn square(&self) -> Nat {
        if self.is_zero() {
            return Nat::zero();
        }
        if self.limbs.len() < KARATSUBA_THRESHOLD {
            self.square_schoolbook()
        } else {
            self.square_karatsuba()
        }
    }

    /// Symmetric schoolbook squaring: accumulate the strictly-upper triangle of
    /// cross products once, double it, then add the diagonal `aᵢ²` terms.
    fn square_schoolbook(&self) -> Nat {
        let n = self.limbs.len();
        let mut cross = alloc::vec![0 as Limb; 2 * n];
        for i in 0..n {
            let mut carry = 0;
            for j in (i + 1)..n {
                let (lo, hi) = mac(cross[i + j], self.limbs[i], self.limbs[j], carry);
                cross[i + j] = lo;
                carry = hi;
            }
            cross[i + n] = carry;
        }
        let mut result = {
            let mut c = Nat { limbs: cross };
            c.normalize();
            c.shl(1) // 2·(sum of cross products)
        };

        // Add the diagonal squares aᵢ² at position 2i.
        let mut diag = alloc::vec![0 as Limb; 2 * n];
        for i in 0..n {
            let sq = self.limbs[i] as u128 * self.limbs[i] as u128;
            let (lo, hi) = (sq as Limb, (sq >> LIMB_BITS) as Limb);
            let (s0, c0) = adc(diag[2 * i], lo, 0);
            diag[2 * i] = s0;
            let (s1, mut carry) = adc(diag[2 * i + 1], hi, c0);
            diag[2 * i + 1] = s1;
            let mut k = 2 * i + 2;
            while carry != 0 && k < 2 * n {
                let (s, c) = adc(diag[k], 0, carry);
                diag[k] = s;
                carry = c;
                k += 1;
            }
        }
        let mut diag = Nat { limbs: diag };
        diag.normalize();
        result = result.add(&diag);
        result
    }

    /// Karatsuba squaring: three half-size squarings.
    fn square_karatsuba(&self) -> Nat {
        let n = self.limbs.len();
        if n < KARATSUBA_THRESHOLD {
            return self.square_schoolbook();
        }
        let half = n / 2;
        let (a0, a1) = self.split_at_limb(half);
        let z0 = a0.square();
        let z2 = a1.square();
        let z1 = a0
            .add(&a1)
            .square()
            .checked_sub(&z0)
            .and_then(|t| t.checked_sub(&z2))
            .expect("karatsuba square middle term is non-negative");
        let bits = (half * LIMB_BITS as usize) as u64;
        z2.shl(2 * bits).add(&z1.shl(bits)).add(&z0)
    }

    /// Returns `(low, high)` where `self == low + high·2^(64·at)`.
    fn split_at_limb(&self, at: usize) -> (Nat, Nat) {
        if at >= self.limbs.len() {
            return (self.clone(), Nat::zero());
        }
        (
            Nat::from_limbs(&self.limbs[..at]),
            Nat::from_limbs(&self.limbs[at..]),
        )
    }

    /// Karatsuba multiplication: three half-size products instead of four.
    fn mul_karatsuba(&self, rhs: &Nat) -> Nat {
        let n = self.limbs.len().max(rhs.limbs.len());
        if self.limbs.len().min(rhs.limbs.len()) < KARATSUBA_THRESHOLD {
            return self.mul_schoolbook(rhs);
        }
        let half = n / 2;
        let (a0, a1) = self.split_at_limb(half);
        let (b0, b1) = rhs.split_at_limb(half);
        let z0 = a0.mul(&b0);
        let z2 = a1.mul(&b1);
        // z1 = (a0+a1)(b0+b1) - z2 - z0
        let z1 = a0
            .add(&a1)
            .mul(&b0.add(&b1))
            .checked_sub(&z2)
            .and_then(|t| t.checked_sub(&z0))
            .expect("karatsuba middle term is non-negative");
        let bits = (half * LIMB_BITS as usize) as u64;
        z2.shl(2 * bits).add(&z1.shl(bits)).add(&z0)
    }

    /// Returns `self << bits`.
    pub fn shl(&self, bits: u64) -> Nat {
        if self.is_zero() || bits == 0 {
            return self.clone();
        }
        let limb_shift = (bits / LIMB_BITS as u64) as usize;
        let bit_shift = (bits % LIMB_BITS as u64) as u32;
        let mut out = alloc::vec![0 as Limb; limb_shift];
        if bit_shift == 0 {
            out.extend_from_slice(&self.limbs);
        } else {
            let mut carry = 0;
            for &l in &self.limbs {
                out.push((l << bit_shift) | carry);
                carry = l >> (LIMB_BITS - bit_shift);
            }
            if carry != 0 {
                out.push(carry);
            }
        }
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }

    /// Returns `self >> bits` (floor division by `2^bits`).
    pub fn shr(&self, bits: u64) -> Nat {
        if self.is_zero() || bits == 0 {
            return self.clone();
        }
        let limb_shift = (bits / LIMB_BITS as u64) as usize;
        let bit_shift = (bits % LIMB_BITS as u64) as u32;
        if limb_shift >= self.limbs.len() {
            return Nat::zero();
        }
        let src = &self.limbs[limb_shift..];
        let mut out = Vec::with_capacity(src.len());
        if bit_shift == 0 {
            out.extend_from_slice(src);
        } else {
            for i in 0..src.len() {
                let lo = src[i] >> bit_shift;
                let hi = src
                    .get(i + 1)
                    .map(|&h| h << (LIMB_BITS - bit_shift))
                    .unwrap_or(0);
                out.push(lo | hi);
            }
        }
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }

    /// Returns the greatest common divisor of `self` and `rhs`.
    ///
    /// Small operands use Stein's binary GCD; large ones use Lehmer's algorithm,
    /// which advances several Euclidean steps per multi-precision operation.
    /// `gcd(0, n) == gcd(n, 0) == n`, and `gcd(0, 0) == 0`.
    pub fn gcd(&self, rhs: &Nat) -> Nat {
        if self.is_zero() {
            return rhs.clone();
        }
        if rhs.is_zero() {
            return self.clone();
        }
        if self.limbs.len().max(rhs.limbs.len()) < LEHMER_THRESHOLD {
            self.gcd_binary(rhs)
        } else {
            self.gcd_lehmer(rhs)
        }
    }

    /// Stein's binary GCD (no division). Precondition: both operands non-zero.
    fn gcd_binary(&self, rhs: &Nat) -> Nat {
        let mut u = self.clone();
        let mut v = rhs.clone();
        let shift = u.trailing_zeros().min(v.trailing_zeros());
        u = u.shr(u.trailing_zeros());
        v = v.shr(v.trailing_zeros());
        loop {
            v = v.shr(v.trailing_zeros());
            if u.cmp_ref(&v) == Ordering::Greater {
                core::mem::swap(&mut u, &mut v);
            }
            v = v
                .checked_sub(&u)
                .expect("binary gcd: v >= u by construction");
            if v.is_zero() {
                break;
            }
        }
        u.shl(shift)
    }

    /// Lehmer's GCD (Knuth TAOCP §4.5.2, Algorithm L): use the leading words to
    /// derive a 2×2 cofactor matrix in single precision, then apply it to the
    /// full operands, doing far fewer multi-precision divisions than plain
    /// Euclid. Precondition: both operands non-zero.
    fn gcd_lehmer(&self, rhs: &Nat) -> Nat {
        use crate::int::Int;

        let mut u = self.clone();
        let mut v = rhs.clone();
        if u.cmp_ref(&v) == Ordering::Less {
            core::mem::swap(&mut u, &mut v);
        }
        while v.limbs.len() > 1 {
            // Leading ~63 bits of u, and of v at the same alignment.
            let shift = u.bit_len().saturating_sub(63);
            let mut x = u.shr(shift).to_u64().unwrap_or(0);
            let mut y = v.shr(shift).to_u64().unwrap_or(0);

            // Single-precision partial Euclid, accumulating [[a,b],[c,d]].
            let (mut a, mut b, mut c, mut d) = (1i128, 0i128, 0i128, 1i128);
            loop {
                let (yc, yd) = (y as i128 + c, y as i128 + d);
                if yc == 0 || yd == 0 {
                    break;
                }
                let q = (x as i128 + a) / yc;
                if q != (x as i128 + b) / yd {
                    break; // Lehmer's exactness test failed
                }
                let (na, nb) = (c, d);
                (c, d) = (a - q * c, b - q * d);
                (a, b) = (na, nb);
                let ny = x as i128 - q * y as i128;
                x = y;
                y = ny as u64;
            }

            if b == 0 {
                // No single-precision progress: one full division step.
                let (_, r) = u.div_rem(&v).expect("v is non-zero");
                u = core::mem::replace(&mut v, r);
            } else {
                // Apply the matrix to the full operands (result stays positive).
                let (ui, vi) = (Int::from(u.clone()), Int::from(v.clone()));
                let nu = lincomb(a, &ui, b, &vi);
                let nv = lincomb(c, &ui, d, &vi);
                u = nu.magnitude();
                v = nv.magnitude();
                if u.cmp_ref(&v) == Ordering::Less {
                    core::mem::swap(&mut u, &mut v);
                }
            }
        }
        // v now fits a single limb: finish in machine words.
        if v.is_zero() {
            return u;
        }
        let vr = v.limbs[0];
        let ur = u.divmod_small(vr).1;
        Nat::from_u64(u64_gcd(vr, ur))
    }

    /// Returns bit `i` (0 = least significant), or `false` past the top.
    #[inline]
    pub fn bit(&self, i: u64) -> bool {
        let limb = (i / LIMB_BITS as u64) as usize;
        match self.limbs.get(limb) {
            Some(&l) => (l >> (i % LIMB_BITS as u64)) & 1 == 1,
            None => false,
        }
    }

    /// Divides by `rhs`, returning `(quotient, remainder)` with
    /// `self == quotient·rhs + remainder` and `remainder < rhs`, or `None` if
    /// `rhs` is zero.
    ///
    /// Dispatches to single-limb division or Knuth's Algorithm D (TAOCP Vol. 2
    /// §4.3.1). Sub-quadratic Burnikel–Ziegler recursive division is a later
    /// milestone; see `ROADMAP.md`.
    pub fn div_rem(&self, rhs: &Nat) -> Option<(Nat, Nat)> {
        if rhs.is_zero() {
            return None;
        }
        match self.cmp_ref(rhs) {
            Ordering::Less => return Some((Nat::zero(), self.clone())),
            Ordering::Equal => return Some((Nat::one(), Nat::zero())),
            Ordering::Greater => {}
        }
        if rhs.limbs.len() == 1 {
            let (q, r) = self.divmod_small(rhs.limbs[0]);
            return Some((q, Nat::from_u64(r)));
        }
        if rhs.limbs.len() >= BZ_THRESHOLD {
            return Some(bz_div_rem(self, rhs));
        }
        Some(self.div_rem_knuth(rhs))
    }

    /// Knuth Algorithm D: schoolbook long division in base `2^64`, with a
    /// normalized divisor and the 2-by-1 limb quotient estimate. Precondition:
    /// `rhs` has ≥ 2 limbs and `self > rhs`.
    fn div_rem_knuth(&self, rhs: &Nat) -> (Nat, Nat) {
        const B: u128 = 1 << LIMB_BITS;
        let n = rhs.limbs.len();
        let m = self.limbs.len() - n;

        // Normalize so the divisor's top limb has its high bit set.
        let shift = rhs.limbs[n - 1].leading_zeros();
        let vn = rhs.shl(shift as u64);
        let vv = &vn.limbs;
        debug_assert_eq!(vv.len(), n);
        let un = self.shl(shift as u64);
        let mut u = un.limbs.clone();
        u.resize(self.limbs.len() + 1, 0); // exactly m + n + 1 limbs

        let (b1, b2) = (vv[n - 1] as u128, vv[n - 2] as u128);
        let mut q = alloc::vec![0 as Limb; m + 1];

        for j in (0..=m).rev() {
            // Estimate the quotient limb from the top two dividend limbs.
            let num = ((u[j + n] as u128) << LIMB_BITS) | u[j + n - 1] as u128;
            let mut qhat = num / b1;
            let mut rhat = num % b1;
            while qhat >= B || qhat * b2 > ((rhat << LIMB_BITS) | u[j + n - 2] as u128) {
                qhat -= 1;
                rhat += b1;
                if rhat >= B {
                    break;
                }
            }

            // Multiply and subtract: u[j..=j+n] -= qhat · vv.
            let mut carry: u128 = 0;
            let mut borrow: i64 = 0;
            for i in 0..n {
                let p = qhat * vv[i] as u128 + carry;
                carry = p >> LIMB_BITS;
                let d = (u[j + i] as i128) - ((p as u64) as i128) - (borrow as i128);
                u[j + i] = d as u64;
                borrow = if d < 0 { 1 } else { 0 };
            }
            let d = (u[j + n] as i128) - (carry as i128) - (borrow as i128);
            u[j + n] = d as u64;

            q[j] = qhat as Limb;
            if d < 0 {
                // qhat was one too large: add the divisor back.
                q[j] -= 1;
                let mut add_carry: u128 = 0;
                for i in 0..n {
                    let s = u[j + i] as u128 + vv[i] as u128 + add_carry;
                    u[j + i] = s as u64;
                    add_carry = s >> LIMB_BITS;
                }
                u[j + n] = (u[j + n] as u128 + add_carry) as u64;
            }
        }

        let mut quotient = Nat { limbs: q };
        quotient.normalize();
        // Denormalize the remainder (the low n limbs of u), undoing the shift.
        let remainder = Nat::from_limbs(&u[..n]).shr(shift as u64);
        (quotient, remainder)
    }

    /// Divides by a single-limb value, returning `(quotient, remainder)`.
    ///
    /// The divisor must be non-zero. This is the primitive behind decimal
    /// formatting; full multi-limb division is a later milestone.
    fn divmod_small(&self, d: Limb) -> (Nat, Limb) {
        debug_assert!(d != 0, "divmod_small by zero");
        let dd = d as u128;
        let mut rem: u128 = 0;
        let mut q = alloc::vec![0 as Limb; self.limbs.len()];
        for i in (0..self.limbs.len()).rev() {
            let cur = (rem << LIMB_BITS) | self.limbs[i] as u128;
            q[i] = (cur / dd) as Limb;
            rem = cur % dd;
        }
        let mut n = Nat { limbs: q };
        n.normalize();
        (n, rem as Limb)
    }

    /// Computes `self · mul + add`, where `mul` and `add` are single limbs.
    ///
    /// This is the primitive behind decimal parsing (`n·10 + digit`).
    fn mul_add_small(&self, mul: Limb, add: Limb) -> Nat {
        let mut out = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry = add as u128;
        for &l in &self.limbs {
            let t = l as u128 * mul as u128 + carry;
            out.push(t as Limb);
            carry = t >> LIMB_BITS;
        }
        while carry != 0 {
            out.push(carry as Limb);
            carry >>= LIMB_BITS;
        }
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }
}

impl Nat {
    /// Returns the value as a `u64` if it fits in a single limb.
    pub fn to_u64(&self) -> Option<u64> {
        match self.limbs.as_slice() {
            [] => Some(0),
            &[only] => Some(only),
            _ => None,
        }
    }

    /// Returns `true` if this value is one.
    #[inline]
    pub fn is_one(&self) -> bool {
        self.limbs.as_slice() == [1]
    }

    /// Returns the little-endian limb slice of the magnitude, normalized so the
    /// most-significant limb is non-zero (empty for zero).
    #[inline]
    pub fn as_limbs(&self) -> &[Limb] {
        &self.limbs
    }

    /// Builds a natural from little-endian limbs (any trailing zeros are
    /// stripped).
    pub fn from_limbs(limbs: &[Limb]) -> Nat {
        let mut n = Nat {
            limbs: limbs.to_vec(),
        };
        n.normalize();
        n
    }

    /// Builds a natural from little-endian bytes.
    pub fn from_bytes_le(bytes: &[u8]) -> Nat {
        let mut limbs = Vec::with_capacity(bytes.len() / 8 + 1);
        for chunk in bytes.chunks(8) {
            let mut limb: Limb = 0;
            for (i, &b) in chunk.iter().enumerate() {
                limb |= (b as Limb) << (8 * i);
            }
            limbs.push(limb);
        }
        let mut n = Nat { limbs };
        n.normalize();
        n
    }

    /// Returns the magnitude as little-endian bytes (no trailing zero bytes;
    /// empty for zero).
    pub fn to_bytes_le(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.limbs.len() * 8);
        for &limb in &self.limbs {
            out.extend_from_slice(&limb.to_le_bytes());
        }
        while matches!(out.last(), Some(&0)) {
            out.pop();
        }
        out
    }

    /// Returns the low `k` bits of this value, i.e. `self mod 2^k`.
    pub fn low_bits(&self, k: u64) -> Nat {
        if k == 0 {
            return Nat::zero();
        }
        let full = (k / LIMB_BITS as u64) as usize;
        let rem = (k % LIMB_BITS as u64) as u32;
        let take = full.min(self.limbs.len());
        let mut out: Vec<Limb> = self.limbs[..take].to_vec();
        if rem > 0 && full < self.limbs.len() {
            while out.len() < full {
                out.push(0);
            }
            out.push(self.limbs[full] & ((1u64 << rem) - 1));
        }
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }

    /// Returns `self` raised to `exp` (`self^0 == 1`), by square-and-multiply.
    pub fn pow(&self, exp: u32) -> Nat {
        let mut result = Nat::one();
        let mut base = self.clone();
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul(&base);
            }
            e >>= 1;
            if e > 0 {
                base = base.square();
            }
        }
        result
    }

    /// Returns the floor of the square root, `⌊√self⌋`, via Newton's method.
    pub fn isqrt(&self) -> Nat {
        if self.is_zero() {
            return Nat::zero();
        }
        if self.bit_len() <= 2 {
            // values 1..=3 all have isqrt 1
            return Nat::one();
        }
        let mut x = Nat::one().shl(self.bit_len().div_ceil(2));
        loop {
            let (q, _) = self.div_rem(&x).expect("x is non-zero");
            let y = x.add(&q).shr(1);
            if y.cmp_ref(&x) != Ordering::Less {
                return x;
            }
            x = y;
        }
    }

    /// Returns the floor of the `k`th root, `⌊self^(1/k)⌋`, for `k >= 1`, by
    /// bitwise binary search.
    pub fn nth_root_floor(&self, k: u32) -> Nat {
        assert!(k >= 1, "nth_root_floor: k must be >= 1");
        if k == 1 || self.is_zero() || self.is_one() {
            return self.clone();
        }
        if k == 2 {
            return self.isqrt();
        }
        let hb = self.bit_len().div_ceil(k as u64);
        let mut root = Nat::zero();
        for bit in (0..=hb).rev() {
            let cand = root.add(&Nat::one().shl(bit));
            if cand.pow(k).cmp_ref(self) != Ordering::Greater {
                root = cand;
            }
        }
        root
    }

    /// Writes the magnitude in the given `radix` (2–36) to `out`.
    pub fn write_radix(&self, out: &mut impl fmt::Write, radix: u32) -> fmt::Result {
        assert!((2..=36).contains(&radix), "radix must be in 2..=36");
        if self.is_zero() {
            return out.write_str("0");
        }
        out.write_str(&self.to_radix_string(radix))
    }

    /// Returns the minimal (no leading zeros) base-`radix` digit string, using
    /// divide-and-conquer: split off a divisor `radix^len ≈ √self`, recurse on
    /// the quotient and remainder, and zero-pad the remainder to `len` digits.
    /// With sub-quadratic multiplication/division this is `O(M(n)·log n)`.
    fn to_radix_string(&self, radix: u32) -> String {
        // Base case: a few limbs go straight through single-limb division.
        if self.limbs.len() <= RADIX_RECURSION_LIMBS {
            return simple_radix_string(self, radix);
        }
        // Build `p = radix^(2^k)` up to ≈ √self (largest with p·p ≤ self).
        let mut p = Nat::from_u64(radix as u64);
        let mut len: usize = 1;
        loop {
            let sq = p.mul(&p);
            if sq.cmp_ref(self) == Ordering::Greater {
                break;
            }
            p = sq;
            len *= 2;
        }
        let (q, r) = self.div_rem(&p).expect("p is non-zero");
        let mut s = q.to_radix_string(radix);
        let r_str = if r.is_zero() {
            String::new()
        } else {
            r.to_radix_string(radix)
        };
        // Zero-pad the low part to exactly `len` digits.
        for _ in 0..len - r_str.len() {
            s.push('0');
        }
        s.push_str(&r_str);
        s
    }
}

/// Number of limbs at or below which radix conversion uses the simple
/// single-limb-division loop rather than recursing.
const RADIX_RECURSION_LIMBS: usize = 3;

/// Minimal base-`radix` digit string via repeated single-limb division (for
/// small values / the recursion base case).
fn simple_radix_string(n: &Nat, radix: u32) -> String {
    if n.is_zero() {
        return String::new();
    }
    let mut n = n.clone();
    let mut buf = Vec::new();
    while !n.is_zero() {
        let (q, d) = n.divmod_small(radix as Limb);
        buf.push(digit_char(d as u32));
        n = q;
    }
    buf.reverse();
    String::from_utf8(buf).unwrap_or_default()
}

/// Maps a digit value `0..36` to its ASCII character (`0-9`, then `a-z`).
#[inline]
fn digit_char(d: u32) -> u8 {
    if d < 10 {
        b'0' + d as u8
    } else {
        b'a' + (d - 10) as u8
    }
}

/// Parses an unsigned integer in the given `radix` (2–36).
pub(crate) fn parse_radix(s: &str, radix: u32) -> Result<Nat> {
    if !(2..=36).contains(&radix) || s.is_empty() {
        return Err(Error::Parse);
    }
    let mut n = Nat::zero();
    for ch in s.chars() {
        let d = ch.to_digit(radix).ok_or(Error::Parse)?;
        n = n.mul_add_small(radix as Limb, d as Limb);
    }
    Ok(n)
}

impl Nat {
    /// Returns `self^exp mod modulus`. Panics if `modulus` is zero.
    ///
    /// Odd moduli use Montgomery reduction; others fall back to
    /// square-and-multiply with a division-based reduction.
    pub fn modpow(&self, exp: &Nat, modulus: &Nat) -> Nat {
        assert!(!modulus.is_zero(), "modpow: zero modulus");
        if modulus.is_one() {
            return Nat::zero();
        }
        if !modulus.is_even() && modulus.limbs.len() >= 2 {
            self.modpow_montgomery(exp, modulus)
        } else {
            self.modpow_simple(exp, modulus)
        }
    }

    /// Square-and-multiply with a division-based reduction after each step.
    fn modpow_simple(&self, exp: &Nat, modulus: &Nat) -> Nat {
        let mut result = Nat::one();
        let mut base = self.div_rem(modulus).expect("non-zero modulus").1;
        let bits = exp.bit_len();
        for i in 0..bits {
            if exp.bit(i) {
                result = result.mul(&base).div_rem(modulus).expect("non-zero").1;
            }
            if i + 1 < bits {
                base = base.square().div_rem(modulus).expect("non-zero").1;
            }
        }
        result
    }

    /// Montgomery-reduction modpow for an odd `modulus > 1`.
    fn modpow_montgomery(&self, exp: &Nat, modulus: &Nat) -> Nat {
        use crate::int::Int;

        let k = modulus.limbs.len();
        let rbits = k as u64 * LIMB_BITS as u64;
        // R = 2^(64k); m' = −m^{-1} mod R; R² mod m for conversion into the domain.
        let r = Nat::one().shl(rbits);
        let minv = Int::from(modulus.clone())
            .modinv(&Int::from(r.clone()))
            .expect("odd modulus is invertible mod 2^k")
            .magnitude();
        let m_prime = r.checked_sub(&minv).unwrap_or_else(Nat::zero);
        let r2 = r.mul(&r).div_rem(modulus).expect("non-zero").1;

        // REDC(t) = (t + ((t mod R)·m' mod R)·m) / R, conditionally reduced.
        let redc = |t: &Nat| -> Nat {
            let u = t.low_bits(rbits).mul(&m_prime).low_bits(rbits);
            let s = t.add(&u.mul(modulus)).shr(rbits);
            if s.cmp_ref(modulus) != Ordering::Less {
                s.checked_sub(modulus).expect("s >= m")
            } else {
                s
            }
        };

        let base_mod = self.div_rem(modulus).expect("non-zero").1;
        let mut base = redc(&base_mod.mul(&r2)); // into Montgomery form
        let mut result = r.div_rem(modulus).expect("non-zero").1; // 1 in Montgomery form
        let bits = exp.bit_len();
        for i in 0..bits {
            if exp.bit(i) {
                result = redc(&result.mul(&base));
            }
            if i + 1 < bits {
                base = redc(&base.square());
            }
        }
        redc(&result) // back out of Montgomery form
    }

    /// Returns the smallest prime strictly greater than `self`, found by
    /// scanning odd candidates with the Miller–Rabin test.
    pub fn next_prime(&self, rng: &mut impl crate::random::RandomSource) -> Nat {
        let two = Nat::from_u64(2);
        if self.cmp_ref(&two) == Ordering::Less {
            return two; // next prime after 0 or 1
        }
        let mut c = self.add(&Nat::one());
        if c.is_even() {
            c = c.add(&Nat::one()); // start at an odd candidate ≥ 3
        }
        loop {
            if c.is_probable_prime(40, rng) {
                return c;
            }
            c = c.add(&two);
        }
    }

    /// Returns the largest prime strictly less than `self`, or `None` if there
    /// is none (`self <= 2`).
    pub fn prev_prime(&self, rng: &mut impl crate::random::RandomSource) -> Option<Nat> {
        let two = Nat::from_u64(2);
        if self.cmp_ref(&two) != Ordering::Greater {
            return None;
        }
        if self.cmp_ref(&Nat::from_u64(3)) == Ordering::Equal {
            return Some(two);
        }
        let mut c = self.checked_sub(&Nat::one()).unwrap();
        if c.is_even() {
            c = c.checked_sub(&Nat::one()).unwrap();
        }
        loop {
            if c.cmp_ref(&two) == Ordering::Less {
                return Some(two);
            }
            if c.is_probable_prime(40, rng) {
                return Some(c);
            }
            c = c.checked_sub(&two).unwrap_or_else(Nat::zero);
        }
    }

    /// Miller–Rabin probable-primality test with `rounds` random witnesses.
    ///
    /// Deterministic for the tiny cases; for larger `self` the probability of a
    /// composite passing is at most `4^-rounds`.
    pub fn is_probable_prime(
        &self,
        rounds: u32,
        rng: &mut impl crate::random::RandomSource,
    ) -> bool {
        let two = Nat::from_u64(2);
        let three = Nat::from_u64(3);
        if self.cmp_ref(&two) == Ordering::Less {
            return false;
        }
        if self.cmp_ref(&three) != Ordering::Greater {
            return true; // 2 or 3
        }
        if self.is_even() {
            return false;
        }
        let one = Nat::one();
        let n1 = self.checked_sub(&one).expect("self >= 1");
        let s = n1.trailing_zeros();
        let d = n1.shr(s);
        let n3 = self.checked_sub(&three).expect("self >= 3");

        'witness: for _ in 0..rounds {
            let a = two.add(&Nat::random_below(&n3, rng).unwrap_or_else(Nat::zero));
            let mut x = a.modpow(&d, self);
            if x == one || x == n1 {
                continue;
            }
            for _ in 1..s {
                x = x.square().div_rem(self).expect("non-zero").1;
                if x == n1 {
                    continue 'witness;
                }
            }
            return false; // definitely composite
        }
        true
    }
}

/// Binary GCD on two machine words.
pub fn u64_gcd(mut u: u64, mut v: u64) -> u64 {
    if u == 0 {
        return v;
    }
    if v == 0 {
        return u;
    }
    let shift = (u | v).trailing_zeros();
    u >>= u.trailing_zeros();
    loop {
        v >>= v.trailing_zeros();
        if u > v {
            core::mem::swap(&mut u, &mut v);
        }
        v -= u;
        if v == 0 {
            break;
        }
    }
    u << shift
}

/// Binary GCD on two 32-bit machine words.
#[inline]
pub fn u_gcd(u: u32, v: u32) -> u32 {
    u64_gcd(u as u64, v as u64) as u32
}

impl PartialOrd for Nat {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Nat {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_ref(other)
    }
}

impl From<u64> for Nat {
    #[inline]
    fn from(v: u64) -> Self {
        Nat::from_u64(v)
    }
}

impl From<u128> for Nat {
    #[inline]
    fn from(v: u128) -> Self {
        Nat::from_u128(v)
    }
}

impl FromStr for Nat {
    type Err = Error;

    /// Parses a non-negative decimal integer. An empty string, or any character
    /// that is not an ASCII digit, is a [`Error::Parse`].
    fn from_str(s: &str) -> Result<Self> {
        if s.is_empty() {
            return Err(Error::Parse);
        }
        let mut n = Nat::zero();
        for b in s.bytes() {
            if !b.is_ascii_digit() {
                return Err(Error::Parse);
            }
            n = n.mul_add_small(10, (b - b'0') as Limb);
        }
        Ok(n)
    }
}

impl fmt::Display for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        f.write_str(&self.to_radix_string(10))
    }
}

impl fmt::LowerHex for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let mut it = self.limbs.iter().rev();
        write!(f, "{:x}", it.next().expect("non-empty checked above"))?;
        for limb in it {
            write!(f, "{limb:016x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Nat({self})")
    }
}

impl core::ops::Add for &Nat {
    type Output = Nat;
    #[inline]
    fn add(self, rhs: &Nat) -> Nat {
        Nat::add(self, rhs)
    }
}

impl core::ops::Mul for &Nat {
    type Output = Nat;
    #[inline]
    fn mul(self, rhs: &Nat) -> Nat {
        Nat::mul(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    /// Reference bit-at-a-time long division, kept only for differential testing
    /// against the production Algorithm-D path.
    fn div_rem_binary(a: &Nat, b: &Nat) -> (Nat, Nat) {
        assert!(!b.is_zero());
        if a.cmp_ref(b) == Ordering::Less {
            return (Nat::zero(), a.clone());
        }
        let one = Nat::one();
        let mut q = Nat::zero();
        let mut r = Nat::zero();
        for i in (0..a.bit_len()).rev() {
            r = r.shl(1);
            if a.bit(i) {
                r = r.add(&one);
            }
            q = q.shl(1);
            if r.cmp_ref(b) != Ordering::Less {
                r = r.checked_sub(b).unwrap();
                q = q.add(&one);
            }
        }
        (q, r)
    }

    fn n(s: &str) -> Nat {
        Nat::from_str(s).unwrap()
    }

    #[test]
    fn knuth_matches_binary_reference() {
        // A spread of dividend/divisor sizes, including multi-limb divisors,
        // exact multiples, and near-boundary values.
        let cases = [
            (
                "340282366920938463463374607431768211456",
                "18446744073709551616",
            ),
            (
                "123456789012345678901234567890123456789",
                "98765432109876543210",
            ),
            ("100000000000000000000000000000000000000", "3"),
            (
                "18446744073709551617000000000000000000000",
                "18446744073709551617",
            ),
            (
                "999999999999999999999999999999999999999999",
                "1000000000000000000001",
            ),
        ];
        for (a_s, b_s) in cases.iter() {
            let (a, b) = (n(a_s), n(b_s));
            let (q, r) = a.div_rem(&b).unwrap();
            let (rq, rr) = div_rem_binary(&a, &b);
            assert_eq!(q, rq, "quotient {a_s}/{b_s}");
            assert_eq!(r, rr, "remainder {a_s}/{b_s}");
            // Reconstruction and range.
            assert_eq!(q.mul(&b).add(&r), a);
            assert!(r.cmp_ref(&b) == Ordering::Less);
        }
    }

    #[test]
    fn ntt_matches_toom3() {
        // NTT multiplication must agree with the (verified) Toom-3 path, and
        // with a value computed a different way.
        let p = Nat::from_u64(10).pow(4000); // ~13k bits, ~208 limbs
        let q = Nat::from_u64(10).pow(4100);
        let mut expected = String::from("1");
        expected.push_str(&"0".repeat(8100));
        assert_eq!(mul_ntt(&p, &q), Nat::from_str(&expected).unwrap());

        let mut state = 0x0f0f_1234_dead_beefu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let build = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let bytes: Vec<u8> = (0..cnt * 8).map(|_| f() as u8).collect();
            Nat::from_bytes_le(&bytes)
        };
        for _ in 0..8 {
            let a = build(200 + (next() % 400) as usize, &mut next);
            let b = build(200 + (next() % 400) as usize, &mut next);
            assert_eq!(mul_ntt(&a, &b), a.mul_toom3(&b), "NTT vs Toom-3 mismatch");
        }
    }

    #[test]
    fn burnikel_ziegler_matches_knuth() {
        // Differential: BZ recursive division must match Knuth Algorithm D over
        // random large operands, and satisfy a == q·b + r with r < b.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let build = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let bytes: Vec<u8> = (0..cnt * 8).map(|_| f() as u8).collect();
            Nat::from_bytes_le(&bytes)
        };
        for _ in 0..25 {
            // Divisor 70–110 limbs (crosses BZ recursion), dividend larger.
            let b = build(70 + (next() % 40) as usize, &mut next);
            let extra = build(30 + (next() % 90) as usize, &mut next);
            let a = b.mul(&extra).add(&build(40, &mut next));
            if b.is_zero() || a.cmp_ref(&b) != Ordering::Greater {
                continue;
            }
            let (q_bz, r_bz) = bz_div_rem(&a, &b);
            let (q_kn, r_kn) = a.div_rem_knuth(&b);
            assert_eq!(q_bz, q_kn, "BZ quotient mismatch");
            assert_eq!(r_bz, r_kn, "BZ remainder mismatch");
            assert_eq!(q_bz.mul(&b).add(&r_bz), a);
            assert!(r_bz.cmp_ref(&b) == Ordering::Less);
        }
    }

    #[test]
    fn montgomery_matches_simple_modpow() {
        // Montgomery-reduction modpow must match the division-based version for
        // random bases/exponents and odd moduli of assorted sizes.
        let mut state = 0xabcd_1234_5678_9999u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let build = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let bytes: Vec<u8> = (0..cnt * 8).map(|_| f() as u8).collect();
            Nat::from_bytes_le(&bytes)
        };
        for _ in 0..20 {
            let base = build(2 + (next() % 8) as usize, &mut next);
            let exp = build(1 + (next() % 4) as usize, &mut next);
            let mut m = build(2 + (next() % 6) as usize, &mut next);
            if m.is_even() {
                m = m.add(&Nat::one()); // make odd
            }
            if m.limbs.len() < 2 {
                continue;
            }
            assert_eq!(
                base.modpow_montgomery(&exp, &m),
                base.modpow_simple(&exp, &m),
                "montgomery vs simple modpow"
            );
        }
    }

    #[test]
    fn lehmer_matches_binary_gcd() {
        // Deterministic pseudo-random large pairs; Lehmer must match binary GCD.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..40 {
            // Build multi-limb operands (20–40 limbs) so Lehmer is exercised.
            let build = |cnt: usize, f: &mut dyn FnMut() -> u64| {
                let bytes: Vec<u8> = (0..cnt * 8).map(|_| f() as u8).collect();
                Nat::from_bytes_le(&bytes)
            };
            let a = build(20 + (next() % 20) as usize, &mut next);
            let b = build(20 + (next() % 20) as usize, &mut next);
            if a.is_zero() || b.is_zero() {
                continue;
            }
            let g_lehmer = a.gcd_lehmer(&b);
            let g_binary = a.gcd_binary(&b);
            assert_eq!(g_lehmer, g_binary, "gcd mismatch");
            // g divides both.
            assert!(a.div_rem(&g_lehmer).unwrap().1.is_zero());
            assert!(b.div_rem(&g_lehmer).unwrap().1.is_zero());
        }
        // A case with a large known common factor.
        let common = Nat::from_u64(10).pow(50);
        let a = common.mul(&Nat::from_u64(7).pow(30));
        let b = common.mul(&Nat::from_u64(11).pow(25));
        assert_eq!(a.gcd_lehmer(&b), common);
    }

    #[test]
    fn knuth_stress_products() {
        // Build large values and divide, checking the identity and the
        // multi-limb divisor path (10^k has many limbs).
        let ten_k = Nat::from_u64(10).pow(60); // ~200 bits, several limbs
        let big = Nat::from_u64(7).pow(200);
        let (q, r) = big.div_rem(&ten_k).unwrap();
        assert_eq!(q.mul(&ten_k).add(&r), big);
        assert!(r.cmp_ref(&ten_k) == Ordering::Less);

        // Exact division: (a*b)/b == a, remainder 0.
        let a = Nat::from_u64(3).pow(150);
        let b = Nat::from_u64(11).pow(80);
        let prod = a.mul(&b);
        let (q2, r2) = prod.div_rem(&b).unwrap();
        assert_eq!(q2, a);
        assert!(r2.is_zero());
    }
}
