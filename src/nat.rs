//! Arbitrary-precision natural numbers (unsigned integers).
//!
//! [`Nat`] stores magnitude as a little-endian `Vec` of 64-bit limbs,
//! kept in a canonical form with no trailing zero limbs (so the value zero is
//! the empty vector). That canonical form makes equality and ordering cheap and
//! lets the derived [`PartialEq`]/[`Eq`] be correct.
//!
//! This is the layer that carries the heavy limb-level algorithms: addition,
//! subtraction, multiplication (schoolbook with a Karatsuba path), division
//! (single-limb plus Knuth Algorithm D), shifts, binary GCD, roots, and radix
//! I/O. Further sub-quadratic work (Toom/FFT multiplication, Burnikel–Ziegler
//! division) is future performance work — see `ROADMAP.md`.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::limb::{LIMB_BITS, Limb, adc, mac, sbb};

/// Operands with fewer than this many limbs use schoolbook multiplication;
/// larger ones recurse via Karatsuba. Chosen conservatively; a tuned crossover
/// is a later milestone (see `ROADMAP.md`).
const KARATSUBA_THRESHOLD: usize = 32;

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
        if self.limbs.len().min(rhs.limbs.len()) < KARATSUBA_THRESHOLD {
            self.mul_schoolbook(rhs)
        } else {
            self.mul_karatsuba(rhs)
        }
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

    /// Returns the greatest common divisor of `self` and `rhs`, via Stein's
    /// binary GCD (no division required).
    ///
    /// `gcd(0, n) == gcd(n, 0) == n`, and `gcd(0, 0) == 0`.
    pub fn gcd(&self, rhs: &Nat) -> Nat {
        if self.is_zero() {
            return rhs.clone();
        }
        if rhs.is_zero() {
            return self.clone();
        }
        let mut u = self.clone();
        let mut v = rhs.clone();
        let shift = u.trailing_zeros().min(v.trailing_zeros());
        u = u.shr(u.trailing_zeros());
        v = v.shr(v.trailing_zeros());
        // Invariant: `u` is odd at the top of every iteration.
        loop {
            v = v.shr(v.trailing_zeros());
            // Both odd here; keep the smaller in `u`.
            if u.cmp_ref(&v) == Ordering::Greater {
                core::mem::swap(&mut u, &mut v);
            }
            // v >= u, so this subtraction never underflows.
            v = v
                .checked_sub(&u)
                .expect("binary gcd: v >= u by construction");
            if v.is_zero() {
                break;
            }
        }
        u.shl(shift)
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
                base = base.mul(&base);
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
        let mut n = self.clone();
        let mut buf = Vec::new();
        while !n.is_zero() {
            let (q, r) = n.divmod_small(radix as Limb);
            buf.push(digit_char(r as u32));
            n = q;
        }
        buf.reverse();
        out.write_str(core::str::from_utf8(&buf).unwrap_or("<nan>"))
    }
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
        // Repeated single-limb division by ten. Chunking by 10^19 is a later
        // performance milestone; correctness first.
        let mut n = self.clone();
        let mut buf = Vec::new();
        while !n.is_zero() {
            let (q, r) = n.divmod_small(10);
            buf.push(b'0' + r as u8);
            n = q;
        }
        buf.reverse();
        // Every byte is an ASCII digit, so this is valid UTF-8.
        f.write_str(core::str::from_utf8(&buf).unwrap_or("<nan>"))
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
