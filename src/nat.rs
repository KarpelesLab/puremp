//! Arbitrary-precision natural numbers (unsigned integers).
//!
//! [`Nat`] stores magnitude as a little-endian `Vec` of 64-bit limbs,
//! kept in a canonical form with no trailing zero limbs (so the value zero is
//! the empty vector). That canonical form makes equality and ordering cheap and
//! lets the derived [`PartialEq`]/[`Eq`] be correct.
//!
//! This is the layer that carries the heavy limb-level algorithms. The scaffold
//! ships the quadratic-time schoolbook routines and the pieces the higher layers
//! already need (addition, subtraction, multiplication, shifts, binary GCD,
//! decimal I/O); sub-quadratic multiplication and division land in later
//! milestones — see `ROADMAP.md`.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::limb::{LIMB_BITS, Limb, adc, mac, sbb};

/// An arbitrary-precision natural number (a non-negative integer).
#[derive(Clone, PartialEq, Eq, Default)]
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

    /// Returns `self · rhs` using the quadratic schoolbook algorithm.
    ///
    /// Sub-quadratic multiplication (Karatsuba, Toom-Cook, FFT) is a later
    /// milestone; see `ROADMAP.md`.
    pub fn mul(&self, rhs: &Nat) -> Nat {
        if self.is_zero() || rhs.is_zero() {
            return Nat::zero();
        }
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
    /// The scaffold implementation is bit-at-a-time long division — simple and
    /// obviously correct. Sub-quadratic schemes (Knuth Algorithm D, then
    /// Burnikel–Ziegler recursive division) are later milestones; see
    /// `ROADMAP.md`.
    pub fn div_rem(&self, rhs: &Nat) -> Option<(Nat, Nat)> {
        if rhs.is_zero() {
            return None;
        }
        if self.cmp_ref(rhs) == Ordering::Less {
            return Some((Nat::zero(), self.clone()));
        }
        let one = Nat::one();
        let mut q = Nat::zero();
        let mut r = Nat::zero();
        for i in (0..self.bit_len()).rev() {
            r = r.shl(1);
            if self.bit(i) {
                r = r.add(&one);
            }
            q = q.shl(1);
            if r.cmp_ref(rhs) != Ordering::Less {
                r = r.checked_sub(rhs).expect("r >= rhs checked");
                q = q.add(&one);
            }
        }
        Some((q, r))
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
