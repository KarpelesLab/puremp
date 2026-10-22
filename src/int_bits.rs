//! Bit-level operations on [`Int`] in the infinite two's-complement view (the
//! one [`Int::bitand`]/[`Int::bitor`]/[`Int::bitxor`] use), rounding shifts, and
//! byte import/export.
//!
//! A negative `x` is read as the infinite bit string `…111·(¬(|x|−1))`: its
//! bits are the complements of the bits of `|x| − 1`, so every query below
//! reduces to a question about a non-negative magnitude. This matches GMP's
//! `mpz_tstbit`/`mpz_setbit`/`mpz_popcount`/`mpz_scan0`/… semantics.
//!
//! - [`Int::test_bit`], [`Int::set_bit`], [`Int::clear_bit`],
//!   [`Int::complement_bit`] — single-bit access.
//! - [`Int::count_ones`], [`Int::hamming_distance`], [`Int::scan0`],
//!   [`Int::scan1`] — population count and bit search.
//! - [`Int::div_2k_floor`], [`Int::div_2k_ceil`], [`Int::rem_2k_trunc`] —
//!   power-of-two division with the other roundings (`>>` is the floor one).
//! - [`Int::to_bytes_be`]/[`Int::from_bytes_be`] (and `_le`) — magnitude
//!   bytes; [`Int::to_signed_bytes_be`]/[`Int::from_signed_bytes_be`] (and
//!   `_le`) — two's-complement bytes.

use alloc::vec::Vec;

use crate::int::{Int, Sign};
use crate::nat::Nat;

/// Index of the first bit `≥ start` of the little-endian `limbs` equal to
/// `want`; bits past the top are `0`, so a search for `0` always succeeds.
fn scan_limbs(limbs: &[u64], start: u64, want: bool) -> Option<u64> {
    let mut i = (start / 64) as usize;
    if i >= limbs.len() {
        return (!want).then_some(start);
    }
    let flip = if want { 0 } else { u64::MAX };
    // Mask off the bits below `start` in its limb.
    let mut w = (limbs[i] ^ flip) & (u64::MAX << (start % 64));
    loop {
        if w != 0 {
            return Some(i as u64 * 64 + w.trailing_zeros() as u64);
        }
        i += 1;
        if i == limbs.len() {
            return (!want).then_some(i as u64 * 64);
        }
        w = limbs[i] ^ flip;
    }
}

/// Population count of little-endian limbs.
fn popcount_limbs(limbs: &[u64]) -> u64 {
    limbs.iter().map(|l| l.count_ones() as u64).sum()
}

/// Returns `x` with bit `i` set (`Some(true)`), cleared (`Some(false)`) or
/// complemented (`None`).
fn with_bit(x: &Nat, i: u64, op: Option<bool>) -> Nat {
    let limb = (i / 64) as usize;
    let mask = 1u64 << (i % 64);
    let mut v = x.as_limbs().to_vec();
    if v.len() <= limb {
        if op == Some(false) {
            return x.clone();
        }
        v.resize(limb + 1, 0);
    }
    match op {
        Some(true) => v[limb] |= mask,
        Some(false) => v[limb] &= !mask,
        None => v[limb] ^= mask,
    }
    Nat::from_limbs(&v)
}

impl Nat {
    /// Returns the number of one bits.
    pub fn count_ones(&self) -> u64 {
        popcount_limbs(self.as_limbs())
    }

    /// Returns the number of bit positions where `self` and `other` differ.
    pub fn hamming_distance(&self, other: &Nat) -> u64 {
        let (a, b) = (self.as_limbs(), other.as_limbs());
        let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        short
            .iter()
            .zip(long)
            .map(|(x, y)| (x ^ y).count_ones() as u64)
            .sum::<u64>()
            + popcount_limbs(&long[short.len()..])
    }

    /// Returns the index of the first zero bit at or above `start` (always
    /// exists: the bits past the top are zero).
    pub fn scan0(&self, start: u64) -> u64 {
        scan_limbs(self.as_limbs(), start, false).expect("a zero bit always exists")
    }

    /// Returns the index of the first one bit at or above `start`, or `None`
    /// if there is none.
    pub fn scan1(&self, start: u64) -> Option<u64> {
        scan_limbs(self.as_limbs(), start, true)
    }

    /// Builds a natural from big-endian bytes (leading zero bytes allowed).
    pub fn from_bytes_be(bytes: &[u8]) -> Nat {
        let mut limbs = Vec::with_capacity(bytes.len() / 8 + 1);
        for chunk in bytes.rchunks(8) {
            let mut buf = [0u8; 8];
            buf[8 - chunk.len()..].copy_from_slice(chunk);
            limbs.push(u64::from_be_bytes(buf));
        }
        Nat::from_limbs(&limbs)
    }

    /// Returns the magnitude as big-endian bytes (no leading zero bytes; empty
    /// for zero).
    pub fn to_bytes_be(&self) -> Vec<u8> {
        let mut out = self.to_bytes_le();
        out.reverse();
        out
    }
}

impl Int {
    /// `|self| − 1` for negative `self`, whose bits are the complements of
    /// `self`'s two's-complement bits.
    fn ones_complement_mag(&self) -> Nat {
        self.magnitude_ref()
            .checked_sub(&Nat::one())
            .expect("negative => |self| ≥ 1")
    }

    /// Returns bit `i` of the two's-complement representation (negative values
    /// have infinitely many leading ones). Compare [`Int::bit`], which reads
    /// the magnitude.
    pub fn test_bit(&self, i: u64) -> bool {
        if !self.is_negative() {
            return self.bit(i);
        }
        // |x| − 1 has ones below tz = trailing_zeros(|x|), a zero at tz and the
        // bits of |x| above; the two's-complement bit is their complement.
        let tz = self.trailing_zeros() as u64;
        match i.cmp(&tz) {
            core::cmp::Ordering::Less => false,
            core::cmp::Ordering::Equal => true,
            core::cmp::Ordering::Greater => !self.bit(i),
        }
    }

    /// Applies a bit operation in the two's-complement view.
    fn modify_bit(&mut self, i: u64, op: Option<bool>) {
        *self = if self.is_negative() {
            // Setting a two's-complement bit clears the bit of |x| − 1.
            let y = with_bit(&self.ones_complement_mag(), i, op.map(|b| !b));
            Int::from_sign_magnitude(Sign::Negative, y.add(&Nat::one()))
        } else {
            Int::from(with_bit(&self.magnitude_ref(), i, op))
        };
    }

    /// Sets bit `i` of the two's-complement representation (GMP's
    /// `mpz_setbit`).
    pub fn set_bit(&mut self, i: u64) {
        self.modify_bit(i, Some(true));
    }

    /// Clears bit `i` of the two's-complement representation (GMP's
    /// `mpz_clrbit`).
    pub fn clear_bit(&mut self, i: u64) {
        self.modify_bit(i, Some(false));
    }

    /// Flips bit `i` of the two's-complement representation (GMP's
    /// `mpz_combit`).
    pub fn complement_bit(&mut self, i: u64) {
        self.modify_bit(i, None);
    }

    /// Returns the two's-complement bitwise NOT, `−self − 1` (GMP's
    /// `mpz_com`; also the `!` operator). See [`Int::bitnot`] for a
    /// fixed-width variant.
    pub fn not(&self) -> Int {
        self.neg().sub(&Int::ONE)
    }

    /// Returns the number of one bits, or `None` for a negative value (which
    /// has infinitely many in two's complement).
    pub fn count_ones(&self) -> Option<u64> {
        (!self.is_negative()).then(|| popcount_limbs(self.limbs()))
    }

    /// Returns the number of bit positions where the two's-complement forms of
    /// `self` and `other` differ, or `None` when their signs differ (infinitely
    /// many positions).
    pub fn hamming_distance(&self, other: &Int) -> Option<u64> {
        match (self.is_negative(), other.is_negative()) {
            (false, false) => Some(
                self.magnitude_ref()
                    .hamming_distance(&other.magnitude_ref()),
            ),
            // ¬y₁ ⊕ ¬y₂ = y₁ ⊕ y₂.
            (true, true) => Some(
                self.ones_complement_mag()
                    .hamming_distance(&other.ones_complement_mag()),
            ),
            _ => None,
        }
    }

    /// Returns the index of the first zero bit at or above `start` in the
    /// two's-complement form, or `None` if there is none (a negative value
    /// whose bits from `start` up are all ones).
    pub fn scan0(&self, start: u64) -> Option<u64> {
        if self.is_negative() {
            self.ones_complement_mag().scan1(start)
        } else {
            scan_limbs(self.limbs(), start, false)
        }
    }

    /// Returns the index of the first one bit at or above `start` in the
    /// two's-complement form, or `None` if there is none (a non-negative value
    /// below `2^start`).
    pub fn scan1(&self, start: u64) -> Option<u64> {
        if self.is_negative() {
            Some(self.ones_complement_mag().scan0(start))
        } else {
            scan_limbs(self.limbs(), start, true)
        }
    }

    /// Whether `|self|` has a one bit below bit `k`, i.e. `2^k ∤ self`.
    fn has_low_bits(&self, k: u64) -> bool {
        !self.is_zero() && (self.trailing_zeros() as u64) < k
    }

    /// Returns `⌊self / 2^k⌋` (rounds toward −∞, GMP's `mpz_fdiv_q_2exp`;
    /// also the `>>` operator).
    pub fn div_2k_floor(&self, k: u64) -> Int {
        let q = self.div_2k_trunc(k);
        if self.is_negative() && self.has_low_bits(k) {
            q.sub(&Int::ONE)
        } else {
            q
        }
    }

    /// Returns `⌈self / 2^k⌉` (rounds toward +∞, GMP's `mpz_cdiv_q_2exp`).
    pub fn div_2k_ceil(&self, k: u64) -> Int {
        let q = self.div_2k_trunc(k);
        if self.is_positive() && self.has_low_bits(k) {
            q.add(&Int::ONE)
        } else {
            q
        }
    }

    /// Returns `self − 2^k·trunc(self / 2^k)`: the low `k` bits of `|self|`
    /// with the sign of `self` (GMP's `mpz_tdiv_r_2exp`). See [`Int::mod_2k`]
    /// for the non-negative remainder.
    pub fn rem_2k_trunc(&self, k: u64) -> Int {
        Int::from_sign_magnitude(self.sign(), self.magnitude_ref().low_bits(k))
    }

    /// Builds a non-negative integer from big-endian magnitude bytes (GMP's
    /// `mpz_import` with most-significant-first order).
    pub fn from_bytes_be(bytes: &[u8]) -> Int {
        Int::from(Nat::from_bytes_be(bytes))
    }

    /// Builds a non-negative integer from little-endian magnitude bytes.
    pub fn from_bytes_le(bytes: &[u8]) -> Int {
        Int::from(Nat::from_bytes_le(bytes))
    }

    /// Returns `|self|` as big-endian bytes, without leading zeros (empty for
    /// zero); the sign is dropped, as in GMP's `mpz_export`.
    pub fn to_bytes_be(&self) -> Vec<u8> {
        self.magnitude_ref().to_bytes_be()
    }

    /// Returns `|self|` as little-endian bytes, without trailing zeros (empty
    /// for zero); the sign is dropped.
    pub fn to_bytes_le(&self) -> Vec<u8> {
        self.magnitude_ref().to_bytes_le()
    }

    /// Builds an integer from little-endian two's-complement bytes (the top
    /// bit of the last byte is the sign; empty input is zero).
    pub fn from_signed_bytes_le(bytes: &[u8]) -> Int {
        let v = Int::from_bytes_le(bytes);
        match bytes.last() {
            Some(&b) if b & 0x80 != 0 => v.sub(&Int::ONE.mul_2k(8 * bytes.len() as u64)),
            _ => v,
        }
    }

    /// Builds an integer from big-endian two's-complement bytes (the top bit
    /// of the first byte is the sign; empty input is zero).
    pub fn from_signed_bytes_be(bytes: &[u8]) -> Int {
        let v = Int::from_bytes_be(bytes);
        match bytes.first() {
            Some(&b) if b & 0x80 != 0 => v.sub(&Int::ONE.mul_2k(8 * bytes.len() as u64)),
            _ => v,
        }
    }

    /// Returns the shortest little-endian two's-complement byte encoding
    /// (`[]` for zero, `[0xff]` for `−1`).
    pub fn to_signed_bytes_le(&self) -> Vec<u8> {
        if !self.is_negative() {
            let mut out = self.to_bytes_le();
            if matches!(out.last(), Some(&b) if b & 0x80 != 0) {
                out.push(0);
            }
            return out;
        }
        // −x = ¬(x − 1): complement the bytes of |x| − 1 and sign-extend.
        let mut out: Vec<u8> = self
            .ones_complement_mag()
            .to_bytes_le()
            .iter()
            .map(|b| !b)
            .collect();
        if !matches!(out.last(), Some(&b) if b & 0x80 != 0) {
            out.push(0xff);
        }
        out
    }

    /// Returns the shortest big-endian two's-complement byte encoding
    /// (`[]` for zero, `[0xff]` for `−1`).
    pub fn to_signed_bytes_be(&self) -> Vec<u8> {
        let mut out = self.to_signed_bytes_le();
        out.reverse();
        out
    }
}

macro_rules! shift_ops {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $method:ident) => {
        impl core::ops::$tr<u64> for Int {
            type Output = Int;
            #[inline]
            fn $m(self, k: u64) -> Int {
                Int::$method(&self, k)
            }
        }
        impl core::ops::$tr<u64> for &Int {
            type Output = Int;
            #[inline]
            fn $m(self, k: u64) -> Int {
                Int::$method(self, k)
            }
        }
        impl core::ops::$atr<u64> for Int {
            #[inline]
            fn $am(&mut self, k: u64) {
                *self = Int::$method(self, k);
            }
        }
    };
}

// `>>` rounds toward −∞, matching the primitive signed integers.
shift_ops!(Shl, shl, ShlAssign, shl_assign, mul_2k);
shift_ops!(Shr, shr, ShrAssign, shr_assign, div_2k_floor);

impl core::ops::Not for Int {
    type Output = Int;
    #[inline]
    fn not(self) -> Int {
        Int::not(&self)
    }
}

impl core::ops::Not for &Int {
    type Output = Int;
    #[inline]
    fn not(self) -> Int {
        Int::not(self)
    }
}
