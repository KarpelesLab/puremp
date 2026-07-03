//! `num-traits` bridge (feature `num-traits`).
//!
//! Implements the common `num-traits` interfaces for [`Int`], [`Rational`], and
//! [`Nat`] so the types drop into generic numeric code written against them.
//! This is optional ecosystem glue; the arithmetic core depends on nothing.

use num_traits::{FromPrimitive, Num, One, Signed, ToPrimitive, Zero};

use crate::int::Int;
use crate::nat::Nat;
use crate::rational::Rational;

// ---- Int ----

impl Zero for Int {
    #[inline]
    fn zero() -> Int {
        Int::ZERO
    }
    #[inline]
    fn is_zero(&self) -> bool {
        Int::is_zero(self)
    }
}

impl One for Int {
    #[inline]
    fn one() -> Int {
        Int::ONE
    }
    #[inline]
    fn is_one(&self) -> bool {
        Int::is_one(self)
    }
}

impl Num for Int {
    type FromStrRadixErr = crate::error::Error;
    #[inline]
    fn from_str_radix(s: &str, radix: u32) -> Result<Int, Self::FromStrRadixErr> {
        Int::from_str_radix(s, radix)
    }
}

impl Signed for Int {
    #[inline]
    fn abs(&self) -> Int {
        Int::abs(self)
    }
    #[inline]
    fn abs_sub(&self, other: &Int) -> Int {
        let d = self.sub(other);
        if d.is_negative() { Int::ZERO } else { d }
    }
    #[inline]
    fn signum(&self) -> Int {
        Int::from_i64(Int::signum(self) as i64)
    }
    #[inline]
    fn is_positive(&self) -> bool {
        Int::is_positive(self)
    }
    #[inline]
    fn is_negative(&self) -> bool {
        Int::is_negative(self)
    }
}

impl ToPrimitive for Int {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        Int::to_i64(self)
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        Int::to_u64(self)
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        Some(Int::to_f64(self))
    }
}

impl FromPrimitive for Int {
    #[inline]
    fn from_i64(n: i64) -> Option<Int> {
        Some(Int::from_i64(n))
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Int> {
        Some(Int::from_u64(n))
    }
    #[inline]
    fn from_i128(n: i128) -> Option<Int> {
        Some(Int::from_i128(n))
    }
    #[inline]
    fn from_u128(n: u128) -> Option<Int> {
        Some(Int::from_u128(n))
    }
}

// ---- Rational ----

impl Zero for Rational {
    #[inline]
    fn zero() -> Rational {
        Rational::ZERO
    }
    #[inline]
    fn is_zero(&self) -> bool {
        Rational::is_zero(self)
    }
}

impl One for Rational {
    #[inline]
    fn one() -> Rational {
        Rational::ONE
    }
    #[inline]
    fn is_one(&self) -> bool {
        Rational::is_one(self)
    }
}

impl Num for Rational {
    type FromStrRadixErr = crate::error::Error;
    /// Parses a decimal rational (`"3"`, `"-3/4"`, `"1.5"`); `radix` must be 10.
    fn from_str_radix(s: &str, radix: u32) -> Result<Rational, Self::FromStrRadixErr> {
        if radix != 10 {
            return Err(crate::error::Error::Parse);
        }
        s.parse()
    }
}

impl Signed for Rational {
    #[inline]
    fn abs(&self) -> Rational {
        Rational::abs(self)
    }
    #[inline]
    fn abs_sub(&self, other: &Rational) -> Rational {
        let d = self.sub(other);
        if d.is_negative() { Rational::ZERO } else { d }
    }
    #[inline]
    fn signum(&self) -> Rational {
        Rational::from_integer(Int::from_i64(Rational::signum(self) as i64))
    }
    #[inline]
    fn is_positive(&self) -> bool {
        Rational::is_positive(self)
    }
    #[inline]
    fn is_negative(&self) -> bool {
        Rational::is_negative(self)
    }
}

// ---- Nat (unsigned; no total Sub, so no `Num`) ----

impl Zero for Nat {
    #[inline]
    fn zero() -> Nat {
        Nat::zero()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        Nat::is_zero(self)
    }
}

impl One for Nat {
    #[inline]
    fn one() -> Nat {
        Nat::one()
    }
    #[inline]
    fn is_one(&self) -> bool {
        Nat::is_one(self)
    }
}
