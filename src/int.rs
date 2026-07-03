//! Arbitrary-precision signed integers.
//!
//! [`Int`] is a [`Sign`] paired with an unsigned [`Nat`] magnitude. The sign is
//! kept canonical: it is [`Sign::Zero`] if and only if the magnitude is zero, so
//! there is a single representation of every value and the derived
//! [`PartialEq`]/[`Eq`] are correct.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use crate::error::{Error, Result};
use crate::nat::Nat;

/// The sign of an [`Int`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sign {
    /// A value strictly less than zero.
    Negative,
    /// The value zero.
    Zero,
    /// A value strictly greater than zero.
    Positive,
}

impl core::ops::Neg for Sign {
    type Output = Sign;
    #[inline]
    fn neg(self) -> Sign {
        match self {
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
            Sign::Positive => Sign::Negative,
        }
    }
}

/// An arbitrary-precision signed integer.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct Int {
    sign: Sign,
    mag: Nat,
}

impl Default for Sign {
    #[inline]
    fn default() -> Self {
        Sign::Zero
    }
}

impl Int {
    /// Returns the integer zero.
    #[inline]
    pub fn zero() -> Self {
        Int {
            sign: Sign::Zero,
            mag: Nat::zero(),
        }
    }

    /// Returns the integer one.
    #[inline]
    pub fn one() -> Self {
        Int {
            sign: Sign::Positive,
            mag: Nat::one(),
        }
    }

    /// Builds an integer from a sign and a magnitude, canonicalizing the sign of
    /// a zero magnitude to [`Sign::Zero`].
    pub fn from_sign_magnitude(sign: Sign, mag: Nat) -> Self {
        if mag.is_zero() {
            Int {
                sign: Sign::Zero,
                mag,
            }
        } else {
            Int { sign, mag }
        }
    }

    /// Builds an integer from an `i64`.
    pub fn from_i64(v: i64) -> Self {
        let (sign, u) = match v.cmp(&0) {
            Ordering::Greater => (Sign::Positive, v as u64),
            Ordering::Less => (Sign::Negative, v.unsigned_abs()),
            Ordering::Equal => (Sign::Zero, 0),
        };
        Int::from_sign_magnitude(sign, Nat::from_u64(u))
    }

    /// Returns the sign of this integer.
    #[inline]
    pub fn sign(&self) -> Sign {
        self.sign
    }

    /// Returns a reference to the unsigned magnitude `|self|`.
    #[inline]
    pub fn magnitude(&self) -> &Nat {
        &self.mag
    }

    /// Returns `true` if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.sign == Sign::Zero
    }

    /// Returns `true` if this value is strictly negative.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.sign == Sign::Negative
    }

    /// Returns the absolute value `|self|` as an [`Int`].
    pub fn abs(&self) -> Int {
        Int::from_sign_magnitude(
            if self.is_zero() {
                Sign::Zero
            } else {
                Sign::Positive
            },
            self.mag.clone(),
        )
    }

    /// Returns `-self`.
    pub fn neg(&self) -> Int {
        Int::from_sign_magnitude(-self.sign, self.mag.clone())
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Int) -> Int {
        match (self.sign, rhs.sign) {
            (Sign::Zero, _) => rhs.clone(),
            (_, Sign::Zero) => self.clone(),
            // Same sign: add magnitudes, keep the sign.
            (a, b) if a == b => Int::from_sign_magnitude(a, self.mag.add(&rhs.mag)),
            // Opposite signs: subtract the smaller magnitude from the larger.
            _ => match self.mag.cmp(&rhs.mag) {
                Ordering::Equal => Int::zero(),
                Ordering::Greater => Int::from_sign_magnitude(
                    self.sign,
                    self.mag.checked_sub(&rhs.mag).expect("mag >= rhs.mag"),
                ),
                Ordering::Less => Int::from_sign_magnitude(
                    rhs.sign,
                    rhs.mag.checked_sub(&self.mag).expect("rhs.mag > mag"),
                ),
            },
        }
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Int) -> Int {
        self.add(&rhs.neg())
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Int) -> Int {
        let sign = match (self.sign, rhs.sign) {
            (Sign::Zero, _) | (_, Sign::Zero) => return Int::zero(),
            (a, b) if a == b => Sign::Positive,
            _ => Sign::Negative,
        };
        Int::from_sign_magnitude(sign, self.mag.mul(&rhs.mag))
    }

    /// Truncated division: returns `(quotient, remainder)` where the quotient
    /// rounds toward zero and the remainder takes the sign of the dividend, so
    /// `self == quotient·rhs + remainder`. Returns `None` if `rhs` is zero.
    ///
    /// Floored and Euclidean variants are a later milestone; see `ROADMAP.md`.
    pub fn div_rem(&self, rhs: &Int) -> Option<(Int, Int)> {
        let (q_mag, r_mag) = self.mag.div_rem(&rhs.mag)?;
        let q_sign = match (self.sign, rhs.sign) {
            (Sign::Zero, _) => Sign::Zero,
            (a, b) if a == b => Sign::Positive,
            _ => Sign::Negative,
        };
        Some((
            Int::from_sign_magnitude(q_sign, q_mag),
            // Remainder keeps the dividend's sign (truncated division).
            Int::from_sign_magnitude(self.sign, r_mag),
        ))
    }

    /// Returns `self` raised to `exp`, by exponentiation-by-squaring.
    ///
    /// `self.pow(0) == 1` for every `self`, including zero.
    pub fn pow(&self, exp: u64) -> Int {
        let mut result = Int::one();
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
}

impl PartialOrd for Int {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Int {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.sign, other.sign) {
            (Sign::Negative, Sign::Zero | Sign::Positive) => Ordering::Less,
            (Sign::Zero, Sign::Positive) => Ordering::Less,
            (Sign::Zero, Sign::Zero) => Ordering::Equal,
            (Sign::Zero, Sign::Negative) => Ordering::Greater,
            (Sign::Positive, Sign::Zero | Sign::Negative) => Ordering::Greater,
            // Same sign: compare magnitudes, reversing for negatives.
            (Sign::Positive, Sign::Positive) => self.mag.cmp(&other.mag),
            (Sign::Negative, Sign::Negative) => other.mag.cmp(&self.mag),
        }
    }
}

impl From<i64> for Int {
    #[inline]
    fn from(v: i64) -> Self {
        Int::from_i64(v)
    }
}

impl From<Nat> for Int {
    #[inline]
    fn from(mag: Nat) -> Self {
        Int::from_sign_magnitude(Sign::Positive, mag)
    }
}

impl FromStr for Int {
    type Err = Error;

    /// Parses a decimal integer with an optional leading `+` or `-`.
    fn from_str(s: &str) -> Result<Self> {
        let (sign, digits) = match s.strip_prefix('-') {
            Some(rest) => (Sign::Negative, rest),
            None => (Sign::Positive, s.strip_prefix('+').unwrap_or(s)),
        };
        let mag = Nat::from_str(digits)?;
        Ok(Int::from_sign_magnitude(sign, mag))
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign == Sign::Negative {
            f.write_str("-")?;
        }
        fmt::Display::fmt(&self.mag, f)
    }
}

impl fmt::Debug for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Int({self})")
    }
}

macro_rules! forward_binop {
    ($trait:ident, $method:ident) => {
        impl core::ops::$trait for &Int {
            type Output = Int;
            #[inline]
            fn $method(self, rhs: &Int) -> Int {
                Int::$method(self, rhs)
            }
        }
        impl core::ops::$trait for Int {
            type Output = Int;
            #[inline]
            fn $method(self, rhs: Int) -> Int {
                Int::$method(&self, &rhs)
            }
        }
    };
}

forward_binop!(Add, add);
forward_binop!(Sub, sub);
forward_binop!(Mul, mul);

impl core::ops::Neg for Int {
    type Output = Int;
    #[inline]
    fn neg(self) -> Int {
        Int::neg(&self)
    }
}

impl core::ops::Neg for &Int {
    type Output = Int;
    #[inline]
    fn neg(self) -> Int {
        Int::neg(self)
    }
}
