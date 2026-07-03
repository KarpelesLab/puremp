//! Arbitrary-precision binary floating-point numbers.
//!
//! [`Float`] is a sign, an integer significand, and a base-2 exponent, carried
//! at a caller-chosen precision. A finite value is
//!
//! ```text
//! value = (-1)^sign · significand · 2^exponent
//! ```
//!
//! The design target is MPFR-class semantics: every operation takes an explicit
//! output precision and a [`RoundingMode`], and returns the correctly-rounded
//! result (the exact real result rounded once, as if computed to infinite
//! precision). The scaffold ships the representation, the rounding-mode
//! vocabulary, and exact conversions; correctly-rounded arithmetic and the
//! transcendental functions are the float milestones in `ROADMAP.md`.

use core::fmt;

use crate::int::{Int, Sign};
use crate::nat::Nat;

/// A rounding direction for a floating-point result, following IEEE 754 and
/// MPFR.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RoundingMode {
    /// Round to the nearest representable value; ties go to the value with an
    /// even significand (IEEE 754 `roundTiesToEven`). The default.
    #[default]
    Nearest,
    /// Round toward zero (truncate).
    TowardZero,
    /// Round toward positive infinity (ceiling).
    TowardPositive,
    /// Round toward negative infinity (floor).
    TowardNegative,
    /// Round away from zero.
    AwayFromZero,
}

/// An arbitrary-precision binary floating-point number.
///
/// This scaffold represents finite values only; the special values
/// (signed zeros, infinities, NaN) and their IEEE interactions are part of the
/// float milestones in `ROADMAP.md`.
#[derive(Clone, PartialEq, Eq)]
pub struct Float {
    /// Sign of the value ([`Sign::Zero`] iff the significand is zero).
    sign: Sign,
    /// Unsigned significand; `value = ±significand · 2^exponent`.
    significand: Nat,
    /// Base-2 exponent.
    exponent: i64,
    /// Working precision in bits (the significand is rounded to this many
    /// significant bits by arithmetic operations).
    precision: u64,
}

impl Float {
    /// Returns a positive zero carried at `precision` bits.
    ///
    /// `precision` is clamped to at least 1.
    pub fn zero(precision: u64) -> Self {
        Float {
            sign: Sign::Zero,
            significand: Nat::zero(),
            exponent: 0,
            precision: precision.max(1),
        }
    }

    /// Builds a [`Float`] holding the exact value of `n`, carried at a precision
    /// wide enough to represent it losslessly (at least 1 bit).
    pub fn from_int_exact(n: &Int) -> Self {
        let significand = n.magnitude().clone();
        let precision = significand.bit_len().max(1);
        Float {
            sign: n.sign(),
            significand,
            exponent: 0,
            precision,
        }
    }

    /// Returns the sign of this value.
    #[inline]
    pub fn sign(&self) -> Sign {
        self.sign
    }

    /// Returns the working precision in bits.
    #[inline]
    pub fn precision(&self) -> u64 {
        self.precision
    }

    /// Returns the base-2 exponent.
    #[inline]
    pub fn exponent(&self) -> i64 {
        self.exponent
    }

    /// Returns the unsigned significand.
    #[inline]
    pub fn significand(&self) -> &Nat {
        &self.significand
    }

    /// Returns `true` if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.sign == Sign::Zero
    }
}

impl fmt::Display for Float {
    /// Formats the exact value as `±significand·2^exponent` (or `0`). Rounded
    /// decimal formatting is a float milestone; see `ROADMAP.md`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        if self.sign == Sign::Negative {
            f.write_str("-")?;
        }
        write!(f, "{}·2^{}", self.significand, self.exponent)
    }
}

impl fmt::Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Float({self} @ {}bit)", self.precision)
    }
}
