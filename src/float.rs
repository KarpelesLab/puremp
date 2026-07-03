//! Arbitrary-precision binary floating-point numbers.
//!
//! [`Float`] is a sign, an integer significand, and a base-2 exponent, carried
//! at a caller-chosen precision. A finite value is
//!
//! ```text
//! value = (-1)^sign · significand · 2^exponent
//! ```
//!
//! Non-zero values are normalized so the significand has exactly `precision`
//! significant bits (its top bit is set). Every arithmetic operation takes an
//! explicit output precision and a [`RoundingMode`] and returns the
//! correctly-rounded result — the exact real result rounded once, as if computed
//! to unbounded precision. This is the MPFR-class contract for the basic
//! operations shipped here (add, sub, mul, div, sqrt); the transcendental
//! functions are a later milestone (see `ROADMAP.md`).
//!
//! This layer is optional and separable — it is not part of the integer/rational
//! core contract, and lives behind the `float` feature.

use core::cmp::Ordering;
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

/// An arbitrary-precision binary floating-point number (finite values only).
///
/// Signed zeros, infinities, and NaN are a later milestone; zero here is a
/// single unsigned value.
#[derive(Clone)]
pub struct Float {
    /// Sign ([`Sign::Zero`] iff the value is zero).
    sign: Sign,
    /// Unsigned significand; `bit_len == precision` for non-zero values.
    significand: Nat,
    /// Base-2 exponent.
    exponent: i64,
    /// Working precision in bits (≥ 1).
    precision: u64,
}

impl Float {
    /// Returns a zero carried at `precision` bits (clamped to ≥ 1).
    pub fn zero(precision: u64) -> Self {
        Float {
            sign: Sign::Zero,
            significand: Nat::zero(),
            exponent: 0,
            precision: precision.max(1),
        }
    }

    /// Rounds `(-1)^sign · mant · 2^exp` to `precision` bits under `mode`.
    fn round_raw(sign: Sign, mant: Nat, exp: i64, precision: u64, mode: RoundingMode) -> Float {
        let precision = precision.max(1);
        if mant.is_zero() || sign == Sign::Zero {
            return Float::zero(precision);
        }
        let bits = mant.bit_len();
        if bits <= precision {
            // Exact: left-align to `precision` bits.
            let shift = precision - bits;
            return Float {
                sign,
                significand: mant.shl(shift),
                exponent: exp - shift as i64,
                precision,
            };
        }
        let drop = bits - precision;
        let low = mant.low_bits(drop);
        let mut hi = mant.shr(drop);
        let mut new_exp = exp + drop as i64;
        let half = Nat::one().shl(drop - 1);
        let round_up = match mode {
            RoundingMode::TowardZero => false,
            RoundingMode::AwayFromZero => !low.is_zero(),
            RoundingMode::TowardPositive => sign == Sign::Positive && !low.is_zero(),
            RoundingMode::TowardNegative => sign == Sign::Negative && !low.is_zero(),
            RoundingMode::Nearest => match low.cmp(&half) {
                Ordering::Greater => true,
                Ordering::Less => false,
                Ordering::Equal => !hi.is_even(), // ties to even
            },
        };
        if round_up {
            hi = hi.add(&Nat::one());
            if hi.bit_len() > precision {
                // Carried out (e.g. 0b111.. + 1): renormalize.
                hi = hi.shr(1);
                new_exp += 1;
            }
        }
        Float {
            sign,
            significand: hi,
            exponent: new_exp,
            precision,
        }
    }

    /// Builds a [`Float`] from an integer, rounded to `precision` bits.
    pub fn from_int(n: &Int, precision: u64, mode: RoundingMode) -> Self {
        Float::round_raw(n.sign(), n.magnitude(), 0, precision, mode)
    }

    /// Re-rounds `self` to a (possibly different) `precision` under `mode`.
    pub fn round(&self, precision: u64, mode: RoundingMode) -> Float {
        Float::round_raw(
            self.sign,
            self.significand.clone(),
            self.exponent,
            precision,
            mode,
        )
    }

    /// Returns `-self` (same precision).
    pub fn neg(&self) -> Float {
        Float {
            sign: -self.sign,
            significand: self.significand.clone(),
            exponent: self.exponent,
            precision: self.precision,
        }
    }

    /// Returns `|self|` (same precision).
    pub fn abs(&self) -> Float {
        Float {
            sign: if self.sign == Sign::Zero {
                Sign::Zero
            } else {
                Sign::Positive
            },
            significand: self.significand.clone(),
            exponent: self.exponent,
            precision: self.precision,
        }
    }

    /// Returns `self + rhs`, correctly rounded to `precision` bits.
    pub fn add(&self, rhs: &Float, precision: u64, mode: RoundingMode) -> Float {
        if self.is_zero() {
            return rhs.round(precision, mode);
        }
        if rhs.is_zero() {
            return self.round(precision, mode);
        }
        // Align to the common (smaller) exponent and add exactly as integers.
        let emin = self.exponent.min(rhs.exponent);
        let a = Int::from_sign_magnitude(
            self.sign,
            self.significand.shl((self.exponent - emin) as u64),
        );
        let b =
            Int::from_sign_magnitude(rhs.sign, rhs.significand.shl((rhs.exponent - emin) as u64));
        let s = a.add(&b);
        Float::round_raw(s.sign(), s.magnitude(), emin, precision, mode)
    }

    /// Returns `self - rhs`, correctly rounded.
    pub fn sub(&self, rhs: &Float, precision: u64, mode: RoundingMode) -> Float {
        self.add(&rhs.neg(), precision, mode)
    }

    /// Returns `self · rhs`, correctly rounded.
    pub fn mul(&self, rhs: &Float, precision: u64, mode: RoundingMode) -> Float {
        if self.is_zero() || rhs.is_zero() {
            return Float::zero(precision);
        }
        let sign = if self.sign == rhs.sign {
            Sign::Positive
        } else {
            Sign::Negative
        };
        let mant = self.significand.mul(&rhs.significand);
        Float::round_raw(sign, mant, self.exponent + rhs.exponent, precision, mode)
    }

    /// Returns `self / rhs`, correctly rounded. Panics if `rhs` is zero.
    pub fn div(&self, rhs: &Float, precision: u64, mode: RoundingMode) -> Float {
        assert!(!rhs.is_zero(), "Float::div: division by zero");
        if self.is_zero() {
            return Float::zero(precision);
        }
        let sign = if self.sign == rhs.sign {
            Sign::Positive
        } else {
            Sign::Negative
        };
        // Compute `precision + 2` guard bits of the quotient, folding the
        // remainder in as a sticky bit for correct rounding.
        let guard = precision + 2;
        let num = self.significand.shl(guard);
        let (mut q, r) = num.div_rem(&rhs.significand).expect("rhs is non-zero");
        if !r.is_zero() && q.is_even() {
            q = q.add(&Nat::one());
        }
        let exp = self.exponent - rhs.exponent - guard as i64;
        Float::round_raw(sign, q, exp, precision, mode)
    }

    /// Returns `√self`, correctly rounded. Panics if `self` is negative.
    pub fn sqrt(&self, precision: u64, mode: RoundingMode) -> Float {
        assert!(self.sign != Sign::Negative, "Float::sqrt: negative operand");
        if self.is_zero() {
            return Float::zero(precision);
        }
        // Make the exponent even so the square root splits cleanly.
        let mut sig = self.significand.clone();
        let mut exp = self.exponent;
        if exp & 1 != 0 {
            sig = sig.shl(1);
            exp -= 1;
        }
        // Scale so the integer square root yields ≥ precision + 2 bits.
        let want = 2 * (precision + 2);
        let cur = sig.bit_len();
        let mut shift = want.saturating_sub(cur);
        if shift & 1 != 0 {
            shift += 1;
        }
        let radicand = sig.shl(shift);
        let mut m = radicand.isqrt();
        // Sticky bit: mark inexactness so rounding sees it.
        if m.mul(&m) != radicand && m.is_even() {
            m = m.add(&Nat::one());
        }
        let result_exp = exp / 2 - (shift / 2) as i64;
        Float::round_raw(Sign::Positive, m, result_exp, precision, mode)
    }

    // --- inspection & conversion ---

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

    /// Returns `true` if this value is strictly negative.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.sign == Sign::Negative
    }

    /// Returns the value as the nearest `f64` (best-effort; may be `±inf`/`0`
    /// on extreme exponents).
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        let mant = Int::from(self.significand.clone()).to_f64();
        let scaled = mant * exp2(self.exponent);
        if self.sign == Sign::Negative {
            -scaled
        } else {
            scaled
        }
    }
}

/// Best-effort `2^e` as an `f64` without `std` (avoids `powi`'s `i32` limit for
/// large exponents by repeated squaring).
fn exp2(e: i64) -> f64 {
    let mut base = if e < 0 { 0.5 } else { 2.0 };
    let mut n = e.unsigned_abs();
    let mut acc = 1.0f64;
    while n > 0 {
        if n & 1 == 1 {
            acc *= base;
        }
        base *= base;
        n >>= 1;
    }
    acc
}

impl Float {
    /// Compares magnitudes `|self|` and `|other|` (both assumed non-zero).
    fn mag_cmp(&self, other: &Float) -> Ordering {
        let emin = self.exponent.min(other.exponent);
        let a = self.significand.shl((self.exponent - emin) as u64);
        let b = other.significand.shl((other.exponent - emin) as u64);
        a.cmp(&b)
    }
}

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Float {}

impl PartialOrd for Float {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Float {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.sign, other.sign) {
            (Sign::Negative, Sign::Zero | Sign::Positive) | (Sign::Zero, Sign::Positive) => {
                Ordering::Less
            }
            (Sign::Zero, Sign::Zero) => Ordering::Equal,
            (Sign::Positive, Sign::Zero | Sign::Negative) | (Sign::Zero, Sign::Negative) => {
                Ordering::Greater
            }
            (Sign::Positive, Sign::Positive) => self.mag_cmp(other),
            (Sign::Negative, Sign::Negative) => other.mag_cmp(self),
        }
    }
}

impl fmt::Display for Float {
    /// Formats the exact value as `±significand·2^exponent` (or `0`). Rounded
    /// decimal formatting is a later milestone; see `ROADMAP.md`.
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
