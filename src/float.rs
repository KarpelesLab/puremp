//! Arbitrary-precision binary floating-point numbers.
//!
//! [`Float`] carries a caller-chosen precision and models the IEEE special
//! values (signed zeros, infinities, NaN) as well as finite normals. A finite
//! non-zero value is
//!
//! ```text
//! value = (-1)^sign · significand · 2^exponent
//! ```
//!
//! with the significand normalized to exactly `precision` bits. Every arithmetic
//! operation takes an explicit output precision and a [`RoundingMode`] and
//! returns the correctly-rounded result. The `*_ternary` variants additionally
//! return the ternary flag (whether the returned value is less than, equal to,
//! or greater than the exact result), matching MPFR.
//!
//! This layer is optional and separable — it is not part of the integer/rational
//! core contract, and lives behind the `float` feature.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use alloc::string::String;

use crate::error::{Error, Result};
use crate::int::{Int, Sign};
use crate::nat::Nat;
use crate::rational::Rational;

/// A rounding direction for a floating-point result, following IEEE 754 and
/// MPFR.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RoundingMode {
    /// Round to the nearest representable value; ties to even. The default.
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

#[derive(Clone, PartialEq, Eq)]
enum Repr {
    /// Not-a-number.
    NaN,
    /// Signed infinity (`true` = negative).
    Inf(bool),
    /// Signed zero (`true` = negative).
    Zero(bool),
    /// A finite non-zero value `(-1)^neg · sig · 2^exp`, `sig` normalized to
    /// `precision` bits.
    Normal { neg: bool, sig: Nat, exp: i64 },
}

/// An arbitrary-precision binary floating-point number.
#[derive(Clone)]
pub struct Float {
    repr: Repr,
    precision: u64,
}

impl Float {
    // --- constructors ---

    /// Returns a NaN carried at `precision` bits.
    pub fn nan(precision: u64) -> Float {
        Float {
            repr: Repr::NaN,
            precision: precision.max(1),
        }
    }

    /// Returns positive infinity.
    pub fn infinity(precision: u64) -> Float {
        Float {
            repr: Repr::Inf(false),
            precision: precision.max(1),
        }
    }

    /// Returns negative infinity.
    pub fn neg_infinity(precision: u64) -> Float {
        Float {
            repr: Repr::Inf(true),
            precision: precision.max(1),
        }
    }

    /// Returns positive zero.
    pub fn zero(precision: u64) -> Float {
        Float {
            repr: Repr::Zero(false),
            precision: precision.max(1),
        }
    }

    /// Returns negative zero.
    pub fn neg_zero(precision: u64) -> Float {
        Float {
            repr: Repr::Zero(true),
            precision: precision.max(1),
        }
    }

    fn zero_signed(neg: bool, precision: u64) -> Float {
        Float {
            repr: Repr::Zero(neg),
            precision: precision.max(1),
        }
    }

    fn inf_signed(neg: bool, precision: u64) -> Float {
        Float {
            repr: Repr::Inf(neg),
            precision: precision.max(1),
        }
    }

    /// Rounds `(-1)^neg · mant · 2^exp` to `precision` bits under `mode`,
    /// returning the value and its ternary flag (`Less`/`Equal`/`Greater` =
    /// returned value vs. exact).
    fn round_raw(
        neg: bool,
        mant: Nat,
        exp: i64,
        precision: u64,
        mode: RoundingMode,
    ) -> (Float, Ordering) {
        let precision = precision.max(1);
        if mant.is_zero() {
            return (Float::zero_signed(neg, precision), Ordering::Equal);
        }
        let bits = mant.bit_len();
        if bits <= precision {
            let shift = precision - bits;
            let repr = Repr::Normal {
                neg,
                sig: mant.shl(shift),
                exp: exp - shift as i64,
            };
            return (Float { repr, precision }, Ordering::Equal);
        }
        let drop = bits - precision;
        let low = mant.low_bits(drop);
        let mut hi = mant.shr(drop);
        let mut new_exp = exp + drop as i64;
        let half = Nat::one().shl(drop - 1);
        let inexact = !low.is_zero();
        let round_up = match mode {
            RoundingMode::TowardZero => false,
            RoundingMode::AwayFromZero => inexact,
            RoundingMode::TowardPositive => !neg && inexact,
            RoundingMode::TowardNegative => neg && inexact,
            RoundingMode::Nearest => match low.cmp(&half) {
                Ordering::Greater => true,
                Ordering::Less => false,
                Ordering::Equal => !hi.is_even(),
            },
        };
        if round_up {
            hi = hi.add(&Nat::one());
            if hi.bit_len() > precision {
                hi = hi.shr(1);
                new_exp += 1;
            }
        }
        let ternary = if !inexact {
            Ordering::Equal
        } else if round_up != neg {
            // rounded up a positive, or truncated a negative: value > exact.
            Ordering::Greater
        } else {
            Ordering::Less
        };
        let repr = Repr::Normal {
            neg,
            sig: hi,
            exp: new_exp,
        };
        (Float { repr, precision }, ternary)
    }

    /// Builds a [`Float`] from an integer, rounded to `precision` bits.
    pub fn from_int(n: &Int, precision: u64, mode: RoundingMode) -> Float {
        Float::round_raw(n.is_negative(), n.magnitude(), 0, precision, mode).0
    }

    /// Builds a [`Float`] from an exact rational, correctly rounded.
    pub fn from_rational(r: &Rational, precision: u64, mode: RoundingMode) -> Float {
        if r.is_zero() {
            return Float::zero(precision);
        }
        let num = r.numerator();
        let den = r.denominator();
        let work_num = num.magnitude().bit_len().max(1);
        let work_den = den.magnitude().bit_len().max(1);
        // Exact integer Floats, then a single correctly-rounded division.
        let fnum = Float::from_int(num, work_num, RoundingMode::TowardZero);
        let fden = Float::from_int(den, work_den, RoundingMode::TowardZero);
        fnum.div(&fden, precision, mode)
    }

    /// Builds a [`Float`] from an `f64` (exact then rounded to `precision`).
    pub fn from_f64(x: f64, precision: u64, mode: RoundingMode) -> Float {
        let bits = x.to_bits();
        let neg = bits >> 63 == 1;
        let exp_field = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & 0x000f_ffff_ffff_ffff;
        if exp_field == 0x7ff {
            return if frac == 0 {
                Float::inf_signed(neg, precision)
            } else {
                Float::nan(precision)
            };
        }
        let (mantissa, exponent) = if exp_field == 0 {
            if frac == 0 {
                return Float::zero_signed(neg, precision);
            }
            (frac, -1074) // subnormal
        } else {
            ((1u64 << 52) | frac, exp_field - 1075)
        };
        Float::round_raw(neg, Nat::from_u64(mantissa), exponent, precision, mode).0
    }

    /// Builds a [`Float`] from an `f32`.
    pub fn from_f32(x: f32, precision: u64, mode: RoundingMode) -> Float {
        Float::from_f64(x as f64, precision, mode)
    }

    /// Re-rounds `self` to a (possibly different) `precision` under `mode`.
    pub fn round(&self, precision: u64, mode: RoundingMode) -> Float {
        self.round_impl(precision, mode).0
    }

    fn round_impl(&self, precision: u64, mode: RoundingMode) -> (Float, Ordering) {
        match &self.repr {
            Repr::NaN => (Float::nan(precision), Ordering::Equal),
            Repr::Inf(neg) => (Float::inf_signed(*neg, precision), Ordering::Equal),
            Repr::Zero(neg) => (Float::zero_signed(*neg, precision), Ordering::Equal),
            Repr::Normal { neg, sig, exp } => {
                Float::round_raw(*neg, sig.clone(), *exp, precision, mode)
            }
        }
    }

    /// Returns `-self` (same precision).
    pub fn neg(&self) -> Float {
        let repr = match &self.repr {
            Repr::NaN => Repr::NaN,
            Repr::Inf(neg) => Repr::Inf(!neg),
            Repr::Zero(neg) => Repr::Zero(!neg),
            Repr::Normal { neg, sig, exp } => Repr::Normal {
                neg: !neg,
                sig: sig.clone(),
                exp: *exp,
            },
        };
        Float {
            repr,
            precision: self.precision,
        }
    }

    /// Returns `|self|` (same precision).
    pub fn abs(&self) -> Float {
        let repr = match &self.repr {
            Repr::NaN => Repr::NaN,
            Repr::Inf(_) => Repr::Inf(false),
            Repr::Zero(_) => Repr::Zero(false),
            Repr::Normal { sig, exp, .. } => Repr::Normal {
                neg: false,
                sig: sig.clone(),
                exp: *exp,
            },
        };
        Float {
            repr,
            precision: self.precision,
        }
    }

    // --- arithmetic ---

    /// Returns `self + rhs`, correctly rounded to `precision` bits.
    pub fn add(&self, rhs: &Float, precision: u64, mode: RoundingMode) -> Float {
        self.add_ternary(rhs, precision, mode).0
    }

    /// Like [`Float::add`], also returning the ternary flag.
    pub fn add_ternary(
        &self,
        rhs: &Float,
        precision: u64,
        mode: RoundingMode,
    ) -> (Float, Ordering) {
        use Repr::*;
        match (&self.repr, &rhs.repr) {
            (NaN, _) | (_, NaN) => (Float::nan(precision), Ordering::Equal),
            (Inf(a), Inf(b)) => {
                if a == b {
                    (Float::inf_signed(*a, precision), Ordering::Equal)
                } else {
                    (Float::nan(precision), Ordering::Equal)
                }
            }
            (Inf(a), _) => (Float::inf_signed(*a, precision), Ordering::Equal),
            (_, Inf(b)) => (Float::inf_signed(*b, precision), Ordering::Equal),
            (Zero(a), Zero(b)) => {
                let neg = if a == b {
                    *a
                } else {
                    mode == RoundingMode::TowardNegative
                };
                (Float::zero_signed(neg, precision), Ordering::Equal)
            }
            (Zero(_), _) => rhs.round_impl(precision, mode),
            (_, Zero(_)) => self.round_impl(precision, mode),
            (Normal { .. }, Normal { .. }) => self.add_normal(rhs, precision, mode),
        }
    }

    fn add_normal(&self, rhs: &Float, precision: u64, mode: RoundingMode) -> (Float, Ordering) {
        let (
            Repr::Normal {
                neg: na,
                sig: sa,
                exp: ea,
            },
            Repr::Normal {
                neg: nb,
                sig: sb,
                exp: eb,
            },
        ) = (&self.repr, &rhs.repr)
        else {
            unreachable!("add_normal called on non-normal operands")
        };
        let emin = (*ea).min(*eb);
        let a = Int::from_sign_magnitude(sign_of(*na), sa.shl((*ea - emin) as u64));
        let b = Int::from_sign_magnitude(sign_of(*nb), sb.shl((*eb - emin) as u64));
        let s = a.add(&b);
        if s.is_zero() {
            let neg = mode == RoundingMode::TowardNegative;
            (Float::zero_signed(neg, precision), Ordering::Equal)
        } else {
            Float::round_raw(s.is_negative(), s.magnitude(), emin, precision, mode)
        }
    }

    /// Returns `self - rhs`, correctly rounded.
    pub fn sub(&self, rhs: &Float, precision: u64, mode: RoundingMode) -> Float {
        self.add(&rhs.neg(), precision, mode)
    }

    /// Like [`Float::sub`], also returning the ternary flag.
    pub fn sub_ternary(
        &self,
        rhs: &Float,
        precision: u64,
        mode: RoundingMode,
    ) -> (Float, Ordering) {
        self.add_ternary(&rhs.neg(), precision, mode)
    }

    /// Returns `self · rhs`, correctly rounded.
    pub fn mul(&self, rhs: &Float, precision: u64, mode: RoundingMode) -> Float {
        self.mul_ternary(rhs, precision, mode).0
    }

    /// Like [`Float::mul`], also returning the ternary flag.
    pub fn mul_ternary(
        &self,
        rhs: &Float,
        precision: u64,
        mode: RoundingMode,
    ) -> (Float, Ordering) {
        use Repr::*;
        match (&self.repr, &rhs.repr) {
            (NaN, _) | (_, NaN) => (Float::nan(precision), Ordering::Equal),
            (Inf(_), Zero(_)) | (Zero(_), Inf(_)) => (Float::nan(precision), Ordering::Equal),
            (Inf(a), other) | (other, Inf(a)) => (
                Float::inf_signed(*a ^ other.sign_bit(), precision),
                Ordering::Equal,
            ),
            (Zero(a), other) | (other, Zero(a)) => (
                Float::zero_signed(*a ^ other.sign_bit(), precision),
                Ordering::Equal,
            ),
            (
                Normal {
                    neg: na,
                    sig: sa,
                    exp: ea,
                },
                Normal {
                    neg: nb,
                    sig: sb,
                    exp: eb,
                },
            ) => Float::round_raw(na ^ nb, sa.mul(sb), ea + eb, precision, mode),
        }
    }

    /// Returns `self / rhs`, correctly rounded. `x/0` is signed infinity and
    /// `0/0` is NaN (no panic).
    pub fn div(&self, rhs: &Float, precision: u64, mode: RoundingMode) -> Float {
        self.div_ternary(rhs, precision, mode).0
    }

    /// Like [`Float::div`], also returning the ternary flag.
    pub fn div_ternary(
        &self,
        rhs: &Float,
        precision: u64,
        mode: RoundingMode,
    ) -> (Float, Ordering) {
        use Repr::*;
        match (&self.repr, &rhs.repr) {
            (NaN, _) | (_, NaN) => (Float::nan(precision), Ordering::Equal),
            (Inf(_), Inf(_)) | (Zero(_), Zero(_)) => (Float::nan(precision), Ordering::Equal),
            (Inf(a), other) => (
                Float::inf_signed(*a ^ other.sign_bit(), precision),
                Ordering::Equal,
            ),
            (other, Inf(b)) => (
                Float::zero_signed(other.sign_bit() ^ *b, precision),
                Ordering::Equal,
            ),
            (other, Zero(b)) => {
                // finite non-zero / 0 = signed infinity.
                (
                    Float::inf_signed(other.sign_bit() ^ *b, precision),
                    Ordering::Equal,
                )
            }
            (Zero(a), other) => (
                Float::zero_signed(*a ^ other.sign_bit(), precision),
                Ordering::Equal,
            ),
            (
                Normal {
                    neg: na,
                    sig: sa,
                    exp: ea,
                },
                Normal {
                    neg: nb,
                    sig: sb,
                    exp: eb,
                },
            ) => {
                let guard = precision + 2;
                let num = sa.shl(guard);
                let (mut q, r) = num.div_rem(sb).expect("divisor is non-zero");
                if !r.is_zero() && q.is_even() {
                    q = q.add(&Nat::one());
                }
                Float::round_raw(na ^ nb, q, ea - eb - guard as i64, precision, mode)
            }
        }
    }

    /// Returns `√self`, correctly rounded. `√(negative)` is NaN, `√(±0) = ±0`.
    pub fn sqrt(&self, precision: u64, mode: RoundingMode) -> Float {
        self.sqrt_ternary(precision, mode).0
    }

    /// Like [`Float::sqrt`], also returning the ternary flag.
    pub fn sqrt_ternary(&self, precision: u64, mode: RoundingMode) -> (Float, Ordering) {
        match &self.repr {
            Repr::NaN => (Float::nan(precision), Ordering::Equal),
            Repr::Inf(true) => (Float::nan(precision), Ordering::Equal),
            Repr::Inf(false) => (Float::infinity(precision), Ordering::Equal),
            Repr::Zero(neg) => (Float::zero_signed(*neg, precision), Ordering::Equal),
            Repr::Normal { neg: true, .. } => (Float::nan(precision), Ordering::Equal),
            Repr::Normal {
                neg: false,
                sig,
                exp,
            } => {
                let mut s = sig.clone();
                let mut e = *exp;
                if e & 1 != 0 {
                    s = s.shl(1);
                    e -= 1;
                }
                let want = 2 * (precision + 2);
                let cur = s.bit_len();
                let mut shift = want.saturating_sub(cur);
                if shift & 1 != 0 {
                    shift += 1;
                }
                let radicand = s.shl(shift);
                let mut m = radicand.isqrt();
                if m.mul(&m) != radicand && m.is_even() {
                    m = m.add(&Nat::one());
                }
                Float::round_raw(false, m, e / 2 - (shift / 2) as i64, precision, mode)
            }
        }
    }

    // --- classification & accessors ---

    /// Returns `true` if this value is NaN.
    #[inline]
    pub fn is_nan(&self) -> bool {
        matches!(self.repr, Repr::NaN)
    }

    /// Returns `true` if this value is `±∞`.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        matches!(self.repr, Repr::Inf(_))
    }

    /// Returns `true` if this value is finite (not NaN or `±∞`).
    #[inline]
    pub fn is_finite(&self) -> bool {
        matches!(self.repr, Repr::Zero(_) | Repr::Normal { .. })
    }

    /// Returns `true` if this value is `±0`.
    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self.repr, Repr::Zero(_))
    }

    /// Returns `true` if the sign bit is set (includes `-0` and `-∞`; `false`
    /// for NaN).
    #[inline]
    pub fn is_sign_negative(&self) -> bool {
        self.repr.sign_bit()
    }

    /// Returns the sign as [`Sign`] (`Zero` for `±0`; `Zero` for NaN).
    pub fn sign(&self) -> Sign {
        match &self.repr {
            Repr::Normal { neg, .. } | Repr::Inf(neg) => sign_of(*neg),
            _ => Sign::Zero,
        }
    }

    /// Returns the working precision in bits.
    #[inline]
    pub fn precision(&self) -> u64 {
        self.precision
    }

    /// Returns the base-2 exponent of a finite non-zero value, else `None`.
    pub fn exponent(&self) -> Option<i64> {
        match &self.repr {
            Repr::Normal { exp, .. } => Some(*exp),
            _ => None,
        }
    }

    /// Returns the unsigned significand of a finite non-zero value, else `None`.
    pub fn significand(&self) -> Option<&Nat> {
        match &self.repr {
            Repr::Normal { sig, .. } => Some(sig),
            _ => None,
        }
    }

    // --- conversions out ---

    /// Returns the exact value as a [`Rational`], or `None` for NaN/`±∞`.
    pub fn to_rational(&self) -> Option<Rational> {
        match &self.repr {
            Repr::Zero(_) => Some(Rational::ZERO),
            Repr::Normal { neg, sig, exp } => {
                let sign = sign_of(*neg);
                Some(if *exp >= 0 {
                    Rational::from_integer(Int::from_sign_magnitude(sign, sig.shl(*exp as u64)))
                } else {
                    let num = Int::from_sign_magnitude(sign, sig.clone());
                    let den = Int::ONE.mul_2k((-exp) as u32);
                    Rational::new(num, den)
                })
            }
            _ => None,
        }
    }

    /// Returns the value as the nearest `f64` (best-effort; may be `±inf`/`0` on
    /// extreme exponents). NaN and `±∞` map to `f64` NaN/`±inf`.
    pub fn to_f64(&self) -> f64 {
        match &self.repr {
            Repr::NaN => f64::NAN,
            Repr::Inf(neg) => {
                if *neg {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            }
            Repr::Zero(neg) => {
                if *neg {
                    -0.0
                } else {
                    0.0
                }
            }
            Repr::Normal { neg, sig, exp } => {
                let mant = Int::from(sig.clone()).to_f64();
                let scaled = mant * exp2(*exp);
                if *neg { -scaled } else { scaled }
            }
        }
    }

    /// Returns the value as the nearest `f32` (via `f64`).
    pub fn to_f32(&self) -> f32 {
        self.to_f64() as f32
    }

    /// Formats the value as a fixed-point decimal string with `frac_digits`
    /// digits after the point, rounded half-up. NaN/`±∞` render as `"NaN"`,
    /// `"inf"`, `"-inf"`.
    pub fn to_decimal_string(&self, frac_digits: u32) -> String {
        match &self.repr {
            Repr::NaN => String::from("NaN"),
            Repr::Inf(true) => String::from("-inf"),
            Repr::Inf(false) => String::from("inf"),
            _ => {
                let r = self.to_rational().expect("finite");
                let mut out = String::new();
                let _ = r.write_decimal(&mut out, frac_digits, false);
                out
            }
        }
    }
}

/// `Sign` from a sign bit.
#[inline]
fn sign_of(neg: bool) -> Sign {
    if neg { Sign::Negative } else { Sign::Positive }
}

impl Repr {
    /// The sign bit (`false` for NaN).
    #[inline]
    fn sign_bit(&self) -> bool {
        match self {
            Repr::NaN => false,
            Repr::Inf(neg) | Repr::Zero(neg) | Repr::Normal { neg, .. } => *neg,
        }
    }
}

/// Best-effort `2^e` as an `f64` by repeated squaring (avoids `powi`'s `i32`
/// range limit).
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
    /// Compares two finite values by magnitude-and-sign (neither NaN).
    fn cmp_finite(&self, other: &Float) -> Ordering {
        let rank = |f: &Float| -> i8 {
            match &f.repr {
                Repr::Inf(true) => -2,
                Repr::Normal { neg: true, .. } => -1,
                Repr::Zero(_) => 0,
                Repr::Normal { neg: false, .. } => 1,
                Repr::Inf(false) => 2,
                Repr::NaN => unreachable!(),
            }
        };
        match rank(self).cmp(&rank(other)) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
        // Same class: only Normal-vs-Normal needs a magnitude compare.
        if let (
            Repr::Normal {
                neg,
                sig: sa,
                exp: ea,
            },
            Repr::Normal {
                sig: sb, exp: eb, ..
            },
        ) = (&self.repr, &other.repr)
        {
            let emin = (*ea).min(*eb);
            let a = sa.shl((*ea - emin) as u64);
            let b = sb.shl((*eb - emin) as u64);
            let m = a.cmp(&b);
            return if *neg { m.reverse() } else { m };
        }
        Ordering::Equal
    }
}

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        Some(self.cmp_finite(other))
    }
}

impl FromStr for Float {
    type Err = Error;

    /// Parses a decimal (`"1.5"`, `"-3/4"`, `"42"`) at 53 bits of precision, or
    /// the tokens `inf`/`-inf`/`nan` (case-insensitive). Use
    /// [`Float::from_rational`] for an explicit precision.
    fn from_str(s: &str) -> Result<Self> {
        match s.trim() {
            t if t.eq_ignore_ascii_case("nan") => Ok(Float::nan(53)),
            t if t.eq_ignore_ascii_case("inf") || t.eq_ignore_ascii_case("+inf") => {
                Ok(Float::infinity(53))
            }
            t if t.eq_ignore_ascii_case("-inf") => Ok(Float::neg_infinity(53)),
            t => {
                let r: Rational = t.parse()?;
                Ok(Float::from_rational(&r, 53, RoundingMode::Nearest))
            }
        }
    }
}

impl fmt::Display for Float {
    /// Formats the exact value as `±significand·2^exponent` (or a special token).
    /// For decimal output use [`Float::to_decimal_string`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.repr {
            Repr::NaN => f.write_str("NaN"),
            Repr::Inf(true) => f.write_str("-inf"),
            Repr::Inf(false) => f.write_str("inf"),
            Repr::Zero(true) => f.write_str("-0"),
            Repr::Zero(false) => f.write_str("0"),
            Repr::Normal { neg, sig, exp } => {
                if *neg {
                    f.write_str("-")?;
                }
                write!(f, "{sig}·2^{exp}")
            }
        }
    }
}

impl fmt::Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Float({self} @ {}bit)", self.precision)
    }
}

// ===========================================================================
// Transcendental functions (M8).
//
// Each public function evaluates at an internal working precision and rounds to
// the caller's target precision under Ziv's strategy: if the working result is
// too close to a rounding boundary to round unambiguously, the working
// precision is increased and the value recomputed. All internal arithmetic uses
// round-to-nearest; only the final rounding honours the caller's mode.
// ===========================================================================

const NEAR: RoundingMode = RoundingMode::Nearest;

/// A finite integer Float at working precision `w`.
fn iflt(k: i64, w: u64) -> Float {
    Float::from_int(&Int::from_i64(k), w, NEAR)
}

/// A finite rational Float `num/den` at working precision `w`.
fn rflt(num: i64, den: i64, w: u64) -> Float {
    Float::from_rational(
        &Rational::new(Int::from_i64(num), Int::from_i64(den)),
        w,
        NEAR,
    )
}

/// True if `term` is negligible relative to `sum` at working precision `w`.
fn negligible(term: &Float, sum: &Float, w: u64) -> bool {
    match (term.exponent(), sum.exponent()) {
        (None, _) => true, // term is zero
        (Some(te), Some(se)) => te < se - (w as i64) - 4,
        (Some(_), None) => false,
    }
}

impl Float {
    /// Multiplies by `2^k` exactly (adjusts the exponent).
    fn scale_pow2(&self, k: i64) -> Float {
        match &self.repr {
            Repr::Normal { neg, sig, exp } => Float {
                repr: Repr::Normal {
                    neg: *neg,
                    sig: sig.clone(),
                    exp: exp + k,
                },
                precision: self.precision,
            },
            _ => self.clone(),
        }
    }

    /// Rounds a finite value to the nearest integer (ties toward +∞ via
    /// `floor(x + 1/2)`), returned as an [`Int`].
    fn round_to_int(&self) -> Int {
        let w = self.precision + 2;
        let shifted = self.add(&rflt(1, 2, w), w, NEAR);
        shifted
            .to_rational()
            .map(|r| r.floor())
            .unwrap_or(Int::ZERO)
    }

    /// Ziv driver: evaluate `f(working_precision)` and round to `prec`,
    /// growing the working precision until the rounding is unambiguous.
    fn ziv<F: Fn(u64) -> Float>(prec: u64, mode: RoundingMode, f: F) -> Float {
        let prec = prec.max(1);
        let mut guard = 48u64;
        loop {
            let val = f(prec + guard);
            if let Some(r) = round_ziv(&val, prec, mode) {
                return r;
            }
            if guard > prec + 4096 {
                return val.round(prec, mode); // give up: best effort
            }
            guard = guard.saturating_mul(2);
        }
    }

    // --- constants ---

    /// Returns π rounded to `precision` bits.
    pub fn pi(precision: u64, mode: RoundingMode) -> Float {
        Float::ziv(precision, mode, pi_at)
    }

    /// Returns ln 2 rounded to `precision` bits.
    pub fn ln2(precision: u64, mode: RoundingMode) -> Float {
        Float::ziv(precision, mode, ln2_at)
    }

    /// Returns Euler's number e rounded to `precision` bits.
    pub fn e(precision: u64, mode: RoundingMode) -> Float {
        Float::ziv(precision, mode, |w| exp_at(&iflt(1, w), w))
    }

    // --- functions ---

    /// Returns `e^self`, correctly rounded. `exp(±∞)`/`exp(0)` handled per IEEE.
    pub fn exp(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(true) => Float::zero(precision),
            Repr::Inf(false) => Float::infinity(precision),
            Repr::Zero(_) => Float::from_int(&Int::ONE, precision, mode),
            Repr::Normal { .. } => {
                let x = self.clone();
                Float::ziv(precision, mode, move |w| exp_at(&x.round(w, NEAR), w))
            }
        }
    }

    /// Returns the natural logarithm `ln(self)`, correctly rounded. `ln(x<0)` is
    /// NaN, `ln(0)` is `−∞`, `ln(+∞)` is `+∞`.
    pub fn ln(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(false) => Float::infinity(precision),
            Repr::Inf(true) => Float::nan(precision),
            Repr::Zero(_) => Float::neg_infinity(precision),
            Repr::Normal { neg: true, .. } => Float::nan(precision),
            Repr::Normal { .. } => {
                let x = self.clone();
                Float::ziv(precision, mode, move |w| ln_at(&x.round(w, NEAR), w))
            }
        }
    }

    /// Returns `sin(self)`, correctly rounded (finite arguments).
    pub fn sin(&self, precision: u64, mode: RoundingMode) -> Float {
        if !self.is_finite() {
            return Float::nan(precision);
        }
        if self.is_zero() {
            return Float::zero_signed(self.is_sign_negative(), precision);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| sin_cos_at(&x.round(w, NEAR), w).0)
    }

    /// Returns `cos(self)`, correctly rounded (finite arguments).
    pub fn cos(&self, precision: u64, mode: RoundingMode) -> Float {
        if !self.is_finite() {
            return Float::nan(precision);
        }
        if self.is_zero() {
            return Float::from_int(&Int::ONE, precision, mode);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| sin_cos_at(&x.round(w, NEAR), w).1)
    }

    /// Returns `tan(self)`, correctly rounded (finite arguments).
    pub fn tan(&self, precision: u64, mode: RoundingMode) -> Float {
        if !self.is_finite() {
            return Float::nan(precision);
        }
        if self.is_zero() {
            return Float::zero_signed(self.is_sign_negative(), precision);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| {
            let (s, c) = sin_cos_at(&x.round(w, NEAR), w);
            s.div(&c, w, NEAR)
        })
    }

    /// Returns `atan(self)`, correctly rounded (finite arguments).
    pub fn atan(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(neg) => {
                // atan(±∞) = ±π/2
                let half_pi = Float::pi(precision + 8, NEAR).scale_pow2(-1);
                if *neg { half_pi.neg() } else { half_pi }.round(precision, mode)
            }
            Repr::Zero(_) => Float::zero_signed(self.is_sign_negative(), precision),
            Repr::Normal { .. } => {
                let x = self.clone();
                Float::ziv(precision, mode, move |w| atan_at(&x.round(w, NEAR), w))
            }
        }
    }
}

/// Returns `Some(rounded)` if `val` can be rounded to `prec` bits unambiguously,
/// else `None` (the caller should recompute at higher precision).
fn round_ziv(val: &Float, prec: u64, mode: RoundingMode) -> Option<Float> {
    match &val.repr {
        Repr::Normal { sig, .. } => {
            let w = sig.bit_len();
            if w <= prec {
                return Some(val.round(prec, mode));
            }
            let drop = w - prec;
            const CHECK: u64 = 24;
            if drop <= CHECK + 1 {
                return None;
            }
            let low = sig.low_bits(drop);
            let margin = Nat::one().shl(drop - CHECK);
            let full = Nat::one().shl(drop);
            let ambiguous = if mode == RoundingMode::Nearest {
                let half = Nat::one().shl(drop - 1);
                let dist = if low >= half {
                    low.checked_sub(&half).unwrap()
                } else {
                    half.checked_sub(&low).unwrap()
                };
                dist < margin
            } else {
                low < margin || full.checked_sub(&low).unwrap() < margin
            };
            if ambiguous {
                None
            } else {
                Some(val.round(prec, mode))
            }
        }
        // NaN / ±∞ / ±0 are exact.
        _ => Some(val.round(prec, mode)),
    }
}

/// `atanh(x) = x + x³/3 + x⁵/5 + …` at working precision `w` (needs `|x| < 1`).
fn atanh_series(x: &Float, w: u64) -> Float {
    let x2 = x.mul(x, w, NEAR);
    let mut pow = x.clone();
    let mut sum = x.clone();
    let mut k: i64 = 1;
    loop {
        pow = pow.mul(&x2, w, NEAR);
        let term = pow.div(&iflt(2 * k + 1, w), w, NEAR);
        sum = sum.add(&term, w, NEAR);
        if negligible(&term, &sum, w) {
            break;
        }
        k += 1;
    }
    sum
}

/// `atan(x) = x − x³/3 + x⁵/5 − …` at working precision `w` (needs `|x| ≤ 1`).
fn atan_series(x: &Float, w: u64) -> Float {
    let x2 = x.mul(x, w, NEAR);
    let mut pow = x.clone();
    let mut sum = x.clone();
    let mut k: i64 = 1;
    loop {
        pow = pow.mul(&x2, w, NEAR);
        let term = pow.div(&iflt(2 * k + 1, w), w, NEAR);
        sum = if k % 2 == 1 {
            sum.sub(&term, w, NEAR)
        } else {
            sum.add(&term, w, NEAR)
        };
        if negligible(&term, &sum, w) {
            break;
        }
        k += 1;
    }
    sum
}

/// π via Machin's formula, `16·atan(1/5) − 4·atan(1/239)`, at precision `w`.
fn pi_at(w: u64) -> Float {
    let a1 = atan_series(&rflt(1, 5, w), w);
    let a2 = atan_series(&rflt(1, 239, w), w);
    iflt(16, w)
        .mul(&a1, w, NEAR)
        .sub(&iflt(4, w).mul(&a2, w, NEAR), w, NEAR)
}

/// ln 2 via `2·atanh(1/3)` at precision `w`.
fn ln2_at(w: u64) -> Float {
    iflt(2, w).mul(&atanh_series(&rflt(1, 3, w), w), w, NEAR)
}

/// `e^x` at precision `w` via range reduction `x = k·ln2 + r` and a Taylor sum.
fn exp_at(x: &Float, w: u64) -> Float {
    let ln2 = ln2_at(w);
    let k = x.div(&ln2, w, NEAR).round_to_int();
    let ki = k.to_i64().unwrap_or(0);
    let r = x.sub(&Float::from_int(&k, w, NEAR).mul(&ln2, w, NEAR), w, NEAR);
    // exp(r) = Σ rⁿ/n!
    let mut term = iflt(1, w);
    let mut sum = iflt(1, w);
    let mut n: i64 = 1;
    loop {
        term = term.mul(&r, w, NEAR).div(&iflt(n, w), w, NEAR);
        sum = sum.add(&term, w, NEAR);
        if negligible(&term, &sum, w) {
            break;
        }
        n += 1;
    }
    sum.scale_pow2(ki)
}

/// `ln(x)` for finite `x > 0` at precision `w`.
fn ln_at(x: &Float, w: u64) -> Float {
    if x.is_zero() {
        return Float::neg_infinity(w);
    }
    let bits = x.significand().map(|s| s.bit_len() as i64).unwrap_or(0);
    let e = x.exponent().unwrap_or(0) + bits - 1; // floor(log2 x)
    let m = x.scale_pow2(-e); // m ∈ [1, 2)
    // ln(m) = 2·atanh((m−1)/(m+1))
    let one = iflt(1, w);
    let y = m.sub(&one, w, NEAR).div(&m.add(&one, w, NEAR), w, NEAR);
    let ln_m = iflt(2, w).mul(&atanh_series(&y, w), w, NEAR);
    iflt(e, w).mul(&ln2_at(w), w, NEAR).add(&ln_m, w, NEAR)
}

/// `(sin x, cos x)` at precision `w`, via reduction to `[-π/4, π/4]`.
fn sin_cos_at(x: &Float, w: u64) -> (Float, Float) {
    let pi = pi_at(w);
    let half_pi = pi.scale_pow2(-1);
    // q = round(x / (π/2)); r = x − q·(π/2) ∈ [−π/4, π/4].
    let q = x.div(&half_pi, w, NEAR).round_to_int();
    let r = x.sub(
        &Float::from_int(&q, w, NEAR).mul(&half_pi, w, NEAR),
        w,
        NEAR,
    );
    let (sr, cr) = sin_cos_series(&r, w);
    // Reconstruct from the quadrant q mod 4.
    let quad = q.rem_euclid(&Int::from_i64(4)).to_i64().unwrap_or(0);
    match quad {
        0 => (sr, cr),
        1 => (cr, sr.neg()),
        2 => (sr.neg(), cr.neg()),
        _ => (cr.neg(), sr),
    }
}

/// `(sin r, cos r)` by Taylor series for small `|r|`, at precision `w`.
fn sin_cos_series(r: &Float, w: u64) -> (Float, Float) {
    let r2 = r.mul(r, w, NEAR);
    // sin
    let mut term = r.clone();
    let mut sin = r.clone();
    let mut n: i64 = 1;
    loop {
        // term_{k} = -term_{k-1} · r² / ((2k)(2k+1))
        term = term
            .mul(&r2, w, NEAR)
            .div(&iflt(2 * n * (2 * n + 1), w), w, NEAR)
            .neg();
        sin = sin.add(&term, w, NEAR);
        if negligible(&term, &sin, w) {
            break;
        }
        n += 1;
    }
    // cos
    let mut cterm = iflt(1, w);
    let mut cos = iflt(1, w);
    let mut m: i64 = 1;
    loop {
        cterm = cterm
            .mul(&r2, w, NEAR)
            .div(&iflt((2 * m - 1) * (2 * m), w), w, NEAR)
            .neg();
        cos = cos.add(&cterm, w, NEAR);
        if negligible(&cterm, &cos, w) {
            break;
        }
        m += 1;
    }
    (sin, cos)
}

/// `atan(x)` for finite non-zero `x` at precision `w`.
///
/// Reduces `|x| > 1` by the complement `atan(x) = π/2 − atan(1/x)`, then halves
/// the argument via `atan(t) = 2·atan(t/(1+√(1+t²)))` until it is small enough
/// for the Taylor series to converge quickly (the raw series is linear near 1).
fn atan_at(x: &Float, w: u64) -> Float {
    let one = iflt(1, w);
    let neg = x.is_sign_negative();
    let ax = x.abs();
    let complement = ax.partial_cmp(&one) == Some(Ordering::Greater);
    let mut arg = if complement {
        one.div(&ax, w, NEAR)
    } else {
        ax
    };

    let quarter = rflt(1, 4, w);
    let mut halvings = 0u32;
    while arg.partial_cmp(&quarter) == Some(Ordering::Greater) {
        let root = one.add(&arg.mul(&arg, w, NEAR), w, NEAR).sqrt(w, NEAR);
        arg = arg.div(&one.add(&root, w, NEAR), w, NEAR);
        halvings += 1;
    }

    let mut result = atan_series(&arg, w);
    for _ in 0..halvings {
        result = result.scale_pow2(1); // × 2
    }
    if complement {
        result = pi_at(w).scale_pow2(-1).sub(&result, w, NEAR);
    }
    if neg { result.neg() } else { result }
}
