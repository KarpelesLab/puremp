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

use alloc::string::{String, ToString};

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

    /// Largest integer `≤ self` (`⌊self⌋`), or `None` if `self` is NaN/∞.
    pub fn floor(&self) -> Option<Int> {
        self.to_rational().map(|r| r.floor())
    }

    /// Smallest integer `≥ self` (`⌈self⌉`), or `None` if `self` is NaN/∞.
    pub fn ceil(&self) -> Option<Int> {
        self.to_rational().map(|r| r.ceil())
    }

    /// `self` truncated toward zero, or `None` if `self` is NaN/∞.
    pub fn trunc(&self) -> Option<Int> {
        self.to_rational().map(|r| r.trunc())
    }

    /// Nearest integer, ties to even (Mathematica's `Round`), or `None` if NaN/∞.
    pub fn round_to_int(&self) -> Option<Int> {
        self.to_rational().map(|r| r.round())
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
            ) => {
                // Fold exact power-of-two factors into the exponents so
                // integer-valued operands multiply at their true size.
                let (sa, ea) = strip_pow2(sa, *ea);
                let (sb, eb) = strip_pow2(sb, *eb);
                Float::round_raw(na ^ nb, sa.mul(&sb), ea + eb, precision, mode)
            }
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
                // Trailing zero bits of a significand are an exact power-of-two
                // factor: fold them into the exponent so integer-valued
                // divisors (e.g. a series' `n · 2^k` at working precision)
                // reduce to a division by a small value.
                let (sa, ea) = strip_pow2(sa, *ea);
                let (sb, eb) = strip_pow2(sb, *eb);
                let (sa, sb) = (sa.as_ref(), sb.as_ref());
                // Shift so the quotient has ≥ precision + 2 bits regardless of
                // the operands' own bit lengths (they may differ from `precision`).
                let guard = (precision as i64 + 2 + sb.bit_len() as i64 - sa.bit_len() as i64)
                    .max(2) as u64;
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

    /// Returns the shortest decimal string that, parsed back at this value's
    /// precision, recovers it exactly — the round-trip-minimal representation
    /// (like `f64`'s `Display`). NaN/`±∞` render as their tokens.
    pub fn to_shortest_string(&self) -> String {
        match &self.repr {
            Repr::NaN => return String::from("NaN"),
            Repr::Inf(true) => return String::from("-inf"),
            Repr::Inf(false) => return String::from("inf"),
            Repr::Zero(_) => return String::from("0"),
            Repr::Normal { .. } => {}
        }
        let value = self.to_rational().expect("finite non-zero");
        let abs = value.abs();

        // Decimal exponent of the leading digit: v ∈ [1, 10) with abs = v·10^e10.
        let ten = Rational::from(Int::from_i64(10));
        let one = Rational::ONE;
        let mut e10 = 0i64;
        let mut v = abs;
        while v >= ten {
            v = v.div(&ten);
            e10 += 1;
        }
        while v < one {
            v = v.mul(&ten);
            e10 -= 1;
        }

        // Try 1, 2, 3, … significant digits until the string round-trips.
        let max_digits = (self.precision as usize) / 3 + 4;
        for d in 1..=max_digits {
            let scale = Rational::from(Int::from_i64(10).pow((d - 1) as u32));
            let scaled = v.mul(&scale);
            // Round half-up to the nearest integer.
            let m = scaled.add(&Rational::power_of_two(-1)).floor();
            let mut ds = m.to_string();
            let exp = e10 + (ds.len() as i64 - d as i64);
            while ds.len() > 1 && ds.ends_with('0') {
                ds.pop();
            }
            let candidate = format_plain(self.is_sign_negative(), &ds, exp);
            if let Ok(r) = candidate.parse::<Rational>()
                && Float::from_rational(&r, self.precision, RoundingMode::Nearest) == *self
            {
                return candidate;
            }
        }
        // Fallback: the exact (long) decimal is guaranteed to round-trip.
        self.to_decimal_string(max_digits as u32)
    }

    /// Returns an exact, losslessly round-trippable string encoding
    /// (`[-]<significand>p<exp>@<precision>`, or `nan@p`/`[-]inf@p`/`[-]0@p`).
    /// See [`Float::from_exact_string`].
    pub fn to_exact_string(&self) -> String {
        match &self.repr {
            Repr::NaN => alloc::format!("nan@{}", self.precision),
            Repr::Inf(neg) => {
                alloc::format!("{}inf@{}", if *neg { "-" } else { "" }, self.precision)
            }
            Repr::Zero(neg) => {
                alloc::format!("{}0@{}", if *neg { "-" } else { "" }, self.precision)
            }
            Repr::Normal { neg, sig, exp } => alloc::format!(
                "{}{sig}p{exp}@{}",
                if *neg { "-" } else { "" },
                self.precision
            ),
        }
    }

    /// Parses the exact encoding produced by [`Float::to_exact_string`].
    pub fn from_exact_string(s: &str) -> Result<Float> {
        let (body, prec_s) = s.rsplit_once('@').ok_or(Error::Parse)?;
        let precision: u64 = prec_s.parse().map_err(|_| Error::Parse)?;
        let (neg, rest) = match body.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, body),
        };
        if rest.eq_ignore_ascii_case("nan") {
            return Ok(Float::nan(precision));
        }
        if rest.eq_ignore_ascii_case("inf") {
            return Ok(Float::inf_signed(neg, precision));
        }
        if rest == "0" {
            return Ok(Float::zero_signed(neg, precision));
        }
        let (sig_s, exp_s) = rest.split_once('p').ok_or(Error::Parse)?;
        let sig = Nat::from_str(sig_s)?;
        let exp: i64 = exp_s.parse().map_err(|_| Error::Parse)?;
        Ok(Float::round_raw(neg, sig, exp, precision, RoundingMode::Nearest).0)
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

/// Formats significant digits `ds` (no trailing zeros) as a plain decimal, where
/// `exp` is the base-10 exponent of the leading digit.
fn format_plain(neg: bool, ds: &str, exp: i64) -> String {
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if exp >= 0 {
        let ip_len = (exp + 1) as usize;
        if ds.len() <= ip_len {
            out.push_str(ds);
            for _ in 0..ip_len - ds.len() {
                out.push('0');
            }
        } else {
            out.push_str(&ds[..ip_len]);
            out.push('.');
            out.push_str(&ds[ip_len..]);
        }
    } else {
        out.push_str("0.");
        for _ in 0..(-exp - 1) {
            out.push('0');
        }
        out.push_str(ds);
    }
    out
}

/// Splits a non-zero significand into its odd part and an adjusted exponent:
/// trailing zero bits are an exact `2^k` factor folded into `exp`. Returns the
/// significand borrowed when there is nothing to strip.
fn strip_pow2(sig: &Nat, exp: i64) -> (alloc::borrow::Cow<'_, Nat>, i64) {
    let tz = sig.trailing_zeros();
    if tz == 0 {
        (alloc::borrow::Cow::Borrowed(sig), exp)
    } else {
        (alloc::borrow::Cow::Owned(sig.shr(tz)), exp + tz as i64)
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

// Value-consuming operators for ergonomics (e.g. `Complex<Float>`). Precision
// policy: the result carries the larger of the two operands' precisions and
// rounds to nearest. For explicit control use the `add`/`sub`/`mul`/`div` methods.
macro_rules! float_binop {
    ($tr:ident, $method:ident, $inherent:ident) => {
        impl core::ops::$tr for Float {
            type Output = Float;
            #[inline]
            fn $method(self, rhs: Float) -> Float {
                let p = self.precision().max(rhs.precision());
                Float::$inherent(&self, &rhs, p, RoundingMode::Nearest)
            }
        }
        impl core::ops::$tr<&Float> for &Float {
            type Output = Float;
            #[inline]
            fn $method(self, rhs: &Float) -> Float {
                let p = self.precision().max(rhs.precision());
                Float::$inherent(self, rhs, p, RoundingMode::Nearest)
            }
        }
    };
}
float_binop!(Add, add, add);
float_binop!(Sub, sub, sub);
float_binop!(Mul, mul, mul);
float_binop!(Div, div, div);

impl core::ops::Neg for Float {
    type Output = Float;
    #[inline]
    fn neg(self) -> Float {
        Float::neg(&self)
    }
}
impl core::ops::Neg for &Float {
    type Output = Float;
    #[inline]
    fn neg(self) -> Float {
        Float::neg(self)
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

/// Rewrites a plain decimal string (as produced by the `to_*_string` methods)
/// into scientific notation `d.dddde±X`. Special tokens pass through.
fn plain_to_scientific(s: &str, upper: bool) -> String {
    if matches!(s, "NaN" | "inf" | "-inf") {
        return String::from(s);
    }
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let e_char = if upper { 'E' } else { 'e' };
    let (int_part, frac_part) = match body.split_once('.') {
        Some((i, f)) => (i, f),
        None => (body, ""),
    };
    let combined: String = int_part.chars().chain(frac_part.chars()).collect();
    let point = int_part.len();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    match combined.find(|c| c != '0') {
        None => {
            // Zero.
            out.push('0');
            out.push(e_char);
            out.push('0');
        }
        Some(p) => {
            let exp = point as i64 - 1 - p as i64;
            let mut sig = &combined[p..];
            sig = sig.trim_end_matches('0');
            let bytes = sig.as_bytes();
            out.push(bytes[0] as char);
            if bytes.len() > 1 {
                out.push('.');
                out.push_str(&sig[1..]);
            }
            out.push(e_char);
            out.push_str(&alloc::format!("{exp}"));
        }
    }
    out
}

impl fmt::Display for Float {
    /// Formats the value in decimal. With a precision (`{:.N}`) it prints `N`
    /// fractional digits (correctly rounded); otherwise it prints the shortest
    /// decimal that round-trips. Special values print as `NaN`/`inf`/`-inf`/`0`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match f.precision() {
            Some(p) => self.to_decimal_string(p as u32),
            None => self.to_shortest_string(),
        };
        f.write_str(&s)
    }
}

impl fmt::LowerExp for Float {
    /// Scientific notation, e.g. `1.5e3`. The mantissa is the shortest that
    /// round-trips; the precision flag is not applied.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&plain_to_scientific(&self.to_shortest_string(), false))
    }
}

impl fmt::UpperExp for Float {
    /// Scientific notation with an uppercase `E`, e.g. `1.5E3`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&plain_to_scientific(&self.to_shortest_string(), true))
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

/// Working precision (bits) at or above which `Float::ln` switches from
/// argument reduction plus the term-by-term `atanh` series (which costs
/// `O(n²)`) to the AGM formula `ln(s) = π/(2·M(1, 4/s))` (`O(M(n)·log n)`).
///
/// The crossover was measured (see the `agm_crossover` ignored bench): AGM is
/// already ~4× faster at 4096 bits and the lead widens with precision (≈25× at
/// 16 kbit, ≈73× at 128 kbit), so the threshold sits just below the smallest
/// measured win. Inputs within `2^-32` of 1 stay on the series regardless (see
/// [`ln_near_one`]).
///
/// (A Brent–Salamin AGM π was also implemented and benchmarked, but the
/// existing Machin binary-split series is already `O(M(n)·log n)` and matched or
/// beat the AGM at every precision — slightly *faster* above ~256 kbit — so no
/// AGM π path is wired in.)
const LN_AGM_THRESHOLD: u64 = 1 << 12;

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
    fn round_half_up_to_int(&self) -> Int {
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

    /// Returns the Euler–Mascheroni constant γ ≈ 0.5772 rounded to `precision`
    /// bits (Mathematica's `EulerGamma`).
    pub fn euler_gamma(precision: u64, mode: RoundingMode) -> Float {
        Float::ziv(precision, mode, gamma_at)
    }

    /// Returns Catalan's constant G ≈ 0.9160 rounded to `precision` bits
    /// (Mathematica's `Catalan`).
    pub fn catalan(precision: u64, mode: RoundingMode) -> Float {
        Float::ziv(precision, mode, catalan_at)
    }

    // --- functions ---

    /// Returns `e^self`, correctly rounded. `exp(±∞)`/`exp(0)` handled per IEEE.
    pub fn exp(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(true) => Float::zero(precision),
            Repr::Inf(false) => Float::infinity(precision),
            Repr::Zero(_) => Float::from_int(&Int::ONE, precision, mode),
            Repr::Normal { .. } => match crate::float_mp::exp_mp(self, precision, mode) {
                // Multi-prime fast path (returns a correctly-rounded result or None).
                Some(v) => v,
                None => {
                    let x = self.clone();
                    Float::ziv(precision, mode, move |w| exp_at(&x.round(w, NEAR), w))
                }
            },
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

    /// Returns `sinh(self) = (eˣ − e⁻ˣ)/2`, correctly rounded.
    pub fn sinh(&self, precision: u64, mode: RoundingMode) -> Float {
        if self.is_nan() {
            return Float::nan(precision);
        }
        if self.is_infinite() {
            return self.clone().round(precision, mode);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| {
            let ex = x.exp(w, NEAR);
            let emx = x.neg().exp(w, NEAR);
            ex.sub(&emx, w, NEAR).scale_pow2(-1)
        })
    }

    /// Returns `cosh(self) = (eˣ + e⁻ˣ)/2`, correctly rounded.
    pub fn cosh(&self, precision: u64, mode: RoundingMode) -> Float {
        if self.is_nan() {
            return Float::nan(precision);
        }
        if self.is_infinite() {
            return Float::infinity(precision);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| {
            let ex = x.exp(w, NEAR);
            let emx = x.neg().exp(w, NEAR);
            ex.add(&emx, w, NEAR).scale_pow2(-1)
        })
    }

    /// Returns `tanh(self) = sinh/cosh`, correctly rounded.
    pub fn tanh(&self, precision: u64, mode: RoundingMode) -> Float {
        if self.is_nan() {
            return Float::nan(precision);
        }
        if self.is_infinite() {
            return Float::from_int(&Int::from_i64(self.signum_i()), precision, mode);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| {
            x.sinh(w, NEAR).div(&x.cosh(w, NEAR), w, NEAR)
        })
    }

    /// Returns `asin(self)` (domain `[-1, 1]`; `NaN` outside), correctly rounded.
    pub fn asin(&self, precision: u64, mode: RoundingMode) -> Float {
        if !self.is_finite() {
            return Float::nan(precision);
        }
        let x = self.clone();
        // asin(x) = atan(x / √(1 − x²)); |x| > 1 yields √(negative) = NaN.
        Float::ziv(precision, mode, move |w| {
            let one = Float::from_int(&Int::ONE, w, NEAR);
            let xr = x.round(w, NEAR);
            let denom = one.sub(&xr.mul(&xr, w, NEAR), w, NEAR).sqrt(w, NEAR);
            xr.div(&denom, w, NEAR).atan(w, NEAR)
        })
    }

    /// Returns `acos(self) = π/2 − asin(self)`, correctly rounded.
    pub fn acos(&self, precision: u64, mode: RoundingMode) -> Float {
        if !self.is_finite() {
            return Float::nan(precision);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| {
            let half_pi = Float::pi(w, NEAR).scale_pow2(-1);
            half_pi.sub(&x.asin(w, NEAR), w, NEAR)
        })
    }

    /// Returns `atan2(self, x)` — the angle of the point `(x, self)` in
    /// `(-π, π]` — correctly rounded (finite arguments).
    pub fn atan2(&self, x: &Float, precision: u64, mode: RoundingMode) -> Float {
        if self.is_nan() || x.is_nan() {
            return Float::nan(precision);
        }
        let (y, x) = (self.clone(), x.clone());
        Float::ziv(precision, mode, move |w| {
            let pi = Float::pi(w, NEAR);
            if x.is_zero() {
                // ±π/2 by the sign of y (0 if y is also zero).
                if y.is_zero() {
                    return Float::zero(w);
                }
                let hp = pi.scale_pow2(-1);
                return if y.is_sign_negative() { hp.neg() } else { hp };
            }
            let base = y.div(&x, w, NEAR).atan(w, NEAR);
            if !x.is_sign_negative() {
                base
            } else if !y.is_sign_negative() {
                base.add(&pi, w, NEAR)
            } else {
                base.sub(&pi, w, NEAR)
            }
        })
    }

    /// Returns `self` raised to the floating exponent `y` via `exp(y·ln self)`.
    /// Defined for `self > 0`; `self == 0` gives `0` (for `y > 0`), and `self < 0`
    /// gives `NaN`.
    pub fn pow(&self, y: &Float, precision: u64, mode: RoundingMode) -> Float {
        if self.is_nan() || y.is_nan() {
            return Float::nan(precision);
        }
        if y.is_zero() {
            return Float::from_int(&Int::ONE, precision, mode);
        }
        if self.is_zero() {
            return if y.is_sign_negative() {
                Float::infinity(precision)
            } else {
                Float::zero(precision)
            };
        }
        if self.is_sign_negative() {
            return Float::nan(precision);
        }
        let (base, y) = (self.clone(), y.clone());
        Float::ziv(precision, mode, move |w| {
            y.mul(&base.ln(w, NEAR), w, NEAR).exp(w, NEAR)
        })
    }

    /// Returns `asinh(self) = ln(self + √(self² + 1))`, correctly rounded.
    pub fn asinh(&self, precision: u64, mode: RoundingMode) -> Float {
        if !self.is_finite() {
            return self.clone().round(precision, mode);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| {
            let one = Float::from_int(&Int::ONE, w, NEAR);
            let xr = x.round(w, NEAR);
            let root = xr.mul(&xr, w, NEAR).add(&one, w, NEAR).sqrt(w, NEAR);
            xr.add(&root, w, NEAR).ln(w, NEAR)
        })
    }

    /// Returns `acosh(self) = ln(self + √(self² − 1))` for `self ≥ 1` (else
    /// `NaN`), correctly rounded.
    pub fn acosh(&self, precision: u64, mode: RoundingMode) -> Float {
        if self.is_nan() {
            return Float::nan(precision);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| {
            let one = Float::from_int(&Int::ONE, w, NEAR);
            let xr = x.round(w, NEAR);
            // √(x²−1) is NaN for x < 1, so the result is NaN there.
            let root = xr.mul(&xr, w, NEAR).sub(&one, w, NEAR).sqrt(w, NEAR);
            xr.add(&root, w, NEAR).ln(w, NEAR)
        })
    }

    /// Returns `atanh(self) = ½·ln((1 + self)/(1 − self))` for `|self| < 1`
    /// (else `±∞`/`NaN`), correctly rounded.
    pub fn atanh(&self, precision: u64, mode: RoundingMode) -> Float {
        if !self.is_finite() {
            return Float::nan(precision);
        }
        let x = self.clone();
        Float::ziv(precision, mode, move |w| {
            let one = Float::from_int(&Int::ONE, w, NEAR);
            let xr = x.round(w, NEAR);
            let ratio = one.add(&xr, w, NEAR).div(&one.sub(&xr, w, NEAR), w, NEAR);
            ratio.ln(w, NEAR).scale_pow2(-1)
        })
    }

    /// Returns the error function `erf(self) = (2/√π)∫₀ˣ e^{−t²} dt`, correctly
    /// rounded. `erf` is odd, `erf(±0) = ±0` and `erf(±∞) = ±1`.
    ///
    /// Moderate arguments use the all-positive (Kummer) series
    /// `erf(x) = (2/√π) e^{−x²} Σ_{n≥0} 2ⁿ x^{2n+1} / (1·3·5···(2n+1))`
    /// (DLMF §7.6, no cancellation); large arguments use `1 − erfc(|x|)` via the
    /// DLMF §7.9 continued fraction.
    pub fn erf(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(neg) => Float::from_int(
                if *neg { &Int::MINUS_ONE } else { &Int::ONE },
                precision,
                mode,
            ),
            Repr::Zero(_) => Float::zero_signed(self.is_sign_negative(), precision),
            Repr::Normal { .. } => {
                let x = self.clone();
                Float::ziv(precision, mode, move |w| erf_at(&x.round(w, NEAR), w))
            }
        }
    }

    /// Returns the complementary error function `erfc(self) = 1 − erf(self)`,
    /// correctly rounded. `erfc(0) = 1`, `erfc(+∞) = 0`, `erfc(−∞) = 2`.
    ///
    /// For small `|x|` this is `1 − erf(|x|)` (adjusted for sign); for large
    /// `|x|` the DLMF §7.9 continued fraction is used directly, avoiding the
    /// catastrophic cancellation of `1 − erf` when `erf(x) ≈ 1`.
    pub fn erfc(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(false) => Float::zero(precision),
            Repr::Inf(true) => Float::from_int(&Int::from_i64(2), precision, mode),
            Repr::Zero(_) => Float::from_int(&Int::ONE, precision, mode),
            Repr::Normal { .. } => {
                let x = self.clone();
                Float::ziv(precision, mode, move |w| erfc_at(&x.round(w, NEAR), w))
            }
        }
    }

    /// Returns the Riemann zeta function `ζ(self)` for a real argument, correctly
    /// rounded. Uses Borwein's acceleration of the alternating eta function
    /// `η(s) = Σ_{k≥1} (−1)^{k−1} k^{−s} = (1 − 2^{1−s}) ζ(s)` (P. Borwein, 2000;
    /// Cohen–Rodriguez-Villegas–Zagier, 2000), converging geometrically with
    /// `~0.4·precision` terms, then `ζ(s) = η(s)/(1 − 2^{1−s})`.
    ///
    /// # Domain
    /// Supported for real `s > 0`, `s ≠ 1`. At the pole `s = 1` returns `+∞`;
    /// `ζ(0) = −1/2` is returned exactly; `s < 0` (and `−∞`) return NaN
    /// (unsupported — the functional equation is not implemented). `ζ(+∞) = 1`.
    /// Near `s = 1` the factor `1 − 2^{1−s}` is tiny, so extra working precision
    /// is spent to keep the result correctly rounded.
    pub fn zeta(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(false) => Float::from_int(&Int::ONE, precision, mode),
            Repr::Inf(true) => Float::nan(precision),
            // ζ(0) = −1/2.
            Repr::Zero(_) => Float::from_rational(
                &Rational::new(Int::MINUS_ONE, Int::from_i64(2)),
                precision,
                mode,
            ),
            Repr::Normal { neg: true, .. } => Float::nan(precision),
            Repr::Normal { .. } => {
                // ζ has a simple pole at s = 1.
                if self.partial_cmp(&Float::from_int(&Int::ONE, self.precision, NEAR))
                    == Some(Ordering::Equal)
                {
                    return Float::infinity(precision);
                }
                let s = self.clone();
                Float::ziv(precision, mode, move |w| zeta_at(&s.round(w, NEAR), w))
            }
        }
    }

    /// Returns the gamma function `Γ(self)` for a real argument, correctly
    /// rounded.
    ///
    /// # Algorithm
    /// Stirling's asymptotic series for `ln Γ(z)` (DLMF 5.11.1)
    ///
    /// ```text
    /// ln Γ(z) = (z − ½) ln z − z + ½ ln(2π) + Σ_{k≥1} B₂ₖ / (2k(2k−1) z^{2k−1})
    /// ```
    ///
    /// with the Bernoulli numbers `B₂ₖ` computed as exact rationals. The argument
    /// is first shifted upward, `Γ(x) = Γ(x+m)/(x(x+1)···(x+m−1))`, so that
    /// `z = x+m ≳ precision/4` makes the asymptotic tail fall below the target,
    /// and `Γ(x)` is recovered by `exp(ln Γ(x))`. For `x < ½` the reflection
    /// formula `Γ(x) Γ(1−x) = π / sin(πx)` (DLMF 5.5.3) maps the argument into the
    /// convergent region.
    ///
    /// # Domain
    /// Defined for all real `x` except the poles `x = 0, −1, −2, …`, which return
    /// NaN. `Γ(+∞) = +∞`; `Γ(−∞)` and NaN return NaN. Integer and half-integer
    /// arguments come out exact after rounding (`Γ(n) = (n−1)!`). Very large
    /// negative arguments near a pole demand extra working precision (the
    /// reflection's `sin(πx)` is tiny there); Ziv supplies it, so accuracy is
    /// maintained for moderate `|x|`.
    pub fn gamma(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(false) => Float::infinity(precision),
            Repr::Inf(true) => Float::nan(precision),
            // Γ has a pole at 0.
            Repr::Zero(_) => Float::nan(precision),
            Repr::Normal { .. } => {
                // Poles at the negative integers.
                if let Some(r) = self.to_rational()
                    && r.denominator() == &Int::ONE
                    && r.numerator().is_negative()
                {
                    return Float::nan(precision);
                }
                let x = self.clone();
                Float::ziv(precision, mode, move |w| gamma_fn_at(&x.round(w, NEAR), w))
            }
        }
    }

    /// Returns the natural logarithm of the gamma function `ln Γ(self)` for a
    /// real argument `x > 0`, correctly rounded.
    ///
    /// Uses the same Stirling core as [`Float::gamma`] (see there). `ln Γ(1) =
    /// ln Γ(2) = 0`; `ln Γ(+∞) = +∞`; `x = 0` returns `+∞` (the pole). Arguments
    /// `x < 0` return NaN — this is the real log-gamma on the positive axis, not
    /// `ln|Γ|`.
    pub fn ln_gamma(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(false) => Float::infinity(precision),
            Repr::Inf(true) => Float::nan(precision),
            Repr::Zero(_) => Float::infinity(precision),
            Repr::Normal { neg: true, .. } => Float::nan(precision),
            Repr::Normal { .. } => {
                let x = self.clone();
                Float::ziv(precision, mode, move |w| ln_gamma_at(&x.round(w, NEAR), w))
            }
        }
    }

    /// Returns the Bessel function of the first kind `Jₙ(self)` for integer order
    /// `n` and real argument `x`, correctly rounded.
    ///
    /// # Algorithm
    /// The ascending power series (DLMF 10.2.2)
    ///
    /// ```text
    /// Jₙ(x) = Σ_{m≥0} (−1)ᵐ / (m! (m+n)!) · (x/2)^{2m+n}
    /// ```
    ///
    /// with `J₋ₙ(x) = (−1)ⁿ Jₙ(x)`. The common factor `(x/2)ⁿ/n!` is pulled out
    /// and the remaining sum is accumulated in scaled-integer arithmetic. Because
    /// the series is alternating it loses about `1.443·|x|` bits to cancellation
    /// (the partial sums reach `≈ e^{|x|}`), so that many guard bits are added.
    ///
    /// # Domain
    /// Any integer `n` and finite real `x`; `Jₙ(0) = 0` for `n ≠ 0`, `J₀(0) = 1`.
    /// Correctly rounded for moderate `|x|` (the guard budget grows linearly with
    /// `|x|`, so very large `|x|` is progressively more expensive). Non-finite `x`
    /// returns NaN.
    pub fn bessel_j(&self, n: i64, precision: u64, mode: RoundingMode) -> Float {
        self.bessel(n, precision, mode, true)
    }

    /// Returns the modified Bessel function of the first kind `Iₙ(self)` for
    /// integer order `n` and real argument `x`, correctly rounded.
    ///
    /// Same ascending series as [`Float::bessel_j`] but without the `(−1)ᵐ` sign,
    /// so every term is positive and there is **no** cancellation (DLMF 10.25.2):
    ///
    /// ```text
    /// Iₙ(x) = Σ_{m≥0} 1 / (m! (m+n)!) · (x/2)^{2m+n}
    /// ```
    ///
    /// with `I₋ₙ(x) = Iₙ(x)`. `Iₙ(0) = 0` for `n ≠ 0`, `I₀(0) = 1`. Non-finite `x`
    /// returns NaN.
    pub fn bessel_i(&self, n: i64, precision: u64, mode: RoundingMode) -> Float {
        self.bessel(n, precision, mode, false)
    }

    /// Shared driver for [`Float::bessel_j`] (`alternating = true`) and
    /// [`Float::bessel_i`] (`alternating = false`).
    fn bessel(&self, n: i64, precision: u64, mode: RoundingMode, alternating: bool) -> Float {
        if !self.is_finite() {
            return Float::nan(precision);
        }
        let order = n.unsigned_abs();
        // Sign flip for negative order: J₋ₙ = (−1)ⁿ Jₙ, I₋ₙ = Iₙ.
        let flip = alternating && n < 0 && order % 2 == 1;
        if self.is_zero() {
            // Jₙ(0) = Iₙ(0) = 0 for n ≠ 0, and = 1 for n = 0.
            return if order == 0 {
                Float::from_int(&Int::ONE, precision, mode)
            } else {
                Float::zero(precision)
            };
        }
        let x = self.clone();
        let res = Float::ziv(precision, mode, move |w| {
            bessel_series_at(order, &x.round(w, NEAR), w, alternating)
        });
        if flip { res.neg() } else { res }
    }

    /// Returns the digamma function `ψ(x) = Γ'(x)/Γ(x)` for a real argument,
    /// correctly rounded.
    ///
    /// # Algorithm
    /// The asymptotic series (DLMF 5.11.2)
    ///
    /// ```text
    /// ψ(z) ≈ ln z − 1/(2z) − Σ_{k≥1} B₂ₖ / (2k · z^{2k})
    /// ```
    ///
    /// with the Bernoulli numbers `B₂ₖ` as exact rationals. The argument is first
    /// shifted upward with the recurrence `ψ(x) = ψ(x+m) − Σ_{j=0}^{m−1} 1/(x+j)`
    /// so that `z = x+m ≳ precision/4` makes the asymptotic tail fall below the
    /// target. For `x < ½` the reflection `ψ(1−x) − ψ(x) = π·cot(πx)` (DLMF 5.5.4)
    /// maps the argument into the convergent region.
    ///
    /// # Domain
    /// Defined for all real `x` except the poles `x = 0, −1, −2, …`, which return
    /// NaN. `ψ(+∞) = +∞`; `ψ(−∞)` and NaN return NaN.
    pub fn digamma(&self, precision: u64, mode: RoundingMode) -> Float {
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(false) => Float::infinity(precision),
            Repr::Inf(true) => Float::nan(precision),
            Repr::Zero(_) => Float::nan(precision),
            Repr::Normal { .. } => {
                if is_nonpos_int(self) {
                    return Float::nan(precision);
                }
                let x = self.clone();
                Float::ziv(precision, mode, move |w| digamma_at(&x.round(w, NEAR), w))
            }
        }
    }

    /// Returns the polygamma function `ψ⁽ⁿ⁾(x)`, the `n`-th derivative of the
    /// digamma function, for a real argument, correctly rounded. `n = 0` is the
    /// digamma function itself.
    ///
    /// # Algorithm
    /// For `n ≥ 1` the asymptotic series (DLMF 5.15.8)
    ///
    /// ```text
    /// ψ⁽ⁿ⁾(z) ≈ (−1)^{n−1} [ (n−1)!/zⁿ + n!/(2 z^{n+1})
    ///                       + Σ_{k≥1} B₂ₖ (2k+n−1)!/(2k)! / z^{2k+n} ]
    /// ```
    ///
    /// with an upward argument shift `ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+m) − (−1)ⁿ n! Σ_{j<m}
    /// 1/(x+j)^{n+1}` (DLMF 5.15.5) pushing `z = x+m` large. Both pieces carry the
    /// sign `(−1)^{n−1}`, so no cancellation occurs.
    ///
    /// # Domain
    /// Defined for all real `x` except the poles `x = 0, −1, −2, …`, which return
    /// NaN. For `n ≥ 1`, `ψ⁽ⁿ⁾(+∞) = 0`; `ψ⁽ⁿ⁾(−∞)` and NaN return NaN.
    pub fn polygamma(&self, n: u64, precision: u64, mode: RoundingMode) -> Float {
        if n == 0 {
            return self.digamma(precision, mode);
        }
        match &self.repr {
            Repr::NaN => Float::nan(precision),
            Repr::Inf(false) => Float::zero(precision),
            Repr::Inf(true) => Float::nan(precision),
            Repr::Zero(_) => Float::nan(precision),
            Repr::Normal { .. } => {
                if is_nonpos_int(self) {
                    return Float::nan(precision);
                }
                let x = self.clone();
                Float::ziv(precision, mode, move |w| {
                    polygamma_at(n, &x.round(w, NEAR), w)
                })
            }
        }
    }

    /// Returns the beta function `B(a, b) = Γ(a)Γ(b)/Γ(a+b)`, correctly rounded.
    ///
    /// # Algorithm
    /// Computed as `sign · exp(ln|Γ(a)| + ln|Γ(b)| − ln|Γ(a+b)|)` so intermediate
    /// gamma values never overflow, with the sign recovered from the three signs
    /// of `Γ` (via reflection for negative arguments; DLMF 5.5.3).
    ///
    /// # Domain
    /// Defined for all real `a`, `b` where neither `a` nor `b` is a non-positive
    /// integer (a pole of `Γ` in the numerator), which return NaN; if only `a+b`
    /// is a non-positive integer the result is `0` (the denominator pole). NaN or
    /// infinite arguments return NaN.
    pub fn beta(a: &Float, b: &Float, precision: u64, mode: RoundingMode) -> Float {
        if !a.is_finite() || !b.is_finite() {
            return Float::nan(precision);
        }
        if is_nonpos_int(a) || is_nonpos_int(b) {
            return Float::nan(precision);
        }
        // Denominator pole `Γ(a+b)` with a finite numerator ⇒ `B = 0` exactly.
        if let (Some(ra), Some(rb)) = (a.to_rational(), b.to_rational()) {
            let sum = ra.add(&rb);
            if sum.denominator() == &Int::ONE && !sum.numerator().is_positive() {
                return Float::zero(precision);
            }
        }
        let a = a.clone();
        let b = b.clone();
        Float::ziv(precision, mode, move |w| {
            beta_at(&a.round(w, NEAR), &b.round(w, NEAR), w)
        })
    }

    /// Returns the Bessel function of the second kind `Yₙ(self)` for integer order
    /// `n` and real argument `x > 0`, correctly rounded.
    ///
    /// # Algorithm
    /// The ascending series (DLMF 10.8.1)
    ///
    /// ```text
    /// Yₙ(x) = (2/π) ln(x/2) Jₙ(x)
    ///         − (1/π) Σ_{k=0}^{n−1} (n−k−1)!/k! (x/2)^{2k−n}
    ///         − (1/π) Σ_{k≥0} (−1)ᵏ (ψ(k+1)+ψ(n+k+1)) / (k!(n+k)!) (x/2)^{2k+n}
    /// ```
    ///
    /// with `ψ(k+1) = −γ + Hₖ` (Euler–Mascheroni constant, harmonic numbers). The
    /// second series is alternating and its partial sums reach `≈ e^{|x|}`, so
    /// about `1.443·|x|` guard bits are added for the cancellation. `Y₋ₙ =
    /// (−1)ⁿ Yₙ`.
    ///
    /// # Domain
    /// Any integer `n` and real `x > 0`. Correctly rounded for moderate `x` (the
    /// guard budget grows linearly with `x`, so very large `x` is progressively
    /// more expensive — in practice good to `x` of a few hundred at reasonable
    /// precision). `Yₙ(0) = −∞`; `x < 0` and non-finite `x` return NaN.
    pub fn bessel_y(&self, n: i64, precision: u64, mode: RoundingMode) -> Float {
        if !self.is_finite() {
            return Float::nan(precision);
        }
        if self.is_zero() {
            return Float::neg_infinity(precision);
        }
        if self.is_sign_negative() {
            return Float::nan(precision);
        }
        let order = n.unsigned_abs();
        let flip = n < 0 && order % 2 == 1;
        let x = self.clone();
        let res = Float::ziv(precision, mode, move |w| {
            bessel_y_at(order, &x.round(w, NEAR), w)
        });
        if flip { res.neg() } else { res }
    }

    /// Returns the modified Bessel function of the second kind `Kₙ(self)` for
    /// integer order `n` and real argument `x > 0`, correctly rounded.
    ///
    /// # Algorithm
    /// The ascending series (DLMF 10.31.1)
    ///
    /// ```text
    /// Kₙ(x) = ½ (x/2)^{−n} Σ_{k=0}^{n−1} (n−k−1)!/k! (−x²/4)ᵏ
    ///         + (−1)^{n+1} ln(x/2) Iₙ(x)
    ///         + (−1)ⁿ ½ (x/2)ⁿ Σ_{k≥0} (ψ(k+1)+ψ(n+k+1)) / (k!(n+k)!) (x²/4)ᵏ
    /// ```
    ///
    /// with `ψ(k+1) = −γ + Hₖ`. The `ln(x/2) Iₙ(x)` term grows like `e^{x}` while
    /// `Kₙ(x) ~ e^{−x}`, so the pieces cancel to `≈ 2.885·x` bits; that many guard
    /// bits are added. `K₋ₙ = Kₙ`.
    ///
    /// # Domain
    /// Any integer `n` and real `x > 0`. Correctly rounded for moderate `x` (the
    /// guard budget grows linearly with `x` — in practice good to `x` of about a
    /// hundred at reasonable precision, less for very large `x`). `Kₙ(0) = +∞`;
    /// `x < 0` and non-finite `x` return NaN.
    pub fn bessel_k(&self, n: i64, precision: u64, mode: RoundingMode) -> Float {
        if !self.is_finite() {
            return Float::nan(precision);
        }
        if self.is_zero() {
            return Float::infinity(precision);
        }
        if self.is_sign_negative() {
            return Float::nan(precision);
        }
        let order = n.unsigned_abs();
        let x = self.clone();
        Float::ziv(precision, mode, move |w| {
            bessel_k_at(order, &x.round(w, NEAR), w)
        })
    }

    /// Sign as `i64` (`-1`/`0`/`1`), for internal use.
    fn signum_i(&self) -> i64 {
        match self.sign() {
            Sign::Negative => -1,
            Sign::Zero => 0,
            Sign::Positive => 1,
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

/// The working scale for an odd Taylor sum on `0 ≤ x < 1` at precision `w`, or
/// `None` when the series collapses to its first term. The scale is anchored
/// to `x`'s magnitude so a small argument keeps `w`-bit *relative* precision
/// (a fixed absolute scale would silently hand Ziv an inaccurate value).
fn odd_series_scale(x: &Float, w: u64) -> Option<u64> {
    let e = match &x.repr {
        Repr::Normal { sig, exp, .. } => exp + sig.bit_len() as i64 - 1, // ⌊log2 x⌋
        _ => return None,                                                // ±0: the sum is x itself
    };
    // The next term is below x·2^(2e); once that clears the guarded target
    // precision, x itself is the correctly-rounded-enough sum.
    if e <= -(w as i64) / 2 - 28 {
        return None;
    }
    Some(w + 32 + (-e).max(0) as u64)
}

/// `atanh(x) = x + x³/3 + x⁵/5 + …` at working precision `w` (needs
/// `0 ≤ x < 1`; the `ln` reduction supplies `x ≤ 1/3`). The sum runs in scaled
/// integer arithmetic: per term one small multiply plus one single-limb
/// division instead of full Float operations.
fn atanh_series(x: &Float, w: u64) -> Float {
    debug_assert!(!x.is_sign_negative(), "atanh series needs x >= 0");
    let Some(n) = odd_series_scale(x, w) else {
        return x.round(w, NEAR);
    };
    let xs = scaled_int(x, n as i64).magnitude();
    let x2 = xs.square().shr(n);
    let mut pow = xs.clone();
    let mut sum = xs;
    let mut k = 1u64;
    loop {
        pow = pow.mul(&x2).shr(n);
        let term = pow.div_rem(&Nat::from_u64(2 * k + 1)).expect("odd > 0").0;
        if term.is_zero() {
            break;
        }
        sum = sum.add(&term);
        k += 1;
    }
    Float::round_raw(false, sum, -(n as i64), w, NEAR).0
}

/// `atan(x) = x − x³/3 + x⁵/5 − …` at working precision `w` (needs
/// `0 ≤ x ≤ 1/4`, supplied by the halving reduction), in scaled integer
/// arithmetic like [`atanh_series`].
fn atan_series(x: &Float, w: u64) -> Float {
    debug_assert!(!x.is_sign_negative(), "atan series needs x >= 0");
    let Some(n) = odd_series_scale(x, w) else {
        return x.round(w, NEAR);
    };
    let xs = scaled_int(x, n as i64).magnitude();
    let x2 = xs.square().shr(n);
    let mut pow = xs.clone();
    let mut sum = xs;
    let mut k = 1u64;
    let mut sub = true;
    loop {
        pow = pow.mul(&x2).shr(n);
        let term = pow.div_rem(&Nat::from_u64(2 * k + 1)).expect("odd > 0").0;
        if term.is_zero() {
            break;
        }
        sum = if sub {
            sum.checked_sub(&term)
                .expect("alternating partial sums stay non-negative")
        } else {
            sum.add(&term)
        };
        sub = !sub;
        k += 1;
    }
    Float::round_raw(false, sum, -(n as i64), w, NEAR).0
}

/// `⌊atan(1/q)·2^n⌋` (within a couple of ulps) via the Taylor series
/// `atan(1/q) = (1/q)·Σ (−1)^k/((2k+1)·q^2k)`, evaluated exactly by binary
/// splitting and finished with one scaled division. Requires `q² ≤ u64::MAX`.
///
/// The single truncated division loses < 1 ulp and the tail of the truncated
/// alternating series is below 1 ulp, well under the callers' guard bits.
fn atan_recip_scaled(q: u64, n: u64) -> Nat {
    let q2 = q * q;
    // Terms shrink by q² ≥ 2^l per step, so n/l + 2 of them bound the tail
    // below 2^-n (floor(log2 q²) only makes the count conservative).
    let l = 63 - q2.leading_zeros() as u64;
    let k = n / l + 2;
    let (num, o, p) = split_atan_sum(0, k, q2, true);
    // Σ = N·q²/(P·O), so atan(1/q) = Σ/q = N·q/(P·O).
    debug_assert!(num.is_positive(), "atan series sum must stay positive");
    num.magnitude()
        .mul(&Nat::from_u64(q))
        .shl(n)
        .div_rem(&p.mul(&o))
        .expect("denominator > 0")
        .0
}

/// π via Machin's formula, `16·atan(1/5) − 4·atan(1/239)`, at precision `w`,
/// evaluated in scaled integer arithmetic.
/// Rounds a precomputed constant `sig / 2^CONST_BITS` to `w` bits, using only the
/// top (significant) limbs. `sig` is little-endian, so its high limbs are last.
fn from_const(sig: &[u64], w: u64) -> Float {
    let keep = (((w + 16) / 64 + 2) as usize).min(sig.len());
    let drop = sig.len() - keep;
    let mag = Nat::from_limbs(&sig[drop..]);
    let exp = drop as i64 * 64 - crate::float_consts::CONST_BITS as i64; // value ≈ mag·2^exp
    Float::round_raw(false, mag, exp, w, NEAR).0
}

/// `ln 2` at `w` bits from the embedded constant, or `None` beyond its length.
/// Exposed for the multi-prime `exp` fast path.
pub(crate) fn ln2_embedded(w: u64) -> Option<Float> {
    (w + 16 <= crate::float_consts::CONST_BITS)
        .then(|| from_const(&crate::float_consts::LN2_SIG, w))
}

/// Rounds `num·2^exp2 / den` (`num ≥ 0`, `den > 0`) to `w` bits, by a single
/// truncating division plus `round_raw` — no rational gcd normalization. Exposed
/// for the multi-prime `exp` fast path's inner rounding.
pub(crate) fn round_ratio(num: &Int, den: &Int, exp2: i64, w: u64, mode: RoundingMode) -> Float {
    let g = w as i64 + 32; // guard bits below the rounding position
    let q = num.mul_2k(g as u32).div_floor(den).magnitude();
    Float::round_raw(false, q, exp2 - g, w, mode).0
}

/// Rounds `sig / 2^total_bits` to `w` bits via `round_raw` (fast — no rational
/// normalization). Exposed for the multi-prime `exp` fast path's embedded logs.
pub(crate) fn round_const_bits(sig: &[u64], total_bits: u64, w: u64) -> Float {
    let keep = (((w + 48) / 64 + 2) as usize).min(sig.len());
    let drop = sig.len() - keep;
    let mag = Nat::from_limbs(&sig[drop..]);
    Float::round_raw(false, mag, drop as i64 * 64 - total_bits as i64, w, NEAR).0
}

/// Euler–Mascheroni constant γ via the Brent–McMillan formula
/// `γ = A(N)/B(N) − ln N`, where `B = Σ (Nᵏ/k!)²` and `A = Σ (Nᵏ/k!)²·Hₖ` with
/// `Hₖ` the k-th harmonic number. The truncation error is `O(e^{-4N})`, so
/// `N = ⌈0.18·n⌉` (with `4N > n·ln2`) drives it below `2⁻ⁿ`.
fn gamma_at(w: u64) -> Float {
    let n = w + 32;
    let bign = (n as i64) * 185 / 1024 + 8; // ≈ 0.18·n, so 4N > n·ln2
    let scale = Int::ONE.mul_2k(n as u32); // 2ⁿ
    let n2 = Int::from_i64(bign * bign);
    let mut t = scale.clone(); // T₀ = (N⁰/0!)²·2ⁿ = 2ⁿ
    let mut b = t.clone(); // B·2ⁿ
    let mut hs = Int::ZERO; // Hₖ·2ⁿ (H₀ = 0)
    let mut a = Int::ZERO; // A·2ⁿ
    let mut k = 1i64;
    loop {
        // Tₖ = Tₖ₋₁·N²/k², Hₖ = Hₖ₋₁ + 1/k.
        t = t.mul(&n2).div_trunc(&Int::from_i64(k * k));
        if t.is_zero() {
            break;
        }
        hs = hs.add(&scale.div_trunc(&Int::from_i64(k)));
        b = b.add(&t);
        // A·2ⁿ += Tₖ·Hₖ = (Tₖ·2ⁿ)·(Hₖ·2ⁿ)/2ⁿ / 2ⁿ.
        a = a.add(&t.mul(&hs).div_2k_trunc(n as u32));
        k += 1;
    }
    let af = Float::round_raw(false, a.magnitude(), -(n as i64), n, NEAR).0;
    let bf = Float::round_raw(false, b.magnitude(), -(n as i64), n, NEAR).0;
    let lnn = Float::from_int(&Int::from_i64(bign), n, NEAR).ln(n, NEAR);
    af.div(&bf, n, NEAR).sub(&lnn, w, NEAR)
}

/// Catalan's constant `G = (π/8)·ln(2+√3) + (3/8)·Σ_{k≥0} 1/((2k+1)²·C(2k,k))`.
/// The sum converges geometrically (`C(2k,k) ~ 4ᵏ`), so `~n/2` terms suffice.
fn catalan_at(w: u64) -> Float {
    let n = w + 32;
    let mut term = Int::ONE.mul_2k(n as u32); // k=0: 1·2ⁿ
    let mut sum = term.clone();
    let mut k = 1i64;
    loop {
        // termₖ = termₖ₋₁·(2k−1)·k / (2·(2k+1)²).
        let num = Int::from_i64((2 * k - 1) * k);
        let den = Int::from_i64(2 * (2 * k + 1) * (2 * k + 1));
        term = term.mul(&num).div_trunc(&den);
        if term.is_zero() {
            break;
        }
        sum = sum.add(&term);
        k += 1;
    }
    let s = Float::round_raw(false, sum.magnitude(), -(n as i64), n, NEAR).0;
    let sqrt3 = Float::from_int(&Int::from_i64(3), n, NEAR).sqrt(n, NEAR);
    let ln_term = Float::from_int(&Int::from_i64(2), n, NEAR)
        .add(&sqrt3, n, NEAR)
        .ln(n, NEAR);
    let eight = Float::from_int(&Int::from_i64(8), n, NEAR);
    let term1 = pi_at(n).mul(&ln_term, n, NEAR).div(&eight, n, NEAR);
    let term2 = s
        .mul(&Float::from_int(&Int::from_i64(3), n, NEAR), n, NEAR)
        .div(&eight, n, NEAR);
    term1.add(&term2, w, NEAR)
}

fn pi_at(w: u64) -> Float {
    if w + 16 <= crate::float_consts::CONST_BITS {
        return from_const(&crate::float_consts::PI_SIG, w);
    }
    let n = w + 32; // guard bits over the series' truncation error
    let a1 = atan_recip_scaled(5, n);
    let a2 = atan_recip_scaled(239, n);
    let pi_scaled = a1
        .shl(4)
        .checked_sub(&a2.shl(2))
        .expect("16·atan(1/5) > 4·atan(1/239)");
    Float::round_raw(false, pi_scaled, -(n as i64), w, NEAR).0
}

/// `⌊log₂|x|⌋` for a finite non-zero [`Float`]; `i64::MIN` for zero/NaN/∞.
fn floor_log2(x: &Float) -> i64 {
    match &x.repr {
        Repr::Normal { sig, exp, .. } => exp + sig.bit_len() as i64 - 1,
        _ => i64::MIN,
    }
}

/// `⌊log₂ w⌋` for `w ≥ 1` (guard-bit budget grows with the iteration count).
fn ilog2(w: u64) -> u64 {
    63 - w.max(1).leading_zeros() as u64
}

/// Arithmetic–geometric mean `M(a, b)` at working precision `w`, for `a, b > 0`.
///
/// Iterates `(a, b) ← ((a+b)/2, √(ab))`; the two sequences converge
/// quadratically to a common limit, so `~log₂ w` steps (one multiply + one
/// square root each) suffice. Stops once the pair agrees to `w` bits.
fn agm(a: &Float, b: &Float, w: u64) -> Float {
    let mut a = a.round(w, NEAR);
    let mut b = b.round(w, NEAR);
    // The two sequences converge quadratically; the initial slow phase (when b
    // starts tiny, as in `ln_agm_at`) adds a further ~log₂ w linear steps, so
    // this cap is never binding in practice. It only guards against a 1-ulp
    // rounding limit cycle near the fixed point, where the values agree to
    // working precision but never become bit-identical.
    let max_iters = 2 * ilog2(w) + 16;
    for _ in 0..max_iters {
        let a1 = a.add(&b, w, NEAR).scale_pow2(-1);
        let b1 = a.mul(&b, w, NEAR).sqrt(w, NEAR);
        let d = a1.sub(&b1, w, NEAR);
        a = a1;
        b = b1;
        // Once |a1 − b1| ≤ 2^(⌊log₂ a1⌋ − w) the two agree to w bits.
        if d.is_zero() || floor_log2(&d) <= floor_log2(&a) - w as i64 {
            return a;
        }
    }
    a.add(&b, w, NEAR).scale_pow2(-1)
}

/// `true` when `x` is within `2^-32` of 1, where `ln(x)` is so small that the
/// AGM formula's fixed *absolute* accuracy cannot deliver `w`-bit *relative*
/// precision (the `ln(s) − m·ln2` subtraction cancels). The `atanh` series is
/// both correct and fast there (its argument is tiny, so it needs few terms),
/// so `ln_at` routes those inputs to the series regardless of precision.
fn ln_near_one(x: &Float) -> bool {
    let d = x.sub(&iflt(1, 64), 64, NEAR);
    d.is_zero() || floor_log2(&d) < -32
}

/// `ln(x)` for finite `x > 0` (and not within `2^-32` of 1) at precision `w`,
/// via the AGM.
///
/// With `s = x·2^m` scaled so `s > 2^{n/2}` (`n` the guarded working precision),
/// Brent's formula gives `ln(s) = π / (2·M(1, 4/s))` to `n` bits, so
/// `ln(x) = ln(s) − m·ln2`. Costs one AGM (`O(M(n)·log n)`) plus a π and a ln2.
///
/// The subtraction cancels `~log₂ n` leading bits (both terms are `~n/2` in
/// magnitude); the wide guard `n − w` absorbs that plus the per-iteration
/// rounding, and `ln_near_one` fences off the inputs where the result itself is
/// too small for any fixed guard.
fn ln_agm_at(x: &Float, w: u64) -> Float {
    let n = w + 64 + 4 * ilog2(w);
    // Scale x up so s ≈ 2^{n/2 + margin}: 4/s is then below 2^{-n/2}, which
    // bounds the formula's O((4/s)²·ln s) error well under 2^{-n}.
    let e = floor_log2(x);
    let target = n as i64 / 2 + 16 + ilog2(n) as i64;
    let m = target - e;
    let s = x.scale_pow2(m); // exact
    let four_over_s = iflt(4, n).div(&s, n, NEAR);
    let agm = agm(&iflt(1, n), &four_over_s, n);
    let ln_s = pi_at(n).div(&agm.scale_pow2(1), n, NEAR); // π/(2·M)
    let m_ln2 = iflt(m, n).mul(&ln2_at(n), n, NEAR);
    ln_s.sub(&m_ln2, n, NEAR).round(w, NEAR)
}

/// Binary-splitting node for `Σ_{k=a}^{b-1} σ^k / ((2k+1)·m^k)` (`σ = −1` when
/// `alternating`, else `+1`): returns `(N, O, P)` with the partial sum equal to
/// `N / (m^(b−1)·O)`, where `O = Π (2k+1)` over the range and `P = m^(b−a)`.
///
/// Splitting the range keeps every multiplication balanced, so the whole sum
/// costs `O(M(n)·log n)` instead of the `O(n²/64)` of term-by-term division.
fn split_atan_sum(a: u64, b: u64, m: u64, alternating: bool) -> (Int, Nat, Nat) {
    // Leaf: fold up to 4 terms in machine words. With `m < 2^16` and
    // `2k+1 < 2^24` every intermediate fits: `|N| ≤ 4·(m·(2b+1))³ < 2^123`,
    // `O < 2^96`, `P ≤ 2^64`.
    if b - a <= 4 && m < (1 << 16) && 2 * b + 1 < (1 << 24) {
        let mut n: i128 = 0;
        let mut o: u128 = 1;
        let mut p: u128 = 1;
        for k in a..b {
            let odd = 2 * k + 1;
            // Append term k: N ← N·m·(2k+1) + σ^k·O, O ← O·(2k+1), P ← P·m.
            let t = if alternating && k & 1 == 1 {
                -(o as i128)
            } else {
                o as i128
            };
            n = n * m as i128 * odd as i128 + t;
            o *= odd as u128;
            p *= m as u128;
        }
        return (Int::from_i128(n), Nat::from_u128(o), Nat::from_u128(p));
    }
    if b - a == 1 {
        // Fallback single-term leaf for parameters beyond the machine-word
        // bounds (astronomical precisions).
        let n = if alternating && a & 1 == 1 {
            Int::MINUS_ONE
        } else {
            Int::ONE
        };
        return (n, Nat::from_u64(2 * a + 1), Nat::from_u64(m));
    }
    let c = a + (b - a) / 2;
    let (n1, o1, p1) = split_atan_sum(a, c, m, alternating);
    let (n2, o2, p2) = split_atan_sum(c, b, m, alternating);
    // N(a,b)/ (m^(b−1)·O1·O2) = N1/(m^(c−1)·O1) + N2/(m^(b−1)·O2).
    let n = n1
        .mul(&Int::from(p2.mul(&o2)))
        .add(&n2.mul(&Int::from(o1.clone())));
    (n, o1.mul(&o2), p1.mul(&p2))
}

/// ln 2 via `2·atanh(1/3) = (2/3)·Σ 1/((2k+1)·9^k)` at precision `w`, evaluated
/// exactly by binary splitting and finished with one scaled division.
fn ln2_at(w: u64) -> Float {
    if w + 16 <= crate::float_consts::CONST_BITS {
        return from_const(&crate::float_consts::LN2_SIG, w);
    }
    let n = w + 32;
    // Each term shrinks by 9 > 2^3, so n/3 + 2 terms push the tail below 2^-n.
    let k = n / 3 + 2;
    let (num, o, p) = split_atan_sum(0, k, 9, false);
    // Σ = N·9/(P·O), so ln 2 = (2/3)·Σ = 6·N/(P·O).
    let sum = num
        .magnitude()
        .mul(&Nat::from_u64(6))
        .shl(n)
        .div_rem(&p.mul(&o))
        .expect("denominator > 0")
        .0;
    Float::round_raw(false, sum, -(n as i64), w, NEAR).0
}

/// `e^x` at precision `w` via range reduction `x = k·ln2 + r` and a Taylor sum.
///
/// A second reduction stage `exp(r) = exp(r/2^j)^(2^j)` with `j ≈ √w` balances
/// series length against squaring count: each halving of `r` removes ~1 term
/// per remaining `w/j` bits, so the term count drops from `O(w/log w)` to
/// `O(√w)` at the cost of `√w` cheap squarings.
fn exp_at(x: &Float, w: u64) -> Float {
    // The squarings amplify rounding error by ~2^j ulps, so work at w + j + 8
    // bits; the returned extra bits keep Ziv's ambiguity check sound.
    let j = w.isqrt().max(1);
    let n = w + j + 8;
    let ln2 = ln2_at(n);
    let k = x.div(&ln2, n, NEAR).round_half_up_to_int();
    let ki = k.to_i64().unwrap_or(0);
    let r = x.sub(&Float::from_int(&k, n, NEAR).mul(&ln2, n, NEAR), n, NEAR);
    let r = r.scale_pow2(-(j as i64)); // exact
    // The Taylor sum runs in scaled integer arithmetic (everything is an
    // integer multiple of 2^-n): per-term cost is one small multiplication and
    // one single-limb division instead of full Float operations. Truncations
    // lose < 1 ulp each, well inside the guard bits above.
    // R = ⌊r·2^n⌋; |r| < ln2/2^(j+1), so |R| < 2^(n−j−1).
    let rs = scaled_int(&r, n as i64);
    // exp(R/2^n) = Σ (R/2^n)^k/k!, all terms scaled by 2^n.
    let mut sum = Int::ONE.mul_2k(n as u32);
    let mut term = sum.clone();
    let mut kk: i64 = 1;
    loop {
        term = term
            .mul(&rs)
            .div_2k_trunc(n as u32)
            .div_trunc(&Int::from_i64(kk));
        if term.is_zero() {
            break;
        }
        sum = sum.add(&term);
        kk += 1;
    }
    // Undo the argument halvings: exp(r) = exp(r/2^j)^(2^j). The sum stays
    // near 2^n (r is tiny), so each squaring is one n-bit multiply.
    for _ in 0..j {
        sum = sum.square().div_2k_trunc(n as u32);
    }
    debug_assert!(sum.is_positive(), "exp series sum must stay positive");
    Float::round_raw(false, sum.magnitude(), -(n as i64), n, NEAR)
        .0
        .scale_pow2(ki)
}

/// `⌊x·2^n⌋` (truncated toward zero) as a signed integer.
fn scaled_int(x: &Float, n: i64) -> Int {
    match &x.repr {
        Repr::Normal { neg, sig, exp } => {
            let e = exp + n;
            let mag = if e >= 0 {
                sig.shl(e as u64)
            } else {
                sig.shr((-e) as u64)
            };
            let v = Int::from(mag);
            if *neg { v.neg() } else { v }
        }
        _ => Int::ZERO,
    }
}

/// `ln(x)` for finite `x > 0` at precision `w`.
fn ln_at(x: &Float, w: u64) -> Float {
    if x.is_zero() {
        return Float::neg_infinity(w);
    }
    if w >= LN_AGM_THRESHOLD && !ln_near_one(x) {
        return ln_agm_at(x, w);
    }
    ln_series_at(x, w)
}

/// `ln(x)` for finite non-zero `x > 0` at precision `w`, via argument reduction
/// to `m ∈ [1, 2)` and the `atanh` series `ln(m) = 2·atanh((m−1)/(m+1))`.
fn ln_series_at(x: &Float, w: u64) -> Float {
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
    let q = x.div(&half_pi, w, NEAR).round_half_up_to_int();
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

/// Working precisions at or above this use the rectangular (baby-step/giant-step)
/// series [`sin_cos_series_rect`]; below it the term-by-term [`sin_cos_series_simple`]
/// wins (its per-term bookkeeping is cheaper than the rectangular block setup).
/// Chosen from the crossover measured by the `bench_sin_cos_rect` harness
/// (≈1.0× at 1024 bits, growing to ~9× at 64k).
const SIN_COS_RECT_THRESHOLD: u64 = 1024;

/// `(sin r, cos r)` for `|r| ≤ π/4` at precision `w`. Dispatches to the O(√T)
/// rectangular series at high precision and the O(T) term-by-term series below
/// the crossover; both are correctly-rounded through the same Ziv wrapper.
fn sin_cos_series(r: &Float, w: u64) -> (Float, Float) {
    if w >= SIN_COS_RECT_THRESHOLD {
        sin_cos_series_rect(r, w)
    } else {
        sin_cos_series_simple(r, w)
    }
}

/// `(sin r, cos r)` by Taylor series for small `|r| ≤ π/4`, at precision `w`,
/// in scaled integer arithmetic (see [`atanh_series`]). Handles the sign of
/// `r` explicitly: both sums run on `z = r² ≥ 0`, whose alternating partial
/// sums stay non-negative.
fn sin_cos_series_simple(r: &Float, w: u64) -> (Float, Float) {
    if r.is_zero() {
        // Exact: sin(±0) = ±0, cos(0) = 1.
        return (r.round(w, NEAR), iflt(1, w));
    }
    let n = w + 32;
    // Both sums run on z = r² at absolute scale 2^-n; they stay in [0.7, 1],
    // so absolute precision is relative precision. The final sin = r·B keeps
    // r's full relative precision (and its sign) through one Float multiply.
    // A tiny r collapses z to zero and the sums to exactly 1: Ziv's boundary
    // test then either rounds correctly (Nearest) or retries (directed modes).
    let z = scaled_int(r, n as i64).magnitude().square().shr(n);
    // cos = Σ (−1)^m z^m/(2m)! : term ← term·z/((2m−1)(2m)).
    let mut term = Nat::one().shl(n);
    let mut cos = term.clone();
    let mut m = 1u64;
    let mut sub = true;
    loop {
        term = term
            .mul(&z)
            .shr(n)
            .div_rem(&Nat::from_u64((2 * m - 1) * (2 * m)))
            .expect("> 0")
            .0;
        if term.is_zero() {
            break;
        }
        cos = if sub {
            cos.checked_sub(&term)
                .expect("alternating partial sums stay non-negative")
        } else {
            cos.add(&term)
        };
        sub = !sub;
        m += 1;
    }
    // sin = r·B(z), B = Σ (−1)^k z^k/(2k+1)! : term ← term·z/((2k)(2k+1)).
    let mut term = Nat::one().shl(n);
    let mut bracket = term.clone();
    let mut k = 1u64;
    let mut sub = true;
    loop {
        term = term
            .mul(&z)
            .shr(n)
            .div_rem(&Nat::from_u64((2 * k) * (2 * k + 1)))
            .expect("> 0")
            .0;
        if term.is_zero() {
            break;
        }
        bracket = if sub {
            bracket
                .checked_sub(&term)
                .expect("alternating partial sums stay non-negative")
        } else {
            bracket.add(&term)
        };
        sub = !sub;
        k += 1;
    }
    let bracket_f = Float::round_raw(false, bracket, -(n as i64), w, NEAR).0;
    (
        r.mul(&bracket_f, w, NEAR),
        Float::round_raw(false, cos, -(n as i64), w, NEAR).0,
    )
}

/// `(sin r, cos r)` for `|r| ≤ π/4` via **rectangular series splitting**
/// (Brent & Zimmermann, *MCA* §4.4.3 / Paterson–Stockmeyer).
///
/// Both `cos = Σ (−1)^m z^m/(2m)!` and `sin/r = Σ (−1)^k z^k/(2k+1)!` (with
/// `z = r²`) have the shape `S = Σ_{m≥0} (−1)^m z^m / ∏_{j=1}^m d(j)` for a
/// small `d(·)`. Splitting the `T`-term sum into `⌈T/C⌉` blocks of width
/// `C ≈ √(2T)` evaluates it with `O(√T)` full-width `z`-multiplies (the `C`
/// baby-step powers `z^j` plus two per block) instead of one per term; the
/// remaining work is small `Nat` products/divides by the block coefficients.
/// Same working scale `2^-n` (`n = w + 32`) and Ziv wrapper as
/// [`sin_cos_series_simple`], so the correctly-rounded result is identical.
fn sin_cos_series_rect(r: &Float, w: u64) -> (Float, Float) {
    if r.is_zero() {
        return (r.round(w, NEAR), iflt(1, w));
    }
    let n = w + 32;
    let z = scaled_int(r, n as i64).magnitude().square().shr(n);
    if z.is_zero() {
        // r so tiny that z underflows the scale: cos = 1, sin = r exactly.
        return (r.round(w, NEAR), iflt(1, w));
    }
    // Block width C ≈ √(2·#terms), shared by both series (same powers of z).
    let t_est = sin_cos_term_count(&z, n);
    let two_t = 2 * t_est;
    let mut c = two_t.isqrt();
    if c * c < two_t {
        c += 1;
    }
    let c = (c as usize).max(2);
    // Baby steps: powers[j] ≈ z^j · 2^n for j = 0..=C (C full-width multiplies).
    let mut powers = alloc::vec::Vec::with_capacity(c + 1);
    powers.push(Nat::one().shl(n)); // z^0 = 1
    powers.push(z.clone()); // z^1
    for j in 2..=c {
        powers.push(powers[j - 1].mul(&z).shr(n));
    }
    // cos: d(m) = (2m−1)(2m); sin bracket: d(k) = (2k)(2k+1).
    let cos = alt_series_rect(n, c, &powers, |m| (2 * m - 1) * (2 * m));
    let bracket = alt_series_rect(n, c, &powers, |k| (2 * k) * (2 * k + 1));
    let bracket_f = Float::round_raw(false, bracket, -(n as i64), w, NEAR).0;
    (
        r.mul(&bracket_f, w, NEAR),
        Float::round_raw(false, cos, -(n as i64), w, NEAR).0,
    )
}

/// Number of terms until `z^m/(2m)! < 2^-n`, i.e. the series has converged to
/// the working scale. A cheap integer magnitude recurrence (log₂ tracked in
/// half-bit units) — only used to size the block width; the block loop itself
/// terminates exactly when its leading term underflows, so an inexact estimate
/// costs at most a few baby-step multiplies. No-`std`-safe (no `f64` intrinsics).
fn sin_cos_term_count(z: &Nat, n: u64) -> u64 {
    // 2·log₂(z/2^n) ≈ 2·bit_len(z) − 1 − 2n  (negative, since z < 2^n).
    let log2z_x2 = 2 * z.bit_len() as i64 - 1 - 2 * n as i64;
    let mut lg2 = 0i64; // running 2·log₂|term|
    let mut m = 0u64;
    loop {
        m += 1;
        let dm = (2 * m - 1) * (2 * m);
        // 2·log₂(dm) ≈ 2·bit_len(dm) − 1.
        let log2dm_x2 = 2 * (64 - dm.leading_zeros() as i64) - 1;
        lg2 += log2z_x2 - log2dm_x2;
        if lg2 < -2 * n as i64 || m >= n {
            return m;
        }
    }
}

/// Evaluates `S = Σ_{m≥0} (−1)^m z^m / ∏_{j=1}^m d(j)` at scale `2^-n` by
/// rectangular splitting, returning the non-negative scaled sum. `powers[j]`
/// must hold `z^j · 2^n` for `j = 0..=c`, and `z < 1` (`d` grows so the sum
/// converges). The running block-leading term `b = t_{iC}·2^n` is advanced by a
/// single full multiply by `z^C` per block; each block sum is one full multiply
/// `b·W` plus small products/divides by the block's integer coefficients.
fn alt_series_rect(n: u64, c: usize, powers: &[Nat], d: impl Fn(u64) -> u64) -> Nat {
    let y = &powers[c]; // z^C · 2^n
    let mut acc = Int::ZERO; // Σ block sums, scale 2^n
    let mut b_mag = powers[0].clone(); // |t_{iC}|·2^n, starts at t_0 = 1 → 2^n
    let mut b_neg = false;
    let mut base = 0u64; // iC, first term index of the current block
    let cap = 2 * (base_cap(c) + c as u64);
    while !b_mag.is_zero() && base <= cap {
        // Block covers m = base .. base+C−1. Build, from the top down,
        //   W    = Σ_{j=0}^{C−1} (−1)^j · num_j · powers[j]   (signed)
        //   num_j = ∏_{l=base+j+1}^{base+C−1} d(l),   num_{C−1} = 1,  num_0 = Den
        // so that the block sum = b · W / (2^n · Den).
        let mut w_acc = Int::ZERO;
        let mut num = Nat::one();
        for j in (0..c).rev() {
            let contrib = Int::from(num.mul(&powers[j]));
            w_acc = if j % 2 == 0 {
                w_acc.add(&contrib)
            } else {
                w_acc.sub(&contrib)
            };
            if j > 0 {
                num = num.mul(&Nat::from_u64(d(base + j as u64)));
            }
        }
        let den = num; // = ∏_{l=base+1}^{base+C−1} d(l)
        // Block sum magnitude = (|b|·|W| ≫ n) / Den, sign = b_neg ⊕ (W<0).
        let block_mag = b_mag
            .mul(&w_acc.magnitude())
            .shr(n)
            .div_rem(&den)
            .expect("den > 0")
            .0;
        let block_neg = b_neg ^ w_acc.is_negative();
        let block = Int::from(block_mag);
        acc = if block_neg {
            acc.sub(&block)
        } else {
            acc.add(&block)
        };
        // Advance b to t_{(i+1)C}·2^n = b·(−1)^C·z^C / (Den·d(base+C)).
        let den_adv = den.mul(&Nat::from_u64(d(base + c as u64)));
        b_mag = b_mag.mul(y).shr(n).div_rem(&den_adv).expect("> 0").0;
        if c % 2 == 1 {
            b_neg = !b_neg;
        }
        base += c as u64;
    }
    debug_assert!(!acc.is_negative(), "series partial sum stays non-negative");
    acc.magnitude()
}

/// A generous hard cap on how many terms the block loop may span, in case the
/// leading-term underflow that normally halts it is delayed by rounding.
fn base_cap(c: usize) -> u64 {
    4 * c as u64 * c as u64 + 64
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

// Above these `x²` bounds a method switches: the Kummer series stays cheap and
// stable while it must carry ~1.44·x² extra bits (its terms peak near e^{x²}),
// and the continued fraction converges quickly only once x is not small.
/// Use the Kummer series for `erf` while `x² ≤` this; else `1 − erfc` (CF).
const ERF_SERIES_MAX_X2: i64 = 25;
/// Compute `erfc = 1 − erf(series)` while `x² ≤` this; else the CF directly.
const ERFC_SERIES_MAX_X2: i64 = 4;

/// `⌈x²⌉` as an `i64`, saturating to `i64::MAX` when it overflows (only reached
/// for enormous arguments, which every caller routes to the "large" branch).
fn ceil_sq_i64(x: &Float, w: u64) -> i64 {
    x.mul(x, w, NEAR)
        .ceil()
        .and_then(|i| i.to_i64())
        .unwrap_or(i64::MAX)
}

/// `erf(x)` for finite non-zero `x` at working precision `w`. Odd in `x`.
fn erf_at(x: &Float, w: u64) -> Float {
    let neg = x.is_sign_negative();
    let a = x.abs();
    let res = if ceil_sq_i64(&a, w) <= ERF_SERIES_MAX_X2 {
        erf_series(&a, w)
    } else {
        // erf(a) = 1 − erfc(a); erfc is tiny here, so no cancellation.
        iflt(1, w).sub(&erfc_cf(&a, w), w, NEAR)
    };
    if neg { res.neg() } else { res }
}

/// `erfc(x)` for finite non-zero `x` at working precision `w`.
fn erfc_at(x: &Float, w: u64) -> Float {
    let neg = x.is_sign_negative();
    let a = x.abs();
    let core = if ceil_sq_i64(&a, w) <= ERFC_SERIES_MAX_X2 {
        // Small a: erfc(a) = 1 − erf(a); erf(a) is well below 1, mild cancel.
        iflt(1, w).sub(&erf_series(&a, w), w, NEAR)
    } else {
        erfc_cf(&a, w)
    };
    // erfc(−a) = 2 − erfc(a).
    if neg {
        iflt(2, w).sub(&core, w, NEAR)
    } else {
        core
    }
}

/// `erf(a)` for `a ≥ 0` via the all-positive Kummer series
/// `erf(a) = (2/√π) e^{−a²} Σ_{n≥0} 2ⁿ a^{2n+1}/(1·3·5···(2n+1))` (DLMF 7.6.2).
///
/// The sum `S = Σ …` is accumulated in scaled-integer arithmetic (every value an
/// integer multiple of `2^-n`, like [`exp_at`]); its terms peak near `e^{a²}`, so
/// the scale carries `⌈1.44·a²⌉` bits beyond `w` to keep the result accurate.
fn erf_series(a: &Float, w: u64) -> Float {
    let a2c = ceil_sq_i64(a, w).max(0) as u64;
    // 1.4427 ≈ log₂ e, so a2c·185/128 ≥ a²·log₂ e bounds the peak-term growth.
    let n = w + a2c.saturating_mul(185) / 128 + 16;
    // Term ratio tₙ/tₙ₋₁ = 2a²/(2n+1); t₀ = a. Everything scaled by 2ⁿ.
    let as_ = scaled_int(a, n as i64).magnitude();
    let two_a2 = as_.square().shr(n).shl(1); // 2a² · 2ⁿ
    let mut term = as_.clone();
    let mut sum = as_;
    let mut k = 1u64;
    loop {
        term = term
            .mul(&two_a2)
            .shr(n)
            .div_rem(&Nat::from_u64(2 * k + 1))
            .expect("odd > 0")
            .0;
        if term.is_zero() {
            break;
        }
        sum = sum.add(&term);
        k += 1;
    }
    let s = Float::round_raw(false, sum, -(n as i64), n, NEAR).0;
    // erf = (2/√π)·e^{−a²}·S.
    let sqrtpi = pi_at(n).sqrt(n, NEAR);
    let factor = iflt(2, n).div(&sqrtpi, n, NEAR);
    let em = a.mul(a, n, NEAR).neg().exp(n, NEAR);
    factor.mul(&em, n, NEAR).mul(&s, n, NEAR).round(w, NEAR)
}

/// `erfc(a)` for `a > 0` via the continued fraction (DLMF 7.9.3)
/// `√π e^{a²} erfc(a) = 1/(a + ½/(a + 1/(a + 3⁄2/(a + …))))`, evaluated by the
/// modified Lentz algorithm. Converges quickly for the `a > 2` callers.
fn erfc_cf(a: &Float, w: u64) -> Float {
    let n = w + 16;
    let one = iflt(1, n);
    let tiny = one.scale_pow2(-4 * (n as i64)); // stand-in for a zero pivot
    let x = a.round(n, NEAR);
    let stop = one.scale_pow2(-(w as i64) - 4);
    // f = b₀ + a₁/(b₁ + a₂/(b₂ + …)) with b₀ = 0, bⱼ = x, a₁ = 1,
    // aⱼ = (j−1)/2 for j ≥ 2.
    let mut f = tiny.clone();
    let mut c = f.clone();
    let mut d = Float::zero(n);
    let mut j = 1u64;
    let cap = 8 * w + 200;
    loop {
        let aj = if j == 1 {
            one.clone()
        } else {
            rflt((j - 1) as i64, 2, n)
        };
        d = x.add(&aj.mul(&d, n, NEAR), n, NEAR);
        if d.is_zero() {
            d = tiny.clone();
        }
        d = one.div(&d, n, NEAR);
        c = x.add(&aj.div(&c, n, NEAR), n, NEAR);
        if c.is_zero() {
            c = tiny.clone();
        }
        let delta = c.mul(&d, n, NEAR);
        f = f.mul(&delta, n, NEAR);
        let conv = delta.sub(&one, n, NEAR).abs().partial_cmp(&stop) == Some(Ordering::Less);
        j += 1;
        if conv || j > cap {
            break;
        }
    }
    // erfc(a) = e^{−a²}/√π · f.
    let sqrtpi = pi_at(n).sqrt(n, NEAR);
    let em = a.mul(a, n, NEAR).neg().exp(n, NEAR);
    em.div(&sqrtpi, n, NEAR).mul(&f, n, NEAR).round(w, NEAR)
}

/// `ζ(s)` for real `s > 0`, `s ≠ 1`, at working precision `w`.
///
/// Computes the alternating eta function `η(s) = Σ_{k≥0} (−1)ᵏ (k+1)^{−s}` by the
/// Cohen–Rodriguez-Villegas–Zagier / Borwein acceleration (error `≤ (3+√8)^{−N}`
/// for the totally-monotonic terms `(k+1)^{−s}`, `s > 0`), then returns
/// `ζ(s) = η(s)/(1 − 2^{1−s})`.
fn zeta_at(s: &Float, w: u64) -> Float {
    // Terms decay like (3+√8)^{−N}, log₂(3+√8) ≈ 2.543; N·2/5 > (w+16)/2.543.
    let nt = ((w + 16) * 2 / 5 + 4) as i128;
    // Exact d = ((3+√8)ᴺ + (3−√8)ᴺ)/2: Dₖ = 6Dₖ₋₁ − Dₖ₋₂, D₀ = 2, D₁ = 6.
    let mut dprev = Int::from_i64(2);
    let mut dcur = Int::from_i64(6);
    let six = Int::from_i64(6);
    for _ in 2..=nt {
        let next = dcur.mul(&six).sub(&dprev);
        dprev = dcur;
        dcur = next;
    }
    let d_int = dcur.div_trunc(&Int::from_i64(2));
    let df = Float::from_int(&d_int, w, NEAR);

    let neg_s = s.neg();
    let ln2 = ln2_at(w);
    // CVZ Algorithm 1: b := −1, c := −d, s := 0; loop; η ≈ Σ cₖ aₖ / d.
    let mut b = Float::from_int(&Int::MINUS_ONE, w, NEAR);
    let mut c = df.neg();
    let mut sum = Float::zero(w);
    for k in 0..nt {
        c = b.sub(&c, w, NEAR);
        // aₖ = (k+1)^{−s} = exp(−s·ln(k+1)).
        let base = Float::from_int(&Int::from_i128(k + 1), w, NEAR);
        let ak = neg_s.mul(&base.ln(w, NEAR), w, NEAR).exp(w, NEAR);
        sum = sum.add(&c.mul(&ak, w, NEAR), w, NEAR);
        // b ← b·(k+N)(k−N)·2 / ((2k+1)(k+1)).
        let num = (k + nt) * (k - nt) * 2;
        let den = (2 * k + 1) * (k + 1);
        b = b
            .mul(&Float::from_int(&Int::from_i128(num), w, NEAR), w, NEAR)
            .div(&Float::from_int(&Int::from_i128(den), w, NEAR), w, NEAR);
    }
    let eta = sum.div(&df, w, NEAR);
    // ζ(s) = η(s) / (1 − 2^{1−s}); 2^{1−s} = exp((1−s)·ln2).
    let one = iflt(1, w);
    let two_pow = one.sub(s, w, NEAR).mul(&ln2, w, NEAR).exp(w, NEAR);
    let factor = one.sub(&two_pow, w, NEAR);
    eta.div(&factor, w, NEAR)
}

/// `⌊log₂|x|⌋` for a finite non-zero `x`, or `i64::MIN` for ±0/±∞/NaN.
fn float_msb(x: &Float) -> i64 {
    match &x.repr {
        Repr::Normal { sig, exp, .. } => exp + sig.bit_len() as i64 - 1,
        _ => i64::MIN,
    }
}

/// An incrementally-extended table of Bernoulli numbers `B₀, B₁, B₂, …` as exact
/// rationals, built via the standard recurrence
/// `B_m = −1/(m+1) · Σ_{j=0}^{m−1} C(m+1, j) B_j` (clean-room; e.g. MCA §4.7.2).
struct BernoulliTable {
    b: alloc::vec::Vec<Rational>,
}

impl BernoulliTable {
    fn new() -> Self {
        BernoulliTable {
            b: alloc::vec![Rational::from_integer(Int::ONE)], // B₀ = 1
        }
    }

    /// Returns `B_{2k}` (`k ≥ 1`), extending the table as needed.
    fn even(&mut self, k: u64) -> Rational {
        let idx = (2 * k) as usize;
        while self.b.len() <= idx {
            let m = self.b.len() as u64; // index of the value being computed
            if m > 1 && m % 2 == 1 {
                // Odd-index Bernoulli numbers (past B₁) vanish.
                self.b.push(Rational::from_integer(Int::ZERO));
                continue;
            }
            // B_m = −1/(m+1) Σ_{j=0}^{m−1} C(m+1, j) B_j.
            let mut sum = Rational::from_integer(Int::ZERO);
            let mut c = Int::ONE; // C(m+1, 0)
            for j in 0..m as usize {
                sum = sum.add(&self.b[j].mul(&Rational::from_integer(c.clone())));
                // C(m+1, j+1) = C(m+1, j) · (m+1−j) / (j+1).
                c = c
                    .mul(&Int::from_u64(m + 1 - j as u64))
                    .div_trunc(&Int::from_u64(j as u64 + 1));
            }
            let bm = sum.neg().div(&Rational::from_integer(Int::from_u64(m + 1)));
            self.b.push(bm);
        }
        self.b[idx].clone()
    }
}

/// Stirling's asymptotic tail `Σ_{k≥1} B₂ₖ / (2k(2k−1) z^{2k−1})` at working
/// precision `w`, for a large positive `z`. Terms are added until one falls below
/// `2^{-w}` (absolute); the series being asymptotic, accumulation also stops if a
/// term stops shrinking (a safety net — `z ≳ w/4` keeps that from happening
/// before convergence).
fn stirling_tail(z: &Float, w: u64) -> Float {
    let z2 = z.mul(z, w, NEAR);
    let mut zpow = z.clone(); // z^{2k−1}, starts at z¹ (k = 1)
    let mut sum = Float::zero(w);
    let mut table = BernoulliTable::new();
    let mut prev_msb = i64::MAX;
    let mut k = 1u64;
    loop {
        // coeff = B₂ₖ / (2k(2k−1)).
        let denom = Int::from_u64(2 * k).mul(&Int::from_u64(2 * k - 1));
        let coeff = table.even(k).div(&Rational::from_integer(denom));
        let term = Float::from_rational(&coeff, w, NEAR).div(&zpow, w, NEAR);
        let msb = float_msb(&term);
        if msb < -(w as i64) {
            break;
        }
        if msb >= prev_msb {
            // Asymptotic minimum reached before convergence — stop (safety).
            break;
        }
        prev_msb = msb;
        sum = sum.add(&term, w, NEAR);
        zpow = zpow.mul(&z2, w, NEAR); // z^{2k+1}
        k += 1;
    }
    sum
}

/// `ln Γ(x)` for finite `x > 0` at working precision `w`, via Stirling with an
/// upward argument shift `Γ(x) = Γ(z)/∏_{j<m}(x+j)`, `z = x+m ≳ w/4`.
fn ln_gamma_at(x: &Float, w: u64) -> Float {
    // Guard against the cancellation in ln Γ(z) − ln P (both ≈ z ln z) and the
    // O(m) rounding of the shift product; both are bounded by a few ·bitlen(w).
    let bw = 64 - w.leading_zeros() as u64;
    let n = w + 2 * bw + 48;
    // Shift so that z = x + m reaches the numeric threshold Z ≈ n/4.
    let z_thresh = (n / 4 + 8) as i64;
    let fx = x.floor().and_then(|i| i.to_i64()).unwrap_or(z_thresh);
    let m = if fx >= z_thresh { 0 } else { z_thresh - fx };
    let z = x.add(&iflt(m, n), n, NEAR);
    // P = ∏_{j=0}^{m−1} (x + j); ln Γ(x) = ln Γ(z) − ln P.
    let mut p = iflt(1, n);
    for j in 0..m {
        p = p.mul(&x.add(&iflt(j, n), n, NEAR), n, NEAR);
    }
    let ln_p = if m > 0 { p.ln(n, NEAR) } else { Float::zero(n) };
    // ln Γ(z) = (z − ½) ln z − z + ½ ln(2π) + tail.
    let ln_z = z.ln(n, NEAR);
    let half = rflt(1, 2, n);
    let ln_2pi_half = pi_at(n).scale_pow2(1).ln(n, NEAR).scale_pow2(-1);
    let ln_gamma_z = z
        .sub(&half, n, NEAR)
        .mul(&ln_z, n, NEAR)
        .sub(&z, n, NEAR)
        .add(&ln_2pi_half, n, NEAR)
        .add(&stirling_tail(&z, n), n, NEAR);
    ln_gamma_z.sub(&ln_p, n, NEAR).round(w, NEAR)
}

/// `Γ(x)` for finite `x ≥ ½` at working precision `w`, via `exp(ln Γ(x))`.
fn gamma_pos_at(x: &Float, w: u64) -> Float {
    // exp needs its argument to ~2^{-w} absolute; ln Γ(x) can be as large as
    // ≈ x·ln x, so carry log₂|ln Γ(x)| extra bits.
    let extra = float_msb(x).max(0) as u64 + 40;
    let ln_g = ln_gamma_at(x, w + extra);
    ln_g.exp(w, NEAR)
}

/// `Γ(x)` for finite `x` (not a non-positive integer) at working precision `w`.
fn gamma_fn_at(x: &Float, w: u64) -> Float {
    let half = rflt(1, 2, w + 8);
    if x.partial_cmp(&half) == Some(Ordering::Less) {
        // Reflection: Γ(x) = π / (sin(πx) · Γ(1−x)), with 1−x > ½.
        // Extra bits cover the tiny sin(πx) near integer arguments.
        let n = w + 40 + float_msb(x).max(0) as u64;
        let one = iflt(1, n);
        let one_minus_x = one.sub(x, n, NEAR);
        let g1 = gamma_pos_at(&one_minus_x, n);
        let sin_pix = pi_at(n).mul(x, n, NEAR).sin(n, NEAR);
        return pi_at(n)
            .div(&sin_pix.mul(&g1, n, NEAR), n, NEAR)
            .round(w, NEAR);
    }
    gamma_pos_at(x, w)
}

/// `Jₙ(x)` (`alternating`) or `Iₙ(x)` at working precision `w`, for finite
/// non-zero `x` and order `n ≥ 0`. The common factor `(x/2)ⁿ/n!` is pulled out
/// so the accumulated sum stays anchored near its leading term `1`; the sum runs
/// in scaled-integer arithmetic like [`exp_at`].
fn bessel_series_at(n: u64, x: &Float, w: u64, alternating: bool) -> Float {
    // Cancellation guard for the alternating (J) case: partial sums reach
    // ≈ e^{|x|}, i.e. ~1.4427·|x| bits (log₂e ≈ 185/128); harmless for I.
    let ax_floor = x.abs().floor().and_then(|i| i.to_i64()).unwrap_or(i64::MAX);
    let x_guard = (((ax_floor as u128 + 1) * 185 / 128).min(u64::MAX as u128)) as u64;
    let ns = w + x_guard + 64;
    // h2 = (x/2)² scaled by 2^{ns} (always non-negative).
    let half = x.scale_pow2(-1);
    let hs = scaled_int(&half, ns as i64);
    let h2 = hs.square().div_2k_trunc(ns as u32); // (x/2)² · 2^{ns}
    // U = Σ_{m≥0} (∓1)ᵐ cₘ, c₀ = 1, cₘ = cₘ₋₁ · (x/2)² / (m(m+n)), scaled by 2^{ns}.
    let scale = Int::ONE.mul_2k(ns as u32);
    let mut c = scale.clone();
    let mut sum = scale.clone();
    let mut m = 1u64;
    loop {
        // divisor = m·(m+n).
        let divisor = Int::from_u128(m as u128 * (m as u128 + n as u128));
        c = c.mul(&h2).div_2k_trunc(ns as u32).div_trunc(&divisor);
        if c.is_zero() {
            break;
        }
        if alternating && m % 2 == 1 {
            sum = sum.sub(&c);
        } else {
            sum = sum.add(&c);
        }
        m += 1;
    }
    let uf = Float::round_raw(sum.is_negative(), sum.magnitude(), -(ns as i64), ns, NEAR).0;
    // prefactor = (x/2)ⁿ / n!.
    let pref = float_powi(&half, n, ns).div(&factorial_float(n, ns), ns, NEAR);
    uf.mul(&pref, ns, NEAR).round(w, NEAR)
}

/// `base^e` (integer exponent) at working precision `w`, by binary exponentiation.
fn float_powi(base: &Float, e: u64, w: u64) -> Float {
    let mut acc = iflt(1, w);
    let mut b = base.clone();
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            acc = acc.mul(&b, w, NEAR);
        }
        e >>= 1;
        if e > 0 {
            b = b.mul(&b, w, NEAR);
        }
    }
    acc
}

/// `n!` as a [`Float`] at working precision `w`.
fn factorial_float(n: u64, w: u64) -> Float {
    let mut f = Int::ONE;
    let mut k = 2u64;
    while k <= n {
        f = f.mul(&Int::from_u64(k));
        k += 1;
    }
    Float::from_int(&f, w, NEAR)
}

/// `n!` as an exact [`Int`].
fn factorial_int(n: u64) -> Int {
    let mut f = Int::ONE;
    let mut k = 2u64;
    while k <= n {
        f = f.mul(&Int::from_u64(k));
        k += 1;
    }
    f
}

/// The `n`-th harmonic number `Hₙ = Σ_{j=1}^{n} 1/j` as a [`Float`] at precision
/// `w` (`H₀ = 0`).
fn harmonic_float(n: u64, w: u64) -> Float {
    let mut s = Float::zero(w);
    for j in 1..=n {
        s = s.add(&iflt(1, w).div(&iflt(j as i64, w), w, NEAR), w, NEAR);
    }
    s
}

/// True if `x` is a non-positive integer (`0, −1, −2, …`), i.e. a pole of `Γ`.
fn is_nonpos_int(x: &Float) -> bool {
    match x.to_rational() {
        Some(r) => r.denominator() == &Int::ONE && !r.numerator().is_positive(),
        None => false,
    }
}

/// Digamma asymptotic tail `Σ_{k≥1} B₂ₖ / (2k·z^{2k})` at precision `w`, for a
/// large positive `z`; mirrors [`stirling_tail`] (asymptotic, with the same
/// safety net if a term stops shrinking).
fn digamma_tail(z: &Float, w: u64) -> Float {
    let z2 = z.mul(z, w, NEAR);
    let mut zpow = z2.clone(); // z^{2k}, starts at z² (k = 1)
    let mut sum = Float::zero(w);
    let mut table = BernoulliTable::new();
    let mut prev_msb = i64::MAX;
    let mut k = 1u64;
    loop {
        // coeff = B₂ₖ / (2k).
        let coeff = table
            .even(k)
            .div(&Rational::from_integer(Int::from_u64(2 * k)));
        let term = Float::from_rational(&coeff, w, NEAR).div(&zpow, w, NEAR);
        let msb = float_msb(&term);
        if msb < -(w as i64) {
            break;
        }
        if msb >= prev_msb {
            break;
        }
        prev_msb = msb;
        sum = sum.add(&term, w, NEAR);
        zpow = zpow.mul(&z2, w, NEAR); // z^{2k+2}
        k += 1;
    }
    sum
}

/// `ψ(x)` for finite `x` at precision `w`, via the upward recurrence
/// `ψ(x) = ψ(z) − Σ_{j<m} 1/(x+j)` with `z = x+m ≳ w/4`.
fn digamma_pos_at(x: &Float, w: u64) -> Float {
    let bw = 64 - w.leading_zeros() as u64;
    let n = w + 2 * bw + 48;
    let z_thresh = (n / 4 + 8) as i64;
    let fx = x.floor().and_then(|i| i.to_i64()).unwrap_or(z_thresh);
    let m = if fx >= z_thresh { 0 } else { z_thresh - fx };
    let z = x.add(&iflt(m, n), n, NEAR);
    // Σ_{j=0}^{m−1} 1/(x+j).
    let mut s = Float::zero(n);
    for j in 0..m {
        let d = x.add(&iflt(j, n), n, NEAR);
        s = s.add(&iflt(1, n).div(&d, n, NEAR), n, NEAR);
    }
    // ψ(z) = ln z − 1/(2z) − tail.
    let ln_z = z.ln(n, NEAR);
    let half_over_z = iflt(1, n).div(&z, n, NEAR).scale_pow2(-1);
    let psi_z = ln_z
        .sub(&half_over_z, n, NEAR)
        .sub(&digamma_tail(&z, n), n, NEAR);
    psi_z.sub(&s, n, NEAR).round(w, NEAR)
}

/// `ψ(x)` for finite `x` (not a non-positive integer) at precision `w`.
fn digamma_at(x: &Float, w: u64) -> Float {
    let half = rflt(1, 2, w + 8);
    if x.partial_cmp(&half) == Some(Ordering::Less) {
        // Reflection: ψ(x) = ψ(1−x) − π·cot(πx), with 1−x > ½.
        let n = w + 48 + float_msb(x).max(0) as u64;
        let one = iflt(1, n);
        let one_minus_x = one.sub(x, n, NEAR);
        let psi1 = digamma_pos_at(&one_minus_x, n);
        let pix = pi_at(n).mul(x, n, NEAR);
        let cot = pix.cos(n, NEAR).div(&pix.sin(n, NEAR), n, NEAR);
        let pcot = pi_at(n).mul(&cot, n, NEAR);
        return psi1.sub(&pcot, n, NEAR).round(w, NEAR);
    }
    digamma_pos_at(x, w)
}

/// `ψ⁽ⁿ⁾(z)` for order `n ≥ 1` and large positive `z`, via the asymptotic series
/// (DLMF 5.15.8).
fn polygamma_asymp(order: u64, z: &Float, w: u64) -> Float {
    let n = order;
    let zn = float_powi(z, n, w); // zⁿ
    let znp1 = zn.mul(z, w, NEAR); // z^{n+1}
    // (n−1)!/zⁿ + n!/(2 z^{n+1}).
    let mut acc = Float::from_int(&factorial_int(n - 1), w, NEAR).div(&zn, w, NEAR);
    acc = acc.add(
        &Float::from_int(&factorial_int(n), w, NEAR)
            .div(&znp1, w, NEAR)
            .scale_pow2(-1),
        w,
        NEAR,
    );
    // Tail Σ_{k≥1} B₂ₖ (2k+n−1)!/(2k)! / z^{2k+n}.
    let z2 = z.mul(z, w, NEAR);
    let mut zpow = znp1.mul(z, w, NEAR); // z^{n+2} (k = 1)
    let mut table = BernoulliTable::new();
    let mut prev_msb = i64::MAX;
    let mut k = 1u64;
    loop {
        // ratio = (2k+n−1)!/(2k)! = ∏_{i=1}^{n−1} (2k+i).
        let mut ratio = Int::ONE;
        for i in 1..n {
            ratio = ratio.mul(&Int::from_u64(2 * k + i));
        }
        let coeff = table.even(k).mul(&Rational::from_integer(ratio));
        let term = Float::from_rational(&coeff, w, NEAR).div(&zpow, w, NEAR);
        let msb = float_msb(&term);
        if msb < -(w as i64) {
            break;
        }
        if msb >= prev_msb {
            break;
        }
        prev_msb = msb;
        acc = acc.add(&term, w, NEAR);
        zpow = zpow.mul(&z2, w, NEAR);
        k += 1;
    }
    // Overall sign (−1)^{n−1}.
    if n % 2 == 1 { acc } else { acc.neg() }
}

/// `ψ⁽ⁿ⁾(x)` for order `n ≥ 1` and finite `x` (not a non-positive integer) at
/// precision `w`, via the upward recurrence
/// `ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(z) − (−1)ⁿ n! Σ_{j<m} 1/(x+j)^{n+1}`, `z = x+m` large.
fn polygamma_at(order: u64, x: &Float, w: u64) -> Float {
    let bw = 64 - w.leading_zeros() as u64;
    let n = w + 2 * bw + 48;
    let z_thresh = ((n / 4 + 8) as i64).max(2 * order as i64 + 8);
    let fx = x.floor().and_then(|i| i.to_i64()).unwrap_or(z_thresh);
    let m = if fx >= z_thresh { 0 } else { z_thresh - fx };
    let z = x.add(&iflt(m, n), n, NEAR);
    let psi_z = polygamma_asymp(order, &z, n);
    // S = n! · Σ_{j<m} 1/(x+j)^{n+1}.
    let fact = Float::from_int(&factorial_int(order), n, NEAR);
    let mut s = Float::zero(n);
    for j in 0..m {
        let d = x.add(&iflt(j, n), n, NEAR);
        let dpow = float_powi(&d, order + 1, n);
        s = s.add(&fact.div(&dpow, n, NEAR), n, NEAR);
    }
    // ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(z) − (−1)ⁿ S.
    let raw = if order.is_multiple_of(2) {
        psi_z.sub(&s, n, NEAR)
    } else {
        psi_z.add(&s, n, NEAR)
    };
    raw.round(w, NEAR)
}

/// `(sign_negative, ln|Γ(x)|)` at precision `w` for finite `x` that is not a
/// non-positive integer; uses reflection (DLMF 5.5.3) for `x < 0`.
fn signed_ln_gamma_at(x: &Float, w: u64) -> (bool, Float) {
    if x.sign() == Sign::Positive {
        return (false, ln_gamma_at(x, w));
    }
    // x < 0: |Γ(x)| = π / (|sin πx| · Γ(1−x)); sign(Γ(x)) = sign(sin πx).
    let n = w + 48 + float_msb(x).max(0) as u64;
    let one = iflt(1, n);
    let one_minus_x = one.sub(x, n, NEAR);
    let lg = ln_gamma_at(&one_minus_x, n);
    let sinpix = pi_at(n).mul(x, n, NEAR).sin(n, NEAR);
    let ln_pi = pi_at(n).ln(n, NEAR);
    let ln_abs_sin = sinpix.abs().ln(n, NEAR);
    let lnabs = ln_pi
        .sub(&ln_abs_sin, n, NEAR)
        .sub(&lg, n, NEAR)
        .round(w, NEAR);
    (sinpix.is_sign_negative(), lnabs)
}

/// `B(a, b) = Γ(a)Γ(b)/Γ(a+b)` at precision `w` via `exp` of log-gammas.
fn beta_at(a: &Float, b: &Float, w: u64) -> Float {
    let n = w + 16;
    let ab = a.add(b, n, NEAR);
    let (sa, la) = signed_ln_gamma_at(a, n);
    let (sb, lb) = signed_ln_gamma_at(b, n);
    let (sab, lab) = signed_ln_gamma_at(&ab, n);
    let v = la.add(&lb, n, NEAR).sub(&lab, n, NEAR);
    let mag = v.exp(n, NEAR);
    let neg = sa ^ sb ^ sab;
    if neg { mag.neg() } else { mag }.round(w, NEAR)
}

/// `Yₙ(x)` for order `n ≥ 0` and finite `x > 0` at precision `w` (DLMF 10.8.1).
fn bessel_y_at(n: u64, x: &Float, w: u64) -> Float {
    let ax_floor = x.abs().floor().and_then(|i| i.to_i64()).unwrap_or(i64::MAX);
    // Alternating series reaches ≈ e^{|x|} → ~1.4427·|x| guard bits.
    let x_guard = (((ax_floor as u128 + 1) * 185 / 128).min(u64::MAX as u128)) as u64;
    let ns = w + x_guard + n.saturating_mul(4) + 96;
    let half = x.scale_pow2(-1); // x/2
    let h2 = half.mul(&half, ns, NEAR); // (x/2)²
    let ln_half = half.ln(ns, NEAR);
    let jn = bessel_series_at(n, x, ns, true); // Jₙ(x)
    let pi = pi_at(ns);
    // (2/π) ln(x/2) Jₙ.
    let term_a = ln_half.mul(&jn, ns, NEAR).scale_pow2(1).div(&pi, ns, NEAR);
    // S1 = (x/2)^{−n} Σ_{k=0}^{n−1} (n−k−1)!/k! (x/2)^{2k}.
    let half_neg_n = if n == 0 {
        iflt(1, ns)
    } else {
        iflt(1, ns).div(&float_powi(&half, n, ns), ns, NEAR)
    };
    let mut s1 = Float::zero(ns);
    let mut hp = iflt(1, ns); // (x/2)^{2k}
    for k in 0..n {
        let coef = Float::from_int(&factorial_int(n - k - 1), ns, NEAR).div(
            &Float::from_int(&factorial_int(k), ns, NEAR),
            ns,
            NEAR,
        );
        s1 = s1.add(&coef.mul(&hp, ns, NEAR), ns, NEAR);
        hp = hp.mul(&h2, ns, NEAR);
    }
    let s1 = s1.mul(&half_neg_n, ns, NEAR);
    // S2 = Σ_{k≥0} (−1)ᵏ (Hₖ + H_{n+k} − 2γ) (x/2)^{2k+n}/(k!(n+k)!).
    let two_gamma = gamma_at(ns).scale_pow2(1);
    let mut e =
        float_powi(&half, n, ns).div(&Float::from_int(&factorial_int(n), ns, NEAR), ns, NEAR);
    let mut hk = Float::zero(ns); // H₀
    let mut hnk = harmonic_float(n, ns); // Hₙ
    let mut s2 = Float::zero(ns);
    let kmin = (ax_floor.max(0) as u64) + 2;
    let mut k = 0u64;
    loop {
        let psi = hk.add(&hnk, ns, NEAR).sub(&two_gamma, ns, NEAR);
        let mut term = psi.mul(&e, ns, NEAR);
        if k % 2 == 1 {
            term = term.neg();
        }
        s2 = s2.add(&term, ns, NEAR);
        if k > kmin && float_msb(&term) < -(ns as i64) {
            break;
        }
        k += 1;
        hk = hk.add(&iflt(1, ns).div(&iflt(k as i64, ns), ns, NEAR), ns, NEAR);
        hnk = hnk.add(
            &iflt(1, ns).div(&iflt((n + k) as i64, ns), ns, NEAR),
            ns,
            NEAR,
        );
        let denom = Int::from_u128(k as u128 * (n as u128 + k as u128));
        e = e
            .mul(&h2, ns, NEAR)
            .div(&Float::from_int(&denom, ns, NEAR), ns, NEAR);
        if e.is_zero() {
            break;
        }
    }
    // Yₙ = term_a − (1/π)(S1 + S2).
    let bracket = s1.add(&s2, ns, NEAR).div(&pi, ns, NEAR);
    term_a.sub(&bracket, ns, NEAR).round(w, NEAR)
}

/// `Kₙ(x)` for order `n ≥ 0` and finite `x > 0` at precision `w` (DLMF 10.31.1).
fn bessel_k_at(n: u64, x: &Float, w: u64) -> Float {
    let ax_floor = x.abs().floor().and_then(|i| i.to_i64()).unwrap_or(i64::MAX);
    // ln(x/2)·Iₙ ≈ e^{x} must cancel to Kₙ ≈ e^{−x} → ~2.885·x guard bits.
    let x_guard = (((ax_floor as u128 + 1) * 370 / 128).min(u64::MAX as u128)) as u64;
    let ns = w + x_guard + n.saturating_mul(4) + 96;
    let half = x.scale_pow2(-1);
    let h2 = half.mul(&half, ns, NEAR);
    let ln_half = half.ln(ns, NEAR);
    let in_ = bessel_series_at(n, x, ns, false); // Iₙ(x)
    // piece1 = ½ (x/2)^{−n} Σ_{k=0}^{n−1} (n−k−1)!/k! (−1)ᵏ (x/2)^{2k}.
    let half_neg_n = if n == 0 {
        iflt(1, ns)
    } else {
        iflt(1, ns).div(&float_powi(&half, n, ns), ns, NEAR)
    };
    let mut t1 = Float::zero(ns);
    let mut hp = iflt(1, ns);
    for k in 0..n {
        let mut coef = Float::from_int(&factorial_int(n - k - 1), ns, NEAR).div(
            &Float::from_int(&factorial_int(k), ns, NEAR),
            ns,
            NEAR,
        );
        if k % 2 == 1 {
            coef = coef.neg();
        }
        t1 = t1.add(&coef.mul(&hp, ns, NEAR), ns, NEAR);
        hp = hp.mul(&h2, ns, NEAR);
    }
    let piece1 = t1.mul(&half_neg_n, ns, NEAR).scale_pow2(-1);
    // piece2 = (−1)^{n+1} ln(x/2) Iₙ.
    let mut piece2 = ln_half.mul(&in_, ns, NEAR);
    if n.is_multiple_of(2) {
        piece2 = piece2.neg();
    }
    // piece3 = (−1)ⁿ ½ Σ_{k≥0} (Hₖ + H_{n+k} − 2γ) (x/2)^{2k+n}/(k!(n+k)!).
    let two_gamma = gamma_at(ns).scale_pow2(1);
    let mut e =
        float_powi(&half, n, ns).div(&Float::from_int(&factorial_int(n), ns, NEAR), ns, NEAR);
    let mut hk = Float::zero(ns);
    let mut hnk = harmonic_float(n, ns);
    let mut t2 = Float::zero(ns);
    let kmin = (ax_floor.max(0) as u64) + 2;
    let mut k = 0u64;
    loop {
        let psi = hk.add(&hnk, ns, NEAR).sub(&two_gamma, ns, NEAR);
        let term = psi.mul(&e, ns, NEAR);
        t2 = t2.add(&term, ns, NEAR);
        if k > kmin && float_msb(&term) < -(ns as i64) {
            break;
        }
        k += 1;
        hk = hk.add(&iflt(1, ns).div(&iflt(k as i64, ns), ns, NEAR), ns, NEAR);
        hnk = hnk.add(
            &iflt(1, ns).div(&iflt((n + k) as i64, ns), ns, NEAR),
            ns,
            NEAR,
        );
        let denom = Int::from_u128(k as u128 * (n as u128 + k as u128));
        e = e
            .mul(&h2, ns, NEAR)
            .div(&Float::from_int(&denom, ns, NEAR), ns, NEAR);
        if e.is_zero() {
            break;
        }
    }
    let mut piece3 = t2.scale_pow2(-1);
    if n % 2 == 1 {
        piece3 = piece3.neg();
    }
    piece1
        .add(&piece2, ns, NEAR)
        .add(&piece3, ns, NEAR)
        .round(w, NEAR)
}

#[cfg(test)]
mod agm_tests {
    extern crate std;
    use std::println;

    use super::*;
    use crate::RoundingMode::{AwayFromZero, Nearest, TowardNegative, TowardPositive, TowardZero};

    const MODES: [RoundingMode; 5] = [
        Nearest,
        TowardZero,
        TowardPositive,
        TowardNegative,
        AwayFromZero,
    ];

    // π via the Machin binary-split series (the production `pi_at` body without
    // the embedded-constant shortcut) — kept here only to benchmark against and
    // differentially validate the AGM π below, which is *not* wired into
    // production because the series matches or beats it at every precision.
    fn pi_series_at(w: u64) -> Float {
        let n = w + 32;
        let a1 = super::atan_recip_scaled(5, n);
        let a2 = super::atan_recip_scaled(239, n);
        let pi_scaled = a1.shl(4).checked_sub(&a2.shl(2)).unwrap();
        Float::round_raw(false, pi_scaled, -(n as i64), w, NEAR).0
    }

    // π via the Gauss–Legendre / Brent–Salamin AGM iteration (test-only; see
    // above). a₀=1, b₀=1/√2, t₀=1/4, p₀=1; step: a←(a+b)/2, b←√(ab),
    // t←t−p(a−a')², p←2p; π≈(a+b)²/(4t).
    fn pi_agm_at(w: u64) -> Float {
        let n = w + 32 + 2 * ilog2(w);
        let mut a = iflt(1, n);
        let mut b = rflt(1, 2, n).sqrt(n, NEAR);
        let mut t = rflt(1, 4, n);
        for p_exp in 0..(2 * ilog2(w) as i64 + 16) {
            let a1 = a.add(&b, n, NEAR).scale_pow2(-1);
            let b1 = a.mul(&b, n, NEAR).sqrt(n, NEAR);
            let diff = a.sub(&a1, n, NEAR);
            let d2 = diff.mul(&diff, n, NEAR).scale_pow2(p_exp);
            t = t.sub(&d2, n, NEAR);
            let conv = a1.sub(&b1, n, NEAR);
            a = a1;
            b = b1;
            if conv.is_zero() || floor_log2(&conv) <= -(w as i64) {
                break;
            }
        }
        let s = a.add(&b, n, NEAR);
        s.mul(&s, n, NEAR)
            .div(&t.scale_pow2(2), n, NEAR)
            .round(w, NEAR)
    }

    // Correctly-rounded π/ln through Ziv, but forcing a specific inner method,
    // so the differential tests can compare the two implementations end-to-end.
    fn pi_series(prec: u64, mode: RoundingMode) -> Float {
        Float::ziv(prec, mode, pi_series_at)
    }
    fn pi_agm(prec: u64, mode: RoundingMode) -> Float {
        Float::ziv(prec, mode, pi_agm_at)
    }
    fn ln_series(x: &Float, prec: u64, mode: RoundingMode) -> Float {
        let x = x.clone();
        Float::ziv(prec, mode, move |w| ln_series_at(&x.round(w, NEAR), w))
    }
    fn ln_agm(x: &Float, prec: u64, mode: RoundingMode) -> Float {
        let x = x.clone();
        Float::ziv(prec, mode, move |w| ln_agm_at(&x.round(w, NEAR), w))
    }

    fn same(a: &Float, b: &Float) -> bool {
        a.to_exact_string() == b.to_exact_string()
    }

    #[test]
    fn agm_quick_sanity() {
        // Fast low-precision checks that the AGM math is correct at all (these
        // call the `_at` functions directly regardless of the wiring
        // thresholds). ln is exercised only for x bounded away from 1, where
        // the AGM formula is accurate.
        for &prec in &[53u64, 200, 2000] {
            for &mode in &MODES {
                assert!(
                    same(&pi_agm(prec, mode), &pi_series(prec, mode)),
                    "quick pi_agm != series at {prec} bits, {mode:?}"
                );
            }
            for k in [2i64, 3, 7, 100, 1000] {
                let x = iflt(k, prec + 64);
                assert!(
                    same(&ln_agm(&x, prec, Nearest), &ln_series(&x, prec, Nearest)),
                    "quick ln_agm != series at {prec} bits, x={k}"
                );
            }
            let half = rflt(1, 2, prec + 64); // 0.5, far enough from 1
            assert!(same(
                &ln_agm(&half, prec, Nearest),
                &ln_series(&half, prec, Nearest)
            ));
        }
        // Known values.
        assert!(same(
            &ln_agm(&iflt(2, 2064), 2000, Nearest),
            &Float::ln2(2000, Nearest)
        ));
        assert!(same(&pi_agm(2000, Nearest), &Float::pi(2000, Nearest)));
    }

    #[test]
    fn ln_near_one_falls_back_to_series() {
        // Where the AGM formula loses relative accuracy (x ≈ 1, tiny result),
        // the wired `Float::ln` must fall back to the series and stay
        // bit-identical to it — above the AGM threshold and in every mode.
        let prec = LN_AGM_THRESHOLD + 500; // exercises the wired dispatch
        let ref_prec = prec;
        // x = 1 exactly (ln = 0), and x = 1 ± 2^-k for k straddling the fence.
        let mut xs = alloc::vec::Vec::new();
        xs.push(iflt(1, prec + 64));
        for k in [10i64, 33, 60, 200] {
            let d = iflt(1, prec + 64).scale_pow2(-k);
            xs.push(iflt(1, prec + 64).add(&d, prec + 64, NEAR));
            xs.push(iflt(1, prec + 64).sub(&d, prec + 64, NEAR));
        }
        for x in &xs {
            for &mode in &MODES {
                assert!(
                    same(&x.ln(prec, mode), &ln_series(x, ref_prec, mode)),
                    "wired ln != series near 1 at {prec} bits, {mode:?}, x={}",
                    x.to_exact_string()
                );
            }
        }
    }

    #[test]
    fn ln_agm_matches_series_all_modes() {
        // A spread of x > 0 (including < 1) and precisions crossing the AGM
        // threshold. LCG for reproducibility.
        let mut state: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };
        for &prec in &[64u64, 777, 9000] {
            let cases = if prec >= 40000 { 3 } else { 8 };
            for _ in 0..cases {
                // x = mantissa · 2^scale, scale in [-40, 40].
                let mant = Int::from_u64(next() | 1);
                let scale = (next() % 81) as i64 - 40;
                let x = Float::from_int(&mant, prec + 64, NEAR).scale_pow2(scale);
                for &mode in &MODES {
                    let s = ln_series(&x, prec, mode);
                    let a = ln_agm(&x, prec, mode);
                    assert!(
                        same(&s, &a),
                        "ln_agm != series at {prec} bits, {mode:?}, x={}",
                        x.to_exact_string()
                    );
                    assert!(
                        same(&x.ln(prec, mode), &s),
                        "Float::ln != series at {prec} bits, {mode:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn ln_agm_known_values() {
        // ln(2) equals the embedded constant; ln(e) = 1; ln(1) = 0.
        for &prec in &[500u64, 3000] {
            let two = iflt(2, prec + 64);
            assert!(
                same(&ln_agm(&two, prec, Nearest), &Float::ln2(prec, Nearest)),
                "ln_agm(2) != ln2 at {prec}"
            );
            let e = Float::e(prec + 64, Nearest);
            let one = iflt(1, prec);
            assert!(
                same(&ln_agm(&e, prec, Nearest), &one),
                "ln_agm(e) != 1 at {prec}"
            );
        }
    }

    #[test]
    #[ignore = "high-precision differential (slow); run with --ignored"]
    fn ln_agm_high_precision_matches_series() {
        // Above the wired threshold, so Float::ln itself takes the AGM path.
        for &prec in &[40000u64, 70000] {
            let x = iflt(3, prec + 64).scale_pow2(-1); // 1.5
            for &mode in &MODES {
                assert!(same(&ln_agm(&x, prec, mode), &ln_series(&x, prec, mode)));
                assert!(same(&x.ln(prec, mode), &ln_series(&x, prec, mode)));
            }
        }
    }

    #[test]
    #[ignore = "timing benchmark; run with --ignored --nocapture"]
    fn agm_crossover() {
        use std::time::Instant;
        fn t<F: Fn() -> Float>(f: F) -> f64 {
            let start = Instant::now();
            let _ = f();
            start.elapsed().as_secs_f64() * 1e3
        }
        println!("\n== π: Machin series vs Brent–Salamin AGM (ms) ==");
        println!(
            "{:>10}  {:>12}  {:>12}  {:>8}",
            "bits", "series", "agm", "speedup"
        );
        for &w in &[
            1u64 << 12,
            1 << 14,
            1 << 15,
            1 << 16,
            1 << 17,
            1 << 18,
            1 << 19,
        ] {
            let ts = t(|| pi_series_at(w));
            let ta = t(|| pi_agm_at(w));
            println!("{w:>10}  {ts:>12.2}  {ta:>12.2}  {:>7.2}x", ts / ta);
        }
        println!("\n== ln: atanh series vs AGM (ms) ==");
        println!(
            "{:>10}  {:>12}  {:>12}  {:>8}",
            "bits", "series", "agm", "speedup"
        );
        let x = iflt(3, 1 << 20).scale_pow2(-1); // 1.5
        for &w in &[
            1u64 << 12,
            1 << 13,
            1 << 14,
            1 << 15,
            1 << 16,
            1 << 17,
            1 << 18,
        ] {
            let xw = x.round(w, NEAR);
            let ts = t(|| ln_series_at(&xw, w));
            let ta = t(|| ln_agm_at(&xw, w));
            println!("{w:>10}  {ts:>12.2}  {ta:>12.2}  {:>7.2}x", ts / ta);
        }
    }
}

#[cfg(test)]
mod sin_cos_rect_tests {
    extern crate std;
    use std::println;

    use super::*;
    use crate::RoundingMode::{AwayFromZero, Nearest, TowardNegative, TowardPositive, TowardZero};

    const MODES: [RoundingMode; 5] = [
        Nearest,
        TowardZero,
        TowardPositive,
        TowardNegative,
        AwayFromZero,
    ];

    /// `sin_cos_at` with a forced choice of series, so the two implementations
    /// can be compared through the identical quadrant reduction + Ziv wrapper.
    fn sin_cos_full(x: &Float, prec: u64, mode: RoundingMode, rect: bool) -> (Float, Float) {
        let series = move |r: &Float, w: u64| {
            if rect {
                sin_cos_series_rect(r, w)
            } else {
                sin_cos_series_simple(r, w)
            }
        };
        let at = move |x: &Float, w: u64| -> (Float, Float) {
            let pi = pi_at(w);
            let half_pi = pi.scale_pow2(-1);
            let q = x.div(&half_pi, w, NEAR).round_half_up_to_int();
            let r = x.sub(
                &Float::from_int(&q, w, NEAR).mul(&half_pi, w, NEAR),
                w,
                NEAR,
            );
            let (sr, cr) = series(&r, w);
            let quad = q.rem_euclid(&Int::from_i64(4)).to_i64().unwrap_or(0);
            match quad {
                0 => (sr, cr),
                1 => (cr, sr.neg()),
                2 => (sr.neg(), cr.neg()),
                _ => (cr.neg(), sr),
            }
        };
        let xs = x.clone();
        let at2 = at;
        let s = Float::ziv(prec, mode, move |w| at2(&xs.round(w, NEAR), w).0);
        let xc = x.clone();
        let c = Float::ziv(prec, mode, move |w| at(&xc.round(w, NEAR), w).1);
        (s, c)
    }

    fn bit_identical(a: &Float, b: &Float) -> bool {
        a.repr == b.repr && a.precision == b.precision
    }

    /// A crude xorshift so tests stay dependency-free and deterministic.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        /// A random finite float at precision `p`, magnitude scaled by `scale`.
        fn float(&mut self, p: u64, scale: i64) -> Float {
            let mant = Int::from_i64((self.next() >> 1) as i64);
            let f = Float::from_int(&mant, p, NEAR);
            let e = (self.next() % 30) as i64 - 15 + scale;
            let f = f.scale_pow2(e);
            if self.next() & 1 == 0 { f.neg() } else { f }
        }
    }

    fn differential(precisions: &[u64], per_prec: usize) {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for &p in precisions {
            for _ in 0..per_prec {
                // Mix tiny, ~1, and large (big quadrant reduction) magnitudes.
                for &scale in &[-20i64, 0, 30] {
                    let x = rng.float(p, scale);
                    for &mode in &MODES {
                        let (ss, sc) = sin_cos_full(&x, p, mode, false);
                        let (rs, rc) = sin_cos_full(&x, p, mode, true);
                        assert!(
                            bit_identical(&ss, &rs),
                            "sin mismatch p={p} mode={mode:?} x={x:?}"
                        );
                        assert!(
                            bit_identical(&sc, &rc),
                            "cos mismatch p={p} mode={mode:?} x={x:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rect_matches_simple_fast() {
        // Small-w edge cases (few blocks) plus a straddle of the 1024 crossover.
        differential(&[300, 900, 1024, 1536, 2048], 5);
    }

    #[test]
    fn rect_known_values() {
        let p = 2048;
        let n = Nearest;
        // sin 0 = 0, cos 0 = 1.
        let (s0, c0) = sin_cos_series_rect(&Float::zero(p + 32), p);
        assert!(s0.is_zero());
        assert!(c0.sub(&iflt(1, p), p, n).is_zero());
        // sin²+cos² = 1 at a generic argument.
        let x = Float::from_int(&Int::from_i64(7), p, n).div(
            &Float::from_int(&Int::from_i64(10), p, n),
            p,
            n,
        );
        let s = x.sin(p, n);
        let c = x.cos(p, n);
        let one = s.mul(&s, p, n).add(&c.mul(&c, p, n), p, n);
        assert!(one.sub(&iflt(1, p), p, n).abs() < rflt(1, 1i64 << 60, p));
        // sin(π/6) ≈ 1/2.
        let pi6 = Float::pi(p, n).div(&Float::from_int(&Int::from_i64(6), p, n), p, n);
        let half = pi6.sin(p, n);
        assert!(half.sub(&rflt(1, 2, p), p, n).abs() < rflt(1, 1i64 << 60, p));
    }

    #[test]
    #[ignore = "heavy: run with --release --ignored"]
    fn rect_matches_simple_heavy() {
        differential(&[3000, 4096, 8192, 16384], 4);
    }

    #[test]
    #[ignore = "benchmark: cargo test --release -- --ignored bench_sin_cos_rect --nocapture"]
    fn bench_sin_cos_rect() {
        use std::time::Instant;
        let n = Nearest;
        fn t(iters: u32, mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let s = Instant::now();
                for _ in 0..iters {
                    f();
                }
                best = best.min(s.elapsed().as_secs_f64() / iters as f64);
            }
            best
        }
        println!("\n  w        simple(ms)   rect(ms)   speedup   Cblock");
        for &w in &[
            256u64, 512, 1024, 1500, 2048, 4096, 8192, 16384, 32768, 65536,
        ] {
            let x = Float::from_int(&Int::from_i64(12345), w + 64, n).div(
                &Float::from_int(&Int::from_i64(10000), w + 64, n),
                w + 64,
                n,
            );
            let iters: u32 = if w <= 1024 {
                200
            } else if w <= 4096 {
                40
            } else if w <= 16384 {
                8
            } else {
                2
            };
            // Reduce once to r ∈ [−π/4, π/4] to time the series itself.
            let z = scaled_int(&x, (w + 32) as i64)
                .magnitude()
                .square()
                .shr(w + 32);
            let c = ((2 * sin_cos_term_count(&z, w + 32)) as f64).sqrt().ceil() as u64;
            let ts = t(iters, || {
                std::hint::black_box(sin_cos_series_simple(&x, w));
            });
            let tr = t(iters, || {
                std::hint::black_box(sin_cos_series_rect(&x, w));
            });
            println!(
                "{w:>6}   {:>10.4} {:>10.4}   {:>6.2}x   {c:>5}",
                ts * 1e3,
                tr * 1e3,
                ts / tr
            );
        }
    }
}
