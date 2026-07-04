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
    let k = x.div(&ln2, n, NEAR).round_to_int();
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

/// `(sin r, cos r)` by Taylor series for small `|r| ≤ π/4`, at precision `w`,
/// in scaled integer arithmetic (see [`atanh_series`]). Handles the sign of
/// `r` explicitly: both sums run on `z = r² ≥ 0`, whose alternating partial
/// sums stay non-negative.
fn sin_cos_series(r: &Float, w: u64) -> (Float, Float) {
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
