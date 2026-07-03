//! Extended rationals with infinities — an exact [`Rational`] augmented with
//! `+∞`, `-∞`, and `NaN`, following the usual IEEE-style arithmetic
//! (`1/0 = +∞`, `-1/0 = -∞`, `0/0 = NaN`, `∞ − ∞ = NaN`, `∞ · 0 = NaN`, …).
//!
//! Unlike floating point this stays exact on the finite part — no rounding — so
//! it is a projectively/affinely extended field useful for computations that may
//! legitimately reach an infinity (limits, continued fractions, homogeneous
//! coordinates) without losing exactness elsewhere.

use core::cmp::Ordering;
use core::fmt;

use crate::int::Int;
use crate::rational::Rational;

/// A rational number extended with signed infinities and NaN.
#[derive(Clone)]
pub enum InfRational {
    /// A finite exact rational.
    Finite(Rational),
    /// Positive infinity.
    PosInf,
    /// Negative infinity.
    NegInf,
    /// Not-a-number (e.g. `0/0`, `∞ − ∞`).
    Nan,
}

impl InfRational {
    /// Wraps a finite rational.
    #[inline]
    pub fn finite(r: Rational) -> InfRational {
        InfRational::Finite(r)
    }

    /// Builds `num / den`, mapping a zero denominator to `±∞` (or `NaN` for
    /// `0/0`).
    pub fn ratio(num: Int, den: Int) -> InfRational {
        if den.is_zero() {
            match num.signum() {
                1 => InfRational::PosInf,
                -1 => InfRational::NegInf,
                _ => InfRational::Nan,
            }
        } else {
            InfRational::Finite(Rational::new(num, den))
        }
    }

    // --- classification ---

    /// Returns `true` if NaN.
    #[inline]
    pub fn is_nan(&self) -> bool {
        matches!(self, InfRational::Nan)
    }
    /// Returns `true` if `±∞`.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        matches!(self, InfRational::PosInf | InfRational::NegInf)
    }
    /// Returns `true` if finite (not NaN or `±∞`).
    #[inline]
    pub fn is_finite(&self) -> bool {
        matches!(self, InfRational::Finite(_))
    }
    /// Returns `true` if a finite zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self, InfRational::Finite(r) if r.is_zero())
    }

    /// Extended sign: `-1`, `0`, or `1` (`0` for NaN and finite zero).
    fn ext_sign(&self) -> i32 {
        match self {
            InfRational::PosInf => 1,
            InfRational::NegInf => -1,
            InfRational::Nan => 0,
            InfRational::Finite(r) => r.signum(),
        }
    }

    /// Returns the finite value, or `None` for `±∞`/`NaN`.
    pub fn to_rational(&self) -> Option<&Rational> {
        match self {
            InfRational::Finite(r) => Some(r),
            _ => None,
        }
    }

    // --- arithmetic ---

    /// Returns `-self`.
    pub fn neg(&self) -> InfRational {
        match self {
            InfRational::Finite(r) => InfRational::Finite(r.neg()),
            InfRational::PosInf => InfRational::NegInf,
            InfRational::NegInf => InfRational::PosInf,
            InfRational::Nan => InfRational::Nan,
        }
    }

    /// Returns `|self|`.
    pub fn abs(&self) -> InfRational {
        match self {
            InfRational::Finite(r) => InfRational::Finite(r.abs()),
            InfRational::PosInf | InfRational::NegInf => InfRational::PosInf,
            InfRational::Nan => InfRational::Nan,
        }
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &InfRational) -> InfRational {
        use InfRational::*;
        match (self, rhs) {
            (Nan, _) | (_, Nan) => Nan,
            (PosInf, NegInf) | (NegInf, PosInf) => Nan,
            (PosInf, _) | (_, PosInf) => PosInf,
            (NegInf, _) | (_, NegInf) => NegInf,
            (Finite(a), Finite(b)) => Finite(a.add(b)),
        }
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &InfRational) -> InfRational {
        self.add(&rhs.neg())
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &InfRational) -> InfRational {
        use InfRational::*;
        if self.is_nan() || rhs.is_nan() {
            return Nan;
        }
        if let (Finite(a), Finite(b)) = (self, rhs) {
            return Finite(a.mul(b));
        }
        // At least one infinity: ∞·0 is NaN, otherwise a signed infinity.
        if self.is_zero() || rhs.is_zero() {
            return Nan;
        }
        if self.ext_sign() * rhs.ext_sign() > 0 {
            PosInf
        } else {
            NegInf
        }
    }

    /// Returns `self / rhs`.
    pub fn div(&self, rhs: &InfRational) -> InfRational {
        use InfRational::*;
        match (self, rhs) {
            (Nan, _) | (_, Nan) => Nan,
            // ∞ / ∞ is NaN.
            (PosInf | NegInf, PosInf | NegInf) => Nan,
            // ∞ / finite = signed ∞.
            (PosInf | NegInf, Finite(b)) => {
                if self.ext_sign() * finite_sign(b) >= 0 {
                    PosInf
                } else {
                    NegInf
                }
            }
            // finite / ∞ = 0.
            (Finite(_), PosInf | NegInf) => Finite(Rational::ZERO),
            (Finite(a), Finite(b)) => {
                if b.is_zero() {
                    match a.signum() {
                        1 => PosInf,
                        -1 => NegInf,
                        _ => Nan, // 0/0
                    }
                } else {
                    Finite(a.div(b))
                }
            }
        }
    }

    /// Returns `1/self` (`1/0 = +∞`, `1/±∞ = 0`, `1/NaN = NaN`).
    pub fn recip(&self) -> InfRational {
        InfRational::Finite(Rational::ONE).div(self)
    }
}

/// Sign of a finite rational treating zero as `+` (for `∞ / finite`).
fn finite_sign(r: &Rational) -> i32 {
    if r.is_negative() { -1 } else { 1 }
}

impl PartialEq for InfRational {
    fn eq(&self, other: &Self) -> bool {
        use InfRational::*;
        match (self, other) {
            (Finite(a), Finite(b)) => a == b,
            (PosInf, PosInf) | (NegInf, NegInf) => true,
            _ => false, // NaN never equal; different classes unequal
        }
    }
}

impl PartialOrd for InfRational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use InfRational::*;
        match (self, other) {
            (Nan, _) | (_, Nan) => None,
            (Finite(a), Finite(b)) => Some(a.cmp(b)),
            (NegInf, NegInf) | (PosInf, PosInf) => Some(Ordering::Equal),
            (NegInf, _) | (_, PosInf) => Some(Ordering::Less),
            (PosInf, _) | (_, NegInf) => Some(Ordering::Greater),
        }
    }
}

impl From<Rational> for InfRational {
    #[inline]
    fn from(r: Rational) -> InfRational {
        InfRational::Finite(r)
    }
}

impl From<Int> for InfRational {
    #[inline]
    fn from(n: Int) -> InfRational {
        InfRational::Finite(Rational::from_integer(n))
    }
}

impl core::str::FromStr for InfRational {
    type Err = crate::error::Error;

    /// Parses `inf`/`+inf`/`-inf`/`nan` (case-insensitive) or a finite rational
    /// (`"3"`, `"-3/4"`, `"1.5"`).
    fn from_str(s: &str) -> crate::error::Result<InfRational> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("nan") {
            Ok(InfRational::Nan)
        } else if t.eq_ignore_ascii_case("inf") || t.eq_ignore_ascii_case("+inf") {
            Ok(InfRational::PosInf)
        } else if t.eq_ignore_ascii_case("-inf") {
            Ok(InfRational::NegInf)
        } else {
            Ok(InfRational::Finite(t.parse()?))
        }
    }
}

impl fmt::Display for InfRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InfRational::Finite(r) => fmt::Display::fmt(r, f),
            InfRational::PosInf => f.write_str("inf"),
            InfRational::NegInf => f.write_str("-inf"),
            InfRational::Nan => f.write_str("NaN"),
        }
    }
}

impl fmt::Debug for InfRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InfRational({self})")
    }
}

macro_rules! inf_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for InfRational {
            type Output = InfRational;
            #[inline]
            fn $m(self, rhs: InfRational) -> InfRational {
                InfRational::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&InfRational> for &InfRational {
            type Output = InfRational;
            #[inline]
            fn $m(self, rhs: &InfRational) -> InfRational {
                InfRational::$m(self, rhs)
            }
        }
        impl core::ops::$atr for InfRational {
            #[inline]
            fn $am(&mut self, rhs: InfRational) {
                *self = InfRational::$m(self, &rhs);
            }
        }
    };
}

inf_binop!(Add, add, AddAssign, add_assign);
inf_binop!(Sub, sub, SubAssign, sub_assign);
inf_binop!(Mul, mul, MulAssign, mul_assign);
inf_binop!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for InfRational {
    type Output = InfRational;
    #[inline]
    fn neg(self) -> InfRational {
        InfRational::neg(&self)
    }
}
