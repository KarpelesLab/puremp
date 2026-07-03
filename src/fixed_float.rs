//! Fixed-precision floating point — a `mpfx`-style convenience wrapper over
//! [`Float`].
//!
//! [`Float`] takes an explicit output precision and [`RoundingMode`] on *every*
//! operation (the flexible MPFR-style interface). [`FixedFloat`] instead bakes a
//! precision and rounding mode into the value, so it supports the ordinary
//! `+ - * /` operators and method-style transcendentals without threading those
//! parameters through each call. Binary operations use the larger of the two
//! operands' precisions and the left operand's rounding mode.

use core::cmp::Ordering;
use core::fmt;

use crate::float::{Float, RoundingMode};
use crate::int::{Int, Sign};

/// A floating-point value carrying a fixed working precision and rounding mode.
#[derive(Clone)]
pub struct FixedFloat {
    value: Float,
    mode: RoundingMode,
}

impl FixedFloat {
    /// Wraps a [`Float`], adopting its precision and the given rounding mode.
    #[inline]
    pub fn from_float(value: Float, mode: RoundingMode) -> FixedFloat {
        FixedFloat { value, mode }
    }

    /// Builds from an integer at `precision` bits.
    pub fn from_int(n: &Int, precision: u64, mode: RoundingMode) -> FixedFloat {
        FixedFloat {
            value: Float::from_int(n, precision, mode),
            mode,
        }
    }

    /// Builds from an `f64` at `precision` bits.
    pub fn from_f64(x: f64, precision: u64, mode: RoundingMode) -> FixedFloat {
        FixedFloat {
            value: Float::from_f64(x, precision, mode),
            mode,
        }
    }

    /// Positive zero at `precision` bits.
    pub fn zero(precision: u64, mode: RoundingMode) -> FixedFloat {
        FixedFloat {
            value: Float::zero(precision),
            mode,
        }
    }

    /// NaN / ±∞ at `precision` bits.
    pub fn nan(precision: u64, mode: RoundingMode) -> FixedFloat {
        FixedFloat {
            value: Float::nan(precision),
            mode,
        }
    }
    /// Positive infinity at `precision` bits.
    pub fn infinity(precision: u64, mode: RoundingMode) -> FixedFloat {
        FixedFloat {
            value: Float::infinity(precision),
            mode,
        }
    }
    /// Negative infinity at `precision` bits.
    pub fn neg_infinity(precision: u64, mode: RoundingMode) -> FixedFloat {
        FixedFloat {
            value: Float::neg_infinity(precision),
            mode,
        }
    }

    /// π at `precision` bits.
    pub fn pi(precision: u64, mode: RoundingMode) -> FixedFloat {
        FixedFloat {
            value: Float::pi(precision, mode),
            mode,
        }
    }
    /// Euler's number e at `precision` bits.
    pub fn e(precision: u64, mode: RoundingMode) -> FixedFloat {
        FixedFloat {
            value: Float::e(precision, mode),
            mode,
        }
    }

    // --- accessors ---

    /// Returns the working precision in bits.
    #[inline]
    pub fn precision(&self) -> u64 {
        self.value.precision()
    }

    /// Returns the rounding mode.
    #[inline]
    pub fn rounding_mode(&self) -> RoundingMode {
        self.mode
    }

    /// Returns the underlying [`Float`].
    #[inline]
    pub fn as_float(&self) -> &Float {
        &self.value
    }

    /// Consumes into the underlying [`Float`].
    #[inline]
    pub fn into_float(self) -> Float {
        self.value
    }

    /// Returns the value as the nearest `f64`.
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }

    /// Returns the shortest round-tripping decimal string.
    #[inline]
    pub fn to_shortest_string(&self) -> alloc::string::String {
        self.value.to_shortest_string()
    }

    /// Returns `true` if this value is NaN.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.value.is_nan()
    }
    /// Returns `true` if this value is `±∞`.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.value.is_infinite()
    }
    /// Returns `true` if this value is finite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }
    /// Returns `true` if this value is `±0`.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
    /// Returns the sign of this value.
    #[inline]
    pub fn sign(&self) -> Sign {
        self.value.sign()
    }

    /// Re-rounds to a different precision (keeping the rounding mode).
    pub fn with_precision(&self, precision: u64) -> FixedFloat {
        FixedFloat {
            value: self.value.round(precision, self.mode),
            mode: self.mode,
        }
    }

    // --- arithmetic (result precision = max of operand precisions) ---

    fn combine(
        &self,
        rhs: &FixedFloat,
        op: fn(&Float, &Float, u64, RoundingMode) -> Float,
    ) -> FixedFloat {
        let precision = self.value.precision().max(rhs.value.precision());
        FixedFloat {
            value: op(&self.value, &rhs.value, precision, self.mode),
            mode: self.mode,
        }
    }

    /// Returns `-self`.
    pub fn neg(&self) -> FixedFloat {
        FixedFloat {
            value: self.value.neg(),
            mode: self.mode,
        }
    }

    /// Returns `|self|`.
    pub fn abs(&self) -> FixedFloat {
        FixedFloat {
            value: self.value.abs(),
            mode: self.mode,
        }
    }

    /// Returns `√self`.
    pub fn sqrt(&self) -> FixedFloat {
        self.unary(Float::sqrt)
    }
    /// Returns `e^self`.
    pub fn exp(&self) -> FixedFloat {
        self.unary(Float::exp)
    }
    /// Returns `ln(self)`.
    pub fn ln(&self) -> FixedFloat {
        self.unary(Float::ln)
    }
    /// Returns `sin(self)`.
    pub fn sin(&self) -> FixedFloat {
        self.unary(Float::sin)
    }
    /// Returns `cos(self)`.
    pub fn cos(&self) -> FixedFloat {
        self.unary(Float::cos)
    }
    /// Returns `tan(self)`.
    pub fn tan(&self) -> FixedFloat {
        self.unary(Float::tan)
    }
    /// Returns `atan(self)`.
    pub fn atan(&self) -> FixedFloat {
        self.unary(Float::atan)
    }

    /// Returns `self` raised to the floating power `exp`.
    pub fn pow(&self, exp: &FixedFloat) -> FixedFloat {
        let precision = self.value.precision().max(exp.value.precision());
        FixedFloat {
            value: self.value.pow(&exp.value, precision, self.mode),
            mode: self.mode,
        }
    }

    fn unary(&self, op: fn(&Float, u64, RoundingMode) -> Float) -> FixedFloat {
        FixedFloat {
            value: op(&self.value, self.value.precision(), self.mode),
            mode: self.mode,
        }
    }
}

impl PartialEq for FixedFloat {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialOrd for FixedFloat {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl fmt::Display for FixedFloat {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

impl fmt::Debug for FixedFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FixedFloat({:?}, {:?})", self.value, self.mode)
    }
}

macro_rules! fixed_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $call:path) => {
        impl core::ops::$tr for FixedFloat {
            type Output = FixedFloat;
            #[inline]
            fn $m(self, rhs: FixedFloat) -> FixedFloat {
                self.combine(&rhs, $call)
            }
        }
        impl core::ops::$tr<&FixedFloat> for &FixedFloat {
            type Output = FixedFloat;
            #[inline]
            fn $m(self, rhs: &FixedFloat) -> FixedFloat {
                self.combine(rhs, $call)
            }
        }
        impl core::ops::$atr for FixedFloat {
            #[inline]
            fn $am(&mut self, rhs: FixedFloat) {
                *self = self.combine(&rhs, $call);
            }
        }
    };
}

fixed_binop!(Add, add, AddAssign, add_assign, Float::add);
fixed_binop!(Sub, sub, SubAssign, sub_assign, Float::sub);
fixed_binop!(Mul, mul, MulAssign, mul_assign, Float::mul);
fixed_binop!(Div, div, DivAssign, div_assign, Float::div);

impl core::ops::Neg for FixedFloat {
    type Output = FixedFloat;
    #[inline]
    fn neg(self) -> FixedFloat {
        FixedFloat::neg(&self)
    }
}
