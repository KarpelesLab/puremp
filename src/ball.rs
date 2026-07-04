//! Ball (midpoint–radius) arithmetic — rigorous real arithmetic in the style of
//! Arb.
//!
//! A [`Ball`] is a real enclosure `mid ± rad` (the set `[mid − rad, mid + rad]`)
//! where the midpoint `mid` is an arbitrary-precision [`Float`] and the radius
//! `rad` is a small, low-precision non-negative `Float` that rigorously bounds
//! the error. Every operation rounds the midpoint at working precision and folds
//! the rounding error into the radius (always rounded *up*), so the true result
//! is guaranteed to lie in the returned ball.
//!
//! This is the mid–rad counterpart to [`Interval`]'s
//! inf–sup form. At high precision it is cheaper: only the midpoint carries full
//! precision, while the radius stays a handful of bits. The two cross-convert via
//! [`Ball::to_interval`] / [`Ball::from_interval`]; `÷` and `sqrt` are computed
//! through the interval form to reuse its proven directed-rounding enclosures.
//!
//! References: J. van der Hoeven, "Ball arithmetic" (2009); F. Johansson, "Arb:
//! Efficient Arbitrary-Precision Midpoint-Radius Interval Arithmetic" (2017).

use core::cmp::Ordering;
use core::fmt;

use crate::float::{Float, RoundingMode};
use crate::int::{Int, Sign};
use crate::interval::Interval;
use crate::rational::Rational;

const DOWN: RoundingMode = RoundingMode::TowardNegative;
const UP: RoundingMode = RoundingMode::TowardPositive;
const NEAR: RoundingMode = RoundingMode::Nearest;

/// Precision (bits) of the radius. The radius only needs to bound an error, so a
/// handful of bits suffice; Arb uses ~30.
const RAD_PREC: u64 = 30;

/// A rigorous real enclosure `mid ± rad`.
#[derive(Clone)]
pub struct Ball {
    mid: Float,
    rad: Float, // ≥ 0, at RAD_PREC bits, always an upper bound
}

/// A non-negative radius `0` at the radius precision.
fn rad_zero() -> Float {
    Float::from_int(&Int::ZERO, RAD_PREC, NEAR)
}

/// The midpoint rounding error to fold into the radius: `0` when the operation
/// was exact (`Ordering::Equal`), else a rigorous ½-ulp bound.
fn round_err(mid: &Float, ord: Ordering) -> Float {
    if ord == Ordering::Equal {
        rad_zero()
    } else {
        half_ulp(mid)
    }
}

/// A rigorous upper bound on the round-to-nearest error of `f` (½ ulp = 2^{e−1}
/// for a value with least-significant-bit weight `2^e`). Returns `+∞` for a
/// non-finite midpoint and `0` for zero.
fn half_ulp(f: &Float) -> Float {
    if f.is_zero() {
        return rad_zero();
    }
    match f.exponent() {
        // ½ ulp = 2^{e−1}; an out-of-`i32` exponent is unreachable for any real
        // value, so fall back to the (still rigorous) unbounded radius.
        Some(e) => i32::try_from(e - 1)
            .map(|k| Float::from_rational(&Rational::power_of_two(k), RAD_PREC, UP))
            .unwrap_or_else(|_| Float::infinity(RAD_PREC)),
        None => Float::infinity(RAD_PREC), // NaN/∞ midpoint → unbounded
    }
}

impl Ball {
    /// A ball `mid ± |rad|` (the radius magnitude is taken and rounded up to
    /// the radius precision).
    pub fn new(mid: Float, rad: Float) -> Ball {
        let rad = rad.abs().round(RAD_PREC, UP);
        Ball { mid, rad }
    }

    /// The exact (zero-radius) ball at `x`.
    pub fn point(x: Float) -> Ball {
        Ball {
            mid: x,
            rad: rad_zero(),
        }
    }

    /// A ball enclosing the integer `n`, midpoint at `precision` bits (exact, so
    /// radius `0`).
    pub fn from_int(n: &Int, precision: u64) -> Ball {
        Ball {
            mid: Float::from_int(n, precision, NEAR),
            rad: rad_zero(),
        }
    }

    /// A ball enclosing the rational `r`, midpoint rounded to `precision` bits and
    /// the rounding error folded into the radius.
    pub fn from_rational(r: &Rational, precision: u64) -> Ball {
        let mid = Float::from_rational(r, precision, NEAR);
        let rad = match mid.to_rational() {
            Some(m) if m == *r => rad_zero(), // exactly representable
            _ => half_ulp(&mid),
        };
        Ball { mid, rad }
    }

    /// A ball enclosing the `f64` value `x` (exact — `f64` is dyadic).
    pub fn from_f64(x: f64, precision: u64) -> Ball {
        Ball {
            mid: Float::from_f64(x, precision, NEAR),
            rad: rad_zero(),
        }
    }

    /// The midpoint.
    #[inline]
    pub fn midpoint(&self) -> &Float {
        &self.mid
    }

    /// The radius (a non-negative upper bound on `|value − midpoint|`).
    #[inline]
    pub fn radius(&self) -> &Float {
        &self.rad
    }

    /// The midpoint's precision (bits).
    #[inline]
    pub fn precision(&self) -> u64 {
        self.mid.precision()
    }

    /// The rigorous lower endpoint `mid − rad` (rounded down).
    pub fn lower(&self) -> Float {
        self.mid.sub(&self.rad, self.precision(), DOWN)
    }

    /// The rigorous upper endpoint `mid + rad` (rounded up).
    pub fn upper(&self) -> Float {
        self.mid.add(&self.rad, self.precision(), UP)
    }

    /// Whether the ball certainly contains `x`.
    pub fn contains(&self, x: &Float) -> bool {
        &self.lower() <= x && x <= &self.upper()
    }

    /// Whether the ball certainly contains zero.
    pub fn contains_zero(&self) -> bool {
        // lower ≤ 0 ≤ upper, checked rigorously via the endpoints.
        self.lower().sign() != Sign::Positive && self.upper().sign() != Sign::Negative
    }

    /// Whether both midpoint and radius are finite.
    pub fn is_finite(&self) -> bool {
        self.mid.is_finite() && self.rad.is_finite()
    }

    fn work_precision(&self, rhs: &Ball) -> u64 {
        self.precision().max(rhs.precision())
    }

    /// `self + rhs`. `rad = rad_a + rad_b + (midpoint rounding error)`, rounded up.
    pub fn add(&self, rhs: &Ball) -> Ball {
        let p = self.work_precision(rhs);
        let (mid, ord) = self.mid.add_ternary(&rhs.mid, p, NEAR);
        let rad = sum_up(&[&self.rad, &rhs.rad, &round_err(&mid, ord)]);
        Ball { mid, rad }
    }

    /// `self − rhs`.
    pub fn sub(&self, rhs: &Ball) -> Ball {
        let p = self.work_precision(rhs);
        let (mid, ord) = self.mid.sub_ternary(&rhs.mid, p, NEAR);
        let rad = sum_up(&[&self.rad, &rhs.rad, &round_err(&mid, ord)]);
        Ball { mid, rad }
    }

    /// `−self` (exact negation; radius unchanged).
    pub fn neg(&self) -> Ball {
        Ball {
            mid: self.mid.neg(),
            rad: self.rad.clone(),
        }
    }

    /// `self · rhs`. `rad = |mid_a|·rad_b + |mid_b|·rad_a + rad_a·rad_b +
    /// ½ulp(mid)`, rounded up.
    pub fn mul(&self, rhs: &Ball) -> Ball {
        let p = self.work_precision(rhs);
        let (mid, ord) = self.mid.mul_ternary(&rhs.mid, p, NEAR);
        let t1 = self.mid.abs().mul(&rhs.rad, RAD_PREC, UP);
        let t2 = rhs.mid.abs().mul(&self.rad, RAD_PREC, UP);
        let t3 = self.rad.mul(&rhs.rad, RAD_PREC, UP);
        let rad = sum_up(&[&t1, &t2, &t3, &round_err(&mid, ord)]);
        Ball { mid, rad }
    }

    /// `self / rhs`. Computed through the interval form so a divisor ball that
    /// contains zero yields the rigorous unbounded result rather than a wrong one.
    pub fn div(&self, rhs: &Ball) -> Ball {
        let p = self.work_precision(rhs);
        Ball::from_interval(&self.to_interval().div(&rhs.to_interval()), p)
    }

    /// Principal square root, via the interval form. A ball dipping below zero is
    /// clamped at zero (as `Interval::sqrt` does).
    pub fn sqrt(&self) -> Ball {
        Ball::from_interval(&self.to_interval().sqrt(), self.precision())
    }

    /// The exponential `e^self`.
    ///
    /// `exp` is monotone increasing, so the enclosure is built from the
    /// endpoints: `exp(lower)` rounded *down* and `exp(upper)` rounded *up*
    /// rigorously bracket the range, and the resulting interval is rebuilt into a
    /// ball (exactly as [`Ball::sqrt`] does through the interval form). Finite for
    /// every finite ball.
    pub fn exp(&self) -> Ball {
        let p = self.precision();
        let lo = self.lower().exp(p, DOWN);
        let hi = self.upper().exp(p, UP);
        Ball::from_interval(&Interval::new(lo, hi, p), p)
    }

    /// The natural logarithm `ln(self)`.
    ///
    /// Domain: the ball must be *strictly positive* — its lower endpoint must be
    /// `> 0`. `ln` is monotone increasing on its domain, so the enclosure is the
    /// endpoint bracket `[ln(lower) (rounded down), ln(upper) (rounded up)]`.
    ///
    /// If the ball is not strictly positive (`lower() ≤ 0`, so it touches or
    /// crosses the branch point at `0`) the result is *indeterminate*: a ball with
    /// a NaN midpoint and `+∞` radius, matching how [`Ball::div`] reports an
    /// unusable enclosure.
    pub fn ln(&self) -> Ball {
        let p = self.precision();
        if self.lower().sign() != Sign::Positive {
            // Not strictly positive: ln is undefined at/below zero. Report an
            // unbounded, indeterminate enclosure rather than a bogus value.
            return Ball::new(Float::nan(p), Float::infinity(RAD_PREC));
        }
        let lo = self.lower().ln(p, DOWN);
        let hi = self.upper().ln(p, UP);
        Ball::from_interval(&Interval::new(lo, hi, p), p)
    }

    /// The inf–sup [`Interval`] enclosing this ball.
    pub fn to_interval(&self) -> Interval {
        Interval::new(self.lower(), self.upper(), self.precision())
    }

    /// The tightest ball enclosing an [`Interval`], midpoint at `precision` bits.
    pub fn from_interval(iv: &Interval, precision: u64) -> Ball {
        let mid = iv.midpoint().round(precision, NEAR);
        // rad ≥ max(mid − lo, hi − mid), rounded up, so [mid ± rad] ⊇ [lo, hi].
        let lo_gap = mid.sub(iv.lower(), RAD_PREC, UP);
        let hi_gap = iv.upper().sub(&mid, RAD_PREC, UP);
        let rad = if lo_gap >= hi_gap { lo_gap } else { hi_gap };
        Ball {
            mid,
            rad: rad.abs().round(RAD_PREC, UP),
        }
    }
}

/// Sum non-negative radii with every step rounded up, so the result is a rigorous
/// upper bound.
fn sum_up(terms: &[&Float]) -> Float {
    let mut acc = rad_zero();
    for t in terms {
        acc = acc.add(t, RAD_PREC, UP);
    }
    acc
}

impl fmt::Display for Ball {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ± {}", self.mid, self.rad)
    }
}

impl fmt::Debug for Ball {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ball({} ± {})", self.mid, self.rad)
    }
}

macro_rules! ball_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr<Ball> for Ball {
            type Output = Ball;
            #[inline]
            fn $m(self, rhs: Ball) -> Ball {
                Ball::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&Ball> for Ball {
            type Output = Ball;
            #[inline]
            fn $m(self, rhs: &Ball) -> Ball {
                Ball::$m(&self, rhs)
            }
        }
        impl core::ops::$tr<Ball> for &Ball {
            type Output = Ball;
            #[inline]
            fn $m(self, rhs: Ball) -> Ball {
                Ball::$m(self, &rhs)
            }
        }
        impl core::ops::$tr<&Ball> for &Ball {
            type Output = Ball;
            #[inline]
            fn $m(self, rhs: &Ball) -> Ball {
                Ball::$m(self, rhs)
            }
        }
        impl core::ops::$atr<Ball> for Ball {
            #[inline]
            fn $am(&mut self, rhs: Ball) {
                *self = Ball::$m(self, &rhs);
            }
        }
        impl core::ops::$atr<&Ball> for Ball {
            #[inline]
            fn $am(&mut self, rhs: &Ball) {
                *self = Ball::$m(self, rhs);
            }
        }
    };
}
ball_binop!(Add, add, AddAssign, add_assign);
ball_binop!(Sub, sub, SubAssign, sub_assign);
ball_binop!(Mul, mul, MulAssign, mul_assign);
ball_binop!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for Ball {
    type Output = Ball;
    #[inline]
    fn neg(self) -> Ball {
        Ball::neg(&self)
    }
}
impl core::ops::Neg for &Ball {
    type Output = Ball;
    #[inline]
    fn neg(self) -> Ball {
        Ball::neg(self)
    }
}
