//! Interval arithmetic with outward rounding (verified computing).
//!
//! An [`Interval`] `[lo, hi]` encloses a set of reals. Every operation rounds
//! the lower endpoint toward −∞ and the upper endpoint toward +∞, so the result
//! provably contains the true result of the operation applied to any members of
//! the operands — the fundamental theorem of interval arithmetic. This gives
//! rigorous error bounds on floating-point computations.

use core::fmt;

use crate::float::{Float, RoundingMode};

const DOWN: RoundingMode = RoundingMode::TowardNegative;
const UP: RoundingMode = RoundingMode::TowardPositive;

/// A closed real interval `[lo, hi]` with [`Float`] endpoints.
#[derive(Clone)]
pub struct Interval {
    lo: Float,
    hi: Float,
    precision: u64,
}

fn fmin(a: Float, b: Float) -> Float {
    if a <= b { a } else { b }
}
fn fmax(a: Float, b: Float) -> Float {
    if a >= b { a } else { b }
}

impl Interval {
    /// Builds `[lo, hi]` at the given working precision. The endpoints are used
    /// as given (assumed already correctly rounded outward).
    pub fn new(lo: Float, hi: Float, precision: u64) -> Interval {
        Interval { lo, hi, precision }
    }

    /// Builds the degenerate interval `[x, x]`.
    pub fn point(x: Float) -> Interval {
        let precision = x.precision();
        Interval {
            lo: x.clone(),
            hi: x,
            precision,
        }
    }

    /// Builds the tightest interval enclosing a rational at `precision` bits.
    pub fn from_rational(r: &crate::rational::Rational, precision: u64) -> Interval {
        Interval {
            lo: Float::from_rational(r, precision, DOWN),
            hi: Float::from_rational(r, precision, UP),
            precision,
        }
    }

    /// Returns the lower endpoint.
    #[inline]
    pub fn lower(&self) -> &Float {
        &self.lo
    }

    /// Returns the upper endpoint.
    #[inline]
    pub fn upper(&self) -> &Float {
        &self.hi
    }

    /// Returns the working precision.
    #[inline]
    pub fn precision(&self) -> u64 {
        self.precision
    }

    /// Returns `true` if the interval contains zero.
    pub fn contains_zero(&self) -> bool {
        let zero = Float::zero(self.precision);
        self.lo <= zero && zero <= self.hi
    }

    /// Returns an upper bound on the width `hi − lo`.
    pub fn width(&self) -> Float {
        self.hi.sub(&self.lo, self.precision, UP)
    }

    /// Returns the midpoint (rounded to nearest).
    pub fn midpoint(&self) -> Float {
        self.lo
            .add(&self.hi, self.precision, RoundingMode::Nearest)
            .mul(
                &Float::from_f64(0.5, self.precision, RoundingMode::Nearest),
                self.precision,
                RoundingMode::Nearest,
            )
    }

    fn prec(&self, rhs: &Interval) -> u64 {
        self.precision.max(rhs.precision)
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Interval) -> Interval {
        let p = self.prec(rhs);
        Interval {
            lo: self.lo.add(&rhs.lo, p, DOWN),
            hi: self.hi.add(&rhs.hi, p, UP),
            precision: p,
        }
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Interval) -> Interval {
        let p = self.prec(rhs);
        Interval {
            lo: self.lo.sub(&rhs.hi, p, DOWN),
            hi: self.hi.sub(&rhs.lo, p, UP),
            precision: p,
        }
    }

    /// Returns `-self`.
    pub fn neg(&self) -> Interval {
        Interval {
            lo: self.hi.neg(),
            hi: self.lo.neg(),
            precision: self.precision,
        }
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Interval) -> Interval {
        let p = self.prec(rhs);
        let ends = [
            (&self.lo, &rhs.lo),
            (&self.lo, &rhs.hi),
            (&self.hi, &rhs.lo),
            (&self.hi, &rhs.hi),
        ];
        let mut lo = self.lo.mul(&rhs.lo, p, DOWN);
        let mut hi = self.lo.mul(&rhs.lo, p, UP);
        for (a, b) in ends {
            lo = fmin(lo, a.mul(b, p, DOWN));
            hi = fmax(hi, a.mul(b, p, UP));
        }
        Interval {
            lo,
            hi,
            precision: p,
        }
    }

    /// Returns `self / rhs`. Panics if `rhs` contains zero.
    pub fn div(&self, rhs: &Interval) -> Interval {
        assert!(
            !rhs.contains_zero(),
            "Interval::div: divisor interval contains zero"
        );
        let p = self.prec(rhs);
        let ends = [
            (&self.lo, &rhs.lo),
            (&self.lo, &rhs.hi),
            (&self.hi, &rhs.lo),
            (&self.hi, &rhs.hi),
        ];
        let mut lo = self.lo.div(&rhs.lo, p, DOWN);
        let mut hi = self.lo.div(&rhs.lo, p, UP);
        for (a, b) in ends {
            lo = fmin(lo, a.div(b, p, DOWN));
            hi = fmax(hi, a.div(b, p, UP));
        }
        Interval {
            lo,
            hi,
            precision: p,
        }
    }

    /// Returns `√self` (requires `lo >= 0`).
    pub fn sqrt(&self) -> Interval {
        Interval {
            lo: self.lo.sqrt(self.precision, DOWN),
            hi: self.hi.sqrt(self.precision, UP),
            precision: self.precision,
        }
    }

    /// Returns the intersection, or `None` if the intervals are disjoint.
    pub fn intersect(&self, rhs: &Interval) -> Option<Interval> {
        let p = self.prec(rhs);
        let lo = fmax(self.lo.clone(), rhs.lo.clone());
        let hi = fmin(self.hi.clone(), rhs.hi.clone());
        if lo <= hi {
            Some(Interval {
                lo,
                hi,
                precision: p,
            })
        } else {
            None
        }
    }

    /// Returns the convex hull (smallest interval containing both).
    pub fn hull(&self, rhs: &Interval) -> Interval {
        Interval {
            lo: fmin(self.lo.clone(), rhs.lo.clone()),
            hi: fmax(self.hi.clone(), rhs.hi.clone()),
            precision: self.prec(rhs),
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.lo, self.hi)
    }
}

impl fmt::Debug for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Interval[{:?}, {:?}]", self.lo, self.hi)
    }
}

macro_rules! iv_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for Interval {
            type Output = Interval;
            #[inline]
            fn $m(self, rhs: Interval) -> Interval {
                Interval::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&Interval> for &Interval {
            type Output = Interval;
            #[inline]
            fn $m(self, rhs: &Interval) -> Interval {
                Interval::$m(self, rhs)
            }
        }
        impl core::ops::$atr for Interval {
            #[inline]
            fn $am(&mut self, rhs: Interval) {
                *self = Interval::$m(self, &rhs);
            }
        }
    };
}

iv_binop!(Add, add, AddAssign, add_assign);
iv_binop!(Sub, sub, SubAssign, sub_assign);
iv_binop!(Mul, mul, MulAssign, mul_assign);
iv_binop!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for Interval {
    type Output = Interval;
    #[inline]
    fn neg(self) -> Interval {
        Interval::neg(&self)
    }
}
