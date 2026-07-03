//! Arbitrary-precision rational numbers (exact `p/q` fractions).
//!
//! [`Rational`] keeps a signed [`Int`] numerator and an unsigned [`Nat`]
//! denominator in canonical form: the denominator is always strictly positive,
//! the fraction is in lowest terms (`gcd(|num|, den) == 1`), and zero is
//! `0/1`. Every value therefore has a unique representation.

use core::cmp::Ordering;
use core::fmt;

use crate::error::{Error, Result};
use crate::int::{Int, Sign};
use crate::nat::Nat;

/// An arbitrary-precision rational number kept in lowest terms.
#[derive(Clone, PartialEq, Eq)]
pub struct Rational {
    /// Signed numerator (carries the sign of the whole value).
    num: Int,
    /// Strictly-positive denominator, coprime with `|num|`.
    den: Nat,
}

impl Rational {
    /// Returns the rational zero (`0/1`).
    #[inline]
    pub fn zero() -> Self {
        Rational {
            num: Int::zero(),
            den: Nat::one(),
        }
    }

    /// Returns the rational one (`1/1`).
    #[inline]
    pub fn one() -> Self {
        Rational {
            num: Int::one(),
            den: Nat::one(),
        }
    }

    /// Builds `num / den`, reducing to lowest terms. Returns
    /// [`Error::DivisionByZero`] if `den` is zero.
    pub fn new(num: Int, den: Int) -> Result<Self> {
        if den.is_zero() {
            return Err(Error::DivisionByZero);
        }
        // The overall sign is the product of the two signs; the stored
        // denominator is always positive.
        let sign = match (num.sign(), den.sign()) {
            (Sign::Zero, _) => Sign::Zero,
            (a, b) if a == b => Sign::Positive,
            _ => Sign::Negative,
        };
        let mut num_mag = num.magnitude().clone();
        let mut den_mag = den.magnitude().clone();
        let g = num_mag.gcd(&den_mag);
        if !g.is_zero() && g != Nat::one() {
            num_mag = num_mag.div_rem(&g).expect("gcd divides numerator").0;
            den_mag = den_mag.div_rem(&g).expect("gcd divides denominator").0;
        }
        Ok(Rational {
            num: Int::from_sign_magnitude(sign, num_mag),
            den: den_mag,
        })
    }

    /// Builds the rational `n/1` from an integer.
    pub fn from_int(n: Int) -> Self {
        Rational {
            num: n,
            den: Nat::one(),
        }
    }

    /// Returns the (signed, reduced) numerator.
    #[inline]
    pub fn numerator(&self) -> &Int {
        &self.num
    }

    /// Returns the (positive, reduced) denominator.
    #[inline]
    pub fn denominator(&self) -> &Nat {
        &self.den
    }

    /// Returns `true` if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    /// Returns `true` if the denominator is one (i.e. the value is an integer).
    #[inline]
    pub fn is_integer(&self) -> bool {
        self.den == Nat::one()
    }

    /// Returns `-self`.
    pub fn neg(&self) -> Rational {
        Rational {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    /// Returns `1/self`, or [`Error::DivisionByZero`] if `self` is zero.
    pub fn recip(&self) -> Result<Rational> {
        Rational::new(Int::from(self.den.clone()), self.num.clone())
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Rational) -> Rational {
        // a/b + c/d = (a·d + c·b) / (b·d), then reduce.
        let ad = self.num.mul(&Int::from(rhs.den.clone()));
        let cb = rhs.num.mul(&Int::from(self.den.clone()));
        let num = ad.add(&cb);
        let den = Int::from(self.den.mul(&rhs.den));
        Rational::new(num, den).expect("product of positive denominators is non-zero")
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Rational) -> Rational {
        self.add(&rhs.neg())
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Rational) -> Rational {
        let num = self.num.mul(&rhs.num);
        let den = Int::from(self.den.mul(&rhs.den));
        Rational::new(num, den).expect("product of positive denominators is non-zero")
    }

    /// Returns `self / rhs`, or [`Error::DivisionByZero`] if `rhs` is zero.
    pub fn div(&self, rhs: &Rational) -> Result<Rational> {
        // a/b ÷ c/d = (a·d) / (b·c).
        let num = self.num.mul(&Int::from(rhs.den.clone()));
        let den = Int::from(self.den.clone()).mul(&rhs.num);
        Rational::new(num, den)
    }
}

impl PartialOrd for Rational {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b vs c/d, with b, d > 0: compare a·d against c·b.
        let lhs = self.num.mul(&Int::from(other.den.clone()));
        let rhs = other.num.mul(&Int::from(self.den.clone()));
        lhs.cmp(&rhs)
    }
}

impl From<Int> for Rational {
    #[inline]
    fn from(n: Int) -> Self {
        Rational::from_int(n)
    }
}

impl fmt::Display for Rational {
    /// Formats as `numerator/denominator`, or just `numerator` when the value is
    /// an integer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_integer() {
            fmt::Display::fmt(&self.num, f)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rational({self})")
    }
}
