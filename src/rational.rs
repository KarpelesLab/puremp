//! Arbitrary-precision rational numbers (exact `p/q` fractions).
//!
//! [`Rational`] keeps a signed numerator and a strictly-positive denominator (both
//! [`Int`]) in canonical form: `den > 0`, `gcd(|num|, den) == 1`, and integers
//! have `den == 1` (so zero is `0/1`). Every value therefore has a unique
//! representation, which makes the derived [`PartialEq`]/[`Eq`]/[`Hash`] correct.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::int::{Int, Sign};
use crate::nat::Nat;

/// An arbitrary-precision rational number kept in lowest terms.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Rational {
    /// Signed numerator (carries the sign of the whole value).
    num: Int,
    /// Strictly-positive denominator, coprime with `|num|`.
    den: Int,
}

impl Rational {
    /// The value `0` (`0/1`).
    pub const ZERO: Rational = Rational {
        num: Int::ZERO,
        den: Int::ONE,
    };
    /// The value `1` (`1/1`).
    pub const ONE: Rational = Rational {
        num: Int::ONE,
        den: Int::ONE,
    };
    /// The value `-1` (`-1/1`).
    pub const MINUS_ONE: Rational = Rational {
        num: Int::MINUS_ONE,
        den: Int::ONE,
    };

    /// Reduces `num/den` (with `den != 0`) to canonical form.
    fn normalize(mut num: Int, mut den: Int) -> Rational {
        debug_assert!(!den.is_zero(), "Rational::normalize with zero denominator");
        if den.is_negative() {
            num = num.neg();
            den = den.neg();
        }
        if num.is_zero() {
            return Rational::ZERO;
        }
        let g = Int::from(num.magnitude().gcd(&den.magnitude()));
        if !g.is_one() {
            num = num.div_exact(&g);
            den = den.div_exact(&g);
        }
        Rational { num, den }
    }

    /// Builds `num / den`, reduced to lowest terms. Panics if `den` is zero.
    pub fn new(num: Int, den: Int) -> Rational {
        assert!(!den.is_zero(), "Rational::new: zero denominator");
        Rational::normalize(num, den)
    }

    /// Builds `num / den`, reduced; returns `None` if `den` is zero.
    pub fn checked_new(num: Int, den: Int) -> Option<Rational> {
        if den.is_zero() {
            None
        } else {
            Some(Rational::normalize(num, den))
        }
    }

    /// Builds the rational `n/1` from an integer.
    #[inline]
    pub fn from_integer(n: Int) -> Rational {
        Rational {
            num: n,
            den: Int::ONE,
        }
    }

    /// Builds `2^k`; `k` may be negative.
    pub fn power_of_two(k: i32) -> Rational {
        if k >= 0 {
            Rational {
                num: Int::ONE.mul_2k(k as u32),
                den: Int::ONE,
            }
        } else {
            Rational {
                num: Int::ONE,
                den: Int::ONE.mul_2k((-(k as i64)) as u32),
            }
        }
    }

    /// Returns the (signed, reduced) numerator.
    #[inline]
    pub fn numerator(&self) -> &Int {
        &self.num
    }

    /// Returns the (positive, reduced) denominator.
    #[inline]
    pub fn denominator(&self) -> &Int {
        &self.den
    }

    // --- predicates ---

    /// Returns `true` if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    /// Returns `true` if this value is one.
    #[inline]
    pub fn is_one(&self) -> bool {
        self.num.is_one() && self.den.is_one()
    }

    /// Returns `true` if this value is minus one.
    #[inline]
    pub fn is_minus_one(&self) -> bool {
        self.num.is_minus_one() && self.den.is_one()
    }

    /// Returns `true` if this value is strictly positive.
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.num.is_positive()
    }

    /// Returns `true` if this value is strictly negative.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.num.is_negative()
    }

    /// Returns `true` if the denominator is one (i.e. the value is an integer).
    #[inline]
    pub fn is_integer(&self) -> bool {
        self.den.is_one()
    }

    /// Returns `-1`, `0`, or `1` according to the sign.
    #[inline]
    pub fn signum(&self) -> i32 {
        self.num.signum()
    }

    // --- arithmetic ---

    /// Returns `-self`.
    pub fn neg(&self) -> Rational {
        Rational {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    /// Returns `|self|`.
    pub fn abs(&self) -> Rational {
        Rational {
            num: self.num.abs(),
            den: self.den.clone(),
        }
    }

    /// Returns `1/self`. Panics if `self` is zero.
    pub fn recip(&self) -> Rational {
        assert!(!self.is_zero(), "Rational::recip: reciprocal of zero");
        Rational::normalize(self.den.clone(), self.num.clone())
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Rational) -> Rational {
        let num = self.num.mul(&rhs.den).add(&rhs.num.mul(&self.den));
        let den = self.den.mul(&rhs.den);
        Rational::normalize(num, den)
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Rational) -> Rational {
        self.add(&rhs.neg())
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Rational) -> Rational {
        Rational::normalize(self.num.mul(&rhs.num), self.den.mul(&rhs.den))
    }

    /// Returns `self / rhs`. Panics if `rhs` is zero.
    pub fn div(&self, rhs: &Rational) -> Rational {
        assert!(!rhs.is_zero(), "Rational::div: division by zero");
        Rational::normalize(self.num.mul(&rhs.den), self.den.mul(&rhs.num))
    }

    /// Returns `self` raised to `n` (negative `n` via the reciprocal). Panics on
    /// `0` to a negative power.
    pub fn pow(&self, n: i32) -> Rational {
        if n >= 0 {
            Rational::normalize(self.num.pow(n as u32), self.den.pow(n as u32))
        } else {
            assert!(!self.is_zero(), "Rational::pow: zero to a negative power");
            let m = n.unsigned_abs();
            Rational::normalize(self.den.pow(m), self.num.pow(m))
        }
    }

    /// Fused multiply-add: `self += a · b`.
    pub fn addmul(&mut self, a: &Rational, b: &Rational) {
        *self = self.add(&a.mul(b));
    }

    /// Fused multiply-subtract: `self -= a · b`.
    pub fn submul(&mut self, a: &Rational, b: &Rational) {
        *self = self.sub(&a.mul(b));
    }

    // --- rounding to Int ---

    /// Returns the greatest integer `≤ self`.
    #[inline]
    pub fn floor(&self) -> Int {
        self.num.div_floor(&self.den)
    }

    /// Returns the least integer `≥ self`.
    pub fn ceil(&self) -> Int {
        // ceil(a/b) = -floor(-a/b) for b > 0.
        self.num.neg().div_floor(&self.den).neg()
    }

    /// Returns `self` truncated toward zero.
    #[inline]
    pub fn trunc(&self) -> Int {
        self.num.div_trunc(&self.den)
    }

    /// Returns `Some(n)` if this value is the integer `n`, else `None`.
    pub fn to_integer(&self) -> Option<Int> {
        self.is_integer().then(|| self.num.clone())
    }

    // --- integer division of rationals ---

    /// Returns `⌊self / b⌋` as an integer. Panics if `b` is zero.
    pub fn div_floor(&self, b: &Rational) -> Int {
        self.div(b).floor()
    }

    /// Returns `self / b` truncated toward zero, as an integer. Panics if `b` is
    /// zero.
    pub fn div_trunc(&self, b: &Rational) -> Int {
        self.div(b).trunc()
    }

    /// Returns the Euclidean remainder `self - b·⌊self/b⌋`-style value in
    /// `[0, |b|)`. Panics if `b` is zero.
    pub fn rem_euclid(&self, b: &Rational) -> Rational {
        let n = self.div(b);
        // Pick the quotient that drives the remainder non-negative.
        let q = if b.is_negative() { n.ceil() } else { n.floor() };
        self.sub(&b.mul(&Rational::from_integer(q)))
    }

    // --- bounded conversion ---

    /// Returns `true` if this value is an integer that fits in an `i64`.
    pub fn fits_i64(&self) -> bool {
        self.is_integer() && self.num.fits_i64()
    }

    /// Returns the value as an `i64` if it is an integer that fits.
    pub fn to_i64(&self) -> Option<i64> {
        self.is_integer().then(|| self.num.to_i64()).flatten()
    }

    /// Returns the value as the nearest `f64` (best-effort).
    pub fn to_f64(&self) -> f64 {
        self.num.to_f64() / self.den.to_f64()
    }

    /// Writes the value as a decimal expansion with `precision` fractional
    /// digits. If `truncate` is true the last digit is chopped; otherwise the
    /// expansion is rounded half-up.
    pub fn write_decimal(
        &self,
        out: &mut impl fmt::Write,
        precision: u32,
        truncate: bool,
    ) -> fmt::Result {
        let ten = Nat::from_u64(10);
        let d = self.den.magnitude();
        let (mut ip, mut rem) = self
            .num
            .magnitude()
            .div_rem(&d)
            .expect("denominator is non-zero");
        let mut frac: Vec<u8> = Vec::with_capacity(precision as usize);
        for _ in 0..precision {
            rem = rem.mul(&ten);
            let (digit, r) = rem.div_rem(&d).expect("denominator is non-zero");
            frac.push(b'0' + digit.to_u64().unwrap_or(0) as u8);
            rem = r;
        }
        if !truncate && rem.mul(&Nat::from_u64(2)).cmp(&d) != Ordering::Less {
            // Round half-up: propagate a carry through the fractional digits and,
            // if it escapes the top, into the integer part.
            let mut carry = true;
            for c in frac.iter_mut().rev() {
                if !carry {
                    break;
                }
                if *c == b'9' {
                    *c = b'0';
                } else {
                    *c += 1;
                    carry = false;
                }
            }
            if carry {
                ip = ip.add(&Nat::one());
            }
        }
        if self.is_negative() {
            out.write_str("-")?;
        }
        write!(out, "{ip}")?;
        if precision > 0 {
            out.write_str(".")?;
            out.write_str(core::str::from_utf8(&frac).unwrap_or(""))?;
        }
        Ok(())
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
        self.num.mul(&other.den).cmp(&other.num.mul(&self.den))
    }
}

impl Default for Rational {
    #[inline]
    fn default() -> Rational {
        Rational::ZERO
    }
}

impl From<Int> for Rational {
    #[inline]
    fn from(n: Int) -> Rational {
        Rational::from_integer(n)
    }
}

impl From<i64> for Rational {
    #[inline]
    fn from(v: i64) -> Rational {
        Rational::from_integer(Int::from_i64(v))
    }
}

impl FromStr for Rational {
    type Err = Error;

    /// Parses `"3"`, `"-3/4"`, or a decimal like `"1.5"`.
    fn from_str(s: &str) -> Result<Self> {
        let (neg, body) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        let sign = if neg { Sign::Negative } else { Sign::Positive };

        if let Some(i) = body.find('/') {
            let num = Nat::from_str(&body[..i])?;
            let den = Nat::from_str(&body[i + 1..])?;
            let den = Int::from(den);
            return Rational::checked_new(Int::from_sign_magnitude(sign, num), den)
                .ok_or(Error::DivisionByZero);
        }

        if let Some(i) = body.find('.') {
            let ip = &body[..i];
            let fp = &body[i + 1..];
            if ip.is_empty() && fp.is_empty() {
                return Err(Error::Parse);
            }
            let mut digits = String::with_capacity(ip.len() + fp.len());
            digits.push_str(ip);
            digits.push_str(fp);
            let mag = if digits.is_empty() {
                Nat::zero()
            } else {
                Nat::from_str(&digits)?
            };
            let den = Nat::from_u64(10).pow(fp.len() as u32);
            return Ok(Rational::normalize(
                Int::from_sign_magnitude(sign, mag),
                Int::from(den),
            ));
        }

        Ok(Rational::from_integer(Int::from_sign_magnitude(
            sign,
            Nat::from_str(body)?,
        )))
    }
}

impl fmt::Display for Rational {
    /// Formats as `numerator/denominator`, or just `numerator` when integer.
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

macro_rules! rat_binops {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr<Rational> for Rational {
            type Output = Rational;
            #[inline]
            fn $m(self, rhs: Rational) -> Rational {
                Rational::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&Rational> for &Rational {
            type Output = Rational;
            #[inline]
            fn $m(self, rhs: &Rational) -> Rational {
                Rational::$m(self, rhs)
            }
        }
        impl core::ops::$atr<Rational> for Rational {
            #[inline]
            fn $am(&mut self, rhs: Rational) {
                *self = Rational::$m(self, &rhs);
            }
        }
        impl core::ops::$atr<&Rational> for Rational {
            #[inline]
            fn $am(&mut self, rhs: &Rational) {
                *self = Rational::$m(self, rhs);
            }
        }
    };
}

rat_binops!(Add, add, AddAssign, add_assign);
rat_binops!(Sub, sub, SubAssign, sub_assign);
rat_binops!(Mul, mul, MulAssign, mul_assign);
rat_binops!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for Rational {
    type Output = Rational;
    #[inline]
    fn neg(self) -> Rational {
        Rational::neg(&self)
    }
}

impl core::ops::Neg for &Rational {
    type Output = Rational;
    #[inline]
    fn neg(self) -> Rational {
        Rational::neg(self)
    }
}
