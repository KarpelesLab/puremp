//! Exact dyadic rationals — numbers of the form `n · 2^-k`, the rationals whose
//! denominator is a power of two.
//!
//! [`Dyadic`] is exact and closed under addition, subtraction, multiplication,
//! and scaling by powers of two (division in general is not — `1/3` is not
//! dyadic). Every dyadic value has a *terminating* decimal expansion, which
//! [`Dyadic`]'s [`Display`](core::fmt::Display) prints exactly.
//!
//! A value is stored as `numerator · 2^-scale` in canonical form: the numerator
//! is odd (so the representation is unique), or the value is zero
//! (`numerator == 0`, `scale == 0`). The `scale` is the number of fractional
//! binary digits; it is signed, so an even integer such as `8 = 1 · 2^-(-3)` has
//! a negative scale after normalization.

use core::cmp::Ordering;
use core::fmt;

use alloc::string::ToString;

use crate::int::{Int, Sign};
use crate::nat::Nat;

/// An exact dyadic rational, `numerator · 2^-scale`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Dyadic {
    numerator: Int,
    scale: i64,
}

impl Dyadic {
    /// The value zero.
    pub fn zero() -> Dyadic {
        Dyadic {
            numerator: Int::ZERO,
            scale: 0,
        }
    }

    /// The value one.
    pub fn one() -> Dyadic {
        Dyadic {
            numerator: Int::ONE,
            scale: 0,
        }
    }

    /// Builds `n · 2^-k`, canonicalizing to an odd numerator (so a positive `k`
    /// scales the value *down*, matching the `n·2⁻ᵏ` convention).
    pub fn new(n: Int, k: i64) -> Dyadic {
        if n.is_zero() {
            return Dyadic::zero();
        }
        let t = n.trailing_zeros();
        if t > 0 {
            // n = m·2^t (m odd): n·2^-k = m·2^-(k - t).
            Dyadic {
                numerator: n.div_2k_trunc(t),
                scale: k - t as i64,
            }
        } else {
            Dyadic {
                numerator: n,
                scale: k,
            }
        }
    }

    /// Builds the dyadic value of an integer.
    #[inline]
    pub fn from_int(n: Int) -> Dyadic {
        Dyadic::new(n, 0)
    }

    /// Returns the (odd, or zero) numerator.
    #[inline]
    pub fn numerator(&self) -> &Int {
        &self.numerator
    }

    /// Returns the scale `k`, so that the value equals `numerator · 2^-k`. A
    /// positive scale is the count of fractional binary digits.
    #[inline]
    pub fn scale(&self) -> i64 {
        self.scale
    }

    /// Returns `true` if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Returns `true` if this value is an integer (`scale <= 0`).
    #[inline]
    pub fn is_integer(&self) -> bool {
        self.is_zero() || self.scale <= 0
    }

    /// Returns the sign of this value.
    #[inline]
    pub fn sign(&self) -> Sign {
        self.numerator.sign()
    }

    /// Returns `-self`.
    pub fn neg(&self) -> Dyadic {
        Dyadic {
            numerator: self.numerator.neg(),
            scale: self.scale,
        }
    }

    /// Returns `|self|`.
    pub fn abs(&self) -> Dyadic {
        Dyadic {
            numerator: self.numerator.abs(),
            scale: self.scale,
        }
    }

    /// Returns `self · 2^k` (exact; `k` may be negative).
    pub fn mul_2k(&self, k: i64) -> Dyadic {
        if self.is_zero() {
            return Dyadic::zero();
        }
        // value · 2^k = numerator · 2^-(scale - k).
        Dyadic {
            numerator: self.numerator.clone(),
            scale: self.scale - k,
        }
    }

    /// Aligns two values to a common (finer) scale, returning their numerators
    /// at that scale and the scale.
    fn aligned(&self, other: &Dyadic) -> (Int, Int, i64) {
        let smax = self.scale.max(other.scale);
        let a = self.numerator.mul_2k((smax - self.scale) as u32);
        let b = other.numerator.mul_2k((smax - other.scale) as u32);
        (a, b, smax)
    }

    /// Returns `self + rhs` (exact).
    pub fn add(&self, rhs: &Dyadic) -> Dyadic {
        if self.is_zero() {
            return rhs.clone();
        }
        if rhs.is_zero() {
            return self.clone();
        }
        let (a, b, smax) = self.aligned(rhs);
        Dyadic::new(a.add(&b), smax)
    }

    /// Returns `self - rhs` (exact).
    pub fn sub(&self, rhs: &Dyadic) -> Dyadic {
        self.add(&rhs.neg())
    }

    /// Returns `self · rhs` (exact).
    pub fn mul(&self, rhs: &Dyadic) -> Dyadic {
        Dyadic::new(self.numerator.mul(&rhs.numerator), self.scale + rhs.scale)
    }

    /// Returns `self` raised to `exp` (exact).
    pub fn pow(&self, exp: u32) -> Dyadic {
        if exp == 0 {
            return Dyadic::one();
        }
        Dyadic::new(self.numerator.pow(exp), self.scale * exp as i64)
    }

    /// Returns the greatest integer `<= self`.
    pub fn floor(&self) -> Int {
        if self.scale <= 0 {
            self.numerator.mul_2k((-self.scale) as u32)
        } else {
            self.numerator
                .div_floor(&Int::ONE.mul_2k(self.scale as u32))
        }
    }

    /// Returns `self` truncated toward zero as an integer.
    pub fn trunc(&self) -> Int {
        if self.scale <= 0 {
            self.numerator.mul_2k((-self.scale) as u32)
        } else {
            self.numerator.div_2k_trunc(self.scale as u32)
        }
    }
}

impl PartialOrd for Dyadic {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dyadic {
    fn cmp(&self, other: &Self) -> Ordering {
        let (a, b, _) = self.aligned(other);
        a.cmp(&b)
    }
}

impl Default for Dyadic {
    #[inline]
    fn default() -> Dyadic {
        Dyadic::zero()
    }
}

impl From<Int> for Dyadic {
    #[inline]
    fn from(n: Int) -> Dyadic {
        Dyadic::from_int(n)
    }
}

impl From<i64> for Dyadic {
    #[inline]
    fn from(v: i64) -> Dyadic {
        Dyadic::from_int(Int::from_i64(v))
    }
}

impl fmt::Display for Dyadic {
    /// Prints the exact (terminating) decimal expansion.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        if self.numerator.is_negative() {
            f.write_str("-")?;
        }
        let mag = self.numerator.magnitude();
        if self.scale <= 0 {
            // Integer: |numerator| << (-scale).
            return fmt::Display::fmt(&mag.shl((-self.scale) as u64), f);
        }
        // value = |numerator| / 2^k = |numerator|·5^k / 10^k; place the point k
        // digits from the right.
        let k = self.scale as u32;
        let scaled = mag.mul(&Nat::from_u64(5).pow(k));
        let digits = scaled.to_string();
        let k = k as usize;
        if digits.len() <= k {
            f.write_str("0.")?;
            for _ in 0..k - digits.len() {
                f.write_str("0")?;
            }
            f.write_str(&digits)
        } else {
            let point = digits.len() - k;
            f.write_str(&digits[..point])?;
            f.write_str(".")?;
            f.write_str(&digits[point..])
        }
    }
}

impl fmt::Debug for Dyadic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dyadic({} · 2^-{})", self.numerator, self.scale)
    }
}

macro_rules! dyadic_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for Dyadic {
            type Output = Dyadic;
            #[inline]
            fn $m(self, rhs: Dyadic) -> Dyadic {
                Dyadic::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&Dyadic> for &Dyadic {
            type Output = Dyadic;
            #[inline]
            fn $m(self, rhs: &Dyadic) -> Dyadic {
                Dyadic::$m(self, rhs)
            }
        }
        impl core::ops::$atr for Dyadic {
            #[inline]
            fn $am(&mut self, rhs: Dyadic) {
                *self = Dyadic::$m(self, &rhs);
            }
        }
    };
}

dyadic_binop!(Add, add, AddAssign, add_assign);
dyadic_binop!(Sub, sub, SubAssign, sub_assign);
dyadic_binop!(Mul, mul, MulAssign, mul_assign);

impl core::ops::Neg for Dyadic {
    type Output = Dyadic;
    #[inline]
    fn neg(self) -> Dyadic {
        Dyadic::neg(&self)
    }
}

// --- conversions to the other numeric types ---

#[cfg(feature = "rational")]
impl Dyadic {
    /// Returns the exact value as a [`Rational`](crate::rational::Rational).
    pub fn to_rational(&self) -> crate::rational::Rational {
        use crate::rational::Rational;
        if self.scale <= 0 {
            Rational::from_integer(self.numerator.mul_2k((-self.scale) as u32))
        } else {
            Rational::new(self.numerator.clone(), Int::ONE.mul_2k(self.scale as u32))
        }
    }

    /// Converts a [`Rational`](crate::rational::Rational) to a [`Dyadic`], or
    /// `None` if its denominator is not a power of two.
    pub fn try_from_rational(r: &crate::rational::Rational) -> Option<Dyadic> {
        let k = r.denominator().is_power_of_two()?;
        Some(Dyadic::new(r.numerator().clone(), k as i64))
    }
}

#[cfg(feature = "float")]
impl Dyadic {
    /// Rounds this value to a [`Float`](crate::float::Float) at `precision` bits.
    pub fn to_float(
        &self,
        precision: u64,
        mode: crate::float::RoundingMode,
    ) -> crate::float::Float {
        // Exact rational → correctly rounded float.
        crate::float::Float::from_rational(&self.to_rational(), precision, mode)
    }

    /// Converts a finite [`Float`](crate::float::Float) to an exact [`Dyadic`],
    /// or `None` for NaN / ±∞ (finite floats are always exactly dyadic).
    pub fn from_float(f: &crate::float::Float) -> Option<Dyadic> {
        if f.is_zero() {
            return Some(Dyadic::zero());
        }
        let sig = f.significand()?; // None for NaN/±∞
        let exp = f.exponent()?; // value = ±sig · 2^exp = ±sig · 2^-(-exp)
        let numerator = Int::from_sign_magnitude(f.sign(), sig.clone());
        Some(Dyadic::new(numerator, -exp))
    }
}

impl core::str::FromStr for Dyadic {
    type Err = crate::error::Error;

    /// Parses a decimal (`"3"`, `"-1.5"`, `"0.25"`); returns
    /// [`Error::Parse`](crate::error::Error::Parse) if the value is not dyadic
    /// (its reduced denominator is not a power of two).
    #[cfg(feature = "rational")]
    fn from_str(s: &str) -> crate::error::Result<Dyadic> {
        let r: crate::rational::Rational = s.parse()?;
        Dyadic::try_from_rational(&r).ok_or(crate::error::Error::Parse)
    }

    /// Without the `rational` feature, only plain integers parse.
    #[cfg(not(feature = "rational"))]
    fn from_str(s: &str) -> crate::error::Result<Dyadic> {
        Ok(Dyadic::from_int(s.parse()?))
    }
}
