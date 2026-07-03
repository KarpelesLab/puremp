//! Arbitrary-precision base-10 floating point.
//!
//! [`Decimal`] stores `coefficient · 10^exponent` with an arbitrary-precision
//! [`Int`] coefficient and an `i64` exponent — the exact analog of Python's
//! `Decimal` / Java's `BigDecimal`. Addition, subtraction, and multiplication
//! are **exact** (base-10 numbers are closed under them); division and explicit
//! rounding take a target number of significant digits (or a fixed exponent,
//! via [`Decimal::quantize`]) and a [`Rounding`] mode.
//!
//! Because trailing zeros are preserved, `"1.50"` and `"1.5"` are distinct
//! representations that compare *equal* — handy for money, where the scale is
//! meaningful.

use core::cmp::Ordering;
use core::fmt;

use alloc::string::{String, ToString};

use crate::int::Int;

/// Rounding modes for [`Decimal`] division and rounding.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Rounding {
    /// Round toward zero (truncate).
    Down,
    /// Round away from zero.
    Up,
    /// Round toward negative infinity.
    Floor,
    /// Round toward positive infinity.
    Ceiling,
    /// Round to nearest; ties away from zero.
    HalfUp,
    /// Round to nearest; ties toward zero.
    HalfDown,
    /// Round to nearest; ties to even (banker's rounding). The default.
    #[default]
    HalfEven,
}

/// An arbitrary-precision base-10 floating-point number, `coefficient · 10^exponent`.
#[derive(Clone)]
pub struct Decimal {
    coeff: Int,
    exp: i64,
}

/// Returns `10^n` as an [`Int`].
fn pow10(n: u32) -> Int {
    Int::from_i64(10).pow(n)
}

/// `10^e` as an `f64` by repeated squaring (no `powi`, so it works in `no_std`).
fn pow10_f64(e: i64) -> f64 {
    let mut base = if e < 0 { 0.1 } else { 10.0 };
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

/// Number of decimal digits in `|n|` (`1` for zero).
fn digit_count(n: &Int) -> usize {
    if n.is_zero() {
        1
    } else {
        n.magnitude().to_string().len()
    }
}

/// Rounds `coeff / 10^drop` to an integer under `mode` (`drop >= 1`).
fn round_div_pow10(coeff: &Int, drop: u32, mode: Rounding) -> Int {
    let divisor = pow10(drop);
    let (q, r) = coeff.div_rem(&divisor).expect("non-zero divisor");
    if r.is_zero() {
        return q;
    }
    let neg = coeff.is_negative();
    let r2 = r.abs().mul_2k(1); // 2·|r|, compared against the divisor for halves
    let increment = match mode {
        Rounding::Down => false,
        Rounding::Up => true,
        Rounding::Floor => neg,
        Rounding::Ceiling => !neg,
        Rounding::HalfUp => r2 >= divisor,
        Rounding::HalfDown => r2 > divisor,
        Rounding::HalfEven => match r2.cmp(&divisor) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => q.is_odd(),
        },
    };
    if increment {
        q.add(&Int::from_i64(coeff.signum() as i64))
    } else {
        q
    }
}

impl Decimal {
    /// The value zero.
    pub fn zero() -> Decimal {
        Decimal {
            coeff: Int::ZERO,
            exp: 0,
        }
    }

    /// The value one.
    pub fn one() -> Decimal {
        Decimal {
            coeff: Int::ONE,
            exp: 0,
        }
    }

    /// Builds `coefficient · 10^exponent` (kept as given; not normalized).
    pub fn new(coefficient: Int, exponent: i64) -> Decimal {
        Decimal {
            coeff: coefficient,
            exp: exponent,
        }
    }

    /// Builds the decimal value of an integer.
    #[inline]
    pub fn from_int(n: Int) -> Decimal {
        Decimal { coeff: n, exp: 0 }
    }

    /// Returns the coefficient.
    #[inline]
    pub fn coefficient(&self) -> &Int {
        &self.coeff
    }

    /// Returns the base-10 exponent.
    #[inline]
    pub fn exponent(&self) -> i64 {
        self.exp
    }

    /// Returns `true` if this value is zero (any exponent).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeff.is_zero()
    }

    /// Returns `-1`, `0`, or `1` according to the sign.
    #[inline]
    pub fn signum(&self) -> i32 {
        self.coeff.signum()
    }

    /// Returns `-self`.
    pub fn neg(&self) -> Decimal {
        Decimal {
            coeff: self.coeff.neg(),
            exp: self.exp,
        }
    }

    /// Returns `|self|`.
    pub fn abs(&self) -> Decimal {
        Decimal {
            coeff: self.coeff.abs(),
            exp: self.exp,
        }
    }

    /// Returns `self · 10^n` (exact; adjusts only the exponent).
    pub fn scaleb(&self, n: i64) -> Decimal {
        Decimal {
            coeff: self.coeff.clone(),
            exp: self.exp + n,
        }
    }

    /// Removes trailing zeros from the coefficient (raising the exponent), giving
    /// the minimal-length equal representation (`"1.500"` → `"1.5"`).
    pub fn normalized(&self) -> Decimal {
        if self.coeff.is_zero() {
            return Decimal::zero();
        }
        let mut coeff = self.coeff.clone();
        let mut exp = self.exp;
        let ten = Int::from_i64(10);
        loop {
            let (q, r) = coeff.div_rem(&ten).unwrap();
            if !r.is_zero() {
                break;
            }
            coeff = q;
            exp += 1;
        }
        Decimal { coeff, exp }
    }

    /// Aligns two values to the common (smaller) exponent, returning the two
    /// coefficients at that scale and the exponent.
    fn aligned(&self, other: &Decimal) -> (Int, Int, i64) {
        let emin = self.exp.min(other.exp);
        let a = self.coeff.mul(&pow10((self.exp - emin) as u32));
        let b = other.coeff.mul(&pow10((other.exp - emin) as u32));
        (a, b, emin)
    }

    /// Returns `self + rhs` (exact).
    pub fn add(&self, rhs: &Decimal) -> Decimal {
        let (a, b, emin) = self.aligned(rhs);
        Decimal {
            coeff: a.add(&b),
            exp: emin,
        }
    }

    /// Returns `self - rhs` (exact).
    pub fn sub(&self, rhs: &Decimal) -> Decimal {
        self.add(&rhs.neg())
    }

    /// Returns `self · rhs` (exact).
    pub fn mul(&self, rhs: &Decimal) -> Decimal {
        Decimal {
            coeff: self.coeff.mul(&rhs.coeff),
            exp: self.exp + rhs.exp,
        }
    }

    /// Rounds to at most `sig_digits` significant digits under `mode`
    /// (`sig_digits >= 1`).
    pub fn round_to_digits(&self, sig_digits: u32, mode: Rounding) -> Decimal {
        if self.coeff.is_zero() {
            return self.clone();
        }
        let digits = digit_count(&self.coeff) as i64;
        let drop = digits - sig_digits.max(1) as i64;
        if drop <= 0 {
            return self.clone();
        }
        let coeff = round_div_pow10(&self.coeff, drop as u32, mode);
        Decimal {
            coeff,
            exp: self.exp + drop,
        }
    }

    /// Rounds to the fixed exponent `target_exp` under `mode` (like Python's
    /// `quantize`): scaling up is exact, scaling down rounds.
    pub fn quantize(&self, target_exp: i64, mode: Rounding) -> Decimal {
        if target_exp <= self.exp {
            // Finer or equal scale: multiply out exactly.
            Decimal {
                coeff: self.coeff.mul(&pow10((self.exp - target_exp) as u32)),
                exp: target_exp,
            }
        } else {
            let drop = (target_exp - self.exp) as u32;
            Decimal {
                coeff: round_div_pow10(&self.coeff, drop, mode),
                exp: target_exp,
            }
        }
    }

    /// Returns `self / rhs` rounded to `sig_digits` significant digits under
    /// `mode`. Panics if `rhs` is zero.
    pub fn div(&self, rhs: &Decimal, sig_digits: u32, mode: Rounding) -> Decimal {
        assert!(!rhs.is_zero(), "Decimal::div: division by zero");
        if self.is_zero() {
            return Decimal::zero();
        }
        let da = digit_count(&self.coeff) as i64;
        let db = digit_count(&rhs.coeff) as i64;
        // Scale the numerator so the quotient has a few more than sig_digits.
        let work = (sig_digits as i64 + 4 + db - da).max(1);
        let scaled = self.coeff.mul(&pow10(work as u32));
        let (mut q, r) = scaled.div_rem(&rhs.coeff).expect("non-zero divisor");
        // Sticky: keep a nonzero low digit when the division was inexact.
        if !r.is_zero() && q.rem_euclid(&Int::from_i64(10)).is_zero() {
            q = q.add(&Int::from_i64(q.signum() as i64));
        }
        Decimal {
            coeff: q,
            exp: self.exp - rhs.exp - work,
        }
        .round_to_digits(sig_digits, mode)
        // Strip the trailing zeros the scaling introduced (e.g. 1/8 → 0.125).
        .normalized()
    }

    /// Returns the value as the nearest `f64` (best-effort).
    pub fn to_f64(&self) -> f64 {
        self.coeff.to_f64() * pow10_f64(self.exp)
    }
}

impl PartialEq for Decimal {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Decimal {}

impl PartialOrd for Decimal {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Decimal {
    fn cmp(&self, other: &Self) -> Ordering {
        let (a, b, _) = self.aligned(other);
        a.cmp(&b)
    }
}

impl Default for Decimal {
    #[inline]
    fn default() -> Decimal {
        Decimal::zero()
    }
}

impl From<Int> for Decimal {
    #[inline]
    fn from(n: Int) -> Decimal {
        Decimal::from_int(n)
    }
}

impl From<i64> for Decimal {
    #[inline]
    fn from(v: i64) -> Decimal {
        Decimal::from_int(Int::from_i64(v))
    }
}

impl core::str::FromStr for Decimal {
    type Err = crate::error::Error;

    /// Parses `"123"`, `"-0.001"`, `"1.5"`, or scientific `"1.5e10"` / `"2E-8"`.
    fn from_str(s: &str) -> crate::error::Result<Decimal> {
        use crate::error::Error;
        let s = s.trim();
        // Split off an optional exponent.
        let (mantissa, exp_part) = match s.find(['e', 'E']) {
            Some(i) => (&s[..i], &s[i + 1..]),
            None => (s, ""),
        };
        let mut exp: i64 = if exp_part.is_empty() {
            0
        } else {
            exp_part.parse().map_err(|_| Error::Parse)?
        };
        let (neg, body) = match mantissa.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, mantissa.strip_prefix('+').unwrap_or(mantissa)),
        };
        // Integer and fractional digit runs.
        let (int_part, frac_part) = match body.find('.') {
            Some(i) => (&body[..i], &body[i + 1..]),
            None => (body, ""),
        };
        if int_part.is_empty() && frac_part.is_empty() {
            return Err(Error::Parse);
        }
        let mut digits = String::with_capacity(int_part.len() + frac_part.len());
        digits.push_str(int_part);
        digits.push_str(frac_part);
        exp -= frac_part.len() as i64;
        let mag: crate::nat::Nat = if digits.is_empty() {
            crate::nat::Nat::zero()
        } else {
            digits.parse().map_err(|_| Error::Parse)?
        };
        let coeff = Int::from_sign_magnitude(
            if neg {
                crate::int::Sign::Negative
            } else {
                crate::int::Sign::Positive
            },
            mag,
        );
        Ok(Decimal::new(coeff, exp))
    }
}

impl fmt::Display for Decimal {
    /// Formats in plain (non-scientific) decimal notation, preserving the scale.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coeff.is_zero() {
            return f.write_str("0");
        }
        if self.coeff.is_negative() {
            f.write_str("-")?;
        }
        let digits = self.coeff.magnitude().to_string();
        if self.exp >= 0 {
            f.write_str(&digits)?;
            for _ in 0..self.exp {
                f.write_str("0")?;
            }
            Ok(())
        } else {
            let k = (-self.exp) as usize;
            if digits.len() > k {
                let point = digits.len() - k;
                f.write_str(&digits[..point])?;
                f.write_str(".")?;
                f.write_str(&digits[point..])
            } else {
                f.write_str("0.")?;
                for _ in 0..k - digits.len() {
                    f.write_str("0")?;
                }
                f.write_str(&digits)
            }
        }
    }
}

impl fmt::Debug for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Decimal({} · 10^{})", self.coeff, self.exp)
    }
}

macro_rules! dec_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for Decimal {
            type Output = Decimal;
            #[inline]
            fn $m(self, rhs: Decimal) -> Decimal {
                Decimal::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&Decimal> for &Decimal {
            type Output = Decimal;
            #[inline]
            fn $m(self, rhs: &Decimal) -> Decimal {
                Decimal::$m(self, rhs)
            }
        }
        impl core::ops::$atr for Decimal {
            #[inline]
            fn $am(&mut self, rhs: Decimal) {
                *self = Decimal::$m(self, &rhs);
            }
        }
    };
}

dec_binop!(Add, add, AddAssign, add_assign);
dec_binop!(Sub, sub, SubAssign, sub_assign);
dec_binop!(Mul, mul, MulAssign, mul_assign);

impl core::ops::Neg for Decimal {
    type Output = Decimal;
    #[inline]
    fn neg(self) -> Decimal {
        Decimal::neg(&self)
    }
}

// --- conversions to the exact rational, when available ---

#[cfg(feature = "rational")]
impl Decimal {
    /// Returns the exact value as a [`Rational`](crate::rational::Rational).
    pub fn to_rational(&self) -> crate::rational::Rational {
        use crate::rational::Rational;
        if self.exp >= 0 {
            Rational::from_integer(self.coeff.mul(&pow10(self.exp as u32)))
        } else {
            Rational::new(self.coeff.clone(), pow10((-self.exp) as u32))
        }
    }
}
