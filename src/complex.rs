//! Generic complex numbers `re + im·i`.
//!
//! [`Complex`] is parameterized over the component type, so it composes with any
//! of the crate's numeric types that expose the relevant operators:
//! `Complex<Int>` (Gaussian integers), `Complex<Rational>`, `Complex<Dyadic>`,
//! `Complex<Decimal>`, and `Complex<FixedFloat>`. (The precision-carrying
//! [`Float`](crate::float::Float) has no plain operators, so use `FixedFloat`
//! for complex floats.)
//!
//! Addition, subtraction, multiplication, negation, and conjugation need only
//! `+ - *` on the component type; complex division additionally needs `/`, so it
//! is available for field-like components (`Rational`, `Decimal`, `FixedFloat`)
//! but not for `Int`.

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A complex number with components of type `T`: `re + im·i`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Complex<T> {
    /// The real part.
    pub re: T,
    /// The imaginary part.
    pub im: T,
}

impl<T> Complex<T> {
    /// Builds `re + im·i`.
    #[inline]
    pub const fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }
}

impl<T: Default> Complex<T> {
    /// Builds a real value `re + 0·i`.
    #[inline]
    pub fn from_real(re: T) -> Complex<T> {
        Complex {
            re,
            im: T::default(),
        }
    }

    /// The imaginary unit `i` (requires a `One`-like value); built from `one`.
    #[inline]
    pub fn imaginary(one: T) -> Complex<T> {
        Complex {
            re: T::default(),
            im: one,
        }
    }
}

impl<T: Default + PartialEq> Complex<T> {
    /// Returns `true` if both components are zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.re == T::default() && self.im == T::default()
    }

    /// Returns `true` if the imaginary part is zero (a real value).
    #[inline]
    pub fn is_real(&self) -> bool {
        self.im == T::default()
    }
}

impl<T> Complex<T>
where
    T: Clone + Neg<Output = T>,
{
    /// Returns the complex conjugate `re − im·i`.
    #[inline]
    pub fn conj(&self) -> Complex<T> {
        Complex {
            re: self.re.clone(),
            im: -self.im.clone(),
        }
    }
}

impl<T> Complex<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Complex<T>) -> Complex<T> {
        Complex {
            re: self.re.clone() + rhs.re.clone(),
            im: self.im.clone() + rhs.im.clone(),
        }
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Complex<T>) -> Complex<T> {
        Complex {
            re: self.re.clone() - rhs.re.clone(),
            im: self.im.clone() - rhs.im.clone(),
        }
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Complex<T>) -> Complex<T> {
        let ac = self.re.clone() * rhs.re.clone();
        let bd = self.im.clone() * rhs.im.clone();
        let ad = self.re.clone() * rhs.im.clone();
        let bc = self.im.clone() * rhs.re.clone();
        Complex {
            re: ac - bd,
            im: ad + bc,
        }
    }

    /// Returns the squared magnitude `re² + im²` (the field norm; for
    /// `Complex<Int>` this is the Gaussian-integer norm).
    pub fn norm_sqr(&self) -> T {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

impl<T> Complex<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    /// Returns `-self`.
    #[inline]
    pub fn neg(&self) -> Complex<T> {
        Complex {
            re: -self.re.clone(),
            im: -self.im.clone(),
        }
    }
}

impl<T> Complex<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    /// Returns `self / rhs = self · conj(rhs) / |rhs|²`. Requires a divisible
    /// component type (not available for `Complex<Int>`).
    pub fn div(&self, rhs: &Complex<T>) -> Complex<T> {
        let denom = rhs.re.clone() * rhs.re.clone() + rhs.im.clone() * rhs.im.clone();
        let re =
            (self.re.clone() * rhs.re.clone() + self.im.clone() * rhs.im.clone()) / denom.clone();
        let im = (self.im.clone() * rhs.re.clone() - self.re.clone() * rhs.im.clone()) / denom;
        Complex { re, im }
    }
}

impl<T> fmt::Display for Complex<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}i", self.re, self.im)
    }
}

macro_rules! complex_binop {
    ($tr:ident, $m:ident, $bound:path, $atr:ident, $am:ident) => {
        // All four owned/borrowed operand combinations, so `a op b`, `a op &b`,
        // `&a op b`, and `&a op &b` all work.
        impl<T> core::ops::$tr<Complex<T>> for Complex<T>
        where
            T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + $bound,
        {
            type Output = Complex<T>;
            #[inline]
            fn $m(self, rhs: Complex<T>) -> Complex<T> {
                Complex::$m(&self, &rhs)
            }
        }
        impl<T> core::ops::$tr<&Complex<T>> for Complex<T>
        where
            T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + $bound,
        {
            type Output = Complex<T>;
            #[inline]
            fn $m(self, rhs: &Complex<T>) -> Complex<T> {
                Complex::$m(&self, rhs)
            }
        }
        impl<T> core::ops::$tr<Complex<T>> for &Complex<T>
        where
            T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + $bound,
        {
            type Output = Complex<T>;
            #[inline]
            fn $m(self, rhs: Complex<T>) -> Complex<T> {
                Complex::$m(self, &rhs)
            }
        }
        impl<T> core::ops::$tr<&Complex<T>> for &Complex<T>
        where
            T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + $bound,
        {
            type Output = Complex<T>;
            #[inline]
            fn $m(self, rhs: &Complex<T>) -> Complex<T> {
                Complex::$m(self, rhs)
            }
        }
        impl<T> core::ops::$atr<Complex<T>> for Complex<T>
        where
            T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + $bound,
        {
            #[inline]
            fn $am(&mut self, rhs: Complex<T>) {
                *self = Complex::$m(self, &rhs);
            }
        }
        impl<T> core::ops::$atr<&Complex<T>> for Complex<T>
        where
            T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + $bound,
        {
            #[inline]
            fn $am(&mut self, rhs: &Complex<T>) {
                *self = Complex::$m(self, rhs);
            }
        }
    };
}

// Add/Sub/Mul only need the ring operators (bound satisfied by the base three).
complex_binop!(Add, add, Mul<Output = T>, AddAssign, add_assign);
complex_binop!(Sub, sub, Mul<Output = T>, SubAssign, sub_assign);
complex_binop!(Mul, mul, Mul<Output = T>, MulAssign, mul_assign);
// Div additionally needs component division.
complex_binop!(Div, div, Div<Output = T>, DivAssign, div_assign);

impl<T> core::ops::Neg for Complex<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    type Output = Complex<T>;
    #[inline]
    fn neg(self) -> Complex<T> {
        Complex::neg(&self)
    }
}
impl<T> core::ops::Neg for &Complex<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    type Output = Complex<T>;
    #[inline]
    fn neg(self) -> Complex<T> {
        Complex::neg(self)
    }
}

/// Inexact complex analysis on `Complex<Float>`. Every result uses the working
/// precision `max(re.precision(), im.precision())`, rounded to nearest, so no
/// precision argument is needed at the call site.
#[cfg(feature = "float")]
impl Complex<crate::float::Float> {
    fn working_precision(&self) -> u64 {
        self.re.precision().max(self.im.precision())
    }

    /// Modulus `|z| = √(re² + im²)`.
    pub fn abs(&self) -> crate::float::Float {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        let g = p + 16;
        let re2 = Float::mul(&self.re, &self.re, g, Nearest);
        let im2 = Float::mul(&self.im, &self.im, g, Nearest);
        Float::add(&re2, &im2, g, Nearest).sqrt(p, Nearest)
    }

    fn abs_at(&self, w: u64) -> crate::float::Float {
        use crate::float::{Float, RoundingMode::Nearest};
        let re2 = Float::mul(&self.re, &self.re, w, Nearest);
        let im2 = Float::mul(&self.im, &self.im, w, Nearest);
        Float::add(&re2, &im2, w, Nearest).sqrt(w, Nearest)
    }

    /// Argument (phase) `arg(z) = atan2(im, re)`, in `(−π, π]`.
    pub fn arg(&self) -> crate::float::Float {
        self.im.atan2(
            &self.re,
            self.working_precision(),
            crate::float::RoundingMode::Nearest,
        )
    }

    /// `e^z = e^{re}·(cos(im) + i·sin(im))`.
    pub fn exp(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        let er = self.re.exp(p, Nearest);
        Complex {
            re: Float::mul(&er, &self.im.cos(p, Nearest), p, Nearest),
            im: Float::mul(&er, &self.im.sin(p, Nearest), p, Nearest),
        }
    }

    /// Principal logarithm `ln(z) = ln|z| + i·arg(z)`.
    pub fn ln(&self) -> Complex<crate::float::Float> {
        use crate::float::RoundingMode::Nearest;
        Complex {
            re: self.abs().ln(self.working_precision(), Nearest),
            im: self.arg(),
        }
    }

    /// Principal square root, `√((|z|+re)/2) + i·sgn(im)·√((|z|−re)/2)`.
    pub fn sqrt(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        use crate::int::{Int, Sign};
        let p = self.working_precision();
        let g = p + 16;
        let modulus = self.abs_at(g);
        let two = Float::from_int(&Int::from_i64(2), g, Nearest);
        let re = Float::div(
            &Float::add(&modulus, &self.re, g, Nearest),
            &two,
            g,
            Nearest,
        )
        .sqrt(p, Nearest);
        let mut im = Float::div(
            &Float::sub(&modulus, &self.re, g, Nearest),
            &two,
            g,
            Nearest,
        )
        .sqrt(p, Nearest);
        if self.im.sign() == Sign::Negative {
            im = im.neg();
        }
        Complex { re, im }
    }

    /// `z^w = exp(w·ln z)` (principal branch).
    pub fn pow(&self, w: &Complex<crate::float::Float>) -> Complex<crate::float::Float> {
        w.mul(&self.ln()).exp()
    }

    /// `sin(a+bi) = sin a·cosh b + i·cos a·sinh b`.
    pub fn sin(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        Complex {
            re: Float::mul(
                &self.re.sin(p, Nearest),
                &self.im.cosh(p, Nearest),
                p,
                Nearest,
            ),
            im: Float::mul(
                &self.re.cos(p, Nearest),
                &self.im.sinh(p, Nearest),
                p,
                Nearest,
            ),
        }
    }

    /// `cos(a+bi) = cos a·cosh b − i·sin a·sinh b`.
    pub fn cos(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        Complex {
            re: Float::mul(
                &self.re.cos(p, Nearest),
                &self.im.cosh(p, Nearest),
                p,
                Nearest,
            ),
            im: Float::mul(
                &self.re.sin(p, Nearest),
                &self.im.sinh(p, Nearest),
                p,
                Nearest,
            )
            .neg(),
        }
    }
}
