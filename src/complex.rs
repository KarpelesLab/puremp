//! Generic complex numbers `re + im·i`.
//!
//! [`Complex`] is parameterized over the component type, so it composes with any
//! of the crate's numeric types that expose the relevant operators:
//! `Complex<Int>` (Gaussian integers), `Complex<Rational>`, `Complex<Dyadic>`,
//! `Complex<Decimal>`, and `Complex<Float>` / `Complex<FixedFloat>` for inexact
//! complex. `Complex<Float>` additionally carries a full analytic suite —
//! `abs`/`arg`, `exp`, `ln`, `sqrt`, `pow`, the trigonometric and hyperbolic
//! functions and their inverses.
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

    // --- private complex-arithmetic helpers (Float lacks +,-,*,/ operators, so
    // every combining step is spelled out via the fully-qualified `Float::*`
    // rounded operations at an explicit working precision) ---

    /// Complex product `a·b`, components rounded to `p` bits.
    fn cmul(a: &Complex<crate::float::Float>, b: &Complex<crate::float::Float>, p: u64) -> Self {
        use crate::float::{Float, RoundingMode::Nearest};
        let ac = Float::mul(&a.re, &b.re, p, Nearest);
        let bd = Float::mul(&a.im, &b.im, p, Nearest);
        let ad = Float::mul(&a.re, &b.im, p, Nearest);
        let bc = Float::mul(&a.im, &b.re, p, Nearest);
        Complex {
            re: Float::sub(&ac, &bd, p, Nearest),
            im: Float::add(&ad, &bc, p, Nearest),
        }
    }

    /// Complex sum `a+b`, components rounded to `p` bits.
    fn cadd(a: &Complex<crate::float::Float>, b: &Complex<crate::float::Float>, p: u64) -> Self {
        use crate::float::{Float, RoundingMode::Nearest};
        Complex {
            re: Float::add(&a.re, &b.re, p, Nearest),
            im: Float::add(&a.im, &b.im, p, Nearest),
        }
    }

    /// Complex quotient `a/b = a·conj(b)/|b|²`, components rounded to `p` bits.
    fn cdiv(a: &Complex<crate::float::Float>, b: &Complex<crate::float::Float>, p: u64) -> Self {
        use crate::float::{Float, RoundingMode::Nearest};
        let br2 = Float::mul(&b.re, &b.re, p, Nearest);
        let bi2 = Float::mul(&b.im, &b.im, p, Nearest);
        let denom = Float::add(&br2, &bi2, p, Nearest);
        let ac = Float::mul(&a.re, &b.re, p, Nearest);
        let bd = Float::mul(&a.im, &b.im, p, Nearest);
        let bc = Float::mul(&a.im, &b.re, p, Nearest);
        let ad = Float::mul(&a.re, &b.im, p, Nearest);
        Complex {
            re: Float::div(&Float::add(&ac, &bd, p, Nearest), &denom, p, Nearest),
            im: Float::div(&Float::sub(&bc, &ad, p, Nearest), &denom, p, Nearest),
        }
    }

    /// Multiply by the imaginary unit: `i·z = −z.im + z.re·i` (exact — a swap
    /// and a negation, no rounding).
    fn mul_i(z: &Complex<crate::float::Float>) -> Self {
        Complex {
            re: crate::float::Float::neg(&z.im),
            im: z.re.clone(),
        }
    }

    /// Multiply by `−i`: `−i·z = z.im − z.re·i` (exact).
    fn mul_neg_i(z: &Complex<crate::float::Float>) -> Self {
        Complex {
            re: z.im.clone(),
            im: crate::float::Float::neg(&z.re),
        }
    }

    /// Round both components to `p` bits (used to bring a guard-precision
    /// intermediate back to the working precision).
    fn round_to(z: &Complex<crate::float::Float>, p: u64) -> Self {
        use crate::float::RoundingMode::Nearest;
        Complex {
            re: z.re.round(p, Nearest),
            im: z.im.round(p, Nearest),
        }
    }

    /// `tan z = sin z / cos z`. Poles at `z = π/2 + kπ`.
    pub fn tan(&self) -> Complex<crate::float::Float> {
        let p = self.working_precision();
        Self::cdiv(&self.sin(), &self.cos(), p)
    }

    /// `cot z = cos z / sin z`. Poles at `z = kπ`.
    pub fn cot(&self) -> Complex<crate::float::Float> {
        let p = self.working_precision();
        Self::cdiv(&self.cos(), &self.sin(), p)
    }

    /// `sinh(a+bi) = sinh a·cos b + i·cosh a·sin b`.
    pub fn sinh(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        Complex {
            re: Float::mul(
                &self.re.sinh(p, Nearest),
                &self.im.cos(p, Nearest),
                p,
                Nearest,
            ),
            im: Float::mul(
                &self.re.cosh(p, Nearest),
                &self.im.sin(p, Nearest),
                p,
                Nearest,
            ),
        }
    }

    /// `cosh(a+bi) = cosh a·cos b + i·sinh a·sin b`.
    pub fn cosh(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        Complex {
            re: Float::mul(
                &self.re.cosh(p, Nearest),
                &self.im.cos(p, Nearest),
                p,
                Nearest,
            ),
            im: Float::mul(
                &self.re.sinh(p, Nearest),
                &self.im.sin(p, Nearest),
                p,
                Nearest,
            ),
        }
    }

    /// `tanh z = sinh z / cosh z`. Poles at `z = i(π/2 + kπ)`.
    pub fn tanh(&self) -> Complex<crate::float::Float> {
        let p = self.working_precision();
        Self::cdiv(&self.sinh(), &self.cosh(), p)
    }

    /// Principal inverse sine `asin z = −i·ln(iz + √(1−z²))`.
    ///
    /// Range: `Re ∈ [−π/2, π/2]`. Branch cuts along the real axis outside
    /// `[−1, 1]`, i.e. `(−∞, −1)` and `(1, ∞)` (DLMF §4.23), inherited from the
    /// principal `√` and `ln`.
    pub fn asin(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        let g = p + 16;
        let one = Float::from_f64(1.0, g, Nearest);
        let z2 = Self::cmul(self, self, g); // z²
        let one_minus = Complex {
            // 1 − z²
            re: Float::sub(&one, &z2.re, g, Nearest),
            im: Float::neg(&z2.im),
        };
        let root = one_minus.sqrt(); // √(1−z²)
        let inner = Self::cadd(&Self::mul_i(self), &root, g); // iz + √(1−z²)
        Self::round_to(&Self::mul_neg_i(&inner.ln()), p)
    }

    /// Principal inverse cosine `acos z = π/2 − asin z`.
    ///
    /// Range: `Re ∈ [0, π]`. Branch cuts along the real axis outside `[−1, 1]`
    /// (DLMF §4.23).
    pub fn acos(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        let g = p + 16;
        let a = self.asin();
        let two = Float::from_f64(2.0, g, Nearest);
        let half_pi = Float::div(&Float::pi(g, Nearest), &two, g, Nearest);
        Self::round_to(
            &Complex {
                re: Float::sub(&half_pi, &a.re, g, Nearest),
                im: Float::neg(&a.im),
            },
            p,
        )
    }

    /// Principal inverse tangent `atan z = (i/2)·ln((i+z)/(i−z))`
    /// (equivalently `(1/2i)·ln((1+iz)/(1−iz))`).
    ///
    /// Range: `Re ∈ [−π/2, π/2]`. Branch cuts along the imaginary axis outside
    /// `[−i, i]`, i.e. `(−i∞, −i)` and `(i, i∞)` (DLMF §4.23).
    pub fn atan(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        let g = p + 16;
        let one = Float::from_f64(1.0, g, Nearest);
        let num = Complex {
            // i + z
            re: self.re.clone(),
            im: Float::add(&self.im, &one, g, Nearest),
        };
        let den = Complex {
            // i − z
            re: Float::neg(&self.re),
            im: Float::sub(&one, &self.im, g, Nearest),
        };
        let l = Self::cdiv(&num, &den, g).ln();
        // (i/2)·(l.re + i·l.im) = −l.im/2 + i·l.re/2
        let two = Float::from_f64(2.0, g, Nearest);
        Self::round_to(
            &Complex {
                re: Float::div(&Float::neg(&l.im), &two, g, Nearest),
                im: Float::div(&l.re, &two, g, Nearest),
            },
            p,
        )
    }

    /// Principal inverse hyperbolic sine `asinh z = ln(z + √(z²+1))`.
    ///
    /// Branch cuts along the imaginary axis outside `[−i, i]`, i.e. `(−i∞, −i)`
    /// and `(i, i∞)` (DLMF §4.37).
    pub fn asinh(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        let g = p + 16;
        let one = Float::from_f64(1.0, g, Nearest);
        let z2 = Self::cmul(self, self, g); // z²
        let z2p1 = Complex {
            // z² + 1
            re: Float::add(&z2.re, &one, g, Nearest),
            im: z2.im.clone(),
        };
        let inner = Self::cadd(self, &z2p1.sqrt(), g); // z + √(z²+1)
        Self::round_to(&inner.ln(), p)
    }

    /// Principal inverse hyperbolic cosine `acosh z = ln(z + √(z−1)·√(z+1))`.
    ///
    /// Range: `Im ∈ [−π, π]`, `Re ≥ 0`. Branch cut along the real axis on
    /// `(−∞, 1)` (DLMF §4.37). The two-factor `√(z−1)·√(z+1)` form selects the
    /// principal branch (unlike `√(z²−1)`).
    pub fn acosh(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        let g = p + 16;
        let one = Float::from_f64(1.0, g, Nearest);
        let zm1 = Complex {
            // z − 1
            re: Float::sub(&self.re, &one, g, Nearest),
            im: self.im.clone(),
        };
        let zp1 = Complex {
            // z + 1
            re: Float::add(&self.re, &one, g, Nearest),
            im: self.im.clone(),
        };
        let root = Self::cmul(&zm1.sqrt(), &zp1.sqrt(), g); // √(z−1)·√(z+1)
        let inner = Self::cadd(self, &root, g); // z + √(z−1)·√(z+1)
        Self::round_to(&inner.ln(), p)
    }

    /// Principal inverse hyperbolic tangent `atanh z = ½·ln((1+z)/(1−z))`.
    ///
    /// Branch cuts along the real axis outside `[−1, 1]`, i.e. `(−∞, −1)` and
    /// `(1, ∞)` (DLMF §4.37).
    pub fn atanh(&self) -> Complex<crate::float::Float> {
        use crate::float::{Float, RoundingMode::Nearest};
        let p = self.working_precision();
        let g = p + 16;
        let one = Float::from_f64(1.0, g, Nearest);
        let num = Complex {
            // 1 + z
            re: Float::add(&one, &self.re, g, Nearest),
            im: self.im.clone(),
        };
        let den = Complex {
            // 1 − z
            re: Float::sub(&one, &self.re, g, Nearest),
            im: Float::neg(&self.im),
        };
        let l = Self::cdiv(&num, &den, g).ln();
        let half = Float::from_f64(0.5, g, Nearest);
        Self::round_to(
            &Complex {
                re: Float::mul(&l.re, &half, g, Nearest),
                im: Float::mul(&l.im, &half, g, Nearest),
            },
            p,
        )
    }
}
