//! Generic dense univariate polynomials `Poly<T>`.
//!
//! Coefficients are stored low-to-high (`coeffs[i]` multiplies `xⁱ`) and kept
//! trimmed so the leading coefficient is nonzero (the zero polynomial has no
//! coefficients). Like [`Complex`](crate::complex::Complex), `Poly<T>` composes
//! with any component type exposing the right operators: ring operations
//! (`+ - *`, evaluation, derivative) need only `+ - *`; polynomial division and
//! GCD additionally need component division, so they are available for field
//! components (`Rational`, `Decimal`, `FixedFloat`) but not for `Int`.
//!
//! `T::default()` is taken to be the additive identity (zero), matching all of
//! this crate's numeric types.

use alloc::vec::Vec;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A dense univariate polynomial with coefficients of type `T`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Poly<T> {
    coeffs: Vec<T>,
}

/// Highest index holding a nonzero entry, or `None` if all are zero.
fn top_nonzero<T: Default + PartialEq>(v: &[T]) -> Option<usize> {
    v.iter().rposition(|c| *c != T::default())
}

impl<T: Clone + Default + PartialEq> Poly<T> {
    /// Builds a polynomial from low-to-high coefficients, trimming trailing zeros.
    pub fn new(mut coeffs: Vec<T>) -> Poly<T> {
        match top_nonzero(&coeffs) {
            Some(i) => coeffs.truncate(i + 1),
            None => coeffs.clear(),
        }
        Poly { coeffs }
    }

    /// The zero polynomial.
    pub fn zero() -> Poly<T> {
        Poly { coeffs: Vec::new() }
    }

    /// The constant polynomial `c`.
    pub fn constant(c: T) -> Poly<T> {
        Poly::new(alloc::vec![c])
    }

    /// The monomial `c·x^degree`.
    pub fn monomial(c: T, degree: usize) -> Poly<T> {
        let mut v = Vec::with_capacity(degree + 1);
        v.resize(degree, T::default());
        v.push(c);
        Poly::new(v)
    }

    /// Returns the coefficients, low-to-high (empty for the zero polynomial).
    #[inline]
    pub fn coeffs(&self) -> &[T] {
        &self.coeffs
    }

    /// Returns `true` if this is the zero polynomial.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Returns the degree, or `None` for the zero polynomial.
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    /// Returns the coefficient of `xⁱ` (zero if out of range).
    pub fn coeff(&self, i: usize) -> T {
        self.coeffs.get(i).cloned().unwrap_or_default()
    }

    /// Returns the leading coefficient, or `None` for the zero polynomial.
    pub fn leading(&self) -> Option<&T> {
        self.coeffs.last()
    }
}

impl<T> Poly<T>
where
    T: Clone + Default + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Poly<T>) -> Poly<T> {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.coeff(i) + rhs.coeff(i));
        }
        Poly::new(out)
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Poly<T>) -> Poly<T> {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.coeff(i) - rhs.coeff(i));
        }
        Poly::new(out)
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Poly<T>) -> Poly<T> {
        if self.is_zero() || rhs.is_zero() {
            return Poly::zero();
        }
        let mut out = alloc::vec![T::default(); self.coeffs.len() + rhs.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in rhs.coeffs.iter().enumerate() {
                let prod = a.clone() * b.clone();
                out[i + j] = out[i + j].clone() + prod;
            }
        }
        Poly::new(out)
    }

    /// Returns `self · scalar`.
    pub fn scalar_mul(&self, scalar: &T) -> Poly<T> {
        Poly::new(
            self.coeffs
                .iter()
                .map(|c| c.clone() * scalar.clone())
                .collect(),
        )
    }

    /// Evaluates the polynomial at `x` (Horner's method).
    pub fn eval(&self, x: &T) -> T {
        let mut acc = T::default();
        for c in self.coeffs.iter().rev() {
            acc = acc * x.clone() + c.clone();
        }
        acc
    }

    /// Returns the formal derivative `d/dx`.
    pub fn derivative(&self) -> Poly<T> {
        if self.coeffs.len() < 2 {
            return Poly::zero();
        }
        let mut out = Vec::with_capacity(self.coeffs.len() - 1);
        for (i, c) in self.coeffs.iter().enumerate().skip(1) {
            // i·c via repeated addition (no integer-scalar trait required).
            let mut acc = T::default();
            for _ in 0..i {
                acc = acc + c.clone();
            }
            out.push(acc);
        }
        Poly::new(out)
    }
}

impl<T> Poly<T>
where
    T: Clone
        + Default
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>,
{
    /// Returns `-self`.
    pub fn neg(&self) -> Poly<T> {
        Poly {
            coeffs: self.coeffs.iter().map(|c| -c.clone()).collect(),
        }
    }
}

impl<T> Poly<T>
where
    T: Clone
        + Default
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    /// Divides `self` by `divisor`, returning `(quotient, remainder)` with
    /// `deg(remainder) < deg(divisor)`. Requires a field component type. Panics
    /// if `divisor` is the zero polynomial.
    pub fn div_rem(&self, divisor: &Poly<T>) -> (Poly<T>, Poly<T>) {
        let dd = divisor
            .degree()
            .expect("Poly::div_rem: division by zero polynomial");
        let lead = divisor.leading().unwrap().clone();
        let mut rem = self.coeffs.clone();
        let mut quot = alloc::vec![T::default(); self.coeffs.len().saturating_sub(dd)];
        while let Some(rd) = top_nonzero(&rem) {
            if rd < dd {
                break;
            }
            let coef = rem[rd].clone() / lead.clone();
            let shift = rd - dd;
            for (i, dc) in divisor.coeffs.iter().enumerate() {
                rem[shift + i] = rem[shift + i].clone() - coef.clone() * dc.clone();
            }
            quot[shift] = coef;
        }
        (Poly::new(quot), Poly::new(rem))
    }

    /// Returns the remainder of `self / divisor`.
    pub fn rem(&self, divisor: &Poly<T>) -> Poly<T> {
        self.div_rem(divisor).1
    }

    /// Returns the monic form (leading coefficient scaled to one), or the zero
    /// polynomial unchanged.
    pub fn monic(&self) -> Poly<T> {
        match self.leading() {
            None => Poly::zero(),
            Some(lead) => {
                let inv_lead = lead.clone();
                Poly::new(
                    self.coeffs
                        .iter()
                        .map(|c| c.clone() / inv_lead.clone())
                        .collect(),
                )
            }
        }
    }

    /// Returns the monic GCD of `self` and `other` (Euclid's algorithm over the
    /// field of coefficients).
    pub fn gcd(&self, other: &Poly<T>) -> Poly<T> {
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a.monic()
    }
}

impl<T: fmt::Display + Clone + Default + PartialEq> fmt::Display for Poly<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate().rev() {
            if *c == T::default() {
                continue;
            }
            if !first {
                f.write_str(" + ")?;
            }
            first = false;
            match i {
                0 => write!(f, "{c}")?,
                1 => write!(f, "{c}·x")?,
                _ => write!(f, "{c}·x^{i}")?,
            }
        }
        Ok(())
    }
}

macro_rules! poly_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl<T> core::ops::$tr for Poly<T>
        where
            T: Clone + Default + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
        {
            type Output = Poly<T>;
            #[inline]
            fn $m(self, rhs: Poly<T>) -> Poly<T> {
                Poly::$m(&self, &rhs)
            }
        }
        impl<T> core::ops::$tr<&Poly<T>> for &Poly<T>
        where
            T: Clone + Default + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
        {
            type Output = Poly<T>;
            #[inline]
            fn $m(self, rhs: &Poly<T>) -> Poly<T> {
                Poly::$m(self, rhs)
            }
        }
        impl<T> core::ops::$atr for Poly<T>
        where
            T: Clone + Default + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
        {
            #[inline]
            fn $am(&mut self, rhs: Poly<T>) {
                *self = Poly::$m(self, &rhs);
            }
        }
    };
}

poly_binop!(Add, add, AddAssign, add_assign);
poly_binop!(Sub, sub, SubAssign, sub_assign);
poly_binop!(Mul, mul, MulAssign, mul_assign);
