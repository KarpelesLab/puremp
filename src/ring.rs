//! The [`Ring`] abstraction: identities relative to a sample element.
//!
//! Generic containers such as [`Poly`](crate::poly::Poly) and
//! [`Matrix`](crate::matrix::Matrix) need an additive zero (and, for identity
//! matrices, a multiplicative one) of the *same* ring as the elements they hold.
//! For the context-free numeric types (`Int`, `Rational`, `Float`, …) that zero
//! is a constant, but the context-carrying rings — [`ModInt`](crate::mod_int::ModInt)
//! (`ℤ/nℤ`) and [`GfElement`](crate::galois::GfElement) (`GF(pᵏ)`) — cannot
//! manufacture their identities out of thin air: a zero only makes sense once you
//! know the modulus or the field.
//!
//! [`Ring`] resolves this by taking `&self` as the context. `a.zero()` returns
//! the additive identity of the same ring as `a`, `a.one()` its multiplicative
//! identity; context-free types simply ignore `self`. This deliberately does
//! *not* touch the external `num-traits` `Zero`/`One` traits, which remain the
//! context-free bridge for the numeric types that have one.

use core::ops::{Add, Mul, Neg, Sub};

/// A ring: a type with `+ − ×` and additive/multiplicative identities that are
/// taken *relative to a sample element* (`&self`).
///
/// Taking `&self` is what lets the context-carrying rings ([`ModInt`], finite
/// field [`GfElement`]) produce identities in their own ring (same modulus /
/// field); the context-free numeric types ignore `self` and return their
/// canonical constants.
///
/// [`ModInt`]: crate::mod_int::ModInt
/// [`GfElement`]: crate::galois::GfElement
pub trait Ring:
    Clone
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// The additive identity of the *same* ring as `self` (same modulus/field).
    fn zero(&self) -> Self;
    /// The multiplicative identity of the same ring as `self`.
    fn one(&self) -> Self;
    /// Whether `self` is the additive identity.
    fn is_zero(&self) -> bool;
}

#[cfg(feature = "int")]
impl Ring for crate::int::Int {
    #[inline]
    fn zero(&self) -> Self {
        crate::int::Int::ZERO
    }
    #[inline]
    fn one(&self) -> Self {
        crate::int::Int::ONE
    }
    #[inline]
    fn is_zero(&self) -> bool {
        crate::int::Int::is_zero(self)
    }
}

#[cfg(feature = "rational")]
impl Ring for crate::rational::Rational {
    #[inline]
    fn zero(&self) -> Self {
        crate::rational::Rational::ZERO
    }
    #[inline]
    fn one(&self) -> Self {
        crate::rational::Rational::ONE
    }
    #[inline]
    fn is_zero(&self) -> bool {
        crate::rational::Rational::is_zero(self)
    }
}

#[cfg(feature = "float")]
impl Ring for crate::float::Float {
    /// Zero at the same working precision as `self`.
    #[inline]
    fn zero(&self) -> Self {
        crate::float::Float::zero(self.precision())
    }
    /// One at the same working precision as `self`.
    #[inline]
    fn one(&self) -> Self {
        crate::float::Float::from_int(
            &crate::int::Int::ONE,
            self.precision(),
            crate::float::RoundingMode::Nearest,
        )
    }
    #[inline]
    fn is_zero(&self) -> bool {
        crate::float::Float::is_zero(self)
    }
}

#[cfg(feature = "dyadic")]
impl Ring for crate::dyadic::Dyadic {
    #[inline]
    fn zero(&self) -> Self {
        crate::dyadic::Dyadic::zero()
    }
    #[inline]
    fn one(&self) -> Self {
        crate::dyadic::Dyadic::one()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        crate::dyadic::Dyadic::is_zero(self)
    }
}

#[cfg(feature = "decimal")]
impl Ring for crate::decimal::Decimal {
    #[inline]
    fn zero(&self) -> Self {
        crate::decimal::Decimal::zero()
    }
    #[inline]
    fn one(&self) -> Self {
        crate::decimal::Decimal::one()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        crate::decimal::Decimal::is_zero(self)
    }
}

#[cfg(feature = "int")]
impl Ring for crate::mod_int::ModInt {
    /// Zero in the same ring `ℤ/nℤ` as `self` (shares the modulus).
    #[inline]
    fn zero(&self) -> Self {
        self.of(crate::int::Int::ZERO)
    }
    /// One in the same ring `ℤ/nℤ` as `self` (shares the modulus).
    #[inline]
    fn one(&self) -> Self {
        self.of(crate::int::Int::ONE)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        crate::mod_int::ModInt::is_zero(self)
    }
}

#[cfg(feature = "galois")]
impl Ring for crate::galois::GfElement {
    /// Zero in the same field `GF(pᵏ)` as `self`.
    #[inline]
    fn zero(&self) -> Self {
        self.field().zero()
    }
    /// One in the same field `GF(pᵏ)` as `self`.
    #[inline]
    fn one(&self) -> Self {
        self.field().one()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        crate::galois::GfElement::is_zero(self)
    }
}

#[cfg(feature = "complex")]
impl<T: Ring> Ring for crate::complex::Complex<T> {
    /// Componentwise zero (`0 + 0·i`), each component in the same ring as
    /// `self`'s.
    #[inline]
    fn zero(&self) -> Self {
        crate::complex::Complex::new(self.re.zero(), self.im.zero())
    }
    /// The multiplicative identity `1 + 0·i`.
    #[inline]
    fn one(&self) -> Self {
        crate::complex::Complex::new(self.re.one(), self.im.zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
}
