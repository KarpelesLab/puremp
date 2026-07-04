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

/// An element of a (commutative) field: a [`Ring`] whose nonzero elements are
/// invertible (it also has division).
///
/// Caveat: `ModInt` is a genuine field only when its modulus is prime, and
/// `Float` only up to rounding. Both implement `Field` so the generic
/// linear-algebra / polynomial machinery can run over them, but it is the
/// caller's responsibility that the ring is actually a field (a prime modulus,
/// numerically well-conditioned data, …).
pub trait Field: Ring + core::ops::Div<Output = Self> {
    /// The multiplicative inverse of `self`, or `None` when `self` is zero.
    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.one() / self.clone())
        }
    }
}

#[cfg(feature = "rational")]
impl Field for crate::rational::Rational {}

#[cfg(feature = "float")]
impl Field for crate::float::Float {}

#[cfg(feature = "int")]
impl Field for crate::mod_int::ModInt {
    #[inline]
    fn inv(&self) -> Option<Self> {
        crate::mod_int::ModInt::inv(self)
    }
}

#[cfg(feature = "galois")]
impl Field for crate::galois::GfElement {
    #[inline]
    fn inv(&self) -> Option<Self> {
        crate::galois::GfElement::inv(self)
    }
}

#[cfg(feature = "complex")]
impl<F: Field> Field for crate::complex::Complex<F> {}

/// A **finite** field `GF(q)`: a [`Field`] that additionally exposes its size
/// `q` (the number of elements) and characteristic `p`.
///
/// This is the marker the generic Cantor–Zassenhaus factorizer
/// (`FactorOverField`, with the `poly` feature) keys on: the algorithm
/// needs the field order `q`, which only *finite* fields have. The infinite
/// fields (`Rational`, `Float`) deliberately do **not** implement it, so the
/// factorizer cannot be misapplied to them.
///
/// Caveat (as with [`Field`]): a [`ModInt`] is a genuine field only when its
/// modulus is prime — it is the caller's responsibility that the ring really is
/// a field.
///
/// [`ModInt`]: crate::mod_int::ModInt
#[cfg(feature = "int")]
pub trait FiniteField: Field {
    /// The field order `q`: the number of elements (`p` for the prime field
    /// `ℤ/pℤ`, `pᵏ` for `GF(pᵏ)`).
    fn order(&self) -> crate::int::Int;

    /// The characteristic `p` (a prime): the additive order of `1`.
    fn characteristic(&self) -> crate::int::Int;

    /// Maps an integer to a field element — a bijection of `[0, q)` onto the
    /// field (the index is first reduced modulo `q`). Lets generic code
    /// enumerate or randomly sample field elements without knowing the concrete
    /// representation. Takes `&self` for the field context (modulus/field data),
    /// not as a value to convert.
    #[allow(clippy::wrong_self_convention)]
    fn from_index(&self, index: &crate::int::Int) -> Self;
}

#[cfg(feature = "int")]
impl FiniteField for crate::mod_int::ModInt {
    /// The order equals the modulus `p` (a genuine field only when it is prime).
    #[inline]
    fn order(&self) -> crate::int::Int {
        self.modulus()
    }
    /// The characteristic equals the modulus `p`.
    #[inline]
    fn characteristic(&self) -> crate::int::Int {
        self.modulus()
    }
    #[inline]
    fn from_index(&self, index: &crate::int::Int) -> Self {
        self.of(index.clone())
    }
}

#[cfg(feature = "galois")]
impl FiniteField for crate::galois::GfElement {
    /// The order `pᵏ`.
    #[inline]
    fn order(&self) -> crate::int::Int {
        self.field().order()
    }
    /// The characteristic `p`.
    #[inline]
    fn characteristic(&self) -> crate::int::Int {
        self.field().characteristic()
    }
    /// Writes `index mod pᵏ` in base `p` (`k` little-endian digits) and reads the
    /// digits back as the coefficient vector of an element of `GF(pᵏ)`.
    fn from_index(&self, index: &crate::int::Int) -> Self {
        let field = self.field();
        let p = field.characteristic();
        let mut n = index.rem_euclid(&field.order());
        let mut digits = alloc::vec::Vec::with_capacity(field.degree());
        for _ in 0..field.degree() {
            digits.push(n.rem_euclid(&p));
            n = n.div_floor(&p);
        }
        field.element(&digits)
    }
}
