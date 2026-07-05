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

    /// Whether this ring's `+`/`−`/`×` are *exact and associative*, so that a
    /// re-associating fast matrix multiply (Strassen–Winograd, which forms
    /// products of sums of blocks and recombines them in a different order) is
    /// **bit-identical** to the naive triple loop.
    ///
    /// The default is `false`: a ring opts in only when it is genuinely exact.
    /// The exact arbitrary-precision rings ([`Int`](crate::int::Int),
    /// [`Rational`](crate::rational::Rational)) set it to `true`; rounding types
    /// such as [`Float`](crate::float::Float) leave it `false`, so their
    /// [`Matrix::mul`](crate::matrix::Matrix::mul) always uses the naive path and
    /// their results are unaffected.
    const REASSOCIATIVE: bool = false;

    /// A cheap proxy for the *cost of multiplying two ring elements of roughly
    /// `self`'s magnitude* — for the arbitrary-precision integers/rationals, the
    /// operand's bit length.
    ///
    /// Strassen–Winograd trades one element multiply (per 8) for a handful of
    /// extra element additions and block allocations; that is a win only when a
    /// multiply is far dearer than an add, i.e. when the operands are large.
    /// [`Matrix::mul`](crate::matrix::Matrix::mul) samples this on one entry to
    /// decide whether to take the Strassen path. It never affects the *result*,
    /// only the path taken; the default `0` (a free multiply) keeps a ring on the
    /// naive path.
    fn multiply_cost_hint(&self) -> u64 {
        0
    }
}

#[cfg(feature = "int")]
impl Ring for crate::int::Int {
    // Arbitrary-precision integer arithmetic is exact and associative.
    const REASSOCIATIVE: bool = true;
    #[inline]
    fn multiply_cost_hint(&self) -> u64 {
        u64::from(self.bit_len())
    }
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
    // Exact reduced fractions: addition and multiplication are exact/associative.
    const REASSOCIATIVE: bool = true;
    #[inline]
    fn multiply_cost_hint(&self) -> u64 {
        // A rational multiply costs roughly both numerator and denominator
        // products (plus a gcd); size it by their combined bit length.
        u64::from(self.numerator().bit_len()) + u64::from(self.denominator().bit_len())
    }
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

#[cfg(feature = "poly")]
impl<T: Ring> Ring for crate::poly::Poly<T> {
    /// The zero polynomial.
    #[inline]
    fn zero(&self) -> Self {
        crate::poly::Poly::zero()
    }
    /// The constant polynomial `1` (its `1` drawn from a coefficient's ring).
    ///
    /// # Panics
    /// On the zero polynomial, which has no coefficient to source the ring from.
    fn one(&self) -> Self {
        crate::poly::Poly::constant(
            self.leading()
                .expect("Poly::one: the zero polynomial has no ring context")
                .one(),
        )
    }
    #[inline]
    fn is_zero(&self) -> bool {
        crate::poly::Poly::is_zero(self)
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
