//! Algebraic number fields `K = ℚ(θ) = ℚ[x]/(T(x))` and their element
//! arithmetic.
//!
//! A [`NumberField`] is presented by a **monic irreducible** polynomial
//! `T ∈ ℚ[x]` of degree `n ≥ 1`; the field is the quotient `ℚ[x]/(T)`, a
//! ℚ-vector space of dimension `n` with the *power basis* `1, θ, θ², …, θⁿ⁻¹`
//! (where `θ` is the residue class of `x`). Its elements
//! ([`NumberFieldElement`]) are polynomials in `θ` of degree `< n`, i.e.
//! [`Poly<Rational>`](crate::poly::Poly) representatives reduced modulo `T`.
//!
//! The core algebraic invariants are read off the **multiplication matrix**
//! `M_α`: the `n × n` matrix over ℚ of the ℚ-linear map "multiply by α" written
//! in the power basis. Then
//!
//! - `norm(α)  = det(M_α)`   (a rational; multiplicative),
//! - `trace(α) = tr(M_α)`    (a rational; additive),
//! - `char_poly(α) = det(x·I − M_α)` (monic, degree `n`),
//! - `min_poly(α)` = the squarefree part of `char_poly(α)` (the minimal monic
//!   annihilator; it divides the characteristic polynomial),
//! - `α` is an *algebraic integer* iff `char_poly(α) ∈ ℤ[x]`.
//!
//! The `n` complex roots `θ_1, …, θ_n` of `T` give the field's embeddings
//! `σ_i : K → ℂ` (`α ↦ α(θ_i)`); the [`signature`](NumberField::signature)
//! `(r₁, r₂)` counts the real embeddings and conjugate complex pairs. As an
//! independent numerical cross-check, `norm(α) = ∏ σ_i(α)` and
//! `trace(α) = Σ σ_i(α)`.
//!
//! This is a clean-room implementation drawn from the open literature (Cohen,
//! *A Course in Computational Algebraic Number Theory*, ch. 3–4; Marcus,
//! *Number Fields*). The defining polynomial is shared cheaply between a field
//! and its elements via [`Rc`], mirroring
//! [`GaloisField`](crate::galois::GaloisField)/[`ModInt`](crate::mod_int::ModInt).

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::fmt;

use crate::complex::Complex;
use crate::float::{Float, RoundingMode};
use crate::int::Int;
use crate::matrix::{Matrix, RingMatrix};
use crate::poly::Poly;
use crate::rational::Rational;
use crate::ring::{Field, Ring};

/// The shared, immutable data of a number field: its monic irreducible defining
/// polynomial and its degree.
struct FieldData {
    /// The monic irreducible defining polynomial `T ∈ ℚ[x]`.
    t: Poly<Rational>,
    /// The degree `n = deg(T) ≥ 1`.
    n: usize,
}

/// A number field `K = ℚ(θ) = ℚ[x]/(T(x))`, presented by a monic irreducible
/// `T ∈ ℚ[x]` of degree `n ≥ 1`.
///
/// Cheap to clone: the defining polynomial is shared (`Rc`), and every
/// [`NumberFieldElement`] carries the same shared context.
#[derive(Clone)]
pub struct NumberField {
    data: Rc<FieldData>,
}

/// An element of a [`NumberField`], stored as its canonical representative: a
/// [`Poly<Rational>`](crate::poly::Poly) of degree `< n` in the power basis
/// `1, θ, …, θⁿ⁻¹`.
#[derive(Clone)]
pub struct NumberFieldElement {
    field: Rc<FieldData>,
    /// The reduced representative, `deg < n` (the zero polynomial for `0`).
    poly: Poly<Rational>,
}

impl NumberField {
    /// Builds the number field `ℚ[x]/(T)` from a defining polynomial `T`.
    ///
    /// `T` must have degree `n ≥ 1` and be **irreducible over ℚ** (verified with
    /// the crate's rational factorization). The polynomial is made monic
    /// internally, so any nonzero leading coefficient is accepted; irreducibility
    /// is scale-invariant. Returns `None` if `T` is constant/zero or reducible.
    pub fn new(t: Poly<Rational>) -> Option<NumberField> {
        let n = t.degree()?;
        if n < 1 {
            return None;
        }
        let tm = t.monic();
        // Irreducible over ℚ ⟺ a single monic irreducible factor of multiplicity 1.
        let factors = tm.factor();
        if factors.len() != 1 || factors[0].1 != 1 {
            return None;
        }
        Some(NumberField {
            data: Rc::new(FieldData { t: tm, n }),
        })
    }

    /// The degree `n = [K : ℚ]` of the field.
    #[inline]
    pub fn degree(&self) -> usize {
        self.data.n
    }

    /// The monic irreducible defining polynomial `T ∈ ℚ[x]`.
    #[inline]
    pub fn defining_polynomial(&self) -> &Poly<Rational> {
        &self.data.t
    }

    /// Builds the element represented by `poly`, reduced modulo `T` to its
    /// canonical representative of degree `< n`.
    pub fn element(&self, poly: Poly<Rational>) -> NumberFieldElement {
        NumberFieldElement {
            field: self.data.clone(),
            poly: reduce(&poly, &self.data.t),
        }
    }

    /// The embedding of a rational `r` as the constant element `r ∈ K`.
    pub fn from_rational(&self, r: Rational) -> NumberFieldElement {
        NumberFieldElement {
            field: self.data.clone(),
            poly: reduce(&Poly::constant(r), &self.data.t),
        }
    }

    /// The additive identity `0 ∈ K`.
    pub fn zero(&self) -> NumberFieldElement {
        NumberFieldElement {
            field: self.data.clone(),
            poly: Poly::zero(),
        }
    }

    /// The multiplicative identity `1 ∈ K`.
    pub fn one(&self) -> NumberFieldElement {
        NumberFieldElement {
            field: self.data.clone(),
            poly: reduce(&Poly::constant(Rational::ONE), &self.data.t),
        }
    }

    /// The generator `θ`, the residue class of `x`.
    pub fn generator(&self) -> NumberFieldElement {
        NumberFieldElement {
            field: self.data.clone(),
            poly: reduce(&Poly::monomial(Rational::ONE, 1), &self.data.t),
        }
    }

    /// The discriminant `disc(T) ∈ ℚ` of the defining polynomial.
    ///
    /// Computed as `(−1)^{n(n−1)/2}·Res(T, T′)` (the defining polynomial is
    /// monic, so no leading-coefficient division is needed), where the resultant
    /// is the determinant of the Sylvester matrix of `T` and its derivative.
    pub fn discriminant(&self) -> Rational {
        let t = &self.data.t;
        let res = resultant(t, &t.derivative());
        let n = self.data.n;
        // Sign (−1)^{n(n−1)/2}.
        if (n * (n - 1) / 2).is_multiple_of(2) {
            res
        } else {
            res.neg()
        }
    }

    /// The signature `(r₁, r₂)`: the number of real embeddings `r₁` and of
    /// conjugate complex-embedding pairs `r₂`, with `r₁ + 2·r₂ = n`.
    ///
    /// `r₁` is the (exact) count of real roots of `T` via Sturm's theorem; since
    /// `T` is irreducible its roots are distinct, so the remaining `n − r₁` roots
    /// form `r₂ = (n − r₁)/2` complex-conjugate pairs.
    pub fn signature(&self) -> (usize, usize) {
        let r1 = self.data.t.real_root_count();
        (r1, (self.data.n - r1) / 2)
    }

    /// The `n` complex roots `θ_i` of `T` (the images of `θ` under the
    /// embeddings `σ_i`), each component computed to `precision` bits.
    ///
    /// The roots are found with the Durand–Kerner (Weierstrass) simultaneous
    /// iteration over [`Complex<Float>`](crate::complex::Complex).
    pub fn complex_roots(&self, precision: u64) -> Vec<Complex<Float>> {
        durand_kerner(&self.data.t, precision)
    }
}

impl NumberFieldElement {
    /// The field this element belongs to.
    pub fn field(&self) -> NumberField {
        NumberField {
            data: self.field.clone(),
        }
    }

    /// The canonical representative, a [`Poly<Rational>`](crate::poly::Poly) of
    /// degree `< n` in the power basis `1, θ, …, θⁿ⁻¹`.
    #[inline]
    pub fn poly(&self) -> &Poly<Rational> {
        &self.poly
    }

    /// Whether this is the additive identity `0`.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.poly.is_zero()
    }

    /// Whether this is the multiplicative identity `1`.
    pub fn is_one(&self) -> bool {
        self.poly.degree() == Some(0) && self.poly.coeff(0).is_one()
    }

    fn same_field(&self, other: &NumberFieldElement) {
        assert!(
            same_field(&self.field, &other.field),
            "NumberFieldElement: operands from different fields"
        );
    }

    #[inline]
    fn wrap(&self, poly: Poly<Rational>) -> NumberFieldElement {
        NumberFieldElement {
            field: self.field.clone(),
            poly,
        }
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &NumberFieldElement) -> NumberFieldElement {
        self.same_field(rhs);
        // Both operands have degree < n; the sum does too — no reduction needed.
        self.wrap(self.poly.add(&rhs.poly))
    }

    /// Returns `self − rhs`.
    pub fn sub(&self, rhs: &NumberFieldElement) -> NumberFieldElement {
        self.same_field(rhs);
        self.wrap(self.poly.sub(&rhs.poly))
    }

    /// Returns `−self`.
    pub fn neg(&self) -> NumberFieldElement {
        self.wrap(self.poly.neg())
    }

    /// Returns `self · rhs` (polynomial product reduced modulo `T`).
    pub fn mul(&self, rhs: &NumberFieldElement) -> NumberFieldElement {
        self.same_field(rhs);
        self.wrap(reduce(&self.poly.mul(&rhs.poly), &self.field.t))
    }

    /// Returns the multiplicative inverse `self⁻¹`, or `None` if `self` is zero.
    ///
    /// Uses the extended Euclidean algorithm in `ℚ[x]`: since `T` is irreducible
    /// and `deg(self) < deg(T)`, `gcd(self, T)` is a nonzero constant `c` with
    /// `u·self + v·T = c`, so `self⁻¹ = (u/c) mod T`.
    pub fn inv(&self) -> Option<NumberFieldElement> {
        if self.poly.is_zero() {
            return None;
        }
        let (g, u, _v) = poly_ext_gcd(&self.poly, &self.field.t);
        // g is a nonzero constant; normalise u by it and reduce mod T.
        let c = g.coeff(0);
        let uinv = u.scalar_mul(&c.recip());
        Some(self.wrap(reduce(&uinv, &self.field.t)))
    }

    /// Returns `self / rhs = self · rhs⁻¹`.
    ///
    /// # Panics
    /// If `rhs` is zero.
    pub fn div(&self, rhs: &NumberFieldElement) -> NumberFieldElement {
        self.mul(&rhs.inv().expect("NumberFieldElement::div: divisor is zero"))
    }

    /// Returns `self` raised to the integer power `exp` (negative exponents
    /// invert first).
    ///
    /// # Panics
    /// If `exp < 0` and `self` is zero.
    pub fn pow(&self, exp: i64) -> NumberFieldElement {
        if exp < 0 {
            return self
                .inv()
                .expect("NumberFieldElement::pow: zero base with negative exponent")
                .pow(-exp);
        }
        // Square-and-multiply, low bit first.
        let mut result = NumberFieldElement {
            field: self.field.clone(),
            poly: reduce(&Poly::constant(Rational::ONE), &self.field.t),
        };
        let mut base = self.clone();
        let mut e = exp as u64;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul(&base);
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    /// The `n × n` **multiplication matrix** `M_α` over ℚ: the matrix of the
    /// ℚ-linear map `β ↦ α·β` in the power basis `1, θ, …, θⁿ⁻¹`. Its column `j`
    /// holds the coordinates of `α·θʲ` reduced modulo `T`.
    pub fn mul_matrix(&self) -> Matrix<Rational> {
        let n = self.field.n;
        let t = &self.field.t;
        let mut data = alloc::vec![Rational::ZERO; n * n];
        // Column j: coordinates of (self · x^j) mod T.
        let mut col = self.poly.clone(); // self · x^0
        for j in 0..n {
            let reduced = reduce(&col, t);
            for i in 0..n {
                data[i * n + j] = reduced.coeff(i);
            }
            // Advance to self · x^(j+1) by multiplying the representative by x.
            if j + 1 < n {
                col = col.mul(&Poly::monomial(Rational::ONE, 1));
            }
        }
        Matrix::new(n, n, data)
    }

    /// The field norm `N(α) = det(M_α) ∈ ℚ` (multiplicative).
    pub fn norm(&self) -> Rational {
        self.mul_matrix().determinant()
    }

    /// The field trace `Tr(α) = tr(M_α) ∈ ℚ` (additive).
    pub fn trace(&self) -> Rational {
        let m = self.mul_matrix();
        let n = self.field.n;
        let mut acc = Rational::ZERO;
        for i in 0..n {
            acc = acc.add(m.get(i, i));
        }
        acc
    }

    /// The characteristic polynomial `det(x·I − M_α) ∈ ℚ[x]`: monic of degree
    /// `n`, low-to-high coefficient order.
    pub fn char_poly(&self) -> Poly<Rational> {
        // `det(x·I − M_α)` via the division-free Samuelson–Berkowitz algorithm.
        Poly::new(self.mul_matrix().charpoly())
    }

    /// The minimal polynomial of `α` over ℚ: the monic annihilator of least
    /// degree. It equals the squarefree part of [`char_poly`](Self::char_poly)
    /// (which is a power of it) and therefore always divides it.
    pub fn min_poly(&self) -> Poly<Rational> {
        self.char_poly().squarefree_part()
    }

    /// Whether `α` is an **algebraic integer**, i.e. its characteristic
    /// polynomial (equivalently its minimal polynomial) has integer
    /// coefficients.
    pub fn is_algebraic_integer(&self) -> bool {
        let cp = self.char_poly();
        let n = cp.degree().unwrap_or(0);
        (0..=n).all(|i| cp.coeff(i).is_integer())
    }

    /// The `n` complex embeddings `σ_i(α) = α(θ_i)`, evaluating the
    /// representative at each complex root of `T`, to `precision` bits.
    ///
    /// Ordered to match [`NumberField::complex_roots`]. Cross-checks:
    /// `norm(α) = ∏ σ_i(α)` and `trace(α) = Σ σ_i(α)`.
    pub fn embeddings(&self, precision: u64) -> Vec<Complex<Float>> {
        let wp = precision + 64;
        let coeffs: Vec<Complex<Float>> = self
            .poly
            .coeffs()
            .iter()
            .map(|c| complex_from_rational(c, wp))
            .collect();
        durand_kerner(&self.field.t, precision)
            .iter()
            .map(|z| {
                let v = horner(&coeffs, z, wp);
                Complex::new(
                    v.re.round(precision, RoundingMode::Nearest),
                    v.im.round(precision, RoundingMode::Nearest),
                )
            })
            .collect()
    }
}

/// Whether two field contexts are the same field (pointer or value equality).
fn same_field(a: &Rc<FieldData>, b: &Rc<FieldData>) -> bool {
    Rc::ptr_eq(a, b) || a.t == b.t
}

/// Reduces `p` to its canonical representative of degree `< deg(T)` (`p mod T`).
fn reduce(p: &Poly<Rational>, t: &Poly<Rational>) -> Poly<Rational> {
    p.div_rem(t).1
}

/// Extended Euclidean algorithm in `ℚ[x]`: returns `(g, u, v)` with
/// `u·a + v·b = g`, where `g = gcd(a, b)`.
fn poly_ext_gcd(
    a: &Poly<Rational>,
    b: &Poly<Rational>,
) -> (Poly<Rational>, Poly<Rational>, Poly<Rational>) {
    let one = Poly::constant(Rational::ONE);
    let zero = Poly::<Rational>::zero();
    let (mut r0, mut r1) = (a.clone(), b.clone());
    let (mut s0, mut s1) = (one.clone(), zero.clone());
    let (mut t0, mut t1) = (zero, one);
    while !r1.is_zero() {
        let (q, r) = r0.div_rem(&r1);
        r0 = r1;
        r1 = r;
        let s = s0.sub(&q.mul(&s1));
        s0 = s1;
        s1 = s;
        let t = t0.sub(&q.mul(&t1));
        t0 = t1;
        t1 = t;
    }
    (r0, s0, t0)
}

/// The resultant `Res(a, b)` of two rational polynomials, as the determinant of
/// their Sylvester matrix.
fn resultant(a: &Poly<Rational>, b: &Poly<Rational>) -> Rational {
    let m = match a.degree() {
        Some(d) => d,
        None => return Rational::ZERO,
    };
    let dn = match b.degree() {
        Some(d) => d,
        None => return Rational::ZERO,
    };
    let size = m + dn;
    if size == 0 {
        // Both constants: Res of two nonzero constants is 1.
        return Rational::ONE;
    }
    // Sylvester matrix: `dn` rows of `a`'s coefficients (high→low) each shifted
    // right by one, then `m` rows of `b`'s coefficients likewise.
    let a_hi: Vec<Rational> = (0..=m).rev().map(|i| a.coeff(i)).collect();
    let b_hi: Vec<Rational> = (0..=dn).rev().map(|i| b.coeff(i)).collect();
    let mut data = alloc::vec![Rational::ZERO; size * size];
    for (row, r) in (0..dn).enumerate() {
        for (k, c) in a_hi.iter().enumerate() {
            data[row * size + r + k] = c.clone();
        }
    }
    for (idx, r) in (0..m).enumerate() {
        let row = dn + idx;
        for (k, c) in b_hi.iter().enumerate() {
            data[row * size + r + k] = c.clone();
        }
    }
    Matrix::new(size, size, data).determinant()
}

/// Builds a real [`Complex<Float>`](crate::complex::Complex) from a rational at
/// working precision `wp`.
fn complex_from_rational(r: &Rational, wp: u64) -> Complex<Float> {
    Complex::new(
        Float::from_rational(r, wp, RoundingMode::Nearest),
        Float::zero(wp),
    )
}

/// Horner evaluation of a polynomial (coefficients low-to-high) at a complex
/// point, at working precision `wp`.
fn horner(coeffs: &[Complex<Float>], z: &Complex<Float>, wp: u64) -> Complex<Float> {
    let mut acc = Complex::new(Float::zero(wp), Float::zero(wp));
    for c in coeffs.iter().rev() {
        acc = acc.mul(z).add(c);
    }
    acc
}

/// Finds all `n` complex roots of a (monic-after-scaling) rational polynomial by
/// the Durand–Kerner simultaneous iteration, each component to `precision` bits.
fn durand_kerner(t: &Poly<Rational>, precision: u64) -> Vec<Complex<Float>> {
    let n = t.degree().expect("durand_kerner: zero polynomial");
    let wp = precision + 64;
    let tm = t.monic();
    let coeffs: Vec<Complex<Float>> = (0..=n)
        .map(|i| complex_from_rational(&tm.coeff(i), wp))
        .collect();

    // Degree 1: single root −a₀ (monic x + a₀).
    if n == 1 {
        let root = coeffs[0].neg();
        return alloc::vec![Complex::new(
            root.re.round(precision, RoundingMode::Nearest),
            root.im.round(precision, RoundingMode::Nearest),
        )];
    }

    // Cauchy bound radius R = 1 + max|aᵢ| (i < n) for the initial layout.
    let mut radius = Float::from_int(&Int::ONE, wp, RoundingMode::Nearest);
    for c in coeffs.iter().take(n) {
        let a = c.abs();
        if a > radius {
            radius = a;
        }
    }
    let radius = Complex::new(radius, Float::zero(wp));

    // Seeds zₖ = R·g^k with g = 0.4 + 0.9i (off-axis, avoids symmetry traps).
    let g = Complex::new(
        Float::from_rational(
            &Rational::new(Int::from_i64(2), Int::from_i64(5)),
            wp,
            RoundingMode::Nearest,
        ),
        Float::from_rational(
            &Rational::new(Int::from_i64(9), Int::from_i64(10)),
            wp,
            RoundingMode::Nearest,
        ),
    );
    let mut z: Vec<Complex<Float>> = Vec::with_capacity(n);
    let mut p = radius.clone();
    for _ in 0..n {
        z.push(p.clone());
        p = p.mul(&g);
    }

    let one = Complex::new(
        Float::from_int(&Int::ONE, wp, RoundingMode::Nearest),
        Float::zero(wp),
    );
    let threshold = Float::from_rational(
        &Rational::power_of_two(-(precision as i32 + 8)),
        wp,
        RoundingMode::Nearest,
    );
    let max_iter = 400 + (precision as usize) / 4;
    for _ in 0..max_iter {
        let mut max_delta = Float::zero(wp);
        // Gauss–Seidel: update z[i] in place using the freshest neighbours.
        for i in 0..n {
            let num = horner(&coeffs, &z[i], wp);
            let mut den = one.clone();
            for j in 0..n {
                if j != i {
                    den = den.mul(&z[i].sub(&z[j]));
                }
            }
            let delta = num.div(&den);
            z[i] = z[i].sub(&delta);
            let d = delta.abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        if max_delta < threshold {
            break;
        }
    }

    z.into_iter()
        .map(|c| {
            Complex::new(
                c.re.round(precision, RoundingMode::Nearest),
                c.im.round(precision, RoundingMode::Nearest),
            )
        })
        .collect()
}

impl Ring for NumberFieldElement {
    // Exact rational arithmetic: associative and exact.
    const REASSOCIATIVE: bool = true;
    const EXACT: bool = true;
    #[inline]
    fn zero(&self) -> Self {
        NumberFieldElement {
            field: self.field.clone(),
            poly: Poly::zero(),
        }
    }
    #[inline]
    fn one(&self) -> Self {
        NumberFieldElement {
            field: self.field.clone(),
            poly: reduce(&Poly::constant(Rational::ONE), &self.field.t),
        }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.poly.is_zero()
    }
}

impl Field for NumberFieldElement {
    #[inline]
    fn inv(&self) -> Option<Self> {
        NumberFieldElement::inv(self)
    }
}

impl PartialEq for NumberFieldElement {
    fn eq(&self, other: &Self) -> bool {
        same_field(&self.field, &other.field) && self.poly == other.poly
    }
}

impl Eq for NumberFieldElement {}

impl fmt::Display for NumberFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.poly.is_zero() {
            return f.write_str("0");
        }
        let deg = self.poly.degree().unwrap_or(0);
        let mut first = true;
        for i in (0..=deg).rev() {
            let c = self.poly.coeff(i);
            if c.is_zero() {
                continue;
            }
            if !first {
                f.write_str(" + ")?;
            }
            first = false;
            match i {
                0 => write!(f, "{c}")?,
                1 if c.is_one() => f.write_str("θ")?,
                1 => write!(f, "{c}·θ")?,
                _ if c.is_one() => write!(f, "θ^{i}")?,
                _ => write!(f, "{c}·θ^{i}")?,
            }
        }
        Ok(())
    }
}

impl fmt::Debug for NumberFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NumberFieldElement({self} in {:?})", self.field())
    }
}

impl fmt::Debug for NumberField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NumberField(ℚ[x]/({}))", self.data.t)
    }
}

macro_rules! nf_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for NumberFieldElement {
            type Output = NumberFieldElement;
            #[inline]
            fn $m(self, rhs: NumberFieldElement) -> NumberFieldElement {
                NumberFieldElement::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&NumberFieldElement> for NumberFieldElement {
            type Output = NumberFieldElement;
            #[inline]
            fn $m(self, rhs: &NumberFieldElement) -> NumberFieldElement {
                NumberFieldElement::$m(&self, rhs)
            }
        }
        impl core::ops::$tr<NumberFieldElement> for &NumberFieldElement {
            type Output = NumberFieldElement;
            #[inline]
            fn $m(self, rhs: NumberFieldElement) -> NumberFieldElement {
                NumberFieldElement::$m(self, &rhs)
            }
        }
        impl core::ops::$tr<&NumberFieldElement> for &NumberFieldElement {
            type Output = NumberFieldElement;
            #[inline]
            fn $m(self, rhs: &NumberFieldElement) -> NumberFieldElement {
                NumberFieldElement::$m(self, rhs)
            }
        }
        impl core::ops::$atr<NumberFieldElement> for NumberFieldElement {
            #[inline]
            fn $am(&mut self, rhs: NumberFieldElement) {
                *self = NumberFieldElement::$m(self, &rhs);
            }
        }
        impl core::ops::$atr<&NumberFieldElement> for NumberFieldElement {
            #[inline]
            fn $am(&mut self, rhs: &NumberFieldElement) {
                *self = NumberFieldElement::$m(self, rhs);
            }
        }
    };
}

nf_binop!(Add, add, AddAssign, add_assign);
nf_binop!(Sub, sub, SubAssign, sub_assign);
nf_binop!(Mul, mul, MulAssign, mul_assign);
nf_binop!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for NumberFieldElement {
    type Output = NumberFieldElement;
    #[inline]
    fn neg(self) -> NumberFieldElement {
        NumberFieldElement::neg(&self)
    }
}
impl core::ops::Neg for &NumberFieldElement {
    type Output = NumberFieldElement;
    #[inline]
    fn neg(self) -> NumberFieldElement {
        NumberFieldElement::neg(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn q(n: i64) -> Rational {
        Rational::from_integer(Int::from_i64(n))
    }

    fn qd(n: i64, d: i64) -> Rational {
        Rational::new(Int::from_i64(n), Int::from_i64(d))
    }

    /// Poly from integer coefficients, low-to-high.
    fn poly(coeffs: &[i64]) -> Poly<Rational> {
        Poly::new(coeffs.iter().map(|&c| q(c)).collect())
    }

    // ℚ(i): x² + 1
    fn field_i() -> NumberField {
        NumberField::new(poly(&[1, 0, 1])).unwrap()
    }
    // ℚ(√2): x² − 2
    fn field_sqrt2() -> NumberField {
        NumberField::new(poly(&[-2, 0, 1])).unwrap()
    }
    // ℚ(∛2): x³ − 2
    fn field_cbrt2() -> NumberField {
        NumberField::new(poly(&[-2, 0, 0, 1])).unwrap()
    }
    // 5th cyclotomic: x⁴ + x³ + x² + x + 1
    fn field_cyclo5() -> NumberField {
        NumberField::new(poly(&[1, 1, 1, 1, 1])).unwrap()
    }
    // A degree-5 irreducible: x⁵ − x − 1
    fn field_deg5() -> NumberField {
        NumberField::new(poly(&[-1, -1, 0, 0, 0, 1])).unwrap()
    }

    #[test]
    fn rejects_reducible() {
        // x² − 1 = (x−1)(x+1)
        assert!(NumberField::new(poly(&[-1, 0, 1])).is_none());
        // x² (not squarefree, reducible)
        assert!(NumberField::new(poly(&[0, 0, 1])).is_none());
        // constant / degree 0
        assert!(NumberField::new(poly(&[5])).is_none());
        // x³ − 1 = (x−1)(x²+x+1)
        assert!(NumberField::new(poly(&[-1, 0, 0, 1])).is_none());
    }

    #[test]
    fn accepts_irreducible() {
        assert_eq!(field_i().degree(), 2);
        assert_eq!(field_sqrt2().degree(), 2);
        assert_eq!(field_cbrt2().degree(), 3);
        assert_eq!(field_cyclo5().degree(), 4);
        assert_eq!(field_deg5().degree(), 5);
    }

    #[test]
    fn inverse_roundtrips() {
        for k in [
            field_i(),
            field_sqrt2(),
            field_cbrt2(),
            field_cyclo5(),
            field_deg5(),
        ] {
            let theta = k.generator();
            let a = theta.add(&k.from_rational(q(3))); // θ + 3
            let ainv = a.inv().unwrap();
            assert!(a.mul(&ainv).is_one());
            // i² = −1 style check via generator powers
            let b = theta.mul(&theta).sub(&k.from_rational(q(1)));
            if !b.is_zero() {
                assert!(b.mul(&b.inv().unwrap()).is_one());
            }
        }
    }

    #[test]
    fn i_squared_is_minus_one() {
        let k = field_i();
        let i = k.generator();
        assert_eq!(i.mul(&i), k.from_rational(q(-1)));
    }

    #[test]
    fn norm_multiplicative_trace_additive() {
        for k in [
            field_i(),
            field_sqrt2(),
            field_cbrt2(),
            field_cyclo5(),
            field_deg5(),
        ] {
            let theta = k.generator();
            let a = theta.mul(&theta).add(&k.from_rational(q(2))); // θ² + 2
            let b = theta.add(&k.from_rational(qd(-3, 2))); // θ − 3/2
            // norm(ab) = norm(a) norm(b)
            let lhs = a.mul(&b).norm();
            let rhs = a.norm().mul(&b.norm());
            assert_eq!(lhs, rhs);
            // trace(a+b) = trace(a) + trace(b)
            let tl = a.add(&b).trace();
            let tr = a.trace().add(&b.trace());
            assert_eq!(tl, tr);
        }
    }

    #[test]
    fn norm_of_rational_is_power() {
        for k in [field_i(), field_cbrt2(), field_cyclo5(), field_deg5()] {
            let n = k.degree() as i32;
            let r = qd(5, 3);
            let elt = k.from_rational(r.clone());
            assert_eq!(elt.norm(), r.pow(n));
        }
    }

    /// Evaluate a rational polynomial at a field element (Σ cᵢ αⁱ).
    fn eval_poly_at(
        cp: &Poly<Rational>,
        alpha: &NumberFieldElement,
        k: &NumberField,
    ) -> NumberFieldElement {
        let deg = cp.degree().unwrap_or(0);
        let mut acc = k.zero();
        let mut power = k.one();
        for i in 0..=deg {
            acc = acc.add(&power.mul(&k.from_rational(cp.coeff(i))));
            if i < deg {
                power = power.mul(alpha);
            }
        }
        acc
    }

    #[test]
    fn cayley_hamilton_and_minpoly_divides() {
        for k in [
            field_i(),
            field_sqrt2(),
            field_cbrt2(),
            field_cyclo5(),
            field_deg5(),
        ] {
            let theta = k.generator();
            let a = theta.mul(&theta).sub(&theta).add(&k.from_rational(q(1)));
            let cp = a.char_poly();
            assert_eq!(cp.degree(), Some(k.degree()));
            // char_poly(a) = 0 in K
            assert!(eval_poly_at(&cp, &a, &k).is_zero());
            // min_poly | char_poly
            let mp = a.min_poly();
            let (q_, r_) = cp.div_rem(&mp);
            assert!(r_.is_zero());
            let _ = q_;
            // min_poly(a) = 0 too
            assert!(eval_poly_at(&mp, &a, &k).is_zero());
        }
    }

    #[test]
    fn generator_is_algebraic_integer() {
        for k in [
            field_i(),
            field_sqrt2(),
            field_cbrt2(),
            field_cyclo5(),
            field_deg5(),
        ] {
            assert!(k.generator().is_algebraic_integer());
            // char_poly(θ) = T
            let cp = k.generator().char_poly();
            assert_eq!(cp, *k.defining_polynomial());
        }
        // Non-integer: θ/2 in ℚ(√2) has char poly x² − 1/2, not integral.
        let k = field_sqrt2();
        let half_theta = k.generator().mul(&k.from_rational(qd(1, 2)));
        assert!(!half_theta.is_algebraic_integer());
    }

    #[test]
    fn discriminants_known() {
        // disc(x² + 1) = −4
        assert_eq!(field_i().discriminant(), q(-4));
        // disc(x² − d) = 4d
        for d in [2, 3, 5, 7] {
            let k = NumberField::new(poly(&[-d, 0, 1])).unwrap();
            assert_eq!(k.discriminant(), q(4 * d));
        }
        // disc(x³ − 2) = −27·4 = −108  (disc(x³+q) = −27q²; q=−2 → −108)
        assert_eq!(field_cbrt2().discriminant(), q(-108));
        // disc of 5th cyclotomic = 125
        assert_eq!(field_cyclo5().discriminant(), q(125));
    }

    #[test]
    fn signatures() {
        assert_eq!(field_i().signature(), (0, 1)); // no real embeddings
        assert_eq!(field_sqrt2().signature(), (2, 0)); // ±√2 real
        assert_eq!(field_cbrt2().signature(), (1, 1)); // one real cube root
        assert_eq!(field_cyclo5().signature(), (0, 2)); // all complex
        assert_eq!(field_deg5().signature(), (1, 2)); // x⁵−x−1 has one real root
    }

    #[test]
    fn ring_field_traits() {
        let k = field_cbrt2();
        let theta = k.generator();
        let one = k.one();
        // via operators
        let a = &theta + &one;
        let b = &a * &theta;
        assert_eq!(b, theta.mul(&theta).add(&theta));
        assert_eq!((&a - &one), theta);
        // pow
        assert_eq!(theta.pow(3), k.from_rational(q(2))); // θ³ = 2
        assert_eq!(theta.pow(0), one);
        assert_eq!(theta.pow(-1), theta.inv().unwrap());
        // Ring trait
        assert!(Ring::is_zero(&k.zero()));
        assert_eq!(Ring::one(&theta), one);
    }

    #[test]
    fn display_smoke() {
        let k = field_cbrt2();
        let theta = k.generator();
        let e = theta.mul(&theta).add(&theta).add(&k.from_rational(q(1)));
        assert_eq!(e.to_string(), "θ^2 + θ + 1");
        assert_eq!(k.zero().to_string(), "0");
    }

    #[test]
    fn embeddings_low_precision() {
        // ℚ(∛2): the real root ≈ 1.2599, checkable at modest precision.
        let k = field_cbrt2();
        let roots = k.complex_roots(64);
        assert_eq!(roots.len(), 3);
        // product of roots of x³ − 2 = 2 (since −(−2)/1 for odd degree → 2)
        // Trace check via θ: Σ σ_i(θ) = 0 (coeff of x² is 0).
        let embs = k.generator().embeddings(64);
        let mut sum = Complex::new(Float::zero(80), Float::zero(80));
        for e in &embs {
            sum = sum.add(e);
        }
        assert!(sum.abs().to_f64().abs() < 1e-12);
    }

    // ---- Slow, high-precision numerical cross-checks (embeddings). ----

    #[test]
    #[ignore = "slow high-precision embedding cross-check"]
    fn norm_trace_via_embeddings() {
        let prec = 256u64;
        let wp = prec + 64;
        for k in [
            field_i(),
            field_sqrt2(),
            field_cbrt2(),
            field_cyclo5(),
            field_deg5(),
        ] {
            let theta = k.generator();
            let a = theta
                .mul(&theta)
                .add(&theta)
                .sub(&k.from_rational(qd(7, 3)));
            let embs = a.embeddings(prec);
            // Σ σ_i(a) ≈ trace(a)
            let mut sum = Complex::new(Float::zero(wp), Float::zero(wp));
            let mut prod = Complex::new(
                Float::from_int(&Int::ONE, wp, RoundingMode::Nearest),
                Float::zero(wp),
            );
            for e in &embs {
                sum = sum.add(e);
                prod = prod.mul(e);
            }
            let tr = Float::from_rational(&a.trace(), wp, RoundingMode::Nearest);
            let nm = Float::from_rational(&a.norm(), wp, RoundingMode::Nearest);
            assert!(
                sum.re
                    .sub(&tr, wp, RoundingMode::Nearest)
                    .abs()
                    .to_f64()
                    .abs()
                    < 1e-30
            );
            assert!(sum.im.abs().to_f64().abs() < 1e-30);
            assert!(
                prod.re
                    .sub(&nm, wp, RoundingMode::Nearest)
                    .abs()
                    .to_f64()
                    .abs()
                    < 1e-30
            );
            assert!(prod.im.abs().to_f64().abs() < 1e-30);
        }
    }
}
