//! Finite (Galois) field extensions `GF(pᵏ)`.
//!
//! [`ModInt`](crate::mod_int::ModInt) already models the prime field `GF(p) =
//! ℤ/pℤ`. This module builds the *extension* fields `GF(pᵏ)` for `k ≥ 1`,
//! represented as `GF(p)[x] / (f)` for a monic irreducible modulus polynomial
//! `f` of degree `k` over `GF(p)`.
//!
//! A [`GaloisField`] is the shared field context: the prime `p`, the degree
//! `k`, and the modulus `f` (as coefficients low-to-high, each reduced into
//! `[0, p)`, leading coefficient `1`). The data is wrapped in an
//! [`Rc`] so [`GfElement`]s produced from one field share it
//! cheaply, exactly as [`ModInt`](crate::mod_int::ModInt) shares its modulus.
//!
//! A [`GfElement`] is the residue class of a polynomial, stored as its canonical
//! representative of degree `< k` (a length-`k` coefficient vector, each entry in
//! `[0, p)`). Elements of *different* fields must not be mixed: every binary
//! operation panics on a field mismatch, mirroring [`ModInt`](crate::mod_int::ModInt).
//!
//! Irreducibility is decided by **Rabin's test** (Lidl & Niederreiter, *Finite
//! Fields*, Thm. 3.x; Menezes et al., *Handbook of Applied Cryptography* §4.5.1):
//! a monic degree-`k` polynomial `f` is irreducible over `GF(p)` iff
//! `x^(pᵏ) ≡ x (mod f)` and `gcd(x^(p^{k/q}) − x, f) = 1` for every prime
//! divisor `q` of `k`. The powers `x^(pᵐ)` are built by iterating the Frobenius
//! (`p`-th power modulo `f`) `m` times.
//!
//! This is a clean-room implementation drawn from the open literature (Lidl &
//! Niederreiter; the HAC §2.6/§4.5; Cohen, *A Course in Computational Algebraic
//! Number Theory* §3.4–3.6; von zur Gathen & Gerhard, *Modern Computer
//! Algebra*); it copies no external source.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::fmt;

use crate::int::Int;

// ===========================================================================
// Internal `GF(p)[x]` arithmetic on `Vec<Int>` (coefficients low-to-high, each
// reduced into `[0, p)`, trailing zeros trimmed so the leading entry is nonzero;
// the empty vector is the zero polynomial).
// ===========================================================================

/// Highest index of a nonzero coefficient, or `None` for the zero polynomial.
fn deg_of(a: &[Int]) -> Option<usize> {
    a.iter().rposition(|c| !c.is_zero())
}

/// Drops trailing (high-degree) zero coefficients.
fn poly_trim(mut a: Vec<Int>) -> Vec<Int> {
    while a.last().is_some_and(Int::is_zero) {
        a.pop();
    }
    a
}

/// Pads (or truncates a trimmed poly) to exactly `len` coefficients with zeros.
fn pad_to(mut a: Vec<Int>, len: usize) -> Vec<Int> {
    a.truncate(len);
    while a.len() < len {
        a.push(Int::ZERO);
    }
    a
}

/// Coefficientwise `self − rhs` modulo `p`.
fn poly_sub(a: &[Int], b: &[Int], p: &Int) -> Vec<Int> {
    let n = a.len().max(b.len());
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(i).cloned().unwrap_or(Int::ZERO);
        let y = b.get(i).cloned().unwrap_or(Int::ZERO);
        r.push(x.sub(&y).rem_euclid(p));
    }
    poly_trim(r)
}

/// Schoolbook `self · rhs`, reducing every coefficient modulo `p`.
fn poly_mul(a: &[Int], b: &[Int], p: &Int) -> Vec<Int> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = alloc::vec![Int::ZERO; a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            r[i + j] = r[i + j].add(&ai.mul(bj));
        }
    }
    poly_trim(r.into_iter().map(|c| c.rem_euclid(p)).collect())
}

/// `self · scalar` modulo `p`.
fn poly_scale(a: &[Int], s: &Int, p: &Int) -> Vec<Int> {
    poly_trim(a.iter().map(|c| c.mul(s).rem_euclid(p)).collect())
}

/// Divides `a` by nonzero `b` over `GF(p)`, returning `(quotient, remainder)`
/// with `deg(remainder) < deg(b)`. The leading coefficient of `b` is inverted
/// modulo the prime `p`.
fn poly_divmod(a: &[Int], b: &[Int], p: &Int) -> (Vec<Int>, Vec<Int>) {
    let db = deg_of(b).expect("poly_divmod: division by zero polynomial");
    let inv_lc = b[db]
        .modinv(p)
        .expect("poly_divmod: leading coefficient must be invertible mod p");
    let mut r = poly_trim(a.to_vec());
    let rd = match deg_of(&r) {
        Some(d) if d >= db => d,
        _ => return (Vec::new(), r),
    };
    let mut q = alloc::vec![Int::ZERO; rd - db + 1];
    while let Some(rd) = deg_of(&r) {
        if rd < db {
            break;
        }
        let d = rd - db;
        let coef = r[rd].mul(&inv_lc).rem_euclid(p);
        for (j, bj) in b.iter().enumerate() {
            r[d + j] = r[d + j].sub(&coef.mul(bj)).rem_euclid(p);
        }
        q[d] = coef;
        r = poly_trim(r);
    }
    (poly_trim(q), r)
}

/// Remainder of `a` divided by nonzero `b` over `GF(p)`.
fn poly_rem(a: &[Int], b: &[Int], p: &Int) -> Vec<Int> {
    poly_divmod(a, b, p).1
}

/// Monic greatest common divisor of `a` and `b` over `GF(p)` (Euclid).
fn poly_gcd(a: &[Int], b: &[Int], p: &Int) -> Vec<Int> {
    let mut a = poly_trim(a.to_vec());
    let mut b = poly_trim(b.to_vec());
    while !b.is_empty() {
        let r = poly_rem(&a, &b, p);
        a = b;
        b = r;
    }
    // Normalize to monic.
    match deg_of(&a) {
        None => a,
        Some(d) => {
            if a[d].is_one() {
                a
            } else {
                let inv = a[d]
                    .modinv(p)
                    .expect("poly_gcd: leading coefficient invertible mod p");
                poly_scale(&a, &inv, p)
            }
        }
    }
}

/// `base^e mod modulus` in `GF(p)[x]` (square-and-multiply, exponent an [`Int`]).
fn poly_powmod(base: &[Int], e: &Int, modulus: &[Int], p: &Int) -> Vec<Int> {
    let mut result = poly_rem(&[Int::ONE], modulus, p);
    let mut b = poly_rem(base, modulus, p);
    for i in 0..e.bit_len() {
        if e.bit(i) {
            result = poly_rem(&poly_mul(&result, &b, p), modulus, p);
        }
        b = poly_rem(&poly_mul(&b, &b, p), modulus, p);
    }
    result
}

/// `x^(pᵐ) mod f`, built by iterating the Frobenius (`p`-th power mod `f`) `m`
/// times starting from `x`.
fn frobenius_pow(f: &[Int], m: usize, p: &Int) -> Vec<Int> {
    let x = alloc::vec![Int::ZERO, Int::ONE];
    let mut h = poly_rem(&x, f, p);
    for _ in 0..m {
        h = poly_powmod(&h, p, f, p);
    }
    h
}

/// Distinct prime divisors of `k` (ascending).
fn distinct_primes(k: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for f in Int::from_u64(k as u64).factorize() {
        let v = f.to_u64().expect("small factor fits in u64") as usize;
        if out.last() != Some(&v) {
            out.push(v);
        }
    }
    out
}

/// Rabin's irreducibility test for a **monic** polynomial `f` of degree `≥ 1`
/// over `GF(p)`.
fn is_irreducible(f: &[Int], p: &Int) -> bool {
    let k = match deg_of(f) {
        Some(d) if d >= 1 => d,
        _ => return false,
    };
    if k == 1 {
        return true; // every monic linear polynomial is irreducible
    }
    let x = alloc::vec![Int::ZERO, Int::ONE];
    // (2) gcd(x^(p^{k/q}) − x, f) = 1 for every prime q | k.
    for q in distinct_primes(k) {
        let h = frobenius_pow(f, k / q, p);
        let diff = poly_sub(&h, &x, p);
        let g = poly_gcd(&diff, f, p);
        if deg_of(&g).is_some_and(|d| d >= 1) {
            return false; // a nontrivial common factor: reducible
        }
    }
    // (1) x^(pᵏ) ≡ x (mod f).
    let h = frobenius_pow(f, k, p);
    poly_sub(&h, &x, p).is_empty()
}

/// Extended Euclid in `GF(p)[x]`: the inverse `u` of `a` modulo `modulus`
/// (`u·a ≡ 1`), or `None` when `a ≡ 0` or `a` is not invertible.
fn poly_inv(a: &[Int], modulus: &[Int], p: &Int) -> Option<Vec<Int>> {
    let a = poly_trim(a.to_vec());
    if a.is_empty() {
        return None;
    }
    // Track only the coefficient of `a`: sᵢ·a ≡ rᵢ (mod modulus).
    let (mut r0, mut r1) = (modulus.to_vec(), a);
    let (mut s0, mut s1): (Vec<Int>, Vec<Int>) = (Vec::new(), alloc::vec![Int::ONE]);
    while !r1.is_empty() {
        let (q, r) = poly_divmod(&r0, &r1, p);
        let s = poly_sub(&s0, &poly_mul(&q, &s1, p), p);
        r0 = r1;
        r1 = r;
        s0 = s1;
        s1 = s;
    }
    // gcd = r0 must be a nonzero constant (guaranteed when `modulus` is
    // irreducible and `a ≠ 0`); otherwise `a` shares a factor and has no inverse.
    if deg_of(&r0) != Some(0) {
        return None;
    }
    let inv_c = r0[0].modinv(p)?;
    Some(poly_rem(&poly_scale(&s0, &inv_c, p), modulus, p))
}

/// Increments `digits` as a little-endian base-`p` counter; returns `false` on
/// wrap-around (every digit was `p−1`).
fn incr_base_p(digits: &mut [Int], p: &Int) -> bool {
    for d in digits.iter_mut() {
        *d = d.add(&Int::ONE);
        if *d < *p {
            return true;
        }
        *d = Int::ZERO;
    }
    false
}

// ===========================================================================
// Public types.
// ===========================================================================

/// The shared field context of `GF(pᵏ)`: the prime `p`, the degree `k`, and the
/// monic irreducible modulus polynomial.
struct FieldData {
    /// The characteristic (a prime).
    p: Int,
    /// The extension degree.
    k: usize,
    /// The modulus `f`, low-to-high, length `k+1`, monic, each entry in `[0, p)`.
    modulus: Vec<Int>,
}

/// A finite field `GF(pᵏ) = GF(p)[x] / (f)` for a monic irreducible `f`.
///
/// The field data (prime, degree, modulus) is reference-counted, so
/// [`GfElement`]s built from it share the context cheaply.
#[derive(Clone)]
pub struct GaloisField {
    data: Rc<FieldData>,
}

/// An element of a [`GaloisField`]: the residue class of a polynomial, stored as
/// its canonical representative of degree `< k`.
#[derive(Clone)]
pub struct GfElement {
    field: Rc<FieldData>,
    /// Coefficients low-to-high, length exactly `k`, each in `[0, p)`.
    coeffs: Vec<Int>,
}

impl GaloisField {
    /// Builds `GF(pᵏ)` from an explicit modulus.
    ///
    /// `p` must be prime and `modulus` a **monic** polynomial (low-to-high
    /// coefficients, leading coefficient `1` after reduction mod `p`) of degree
    /// `k ≥ 1` that is **irreducible** over `GF(p)`. Returns `None` if `p` is not
    /// prime, if the modulus is not monic of positive degree, or if it is
    /// reducible.
    pub fn new(p: Int, modulus: &[Int]) -> Option<GaloisField> {
        if !p.is_prime_bpsw() {
            return None;
        }
        let reduced: Vec<Int> = modulus.iter().map(|c| c.rem_euclid(&p)).collect();
        if reduced.len() < 2 {
            return None; // degree must be ≥ 1
        }
        let k = reduced.len() - 1;
        if !reduced[k].is_one() {
            return None; // not monic (leading coefficient must be 1)
        }
        if !is_irreducible(&reduced, &p) {
            return None;
        }
        Some(GaloisField {
            data: Rc::new(FieldData {
                p,
                k,
                modulus: reduced,
            }),
        })
    }

    /// Builds `GF(pᵏ)`, automatically finding a monic irreducible modulus of
    /// degree `k` over `GF(p)`.
    ///
    /// Candidate monic polynomials `xᵏ + c_{k-1}xᵏ⁻¹ + … + c₀` are enumerated
    /// deterministically (the low coefficients as a base-`p` counter) and each is
    /// tested with Rabin's irreducibility test; the first irreducible one is
    /// used. Returns `None` if `p` is not prime or `k == 0`.
    pub fn create(p: Int, k: usize) -> Option<GaloisField> {
        if k == 0 || !p.is_prime_bpsw() {
            return None;
        }
        let mut low = alloc::vec![Int::ZERO; k];
        loop {
            let mut modulus = low.clone();
            modulus.push(Int::ONE); // monic degree-k candidate
            if is_irreducible(&modulus, &p) {
                return Some(GaloisField {
                    data: Rc::new(FieldData { p, k, modulus }),
                });
            }
            if !incr_base_p(&mut low, &p) {
                return None; // exhausted (unreachable: irreducibles always exist)
            }
        }
    }

    /// The characteristic `p`.
    #[inline]
    pub fn characteristic(&self) -> Int {
        self.data.p.clone()
    }

    /// The extension degree `k`.
    #[inline]
    pub fn degree(&self) -> usize {
        self.data.k
    }

    /// The field order `pᵏ` (the number of elements).
    #[inline]
    pub fn order(&self) -> Int {
        self.data.p.pow(self.data.k as u32)
    }

    /// The modulus polynomial coefficients, low-to-high (length `k+1`, monic).
    #[inline]
    pub fn modulus(&self) -> &[Int] {
        &self.data.modulus
    }

    /// Builds an element from polynomial coefficients (low-to-high); the input is
    /// reduced modulo `p` and modulo the field modulus, so any length is accepted.
    pub fn element(&self, coeffs: &[Int]) -> GfElement {
        let reduced: Vec<Int> = coeffs.iter().map(|c| c.rem_euclid(&self.data.p)).collect();
        let residue = poly_rem(&reduced, &self.data.modulus, &self.data.p);
        GfElement {
            field: self.data.clone(),
            coeffs: pad_to(residue, self.data.k),
        }
    }

    /// The additive identity `0`.
    #[inline]
    pub fn zero(&self) -> GfElement {
        GfElement {
            field: self.data.clone(),
            coeffs: alloc::vec![Int::ZERO; self.data.k],
        }
    }

    /// The multiplicative identity `1`.
    pub fn one(&self) -> GfElement {
        self.from_int(&Int::ONE)
    }

    /// The generator `a`, the residue class of `x` (`[0, 1, 0, …]`).
    pub fn generator(&self) -> GfElement {
        self.element(&[Int::ZERO, Int::ONE])
    }

    /// The constant element embedding a `GF(p)` value `c` (`c mod p`).
    pub fn from_int(&self, c: &Int) -> GfElement {
        let mut coeffs = alloc::vec![Int::ZERO; self.data.k];
        coeffs[0] = c.rem_euclid(&self.data.p);
        GfElement {
            field: self.data.clone(),
            coeffs,
        }
    }

    /// The Frobenius endomorphism `elem ↦ elemᵖ`. It fixes `GF(p)`, satisfies
    /// `(a+b)ᵖ = aᵖ + bᵖ`, and `elem^(pᵏ) = elem` for every element. Panics if
    /// `elem` belongs to a different field.
    pub fn frobenius(&self, elem: &GfElement) -> GfElement {
        assert!(
            same_field(&self.data, &elem.field),
            "GaloisField::frobenius: element from a different field"
        );
        elem.pow(&self.data.p)
    }

    /// A primitive element: a generator of the (cyclic) multiplicative group
    /// `GF(pᵏ)*`, i.e. an element of multiplicative order `pᵏ − 1`.
    ///
    /// Candidates are enumerated deterministically and each is checked with the
    /// order test `g^{(pᵏ−1)/q} ≠ 1` for every prime `q | pᵏ − 1` (reusing
    /// integer factorization). Such an element always exists.
    pub fn primitive_element(&self) -> GfElement {
        let one = self.one();
        let n = self.order().sub(&Int::ONE); // pᵏ − 1
        // Distinct prime divisors of the group order.
        let mut primes: Vec<Int> = Vec::new();
        for q in n.factorize() {
            if primes.last() != Some(&q) {
                primes.push(q);
            }
        }
        let mut digits = alloc::vec![Int::ZERO; self.data.k];
        while incr_base_p(&mut digits, &self.data.p) {
            let cand = self.element(&digits);
            if cand.is_zero() {
                continue;
            }
            let is_primitive = primes.iter().all(|q| cand.pow(&n.div_exact(q)) != one);
            if is_primitive {
                return cand;
            }
        }
        unreachable!("a primitive element always exists in a finite field");
    }
}

/// Whether two field contexts are the same field (pointer or value equality).
fn same_field(a: &Rc<FieldData>, b: &Rc<FieldData>) -> bool {
    Rc::ptr_eq(a, b) || (a.p == b.p && a.modulus == b.modulus)
}

impl GfElement {
    /// The field this element belongs to.
    pub fn field(&self) -> GaloisField {
        GaloisField {
            data: self.field.clone(),
        }
    }

    /// The canonical residue coefficients, low-to-high (length `k`, each in
    /// `[0, p)`).
    #[inline]
    pub fn to_coefficients(&self) -> &[Int] {
        &self.coeffs
    }

    /// Returns `true` if this is the zero element.
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(Int::is_zero)
    }

    /// Returns `true` if this is the multiplicative identity `1`.
    pub fn is_one(&self) -> bool {
        self.coeffs[0].is_one() && self.coeffs[1..].iter().all(Int::is_zero)
    }

    fn same_field(&self, other: &GfElement) {
        assert!(
            same_field(&self.field, &other.field),
            "GfElement: operands from different fields"
        );
    }

    #[inline]
    fn wrap(&self, coeffs: Vec<Int>) -> GfElement {
        GfElement {
            field: self.field.clone(),
            coeffs,
        }
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &GfElement) -> GfElement {
        self.same_field(rhs);
        let p = &self.field.p;
        let coeffs = (0..self.field.k)
            .map(|i| self.coeffs[i].add(&rhs.coeffs[i]).rem_euclid(p))
            .collect();
        self.wrap(coeffs)
    }

    /// Returns `self − rhs`.
    pub fn sub(&self, rhs: &GfElement) -> GfElement {
        self.same_field(rhs);
        let p = &self.field.p;
        let coeffs = (0..self.field.k)
            .map(|i| self.coeffs[i].sub(&rhs.coeffs[i]).rem_euclid(p))
            .collect();
        self.wrap(coeffs)
    }

    /// Returns `−self`.
    pub fn neg(&self) -> GfElement {
        let p = &self.field.p;
        let coeffs = self.coeffs.iter().map(|c| c.neg().rem_euclid(p)).collect();
        self.wrap(coeffs)
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &GfElement) -> GfElement {
        self.same_field(rhs);
        let p = &self.field.p;
        let prod = poly_mul(&self.coeffs, &rhs.coeffs, p);
        let residue = poly_rem(&prod, &self.field.modulus, p);
        self.wrap(pad_to(residue, self.field.k))
    }

    /// Returns the multiplicative inverse `self⁻¹`, or `None` if `self` is zero.
    ///
    /// Computed by the extended Euclidean algorithm in `GF(p)[x]`; cross-checks
    /// against `self^(pᵏ−2)`.
    pub fn inv(&self) -> Option<GfElement> {
        let p = &self.field.p;
        let u = poly_inv(&self.coeffs, &self.field.modulus, p)?;
        Some(self.wrap(pad_to(u, self.field.k)))
    }

    /// Returns `self / rhs = self · rhs⁻¹`. Panics if `rhs` is zero.
    pub fn div(&self, rhs: &GfElement) -> GfElement {
        self.mul(
            &rhs.inv()
                .expect("GfElement::div: divisor is not invertible"),
        )
    }

    /// Returns `self` raised to `exp` (negative exponents invert first).
    pub fn pow(&self, exp: &Int) -> GfElement {
        if exp.is_negative() {
            return self
                .inv()
                .expect("GfElement::pow: base not invertible for a negative exponent")
                .pow(&exp.abs());
        }
        // Square-and-multiply, low bit first.
        let mut result = GfElement {
            field: self.field.clone(),
            coeffs: {
                let mut c = alloc::vec![Int::ZERO; self.field.k];
                c[0] = Int::ONE;
                c
            },
        };
        let mut base = self.clone();
        for i in 0..exp.bit_len() {
            if exp.bit(i) {
                result = result.mul(&base);
            }
            base = base.mul(&base);
        }
        result
    }
}

impl PartialEq for GfElement {
    fn eq(&self, other: &Self) -> bool {
        same_field(&self.field, &other.field) && self.coeffs == other.coeffs
    }
}

impl Eq for GfElement {}

impl fmt::Display for GfElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate().rev() {
            if c.is_zero() {
                continue;
            }
            if !first {
                f.write_str(" + ")?;
            }
            first = false;
            match i {
                0 => write!(f, "{c}")?,
                1 if c.is_one() => f.write_str("a")?,
                1 => write!(f, "{c}·a")?,
                _ if c.is_one() => write!(f, "a^{i}")?,
                _ => write!(f, "{c}·a^{i}")?,
            }
        }
        Ok(())
    }
}

impl fmt::Debug for GfElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GfElement({} in GF({}^{}))",
            self, self.field.p, self.field.k
        )
    }
}

impl fmt::Debug for GaloisField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GaloisField(GF({}^{}))", self.data.p, self.data.k)
    }
}

macro_rules! gf_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for GfElement {
            type Output = GfElement;
            #[inline]
            fn $m(self, rhs: GfElement) -> GfElement {
                GfElement::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&GfElement> for GfElement {
            type Output = GfElement;
            #[inline]
            fn $m(self, rhs: &GfElement) -> GfElement {
                GfElement::$m(&self, rhs)
            }
        }
        impl core::ops::$tr<GfElement> for &GfElement {
            type Output = GfElement;
            #[inline]
            fn $m(self, rhs: GfElement) -> GfElement {
                GfElement::$m(self, &rhs)
            }
        }
        impl core::ops::$tr<&GfElement> for &GfElement {
            type Output = GfElement;
            #[inline]
            fn $m(self, rhs: &GfElement) -> GfElement {
                GfElement::$m(self, rhs)
            }
        }
        impl core::ops::$atr<GfElement> for GfElement {
            #[inline]
            fn $am(&mut self, rhs: GfElement) {
                *self = GfElement::$m(self, &rhs);
            }
        }
        impl core::ops::$atr<&GfElement> for GfElement {
            #[inline]
            fn $am(&mut self, rhs: &GfElement) {
                *self = GfElement::$m(self, rhs);
            }
        }
    };
}

gf_binop!(Add, add, AddAssign, add_assign);
gf_binop!(Sub, sub, SubAssign, sub_assign);
gf_binop!(Mul, mul, MulAssign, mul_assign);
gf_binop!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for GfElement {
    type Output = GfElement;
    #[inline]
    fn neg(self) -> GfElement {
        GfElement::neg(&self)
    }
}
impl core::ops::Neg for &GfElement {
    type Output = GfElement;
    #[inline]
    fn neg(self) -> GfElement {
        GfElement::neg(self)
    }
}
