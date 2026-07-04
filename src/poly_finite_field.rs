//! Univariate polynomial factorization over a finite field `GF(q)`
//! (Cantor–Zassenhaus).
//!
//! Where [`Poly<Rational>::factor`](crate::poly::Poly::factor) factors over `ℚ`
//! (Berlekamp–Zassenhaus), this module factors over any *finite* field — the
//! prime fields `GF(p)` ([`ModInt`](crate::mod_int::ModInt) with a prime
//! modulus) and the extensions `GF(pᵏ)` ([`GfElement`](crate::galois::GfElement))
//! — through the [`FactorOverField`] trait, blanket-implemented for every
//! `Poly<T>` whose coefficient type is a [`FiniteField`].
//!
//! The pipeline is the classical Cantor–Zassenhaus three-stage factorization
//! (Cantor & Zassenhaus 1981; von zur Gathen & Gerhard, *Modern Computer
//! Algebra* Ch. 14; Cohen, *A Course in Computational Algebraic Number Theory*
//! §3.4), all in monic `GF(q)[x]`:
//!
//! 1. **Square-free factorization** (finite-field aware, so it handles the
//!    characteristic-`p` case `f = g(xᵖ)` by taking a `p`-th root and recursing).
//! 2. **Distinct-degree factorization**: for `d = 1, 2, …`, peel off the product
//!    `g_d = gcd(x^{qᵈ} − x, f)` of the degree-`d` irreducible factors.
//! 3. **Equal-degree (Cantor–Zassenhaus) splitting**: split each `g_d` into its
//!    `r = deg/d` irreducibles with random splitting polynomials — the
//!    `a^{(qᵈ−1)/2} − 1` trick for odd `q`, and the trace map
//!    `a + a^p + a^{p²} + …` for characteristic 2.
//!
//! Randomness is drawn from a deterministic, reproducible [`SeedRng`] seeded from
//! the polynomial, so the factorization of a given input is reproducible.
//!
//! This is a clean-room implementation from the open literature; it copies no
//! external source.

use alloc::vec;
use alloc::vec::Vec;

use crate::int::Int;
use crate::poly::Poly;
use crate::random::SeedRng;
use crate::ring::{FiniteField, Ring};

// ---------------------------------------------------------------------------
// Small generic helpers over `GF(q)[x]` (`Poly<T>`, `T: FiniteField`).
// ---------------------------------------------------------------------------

/// Whether `f` is constant (degree `0`) or the zero polynomial.
fn is_constant<T: Ring>(f: &Poly<T>) -> bool {
    f.degree().is_none_or(|d| d == 0)
}

/// `base^e` for a field element (square-and-multiply). `e ≥ 0`.
fn field_pow<T: Ring>(base: &T, e: &Int) -> T {
    let mut result = base.one();
    let mut b = base.clone();
    for i in 0..e.bit_len() {
        if e.bit(i) {
            result = result * b.clone();
        }
        let sq = b.clone() * b.clone();
        b = sq;
    }
    result
}

/// `base^e mod modulus` in `GF(q)[x]` (square-and-multiply, reducing each step).
fn poly_powmod<T: FiniteField>(base: &Poly<T>, e: &Int, modulus: &Poly<T>) -> Poly<T> {
    let one = modulus
        .leading()
        .expect("poly_powmod: modulus must be nonzero")
        .one();
    let mut result = Poly::constant(one);
    let mut b = base.rem(modulus);
    for i in 0..e.bit_len() {
        if e.bit(i) {
            result = result.mul(&b).rem(modulus);
        }
        b = b.mul(&b).rem(modulus);
    }
    result
}

/// `log_p(q)` — the extension degree `s` with `q = pˢ` (`1` for a prime field).
fn log_p(q: &Int, p: &Int) -> usize {
    let mut s = 0usize;
    let mut t = q.clone();
    while t > Int::ONE {
        t = t.div_exact(p);
        s += 1;
    }
    s
}

// ---------------------------------------------------------------------------
// Stage 1: square-free factorization (finite-field aware).
// ---------------------------------------------------------------------------

/// The `p`-th root of `c`, a perfect `p`-th power `c = D(x)ᵖ = Σ Dⱼᵖ · x^{pj}`.
///
/// In characteristic `p`, `(Σ aⱼ xʲ)ᵖ = Σ aⱼᵖ x^{pj}`, so `c` has nonzero
/// coefficients only at exponents divisible by `p`, and the root's coefficient
/// `Dⱼ = c_{pj}^{1/p} = c_{pj}^{q/p}` (every field element `y` satisfies
/// `y^q = y`, so `(y^{q/p})ᵖ = y^q = y`).
fn pth_root<T: FiniteField>(c: &Poly<T>, p: usize, p_int: &Int) -> Poly<T> {
    let q_over_p = c
        .leading()
        .expect("pth_root: nonzero")
        .order()
        .div_exact(p_int);
    let coeffs: Vec<T> = c
        .coeffs()
        .iter()
        .step_by(p)
        .map(|coef| field_pow(coef, &q_over_p))
        .collect();
    Poly::new(coeffs)
}

/// Square-free factorization of monic `f` (Musser's finite-field algorithm; von
/// zur Gathen & Gerhard §14.6): appends `(gᵢ, i·scale)` pairs to `out` with
/// `f = ∏ gᵢⁱ`, each `gᵢ` monic, square-free and pairwise coprime. `scale`
/// accumulates the powers of `p` contributed by the `p`-th-root recursion.
fn sff_into<T: FiniteField>(
    f: &Poly<T>,
    scale: usize,
    p_int: &Int,
    out: &mut Vec<(Poly<T>, usize)>,
) {
    // c = gcd(f, f') collects every repeated factor (also c = f when f' = 0);
    // w = f / c is the product of the factors taken once.
    let mut c = f.gcd(&f.derivative());
    let mut w = f.div_rem(&c).0;
    let mut i = 1usize;
    while !is_constant(&w) {
        let y = w.gcd(&c);
        let fac = w.div_rem(&y).0; // factors whose multiplicity is exactly i
        if !is_constant(&fac) {
            out.push((fac.monic(), i * scale));
        }
        c = c.div_rem(&y).0;
        w = y;
        i += 1;
    }
    // Whatever remains in c is a perfect p-th power (every multiplicity was a
    // multiple of p): take its p-th root and recurse with the powers scaled by p.
    if !is_constant(&c) {
        let p = p_int
            .to_u64()
            .expect("characteristic fits in u64 for a p-th-power factor") as usize;
        let root = pth_root(&c, p, p_int);
        sff_into(&root, scale * p, p_int, out);
    }
}

/// Square-free factorization of monic `f` into `(square-free part, multiplicity)`.
fn squarefree<T: FiniteField>(f: &Poly<T>) -> Vec<(Poly<T>, usize)> {
    let p = f.leading().expect("squarefree: nonzero").characteristic();
    let mut out = Vec::new();
    sff_into(f, 1, &p, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Stage 2: distinct-degree factorization.
// ---------------------------------------------------------------------------

/// Distinct-degree factorization of a square-free monic `f`: returns `(d, g_d)`
/// where `g_d` is the product of all the monic irreducible factors of degree `d`.
fn distinct_degree<T: FiniteField>(f: &Poly<T>) -> Vec<(usize, Poly<T>)> {
    let sample = f.leading().expect("distinct_degree: nonzero").clone();
    let q = sample.order();
    let x = Poly::monomial(sample.one(), 1);
    let mut out = Vec::new();
    let mut fstar = f.clone();
    let mut d = 1usize;
    // h = x^{qᵈ} mod fstar, maintained by iterating the q-power Frobenius.
    let mut h = poly_powmod(&x, &q, &fstar); // x^q mod fstar  (d = 1)
    while fstar.degree().is_some_and(|deg| deg >= 2 * d) {
        let g = fstar.gcd(&h.sub(&x)); // gcd(fstar, x^{qᵈ} − x)
        if g.degree().is_some_and(|dg| dg > 0) {
            out.push((d, g.clone()));
            fstar = fstar.div_rem(&g).0;
        }
        d += 1;
        if fstar.degree().is_some_and(|deg| deg >= 2 * d) {
            h = poly_powmod(&h, &q, &fstar); // (x^{q^{d-1}})^q = x^{qᵈ}
        }
    }
    // A nonconstant remainder is a single irreducible of its own degree.
    if let Some(deg) = fstar.degree().filter(|&deg| deg > 0) {
        out.push((deg, fstar));
    }
    out
}

// ---------------------------------------------------------------------------
// Stage 3: equal-degree (Cantor–Zassenhaus) splitting.
// ---------------------------------------------------------------------------

/// A deterministic PRNG seeded from `f`, so a given polynomial always factors the
/// same way.
fn seed_rng_from<T: FiniteField>(f: &Poly<T>) -> SeedRng {
    let q = f.leading().and_then(|c| c.order().to_u64()).unwrap_or(0);
    let deg = f.degree().unwrap_or(0) as u64;
    let seed = 0x9E37_79B9_7F4A_7C15u64
        .wrapping_mul(deg.wrapping_add(1))
        .wrapping_add(q)
        .wrapping_add(f.coeffs().len() as u64);
    SeedRng::new(seed | 1)
}

/// A random polynomial of degree `< n` over the field of `sample`.
fn random_poly<T: FiniteField>(n: usize, sample: &T, rng: &mut SeedRng) -> Poly<T> {
    let q = sample.order();
    let coeffs: Vec<T> = (0..n)
        .map(|_| {
            let idx = Int::random_below(&q, rng).unwrap_or(Int::ZERO);
            sample.from_index(&idx)
        })
        .collect();
    Poly::new(coeffs)
}

/// Equal-degree factorization: `f` is a product of `r = deg(f)/d` monic
/// irreducibles each of degree `d`; split it into them.
fn equal_degree<T: FiniteField>(f: &Poly<T>, d: usize) -> Vec<Poly<T>> {
    let total = f.degree().expect("equal_degree: nonzero");
    let r = total / d;
    if r <= 1 {
        return vec![f.monic()];
    }
    let sample = f.leading().unwrap().clone();
    let q = sample.order();
    let p = sample.characteristic();
    let is_char2 = p == Int::from_u64(2);
    // Odd q: the exponent (qᵈ − 1)/2. Char 2: the number of trace terms s·d.
    let (exp_odd, trace_len) = if is_char2 {
        (Int::ZERO, log_p(&q, &p) * d)
    } else {
        let half = q.pow(d as u32).sub(&Int::ONE).div_exact(&Int::from_u64(2));
        (half, 0)
    };

    let mut rng = seed_rng_from(f);
    let mut factors = vec![f.monic()];
    while factors.len() < r {
        let a = random_poly(total, &sample, &mut rng);
        // Splitting polynomial b: gcd(h, b) separates roughly half the factors.
        let b = if is_char2 {
            // Trace map  b = a + a^p + a^{p²} + … + a^{p^{s·d − 1}}  (mod f).
            let mut term = a.rem(f);
            let mut acc = term.clone();
            for _ in 1..trace_len {
                term = poly_powmod(&term, &p, f);
                acc = acc.add(&term);
            }
            acc
        } else {
            // b = a^{(qᵈ − 1)/2} − 1  (mod f).
            poly_powmod(&a, &exp_odd, f).sub(&Poly::constant(sample.one()))
        };

        let mut next = Vec::with_capacity(factors.len());
        for h in core::mem::take(&mut factors) {
            let dh = h.degree().unwrap();
            if dh == d {
                next.push(h);
                continue;
            }
            let g = h.gcd(&b);
            let dg = g.degree().unwrap_or(0);
            if dg > 0 && dg < dh {
                let other = h.div_rem(&g).0;
                next.push(g.monic());
                next.push(other.monic());
            } else {
                next.push(h);
            }
        }
        factors = next;
    }
    factors
}

// ---------------------------------------------------------------------------
// Driver + public trait.
// ---------------------------------------------------------------------------

/// Runs the full Cantor–Zassenhaus pipeline on `f`, returning irreducible monic
/// factors with multiplicities.
fn factor_ff<T: FiniteField>(f: &Poly<T>) -> Vec<(Poly<T>, usize)> {
    if is_constant(f) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (sqfree, mult) in squarefree(&f.monic()) {
        for (d, g_d) in distinct_degree(&sqfree) {
            for irr in equal_degree(&g_d, d) {
                out.push((irr, mult));
            }
        }
    }
    out
}

/// Factorization of a univariate polynomial over a finite field `GF(q)`, by
/// Cantor–Zassenhaus.
///
/// Implemented for every `Poly<T>` whose coefficients form a
/// [`FiniteField`](crate::ring::FiniteField) — the prime fields `GF(p)`
/// ([`ModInt`](crate::mod_int::ModInt) with a prime modulus) and the extensions
/// `GF(pᵏ)` ([`GfElement`](crate::galois::GfElement)). Bring the trait into scope
/// to use it:
///
/// ```
/// # #[cfg(all(feature = "poly", feature = "int"))] {
/// use puremp::{FactorOverField, Int, ModInt, Poly};
///
/// // x² − 1 = (x − 1)(x + 1) over GF(5).
/// let m = |v: i64| ModInt::new(Int::from_i64(v), Int::from_i64(5));
/// let f = Poly::new(vec![m(-1), m(0), m(1)]);
/// let factors = f.factor();
/// assert_eq!(factors.len(), 2);
/// # }
/// ```
///
/// The infinite fields (`Rational`, `Float`) intentionally do not implement
/// [`FiniteField`], so this trait cannot be applied to them; `Poly<Rational>`
/// keeps its own inherent [`factor`](crate::poly::Poly::factor) over `ℚ`.
pub trait FactorOverField<T: FiniteField> {
    /// Factors into monic irreducible factors with multiplicities, as
    /// `(factor, multiplicity)` pairs. The product of `factorⁱ` equals `self`
    /// made monic. Constants (and the zero polynomial) yield an empty list.
    fn factor(&self) -> Vec<(Poly<T>, usize)>;

    /// Whether `self` is irreducible over the field (degree `≥ 1` with a single
    /// factor of multiplicity one).
    fn is_irreducible(&self) -> bool;

    /// The square-free factorization: `(square-free part, multiplicity)` pairs
    /// with `self` (made monic) equal to the product of each part raised to its
    /// multiplicity. Each part is itself square-free but need not be irreducible.
    fn squarefree_factorization(&self) -> Vec<(Poly<T>, usize)>;
}

impl<T: FiniteField> FactorOverField<T> for Poly<T> {
    fn factor(&self) -> Vec<(Poly<T>, usize)> {
        factor_ff(self)
    }

    fn is_irreducible(&self) -> bool {
        if is_constant(self) {
            return false;
        }
        let factors = factor_ff(self);
        factors.len() == 1 && factors[0].1 == 1
    }

    fn squarefree_factorization(&self) -> Vec<(Poly<T>, usize)> {
        if is_constant(self) {
            return Vec::new();
        }
        squarefree(&self.monic())
    }
}
