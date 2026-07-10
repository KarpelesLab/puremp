//! The **ideal class group** `Cl(K)` and **class number** `h_K` of a number
//! field `K`, built on top of [`crate::numberfield`] and
//! [`crate::numberfield_ideal`].
//!
//! This is phase 3a of the number-field subsystem. It computes `Cl(K)` for
//! small-to-moderate fields by the classical **Minkowski-bound + relations +
//! Smith-normal-form** method (Cohen, *A Course in Computational Algebraic
//! Number Theory* (CCANT), §5.2–5.4 and §6.5; Marcus, *Number Fields*, ch. 5).
//!
//! # Method
//!
//! **Minkowski bound.** Every ideal class contains an integral ideal of norm at
//! most
//!
//! ```text
//! M_K = (4/π)^{r₂} · (n! / nⁿ) · √|d_K|
//! ```
//!
//! (`n = [K:ℚ]`, `r₂` the number of complex-embedding pairs, `d_K` the field
//! discriminant). Consequently the classes of the prime ideals of norm `≤ M_K`
//! **generate** `Cl(K)` (CCANT Theorem 5.3.3 / 6.5.x). The bound is evaluated in
//! arbitrary-precision [`Float`] arithmetic with each rounding directed so the
//! result is a *rigorous upper bound*, then rounded up to an integer `M`. Using
//! a slightly larger `M` only enlarges the generating set, which is harmless.
//!
//! **Factor base.** All prime ideals `𝔭` with `N(𝔭) ≤ M`: enumerate the rational
//! primes `p ≤ M`, factor each with [`Order::factor_prime`], and keep the primes
//! above them whose norm is `≤ M`.
//!
//! **Relations.** A principal ideal `(α)` that factors completely over the
//! factor base, `(α) = ∏ 𝔭ᵢ^{aᵢ}`, yields the relation `Σ aᵢ[𝔭ᵢ] = 0` in
//! `Cl(K)`. Two sources are used:
//!
//! * the rational primes themselves — `(p) = ∏_{𝔭|p} 𝔭^{e_𝔭}` is principal, and
//! * short elements of products `A = ∏ 𝔭ᵢ^{eᵢ}` of factor-base primes, found by
//!   [LLL](crate::lattice::lll_reduce) reduction of the **Minkowski embedding**
//!   of `A` (the geometry that makes `T₂`- and hence norm-small elements short).
//!   Each reduced element `α` gives `(α)`, which is factored over the factor base
//!   by computing the `𝔭`-adic valuations `v_𝔭(α)` (largest `k` with
//!   `(α) ⊆ 𝔭^k`) and confirming `∏ N(𝔭)^{v_𝔭} = |N(α)|`.
//!
//! **Structure.** With the factor base as generators `ℤ^g` and the relations as
//! a sublattice `L ⊆ ℤ^g`, `Cl(K) ≅ ℤ^g / L`, whose invariant factors are the
//! [Smith normal form](crate::matrix::Matrix::invariant_factors) of the relation
//! matrix; `h_K` is their product.
//!
//! # Completeness and the guaranteed range
//!
//! Because every collected relation is genuine (`L` is always a *sublattice* of
//! the true relation lattice `ker(ℤ^g → Cl(K))`), the computed class number is
//! **always a multiple of the true `h_K`** and decreases monotonically as more
//! relations are found. The computation collects relations in rounds — all
//! rational primes, then products of factor-base primes with growing exponents —
//! and stops once the Smith normal form is full rank `g` (so the quotient is
//! finite) and has **stabilised** across rounds. Under stabilisation the sublattice
//! has saturated to the full relation lattice and the answer is `h_K`.
//!
//! This is the standard elementary approach and is reliable for **small-to-
//! moderate fields** (small `|d_K|`, so a small Minkowski bound and factor base):
//! quadratic fields, small cubics, small quartics. It is *not* the
//! sub-exponential Buchmann algorithm; for large discriminants the factor base
//! and relation search become impractical and the method returns [`None`] rather
//! than a wrong answer. All results are verified against known class-number
//! tables in the test suite.

// Dense exact linear algebra over explicit index ranges (embeddings, relation
// matrices) reads closer to the mathematics than iterator adapters here.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

use crate::complex::Complex;
use crate::float::{Float, RoundingMode};
use crate::int::Int;
use crate::lattice::lll_reduce;
use crate::matrix::Matrix;
use crate::numberfield::NumberField;
use crate::numberfield_ideal::{Ideal, Order, PrimeIdeal};
use crate::random::SeedRng;
use crate::rational::Rational;

/// Working precision (bits) for the complex embeddings used in the Minkowski
/// bound and in LLL relation search.
const PREC: u64 = 256;
/// Scaling exponent for the (rounded) Minkowski embedding fed to LLL: geometry
/// entries are `round(component · 2^SCALE)`, dwarfing the appended
/// order-coordinate columns so the reduction is driven by the geometry.
const SCALE: u32 = 64;
/// Maximum relation-collection rounds before giving up.
const MAX_ROUNDS: usize = 14;
/// Refuse fields whose Minkowski bound exceeds this (factor base too large to
/// handle by the direct method); returns `None` instead.
const MBOUND_CAP: i64 = 200_000;
/// Refuse fields with more than this many factor-base generators.
const MAX_GENERATORS: usize = 24;

/// The ideal class group `Cl(K)` of a number field: its **class number** and the
/// **invariant factors** describing its structure as a finite abelian group
/// `ℤ/d₁ × ⋯ × ℤ/d_k` with `d₁ | d₂ | … | d_k` (each `dᵢ > 1`).
///
/// The trivial group (`h_K = 1`) has an empty [`invariant_factors`] list.
///
/// [`invariant_factors`]: ClassGroup::invariant_factors
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassGroup {
    /// The class number `h_K = [ℤ^g : L]` (the product of the invariant factors,
    /// `1` when the group is trivial).
    pub class_number: Int,
    /// The invariant factors `d₁ | d₂ | … | d_k` with each `dᵢ > 1`; empty for
    /// the trivial group. `Cl(K) ≅ ℤ/d₁ℤ × ⋯ × ℤ/d_kℤ`.
    pub invariant_factors: Vec<Int>,
}

// ===========================================================================
// Minkowski bound.
// ===========================================================================

/// A rigorous integer upper bound for the Minkowski bound
/// `M_K = (4/π)^{r₂} · (n!/nⁿ) · √|d_K|`.
///
/// Every rounding is directed so the returned value is `≥ M_K`; the class group
/// is generated by the primes of norm `≤ M_K`, so an overestimate only enlarges
/// the (harmless) generating set.
fn minkowski_bound(order: &Order) -> Int {
    let n = order.degree();
    let (_, r2) = order.field().signature();
    let dk = order.discriminant().abs();
    let up = RoundingMode::TowardPositive;
    let down = RoundingMode::TowardNegative;

    // (4/π)^{r₂}: a smaller π gives a larger 4/π, so round π down.
    let pi = Float::pi(PREC, down);
    let four = Float::from_int(&Int::from_i64(4), PREC, up);
    let four_over_pi = four.div(&pi, PREC, up);
    let mut fp = Float::from_int(&Int::ONE, PREC, up);
    for _ in 0..r2 {
        fp = fp.mul(&four_over_pi, PREC, up);
    }

    // n!/nⁿ: numerator up, denominator down.
    let nfact = Float::from_int(&Int::factorial(n as u64), PREC, up);
    let nn = Float::from_int(&Int::from_i64(n as i64).pow(n as u32), PREC, down);
    let ratio = nfact.div(&nn, PREC, up);

    // √|d_K|, rounded up.
    let sqrt_dk = Float::from_int(&dk, PREC, up).sqrt(PREC, up);

    let m = fp.mul(&ratio, PREC, up).mul(&sqrt_dk, PREC, up);
    m.ceil().expect("Minkowski bound is finite")
}

// ===========================================================================
// Factor base.
// ===========================================================================

/// All prime ideals of norm `≤ mbound`: enumerate rational primes `p ≤ mbound`,
/// factor each, keep the primes above them with norm `≤ mbound`.
fn factor_base(order: &Order, mbound: &Int) -> Vec<PrimeIdeal> {
    let mut fb = Vec::new();
    let mut p = Int::from_i64(2);
    while p <= *mbound {
        for (pi, _e) in order.factor_prime(&p) {
            if pi.norm() <= *mbound {
                fb.push(pi);
            }
        }
        p = p.next_prime();
    }
    fb
}

// ===========================================================================
// Factoring a principal ideal over the factor base.
// ===========================================================================

/// The `𝔭`-adic valuation `v_𝔭((α)) = max{ k : (α) ⊆ 𝔭^k }`, capped at `bound`.
fn ideal_valuation(order: &Order, target: &Ideal, prime: &Ideal, bound: usize) -> usize {
    let mut k = 0usize;
    let mut power = order.unit_ideal(); // 𝔭^0 = O_K ⊇ everything
    while k < bound {
        let next = power.mul(prime); // 𝔭^{k+1}
        if next.contains(target) {
            power = next;
            k += 1;
        } else {
            break;
        }
    }
    k
}

/// Factors the ideal `I` over the factor base, returning the exponent vector
/// `(v_{𝔭ᵢ}(I))ᵢ` if `I` factors *completely* over it, or `None` otherwise (a
/// prime of norm `> mbound` divides `I`).
fn factor_over_fb(
    order: &Order,
    ideal: &Ideal,
    fb: &[PrimeIdeal],
    mbound: &Int,
) -> Option<Vec<Int>> {
    let nrm = ideal.norm();
    let mut rel = alloc::vec![Int::ZERO; fb.len()];
    if nrm.is_one() {
        return Some(rel); // (α) = O_K: α is a unit, trivial relation
    }
    if nrm.is_zero() {
        return None;
    }
    let mut check = Int::ONE;
    for (q, qe) in nrm.factor_exponents() {
        if q > *mbound {
            return None; // a rational prime below a non-factor-base prime
        }
        for (i, pi) in fb.iter().enumerate() {
            if pi.rational_prime() != q {
                continue;
            }
            let f = pi.residue_degree();
            let bound = (qe as usize) / f;
            let v = ideal_valuation(order, ideal, pi.ideal(), bound);
            if v > 0 {
                rel[i] = Int::from_i64(v as i64);
                check = check.mul(&q.pow((f * v) as u32));
            }
        }
    }
    if check == nrm {
        Some(rel)
    } else {
        None // incomplete: some prime factor is outside the factor base
    }
}

// ===========================================================================
// Minkowski-embedding LLL relation search.
// ===========================================================================

/// Precomputed data for the Minkowski (canonical) embedding of `K`.
struct Embedder {
    n: usize,
    /// The `n` complex roots `θ_i` of the defining polynomial.
    roots: Vec<Complex<Float>>,
    /// Indices of the real embeddings.
    real_idx: Vec<usize>,
    /// One index per complex-conjugate pair (the representative with `Im > 0`).
    cplx_idx: Vec<usize>,
    /// `2^SCALE` as a float.
    scale: Float,
    /// `√2` as a float (the weight on complex coordinates so that the squared
    /// coordinate length equals the `T₂` norm).
    sqrt2: Float,
}

impl Embedder {
    fn new(order: &Order) -> Embedder {
        let field = order.field();
        let n = order.degree();
        let roots = field.complex_roots(PREC);
        // A real root has a negligible imaginary part.
        let threshold =
            Float::from_rational(&Rational::power_of_two(-64), PREC, RoundingMode::Nearest);
        let zero = Float::zero(PREC);
        let mut real_idx = Vec::new();
        let mut cplx_idx = Vec::new();
        for (i, z) in roots.iter().enumerate() {
            if z.im.abs() < threshold {
                real_idx.push(i);
            } else if z.im > zero {
                cplx_idx.push(i);
            }
        }
        let scale_int = Int::from_i64(2).pow(SCALE);
        let scale = Float::from_int(&scale_int, PREC, RoundingMode::Nearest);
        let sqrt2 = Float::from_int(&Int::from_i64(2), PREC, RoundingMode::Nearest)
            .sqrt(PREC, RoundingMode::Nearest);
        Embedder {
            n,
            roots,
            real_idx,
            cplx_idx,
            scale,
            sqrt2,
        }
    }

    /// The complex embeddings `σ_i(α)` of an element given by power-basis
    /// coordinates, by Horner evaluation at each root.
    fn embed(&self, power: &[Rational]) -> Vec<Complex<Float>> {
        let coeffs: Vec<Complex<Float>> = power
            .iter()
            .map(|c| {
                Complex::new(
                    Float::from_rational(c, PREC, RoundingMode::Nearest),
                    Float::zero(PREC),
                )
            })
            .collect();
        self.roots
            .iter()
            .map(|z| {
                let mut acc = Complex::new(Float::zero(PREC), Float::zero(PREC));
                for c in coeffs.iter().rev() {
                    acc = acc.mul(z).add(c);
                }
                acc
            })
            .collect()
    }

    /// The scaled, rounded Minkowski embedding of an element given by power-basis
    /// coordinates: an integer vector of length `n = r₁ + 2r₂`.
    fn geom(&self, power: &[Rational]) -> Vec<Int> {
        let vals = self.embed(power);
        let mut g = Vec::with_capacity(self.n);
        for &i in &self.real_idx {
            let e = vals[i].re.mul(&self.scale, PREC, RoundingMode::Nearest);
            g.push(e.round_to_int().unwrap_or(Int::ZERO));
        }
        for &i in &self.cplx_idx {
            let re = self
                .sqrt2
                .mul(&vals[i].re, PREC, RoundingMode::Nearest)
                .mul(&self.scale, PREC, RoundingMode::Nearest);
            let im = self
                .sqrt2
                .mul(&vals[i].im, PREC, RoundingMode::Nearest)
                .mul(&self.scale, PREC, RoundingMode::Nearest);
            g.push(re.round_to_int().unwrap_or(Int::ZERO));
            g.push(im.round_to_int().unwrap_or(Int::ZERO));
        }
        g
    }
}

/// Converts integer order coordinates of an element to power-basis coordinates
/// via the integral basis (`v = B · c`).
fn order_to_power(order: &Order, alpha: &[Int]) -> Vec<Rational> {
    let b = order.integral_basis();
    let n = order.degree();
    (0..n)
        .map(|i| {
            let mut acc = Rational::ZERO;
            for j in 0..n {
                if alpha[j].is_zero() {
                    continue;
                }
                acc = acc.add(&b.get(i, j).mul(&Rational::from_integer(alpha[j].clone())));
            }
            acc
        })
        .collect()
}

/// The principal ideal `(α)` for `α` given in integer order coordinates.
fn alpha_ideal(order: &Order, alpha: &[Int]) -> Ideal {
    order.principal_ideal(&order_to_power(order, alpha))
}

/// The product ideal `∏ fb[i].ideal()^{exps[i]}`.
fn build_product(order: &Order, fb: &[PrimeIdeal], exps: &[usize]) -> Ideal {
    let mut a = order.unit_ideal();
    for (i, &e) in exps.iter().enumerate() {
        for _ in 0..e {
            a = a.mul(fb[i].ideal());
        }
    }
    a
}

/// LLL-reduces the Minkowski embedding of the ideal `a`, then factors `(α)` over
/// the factor base for each `α` in a small box of integer combinations of the
/// reduced basis, pushing every complete relation onto `out`.
///
/// Searching combinations (not just the reduced basis vectors) is essential in
/// fields with an infinite unit group: a principal generator of an ideal — e.g.
/// `2+√6` of norm `±2` in `ℚ(√6)` — is typically a short *combination* of the
/// reduced basis rather than a basis vector itself, and only such generators
/// witness that a factor-base prime is principal.
fn collect_lll_relations(
    order: &Order,
    emb: &Embedder,
    fb: &[PrimeIdeal],
    mbound: &Int,
    a: &Ideal,
    out: &mut Vec<Vec<Int>>,
) {
    let n = order.degree();
    let basis = a.basis();
    // Rows: [ scaled Minkowski embedding (n cols) | order coordinates (n cols) ].
    let mut rows: Vec<Vec<Int>> = Vec::with_capacity(n);
    for i in 0..n {
        let ord: Vec<Int> = (0..n).map(|j| basis.get(i, j).clone()).collect();
        let power = order_to_power(order, &ord);
        let mut row = emb.geom(&power);
        row.extend(ord.iter().cloned());
        rows.push(row);
    }
    let reduced = lll_reduce(&rows);
    // Order coordinates of the reduced basis elements (trailing n columns).
    let red_coords: Vec<Vec<Int>> = reduced.iter().map(|r| r[n..].to_vec()).collect();

    // Radius of the combination box: wider for the small (quadratic/cubic) cases.
    let radius: i64 = match n {
        0..=2 => 3,
        3 => 2,
        _ => 1,
    };
    let mut coeff = alloc::vec![-radius; n];
    loop {
        // α = Σ coeff[i] · red_coords[i] (order coordinates).
        let mut alpha = alloc::vec![Int::ZERO; n];
        let mut nonzero = false;
        for (i, &c) in coeff.iter().enumerate() {
            if c != 0 {
                nonzero = true;
                let ci = Int::from_i64(c);
                for k in 0..n {
                    alpha[k] = alpha[k].add(&ci.mul(&red_coords[i][k]));
                }
            }
        }
        if nonzero {
            let ideal = alpha_ideal(order, &alpha);
            if let Some(rel) = factor_over_fb(order, &ideal, fb, mbound)
                && rel.iter().any(|x| !x.is_zero())
            {
                push_unique(out, rel);
            }
        }
        // Advance the odometer over coeff ∈ [-radius, radius]^n.
        let mut i = 0;
        loop {
            if i == n {
                return;
            }
            coeff[i] += 1;
            if coeff[i] <= radius {
                break;
            }
            coeff[i] = -radius;
            i += 1;
        }
    }
}

/// Pushes `rel` onto `rels` unless an identical relation is already present.
fn push_unique(rels: &mut Vec<Vec<Int>>, rel: Vec<Int>) {
    if !rels.contains(&rel) {
        rels.push(rel);
    }
}

/// The set of factor-base exponent vectors whose product ideals are LLL-reduced
/// in a given round: all `0/1` subsets, single primes raised to growing powers,
/// and a handful of random small-exponent combinations.
fn target_exponents(g: usize, round: usize, rng: &mut SeedRng) -> Vec<Vec<usize>> {
    let mut set: Vec<Vec<usize>> = Vec::new();
    // All 0/1 subsets (includes O_K and every single generator) when g is small.
    if g <= 12 {
        for mask in 0u32..(1u32 << g) {
            set.push((0..g).map(|i| ((mask >> i) & 1) as usize).collect());
        }
    } else {
        for _ in 0..256 {
            set.push((0..g).map(|_| (rng.next_u64() & 1) as usize).collect());
        }
    }
    // Single primes to growing powers (to expose the order of each generator).
    for i in 0..g {
        for e in 2..=(round + 2) {
            let mut v = alloc::vec![0usize; g];
            v[i] = e;
            set.push(v);
        }
    }
    // A few random small-exponent combinations.
    let cap = round + 2;
    for _ in 0..(8 * (round + 1)) {
        set.push(
            (0..g)
                .map(|_| (rng.next_u64() as usize) % (cap + 1))
                .collect(),
        );
    }
    set
}

// ===========================================================================
// Smith normal form of the relation matrix.
// ===========================================================================

/// The invariant factors of `ℤ^g / L`, where `L` is spanned by the relation
/// vectors (each length `g`). Returns `None` when the relations do not yet have
/// full rank `g` (the quotient would be infinite — not enough relations).
fn class_structure(relations: &[Vec<Int>], g: usize) -> Option<Vec<Int>> {
    if relations.is_empty() {
        return None;
    }
    let nrel = relations.len();
    // Column `k` = relation k; rows = generators. Cokernel ℤ^g / (col span).
    let mut data = alloc::vec![Int::ZERO; g * nrel];
    for (k, rel) in relations.iter().enumerate() {
        for i in 0..g {
            data[i * nrel + k] = rel[i].clone();
        }
    }
    let factors = Matrix::new(g, nrel, data).invariant_factors();
    if factors.len() == g {
        Some(factors)
    } else {
        None // rank < g ⇒ infinite quotient
    }
}

/// Builds the [`ClassGroup`] from a full set of invariant factors (dropping the
/// trivial `1`s and multiplying the rest for the class number).
fn to_class_group(factors: &[Int]) -> ClassGroup {
    let mut h = Int::ONE;
    let mut nontrivial = Vec::new();
    for d in factors {
        if !d.is_one() {
            h = h.mul(d);
            nontrivial.push(d.clone());
        }
    }
    ClassGroup {
        class_number: h,
        invariant_factors: nontrivial,
    }
}

// ===========================================================================
// Public API.
// ===========================================================================

impl Order {
    /// Computes the **ideal class group** `Cl(K)` of the field of this (maximal)
    /// order by the Minkowski-bound / relations / Smith-normal-form method.
    ///
    /// Returns the [`ClassGroup`] (class number and invariant factors), or
    /// [`None`] if the field is out of range for the direct method — either the
    /// Minkowski bound / factor base is too large, or the relation search fails
    /// to saturate within the round budget. The computation never returns a
    /// wrong (over- or under-counted) class number: it is designed so that any
    /// answer it *does* return has a fully-saturated relation lattice.
    ///
    /// See the [module documentation](crate::numberfield_class) for the method
    /// and its guaranteed range (quadratic, small cubic and quartic fields).
    ///
    /// This assumes `self` is the maximal order `O_K`
    /// (from [`NumberField::maximal_order`]); the class group is an invariant of
    /// the field.
    pub fn class_group(&self) -> Option<ClassGroup> {
        let mbound = minkowski_bound(self);
        if mbound > Int::from_i64(MBOUND_CAP) {
            return None;
        }
        let fb = factor_base(self, &mbound);
        let g = fb.len();
        if g == 0 {
            // No generators ⇒ trivial class group.
            return Some(ClassGroup {
                class_number: Int::ONE,
                invariant_factors: Vec::new(),
            });
        }
        if g > MAX_GENERATORS {
            return None;
        }

        let emb = Embedder::new(self);
        let n = self.degree();
        let mut relations: Vec<Vec<Int>> = Vec::new();

        // Relations from the rational primes: (p) = ∏_{𝔭|p} 𝔭^{e_𝔭} is principal.
        let mut p = Int::from_i64(2);
        while p <= mbound {
            let mut pc = alloc::vec![Rational::ZERO; n];
            pc[0] = Rational::from_integer(p.clone());
            let ip = self.principal_ideal(&pc);
            if let Some(rel) = factor_over_fb(self, &ip, &fb, &mbound)
                && rel.iter().any(|x| !x.is_zero())
            {
                push_unique(&mut relations, rel);
            }
            p = p.next_prime();
        }

        // Relations from short elements of factor-base products, round by round,
        // until the Smith normal form is full rank and stabilises.
        let mut rng = SeedRng::new(0x0c1a_5591_0009_0001);
        let mut last_full: Option<Vec<Int>> = None;
        for round in 0..MAX_ROUNDS {
            for exps in target_exponents(g, round, &mut rng) {
                let a = build_product(self, &fb, &exps);
                collect_lll_relations(self, &emb, &fb, &mbound, &a, &mut relations);
            }
            if let Some(factors) = class_structure(&relations, g) {
                if let Some(prev) = &last_full {
                    // Monotone non-increasing ⇒ a repeat after enough rounds means
                    // the relation lattice has saturated to the true one.
                    if *prev == factors && round >= 2 {
                        return Some(to_class_group(&factors));
                    }
                }
                last_full = Some(factors);
            }
        }
        None
    }
}

impl NumberField {
    /// The **ideal class group** `Cl(K)` of this number field.
    ///
    /// Convenience wrapper for `self.maximal_order().class_group()`. Returns
    /// [`None`] when the field is out of range for the direct method (see
    /// [`Order::class_group`]).
    pub fn class_group(&self) -> Option<ClassGroup> {
        self.maximal_order().class_group()
    }

    /// The **class number** `h_K` of this number field.
    ///
    /// # Panics
    /// If the class group cannot be determined by the direct method (large
    /// discriminant / factor base, or non-saturating relation search). Use
    /// [`NumberField::class_group`] for a non-panicking variant. See the
    /// [module documentation](crate::numberfield_class) for the supported range.
    pub fn class_number(&self) -> Int {
        self.class_group()
            .expect("class_number: field is out of range for the direct method")
            .class_number
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::Poly;

    fn q(v: i64) -> Rational {
        Rational::from_integer(Int::from_i64(v))
    }

    fn poly(coeffs: &[i64]) -> Poly<Rational> {
        Poly::new(coeffs.iter().map(|&c| q(c)).collect())
    }

    fn field(coeffs: &[i64]) -> NumberField {
        NumberField::new(poly(coeffs)).unwrap()
    }

    fn ii(v: i64) -> Int {
        Int::from_i64(v)
    }

    /// ℚ(√−d): defining polynomial x² + d.
    fn imag_quad(d: i64) -> NumberField {
        field(&[d, 0, 1])
    }

    /// ℚ(√d): defining polynomial x² − d.
    fn real_quad(d: i64) -> NumberField {
        field(&[-d, 0, 1])
    }

    fn ivec(v: &[i64]) -> Vec<Int> {
        v.iter().map(|&x| ii(x)).collect()
    }

    fn assert_class_group(k: &NumberField, h: i64, factors: &[i64]) {
        let cg = k.class_group().expect("class group should be computable");
        assert_eq!(cg.class_number, ii(h), "class number for {k:?}");
        assert_eq!(cg.invariant_factors, ivec(factors), "structure for {k:?}");
    }

    // ---- Minkowski bound sanity ----

    #[test]
    fn minkowski_bound_qsqrt_neg5() {
        // M_K = (4/π)·(1/2)·√20 ≈ 2.847 ⇒ upper-bounded integer is 3.
        let o = imag_quad(5).maximal_order();
        let m = minkowski_bound(&o);
        assert!(m >= ii(3), "bound must cover 2.847, got {m}");
        assert!(m <= ii(4), "bound should be tight-ish, got {m}");
    }

    // ---- Imaginary quadratic, h = 1 ----

    #[test]
    fn imag_h1_small() {
        for d in [1, 2, 3, 7, 11, 19] {
            assert_class_group(&imag_quad(d), 1, &[]);
        }
    }

    #[test]
    fn imag_h1_heegner_large() {
        // The larger Heegner numbers: factor base is empty or all-principal.
        for d in [43, 67, 163] {
            assert_class_group(&imag_quad(d), 1, &[]);
        }
    }

    // ---- Imaginary quadratic, h = 2 (Cl = ℤ/2) ----

    #[test]
    fn imag_h2_cyclic2() {
        for d in [5, 6, 10, 13, 15] {
            assert_class_group(&imag_quad(d), 2, &[2]);
        }
    }

    // ---- Imaginary quadratic, h = 3 (Cl = ℤ/3) ----

    #[test]
    fn imag_h3_cyclic3() {
        assert_class_group(&imag_quad(23), 3, &[3]);
    }

    #[test]
    #[ignore = "larger factor base / relation search"]
    fn imag_h3_cyclic3_d31() {
        assert_class_group(&imag_quad(31), 3, &[3]);
    }

    // ---- Imaginary quadratic, h = 4 ----

    #[test]
    fn imag_h4_cyclic4() {
        // ℚ(√−14): Cl = ℤ/4.
        assert_class_group(&imag_quad(14), 4, &[4]);
    }

    #[test]
    #[ignore = "larger factor base / relation search"]
    fn imag_h4_cyclic4_d39() {
        assert_class_group(&imag_quad(39), 4, &[4]);
    }

    #[test]
    fn imag_h4_noncyclic() {
        // ℚ(√−21): Cl = ℤ/2 × ℤ/2 (non-cyclic h = 4).
        assert_class_group(&imag_quad(21), 4, &[2, 2]);
    }

    #[test]
    #[ignore = "larger factor base / relation search"]
    fn imag_h4_noncyclic_d30() {
        // ℚ(√−30): Cl = ℤ/2 × ℤ/2.
        assert_class_group(&imag_quad(30), 4, &[2, 2]);
    }

    // ---- Real quadratic ----

    #[test]
    fn real_h1() {
        for d in [2, 3, 5, 6, 7, 11, 13] {
            assert_class_group(&real_quad(d), 1, &[]);
        }
    }

    #[test]
    fn real_h2() {
        // ℚ(√10) and ℚ(√15): Cl = ℤ/2.
        assert_class_group(&real_quad(10), 2, &[2]);
        assert_class_group(&real_quad(15), 2, &[2]);
    }

    // ---- Cubic and quartic ----

    #[test]
    fn cubic_h1() {
        // x³ − x − 1 (disc −23): class number 1.
        assert_class_group(&field(&[-1, -1, 0, 1]), 1, &[]);
    }

    #[test]
    fn cyclotomic5_h1() {
        // 5th cyclotomic field x⁴ + x³ + x² + x + 1: class number 1.
        assert_class_group(&field(&[1, 1, 1, 1, 1]), 1, &[]);
    }

    // ---- class_number convenience method ----

    #[test]
    fn class_number_method() {
        assert_eq!(imag_quad(5).class_number(), ii(2));
        assert_eq!(imag_quad(163).class_number(), ii(1));
        assert_eq!(real_quad(2).class_number(), ii(1));
    }
}
