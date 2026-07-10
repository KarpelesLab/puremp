//! The unit group `O_K^*`, fundamental units, roots of unity, and the regulator
//! of a [`NumberField`] `K = ℚ(θ) = ℚ[x]/(T)`.
//!
//! This is phase 3b of the number-field subsystem, built on top of
//! [`crate::numberfield`] and [`crate::numberfield_ideal`]. It assumes the
//! defining polynomial `T` is **monic with integer coefficients** (so that `θ`
//! is an algebraic integer and `ℤ[θ] ⊆ O_K`).
//!
//! # Dirichlet's unit theorem
//!
//! For a number field of signature `(r₁, r₂)` the unit group is
//! `O_K^* ≅ μ_K × ℤ^{r}` with **unit rank** `r = r₁ + r₂ − 1`, where `μ_K` is
//! the finite cyclic group of roots of unity in `K`. The **regulator** `R_K` is
//! the covolume of the unit lattice under the logarithmic embedding
//! `ε ↦ (dᵢ·log|σᵢ(ε)|)_i` (`dᵢ = 1` at a real place, `2` at a complex place),
//! restricted to any `r` of the `r₁ + r₂` infinite places:
//! `R = |det( dᵢ·log|σᵢ(εⱼ)| )_{i,j=1..r}|`.
//!
//! # What is provided
//!
//! - [`Order::unit_group`] returns a [`UnitGroup`] carrying the rank, the order
//!   `w_K` and a generator of `μ_K`, the known fundamental units, and the
//!   regulator.
//! - [`NumberField::fundamental_unit`] — the fundamental unit `ε > 1` of a
//!   **real quadratic** field, computed exactly.
//! - [`NumberField::regulator`] — the regulator at a requested `Float`
//!   precision.
//!
//! # Algorithms (clean-room, from the open literature)
//!
//! **Roots of unity.** `μ_K` is cyclic and always contains `−1`, so `2 | w_K`.
//! A primitive `m`-th root of unity lies in `K` iff the `m`-th cyclotomic
//! polynomial `Φ_m` has a root in `K`; this can only happen when `φ(m) | n`.
//! Existence is decided with **Trager's norm test** (Cohen, *CCANT* §3.6.2):
//! form `N(x) = Res_y(T(y), Φ_m(x − s·y)) = ∏_i Φ_m(x − s·θ_i)` for a small
//! shift `s` making `N` squarefree, and check whether `N` has an irreducible
//! rational factor of degree exactly `n` (⟺ `Φ_m` has a linear factor over
//! `K`). The resultant `N` is an integer polynomial and is obtained exactly by
//! evaluating the product at the numeric roots `θ_i` and rounding. The generator
//! is recovered as the linear factor of `gcd_{K[x]}(Φ_m(x), N_i(x + s·θ))`.
//!
//! **Real-quadratic fundamental unit.** The fundamental unit of the real
//! quadratic field of squarefree radicand `d` is `ε = (T + U√d)/2` where
//! `(T, U)` is the smallest positive solution of the Pell-like equation
//! `T² − d·U² = ±4` (Cohen, *CCANT* §5.7). It is obtained from the continued
//! fraction of `√d`: this yields the fundamental solution `η₀ = h + k√d` of the
//! standard Pell equation `x² − d·y² = ±1`; when `d ≡ 1 (mod 4)` a smaller
//! "half-integer" unit `ε = (a + b√d)/2` with `a, b` odd may exist (then
//! `ε³ = η₀`), detected by solving `d·b³ + 3ν·b = 2k` (`ν = N(η₀) = ±1`) for a
//! positive integer `b`.
//!
//! **Regulator.** `R = |det|` of the `r × r` logarithmic-embedding matrix, at
//! `Float` precision.
//!
//! # Scope and limitations
//!
//! Fundamental units are delivered **exactly for real quadratic fields**. Roots
//! of unity, the unit rank, and (given a set of fundamental units) the regulator
//! are computed for **any** field. A general fundamental-unit search for rank
//! `r ≥ 2` (via `S`-unit / relation lattices) is **not** implemented; for such
//! fields [`UnitGroup::fundamental_units`] is empty and the regulator is
//! reported as NaN. See the crate roadmap.

// Dense index-range linear algebra (log-embedding matrices, resultant products)
// reads closer to the mathematics with explicit indices than iterator adapters.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;
use core::fmt;

use crate::complex::Complex;
use crate::float::{Float, RoundingMode};
use crate::int::Int;
use crate::numberfield::{NumberField, NumberFieldElement};
use crate::numberfield_ideal::Order;
use crate::poly::Poly;
use crate::rational::Rational;

const NEAR: RoundingMode = RoundingMode::Nearest;

// ===========================================================================
// UnitGroup.
// ===========================================================================

/// The unit group `O_K^* ≅ μ_K × ℤ^r` of a number field, as returned by
/// [`Order::unit_group`].
///
/// Carries the [`rank`](UnitGroup::rank) `r = r₁ + r₂ − 1`, the roots of unity
/// `μ_K` (its order [`torsion_order`](UnitGroup::torsion_order) `w_K` and a
/// [`torsion_generator`](UnitGroup::torsion_generator)), the known
/// [`fundamental_units`](UnitGroup::fundamental_units), and the
/// [`regulator`](UnitGroup::regulator).
#[derive(Clone)]
pub struct UnitGroup {
    field: NumberField,
    rank: usize,
    torsion_order: Int,
    torsion_generator: NumberFieldElement,
    fundamental_units: Vec<NumberFieldElement>,
}

impl fmt::Debug for UnitGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UnitGroup(rank {}, w_K {}, fundamental units {})",
            self.rank,
            self.torsion_order,
            self.fundamental_units.len()
        )
    }
}

impl UnitGroup {
    /// The unit rank `r = r₁ + r₂ − 1`.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The order `w_K = |μ_K|` of the group of roots of unity in `K`.
    #[inline]
    pub fn torsion_order(&self) -> Int {
        self.torsion_order.clone()
    }

    /// A generator of the cyclic group `μ_K` of roots of unity (a primitive
    /// `w_K`-th root of unity in `K`).
    pub fn torsion_generator(&self) -> NumberFieldElement {
        self.torsion_generator.clone()
    }

    /// The known fundamental units `ε₁, …, ε_r`.
    ///
    /// This is complete (`r` units) for real quadratic fields and empty for
    /// fields of rank `r ≥ 2` (whose fundamental-unit search is not
    /// implemented); see the module documentation.
    pub fn fundamental_units(&self) -> &[NumberFieldElement] {
        &self.fundamental_units
    }

    /// The regulator `R_K` at `precision` bits.
    ///
    /// Returns `1` for rank `0` (the empty determinant). For rank `r ≥ 1` it is
    /// the `|det|` of the `r × r` logarithmic-embedding matrix of the
    /// [`fundamental_units`](UnitGroup::fundamental_units); if fewer than `r`
    /// units are known (rank `≥ 2`, unimplemented) it returns NaN.
    pub fn regulator(&self, precision: u64) -> Float {
        if self.rank == 0 {
            return Float::from_int(&Int::ONE, precision, NEAR);
        }
        if self.fundamental_units.len() != self.rank {
            return Float::nan(precision);
        }
        regulator_from_units(&self.field, &self.fundamental_units, precision)
    }
}

impl Order {
    /// The unit group `O_K^*` of this order (assumed the maximal order).
    ///
    /// Computes the unit rank, the roots of unity `μ_K`, and — for real
    /// quadratic fields — the fundamental unit and regulator. See the
    /// [module documentation](crate::numberfield_units) for scope.
    pub fn unit_group(&self) -> UnitGroup {
        let field = self.field();
        unit_group_of(&field)
    }
}

impl NumberField {
    /// The fundamental unit `ε > 1` of a **real quadratic** field, or `None` if
    /// the field is not real quadratic.
    ///
    /// Computed exactly from the continued fraction of `√d` / the Pell equation;
    /// see the [module documentation](crate::numberfield_units).
    pub fn fundamental_unit(&self) -> Option<NumberFieldElement> {
        real_quadratic_fundamental_unit(self)
    }

    /// The regulator `R_K` at `precision` bits (see [`UnitGroup::regulator`]).
    pub fn regulator(&self, precision: u64) -> Float {
        unit_group_of(self).regulator(precision)
    }
}

/// Builds the [`UnitGroup`] of `field`.
fn unit_group_of(field: &NumberField) -> UnitGroup {
    let (r1, r2) = field.signature();
    let rank = r1 + r2 - 1;
    let (torsion_order, torsion_generator) = roots_of_unity(field);

    let mut fundamental_units = Vec::new();
    if rank == 1 && r1 == 2 {
        // Real quadratic: fundamental unit computed exactly.
        if let Some(eps) = real_quadratic_fundamental_unit(field) {
            fundamental_units.push(eps);
        }
    }

    UnitGroup {
        field: field.clone(),
        rank,
        torsion_order,
        torsion_generator,
        fundamental_units,
    }
}

// ===========================================================================
// Real-quadratic fundamental unit (continued fractions / Pell).
// ===========================================================================

/// The fundamental unit `ε > 1` of a real quadratic field, or `None` if `field`
/// is not real quadratic (degree 2 with two real embeddings).
fn real_quadratic_fundamental_unit(field: &NumberField) -> Option<NumberFieldElement> {
    if field.degree() != 2 {
        return None;
    }
    let (r1, _r2) = field.signature();
    if r1 != 2 {
        return None;
    }
    // T = x² + p·x + q (monic), θ² + p·θ + q = 0.
    let t = field.defining_polynomial();
    let p = t.coeff(1);
    let q = t.coeff(0);
    // Δ = p² − 4q > 0; write Δ = s²·d with d squarefree, s = f/den rational.
    let delta = p
        .mul(&p)
        .sub(&Rational::from_integer(Int::from_i64(4)).mul(&q));
    debug_assert!(delta.is_positive(), "real quadratic ⇒ Δ > 0");
    // N = num·den; √Δ = √N / den. Factor out the square part of N.
    let big_n = delta.numerator().mul(delta.denominator());
    let (d, f) = squarefree_decompose(&big_n);
    let den = delta.denominator().clone();
    // √d = (2θ + p) · (den / f)   (up to sign; fixed below).
    let scale = Rational::new(den, f);
    let two = Rational::from_integer(Int::from_i64(2));
    let sqrt_d = field.element(Poly::new(alloc::vec![p.mul(&scale), two.mul(&scale)]));

    // Pell fundamental solution of x² − d·y² = ±1 (from CF of √d).
    let (h, k, nu) = pell_fundamental(&d);

    // Coefficients (a_num, b_num) of ε = (a_num + b_num·√d)/2.
    let (a_num, b_num) = if d.rem_euclid(&Int::from_i64(4)).is_one() {
        // d ≡ 1 (mod 4): a smaller half-integer unit ε=(a+b√d)/2 with ε³=η₀ may
        // exist. Solve d·b³ + 3ν·b = 2k for a positive integer b.
        match solve_cube(&d, nu, &k) {
            Some(b) => {
                // a = 2h / (d·b² + ν).
                let denom = d.mul(&b.mul(&b)).add(&Int::from_i64(nu as i64));
                let a = h.mul(&Int::from_i64(2)).div_exact(&denom);
                (a, b)
            }
            None => (h.mul(&Int::from_i64(2)), k.mul(&Int::from_i64(2))),
        }
    } else {
        // d ≡ 2,3 (mod 4): O_K = ℤ[√d], ε = h + k√d.
        (h.mul(&Int::from_i64(2)), k.mul(&Int::from_i64(2)))
    };

    // ε = (a_num + b_num·√d)/2, with √d = ±sqrt_d; pick the sign giving |ε| > 1.
    let half_a = Rational::new(a_num, Int::from_i64(2));
    let half_b = Rational::new(b_num, Int::from_i64(2));
    let base = field.from_rational(half_a);
    let coeff_b = field.from_rational(half_b);
    let cand = base.add(&coeff_b.mul(&sqrt_d));
    let eps = if embedding_abs_gt_one(&cand) {
        cand
    } else {
        base.sub(&coeff_b.mul(&sqrt_d))
    };
    Some(eps)
}

/// Whether some embedding of `elt` has absolute value `> 1` (used to pick the
/// fundamental unit `ε > 1` over its conjugate `±1/ε`).
fn embedding_abs_gt_one(elt: &NumberFieldElement) -> bool {
    let prec = 96u64;
    let one = Float::from_int(&Int::ONE, prec, NEAR);
    elt.embeddings(prec).iter().any(|z| z.abs() > one)
}

/// Squarefree decomposition of a positive integer `m = f²·d` with `d`
/// squarefree; returns `(d, f)`.
fn squarefree_decompose(m: &Int) -> (Int, Int) {
    let mut d = Int::ONE;
    let mut f = Int::ONE;
    for (p, e) in m.factor_exponents() {
        if e % 2 == 1 {
            d = d.mul(&p);
        }
        if e / 2 > 0 {
            f = f.mul(&p.pow(e / 2));
        }
    }
    (d, f)
}

/// Floor of `√m` for a non-negative integer `m`.
fn isqrt_int(m: &Int) -> Int {
    Int::from(m.magnitude().isqrt())
}

/// The fundamental solution `(h, k)` of the Pell equation `x² − d·y² = ±1`
/// (minimal positive), together with its norm `ν = h² − d·k² ∈ {+1, −1}`, via
/// the continued fraction of `√d` (`d` a positive non-square).
fn pell_fundamental(d: &Int) -> (Int, Int, i32) {
    let a0 = isqrt_int(d);
    // PQa recursion for √d: m₀=0, q₀=1, a=a0.
    let mut m = Int::ZERO;
    let mut qv = Int::ONE;
    let mut a = a0.clone();
    // Convergents: h_{-1}=1, h₀=a0 ; k_{-1}=0, k₀=1.
    let mut h_prev = Int::ONE;
    let mut h = a0.clone();
    let mut k_prev = Int::ZERO;
    let mut k = Int::ONE;

    loop {
        // Advance (i → i+1).
        m = a.mul(&qv).sub(&m); // m_{i+1} = a_i·q_i − m_i
        let d_minus = d.sub(&m.mul(&m));
        qv = d_minus.div_exact(&qv); // q_{i+1} = (d − m²)/q_i
        if qv.is_one() {
            // End of period: (h, k) is the fundamental solution.
            let nu = h.mul(&h).sub(&d.mul(&k.mul(&k)));
            let sign = if nu.is_negative() { -1 } else { 1 };
            return (h, k, sign);
        }
        a = a0.add(&m).div_floor(&qv); // a_{i+1} = ⌊(a0 + m)/q⌋
        let h_new = a.mul(&h).add(&h_prev);
        h_prev = h;
        h = h_new;
        let k_new = a.mul(&k).add(&k_prev);
        k_prev = k;
        k = k_new;
    }
}

/// Solves `d·b³ + 3ν·b = 2k` for a positive integer `b`, returning it if it
/// exists (detects the half-integer fundamental unit in the `d ≡ 1 (mod 4)`
/// case). `d·b³ + 3ν·b` is strictly increasing in `b` for `b ≥ 1` (`d ≥ 5`), so
/// there is at most one root.
fn solve_cube(d: &Int, nu: i32, k: &Int) -> Option<Int> {
    let target = k.mul(&Int::from_i64(2));
    let three_nu = Int::from_i64(3 * nu as i64);
    // b ≤ ⌊(2k/d)^{1/3}⌋ + 2.
    let bound = {
        let quotient = target.div_floor(d);
        let root = quotient.magnitude().nth_root_floor(3);
        Int::from(root).add(&Int::from_i64(2))
    };
    let mut b = Int::ONE;
    while b <= bound {
        let val = d.mul(&b.pow(3)).add(&three_nu.mul(&b));
        if val == target {
            return Some(b);
        }
        if val > target {
            return None;
        }
        b = b.add(&Int::ONE);
    }
    None
}

// ===========================================================================
// Roots of unity μ_K (Trager's norm test).
// ===========================================================================

/// The group of roots of unity `μ_K`: its order `w_K` and a generator (a
/// primitive `w_K`-th root of unity in `K`).
fn roots_of_unity(field: &NumberField) -> (Int, NumberFieldElement) {
    let n = field.degree();
    // Candidate orders m with φ(m) | n; μ_K is cyclic so w_K is the largest such
    // m for which K contains a primitive m-th root of unity (baseline 2: −1).
    let cap = 2 * (n as u64) * (n as u64) + 64;
    let mut best = 2u64;
    for m in 3..=cap {
        if !(n as u64).is_multiple_of(euler_phi(m)) {
            continue;
        }
        if m > best && field_contains_primitive_root(field, m) {
            best = m;
        }
    }
    let generator = if best == 2 {
        field.from_rational(Rational::from_integer(Int::from_i64(-1)))
    } else {
        extract_primitive_root(field, best)
            .unwrap_or_else(|| field.from_rational(Rational::from_integer(Int::from_i64(-1))))
    };
    (Int::from_i64(best as i64), generator)
}

/// Euler's totient `φ(m)`.
fn euler_phi(m: u64) -> u64 {
    let mut n = m;
    let mut result = m;
    let mut p = 2u64;
    while p * p <= n {
        if n.is_multiple_of(p) {
            while n.is_multiple_of(p) {
                n /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if n > 1 {
        result -= result / n;
    }
    result
}

/// The `m`-th cyclotomic polynomial `Φ_m` as a monic integer polynomial
/// (coefficients low-to-high), computed from `x^m − 1 = ∏_{d|m} Φ_d`.
fn cyclotomic(m: u64) -> Poly<Rational> {
    // Numerator x^m − 1.
    let mut coeffs = alloc::vec![Rational::ZERO; (m + 1) as usize];
    coeffs[0] = Rational::from_integer(Int::from_i64(-1));
    coeffs[m as usize] = Rational::ONE;
    let mut num = Poly::new(coeffs);
    for dd in 1..m {
        if m.is_multiple_of(dd) {
            let phi_d = cyclotomic(dd);
            num = num.div_rem(&phi_d).0;
        }
    }
    num
}

/// The distinct prime factors of `m > 0`.
fn prime_factors(mut m: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut p = 2u64;
    while p * p <= m {
        if m.is_multiple_of(p) {
            out.push(p);
            while m.is_multiple_of(p) {
                m /= p;
            }
        }
        p += 1;
    }
    if m > 1 {
        out.push(m);
    }
    out
}

/// Whether `K` contains a primitive `m`-th root of unity, via Trager's norm
/// test: `Φ_m` has a root in `K` iff the resultant `N(x) = ∏_i Φ_m(x − s·θ_i)`
/// (for a shift `s` making it squarefree) has an irreducible rational factor of
/// degree exactly `n = [K:ℚ]`.
fn field_contains_primitive_root(field: &NumberField, m: u64) -> bool {
    let n = field.degree();
    let phi = cyclotomic(m);
    let e = phi.degree().unwrap_or(0);
    if e == 0 || !n.is_multiple_of(e) {
        return false;
    }
    let g: Vec<Int> = (0..=e).map(|i| phi.coeff(i).numerator().clone()).collect();
    for prec in [160u64, 320, 640] {
        let roots = field.complex_roots(prec);
        for s in 0..=(2 * n as i64 + 4) {
            if let Some(np) = shifted_resultant(&g, &roots, s, prec + 64) {
                let npoly = Poly::new(np.into_iter().map(Rational::from_integer).collect());
                if !is_squarefree(&npoly) {
                    continue;
                }
                return npoly
                    .factor()
                    .iter()
                    .any(|(fac, _)| fac.degree() == Some(n));
            }
            // Rounding failed at this precision: try a larger shift, then bump.
        }
    }
    false
}

/// Recovers a primitive `m`-th root of unity in `K` (assuming it exists) as the
/// linear factor of `gcd_{K[x]}(Φ_m(x), N_i(x + s·θ))`, where `N_i` is the
/// degree-`n` rational factor of the Trager resultant.
fn extract_primitive_root(field: &NumberField, m: u64) -> Option<NumberFieldElement> {
    let n = field.degree();
    let phi = cyclotomic(m);
    let e = phi.degree().unwrap_or(0);
    if e == 0 || !n.is_multiple_of(e) {
        return None;
    }
    let g: Vec<Int> = (0..=e).map(|i| phi.coeff(i).numerator().clone()).collect();
    for prec in [160u64, 320, 640] {
        let roots = field.complex_roots(prec);
        for s in 0..=(2 * n as i64 + 4) {
            let np = match shifted_resultant(&g, &roots, s, prec + 64) {
                Some(v) => v,
                None => continue,
            };
            let npoly = Poly::new(np.into_iter().map(Rational::from_integer).collect());
            if !is_squarefree(&npoly) {
                continue;
            }
            let (ni, _) = npoly
                .factor()
                .into_iter()
                .find(|(fac, _)| fac.degree() == Some(n))?;
            if let Some(zeta) = root_via_gcd(field, &phi, &ni, s, m) {
                return Some(zeta);
            }
        }
    }
    None
}

/// Computes `gcd_{K[x]}(Φ_m(x), N_i(x + s·θ))` and returns its linear root, if
/// that root is a primitive `m`-th root of unity.
fn root_via_gcd(
    field: &NumberField,
    phi: &Poly<Rational>,
    ni: &Poly<Rational>,
    s: i64,
    m: u64,
) -> Option<NumberFieldElement> {
    let one_k = field.one();
    // Φ_m lifted to K[x] with constant field coefficients.
    let g_k = Poly::new(
        (0..=phi.degree().unwrap_or(0))
            .map(|i| field.from_rational(phi.coeff(i)))
            .collect(),
    );
    // N_i(x + s·θ) ∈ K[x]: Horner in K[x] with the linear (x + s·θ).
    let s_theta = field
        .generator()
        .mul(&field.from_rational(Rational::from_integer(Int::from_i64(s))));
    let lin = Poly::new(alloc::vec![s_theta, one_k.clone()]);
    let deg = ni.degree().unwrap_or(0);
    let mut m_k = Poly::<NumberFieldElement>::zero();
    for i in (0..=deg).rev() {
        m_k = m_k.mul(&lin);
        m_k = m_k.add(&Poly::constant(field.from_rational(ni.coeff(i))));
    }
    let g = g_k.gcd(&m_k);
    if g.degree() != Some(1) {
        return None;
    }
    // Monic linear a·x + b ⇒ root ζ = −b/a.
    let a = g.coeff(1);
    let b = g.coeff(0);
    let zeta = b.neg().div(&a);
    // Verify ζ is a primitive m-th root of unity in K.
    if zeta.pow(m as i64) != field.one() {
        return None;
    }
    for p in prime_factors(m) {
        if zeta.pow((m / p) as i64) == field.one() {
            return None;
        }
    }
    Some(zeta)
}

/// Whether a rational polynomial is squarefree (`gcd(f, f')` is constant).
fn is_squarefree(f: &Poly<Rational>) -> bool {
    let d = f.derivative();
    if d.is_zero() {
        return false;
    }
    f.gcd(&d).degree() == Some(0)
}

/// The Trager resultant `N(x) = ∏_i g(x − s·θ_i)` as an exact integer polynomial
/// (coefficients low-to-high), evaluated at the numeric roots `θ_i` and rounded;
/// returns `None` if the rounding is not safe at working precision `wp`.
fn shifted_resultant(g: &[Int], roots: &[Complex<Float>], s: i64, wp: u64) -> Option<Vec<Int>> {
    let s_fl = Float::from_int(&Int::from_i64(s), wp, NEAR);
    let mut acc: Vec<Complex<Float>> = alloc::vec![Complex::new(
        Float::from_int(&Int::ONE, wp, NEAR),
        Float::zero(wp)
    )];
    for theta in roots {
        // a = −s·θ, then factor = g(x + a).
        let a = Complex::new(
            theta.re.mul(&s_fl, wp, NEAR).neg(),
            theta.im.mul(&s_fl, wp, NEAR).neg(),
        );
        let mut factor: Vec<Complex<Float>> = Vec::new();
        for i in (0..g.len()).rev() {
            factor = poly_mul_linear(&factor, &a, wp);
            let gk = Complex::new(Float::from_int(&g[i], wp, NEAR), Float::zero(wp));
            if factor.is_empty() {
                factor.push(gk);
            } else {
                factor[0] = factor[0].add(&gk);
            }
        }
        acc = poly_mul_complex(&acc, &factor, wp);
    }
    let mut out = Vec::with_capacity(acc.len());
    for z in &acc {
        out.push(round_complex_to_int(z, wp)?);
    }
    Some(out)
}

/// `f · (x + a)` for a complex-coefficient polynomial `f` (low-to-high).
fn poly_mul_linear(f: &[Complex<Float>], a: &Complex<Float>, wp: u64) -> Vec<Complex<Float>> {
    if f.is_empty() {
        return Vec::new();
    }
    let mut r = alloc::vec![Complex::new(Float::zero(wp), Float::zero(wp)); f.len() + 1];
    for (i, c) in f.iter().enumerate() {
        r[i + 1] = r[i + 1].add(c);
        r[i] = r[i].add(&c.mul(a));
    }
    r
}

/// Product of two complex-coefficient polynomials (low-to-high).
fn poly_mul_complex(a: &[Complex<Float>], b: &[Complex<Float>], wp: u64) -> Vec<Complex<Float>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out =
        alloc::vec![Complex::new(Float::zero(wp), Float::zero(wp)); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] = out[i + j].add(&x.mul(y));
        }
    }
    out
}

/// Rounds a near-integer complex value to the nearest integer, or `None` if it
/// is not within `1/4` of a real integer (rounding unsafe at this precision).
fn round_complex_to_int(z: &Complex<Float>, wp: u64) -> Option<Int> {
    let quarter = Float::from_rational(&Rational::new(Int::ONE, Int::from_i64(4)), wp, NEAR);
    if z.im.abs() >= quarter {
        return None;
    }
    let rounded = z.re.round_to_int()?;
    let diff =
        z.re.sub(&Float::from_int(&rounded, wp, NEAR), wp, NEAR)
            .abs();
    if diff < quarter { Some(rounded) } else { None }
}

// ===========================================================================
// Regulator.
// ===========================================================================

/// The regulator `R = |det|` of the `r × r` logarithmic-embedding matrix of the
/// given fundamental units (`units.len()` must equal the rank `r`).
fn regulator_from_units(
    field: &NumberField,
    units: &[NumberFieldElement],
    precision: u64,
) -> Float {
    let r = units.len();
    let wp = precision + 64;
    // Infinite places: real embeddings (weight 1) and one representative per
    // complex-conjugate pair (weight 2). There are r₁ + r₂ = r + 1 of them;
    // drop the last to get an r × r matrix.
    let (place_idx, place_wt) = places(field, wp);
    debug_assert_eq!(place_idx.len(), r + 1);

    // matrix[i][j] = wt_i · log|σ_{place_i}(ε_j)|, i,j = 0..r.
    let mut matrix: Vec<Vec<Float>> = Vec::with_capacity(r);
    for i in 0..r {
        let mut row = Vec::with_capacity(r);
        for eps in units.iter() {
            let embs = eps.embeddings(wp);
            let idx = place_idx[i];
            let mag = embs[idx].abs();
            let mut entry = mag.ln(wp, NEAR);
            if place_wt[i] == 2 {
                entry = entry.mul(&Float::from_int(&Int::from_i64(2), wp, NEAR), wp, NEAR);
            }
            row.push(entry);
        }
        matrix.push(row);
    }
    let det = float_determinant(matrix, wp);
    det.abs().round(precision, NEAR)
}

/// The infinite places of `field`: indices (into the embedding vector) and
/// weights (`1` real, `2` complex). One representative per complex pair.
fn places(field: &NumberField, wp: u64) -> (Vec<usize>, Vec<usize>) {
    let roots = field.complex_roots(wp);
    let threshold = Float::from_rational(&Rational::power_of_two(-(wp as i32 / 4)), wp, NEAR);
    let mut idx = Vec::new();
    let mut wt = Vec::new();
    for (i, z) in roots.iter().enumerate() {
        if z.im.abs() < threshold {
            idx.push(i);
            wt.push(1);
        } else if z.im > Float::zero(wp) {
            // One representative per conjugate pair (positive imaginary part).
            idx.push(i);
            wt.push(2);
        }
    }
    (idx, wt)
}

/// Signed determinant of a square `Float` matrix by Gaussian elimination with
/// partial pivoting.
fn float_determinant(mut a: Vec<Vec<Float>>, wp: u64) -> Float {
    let n = a.len();
    if n == 0 {
        return Float::from_int(&Int::ONE, wp, NEAR);
    }
    let mut det = Float::from_int(&Int::ONE, wp, NEAR);
    for col in 0..n {
        // Partial pivot: largest |a[row][col]| for row ≥ col.
        let mut piv = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if a[piv][col].is_zero() {
            return Float::zero(wp);
        }
        if piv != col {
            a.swap(piv, col);
            det = det.neg();
        }
        det = det.mul(&a[col][col], wp, NEAR);
        // Eliminate below.
        for row in (col + 1)..n {
            let factor = a[row][col].div(&a[col][col], wp, NEAR);
            for c in col..n {
                let sub = factor.mul(&a[col][c], wp, NEAR);
                a[row][c] = a[row][c].sub(&sub, wp, NEAR);
            }
        }
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(v: i64) -> Rational {
        Rational::from_integer(Int::from_i64(v))
    }

    fn poly(coeffs: &[i64]) -> Poly<Rational> {
        Poly::new(coeffs.iter().map(|&c| q(c)).collect())
    }

    fn field(coeffs: &[i64]) -> NumberField {
        NumberField::new(poly(coeffs)).unwrap()
    }

    /// ℚ(√m) via x² − m.
    fn qsqrt(m: i64) -> NumberField {
        field(&[-m, 0, 1])
    }

    // ---- Real quadratic fundamental units (known values). ----

    /// Checks that `eps` is a genuine unit `> 1` with norm ±1 and an integral
    /// inverse (a true unit of O_K).
    fn check_unit(eps: &NumberFieldElement) {
        let nrm = eps.norm();
        assert!(nrm.is_one() || nrm == q(-1), "norm must be ±1, got {nrm}");
        assert!(eps.is_algebraic_integer(), "ε must be an algebraic integer");
        let inv = eps.inv().unwrap();
        assert!(
            inv.is_algebraic_integer(),
            "ε⁻¹ must be an algebraic integer"
        );
        assert!(eps.mul(&inv).is_one());
        assert!(embedding_abs_gt_one(eps), "ε must exceed 1");
    }

    #[test]
    fn fundamental_unit_sqrt2() {
        // 1 + √2
        let k = qsqrt(2);
        let eps = k.fundamental_unit().unwrap();
        check_unit(&eps);
        assert_eq!(eps, k.generator().add(&k.from_rational(q(1))));
    }

    #[test]
    fn fundamental_unit_sqrt3() {
        // 2 + √3
        let k = qsqrt(3);
        let eps = k.fundamental_unit().unwrap();
        check_unit(&eps);
        assert_eq!(eps, k.generator().add(&k.from_rational(q(2))));
    }

    #[test]
    fn fundamental_unit_sqrt5() {
        // (1 + √5)/2
        let k = qsqrt(5);
        let eps = k.fundamental_unit().unwrap();
        check_unit(&eps);
        let half = Rational::new(Int::ONE, Int::from_i64(2));
        let expected = k
            .from_rational(half.clone())
            .add(&k.generator().mul(&k.from_rational(half)));
        assert_eq!(eps, expected);
    }

    #[test]
    fn fundamental_unit_sqrt6() {
        // 5 + 2√6
        let k = qsqrt(6);
        let eps = k.fundamental_unit().unwrap();
        check_unit(&eps);
        assert_eq!(
            eps,
            k.from_rational(q(5))
                .add(&k.generator().mul(&k.from_rational(q(2))))
        );
    }

    #[test]
    fn fundamental_unit_sqrt7() {
        // 8 + 3√7
        let k = qsqrt(7);
        let eps = k.fundamental_unit().unwrap();
        check_unit(&eps);
        assert_eq!(
            eps,
            k.from_rational(q(8))
                .add(&k.generator().mul(&k.from_rational(q(3))))
        );
    }

    #[test]
    fn fundamental_unit_sqrt13() {
        // (3 + √13)/2
        let k = qsqrt(13);
        let eps = k.fundamental_unit().unwrap();
        check_unit(&eps);
        let half = Rational::new(Int::ONE, Int::from_i64(2));
        let expected = k
            .from_rational(Rational::new(Int::from_i64(3), Int::from_i64(2)))
            .add(&k.generator().mul(&k.from_rational(half)));
        assert_eq!(eps, expected);
    }

    #[test]
    fn fundamental_unit_via_integral_generator() {
        // ℚ(√5) presented by x² − x − 1 (θ = (1+√5)/2 = the fundamental unit).
        let k = field(&[-1, -1, 1]);
        let eps = k.fundamental_unit().unwrap();
        check_unit(&eps);
        assert_eq!(eps, k.generator());
    }

    #[test]
    fn imaginary_quadratic_has_no_fundamental_unit() {
        assert!(qsqrt(-1).fundamental_unit().is_none());
        assert!(qsqrt(-3).fundamental_unit().is_none());
    }

    #[test]
    #[ignore = "slow: large-period continued fraction (ℚ(√94))"]
    fn fundamental_unit_sqrt94() {
        // 2143295 + 221064√94
        let k = qsqrt(94);
        let eps = k.fundamental_unit().unwrap();
        check_unit(&eps);
        let expected = k
            .from_rational(q(2143295))
            .add(&k.generator().mul(&k.from_rational(q(221064))));
        assert_eq!(eps, expected);
    }

    // ---- Regulator. ----

    #[test]
    fn regulator_matches_log_eps() {
        // R = log ε for a real quadratic field.
        let prec = 128u64;
        for m in [2i64, 3, 5, 6, 7, 13] {
            let k = qsqrt(m);
            let eps = k.fundamental_unit().unwrap();
            let r = k.regulator(prec);
            // Reference: log of the larger embedding magnitude.
            let embs = eps.embeddings(prec + 32);
            let mag = embs
                .iter()
                .map(|z| z.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let reference = mag.ln(prec + 32, NEAR);
            let diff = r.sub(&reference, prec + 32, NEAR).abs();
            assert!(
                diff.to_f64() < 1e-25,
                "regulator mismatch for √{m}: {r} vs {reference}"
            );
        }
    }

    #[test]
    fn regulator_rank_zero_is_one() {
        // Imaginary quadratic: rank 0, regulator 1.
        let k = qsqrt(-1); // ℚ(i)
        let r = k.regulator(64);
        let one = Float::from_int(&Int::ONE, 64, NEAR);
        assert!(r.sub(&one, 64, NEAR).abs().to_f64() < 1e-18);
    }

    // ---- Roots of unity (known w_K). ----

    #[test]
    fn roots_of_unity_gaussian() {
        // ℚ(i): w_K = 4, generator a primitive 4th root (i² = −1, i⁴ = 1).
        let k = qsqrt(-1);
        let ug = k.maximal_order().unit_group();
        assert_eq!(ug.torsion_order(), Int::from_i64(4));
        let g = ug.torsion_generator();
        assert_eq!(g.pow(4), k.one());
        assert!(g.pow(2) != k.one());
    }

    #[test]
    fn roots_of_unity_eisenstein() {
        // ℚ(√−3): w_K = 6.
        let k = qsqrt(-3);
        let ug = k.maximal_order().unit_group();
        assert_eq!(ug.torsion_order(), Int::from_i64(6));
        let g = ug.torsion_generator();
        assert_eq!(g.pow(6), k.one());
        assert!(g.pow(3) != k.one());
        assert!(g.pow(2) != k.one());
    }

    #[test]
    fn roots_of_unity_real_quadratic() {
        // Real fields: only ±1, w_K = 2.
        for m in [2i64, 3, 5] {
            let k = qsqrt(m);
            let ug = k.maximal_order().unit_group();
            assert_eq!(ug.torsion_order(), Int::from_i64(2), "√{m}");
            assert_eq!(ug.torsion_generator(), k.from_rational(q(-1)));
        }
    }

    #[test]
    fn roots_of_unity_cyclotomic5() {
        // ℚ(ζ₅): w_K = 10.
        let k = field(&[1, 1, 1, 1, 1]);
        let ug = k.maximal_order().unit_group();
        assert_eq!(ug.torsion_order(), Int::from_i64(10));
        let g = ug.torsion_generator();
        assert_eq!(g.pow(10), k.one());
        assert!(g.pow(5) != k.one());
        assert!(g.pow(2) != k.one());
    }

    // ---- Unit rank across signatures. ----

    #[test]
    fn ranks() {
        // Imaginary quadratic r=0.
        assert_eq!(qsqrt(-1).maximal_order().unit_group().rank(), 0);
        assert_eq!(qsqrt(-3).maximal_order().unit_group().rank(), 0);
        // Real quadratic r=1.
        assert_eq!(qsqrt(2).maximal_order().unit_group().rank(), 1);
        // ℚ(∛2): (r1,r2)=(1,1) ⇒ r=1.
        assert_eq!(field(&[-2, 0, 0, 1]).maximal_order().unit_group().rank(), 1);
        // Totally real cubic x³ − 3x − 1: (r1,r2)=(3,0) ⇒ r=2.
        assert_eq!(
            field(&[-1, -3, 0, 1]).maximal_order().unit_group().rank(),
            2
        );
    }

    #[test]
    fn rank_two_partial() {
        // Honest-partial: rank ≥ 2 fundamental-unit search is unimplemented.
        let k = field(&[-1, -3, 0, 1]); // totally real cubic, rank 2
        let ug = k.maximal_order().unit_group();
        assert_eq!(ug.rank(), 2);
        assert_eq!(ug.torsion_order(), Int::from_i64(2));
        assert!(ug.fundamental_units().is_empty());
        assert!(ug.regulator(64).is_nan());
    }

    // ---- Internal helpers. ----

    #[test]
    fn pell_small() {
        // x² − 2y² = ±1 minimal: (1,1), norm −1.
        let (h, k, nu) = pell_fundamental(&Int::from_i64(2));
        assert_eq!(h, Int::ONE);
        assert_eq!(k, Int::ONE);
        assert_eq!(nu, -1);
        // x² − 3y² : (2,1), norm +1.
        let (h, k, nu) = pell_fundamental(&Int::from_i64(3));
        assert_eq!(h, Int::from_i64(2));
        assert_eq!(k, Int::ONE);
        assert_eq!(nu, 1);
    }

    #[test]
    fn cyclotomic_values() {
        // Φ₄ = x² + 1, Φ₆ = x² − x + 1.
        assert_eq!(cyclotomic(4), poly(&[1, 0, 1]));
        assert_eq!(cyclotomic(6), poly(&[1, -1, 1]));
        assert_eq!(euler_phi(10), 4);
        assert_eq!(euler_phi(12), 4);
    }
}
