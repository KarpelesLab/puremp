//! Lattice basis reduction.
//!
//! [`lll_reduce`] applies the Lenstra–Lenstra–Lovász algorithm to an integer
//! lattice basis, returning a reduced basis of short, nearly-orthogonal vectors
//! spanning the *same* lattice (the change of basis is unimodular). The
//! Gram–Schmidt bookkeeping is carried in exact [`Rational`] arithmetic, so the
//! reduction is exact — no floating-point heuristics — matching the rest of the
//! crate.
//!
//! LLL underpins integer-relation detection, minimal-polynomial recovery for
//! algebraic numbers, Diophantine approximation, and polynomial factorization.
//!
//! Reference: A. K. Lenstra, H. W. Lenstra, L. Lovász, *Factoring Polynomials
//! with Rational Coefficients*, Mathematische Annalen 261 (1982).

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::int::Int;
use crate::rational::Rational;

/// LLL-reduces `basis` with the classic reduction parameter `δ = 3/4`.
///
/// See [`lll_reduce_delta`] for the general form and the precise contract.
pub fn lll_reduce(basis: &[Vec<Int>]) -> Vec<Vec<Int>> {
    lll_reduce_delta(basis, &Rational::new(Int::from_i64(3), Int::from_i64(4)))
}

/// LLL-reduces `basis` (rows = basis vectors, all of equal dimension) with
/// reduction parameter `delta`, which must lie in `(1/4, 1]` — larger values
/// give a better-reduced basis but more work (`3/4` is the classic choice, and
/// values up to `0.99` are common). The vectors must be linearly independent;
/// if a degenerate (zero-length Gram–Schmidt) vector is encountered the input is
/// returned unchanged.
///
/// The returned basis spans the same lattice as the input and is LLL-reduced:
/// its Gram–Schmidt coefficients satisfy `|μ_{i,j}| ≤ 1/2` and the Lovász
/// condition `‖b*_k‖² ≥ (δ − μ_{k,k-1}²)‖b*_{k-1}‖²` holds for every `k`.
pub fn lll_reduce_delta(basis: &[Vec<Int>], delta: &Rational) -> Vec<Vec<Int>> {
    let n = basis.len();
    if n <= 1 {
        return basis.to_vec();
    }
    let dim = basis[0].len();
    assert!(
        basis.iter().all(|v| v.len() == dim),
        "lll_reduce: all basis vectors must share the same dimension"
    );

    let mut b: Vec<Vec<Int>> = basis.to_vec();
    let (mut mu, mut bstar_norm) = match gram_schmidt(&b) {
        Some(gs) => gs,
        None => return basis.to_vec(), // linearly dependent input
    };
    let half = Rational::new(Int::ONE, Int::from_i64(2));

    let mut k = 1;
    while k < n {
        size_reduce(&mut b, &mut mu, k, k - 1, &half);

        // Lovász condition: ‖b*_k‖² ≥ (δ − μ_{k,k-1}²)·‖b*_{k-1}‖².
        let mk = mu[k][k - 1].clone();
        let bound = Rational::mul(
            &Rational::sub(delta, &Rational::mul(&mk, &mk)),
            &bstar_norm[k - 1],
        );
        if bstar_norm[k].cmp(&bound) != Ordering::Less {
            // Fully size-reduce b_k against the remaining earlier vectors.
            for l in (0..k - 1).rev() {
                size_reduce(&mut b, &mut mu, k, l, &half);
            }
            k += 1;
        } else {
            swap_step(&mut b, &mut mu, &mut bstar_norm, k, n);
            k = if k > 1 { k - 1 } else { 1 };
        }
    }
    b
}

/// Gram–Schmidt orthogonalization in exact rationals. Returns the coefficient
/// matrix `μ` (only entries `j < i` are meaningful) and the squared norms
/// `‖b*_i‖²`, or `None` if the basis is linearly dependent.
fn gram_schmidt(b: &[Vec<Int>]) -> Option<(Vec<Vec<Rational>>, Vec<Rational>)> {
    let n = b.len();
    let dim = b[0].len();
    let mut mu = vec![vec![Rational::ZERO; n]; n];
    let mut norm = vec![Rational::ZERO; n];
    let mut bstar: Vec<Vec<Rational>> = Vec::with_capacity(n);

    for i in 0..n {
        let mut bi: Vec<Rational> = b[i]
            .iter()
            .map(|x| Rational::from_integer(x.clone()))
            .collect();
        for j in 0..i {
            // μ_{i,j} = ⟨b_i, b*_j⟩ / ‖b*_j‖².
            let dot = dot_ir(&b[i], &bstar[j]);
            mu[i][j] = Rational::div(&dot, &norm[j]);
            for t in 0..dim {
                bi[t] = Rational::sub(&bi[t], &Rational::mul(&mu[i][j], &bstar[j][t]));
            }
        }
        norm[i] = dot_rr(&bi, &bi);
        if norm[i].numerator().is_zero() {
            return None; // dependent vector
        }
        bstar.push(bi);
    }
    Some((mu, norm))
}

/// Size-reduction step: if `|μ_{k,l}| > 1/2`, subtract `round(μ_{k,l})·b_l` from
/// `b_k` and update the affected `μ` entries.
fn size_reduce(b: &mut [Vec<Int>], mu: &mut [Vec<Rational>], k: usize, l: usize, half: &Rational) {
    if mu[k][l].abs().cmp(half) != Ordering::Greater {
        return;
    }
    let q = round_to_int(&mu[k][l]);
    if q.is_zero() {
        return;
    }
    let dim = b[k].len();
    let bl = b[l].clone();
    for t in 0..dim {
        b[k][t] = b[k][t].sub(&q.mul(&bl[t]));
    }
    let qr = Rational::from_integer(q);
    let mul = mu[l].clone();
    for j in 0..l {
        mu[k][j] = Rational::sub(&mu[k][j], &Rational::mul(&qr, &mul[j]));
    }
    mu[k][l] = Rational::sub(&mu[k][l], &qr);
}

/// Swaps `b_k` and `b_{k-1}` and updates the Gram–Schmidt data in place, per the
/// standard LLL swap recurrences.
#[allow(clippy::needless_range_loop)] // the index drives two distinct μ rows together
fn swap_step(
    b: &mut [Vec<Int>],
    mu: &mut [Vec<Rational>],
    norm: &mut [Rational],
    k: usize,
    n: usize,
) {
    let mu_old = mu[k][k - 1].clone();
    b.swap(k, k - 1);
    for j in 0..k - 1 {
        let tmp = mu[k][j].clone();
        mu[k][j] = mu[k - 1][j].clone();
        mu[k - 1][j] = tmp;
    }
    // New norms: B' = ‖b*_k‖² + μ²·‖b*_{k-1}‖².
    let bnew = Rational::add(
        &norm[k],
        &Rational::mul(&Rational::mul(&mu_old, &mu_old), &norm[k - 1]),
    );
    // New μ_{k,k-1} = μ_old·‖b*_{k-1}‖² / B'.
    mu[k][k - 1] = Rational::div(&Rational::mul(&mu_old, &norm[k - 1]), &bnew);
    norm[k] = Rational::div(&Rational::mul(&norm[k - 1], &norm[k]), &bnew);
    norm[k - 1] = bnew;

    let mk = mu[k][k - 1].clone();
    for i in k + 1..n {
        let t = mu[i][k].clone();
        mu[i][k] = Rational::sub(&mu[i][k - 1], &Rational::mul(&mu_old, &t));
        mu[i][k - 1] = Rational::add(&t, &Rational::mul(&mk, &mu[i][k]));
    }
}

/// Nearest integer to a rational (ties round up), via `⌊(2p + q) / 2q⌋` with the
/// denominator `q > 0` (guaranteed by `Rational` normalization).
fn round_to_int(r: &Rational) -> Int {
    let two = Int::from_i64(2);
    let num2 = r.numerator().mul(&two);
    let den = r.denominator();
    num2.add(den).div_floor(&den.mul(&two))
}

/// Dot product of an integer vector with a rational vector.
fn dot_ir(a: &[Int], b: &[Rational]) -> Rational {
    let mut acc = Rational::ZERO;
    for (x, y) in a.iter().zip(b) {
        acc.addmul(&Rational::from_integer(x.clone()), y);
    }
    acc
}

/// Dot product of two rational vectors.
fn dot_rr(a: &[Rational], b: &[Rational]) -> Rational {
    let mut acc = Rational::ZERO;
    for (x, y) in a.iter().zip(b) {
        acc.addmul(x, y);
    }
    acc
}

#[cfg(feature = "float")]
pub use relations::{find_integer_relation, minimal_polynomial};

/// Integer-relation detection and minimal-polynomial recovery, built on LLL.
#[cfg(feature = "float")]
mod relations {
    use super::{Int, Rational, Vec, lll_reduce, round_to_int};
    use crate::float::{Float, RoundingMode};

    /// Searches for a small nonzero **integer relation** among the real numbers
    /// `xs`: a vector of integers `a`, not all zero, with `Σ aᵢ·xᵢ ≈ 0`. Returns
    /// `None` if no relation is found at the requested precision (e.g. the values
    /// are rationally independent, like `[1, π]`).
    ///
    /// `scale_bits` controls detection precision: the search weights the values by
    /// `2^scale_bits`, so it should be comfortably below the accuracy of the input
    /// `Float`s (a common choice is ~70–90% of their precision). Larger values
    /// detect finer relations but demand more accurate inputs and reject spurious
    /// ones more strictly. The returned relation is sign-normalized (first nonzero
    /// entry positive) and, for a genuine relation, its entries are the small
    /// integers you'd expect (e.g. `[2, -1]` for `[ln 2, ln 4]`).
    pub fn find_integer_relation(xs: &[Float], scale_bits: u64) -> Option<Vec<Int>> {
        let n = xs.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return xs[0].is_zero().then(|| alloc::vec![Int::ONE]);
        }
        // Lattice rows: row i = eᵢ (weight 1, keeps coefficients small) augmented
        // with round(2^scale · xᵢ) (weight 2^scale, drives the combination to 0).
        let pow2 = Rational::from_integer(Int::ONE.mul_2k(scale_bits as u32));
        let mut basis: Vec<Vec<Int>> = Vec::with_capacity(n);
        for (i, x) in xs.iter().enumerate() {
            let r = x.to_rational()?; // non-finite input → give up
            let mut row = alloc::vec![Int::ZERO; n + 1];
            row[i] = Int::ONE;
            row[n] = round_to_int(&Rational::mul(&r, &pow2));
            basis.push(row);
        }
        let reduced = lll_reduce(&basis);

        // The shortest reduced vector's identity part is the candidate relation.
        // For a lattice of determinant ~2^scale in n dimensions, a *generic*
        // (relation-free) shortest vector has entries ~2^(scale/n). A genuine
        // relation makes the shortest vector far shorter — small coefficients and a
        // near-zero last coordinate (2^scale·Σaᵢxᵢ up to rounding). So accept only
        // when every entry, including the last, sits well below that generic bound.
        let short = &reduced[0];
        let cand = &short[..n];
        if cand.iter().all(Int::is_zero) {
            return None;
        }
        let threshold_bits = scale_bits / (2 * n as u64);
        if short
            .iter()
            .any(|e| u64::from(e.bit_len()) > threshold_bits)
        {
            return None;
        }
        Some(normalize_sign(cand.to_vec()))
    }

    /// Recovers the **minimal polynomial** of the real algebraic number
    /// approximated by `alpha`, as integer coefficients `[a₀, a₁, …, a_d]` (lowest
    /// degree first, so `a₀ + a₁·α + … + a_d·αᵈ = 0`), searching degrees
    /// `1..=max_degree`. Returns the least-degree relation found, or `None`.
    ///
    /// `alpha` must be accurate to well beyond `scale_bits` bits for the answer to
    /// be trustworthy; recovered polynomials should be checked by the caller (e.g.
    /// evaluate at a sharper approximation, or confirm irreducibility).
    pub fn minimal_polynomial(
        alpha: &Float,
        max_degree: usize,
        scale_bits: u64,
    ) -> Option<Vec<Int>> {
        let prec = alpha.precision();
        let m = RoundingMode::Nearest;
        // Powers 1, α, α², …, α^max_degree.
        let mut powers = Vec::with_capacity(max_degree + 1);
        powers.push(Float::from_int(&Int::ONE, prec, m));
        for _ in 1..=max_degree {
            powers.push(powers.last().unwrap().mul(alpha, prec, m));
        }
        (1..=max_degree).find_map(|d| find_integer_relation(&powers[..=d], scale_bits))
    }

    /// Canonicalizes a relation so its first nonzero entry is positive.
    fn normalize_sign(mut v: Vec<Int>) -> Vec<Int> {
        if let Some(first) = v.iter().find(|c| !c.is_zero()) {
            if first.is_negative() {
                for c in &mut v {
                    *c = c.neg();
                }
            }
        }
        v
    }
}
