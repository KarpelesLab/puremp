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
pub use relations::{find_integer_relation, minimal_polynomial, pslq, pslq_with};

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

    /// Searches for a small nonzero **integer relation** among the real numbers
    /// `xs` using the **PSLQ** algorithm: an integer vector `a`, not all zero,
    /// with `Σ aᵢ·xᵢ ≈ 0`. Returns `None` if no relation is found within the
    /// precision/iteration budget (e.g. the values are rationally independent,
    /// like `[1, √2, π]`).
    ///
    /// This is the one-level PSLQ of Ferguson, Bailey & Arno (*Analysis of the
    /// PSLQ Integer Relation Algorithm*, Math. Comp. 68 (1999)), using the
    /// standard reduction parameter `γ = 2/√3`. The internal linear algebra runs
    /// in [`Float`] at a working precision above the input's noise floor, while
    /// the change-of-basis matrices — and hence the returned relation — are kept
    /// as exact [`Int`]s.
    ///
    /// The `xs` must all be finite and computed to `precision` bits. A relation
    /// is accepted once the smallest transformed coordinate falls below
    /// `2^-(precision − precision/4)`; see [`pslq_with`] for full control over the
    /// iteration cap and detection threshold. The returned relation is
    /// sign-normalized (first nonzero entry positive).
    ///
    /// # Examples
    ///
    /// ```
    /// use puremp::{Float, Int, RoundingMode};
    /// use puremp::lattice::pslq;
    ///
    /// let p = 256;
    /// let m = RoundingMode::Nearest;
    /// // φ = (1+√5)/2 satisfies φ² − φ − 1 = 0, so {1, φ, φ²} has relation (−1,−1,1).
    /// let phi = Float::from_int(&Int::from(5), p, m)
    ///     .sqrt(p, m)
    ///     .add(&Float::from_int(&Int::ONE, p, m), p, m)
    ///     .div(&Float::from_int(&Int::from(2), p, m), p, m);
    /// let xs = [
    ///     Float::from_int(&Int::ONE, p, m),
    ///     phi.clone(),
    ///     phi.mul(&phi, p, m),
    /// ];
    /// // Sign-normalized (first nonzero entry positive): −φ²+φ+1 = 0.
    /// let rel = pslq(&xs, p).unwrap();
    /// assert_eq!(rel, [Int::from(1), Int::from(1), Int::from(-1)]);
    /// ```
    pub fn pslq(xs: &[Float], precision: u64) -> Option<Vec<Int>> {
        let n = xs.len();
        let detect_bits = precision - precision / 4;
        // O(n²·precision) iterations comfortably bounds the tiny relations this
        // targets while keeping the "no relation" case terminating quickly.
        let max_iters = (precision as usize + 20) * n.max(1) * n.max(1);
        pslq_with(xs, precision, max_iters, detect_bits)
    }

    /// [`pslq`] with explicit control over the search budget.
    ///
    /// `max_iters` caps the number of PSLQ iterations (returning `None` on
    /// exhaustion), and `detect_bits` sets the detection threshold: a relation is
    /// reported once the smallest transformed coordinate drops below
    /// `2^-detect_bits`. `detect_bits` should sit comfortably below `precision`
    /// (the input's accuracy) yet well above the size of any spurious short
    /// combination — `precision − precision/4` is the default and works for
    /// small-coefficient relations.
    pub fn pslq_with(
        xs: &[Float],
        precision: u64,
        max_iters: usize,
        detect_bits: u64,
    ) -> Option<Vec<Int>> {
        let n = xs.len();
        if n == 0 {
            return None;
        }
        if xs.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if n == 1 {
            return xs[0].is_zero().then(|| alloc::vec![Int::ONE]);
        }

        let m = RoundingMode::Nearest;
        // Work above the input's noise floor so the linear-algebra rounding does
        // not dominate the ~2^-precision residual of a genuine relation.
        let wp = precision + 64;
        let x: Vec<Float> = xs.iter().map(|v| v.round(wp, m)).collect();

        // Partial norms s_k = ‖(x_k, …, x_{n-1})‖.
        let mut s = alloc::vec![Float::zero(wp); n];
        let mut acc = Float::zero(wp);
        for k in (0..n).rev() {
            acc = acc.add(&x[k].mul(&x[k], wp, m), wp, m);
            s[k] = acc.sqrt(wp, m);
        }
        if s[0].is_zero() {
            // All inputs are zero: e₀ is a relation.
            let mut a = alloc::vec![Int::ZERO; n];
            a[0] = Int::ONE;
            return Some(a);
        }

        // Normalize so ‖y‖ = 1 (t = 1/s₀); y stays proportional to x·B throughout.
        let one = Float::from_int(&Int::ONE, wp, m);
        let t0 = one.div(&s[0], wp, m);
        let mut y: Vec<Float> = x.iter().map(|xi| t0.mul(xi, wp, m)).collect();
        for sk in &mut s {
            *sk = t0.mul(sk, wp, m);
        }

        // Initial lower-trapezoidal H (n×(n−1)).
        let mut h = alloc::vec![alloc::vec![Float::zero(wp); n - 1]; n];
        for j in 0..n - 1 {
            h[j][j] = s[j + 1].div(&s[j], wp, m);
            let sj = s[j].mul(&s[j + 1], wp, m);
            for i in j + 1..n {
                h[i][j] = y[i].mul(&y[j], wp, m).div(&sj, wp, m).neg();
            }
        }

        // Exact change-of-basis matrices (A = B⁻¹), both starting as identity.
        let mut a_mat = alloc::vec![alloc::vec![Int::ZERO; n]; n];
        let mut b_mat = alloc::vec![alloc::vec![Int::ZERO; n]; n];
        for i in 0..n {
            a_mat[i][i] = Int::ONE;
            b_mat[i][i] = Int::ONE;
        }

        // γ = 2/√3 and its powers γ^(k+1) used in the row-selection weight.
        let gamma = Float::from_int(&Int::from_i64(2), wp, m).div(
            &Float::from_int(&Int::from_i64(3), wp, m).sqrt(wp, m),
            wp,
            m,
        );
        let mut gpow = Vec::with_capacity(n - 1);
        let mut g = gamma.clone();
        for _ in 0..n - 1 {
            gpow.push(g.clone());
            g = g.mul(&gamma, wp, m);
        }

        // Any spurious near-zero combination reaching ε has coefficients larger
        // than ~2^(detect_bits/(n−1)); a genuine small relation stays far below
        // this, so reject candidates whose entries exceed half that exponent.
        let coeff_bits = detect_bits / (2 * (n as u64 - 1));

        // Detection threshold ε = 2^-detect_bits.
        let eps = Float::from_rational(
            &Rational::new(
                Int::ONE,
                Int::ONE.mul_2k(detect_bits.min(u32::MAX as u64) as u32),
            ),
            wp,
            m,
        );

        hermite_reduce(n, wp, m, &mut y, &mut h, &mut a_mat, &mut b_mat);

        for _ in 0..max_iters {
            // Select the row maximizing γ^(k+1)·|H_kk|.
            let mut sel = 0;
            let mut best = gpow[0].mul(&h[0][0].abs(), wp, m);
            for k in 1..n - 1 {
                let v = gpow[k].mul(&h[k][k].abs(), wp, m);
                if v > best {
                    best = v;
                    sel = k;
                }
            }

            // Swap rows sel, sel+1 of y/H/A and the matching columns of B.
            y.swap(sel, sel + 1);
            h.swap(sel, sel + 1);
            a_mat.swap(sel, sel + 1);
            for row in &mut b_mat {
                row.swap(sel, sel + 1);
            }

            // Corner (Givens) update to restore the trapezoidal shape of H.
            if sel < n - 2 {
                let hmm = h[sel][sel].clone();
                let hmm1 = h[sel][sel + 1].clone();
                let t = hmm
                    .mul(&hmm, wp, m)
                    .add(&hmm1.mul(&hmm1, wp, m), wp, m)
                    .sqrt(wp, m);
                let t1 = hmm.div(&t, wp, m);
                let t2 = hmm1.div(&t, wp, m);
                for row in h.iter_mut().take(n).skip(sel) {
                    let t3 = row[sel].clone();
                    let t4 = row[sel + 1].clone();
                    row[sel] = t1.mul(&t3, wp, m).add(&t2.mul(&t4, wp, m), wp, m);
                    row[sel + 1] = t1.mul(&t4, wp, m).sub(&t2.mul(&t3, wp, m), wp, m);
                }
            }

            hermite_reduce(n, wp, m, &mut y, &mut h, &mut a_mat, &mut b_mat);

            // A near-zero coordinate y_k means column k of B is the relation.
            let mut kmin = 0;
            let mut ymin = y[0].abs();
            for (k, yk) in y.iter().enumerate().skip(1) {
                let ak = yk.abs();
                if ak < ymin {
                    ymin = ak;
                    kmin = k;
                }
            }
            if ymin < eps {
                let rel: Vec<Int> = (0..n).map(|j| b_mat[j][kmin].clone()).collect();
                // A genuine relation drives the residual to the input's noise
                // floor with *small* coefficients. Any near-zero combination of
                // rationally independent reals instead needs huge coefficients
                // (|a·x| ≳ ‖a‖^-(n-1) by Dirichlet), so an oversized candidate
                // means no small relation is certifiable at this precision.
                if rel.iter().any(|c| u64::from(c.bit_len()) > coeff_bits) {
                    return None;
                }
                if rel.iter().any(|c| !c.is_zero()) {
                    return Some(normalize_sign(rel));
                }
            }
        }
        None
    }

    /// One Hermite (size-) reduction sweep of PSLQ: makes `|H_ij| ≤ ½|H_jj|` for
    /// `j < i`, applying the matching integer updates to `y`, `A` and `B`.
    #[allow(clippy::too_many_arguments)]
    fn hermite_reduce(
        n: usize,
        wp: u64,
        m: RoundingMode,
        y: &mut [Float],
        h: &mut [Vec<Float>],
        a: &mut [Vec<Int>],
        b: &mut [Vec<Int>],
    ) {
        for i in 1..n {
            for j in (0..i).rev() {
                let q = match h[i][j].div(&h[j][j], wp, m).round_to_int() {
                    Some(q) if !q.is_zero() => q,
                    _ => continue,
                };
                let qf = Float::from_int(&q, wp, m);
                // y_j += q·y_i.
                let dy = qf.mul(&y[i], wp, m);
                y[j] = y[j].add(&dy, wp, m);
                // H_ik -= q·H_jk for k ≤ j.
                let hj = h[j].clone();
                for k in 0..=j {
                    let d = qf.mul(&hj[k], wp, m);
                    h[i][k] = h[i][k].sub(&d, wp, m);
                }
                // A_ik -= q·A_jk.
                let aj = a[j].clone();
                for k in 0..n {
                    a[i][k] = a[i][k].sub(&q.mul(&aj[k]));
                }
                // B_kj += q·B_ki.
                for row in b.iter_mut() {
                    let bki = row[i].clone();
                    row[j] = row[j].add(&q.mul(&bki));
                }
            }
        }
    }

    /// Canonicalizes a relation so its first nonzero entry is positive.
    fn normalize_sign(mut v: Vec<Int>) -> Vec<Int> {
        if let Some(first) = v.iter().find(|c| !c.is_zero())
            && first.is_negative()
        {
            for c in &mut v {
                *c = c.neg();
            }
        }
        v
    }
}
