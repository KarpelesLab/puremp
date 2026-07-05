//! Exact eigenvalues of rational matrices.
//!
//! These tests exercise `Matrix<Rational>::characteristic_polynomial`,
//! `real_eigenvalues`, and `real_eigenvalues_with_multiplicity` against matrices
//! with *known* spectra:
//!
//! - diagonal / triangular matrices (spectrum = diagonal),
//! - small symmetric matrices with integer eigenvalues,
//! - a matrix with irrational eigenvalues `±√2`,
//! - a companion matrix whose eigenvalues are a known polynomial's roots,
//! - a repeated eigenvalue (algebraic multiplicity), and
//! - a rotation with a purely complex spectrum (no real eigenvalues → empty).
//!
//! Each eigenvalue `λ` is verified *exactly* (charpoly(λ) = 0, evaluated with
//! `Algebraic` arithmetic) and *numerically* (det(A − λ·I) ≈ 0 via `to_float`).

#![cfg(feature = "algebraic")]

use puremp::{Algebraic, FieldMatrix, Float, Int, Matrix, Rational, RoundingMode};

const N: RoundingMode = RoundingMode::Nearest;
const PREC: u64 = 128;

/// Builds a rational matrix from integer rows.
fn mat(rows: &[&[i64]]) -> Matrix<Rational> {
    let data: Vec<Vec<Rational>> = rows
        .iter()
        .map(|r| r.iter().map(|&x| Rational::from(x)).collect())
        .collect();
    Matrix::from_rows(data)
}

/// Evaluates the (rational-coefficient) characteristic polynomial at an algebraic
/// point via Horner, returning an exact `Algebraic`.
fn charpoly_at(m: &Matrix<Rational>, x: &Algebraic) -> Algebraic {
    let cp = m.characteristic_polynomial();
    let coeffs = cp.coeffs();
    let mut acc = Algebraic::from_int(Int::from_i64(0));
    for c in coeffs.iter().rev() {
        acc = acc.mul(x).add(&Algebraic::from_rational(c.clone()));
    }
    acc
}

/// `|det(A − λ·I)|` as an `f64`, computed over `Float` (numeric sanity check).
fn det_shifted_f64(m: &Matrix<Rational>, lambda: &Algebraic) -> f64 {
    let n = m.rows();
    let lf = lambda.to_float(PREC, N);
    let zero = Float::from_int(&Int::from_i64(0), PREC, N);
    let mut b = Matrix::<Float>::filled(zero, n, n);
    for i in 0..n {
        for j in 0..n {
            let mut e = Float::from_rational(m.get(i, j), PREC, N);
            if i == j {
                e = e.sub(&lf, PREC, N);
            }
            b.set(i, j, e);
        }
    }
    FieldMatrix::determinant(&b).abs().to_f64()
}

/// Asserts every returned eigenvalue exactly annihilates the char poly and is a
/// numeric eigenvalue.
fn assert_eigen(m: &Matrix<Rational>, evs: &[Algebraic]) {
    for lambda in evs {
        assert_eq!(
            charpoly_at(m, lambda).signum(),
            0,
            "charpoly({lambda}) must be exactly zero"
        );
        assert!(
            det_shifted_f64(m, lambda) < 1e-20,
            "det(A − λ·I) must be ≈ 0 for λ = {lambda}"
        );
    }
}

fn alg_ints(vals: &[i64]) -> Vec<Algebraic> {
    vals.iter()
        .map(|&v| Algebraic::from_int(Int::from_i64(v)))
        .collect()
}

#[test]
fn diagonal_matrix() {
    let m = mat(&[&[2, 0, 0], &[0, 3, 0], &[0, 0, 5]]);
    let evs = m.real_eigenvalues();
    assert_eq!(evs, alg_ints(&[2, 3, 5]));
    assert_eigen(&m, &evs);
}

#[test]
fn triangular_matrix() {
    // Upper triangular ⇒ eigenvalues are the diagonal {1, 4, 6}.
    let m = mat(&[&[1, 2, 3], &[0, 4, 5], &[0, 0, 6]]);
    let evs = m.real_eigenvalues();
    assert_eq!(evs, alg_ints(&[1, 4, 6]));
    assert_eigen(&m, &evs);
}

#[test]
fn symmetric_2x2() {
    // [[2,1],[1,2]] → {1, 3}.
    let m = mat(&[&[2, 1], &[1, 2]]);
    let evs = m.real_eigenvalues();
    assert_eq!(evs, alg_ints(&[1, 3]));
    assert_eigen(&m, &evs);

    // [[0,1],[1,0]] → {−1, 1}.
    let m = mat(&[&[0, 1], &[1, 0]]);
    let evs = m.real_eigenvalues();
    assert_eq!(evs, alg_ints(&[-1, 1]));
    assert_eigen(&m, &evs);
}

#[test]
fn irrational_eigenvalues_sqrt2() {
    // [[0,2],[1,0]] → characteristic polynomial x² − 2 → eigenvalues ±√2.
    let m = mat(&[&[0, 2], &[1, 0]]);
    let cp = m.characteristic_polynomial();
    // x² − 2, low-to-high: [-2, 0, 1].
    assert_eq!(
        cp.coeffs().to_vec(),
        vec![Rational::from(-2), Rational::from(0), Rational::from(1)]
    );

    let evs = m.real_eigenvalues();
    assert_eq!(evs.len(), 2);

    // √2 as an independent Algebraic (positive root of x² − 2).
    let sqrt2 = Algebraic::from_int(Int::from_i64(2)).sqrt();
    assert_eq!(evs[0], sqrt2.clone().neg()); // −√2
    assert_eq!(evs[1], sqrt2); // +√2

    // Each returned eigenvalue squares to 2.
    for lambda in &evs {
        assert_eq!(
            lambda.mul(lambda),
            Algebraic::from_int(Int::from_i64(2)),
            "λ² must equal 2"
        );
    }
    assert_eigen(&m, &evs);
}

#[test]
fn companion_matrix_known_roots() {
    // Companion matrix of x³ − 6x² + 11x − 6 = (x−1)(x−2)(x−3), using the same
    // sub-diagonal-ones layout as the crate's internal companion(): last column
    // holds −cᵢ, sub-diagonal is ones.
    // c₀ = −6, c₁ = 11, c₂ = −6  ⇒  last column = [6, −11, 6].
    let m = mat(&[&[0, 0, 6], &[1, 0, -11], &[0, 1, 6]]);
    let cp = m.characteristic_polynomial();
    assert_eq!(
        cp.coeffs().to_vec(),
        vec![
            Rational::from(-6),
            Rational::from(11),
            Rational::from(-6),
            Rational::from(1)
        ]
    );
    let evs = m.real_eigenvalues();
    assert_eq!(evs, alg_ints(&[1, 2, 3]));
    assert_eigen(&m, &evs);
}

#[test]
fn repeated_eigenvalue_multiplicity() {
    // Jordan block [[2,1],[0,2]] → single eigenvalue 2 with algebraic mult 2.
    let m = mat(&[&[2, 1], &[0, 2]]);
    let evs = m.real_eigenvalues();
    assert_eq!(evs, alg_ints(&[2])); // distinct → 2 appears once
    assert_eigen(&m, &evs);

    let with_mult = m.real_eigenvalues_with_multiplicity();
    assert_eq!(with_mult.len(), 1);
    assert_eq!(with_mult[0].0, Algebraic::from_int(Int::from_i64(2)));
    assert_eq!(with_mult[0].1, 2);

    // A larger example: diag(2,2,5) → 2 (mult 2), 5 (mult 1).
    let m = mat(&[&[2, 0, 0], &[0, 2, 0], &[0, 0, 5]]);
    let with_mult = m.real_eigenvalues_with_multiplicity();
    assert_eq!(with_mult.len(), 2);
    assert_eq!(with_mult[0].0, Algebraic::from_int(Int::from_i64(2)));
    assert_eq!(with_mult[0].1, 2);
    assert_eq!(with_mult[1].0, Algebraic::from_int(Int::from_i64(5)));
    assert_eq!(with_mult[1].1, 1);
}

#[test]
fn complex_pair_returns_no_real_eigenvalues() {
    // Rotation by 90°: [[0,−1],[1,0]] has eigenvalues ±i (purely complex).
    let m = mat(&[&[0, -1], &[1, 0]]);
    let cp = m.characteristic_polynomial();
    // x² + 1, low-to-high: [1, 0, 1].
    assert_eq!(
        cp.coeffs().to_vec(),
        vec![Rational::from(1), Rational::from(0), Rational::from(1)]
    );
    // No real eigenvalues — empty, no panic.
    assert!(m.real_eigenvalues().is_empty());
    assert!(m.real_eigenvalues_with_multiplicity().is_empty());
}

#[test]
fn mixed_real_and_complex_spectrum() {
    // Block-diagonal: a real eigenvalue 7 plus a complex pair (rotation block).
    // [[7,0,0],[0,0,-1],[0,1,0]] → real spectrum {7}, complex ±i dropped.
    let m = mat(&[&[7, 0, 0], &[0, 0, -1], &[0, 1, 0]]);
    let evs = m.real_eigenvalues();
    assert_eq!(evs, alg_ints(&[7]));
    assert_eigen(&m, &evs);
    let with_mult = m.real_eigenvalues_with_multiplicity();
    assert_eq!(with_mult.len(), 1);
    assert_eq!(with_mult[0].1, 1);
}
