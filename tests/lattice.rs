#![cfg(all(feature = "lattice", feature = "matrix"))]
#![allow(clippy::needless_range_loop)] // explicit indices track parallel vectors/matrices
//! LLL reduction: verify the output is genuinely reduced and spans the same
//! lattice, using an independent Gram–Schmidt and a unimodularity check.

use puremp::lattice::{lll_reduce, lll_reduce_delta};
use puremp::{Int, Matrix, Rational};

fn iv(v: &[i64]) -> Vec<Int> {
    v.iter().map(|&x| Int::from_i64(x)).collect()
}

fn half() -> Rational {
    Rational::new(Int::from_i64(1), Int::from_i64(2))
}

/// Independent exact Gram–Schmidt (does not share code with the implementation).
fn gram_schmidt(b: &[Vec<Int>]) -> (Vec<Vec<Rational>>, Vec<Rational>) {
    let n = b.len();
    let dim = b[0].len();
    let mut mu = vec![vec![Rational::ZERO; n]; n];
    let mut norm = vec![Rational::ZERO; n];
    let mut bstar: Vec<Vec<Rational>> = Vec::new();
    for i in 0..n {
        let mut bi: Vec<Rational> = b[i]
            .iter()
            .map(|x| Rational::from_integer(x.clone()))
            .collect();
        for j in 0..i {
            let mut dot = Rational::ZERO;
            for t in 0..dim {
                dot = dot.add(&Rational::from_integer(b[i][t].clone()).mul(&bstar[j][t]));
            }
            mu[i][j] = dot.div(&norm[j]);
            for t in 0..dim {
                bi[t] = bi[t].sub(&mu[i][j].mul(&bstar[j][t]));
            }
        }
        let mut nn = Rational::ZERO;
        for t in 0..dim {
            nn = nn.add(&bi[t].mul(&bi[t]));
        }
        norm[i] = nn;
        bstar.push(bi);
    }
    (mu, norm)
}

fn assert_reduced(b: &[Vec<Int>], delta: &Rational) {
    let (mu, norm) = gram_schmidt(b);
    for i in 0..b.len() {
        for j in 0..i {
            assert!(
                mu[i][j].abs() <= half(),
                "not size-reduced at ({i},{j}): {}",
                mu[i][j]
            );
        }
    }
    for k in 1..b.len() {
        let m = &mu[k][k - 1];
        let bound = delta.sub(&m.mul(m)).mul(&norm[k - 1]);
        assert!(norm[k] >= bound, "Lovász condition fails at k={k}");
    }
}

/// Squared lattice volume det(B·Bᵀ) = ∏‖b*ᵢ‖², invariant under unimodular change.
fn gram_det(b: &[Vec<Int>]) -> Rational {
    let (_, norm) = gram_schmidt(b);
    norm.iter().fold(Rational::ONE, |a, x| a.mul(x))
}

/// Square case: output = U·input with U integer and det U = ±1.
fn assert_same_lattice_square(input: &[Vec<Int>], output: &[Vec<Int>]) {
    let n = input.len();
    let to_m = |rows: &[Vec<Int>]| {
        let data: Vec<Rational> = rows
            .iter()
            .flatten()
            .map(|x| Rational::from_integer(x.clone()))
            .collect();
        Matrix::new(n, n, data)
    };
    let inv = to_m(input).inverse().expect("independent basis");
    let u = to_m(output).mul(&inv);
    for i in 0..n {
        for j in 0..n {
            assert!(
                u.get(i, j).denominator() == &Int::ONE,
                "U not integer at ({i},{j})"
            );
        }
    }
    let d = u.determinant();
    assert!(
        d == Rational::ONE || d == Rational::from_integer(Int::from_i64(-1)),
        "det U = {d}, expected ±1"
    );
}

#[test]
fn reduced_and_same_lattice_random() {
    let delta = Rational::new(Int::from_i64(3), Int::from_i64(4));
    let mut seed = 0x1EE7_C0DEu64;
    let next = |s: &mut u64| {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*s >> 40) as i64 % 21) - 10
    };
    for n in 2..=5usize {
        for _ in 0..40 {
            let rows: Vec<Vec<Int>> = (0..n)
                .map(|_| iv(&(0..n).map(|_| next(&mut seed)).collect::<Vec<_>>()))
                .collect();
            // Skip singular bases (not a lattice basis) — detected without the
            // unguarded Gram–Schmidt.
            let data: Vec<Rational> = rows
                .iter()
                .flatten()
                .map(|x| Rational::from_integer(x.clone()))
                .collect();
            if Matrix::new(n, n, data).determinant() == Rational::ZERO {
                continue;
            }
            let gm = gram_det(&rows);
            let red = lll_reduce_delta(&rows, &delta);
            assert_reduced(&red, &delta);
            assert_same_lattice_square(&rows, &red);
            assert_eq!(gram_det(&red), gm, "lattice volume changed (n={n})");
        }
    }
}

#[test]
fn finds_short_vectors_in_bad_basis() {
    // A skewed basis of Z²: [[1, 1000000], [0, 1]] contains the short vector
    // (1,0) via b0 - 1000000·b1. LLL must surface something short.
    let bad = vec![iv(&[1, 1_000_000]), iv(&[0, 1])];
    let red = lll_reduce(&bad);
    assert_reduced(&red, &Rational::new(Int::from_i64(3), Int::from_i64(4)));
    assert_same_lattice_square(&bad, &red);
    // Shortest vector of this lattice is (1,0); the first reduced vector matches
    // it up to sign/order.
    let norms: Vec<Int> = red
        .iter()
        .map(|v| v.iter().fold(Int::ZERO, |a, x| a.add(&x.mul(x))))
        .collect();
    assert!(
        norms.iter().any(|nrm| nrm == &Int::ONE),
        "no unit-length vector found: {norms:?}"
    );
}

#[test]
fn classic_example_is_reduced() {
    // A 3×3 lattice with a deliberately poor basis.
    let basis = vec![iv(&[1, 1, 1]), iv(&[-1, 0, 2]), iv(&[3, 5, 6])];
    let delta = Rational::new(Int::from_i64(3), Int::from_i64(4));
    let red = lll_reduce_delta(&basis, &delta);
    assert_reduced(&red, &delta);
    assert_same_lattice_square(&basis, &red);
}

#[test]
fn edge_cases() {
    // Single vector: returned unchanged.
    assert_eq!(lll_reduce(&[iv(&[3, 4])]), vec![iv(&[3, 4])]);
    // Already reduced (identity) stays a valid reduced basis.
    let id = vec![iv(&[1, 0]), iv(&[0, 1])];
    let red = lll_reduce(&id);
    assert_reduced(&red, &Rational::new(Int::from_i64(3), Int::from_i64(4)));
    // Linearly dependent input: returned unchanged (cannot reduce).
    let dep = vec![iv(&[2, 4]), iv(&[1, 2])];
    assert_eq!(lll_reduce(&dep), dep);
}
