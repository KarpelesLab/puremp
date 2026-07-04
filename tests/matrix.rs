//! Tests for generic matrices and exact linear algebra.
#![cfg(all(feature = "matrix", feature = "rational"))]

use puremp::{Int, Matrix, Rational};

fn mi(rows: usize, cols: usize, d: &[i64]) -> Matrix<Int> {
    Matrix::new(rows, cols, d.iter().map(|&x| Int::from(x)).collect())
}
fn mr(rows: usize, cols: usize, d: &[i64]) -> Matrix<Rational> {
    Matrix::new(rows, cols, d.iter().map(|&x| Rational::from(x)).collect())
}

#[test]
fn ring_operations() {
    let a = mi(2, 2, &[1, 2, 3, 4]);
    let b = mi(2, 2, &[5, 6, 7, 8]);
    // product [[19,22],[43,50]]
    assert_eq!(&a * &b, mi(2, 2, &[19, 22, 43, 50]));
    assert_eq!(&a + &b, mi(2, 2, &[6, 8, 10, 12]));
    assert_eq!(a.transpose(), mi(2, 2, &[1, 3, 2, 4]));
    // identity is a mul unit
    assert_eq!(&a * &Matrix::<Int>::identity(2), a);
}

#[test]
fn integer_determinant_bareiss() {
    assert_eq!(mi(2, 2, &[1, 2, 3, 4]).determinant().to_string(), "-2");
    assert_eq!(
        mi(3, 3, &[6, 1, 1, 4, -2, 5, 2, 8, 7])
            .determinant()
            .to_string(),
        "-306"
    );
    // singular
    assert_eq!(
        mi(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9])
            .determinant()
            .to_string(),
        "0"
    );
    // identity
    assert_eq!(Matrix::<Int>::identity(5).determinant().to_string(), "1");
}

#[test]
fn rational_inverse_solve_rank() {
    let a = mr(2, 2, &[4, 7, 2, 6]);
    let inv = a.inverse().unwrap();
    // A * A^-1 == I
    assert_eq!(&a * &inv, Matrix::<Rational>::identity(2));
    assert_eq!(a.determinant().to_string(), "10");

    // Solve [[2,1],[1,3]] x = [1,2] -> x = [1/5, 3/5]
    let m = mr(2, 2, &[2, 1, 1, 3]);
    let x = m.solve(&[Rational::from(1), Rational::from(2)]).unwrap();
    assert_eq!(x[0].to_string(), "1/5");
    assert_eq!(x[1].to_string(), "3/5");

    // singular matrix: no inverse, rank 1
    let s = mr(2, 2, &[1, 2, 2, 4]);
    assert!(s.inverse().is_none());
    assert_eq!(s.rank(), 1);
    assert_eq!(mr(3, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1]).rank(), 3);
}

#[test]
fn fraction_free_inverse_solve_with_pivots() {
    // Zero leading pivot forces the fraction-free path to bail to the exact
    // rational fallback — result must still be correct.
    let m = mr(3, 3, &[0, 1, 2, 1, 0, 3, 4, 5, 0]);
    let inv = m.inverse().expect("nonsingular");
    assert_eq!(&m * &inv, Matrix::<Rational>::identity(3));
    let x = m
        .solve(&[Rational::from(1), Rational::from(2), Rational::from(3)])
        .unwrap();
    // A·x == b
    for i in 0..3 {
        let mut acc = Rational::ZERO;
        for j in 0..3 {
            acc = acc.add(&m.get(i, j).mul(&x[j]));
        }
        assert_eq!(acc, Rational::from((i + 1) as i64));
    }
    // Fractional entries + a zero pivot.
    let mf = mr(2, 2, &[0, 3, 2, 5]).scalar_mul(&Rational::new(1.into(), 3.into()));
    assert_eq!(
        &mf * &mf.inverse().unwrap(),
        Matrix::<Rational>::identity(2)
    );
    // Singular matrix still detected.
    assert!(mr(2, 2, &[1, 2, 2, 4]).inverse().is_none());
    assert!(mr(2, 2, &[0, 0, 1, 2]).inverse().is_none());
}
