#![cfg(all(feature = "matrix", feature = "int"))]
//! Division-free determinant & characteristic polynomial (Samuelson–Berkowitz)
//! over commutative rings — including non-field rings where Bareiss/Gaussian
//! cannot be used.

use puremp::{Int, Matrix, RingMatrix};

fn im(rows: &[&[i64]]) -> Matrix<Int> {
    Matrix::from_rows(
        rows.iter()
            .map(|r| r.iter().map(|&x| Int::from_i64(x)).collect())
            .collect(),
    )
}

#[test]
fn berkowitz_matches_bareiss_over_int() {
    let cases: &[&[&[i64]]] = &[
        &[&[5]],
        &[&[1, 2], &[3, 4]],
        &[&[2, 0, 1], &[3, 1, 4], &[1, 5, 9]],
        &[&[1, 2, 3, 4], &[5, 6, 0, 8], &[9, 1, 2, 3], &[4, 5, 6, 0]],
        &[&[0, 0, 0], &[1, 2, 3], &[4, 5, 6]], // singular
    ];
    for rows in cases {
        let m = im(rows);
        assert_eq!(
            RingMatrix::det(&m),
            m.determinant(),
            "det mismatch for {rows:?}"
        );
    }
}

#[test]
fn charpoly_known_values() {
    // [[1,2],[3,4]] → x² − 5x − 2, low-to-high [-2, -5, 1]
    let m = im(&[&[1, 2], &[3, 4]]);
    assert_eq!(
        m.charpoly(),
        vec![Int::from_i64(-2), Int::from_i64(-5), Int::from_i64(1)]
    );
    assert_eq!(RingMatrix::det(&m), Int::from_i64(-2));
    // 1×1 → x − 7
    let m1 = im(&[&[7]]);
    assert_eq!(m1.charpoly(), vec![Int::from_i64(-7), Int::from_i64(1)]);
    assert_eq!(RingMatrix::det(&m1), Int::from_i64(7));
}

#[test]
fn determinant_over_composite_modint() {
    use puremp::ModInt;
    let m6 = Int::from_i64(6); // composite → ℤ/6ℤ is NOT a field
    let e = |v: i64| ModInt::new(Int::from_i64(v), m6.clone());
    let m = Matrix::from_rows(vec![
        vec![e(2), e(3), e(1)],
        vec![e(4), e(0), e(5)],
        vec![e(1), e(2), e(3)],
    ]);
    // det is a ring homomorphism: computing over ℤ then reducing mod 6 must agree.
    let int_det = im(&[&[2, 3, 1], &[4, 0, 5], &[1, 2, 3]]).determinant();
    assert_eq!(RingMatrix::det(&m), ModInt::new(int_det, m6));
}

#[cfg(feature = "poly")]
#[test]
fn determinant_over_polynomial_ring() {
    use puremp::Poly;
    let x = Poly::new(vec![Int::from_i64(0), Int::from_i64(1)]); // x
    let one = Poly::new(vec![Int::from_i64(1)]);
    // det [[x,1],[1,x]] = x² − 1
    let m = Matrix::from_rows(vec![vec![x.clone(), one.clone()], vec![one, x]]);
    assert_eq!(
        RingMatrix::det(&m),
        Poly::new(vec![Int::from_i64(-1), Int::from_i64(0), Int::from_i64(1)])
    );
}

#[cfg(feature = "rational")]
#[test]
fn berkowitz_matches_bareiss_over_rational() {
    use puremp::Rational;
    let r = |n: i64, d: i64| Rational::new(Int::from_i64(n), Int::from_i64(d));
    let m = Matrix::from_rows(vec![vec![r(1, 2), r(1, 3)], vec![r(1, 4), r(2, 3)]]);
    assert_eq!(RingMatrix::det(&m), m.determinant());
}
