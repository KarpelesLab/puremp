//! Tests for the generic [`FieldMatrix`] linear algebra (Gaussian elimination
//! with pivoting) over arbitrary fields: `GF(p)` ([`ModInt`] with a prime
//! modulus), `GF(pᵏ)` ([`GfElement`]), and — as a correctness oracle —
//! cross-checked against the trusted concrete Bareiss/fraction-free path for
//! [`Matrix<Rational>`].
#![cfg(all(feature = "matrix", feature = "int"))]

use puremp::{FieldMatrix, Int, Matrix, ModInt};

// ---- helpers over GF(p) = ℤ/pℤ ----

fn mm(p: i64, rows: usize, cols: usize, d: &[i64]) -> Matrix<ModInt> {
    let data = d
        .iter()
        .map(|&x| ModInt::new(Int::from(x), Int::from(p)))
        .collect();
    Matrix::new(rows, cols, data)
}
fn me(p: i64, x: i64) -> ModInt {
    ModInt::new(Int::from(x), Int::from(p))
}

#[test]
fn gf_prime_determinant_by_hand() {
    // det[[1,2],[3,4]] = 1·4 − 2·3 = −2 ≡ 3 (mod 5).
    assert_eq!(mm(5, 2, 2, &[1, 2, 3, 4]).determinant(), me(5, 3));
    // det[[6,1,1],[4,-2,5],[2,8,7]] = −306 ≡ 2 (mod 7).
    assert_eq!(
        mm(7, 3, 3, &[6, 1, 1, 4, -2, 5, 2, 8, 7]).determinant(),
        me(7, 2)
    );
    // Triangular determinant is the product of the diagonal: 2·3·4 = 24 ≡ 3 (7).
    assert_eq!(
        mm(7, 3, 3, &[2, 5, 6, 0, 3, 1, 0, 0, 4]).determinant(),
        me(7, 3)
    );
}

#[test]
fn gf_prime_inverse_roundtrip() {
    let a = mm(5, 2, 2, &[1, 2, 3, 4]); // det 3, invertible over GF(5)
    let inv = a.inverse().expect("invertible");
    let id = Matrix::identity_like(a.get(0, 0), 2);
    assert_eq!(&a * &inv, id);
    assert_eq!(&inv * &a, id);

    // A larger invertible example over GF(7).
    let b = mm(7, 3, 3, &[2, 1, 1, 1, 3, 2, 1, 0, 0]);
    let binv = b.inverse().expect("invertible");
    let id3 = Matrix::identity_like(b.get(0, 0), 3);
    assert_eq!(&b * &binv, id3);
    assert_eq!(&binv * &b, id3);
}

#[test]
fn gf_prime_solve() {
    let a = mm(7, 3, 3, &[2, 1, 1, 1, 3, 2, 1, 0, 0]);
    let b = [me(7, 3), me(7, 5), me(7, 6)];
    let x = a.solve(&b).expect("unique solution");
    // Verify A·x = b.
    let xm = Matrix::new(3, 1, x.to_vec());
    let bm = Matrix::new(3, 1, b.to_vec());
    assert_eq!(&a * &xm, bm);
}

#[test]
fn gf_prime_rank_deficient() {
    // Row 2 = 2·row 1; rows 1 and 3 independent ⇒ rank 2 over GF(5).
    let a = mm(5, 3, 3, &[1, 2, 3, 2, 4, 6, 1, 1, 1]);
    assert_eq!(a.rank(), 2);
    // A full-rank 3×3.
    let b = mm(7, 3, 3, &[2, 1, 1, 1, 3, 2, 1, 0, 0]);
    assert_eq!(b.rank(), 3);
}

#[test]
fn gf_prime_singular() {
    // [[1,2],[2,4]] has det 0 (mod 5): rows proportional.
    let s = mm(5, 2, 2, &[1, 2, 2, 4]);
    assert_eq!(s.determinant(), me(5, 0));
    assert!(s.inverse().is_none());
    assert!(s.solve(&[me(5, 1), me(5, 0)]).is_none());
}

// ---- GF(pᵏ) via GfElement ----

#[cfg(feature = "galois")]
mod extension {
    use super::*;
    use puremp::{GaloisField, GfElement};

    fn gel(f: &GaloisField, coeffs: &[i64]) -> GfElement {
        let c: std::vec::Vec<Int> = coeffs.iter().map(|&x| Int::from(x)).collect();
        f.element(&c)
    }

    #[test]
    fn gf256_inverse_and_det() {
        let f = GaloisField::create(Int::from(2), 8).expect("GF(2^8)");
        // A = [[a, b],[c, d]] with entries the residues of x, x+1, 1, x.
        let a = gel(&f, &[0, 1]); // x
        let b = gel(&f, &[1, 1]); // x + 1
        let c = gel(&f, &[1]); // 1
        let d = gel(&f, &[0, 1]); // x
        let m = Matrix::new(2, 2, std::vec![a.clone(), b.clone(), c.clone(), d.clone()]);
        // det = a·d − b·c (char 2 ⇒ subtraction is addition), computed directly.
        let expected = a.mul(&d).sub(&b.mul(&c));
        assert_eq!(m.determinant(), expected);
        assert!(!m.determinant().is_zero()); // this A is invertible

        let inv = m.inverse().expect("invertible");
        let id = Matrix::identity_like(m.get(0, 0), 2);
        assert_eq!(&m * &inv, id);
        assert_eq!(&inv * &m, id);

        // solve then verify.
        let rhs = std::vec![gel(&f, &[1, 0]), gel(&f, &[0, 1])];
        let x = m.solve(&rhs).expect("unique solution");
        let xm = Matrix::new(2, 1, x);
        let bm = Matrix::new(2, 1, rhs);
        assert_eq!(&m * &xm, bm);
    }

    #[test]
    fn gf9_det_rank_singular() {
        let f = GaloisField::create(Int::from(3), 2).expect("GF(3^2)");
        // Diagonal ⇒ determinant is the product of the diagonal entries.
        let diag = Matrix::new(
            2,
            2,
            std::vec![gel(&f, &[2, 1]), f.zero(), f.zero(), gel(&f, &[1, 1])],
        );
        let prod = gel(&f, &[2, 1]).mul(&gel(&f, &[1, 1]));
        assert_eq!(diag.determinant(), prod);
        assert_eq!(diag.rank(), 2);

        // Singular: two equal rows ⇒ det 0, no inverse, no solve.
        let g = gel(&f, &[1, 2]);
        let h = gel(&f, &[2, 0]);
        let sing = Matrix::new(2, 2, std::vec![g.clone(), h.clone(), g.clone(), h.clone()]);
        assert!(sing.determinant().is_zero());
        assert_eq!(sing.rank(), 1);
        assert!(sing.inverse().is_none());
        assert!(sing.solve(&std::vec![f.zero(), f.one()]).is_none());
    }
}

// ---- cross-check the generic Gaussian path against the trusted Bareiss /
// fraction-free concrete Rational implementation ----

#[cfg(feature = "rational")]
mod cross_check {
    use puremp::{FieldMatrix, Int, Matrix, Rational};

    fn mr(rows: usize, cols: usize, d: &[i64]) -> Matrix<Rational> {
        Matrix::new(rows, cols, d.iter().map(|&x| Rational::from(x)).collect())
    }
    fn frac(rows: usize, cols: usize, num: &[i64], den: &[i64]) -> Matrix<Rational> {
        let data = num
            .iter()
            .zip(den)
            .map(|(&n, &d)| Rational::new(Int::from(n), Int::from(d)))
            .collect();
        Matrix::new(rows, cols, data)
    }

    #[test]
    fn determinant_agrees_with_bareiss() {
        let cases = [
            mr(1, 1, &[7]),
            mr(2, 2, &[1, 2, 3, 4]),
            mr(3, 3, &[6, 1, 1, 4, -2, 5, 2, 8, 7]),
            mr(3, 3, &[2, 0, 1, 3, 1, 4, 1, 1, 1]),
            mr(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]), // singular ⇒ 0
            frac(2, 2, &[1, 2, 1, 5], &[2, 3, 7, 6]),
        ];
        for m in &cases {
            // inherent Bareiss/fraction-free vs. explicit generic Gaussian.
            assert_eq!(m.determinant(), FieldMatrix::determinant(m));
        }
    }

    #[test]
    fn inverse_agrees_with_concrete() {
        let cases = [
            mr(2, 2, &[1, 2, 3, 4]),
            mr(3, 3, &[2, 0, 1, 3, 1, 4, 1, 1, 1]),
            frac(2, 2, &[1, 2, 1, 5], &[2, 3, 7, 6]),
        ];
        for m in &cases {
            let concrete = m.inverse();
            let generic = FieldMatrix::inverse(m);
            assert_eq!(concrete, generic);
            assert!(concrete.is_some());
        }
        // A singular matrix ⇒ both return None.
        let s = mr(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(s.inverse().is_none());
        assert!(FieldMatrix::inverse(&s).is_none());
    }

    #[test]
    fn solve_agrees_with_concrete() {
        let m = mr(3, 3, &[2, 0, 1, 3, 1, 4, 1, 1, 1]);
        let b = [Rational::from(1), Rational::from(2), Rational::from(3)];
        let concrete = m.solve(&b);
        let generic = FieldMatrix::solve(&m, &b);
        assert_eq!(concrete, generic);
        assert!(concrete.is_some());

        // Singular ⇒ both None.
        let s = mr(2, 2, &[1, 2, 2, 4]);
        let bb = [Rational::from(1), Rational::from(0)];
        assert!(s.solve(&bb).is_none());
        assert!(FieldMatrix::solve(&s, &bb).is_none());
    }
}
