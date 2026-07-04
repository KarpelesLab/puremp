//! Exercises the generic [`Ring`] abstraction: `Poly<T>` and `Matrix<T>` now work
//! over the context-carrying rings `ModInt` (ℤ/nℤ) and `GfElement` (GF(pᵏ)),
//! whose zero/one need a runtime modulus/field and so cannot come from `Default`.
//!
//! Every check either reduces a hand computation or cross-checks the ring result
//! against ordinary integer arithmetic reduced modulo the characteristic.

#![cfg(all(
    feature = "poly",
    feature = "matrix",
    feature = "galois",
    feature = "complex"
))]

use puremp::{Complex, GaloisField, GfElement, Int, Matrix, ModInt, Poly, Ring};

fn zn(sample: &ModInt, v: i64) -> ModInt {
    sample.of(Int::from_i64(v))
}

// ---------------------------------------------------------------------------
// Poly<ModInt> over ℤ/pℤ.
// ---------------------------------------------------------------------------

#[test]
fn poly_modint_square_reduces_mod_5() {
    let base = ModInt::new(Int::ZERO, Int::from_i64(5));
    let one = zn(&base, 1);
    // (x + 1)² over ℤ/5ℤ.
    let x_plus_1 = Poly::new(alloc_vec(&[one.clone(), one.clone()]));
    let sq = x_plus_1.mul(&x_plus_1);
    // Expect x² + 2x + 1, all coefficients already reduced mod 5.
    let expect = Poly::new(alloc_vec(&[zn(&base, 1), zn(&base, 2), zn(&base, 1)]));
    assert_eq!(sq, expect);
    // Coefficients are the residues 1, 2, 1.
    let residues: alloc::vec::Vec<i64> = sq
        .coeffs()
        .iter()
        .map(|c| c.residue().to_string().parse().unwrap())
        .collect();
    assert_eq!(residues, [1, 2, 1]);
}

#[test]
fn poly_modint_product_and_eval_cross_check() {
    let base = ModInt::new(Int::ZERO, Int::from_i64(5));
    // (2x + 3)(x + 4) over ℤ/5ℤ = 2x² + 11x + 12 ≡ 2x² + x + 2.
    let a = Poly::new(alloc_vec(&[zn(&base, 3), zn(&base, 2)]));
    let b = Poly::new(alloc_vec(&[zn(&base, 4), zn(&base, 1)]));
    let prod = a.mul(&b);
    let expect = Poly::new(alloc_vec(&[zn(&base, 2), zn(&base, 1), zn(&base, 2)]));
    assert_eq!(prod, expect);

    // eval at x = 3: (2·3+3)(3+4) = 9·7 = 63 ≡ 3 (mod 5).
    let at = zn(&base, 3);
    let got = prod.eval(&at);
    assert_eq!(got, zn(&base, 3));
    // Cross-check against plain integer arithmetic reduced mod 5.
    let int_val = ((2 * 3 + 3) * (3 + 4)) % 5;
    assert_eq!(got, zn(&base, int_val));
}

// ---------------------------------------------------------------------------
// Poly<GfElement> over GF(3²).
// ---------------------------------------------------------------------------

#[test]
fn poly_gf_linear_product_identity() {
    let f = GaloisField::create(Int::from_i64(3), 2).expect("GF(3^2) exists");
    let a = f.generator(); // residue class of x
    let b = f.from_int(&Int::from_i64(2)); // the constant 2 ∈ GF(3)
    let one = f.one();

    // (x − a)(x − b) = x² − (a + b)x + a·b.
    let p1 = Poly::new(alloc_vec(&[a.neg(), one.clone()]));
    let p2 = Poly::new(alloc_vec(&[b.neg(), one.clone()]));
    let prod = p1.mul(&p2);

    let expect = Poly::new(alloc_vec(&[a.mul(&b), a.add(&b).neg(), one.clone()]));
    assert_eq!(prod, expect);

    // eval(x − a) at a is 0; the product vanishes at both a and b.
    assert!(p1.eval(&a).is_zero());
    assert!(prod.eval(&a).is_zero());
    assert!(prod.eval(&b).is_zero());
}

#[test]
fn poly_gf_division_uses_leading_inverse() {
    // Over GF(3²): (x² − 1) / (x − 1) = x + 1, exactly.
    let f = GaloisField::create(Int::from_i64(3), 2).expect("GF(3^2) exists");
    let one = f.one();
    let x2_minus_1 = Poly::new(alloc_vec(&[one.neg(), f.zero(), one.clone()]));
    let x_minus_1 = Poly::new(alloc_vec(&[one.neg(), one.clone()]));
    let (q, r) = x2_minus_1.div_rem(&x_minus_1);
    assert!(r.is_zero());
    let expect = Poly::new(alloc_vec(&[one.clone(), one.clone()])); // x + 1
    assert_eq!(q, expect);

    // gcd(x² − 1, x − 1) is monic x − 1.
    let g = x2_minus_1.gcd(&x_minus_1);
    assert_eq!(g, x_minus_1);
}

// ---------------------------------------------------------------------------
// Matrix<ModInt> over ℤ/nℤ.
// ---------------------------------------------------------------------------

#[test]
fn matrix_modint_mul_and_identity() {
    let base = ModInt::new(Int::ZERO, Int::from_i64(7));
    // A = [[1,2],[3,4]], B = [[5,6],[0,1]] over ℤ/7ℤ.
    let a = Matrix::from_rows(alloc_vec(&[
        alloc_vec(&[zn(&base, 1), zn(&base, 2)]),
        alloc_vec(&[zn(&base, 3), zn(&base, 4)]),
    ]));
    let b = Matrix::from_rows(alloc_vec(&[
        alloc_vec(&[zn(&base, 5), zn(&base, 6)]),
        alloc_vec(&[zn(&base, 0), zn(&base, 1)]),
    ]));
    let c = a.mul(&b);
    // Hand computation mod 7:
    // [1·5+2·0, 1·6+2·1] = [5, 8≡1]
    // [3·5+4·0, 3·6+4·1] = [15≡1, 22≡1]
    let expect = Matrix::from_rows(alloc_vec(&[
        alloc_vec(&[zn(&base, 5), zn(&base, 1)]),
        alloc_vec(&[zn(&base, 1), zn(&base, 1)]),
    ]));
    assert_eq!(c, expect);

    // A · I = A using identity_like.
    let id = Matrix::identity_like(&base, 2);
    assert_eq!(a.mul(&id), a);
    assert_eq!(id.mul(&a), a);
}

#[test]
fn matrix_modint_context_constructors() {
    let base = ModInt::new(Int::ZERO, Int::from_i64(11));
    let z = Matrix::zeros_like(&base, 2, 3);
    assert_eq!(z.rows(), 2);
    assert_eq!(z.cols(), 3);
    assert!(z.as_slice().iter().all(ModInt::is_zero));

    let filled = Matrix::filled(zn(&base, 4), 2, 2);
    assert!(filled.as_slice().iter().all(|c| *c == zn(&base, 4)));

    let id = Matrix::identity_like(&base, 3);
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { zn(&base, 1) } else { zn(&base, 0) };
            assert_eq!(*id.get(i, j), want);
        }
    }
}

// ---------------------------------------------------------------------------
// Ring sanity across the impls, and identity-carries-context.
// ---------------------------------------------------------------------------

fn ring_axioms<T: Ring + core::fmt::Debug>(a: &T) {
    let z = a.zero();
    let o = a.one();
    // Additive identity: a + 0 = a and 0 + a = a.
    assert_eq!(a.clone() + z.clone(), *a);
    assert_eq!(z.clone() + a.clone(), *a);
    // Multiplicative identity: a · 1 = a and 1 · a = a.
    assert_eq!(a.clone() * o.clone(), *a);
    assert_eq!(o.clone() * a.clone(), *a);
    // is_zero agrees with equality to zero.
    assert!(z.is_zero());
    assert_eq!(a.is_zero(), *a == z);
}

#[test]
fn ring_axioms_all_impls() {
    ring_axioms(&Int::from_i64(6));
    ring_axioms(&puremp::Rational::new(Int::from_i64(3), Int::from_i64(4)));
    ring_axioms(&puremp::Dyadic::from_int(Int::from_i64(5)));
    ring_axioms(&puremp::Decimal::from_int(Int::from_i64(9)));
    ring_axioms(&Complex::new(Int::from_i64(2), Int::from_i64(-3)));

    let m = zn(&ModInt::new(Int::ZERO, Int::from_i64(13)), 8);
    ring_axioms(&m);

    let f = GaloisField::create(Int::from_i64(2), 8).expect("GF(2^8) exists");
    ring_axioms(&f.generator());
}

#[test]
fn modint_identities_carry_modulus() {
    let m = zn(&ModInt::new(Int::ZERO, Int::from_i64(97)), 42);
    assert_eq!(m.zero().modulus(), Int::from_i64(97));
    assert_eq!(m.one().modulus(), Int::from_i64(97));
    assert!(m.zero().is_zero());
    assert_eq!(m.one(), zn(&m, 1));
}

#[test]
fn gf_identities_carry_field() {
    let f = GaloisField::create(Int::from_i64(3), 2).expect("GF(3^2) exists");
    let g = f.generator();
    let zero: GfElement = g.zero();
    let one: GfElement = g.one();
    assert_eq!(zero.field().characteristic(), Int::from_i64(3));
    assert_eq!(zero.field().degree(), 2);
    assert!(zero.is_zero());
    assert!(one.is_one());
    // one lives in the same field and multiplies correctly.
    assert_eq!(g.mul(&one), g);
}

// A tiny local `alloc::vec!`-free helper so this test file needs no extern alloc
// boilerplate: build a Vec by cloning a slice.
extern crate alloc;
fn alloc_vec<T: Clone>(items: &[T]) -> alloc::vec::Vec<T> {
    items.to_vec()
}
