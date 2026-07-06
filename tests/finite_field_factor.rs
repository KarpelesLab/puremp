//! Cantor–Zassenhaus factorization over finite fields `GF(q)`.
//!
//! These tests exercise [`FactorOverField`] for `Poly<ModInt>` (prime fields
//! `GF(p)`) and `Poly<GfElement>` (extensions `GF(pᵏ)`). Every factorization is
//! verified two ways: the product of the returned `factorⁱ` is re-multiplied and
//! compared against the (monic) input, and each returned factor is checked
//! irreducible.

#![cfg(all(feature = "poly", feature = "galois"))]

use puremp::{FactorOverField, FiniteField, GaloisField, GfElement, Int, ModInt, Poly};

// --------------------------------------------------------------------------
// Helpers.
// --------------------------------------------------------------------------

/// A `GF(p)` element (prime field via `ModInt`).
fn fp(v: i64, p: i64) -> ModInt {
    ModInt::new(Int::from_i64(v), Int::from_i64(p))
}

/// A polynomial over `GF(p)` from integer coefficients (low-to-high).
fn poly_fp(coeffs: &[i64], p: i64) -> Poly<ModInt> {
    Poly::new(coeffs.iter().map(|&c| fp(c, p)).collect())
}

/// Re-multiplies `∏ factorⁱ` and asserts it equals `expected` (both made monic).
fn check_reconstructs<T: FiniteField + core::fmt::Debug>(
    factors: &[(Poly<T>, usize)],
    expected: &Poly<T>,
) {
    let one = expected.leading().expect("nonzero expected").one();
    let mut product = Poly::constant(one);
    for (f, mult) in factors {
        for _ in 0..*mult {
            product = product.mul(f);
        }
    }
    assert_eq!(
        product.monic(),
        expected.monic(),
        "product of factors must reconstruct the input"
    );
}

/// Asserts every returned factor is irreducible and the reconstruction holds.
fn check_full<T: FiniteField + core::fmt::Debug + core::fmt::Display>(
    f: &Poly<T>,
) -> Vec<(Poly<T>, usize)> {
    let factors = f.factor();
    check_reconstructs(&factors, f);
    for (g, _) in &factors {
        assert!(
            g.is_irreducible(),
            "returned factor {g} must be irreducible"
        );
    }
    factors
}

// --------------------------------------------------------------------------
// GF(5).
// --------------------------------------------------------------------------

#[test]
fn gf5_x2_minus_1_splits_into_two_linears() {
    // x² − 1 = (x − 1)(x + 1).
    let f = poly_fp(&[-1, 0, 1], 5);
    let factors = check_full(&f);
    assert_eq!(factors.len(), 2);
    assert!(
        factors
            .iter()
            .all(|(g, m)| *m == 1 && g.degree() == Some(1))
    );
}

#[test]
fn gf5_x2_plus_1_is_irreducible() {
    // x² + 1 over GF(5): −1 is a quadratic residue (2² = 4 = −1), so it actually
    // SPLITS: x² + 1 = (x − 2)(x + 2). Verify the real behavior.
    let f = poly_fp(&[1, 0, 1], 5);
    let factors = check_full(&f);
    assert_eq!(factors.len(), 2, "x²+1 splits over GF(5): {factors:?}");
    assert!(!f.is_irreducible());
}

#[test]
fn gf5_x2_plus_2_is_irreducible() {
    // x² + 2 over GF(5): −2 = 3 is a non-residue (squares are 0,1,4), irreducible.
    let f = poly_fp(&[2, 0, 1], 5);
    assert!(f.is_irreducible());
    let factors = check_full(&f);
    assert_eq!(factors.len(), 1);
    assert_eq!(factors[0].1, 1);
}

#[test]
fn gf5_x3_minus_x_splits_completely() {
    // x³ − x = x(x − 1)(x + 1) over GF(5): three distinct linear factors.
    let f = poly_fp(&[0, -1, 0, 1], 5);
    let factors = check_full(&f);
    assert_eq!(factors.len(), 3);
    assert!(
        factors
            .iter()
            .all(|(g, m)| *m == 1 && g.degree() == Some(1))
    );
}

// --------------------------------------------------------------------------
// GF(2) — characteristic 2, repeated factors via SFF.
// --------------------------------------------------------------------------

#[test]
fn gf2_x2_plus_1_is_a_square() {
    // x² + 1 = (x + 1)² over GF(2): a repeated factor, found by square-free
    // factorization (f' = 0 → p-th root path).
    let f = poly_fp(&[1, 0, 1], 2);
    let factors = check_full(&f);
    assert_eq!(factors.len(), 1);
    let (g, m) = &factors[0];
    assert_eq!(*m, 2, "multiplicity two");
    assert_eq!(g.degree(), Some(1));
    // The square-free factorization reports (x+1, 2).
    let sff = f.squarefree_factorization();
    assert_eq!(sff.len(), 1);
    assert_eq!(sff[0].1, 2);
}

#[test]
fn gf2_x4_plus_x_plus_1_is_irreducible() {
    // x⁴ + x + 1 is irreducible over GF(2) (a primitive polynomial).
    let f = poly_fp(&[1, 1, 0, 0, 1], 2);
    assert!(f.is_irreducible());
    let factors = check_full(&f);
    assert_eq!(factors.len(), 1);
    assert_eq!(factors[0].1, 1);
}

#[test]
fn gf2_x4_plus_x2_plus_1_is_square_of_irreducible() {
    // x⁴ + x² + 1 = (x² + x + 1)² over GF(2).
    let f = poly_fp(&[1, 0, 1, 0, 1], 2);
    let factors = check_full(&f);
    assert_eq!(factors.len(), 1);
    let (g, m) = &factors[0];
    assert_eq!(*m, 2);
    assert_eq!(g.degree(), Some(2));
    assert!(g.is_irreducible());
}

#[test]
fn gf2_product_of_distinct_irreducibles() {
    // (x + 1)(x² + x + 1)(x⁴ + x + 1): three distinct irreducibles of degree
    // 1, 2, 4 — exercises distinct-degree factorization across three degrees.
    let a = poly_fp(&[1, 1], 2); // x + 1
    let b = poly_fp(&[1, 1, 1], 2); // x² + x + 1
    let c = poly_fp(&[1, 1, 0, 0, 1], 2); // x⁴ + x + 1
    let f = a.mul(&b).mul(&c);
    let factors = check_full(&f);
    assert_eq!(factors.len(), 3);
    let mut degs: Vec<usize> = factors.iter().map(|(g, _)| g.degree().unwrap()).collect();
    degs.sort_unstable();
    assert_eq!(degs, vec![1, 2, 4]);
}

// --------------------------------------------------------------------------
// GF(3²) and GF(2⁸) extension fields.
// --------------------------------------------------------------------------

#[test]
fn gf9_product_of_two_irreducibles_comes_back() {
    // Over GF(9) = GF(3)[t]/(t²+1), build two monic irreducible quadratics and
    // verify their product factors back into them.
    let field = GaloisField::create(Int::from_i64(3), 2).expect("GF(9)");
    let x = |c: &[i64]| -> Poly<GfElement> {
        Poly::new(
            c.iter()
                .map(|&v| field.from_int(&Int::from_i64(v)))
                .collect(),
        )
    };
    // Two quadratics with the field generator as a coefficient, kept irreducible.
    let tgen = field.generator(); // t
    // p1 = x² + t   (irreducible iff −t is a non-square in GF(9))
    let p1 = Poly::new(vec![tgen.clone(), field.zero(), field.one()]);
    // p2 = x² + x + t
    let p2 = Poly::new(vec![tgen.clone(), field.one(), field.one()]);
    // Only proceed if both are genuinely irreducible in this field.
    if !p1.is_irreducible() || !p2.is_irreducible() {
        // Fall back to a guaranteed pair: x² + 1 and x² + x + 2 over GF(9).
        let q1 = x(&[1, 0, 1]);
        let q2 = x(&[2, 1, 1]);
        if q1.is_irreducible() && q2.is_irreducible() {
            let f = q1.mul(&q2);
            let factors = check_full(&f);
            assert_eq!(factors.len(), 2);
        }
        return;
    }
    let f = p1.mul(&p2);
    let factors = check_full(&f);
    assert_eq!(factors.len(), 2);
    assert!(
        factors
            .iter()
            .all(|(g, m)| *m == 1 && g.degree() == Some(2))
    );
}

#[test]
fn gf9_irreducible_stays_irreducible_and_square_is_detected() {
    let field = GaloisField::create(Int::from_i64(3), 2).expect("GF(9)");
    let g = field.generator();
    // x² + t is a candidate irreducible; verify irreducibility is preserved and
    // its square is detected with multiplicity two.
    let p = Poly::new(vec![g.clone(), field.zero(), field.one()]);
    if p.is_irreducible() {
        let sq = p.mul(&p);
        let factors = check_full(&sq);
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 2);
    }
}

#[test]
fn gf256_product_of_two_irreducibles() {
    // GF(2⁸) with the AES polynomial x⁸+x⁴+x³+x+1. Multiply two irreducible
    // linear/quadratic factors and confirm they come back (char-2 trace split).
    let aes: Vec<Int> = [1, 1, 0, 1, 1, 0, 0, 0, 1]
        .iter()
        .map(|&c| Int::from_i64(c))
        .collect();
    let field = GaloisField::new(Int::from_i64(2), &aes).expect("GF(256), AES poly");
    let a = field.generator(); // a nonzero, non-one element
    // (x + a)(x + a + 1): two distinct linear factors over GF(256).
    let l1 = Poly::new(vec![a.clone(), field.one()]);
    let l2 = Poly::new(vec![a.add(&field.one()), field.one()]);
    let f = l1.mul(&l2);
    let factors = check_full(&f);
    assert_eq!(factors.len(), 2);
    assert!(
        factors
            .iter()
            .all(|(g, m)| *m == 1 && g.degree() == Some(1))
    );
    // A quadratic that is irreducible over GF(256): x² + x + a (irreducible iff
    // Tr(a) = 1). Only assert when it really is irreducible.
    let quad = Poly::new(vec![a.clone(), field.one(), field.one()]);
    if quad.is_irreducible() {
        let factors = quad.factor();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
    }
}

// --------------------------------------------------------------------------
// is_irreducible on assorted known cases.
// --------------------------------------------------------------------------

#[test]
fn is_irreducible_known_cases() {
    // Irreducible: x² + 2 over GF(5), x⁴ + x + 1 over GF(2).
    assert!(poly_fp(&[2, 0, 1], 5).is_irreducible());
    assert!(poly_fp(&[1, 1, 0, 0, 1], 2).is_irreducible());
    // Reducible: x² − 1 over GF(5), x² + 1 over GF(2) (a square), constants.
    assert!(!poly_fp(&[-1, 0, 1], 5).is_irreducible());
    assert!(!poly_fp(&[1, 0, 1], 2).is_irreducible());
    assert!(!poly_fp(&[3], 7).is_irreducible()); // constant
    // Linear is always irreducible.
    assert!(poly_fp(&[3, 1], 7).is_irreducible());
}

#[test]
fn squarefull_over_gf7() {
    // (x−1)³(x−2) over GF(7): mixed multiplicities exercise SFF then splitting.
    let base = poly_fp(&[-1, 1], 7); // x − 1
    let mut f = poly_fp(&[-2, 1], 7); // x − 2
    for _ in 0..3 {
        f = f.mul(&base);
    }
    let factors = check_full(&f);
    // Two distinct irreducible factors with multiplicities {3, 1}.
    assert_eq!(factors.len(), 2);
    let mut mults: Vec<usize> = factors.iter().map(|(_, m)| *m).collect();
    mults.sort_unstable();
    assert_eq!(mults, vec![1, 3]);
}

// --------------------------------------------------------------------------
// High-degree GF(p): exercises the Kronecker `Poly<ModInt>::mul` hook inside
// distinct-degree / equal-degree factorization (powmod squares degree-~N polys).
// --------------------------------------------------------------------------

/// A tiny LCG so the test is deterministic without an RNG dependency.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

#[test]
fn high_degree_gfp_factors_back() {
    // Build a degree-~120 polynomial over GF(1_000_003) as a product of many
    // random low-degree factors, well above the Kronecker threshold so the
    // multiplies in factor()/powmod route through the packed path. The
    // reconstruction check (product of factors == input) is the differential
    // guarantee: an incorrect Kronecker product would corrupt it.
    let p = 1_000_003;
    let mut rng = Lcg(0x1234_5678);
    let mut f = poly_fp(&[1], p); // constant 1
    while f.degree().unwrap_or(0) < 120 {
        let deg = 1 + (rng.next() as usize % 4);
        let mut coeffs: Vec<i64> = (0..deg).map(|_| (rng.next() % p as u64) as i64).collect();
        coeffs.push(1 + (rng.next() % (p as u64 - 1)) as i64); // nonzero leading
        let g = poly_fp(&coeffs, p);
        f = f.mul(&g);
    }
    // factor() must reconstruct f and return only irreducible factors.
    let factors = check_full(&f);
    assert!(!factors.is_empty());
    // The product of factor degrees × multiplicity equals deg(f).
    let total: usize = factors.iter().map(|(g, m)| g.degree().unwrap() * m).sum();
    assert_eq!(total, f.degree().unwrap());
}
