//! Tests for general real algebraic numbers.
#![cfg(feature = "algebraic")]

use puremp::{Algebraic, Int, Poly, Rational, RoundingMode};

const N: RoundingMode = RoundingMode::Nearest;

fn poly(cs: &[i64]) -> Poly<Rational> {
    Poly::new(cs.iter().map(|&c| Rational::from(c)).collect())
}

// The positive root of x^2 - k, i.e. √k.
fn root_sqrt(k: i64) -> Algebraic {
    Algebraic::new(
        poly(&[-k, 0, 1]),
        Rational::from(0),
        Rational::from(k.max(1)),
    )
}

#[test]
fn sturm_isolation_and_value() {
    let r2 = root_sqrt(2);
    assert!((r2.to_float(60, N).to_f64() - core::f64::consts::SQRT_2).abs() < 1e-15);
    assert_eq!(r2.signum(), 1);
    assert!(!r2.is_rational());

    // rational collapses
    let three = Algebraic::from_int(Int::from(3));
    assert!(three.is_rational());
    assert_eq!(three.to_float(30, N).to_f64(), 3.0);

    // ordering: 1 < √2 < √3 < 2  (cross-"field" comparison, unlike Quadratic)
    assert!(Algebraic::from_int(Int::ONE) < r2);
    assert!(root_sqrt(2) < root_sqrt(3));
    assert!(root_sqrt(3) < Algebraic::from_int(Int::from(2)));
    // equal values compare equal even with different-looking intervals
    assert_eq!(root_sqrt(2), root_sqrt(2));
    assert!(root_sqrt(2).signum() > 0);
    assert_eq!(root_sqrt(2).neg().signum(), -1);
}

#[test]
fn field_arithmetic_via_resultants() {
    let r2 = root_sqrt(2);
    let r3 = root_sqrt(3);

    // √2 · √3 = √6
    let prod = r2.mul(&r3);
    assert!((prod.to_float(60, N).to_f64() - 6.0f64.sqrt()).abs() < 1e-14);
    assert_eq!(prod, root_sqrt(6));

    // √2 + √3 ≈ 3.1462; it is a root of x^4 - 10x^2 + 1
    let sum = r2.add(&r3);
    assert!((sum.to_float(50, N).to_f64() - (2.0f64.sqrt() + 3.0f64.sqrt())).abs() < 1e-13);
    assert_eq!(sum.defining_polynomial(), &poly(&[1, 0, -10, 0, 1]).monic());

    // 1/√2 = √2/2  → √2 · (1/√2) = 1
    let inv = r2.recip();
    assert!((inv.to_float(50, N).to_f64() - 1.0 / 2.0f64.sqrt()).abs() < 1e-14);
    assert_eq!(inv.mul(&r2), Algebraic::from_int(Int::ONE));

    // sqrt of an algebraic: √(√2) = 2^(1/4)
    let q = r2.sqrt();
    assert!((q.to_float(50, N).to_f64() - 2.0f64.powf(0.25)).abs() < 1e-13);

    // golden ratio as an algebraic root of x^2 - x - 1
    let phi = Algebraic::new(poly(&[-1, -1, 1]), Rational::from(1), Rational::from(2));
    assert_eq!(phi.mul(&phi), phi.add(&Algebraic::from_int(Int::ONE)));
}

#[test]
fn real_roots_of_polynomial() {
    // x^3 - 2x  = x(x^2 - 2): roots -√2, 0, √2
    let roots = Algebraic::real_roots_of(&poly(&[0, -2, 0, 1]));
    assert_eq!(roots.len(), 3);
    assert_eq!(roots[0].signum(), -1);
    assert_eq!(roots[1].signum(), 0);
    assert_eq!(roots[2].signum(), 1);
    assert!((roots[2].to_float(50, N).to_f64() - 2.0f64.sqrt()).abs() < 1e-13);
    // roots come back sorted
    assert!(roots[0] < roots[1] && roots[1] < roots[2]);
    // no real roots
    assert!(Algebraic::real_roots_of(&poly(&[1, 0, 1])).is_empty());
}
