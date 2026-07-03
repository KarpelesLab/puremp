//! Tests for exact quadratic-irrational arithmetic.
#![cfg(feature = "algebraic")]

use puremp::{Int, Quadratic, Rational, RoundingMode};

fn q(a: i64, b: i64, d: i64) -> Quadratic {
    Quadratic::new(Rational::from(a), Rational::from(b), Int::from(d))
}

#[test]
fn field_arithmetic_is_exact() {
    let r2 = Quadratic::sqrt(Int::from(2));
    // (√2)² == 2
    assert_eq!(r2.mul(&r2), Quadratic::from(Int::from(2)));
    // (1+√2)(1-√2) = 1 - 2 = -1  (norm)
    let x = q(1, 1, 2);
    assert_eq!(x.mul(&x.conjugate()), Quadratic::from(Int::from(-1)));
    assert_eq!(x.norm().to_string(), "-1");
    // 1/(1+√2) = -1+√2  (since norm -1)
    assert_eq!(x.recip(), q(-1, 1, 2));
    // (1+√2)^2 = 3 + 2√2
    assert_eq!(x.pow(2), q(3, 2, 2));

    // canonicalization: √8 = 2√2
    let r8 = Quadratic::sqrt(Int::from(8));
    assert_eq!(r8, q(0, 2, 2));
    // √9 collapses to the rational 3
    assert!(Quadratic::sqrt(Int::from(9)).is_rational());
    assert_eq!(Quadratic::sqrt(Int::from(9)), Quadratic::from(Int::from(3)));

    // golden ratio φ = (1+√5)/2 satisfies φ² = φ + 1
    let phi = Quadratic::new(
        Rational::new(1.into(), 2.into()),
        Rational::new(1.into(), 2.into()),
        Int::from(5),
    );
    assert_eq!(phi.pow(2), phi.add(&Quadratic::from(Int::ONE)));
}

#[test]
fn ordering_and_float() {
    let n = RoundingMode::Nearest;
    let r2 = Quadratic::sqrt(Int::from(2));
    // √2 ≈ 1.41421356
    assert!((r2.to_float(60, n).to_f64() - core::f64::consts::SQRT_2).abs() < 1e-15);
    // exact ordering: 1 < √2 < 3/2
    assert!(Quadratic::from(Int::ONE) < r2);
    assert!(r2 < Quadratic::rational(Rational::new(3.into(), 2.into())));
    // -√2 < 0 < √2
    assert!(r2.neg() < Quadratic::from(Int::ZERO));
    // 1+√2 vs 2 (a>0,b>0)
    assert!(q(1, 1, 2) > Quadratic::from(Int::from(2)));
    // 3-√2 vs 2: a=1>0,b=-1<0 -> 1 vs √2 -> 1<√2 so 3-√2<2? 3-1.414=1.586<2 yes
    assert!(q(3, -1, 2) < Quadratic::from(Int::from(2)));

    // different fields are unordered
    assert!(
        Quadratic::sqrt(Int::from(2))
            .partial_cmp(&q(0, 1, 3))
            .is_none()
    );
}
