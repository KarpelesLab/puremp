//! Tests for generic polynomials.
#![cfg(all(feature = "poly", feature = "rational"))]

use puremp::{Int, Poly, Rational};

fn pint(cs: &[i64]) -> Poly<Int> {
    Poly::new(cs.iter().map(|&c| Int::from(c)).collect())
}
fn prat(cs: &[i64]) -> Poly<Rational> {
    Poly::new(cs.iter().map(|&c| Rational::from(c)).collect())
}

#[test]
fn ring_operations() {
    // (x + 1)(x - 1) = x^2 - 1
    let a = pint(&[1, 1]);
    let b = pint(&[-1, 1]);
    assert_eq!(&a * &b, pint(&[-1, 0, 1]));
    assert_eq!((&a + &b), pint(&[0, 2]));
    assert_eq!(a.degree(), Some(1));
    assert_eq!(Poly::<Int>::zero().degree(), None);

    // eval: (x^2 - 1) at x=3 -> 8
    assert_eq!((&a * &b).eval(&Int::from(3)), Int::from(8));

    // derivative of x^3 + 2x^2 + 1 = 3x^2 + 4x
    assert_eq!(pint(&[1, 0, 2, 1]).derivative(), pint(&[0, 4, 3]));

    // Display
    assert_eq!(pint(&[-1, 0, 1]).to_string(), "1·x^2 + -1");
}

#[test]
fn field_division_and_gcd() {
    // (x^2 - 1) / (x - 1) = x + 1, remainder 0
    let num = prat(&[-1, 0, 1]);
    let den = prat(&[-1, 1]);
    let (q, r) = num.div_rem(&den);
    assert_eq!(q, prat(&[1, 1]));
    assert!(r.is_zero());

    // division with remainder: (x^2 + 1) / (x - 1) = (x + 1) rem 2
    let (q2, r2) = prat(&[1, 0, 1]).div_rem(&prat(&[-1, 1]));
    assert_eq!(q2, prat(&[1, 1]));
    assert_eq!(r2, prat(&[2]));

    // gcd: gcd(x^2-1, x^2-2x+1) = (x-1) monic
    // x^2-1 = (x-1)(x+1); x^2-2x+1 = (x-1)^2  -> gcd x-1
    let g = prat(&[-1, 0, 1]).gcd(&prat(&[1, -2, 1]));
    assert_eq!(g, prat(&[-1, 1])); // monic x - 1
    assert!(g.leading().unwrap().is_one());
}

#[test]
fn real_root_isolation_and_approximation() {
    use puremp::{Float, RoundingMode};
    let n = RoundingMode::Nearest;

    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6 : three real roots
    let p = prat(&[-6, 11, -6, 1]);
    assert_eq!(p.real_root_count(), 3);
    let roots = p.real_roots(53, n);
    assert_eq!(roots.len(), 3);
    let vals: Vec<f64> = roots.iter().map(Float::to_f64).collect();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);

    // x^2 - 2 : two irrational roots ±√2
    let p2 = prat(&[-2, 0, 1]);
    assert_eq!(p2.real_root_count(), 2);
    let r = p2.real_roots(60, n);
    assert!((r[0].to_f64() + core::f64::consts::SQRT_2).abs() < 1e-15);
    assert!((r[1].to_f64() - core::f64::consts::SQRT_2).abs() < 1e-15);

    // x^2 + 1 : no real roots
    assert_eq!(prat(&[1, 0, 1]).real_root_count(), 0);
    assert!(prat(&[1, 0, 1]).real_roots(30, n).is_empty());

    // count in a sub-interval: (x-1)(x-2)(x-3), roots in (1.5, 3.5] -> 2
    assert_eq!(
        p.count_real_roots_in(
            &Rational::new(3.into(), 2.into()),
            &Rational::new(7.into(), 2.into())
        ),
        2
    );

    // repeated root: (x-1)^2 has one distinct real root
    assert_eq!(prat(&[1, -2, 1]).real_root_count(), 1);
}
