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
