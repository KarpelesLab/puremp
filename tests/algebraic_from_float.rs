#![cfg(all(feature = "algebraic", feature = "lattice"))]
//! `Algebraic::from_float`: recover exact algebraic numbers from Float
//! approximations via LLL minimal-polynomial recovery + root isolation.

use puremp::{Algebraic, Float, Int, Rational, RoundingMode};

const P: u64 = 320;
const M: RoundingMode = RoundingMode::Nearest;

fn f(n: i64) -> Float {
    Float::from_int(&Int::from_i64(n), P, M)
}
fn alg_int(n: i64) -> Algebraic {
    Algebraic::from_int(Int::from_i64(n))
}

#[test]
fn recovers_sqrt2_with_sign() {
    let s2 = f(2).sqrt(P, M);
    let a = Algebraic::from_float(&s2, 4).expect("recover √2");
    assert_eq!(a.mul(&a), alg_int(2)); // a² = 2 exactly
    assert_eq!(a.signum(), 1); // positive root chosen

    let b = Algebraic::from_float(&s2.neg(), 4).expect("recover −√2");
    assert_eq!(b.mul(&b), alg_int(2));
    assert_eq!(b.signum(), -1); // negative root chosen
    assert_eq!(b, a.neg());
}

#[test]
fn recovers_golden_ratio() {
    // φ = (1+√5)/2 satisfies φ² = φ + 1.
    let phi = f(1).add(&f(5).sqrt(P, M), P, M).div(&f(2), P, M);
    let a = Algebraic::from_float(&phi, 4).expect("recover φ");
    assert_eq!(a.mul(&a), a.add(&alg_int(1)));
}

fn close(a: &Algebraic, x: &Float) -> bool {
    a.to_float(64, M).sub(x, 64, M).abs() < f(1).div(&f(1_000_000), 64, M)
}

#[test]
fn recovers_cube_root_2() {
    let cbrt2 = f(2).pow(&f(1).div(&f(3), P, M), P, M);
    let a = Algebraic::from_float(&cbrt2, 4).expect("recover ∛2");
    assert_eq!(a.defining_polynomial().coeffs().len(), 4); // x³ − 2
    assert!(close(&a, &cbrt2)); // and it is ∛2 (cheap check; Algebraic mul is costly)
}

#[test]
fn recovers_degree_four() {
    // √2 + √3, minimal polynomial x⁴ − 10x² + 1.
    let s = f(2).sqrt(P, M).add(&f(3).sqrt(P, M), P, M);
    let a = Algebraic::from_float(&s, 5).expect("recover √2+√3");
    assert_eq!(a.defining_polynomial().coeffs().len(), 5); // x⁴ − 10x² + 1
    assert!(close(&a, &s)); // and it really is √2+√3 ≈ 3.146
    // The defining polynomial is exactly x⁴ − 10x² + 1 (monic).
    let c: Vec<Rational> = a.defining_polynomial().coeffs().to_vec();
    let want = [1, 0, -10, 0, 1].map(|k| Rational::from_integer(Int::from_i64(k)));
    assert_eq!(c, want);
}

#[test]
fn recovers_rational() {
    // A rational value comes back as a degree-1 algebraic equal to itself.
    let three_halves = f(3).div(&f(2), P, M);
    let a = Algebraic::from_float(&three_halves, 4).expect("recover 3/2");
    assert!(a.is_rational());
    assert_eq!(
        a,
        Algebraic::from_rational(Rational::new(Int::from_i64(3), Int::from_i64(2)))
    );
}

#[test]
fn rejects_transcendental() {
    // π has no minimal polynomial of degree ≤ 6.
    let pi = Float::pi(P, M);
    assert!(Algebraic::from_float(&pi, 6).is_none());
}
