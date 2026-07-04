//! Integration tests for the experimental-math `identify` module: the inverse
//! symbolic calculator [`puremp::identify`] and Machin-like formula discovery
//! [`puremp::machin_like`].
//!
//! Each test drives a *known* closed form to a few hundred bits and checks that
//! PSLQ recovers it (as a rendered `Display` or exact coefficients), plus a
//! negative case that must not be mis-identified.

use puremp::identify::{identify, identify_with, machin_like};
use puremp::{Float, Int, RoundingMode};

const PREC: u64 = 400;
const M: RoundingMode = RoundingMode::Nearest;

fn f(n: i64) -> Float {
    Float::from_int(&Int::from_i64(n), PREC, M)
}

/// ζ(2) = π²/6 is recognized as `π²/6`.
#[test]
fn identifies_pi_squared_over_six() {
    let zeta2 = f(2).zeta(PREC, M);
    let id = identify(&zeta2, PREC).expect("π²/6 should be identified");
    assert_eq!(id.to_string(), "π²/6");
    // Denominator is 6, single numerator term π² with coefficient 1.
    assert_eq!(id.x_coeff(), &Int::from_i64(6));
    assert_eq!(id.terms().len(), 1);
    assert_eq!(id.terms()[0].0, Int::from_i64(1));
    assert_eq!(id.terms()[0].1, "π²");
}

/// 2·ln2 − 1 is recognized, exercising a multi-term closed form.
#[test]
fn identifies_two_ln2_minus_one() {
    let x = f(2).mul(&Float::ln2(PREC, M), PREC, M).sub(&f(1), PREC, M);
    let id = identify(&x, PREC).expect("2·ln2 − 1 should be identified");
    assert_eq!(id.to_string(), "2·ln2 − 1");
    assert_eq!(id.x_coeff(), &Int::from_i64(1));
}

/// A plain rational 3/4 is recognized via the "1" basis constant.
#[test]
fn identifies_rational() {
    let x = f(3).div(&f(4), PREC, M);
    let id = identify(&x, PREC).expect("3/4 should be identified");
    assert_eq!(id.to_string(), "3/4");
    assert_eq!(id.x_coeff(), &Int::from_i64(4));
}

/// √2 is recognized as √2.
#[test]
fn identifies_sqrt2() {
    let x = f(2).sqrt(PREC, M);
    let id = identify(&x, PREC).expect("√2 should be identified");
    assert_eq!(id.to_string(), "√2");
    assert_eq!(id.x_coeff(), &Int::from_i64(1));
}

/// A deliberately unrelated value (π + 0.12345) is not mis-identified: the only
/// exact relation has an oversized coefficient, which PSLQ's guard rejects.
#[test]
fn rejects_unrelated_value() {
    let offset = Float::from_rational(
        &puremp::Rational::new(Int::from_i64(12345), Int::from_i64(100000)),
        PREC,
        M,
    );
    let x = Float::pi(PREC, M).add(&offset, PREC, M);
    assert!(
        identify(&x, PREC).is_none(),
        "π + 0.12345 must not be identified"
    );
}

/// A custom single-constant basis recovers x = 3·√7.
#[test]
fn identify_with_custom_basis() {
    let sqrt7 = f(7).sqrt(PREC, M);
    let x = f(3).mul(&sqrt7, PREC, M);
    let basis = [("√7", sqrt7)];
    let id = identify_with(&x, PREC, &basis).expect("3·√7 should be identified");
    assert_eq!(id.to_string(), "3·√7");
}

/// Numerically verify a recovered Machin relation: |a₀·π/4 + Σ aᵢ·atan(1/nᵢ)| is
/// tiny at the working precision.
fn assert_relation_vanishes(denoms: &[i64], rel: &[Int]) {
    let one = f(1);
    let mut acc = Float::zero(PREC);
    // a₀·(π/4).
    let pi4 = Float::pi(PREC, M).div(&f(4), PREC, M);
    acc = acc.add(
        &Float::from_int(&rel[0], PREC, M).mul(&pi4, PREC, M),
        PREC,
        M,
    );
    for (i, &n) in denoms.iter().enumerate() {
        let atan = one.div(&f(n), PREC, M).atan(PREC, M);
        acc = acc.add(
            &Float::from_int(&rel[i + 1], PREC, M).mul(&atan, PREC, M),
            PREC,
            M,
        );
    }
    // Should be far below the input accuracy.
    let bound = Float::from_rational(
        &puremp::Rational::new(Int::ONE, Int::ONE.mul_2k(200)),
        PREC,
        M,
    );
    assert!(
        acc.abs() < bound,
        "relation residual not tiny: {}",
        acc.to_f64()
    );
}

/// Machin's formula: π/4 = 4·atan(1/5) − atan(1/239), i.e. relation (1, −4, 1).
#[test]
fn machin_classic() {
    let denoms = [5, 239];
    let rel = machin_like(&denoms, PREC).expect("Machin's formula should be found");
    let want = [1i64, -4, 1].map(Int::from_i64);
    assert_eq!(rel, want);
    assert_relation_vanishes(&denoms, &rel);
}

/// Euler's two-term formula: π/4 = atan(1/2) + atan(1/3), i.e. relation (1, −1, −1).
#[test]
fn machin_euler_two_three() {
    let denoms = [2, 3];
    let rel = machin_like(&denoms, PREC).expect("π/4 = atan(1/2) + atan(1/3) should be found");
    let want = [1i64, -1, -1].map(Int::from_i64);
    assert_eq!(rel, want);
    assert_relation_vanishes(&denoms, &rel);
}

/// Denominators below 2 are rejected.
#[test]
fn machin_rejects_small_denominators() {
    assert!(machin_like(&[1, 5], PREC).is_none());
}
