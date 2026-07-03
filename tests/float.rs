//! Tests for the optional arbitrary-precision `Float` layer.
#![cfg(feature = "float")]

use puremp::{Float, Int, RoundingMode};

fn from_i64(v: i64, prec: u64) -> Float {
    Float::from_int(&Int::from_i64(v), prec, RoundingMode::Nearest)
}

#[test]
fn exact_small_arithmetic_matches_f64() {
    let prec = 53;
    let n = RoundingMode::Nearest;
    let a = from_i64(3, prec);
    let b = from_i64(4, prec);
    assert_eq!(a.add(&b, prec, n).to_f64(), 7.0);
    assert_eq!(a.sub(&b, prec, n).to_f64(), -1.0);
    assert_eq!(a.mul(&b, prec, n).to_f64(), 12.0);
    // 3/4 is exact in binary.
    assert_eq!(a.div(&b, prec, n).to_f64(), 0.75);
    // 1/2 + 1/4 == 3/4
    let half = Float::from_int(&Int::ONE, prec, n).div(&from_i64(2, prec), prec, n);
    let quarter = Float::from_int(&Int::ONE, prec, n).div(&from_i64(4, prec), prec, n);
    assert_eq!(half.add(&quarter, prec, n).to_f64(), 0.75);
}

#[test]
fn sqrt_is_correctly_rounded() {
    let prec = 60;
    let n = RoundingMode::Nearest;
    // sqrt(2) to high precision, checked against f64 and by squaring.
    let two = from_i64(2, prec);
    let s = two.sqrt(prec, n);
    assert!((s.to_f64() - core::f64::consts::SQRT_2).abs() < 1e-15);
    // s*s rounds back to approximately 2.
    assert!((s.mul(&s, prec, n).to_f64() - 2.0).abs() < 1e-15);
    // Perfect square is exact.
    assert_eq!(from_i64(144, 20).sqrt(20, n).to_f64(), 12.0);
}

#[test]
fn rounding_modes_direct() {
    // 1/3 at 4 bits: the exact value is 0.0101010..._2.
    let prec = 4;
    let one = Float::from_int(&Int::ONE, 32, RoundingMode::Nearest);
    let three = from_i64(3, 32);
    let down = one.div(&three, prec, RoundingMode::TowardZero);
    let up = one.div(&three, prec, RoundingMode::TowardPositive);
    // Directed rounding must bracket the true value, and up > down.
    assert!(down.to_f64() < 1.0 / 3.0);
    assert!(up.to_f64() > 1.0 / 3.0);
    assert!(up > down);

    // Negative operand: TowardNegative rounds a -1/3 down (more negative) vs
    // TowardZero.
    let neg = one.neg().div(&three, prec, RoundingMode::TowardNegative);
    let negz = one.neg().div(&three, prec, RoundingMode::TowardZero);
    assert!(neg.to_f64() < negz.to_f64());
}

#[test]
fn ordering_and_sign() {
    let prec = 32;
    let a = from_i64(-5, prec);
    let b = from_i64(2, prec);
    assert!(a < b);
    assert!(a.abs() > b);
    assert!(a.neg() > b);
    assert!(Float::zero(prec) > a);
    assert!(Float::zero(prec) < b);
    // Equal values at different precisions compare equal.
    assert_eq!(from_i64(6, 10), from_i64(6, 40));
}

#[test]
fn precision_growth_keeps_value() {
    // 1/3 at low precision, re-rounded to higher precision, then *3 ~ 1.
    let n = RoundingMode::Nearest;
    let third = Float::from_int(&Int::ONE, 100, n).div(&from_i64(3, 100), 100, n);
    let back = third.mul(&from_i64(3, 100), 100, n);
    assert!((back.to_f64() - 1.0).abs() < 1e-28);
    assert_eq!(third.precision(), 100);
}
