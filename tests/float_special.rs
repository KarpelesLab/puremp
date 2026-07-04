//! Tests for the special functions on the arbitrary-precision `Float` layer:
//! the error functions `erf`/`erfc` and the Riemann zeta function `zeta`.
//!
//! Each function is checked against a *known closed form* to many digits: the
//! even zeta values against `πⁿ`, `erf` against tabulated constants and its own
//! odd symmetry, and every result for precision-stability (Ziv consistency: a
//! higher-precision result rounded down to a lower precision reproduces the
//! lower-precision result exactly).
#![cfg(feature = "float")]

use puremp::{Float, Int, RoundingMode};

const N: RoundingMode = RoundingMode::Nearest;

fn int(v: i64, prec: u64) -> Float {
    Float::from_int(&Int::from_i64(v), prec, N)
}

fn f64f(x: f64, prec: u64) -> Float {
    Float::from_f64(x, prec, N)
}

/// Decimal prefix used to compare against literal constants.
fn dec(x: &Float, digits: u32) -> String {
    x.to_decimal_string(digits)
}

// --- erf / erfc ---------------------------------------------------------

#[test]
fn erf_zero_and_infinities() {
    // erf(±0) = ±0, erf(±∞) = ±1, erfc(0) = 1, erfc(±∞) = {0, 2}.
    assert!(Float::zero(64).erf(64, N).is_zero());
    assert!(Float::neg_zero(64).erf(64, N).is_sign_negative());
    assert_eq!(Float::infinity(64).erf(64, N).to_f64(), 1.0);
    assert_eq!(Float::neg_infinity(64).erf(64, N).to_f64(), -1.0);
    assert_eq!(Float::zero(64).erfc(64, N).to_f64(), 1.0);
    assert_eq!(Float::infinity(64).erfc(64, N).to_f64(), 0.0);
    assert_eq!(Float::neg_infinity(64).erfc(64, N).to_f64(), 2.0);
    assert!(Float::nan(64).erf(64, N).is_nan());
    assert!(Float::nan(64).erfc(64, N).is_nan());
}

#[test]
fn erf_one_known_value() {
    // erf(1) = 0.84270079294971486934122063508260925929606699796630...
    let e1 = int(1, 200).erf(200, N);
    assert_eq!(dec(&e1, 32), "0.84270079294971486934122063508261");
    // Sanity: the classic 15+ digit value.
    assert!((e1.to_f64() - 0.8427007929497149).abs() < 1e-15);
}

#[test]
fn erf_is_odd() {
    for &x in &[0.3f64, 1.0, 2.5, 4.0] {
        let prec = 180;
        let pos = f64f(x, prec).erf(prec, N);
        let neg = f64f(-x, prec).erf(prec, N);
        assert_eq!(pos.neg().to_decimal_string(40), neg.to_decimal_string(40));
    }
}

#[test]
fn erf_erfc_complementary() {
    // erf(x) + erfc(x) = 1 for a spread of magnitudes (small, series, and CF).
    for &x in &[0.25f64, 0.75, 1.5, 3.0, 6.0] {
        let prec = 220;
        let e = f64f(x, prec).erf(prec, N);
        let c = f64f(x, prec).erfc(prec, N);
        let sum = e.add(&c, prec, N);
        assert_eq!(dec(&sum, 50), int(1, prec).to_decimal_string(50));
    }
}

#[test]
fn erf_large_saturates_and_erfc_small() {
    // erf large ≈ 1, erfc large ≈ 0 (computed via the continued fraction).
    let e = int(6, 200).erf(200, N);
    assert!((e.to_f64() - 1.0).abs() < 1e-16);
    // erfc(3) = 2.20904969985854413727761295167...e-5.
    let c3 = int(3, 200).erfc(200, N);
    assert_eq!(dec(&c3, 22), "0.0000220904969985854414");
    // erfc(5) = 1.5374597944280348501883434853...e-12.
    let c5 = int(5, 200).erfc(200, N);
    assert!((c5.to_f64() - 1.537459794428035e-12).abs() < 1e-24);
}

#[test]
fn erfc_negative_argument() {
    // erfc(−x) = 2 − erfc(x) = 1 + erf(x).
    let prec = 200;
    let c = f64f(-1.5, prec).erfc(prec, N);
    let expect = int(2, prec).sub(&f64f(1.5, prec).erfc(prec, N), prec, N);
    assert_eq!(dec(&c, 45), dec(&expect, 45));
}

// --- zeta ---------------------------------------------------------------

/// π to `prec` bits.
fn pi(prec: u64) -> Float {
    Float::pi(prec, N)
}

#[test]
fn zeta_even_closed_forms() {
    let prec = 256;
    let cmp = 60;

    // ζ(2) = π²/6.
    let z2 = int(2, prec).zeta(prec, N);
    let p = pi(prec);
    let p2 = p.mul(&p, prec, N);
    let z2_exact = p2.div(&int(6, prec), prec, N);
    assert_eq!(dec(&z2, cmp), dec(&z2_exact, cmp));

    // ζ(4) = π⁴/90.
    let z4 = int(4, prec).zeta(prec, N);
    let p4 = p2.mul(&p2, prec, N);
    let z4_exact = p4.div(&int(90, prec), prec, N);
    assert_eq!(dec(&z4, cmp), dec(&z4_exact, cmp));

    // ζ(6) = π⁶/945.
    let z6 = int(6, prec).zeta(prec, N);
    let p6 = p4.mul(&p2, prec, N);
    let z6_exact = p6.div(&int(945, prec), prec, N);
    assert_eq!(dec(&z6, cmp), dec(&z6_exact, cmp));
}

#[test]
fn zeta_known_values() {
    // ζ(3) = Apéry's constant 1.20205690315959428539973816151...
    let z3 = int(3, 200).zeta(200, N);
    assert_eq!(dec(&z3, 30), "1.202056903159594285399738161511");
    // ζ(1/2) = −1.46035450880958681289499152515...
    let zh = f64f(0.5, 200).zeta(200, N);
    assert_eq!(dec(&zh, 30), "-1.460354508809586812889499152515");
}

#[test]
fn zeta_special_points() {
    // ζ(0) = −1/2 exactly; ζ(+∞) = 1; pole at s = 1 → +∞; s < 0 unsupported.
    assert_eq!(Float::zero(64).zeta(64, N).to_f64(), -0.5);
    assert_eq!(Float::infinity(64).zeta(64, N).to_f64(), 1.0);
    let pole = int(1, 64).zeta(64, N);
    assert!(pole.is_infinite() && !pole.is_sign_negative());
    assert!(int(-2, 64).zeta(64, N).is_nan());
    assert!(Float::neg_infinity(64).zeta(64, N).is_nan());
    assert!(Float::nan(64).zeta(64, N).is_nan());
}

// --- Ziv / precision stability ------------------------------------------

/// A value computed at high precision, rounded down to a lower precision,
/// must equal the value computed directly at the lower precision.
fn ziv_consistent<Fun: Fn(u64) -> Float>(lo: u64, hi: u64, g: Fun) {
    let low = g(lo);
    let high_rounded = g(hi).round(lo, N);
    assert_eq!(
        low.to_exact_string(),
        high_rounded.to_exact_string(),
        "Ziv inconsistency between {lo} and {hi} bits"
    );
}

#[test]
fn erf_precision_stable() {
    ziv_consistent(80, 200, |p| f64f(1.0, p + 8).erf(p, N));
    ziv_consistent(80, 200, |p| f64f(0.4, p + 8).erf(p, N));
    ziv_consistent(80, 200, |p| f64f(3.5, p + 8).erf(p, N));
}

#[test]
fn erfc_precision_stable() {
    ziv_consistent(80, 200, |p| f64f(0.5, p + 8).erfc(p, N));
    ziv_consistent(80, 200, |p| f64f(4.0, p + 8).erfc(p, N));
}

#[test]
fn zeta_precision_stable() {
    ziv_consistent(80, 220, |p| int(2, p + 8).zeta(p, N));
    ziv_consistent(80, 220, |p| int(3, p + 8).zeta(p, N));
    ziv_consistent(80, 220, |p| f64f(0.5, p + 8).zeta(p, N));
    ziv_consistent(80, 220, |p| f64f(1.5, p + 8).zeta(p, N));
}
