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

#[test]
fn special_values() {
    let p = 53;
    let inf = Float::infinity(p);
    let ninf = Float::neg_infinity(p);
    let nan = Float::nan(p);
    let one = from_i64(1, p);
    let zero = Float::zero(p);

    assert!(inf.is_infinite() && !inf.is_finite());
    assert!(nan.is_nan());
    assert!(zero.is_zero() && zero.is_finite());

    let n = RoundingMode::Nearest;
    // inf arithmetic
    assert!(inf.add(&one, p, n).is_infinite());
    assert!(inf.add(&ninf, p, n).is_nan()); // ∞ + (−∞) = NaN
    assert!(inf.mul(&zero, p, n).is_nan()); // ∞ · 0 = NaN
    assert_eq!(one.div(&zero, p, n), inf); // 1/0 = +∞
    assert!(zero.div(&zero, p, n).is_nan()); // 0/0 = NaN
    assert!(one.div(&inf, p, n).is_zero()); // 1/∞ = 0
    assert!(from_i64(-4, p).sqrt(p, n).is_nan());
    assert!(inf.sqrt(p, n).is_infinite());

    // NaN compares unordered; NaN != NaN.
    assert!(nan.partial_cmp(&one).is_none());
    assert_ne!(nan, nan);

    // signed zero
    let nzero = Float::neg_zero(p);
    assert!(nzero.is_sign_negative());
    assert_eq!(nzero, zero); // −0 == +0 in value
    assert_eq!(
        one.neg().mul(&zero, p, n).to_f64().to_bits(),
        (-0.0f64).to_bits()
    );
}

#[test]
fn f64_roundtrip_and_conversion() {
    let p = 53;
    let n = RoundingMode::Nearest;
    for &x in &[
        0.0f64,
        1.0,
        -1.0,
        0.5,
        0.1,
        -123.456,
        core::f64::consts::PI,
        1e300,
        1e-300,
    ] {
        let f = Float::from_f64(x, p, n);
        assert_eq!(f.to_f64(), x, "roundtrip {x}");
    }
    assert!(Float::from_f64(f64::NAN, p, n).is_nan());
    assert!(Float::from_f64(f64::INFINITY, p, n).is_infinite());
    assert_eq!(Float::from_f32(0.25f32, p, n).to_f64(), 0.25);
}

#[test]
fn ternary_flag() {
    // 1/3 at low precision is inexact; nearest rounds it either way, but the
    // ternary must agree with the sign of (rounded - exact).
    let p = 8;
    let one = from_i64(1, 60);
    let three = from_i64(3, 60);
    let (q, t) = one.div_ternary(&three, p, RoundingMode::TowardZero);
    assert_eq!(t, core::cmp::Ordering::Less); // truncation of a positive underestimates
    assert!(q.to_f64() < 1.0 / 3.0);

    let (q2, t2) = one.div_ternary(&three, p, RoundingMode::TowardPositive);
    assert_eq!(t2, core::cmp::Ordering::Greater);
    assert!(q2.to_f64() > 1.0 / 3.0);

    // Exact operation reports Equal.
    let (_, te) = from_i64(2, p).add_ternary(&from_i64(3, p), p, RoundingMode::Nearest);
    assert_eq!(te, core::cmp::Ordering::Equal);
}

#[test]
fn decimal_and_rational_io() {
    let n = RoundingMode::Nearest;
    // Parse decimal string -> Float -> decimal string.
    let f: Float = "1.5".parse().unwrap();
    assert_eq!(f.to_decimal_string(1), "1.5");
    let g: Float = "-0.25".parse().unwrap();
    assert_eq!(g.to_decimal_string(2), "-0.25");
    assert!("inf".parse::<Float>().unwrap().is_infinite());
    assert!("nan".parse::<Float>().unwrap().is_nan());

    // Exact rational conversion round-trips a dyadic value.
    let three_quarters = Float::from_rational(&"3/4".parse().unwrap(), 53, n);
    assert_eq!(three_quarters.to_f64(), 0.75);
    let back = three_quarters.to_rational().unwrap();
    assert_eq!(back.to_string(), "3/4");
    assert!(Float::nan(53).to_rational().is_none());
}

#[test]
fn transcendentals_match_f64() {
    let p = 60;
    let n = RoundingMode::Nearest;
    let approx = |f: Float| f.to_f64();

    assert!((approx(Float::pi(p, n)) - core::f64::consts::PI).abs() < 1e-15);
    assert!((approx(Float::e(p, n)) - core::f64::consts::E).abs() < 1e-15);
    assert!((approx(Float::ln2(p, n)) - core::f64::consts::LN_2).abs() < 1e-15);

    let two = from_i64(2, p);
    assert!((approx(two.ln(p, n)) - core::f64::consts::LN_2).abs() < 1e-15);
    assert!((approx(from_i64(1, p).exp(p, n)) - core::f64::consts::E).abs() < 1e-15);
    // exp(ln(5)) == 5
    let five = from_i64(5, p);
    assert!((approx(five.ln(p, n).exp(p, n)) - 5.0).abs() < 1e-14);

    // sin/cos/tan/atan at x = 1
    let one = from_i64(1, p);
    assert!((approx(one.sin(p, n)) - 1.0f64.sin()).abs() < 1e-15);
    assert!((approx(one.cos(p, n)) - 1.0f64.cos()).abs() < 1e-15);
    assert!((approx(one.tan(p, n)) - 1.0f64.tan()).abs() < 1e-14);
    assert!((approx(one.atan(p, n)) - 1.0f64.atan()).abs() < 1e-15);

    // sin²+cos² == 1
    let x = from_i64(3, p);
    let s = x.sin(p, n);
    let c = x.cos(p, n);
    assert!((approx(s.mul(&s, p, n).add(&c.mul(&c, p, n), p, n)) - 1.0).abs() < 1e-15);

    // atan(∞) = π/2
    assert!((approx(Float::infinity(p).atan(p, n)) - core::f64::consts::FRAC_PI_2).abs() < 1e-15);
}

#[test]
fn pi_high_precision_digits() {
    // 200-bit π, checked against the known decimal expansion. 15 fractional
    // digits is a rounding-stable prefix (the 16th digit is 2).
    let pi = Float::pi(200, RoundingMode::Nearest);
    assert_eq!(pi.to_decimal_string(15), "3.141592653589793");
    // A longer prefix, correctly rounded to 20 fractional digits.
    assert_eq!(pi.to_decimal_string(20), "3.14159265358979323846");
}
