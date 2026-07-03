//! Tests for the optional arbitrary-precision `Float` layer.
#![cfg(feature = "float")]

use puremp::{Float, Int, Rational, RoundingMode};

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

#[test]
fn more_transcendentals_match_f64() {
    let p = 60;
    let n = RoundingMode::Nearest;
    let approx = |f: Float| f.to_f64();
    let x = Rational::new(Int::from_i64(3), Int::from_i64(5));
    let xf = Float::from_rational(&x, p, n); // 0.6

    assert!((approx(xf.sinh(p, n)) - 0.6f64.sinh()).abs() < 1e-15);
    assert!((approx(xf.cosh(p, n)) - 0.6f64.cosh()).abs() < 1e-15);
    assert!((approx(xf.tanh(p, n)) - 0.6f64.tanh()).abs() < 1e-15);
    assert!((approx(xf.asin(p, n)) - 0.6f64.asin()).abs() < 1e-15);
    assert!((approx(xf.acos(p, n)) - 0.6f64.acos()).abs() < 1e-15);

    // asin/acos domain: |x|>1 -> NaN.
    assert!(from_i64(2, p).asin(p, n).is_nan());

    // atan2 across quadrants.
    let y = from_i64(1, p);
    let xn = from_i64(-1, p);
    assert!((approx(y.atan2(&xn, p, n)) - 1.0f64.atan2(-1.0)).abs() < 1e-15);
    assert!((approx(y.neg().atan2(&xn, p, n)) - (-1.0f64).atan2(-1.0)).abs() < 1e-15);
    assert!((approx(y.atan2(&Float::zero(p), p, n)) - core::f64::consts::FRAC_PI_2).abs() < 1e-15);

    // pow: 2^10 == 1024, and 2^0.5 == sqrt(2).
    let two = from_i64(2, p);
    assert!((approx(two.pow(&from_i64(10, p), p, n)) - 1024.0).abs() < 1e-10);
    let half = Float::from_rational(&Rational::new(Int::ONE, Int::from_i64(2)), p, n);
    assert!((approx(two.pow(&half, p, n)) - core::f64::consts::SQRT_2).abs() < 1e-15);
}

#[test]
fn inverse_hyperbolics_match_f64() {
    let p = 60;
    let n = RoundingMode::Nearest;
    let a = |f: Float| f.to_f64();
    let x = Float::from_rational(&Rational::new(Int::from_i64(3), Int::from_i64(5)), p, n); // 0.6
    assert!((a(x.asinh(p, n)) - 0.6f64.asinh()).abs() < 1e-15);
    assert!((a(x.atanh(p, n)) - 0.6f64.atanh()).abs() < 1e-15);
    let two = from_i64(2, p);
    assert!((a(two.acosh(p, n)) - 2.0f64.acosh()).abs() < 1e-15);
    // Domain errors.
    assert!(from_i64(0, p).acosh(p, n).is_nan()); // acosh(0) undefined
}

#[test]
fn shortest_decimal_round_trips() {
    let n = RoundingMode::Nearest;
    // Values built from f64 must produce a short string that round-trips.
    for &x in &[1.5f64, 0.1, -0.25, 123.0, 0.001, 1000000.0, -12.5, 6.022e5] {
        let f = Float::from_f64(x, 53, n);
        let s = f.to_shortest_string();
        let back: Rational = s.parse().unwrap();
        assert_eq!(
            Float::from_rational(&back, 53, n),
            f,
            "shortest {s} for {x} must round-trip"
        );
    }
    // Exact small values are minimal.
    assert_eq!(Float::from_f64(1.5, 53, n).to_shortest_string(), "1.5");
    assert_eq!(Float::from_f64(-0.25, 53, n).to_shortest_string(), "-0.25");
    assert_eq!(Float::from_f64(100.0, 53, n).to_shortest_string(), "100");
    assert_eq!(Float::zero(53).to_shortest_string(), "0");
    assert_eq!(Float::infinity(53).to_shortest_string(), "inf");

    // High-precision π round-trips through its shortest form.
    let pi = Float::pi(120, n);
    let s = pi.to_shortest_string();
    let back: Rational = s.parse().unwrap();
    assert_eq!(Float::from_rational(&back, 120, n), pi);
}

#[test]
fn division_and_sqrt_are_correctly_rounded() {
    let n = RoundingMode::Nearest;
    // Division at precision p must equal the exact rational a/b rounded to p bits
    // (from_rational is the correctly-rounded reference).
    for &(a, b) in &[
        (1, 3),
        (2, 7),
        (355, 113),
        (22, 7),
        (1, 1000),
        (999, 1000),
        (7, 9),
        (123, 457),
    ] {
        for &p in &[24u64, 53, 113] {
            let fa = Float::from_int(&Int::from_i64(a), p, n);
            let fb = Float::from_int(&Int::from_i64(b), p, n);
            let got = fa.div(&fb, p, n);
            let want =
                Float::from_rational(&Rational::new(Int::from_i64(a), Int::from_i64(b)), p, n);
            assert_eq!(got, want, "{a}/{b} @ {p} correctly rounded");
        }
    }

    // sqrt at precision p equals the higher-precision sqrt rounded down to p, and
    // its square brackets the input (within one ulp).
    for &v in &[2i64, 3, 5, 7, 10, 1000, 123456789] {
        let p = 60;
        let f = Float::from_int(&Int::from_i64(v), p, n);
        let s = f.sqrt(p, n);
        let s_hi = Float::from_int(&Int::from_i64(v), p + 80, n).sqrt(p + 80, n);
        assert_eq!(s, s_hi.round(p, n), "sqrt({v}) @ {p} correctly rounded");
    }

    // Directed rounding brackets the exact value.
    let fa = Float::from_int(&Int::from_i64(2), 53, n);
    let fb = Float::from_int(&Int::from_i64(3), 53, n);
    let down = fa.div(&fb, 53, RoundingMode::TowardNegative);
    let up = fa.div(&fb, 53, RoundingMode::TowardPositive);
    assert!(down < up);
    let exact = Rational::new(Int::from_i64(2), Int::from_i64(3));
    assert!(down.to_rational().unwrap() < exact && exact < up.to_rational().unwrap());
}

#[test]
fn display_decimal_and_scientific() {
    use puremp::{Float, RoundingMode};
    let n = RoundingMode::Nearest;
    let f = |x: f64| Float::from_f64(x, 53, n);

    // Display is now decimal (shortest round-tripping)
    assert_eq!(f(1.5).to_string(), "1.5");
    assert_eq!(f(-0.25).to_string(), "-0.25");
    assert_eq!(Float::nan(53).to_string(), "NaN");
    assert_eq!(Float::infinity(53).to_string(), "inf");

    // {:.N} → N fractional digits, correctly rounded
    assert_eq!(format!("{:.2}", f(1.0).div(&f(3.0), 53, n)), "0.33");
    assert_eq!(format!("{:.4}", f(2.0).div(&f(3.0), 53, n)), "0.6667");

    // scientific {:e} / {:E}
    assert_eq!(format!("{:e}", f(1500.0)), "1.5e3");
    assert_eq!(format!("{:e}", f(0.00123)), "1.23e-3");
    assert_eq!(format!("{:E}", f(3.25)), "3.25E0");
    assert_eq!(format!("{:e}", Float::zero(53)), "0e0");
    assert_eq!(format!("{:e}", f(-42.0)), "-4.2e1");
}
