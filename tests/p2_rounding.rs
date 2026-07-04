#![cfg(all(feature = "float", feature = "algebraic"))]
//! P2: Float→Int rounding, Rational::round (half-even), Algebraic exactness/to_f64.

use core::str::FromStr;
use puremp::{Algebraic, Float, Int, Rational, RoundingMode};

fn r(n: i64, d: i64) -> Rational {
    Rational::new(Int::from_i64(n), Int::from_i64(d))
}

#[test]
fn rational_round_half_even() {
    assert_eq!(r(5, 2).round().to_string(), "2"); // 2.5 → 2
    assert_eq!(r(7, 2).round().to_string(), "4"); // 3.5 → 4
    assert_eq!(r(-5, 2).round().to_string(), "-2"); // -2.5 → -2
    assert_eq!(r(1, 3).round().to_string(), "0");
    assert_eq!(r(2, 3).round().to_string(), "1");
    assert_eq!(r(3, 1).round().to_string(), "3");
}

#[test]
fn float_to_int_rounding() {
    let x = Float::from_str("-3.5").unwrap();
    assert_eq!(x.floor().unwrap().to_string(), "-4");
    assert_eq!(x.ceil().unwrap().to_string(), "-3");
    assert_eq!(x.trunc().unwrap().to_string(), "-3");
    assert_eq!(x.round_to_int().unwrap().to_string(), "-4"); // -3.5 ties to even (-4)
    assert_eq!(
        Float::from_str("2.5")
            .unwrap()
            .round_to_int()
            .unwrap()
            .to_string(),
        "2"
    );
    // NaN/inf → None
    assert!(Float::nan(53).floor().is_none());
}

#[test]
fn algebraic_exact_sqrt() {
    // Sqrt[2]*Sqrt[2] == 2 exactly.
    let sqrt2 = Algebraic::from_int(Int::from_i64(2)).sqrt();
    let two = sqrt2.mul(&sqrt2);
    assert!(two.is_rational());
    assert_eq!(two.to_f64(), 2.0);
    // (1+√2)^2 == 3 + 2√2  → numerically 5.828...
    let one_plus = Algebraic::from_int(Int::ONE).add(&sqrt2);
    let sq = one_plus.mul(&one_plus);
    assert!((sq.to_f64() - (3.0 + 2.0 * 2f64.sqrt())).abs() < 1e-12);
    // sqrt of a rational, exact: sqrt(1/4) = 1/2
    let half = Algebraic::from_rational(r(1, 4)).sqrt();
    assert!(half.is_rational());
    assert!((half.to_f64() - 0.5).abs() < 1e-15);
    let _ = RoundingMode::Nearest;
}
