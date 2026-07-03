//! Tests for the base-10 Decimal type.
#![cfg(feature = "decimal")]

use puremp::{Decimal, Rounding};

fn d(s: &str) -> Decimal {
    s.parse().expect("decimal literal")
}

#[test]
fn exact_arithmetic_and_scale() {
    // 0.1 + 0.2 == 0.3 exactly (the classic binary-float trap)
    assert_eq!((d("0.1") + d("0.2")).to_string(), "0.3");
    assert_eq!((d("1.50") * d("1.5")).to_string(), "2.250"); // scale adds
    assert_eq!((d("100") - d("0.01")).to_string(), "99.99");
    assert_eq!(d("-2.5").neg().to_string(), "2.5");

    // trailing zeros preserved but compare equal
    assert_eq!(d("1.50"), d("1.5"));
    assert_eq!(d("1.50").normalized().to_string(), "1.5");
    assert!(d("2.5") > d("2.49"));

    // parsing incl. scientific notation
    assert_eq!(d("1.5e3").to_string(), "1500");
    assert_eq!(d("2E-8").to_string(), "0.00000002");
    assert_eq!(d("-0.001").to_string(), "-0.001");
}

#[test]
fn division_and_rounding() {
    let he = Rounding::HalfEven;
    // 1/3 to 10 significant digits
    assert_eq!(d("1").div(&d("3"), 10, he).to_string(), "0.3333333333");
    // 2/3
    assert_eq!(d("2").div(&d("3"), 5, he).to_string(), "0.66667");
    // exact division stays exact-ish
    assert_eq!(d("1").div(&d("8"), 20, he).to_string(), "0.125");

    // rounding modes on 2.5 to integer
    assert_eq!(
        d("2.5").round_to_digits(1, Rounding::HalfEven).to_string(),
        "2"
    );
    assert_eq!(
        d("3.5").round_to_digits(1, Rounding::HalfEven).to_string(),
        "4"
    );
    assert_eq!(
        d("2.5").round_to_digits(1, Rounding::HalfUp).to_string(),
        "3"
    );
    assert_eq!(d("2.5").round_to_digits(1, Rounding::Down).to_string(), "2");
    assert_eq!(
        d("-2.5").round_to_digits(1, Rounding::Ceiling).to_string(),
        "-2"
    );
    assert_eq!(
        d("-2.5").round_to_digits(1, Rounding::Floor).to_string(),
        "-3"
    );

    // quantize to money scale (2 fractional digits)
    assert_eq!(
        d("1.005").quantize(-2, Rounding::HalfUp).to_string(),
        "1.01"
    );
    assert_eq!(
        d("1.004").quantize(-2, Rounding::HalfEven).to_string(),
        "1.00"
    );
    assert_eq!(d("2").quantize(-2, Rounding::HalfEven).to_string(), "2.00"); // scale up exact
}

#[cfg(feature = "rational")]
#[test]
fn rational_conversion() {
    assert_eq!(d("0.75").to_rational().to_string(), "3/4");
    assert_eq!(d("1.5e3").to_rational().to_string(), "1500");
}
