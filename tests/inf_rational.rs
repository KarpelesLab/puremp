//! Tests for the extended-rational type (±∞ / NaN).
#![cfg(feature = "rational")]

use puremp::{InfRational, Int};

fn r(s: &str) -> InfRational {
    s.parse().expect("inf-rational literal")
}

#[test]
fn ieee_style_arithmetic() {
    let inf = r("inf");
    let ninf = r("-inf");
    let nan = r("nan");
    let one = r("1");
    let zero = r("0");
    let half = r("1/2");

    assert!(inf.is_infinite() && !inf.is_finite());
    assert!(nan.is_nan());
    assert!(zero.is_zero());

    // addition
    assert_eq!(&inf + &one, inf);
    assert!((&inf + &ninf).is_nan()); // ∞ + (−∞)
    assert_eq!((&half + &half).to_string(), "1");

    // multiplication
    assert!((&inf * &zero).is_nan()); // ∞ · 0
    assert_eq!(&inf * &r("-2"), ninf); // ∞ · (−2)
    assert_eq!((&half * &half).to_string(), "1/4");

    // division
    assert_eq!(&one / &zero, inf); // 1/0
    assert_eq!(&r("-1") / &zero, ninf); // −1/0
    assert!((&zero / &zero).is_nan()); // 0/0
    assert!((&inf / &inf).is_nan()); // ∞/∞
    assert_eq!(&one / &inf, zero); // 1/∞ = 0
    assert_eq!((&r("2/3") / &r("4/5")).to_string(), "5/6");

    // recip / ratio construction
    assert_eq!(InfRational::ratio(Int::from(1), Int::from(0)), inf);
    assert_eq!(r("2").recip().to_string(), "1/2");
    assert_eq!(zero.recip(), inf);
}

#[test]
fn ordering_and_display() {
    let inf = r("inf");
    let ninf = r("-inf");
    assert!(ninf < r("-1000000"));
    assert!(r("1000000") < inf);
    assert!(ninf < inf);
    assert!(r("1/3") < r("1/2"));
    // NaN is unordered and never equal.
    assert!(r("nan").partial_cmp(&r("1")).is_none());
    assert_ne!(r("nan"), r("nan"));
    assert_eq!(inf, r("inf"));

    assert_eq!(r("inf").to_string(), "inf");
    assert_eq!(r("-inf").to_string(), "-inf");
    assert_eq!(r("nan").to_string(), "NaN");
    assert_eq!(r("-3/4").to_string(), "-3/4");
    assert!(r("inf").to_rational().is_none());
    assert_eq!(r("5/2").to_rational().unwrap().to_string(), "5/2");
}
