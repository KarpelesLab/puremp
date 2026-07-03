//! Tests for the exact dyadic rational type.
#![cfg(feature = "dyadic")]

use puremp::{Dyadic, Int};

fn d(s: &str) -> Dyadic {
    s.parse().expect("dyadic literal")
}

#[test]
fn arithmetic_is_exact() {
    // 0.5 + 0.25 == 0.75
    assert_eq!((d("0.5") + d("0.25")).to_string(), "0.75");
    // 1.5 * 1.5 == 2.25
    assert_eq!((d("1.5") * d("1.5")).to_string(), "2.25");
    assert_eq!((d("1") - d("0.125")).to_string(), "0.875");
    assert_eq!(d("-0.25").neg().to_string(), "0.25");
    assert_eq!(d("3").pow(3).to_string(), "27");
    // mul_2k
    assert_eq!(d("1").mul_2k(-3).to_string(), "0.125");
    assert_eq!(d("0.125").mul_2k(3).to_string(), "1");
}

#[test]
fn canonical_and_ordering() {
    // new(n, k) == n·2^-k. 4·2^-2 == 1 normalizes to numerator 1, scale 0.
    let x = Dyadic::new(Int::from(4), 2);
    assert_eq!(x.numerator(), &Int::from(1));
    assert_eq!(x.scale(), 0);
    assert_eq!(x.to_string(), "1");
    // 3·2^-3 == 0.375
    assert_eq!(Dyadic::new(Int::from(3), 3).to_string(), "0.375");
    // a positive k divides; equal values compare equal regardless of construction
    assert_eq!(Dyadic::new(Int::from(2), 1), d("1")); // 2·2^-1 == 1
    // a large even integer gets a negative scale after normalization
    let eight = Dyadic::from(8i64);
    assert_eq!(eight.numerator(), &Int::from(1));
    assert_eq!(eight.scale(), -3); // 8 == 1·2^-(-3)
    assert!(d("0.5") < d("0.75"));
    assert!(d("-1") < d("0.001953125")); // -1 < 2^-9
    assert_eq!(d("0").to_string(), "0");
}

#[test]
fn parsing_rejects_non_dyadic() {
    assert!("0.1".parse::<Dyadic>().is_err()); // 1/10 is not dyadic
    assert!("1/3".parse::<Dyadic>().is_err());
    assert_eq!("0.0625".parse::<Dyadic>().unwrap().to_string(), "0.0625"); // 1/16
    // large exponent decimal
    assert_eq!(d("0.00390625").to_string(), "0.00390625"); // 2^-8
}

#[test]
fn floor_trunc() {
    assert_eq!(d("2.75").floor().to_string(), "2");
    assert_eq!(d("-2.75").floor().to_string(), "-3");
    assert_eq!(d("-2.75").trunc().to_string(), "-2");
    assert_eq!(d("2.75").trunc().to_string(), "2");
}

#[cfg(feature = "rational")]
#[test]
fn rational_conversions() {
    use puremp::Rational;
    assert_eq!(d("0.75").to_rational().to_string(), "3/4");
    let r: Rational = "3/8".parse().unwrap();
    assert_eq!(Dyadic::try_from_rational(&r).unwrap().to_string(), "0.375");
    assert!(Dyadic::try_from_rational(&"1/3".parse().unwrap()).is_none());
}

#[cfg(feature = "float")]
#[test]
fn float_conversions() {
    use puremp::RoundingMode;
    let n = RoundingMode::Nearest;
    // 0.375 is exactly representable
    assert_eq!(d("0.375").to_float(53, n).to_f64(), 0.375);
    // round-trip a float back to an exact dyadic
    let f = puremp::Float::from_f64(0.5, 53, n);
    assert_eq!(Dyadic::from_float(&f).unwrap().to_string(), "0.5");
    assert!(Dyadic::from_float(&puremp::Float::nan(53)).is_none());
}
