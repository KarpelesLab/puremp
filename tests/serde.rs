//! Serde round-trip tests (the `serde` feature), driven through `serde_json`.
#![cfg(feature = "serde")]

use puremp::{Float, Int, Nat, Rational, RoundingMode};

#[test]
fn json_roundtrip_integers_and_rationals() {
    let i: Int = "-123456789012345678901234567890".parse().unwrap();
    let json = serde_json::to_string(&i).unwrap();
    assert_eq!(json, "\"-123456789012345678901234567890\"");
    assert_eq!(serde_json::from_str::<Int>(&json).unwrap(), i);

    let n: Nat = "340282366920938463463374607431768211456".parse().unwrap();
    assert_eq!(
        serde_json::from_str::<Nat>(&serde_json::to_string(&n).unwrap()).unwrap(),
        n
    );

    let r: Rational = "-22/7".parse().unwrap();
    assert_eq!(serde_json::to_string(&r).unwrap(), "\"-22/7\"");
    assert_eq!(serde_json::from_str::<Rational>("\"-22/7\"").unwrap(), r);

    // Round-trips inside a container.
    let v = vec![Int::from_i64(1), Int::from_i64(-2), Int::from_i64(3)];
    let json = serde_json::to_string(&v).unwrap();
    assert_eq!(serde_json::from_str::<Vec<Int>>(&json).unwrap(), v);
}

#[test]
fn json_roundtrip_float_is_exact() {
    // High-precision π must survive the round-trip bit-for-bit (exact encoding).
    let pi = Float::pi(200, RoundingMode::Nearest);
    let json = serde_json::to_string(&pi).unwrap();
    let back: Float = serde_json::from_str(&json).unwrap();
    assert_eq!(back, pi);
    assert_eq!(back.precision(), 200);

    // Special values too.
    for f in [
        Float::nan(53),
        Float::infinity(53),
        Float::neg_infinity(53),
        Float::neg_zero(53),
    ] {
        let back: Float = serde_json::from_str(&serde_json::to_string(&f).unwrap()).unwrap();
        assert_eq!(back.is_nan(), f.is_nan());
        assert_eq!(back.is_infinite(), f.is_infinite());
        assert_eq!(back.is_sign_negative(), f.is_sign_negative());
    }
}
