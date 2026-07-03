//! Tests for the fixed-precision float wrapper.
#![cfg(feature = "float")]

use puremp::{FixedFloat, Int, RoundingMode};

fn f(v: i64, p: u64) -> FixedFloat {
    FixedFloat::from_int(&Int::from_i64(v), p, RoundingMode::Nearest)
}

#[test]
fn operators_use_baked_precision() {
    let a = f(3, 60);
    let b = f(4, 60);
    // Ergonomic operators — no per-op precision/mode.
    assert_eq!((&a + &b).to_f64(), 7.0);
    assert_eq!((&a - &b).to_f64(), -1.0);
    assert_eq!((&a * &b).to_f64(), 12.0);
    assert_eq!((&a / &b).to_f64(), 0.75);
    assert_eq!((-a.clone()).to_f64(), -3.0);

    // *Assign
    let mut acc = f(10, 60);
    acc += f(5, 60);
    assert_eq!(acc.to_f64(), 15.0);

    // precision carries through; mixed precision takes the max.
    assert_eq!((&f(1, 20) + &f(1, 80)).precision(), 80);
}

#[test]
fn transcendentals_and_compare() {
    let n = RoundingMode::Nearest;
    let two = FixedFloat::from_int(&Int::from_i64(2), 80, n);
    assert!((two.sqrt().to_f64() - core::f64::consts::SQRT_2).abs() < 1e-15);
    assert!((FixedFloat::pi(80, n).to_f64() - core::f64::consts::PI).abs() < 1e-15);
    let one = FixedFloat::from_int(&Int::ONE, 80, n);
    assert!((one.exp().to_f64() - core::f64::consts::E).abs() < 1e-15);
    // (e^x).ln() == x
    let x = FixedFloat::from_f64(1.75, 80, n);
    assert!((x.exp().ln().to_f64() - 1.75).abs() < 1e-14);

    assert!(f(-5, 40) < f(2, 40));
    assert_eq!(f(6, 10), f(6, 40)); // equal values across precisions
    assert!(FixedFloat::nan(40, n).partial_cmp(&f(1, 40)).is_none());

    // pow and shortest string
    let sqrt2 = two.pow(&FixedFloat::from_f64(0.5, 80, n));
    assert!((sqrt2.to_f64() - core::f64::consts::SQRT_2).abs() < 1e-15);
    assert_eq!(FixedFloat::from_f64(1.5, 53, n).to_shortest_string(), "1.5");
}
