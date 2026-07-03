//! Tests for interval arithmetic.
#![cfg(feature = "interval")]

use puremp::{Float, Interval, Rational, RoundingMode};

fn iv(lo: f64, hi: f64, p: u64) -> Interval {
    let n = RoundingMode::Nearest;
    Interval::new(Float::from_f64(lo, p, n), Float::from_f64(hi, p, n), p)
}

#[test]
fn enclosure() {
    // [1,2] + [3,4] = [4,6]
    let s = &iv(1.0, 2.0, 53) + &iv(3.0, 4.0, 53);
    assert_eq!(s.lower().to_f64(), 4.0);
    assert_eq!(s.upper().to_f64(), 6.0);

    // [1,2] - [3,4] = [-3, -1]
    let d = &iv(1.0, 2.0, 53) - &iv(3.0, 4.0, 53);
    assert_eq!(d.lower().to_f64(), -3.0);
    assert_eq!(d.upper().to_f64(), -1.0);

    // [-1,2] * [3,4] = [-4, 8]
    let m = &iv(-1.0, 2.0, 53) * &iv(3.0, 4.0, 53);
    assert_eq!(m.lower().to_f64(), -4.0);
    assert_eq!(m.upper().to_f64(), 8.0);

    // [1,2]/[4,8] = [0.125, 0.5]
    let q = &iv(1.0, 2.0, 53) / &iv(4.0, 8.0, 53);
    assert_eq!(q.lower().to_f64(), 0.125);
    assert_eq!(q.upper().to_f64(), 0.5);

    assert!(iv(-1.0, 1.0, 53).contains_zero());
    assert!(!iv(1.0, 2.0, 53).contains_zero());
}

#[test]
fn outward_rounding_encloses_third() {
    // 1/3 is not representable; the interval must strictly straddle it.
    let third = Interval::from_rational(&Rational::new(1i64.into(), 3i64.into()), 20);
    // lo < 1/3 < hi
    assert!(third.lower().to_f64() < 1.0 / 3.0);
    assert!(third.upper().to_f64() > 1.0 / 3.0);
    // width is positive but tiny
    assert!(third.width().to_f64() > 0.0);

    // sqrt([2,2]) encloses √2
    let r2 = Interval::point(Float::from_f64(2.0, 60, RoundingMode::Nearest)).sqrt();
    assert!(r2.lower().to_f64() <= core::f64::consts::SQRT_2);
    assert!(r2.upper().to_f64() >= core::f64::consts::SQRT_2);

    // hull and intersect
    let h = iv(1.0, 3.0, 53).hull(&iv(2.0, 5.0, 53));
    assert_eq!(h.lower().to_f64(), 1.0);
    assert_eq!(h.upper().to_f64(), 5.0);
    let i = iv(1.0, 3.0, 53).intersect(&iv(2.0, 5.0, 53)).unwrap();
    assert_eq!(i.lower().to_f64(), 2.0);
    assert_eq!(i.upper().to_f64(), 3.0);
    assert!(iv(1.0, 2.0, 53).intersect(&iv(3.0, 4.0, 53)).is_none());
}
