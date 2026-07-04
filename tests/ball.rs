#![cfg(feature = "ball")]
//! Ball (mid-rad) arithmetic — soundness (the ball must always enclose the true
//! value, checked against exact rational computation) plus ergonomics.

use puremp::{Ball, Float, Int, Rational, RoundingMode};

const M: RoundingMode = RoundingMode::Nearest;

fn r(n: i64, d: i64) -> Rational {
    Rational::new(Int::from_i64(n), Int::from_i64(d))
}
fn q(n: i64) -> Rational {
    Rational::from_integer(Int::from_i64(n))
}
/// The exact interval [lower, upper] as rationals must bracket `x`.
fn encloses(b: &Ball, x: &Rational) -> bool {
    let lo = b.lower().to_rational().unwrap();
    let hi = b.upper().to_rational().unwrap();
    &lo <= x && x <= &hi
}

#[test]
fn soundness_add_sub_mul_against_exact() {
    for p in [24u64, 53, 100, 300] {
        for (an, ad, bn, bd) in [
            (1, 3, 1, 7),
            (2, 5, 3, 11),
            (-7, 9, 5, 13),
            (99991, 100003, 1, 7),
        ] {
            let (a, b) = (r(an, ad), r(bn, bd));
            let (ba, bb) = (Ball::from_rational(&a, p), Ball::from_rational(&b, p));
            assert!(
                encloses(&ba.add(&bb), &a.add(&b)),
                "add {an}/{ad}+{bn}/{bd}@{p}"
            );
            assert!(encloses(&ba.sub(&bb), &a.sub(&b)), "sub@{p}");
            assert!(encloses(&ba.mul(&bb), &a.mul(&b)), "mul@{p}");
        }
    }
}

#[test]
fn soundness_div_and_sqrt() {
    let p = 120;
    let (a, b) = (r(22, 7), r(3, 5));
    let (ba, bb) = (Ball::from_rational(&a, p), Ball::from_rational(&b, p));
    assert!(encloses(&ba.div(&bb), &a.div(&b))); // 110/21 exactly enclosed

    // sqrt brackets the true (irrational) root: lower² ≤ 2 ≤ upper²
    let root2 = Ball::from_int(&Int::from_i64(2), p).sqrt();
    let lo = root2.lower().to_rational().unwrap();
    let hi = root2.upper().to_rational().unwrap();
    assert!(
        lo.mul(&lo) <= q(2) && q(2) <= hi.mul(&hi),
        "sqrt2 brackets √2"
    );
    // perfect square is exact
    assert!(encloses(&Ball::from_rational(&r(1, 4), p).sqrt(), &r(1, 2)));
}

#[test]
fn error_accumulates_but_stays_sound() {
    // Sum 1/3 a hundred times; the exact value 100/3 must stay enclosed as the
    // radius grows.
    let p = 60;
    let third = Ball::from_rational(&r(1, 3), p);
    let mut acc = Ball::from_int(&Int::ZERO, p);
    for _ in 0..100 {
        acc = acc.add(&third);
    }
    assert!(encloses(&acc, &r(100, 3)));
    assert!(acc.radius().sign() != puremp::Sign::Negative); // radius ≥ 0
}

#[test]
fn point_balls_are_exact() {
    let x = Float::from_f64(0.25, 64, M); // dyadic → exact
    let b = Ball::point(x.clone());
    assert!(b.radius().is_zero());
    // point ⊗ point of dyadics stays exact
    let prod = b.mul(&Ball::point(Float::from_f64(0.5, 64, M)));
    assert!(prod.radius().is_zero());
    assert!(encloses(&prod, &r(1, 8)));
    assert!(b.contains(&x));
}

#[test]
fn contains_zero_and_endpoints() {
    let straddle = Ball::new(Float::from_f64(0.1, 64, M), Float::from_f64(0.5, 32, M));
    assert!(straddle.contains_zero());
    let away = Ball::from_rational(&r(1, 3), 64);
    assert!(!away.contains_zero());
    assert!(away.lower() <= away.upper());
}

#[test]
fn operators_all_combinations() {
    let a = Ball::from_int(&Int::from_i64(6), 64);
    let b = Ball::from_int(&Int::from_i64(4), 64);
    assert!(encloses(&(&a + &b), &q(10)));
    assert!(encloses(&(a.clone() - &b), &q(2)));
    assert!(encloses(&(&a * b.clone()), &q(24)));
    assert!(encloses(&(a.clone() / b.clone()), &r(3, 2)));
    assert!(encloses(&(-&a), &q(-6)));
    let mut c = a.clone();
    c += &b;
    assert!(encloses(&c, &q(10)));
}
