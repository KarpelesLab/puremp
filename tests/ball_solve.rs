#![cfg(feature = "ball")]
//! Rigorous root solving and monotone transcendentals on [`Ball`].
//!
//! Every check here is *rigorous*: the ball returned by a solver, or produced by
//! `Ball::exp` / `Ball::ln`, must genuinely enclose the true value. True
//! transcendental values are bracketed by directed-rounded references
//! (`f(x, DOWN) ≤ true ≤ f(x, UP)`) and all comparisons are done on exact
//! rationals so no floating-point slop can hide a violation.

use puremp::{Ball, Float, Int, Rational, RoundingMode, bisect_root};

const M: RoundingMode = RoundingMode::Nearest;
const DOWN: RoundingMode = RoundingMode::TowardNegative;
const UP: RoundingMode = RoundingMode::TowardPositive;

/// Working precision for the balls under test.
const P: u64 = 100;
/// Higher precision for reference bounds on transcendental constants.
const REF: u64 = 260;

fn fr(f: &Float) -> Rational {
    f.to_rational().unwrap()
}

fn int_ball(n: i64, prec: u64) -> Ball {
    Ball::from_int(&Int::from_i64(n), prec)
}

/// Assert that ball `b` encloses every real in `[lo, hi]` (a bracket around a
/// true value), using exact rational comparison of the endpoints.
fn ball_encloses_bracket(b: &Ball, lo: &Float, hi: &Float) {
    assert!(
        fr(&b.lower()) <= fr(lo),
        "ball lower {} not ≤ bracket lower {}",
        b.lower(),
        lo
    );
    assert!(
        fr(hi) <= fr(&b.upper()),
        "ball upper {} not ≥ bracket upper {}",
        b.upper(),
        hi
    );
}

#[test]
fn bisect_encloses_sqrt2() {
    // f(x) = x² − 2 has a root at √2 in [1, 2].
    let two = int_ball(2, P);
    let f = |x: &Ball| x.mul(x).sub(&two);

    let root = bisect_root(
        f,
        &Float::from_f64(1.0, P, M),
        &Float::from_f64(2.0, P, M),
        P,
        200,
    )
    .expect("certified sign change on [1,2]");

    // Rigorous: lower² ≤ 2 ≤ upper², all as exact rationals.
    let lo = fr(&root.lower());
    let hi = fr(&root.upper());
    let two_q = Rational::from_integer(Int::from_i64(2));
    assert!(&lo * &lo <= two_q, "lower² = {} > 2", &lo * &lo);
    assert!(two_q <= &hi * &hi, "upper² = {} < 2", &hi * &hi);

    // And the enclosure should be genuinely tight after 200 bisections.
    assert!(
        fr(&root.upper()) - fr(&root.lower()) < Rational::new(Int::ONE, Int::from_i64(1_000_000))
    );
}

#[test]
fn bisect_encloses_ln2() {
    // f(x) = exp(x) − 2 has a root at ln 2 in [0, 1].
    let two = int_ball(2, P);
    let f = |x: &Ball| x.exp().sub(&two);

    let root = bisect_root(
        f,
        &Float::from_f64(0.0, P, M),
        &Float::from_f64(1.0, P, M),
        P,
        200,
    )
    .expect("certified sign change on [0,1]");

    // True ln 2 lies in [ln2(DOWN), ln2(UP)] at high precision; the root ball
    // must enclose that whole bracket.
    let ln2_lo = Float::ln2(REF, DOWN);
    let ln2_hi = Float::ln2(REF, UP);
    ball_encloses_bracket(&root, &ln2_lo, &ln2_hi);
}

#[test]
fn exp_ln_enclose_true_values() {
    // For several sample balls, exp/ln must enclose the exact function value at
    // the midpoint and at both endpoints.
    for (mid, rad) in [(0.5_f64, 0.01_f64), (1.5, 0.05), (2.25, 0.1), (0.1, 0.001)] {
        let b = Ball::new(Float::from_f64(mid, P, M), Float::from_f64(rad, P, M));

        let eb = b.exp();
        // Sample points: lower endpoint, midpoint, upper endpoint.
        for x in [b.lower(), b.midpoint().clone(), b.upper()] {
            let lo = x.exp(REF, DOWN);
            let hi = x.exp(REF, UP);
            ball_encloses_bracket(&eb, &lo, &hi);
        }

        // ln on a strictly positive ball.
        let lb = b.ln();
        for x in [b.lower(), b.midpoint().clone(), b.upper()] {
            let lo = x.ln(REF, DOWN);
            let hi = x.ln(REF, UP);
            ball_encloses_bracket(&lb, &lo, &hi);
        }
    }
}

#[test]
fn exp_ln_round_trip() {
    // ln(exp(b)) must contain the original midpoint.
    for mid in [0.5_f64, 1.5, 2.25, 3.0] {
        let b = Ball::new(Float::from_f64(mid, P, M), Float::from_f64(0.01, P, M));
        let rt = b.exp().ln();
        let m = fr(b.midpoint());
        assert!(
            fr(&rt.lower()) <= m && m <= fr(&rt.upper()),
            "round-trip ball [{}, {}] does not contain midpoint {}",
            rt.lower(),
            rt.upper(),
            b.midpoint()
        );
    }
}

#[test]
fn ln_nonpositive_is_indeterminate() {
    // A ball touching/crossing zero is outside ln's domain: NaN mid, ∞ radius.
    let b = Ball::new(Float::from_f64(0.0, P, M), Float::from_f64(0.5, P, M));
    let r = b.ln();
    assert!(!r.is_finite());
    assert!(r.midpoint().is_nan());
}

#[test]
fn no_sign_change_returns_none() {
    // f(x) = x² + 1 is strictly positive everywhere — no certified sign change.
    let one = int_ball(1, P);
    let f = |x: &Ball| x.mul(x).add(&one);
    let out = bisect_root(
        f,
        &Float::from_f64(-1.0, P, M),
        &Float::from_f64(1.0, P, M),
        P,
        50,
    );
    assert!(out.is_none());
}
