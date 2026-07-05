//! Tests for a second batch of special functions on the arbitrary-precision
//! `Float` layer: the digamma / polygamma functions `ψ⁽ⁿ⁾`, the beta function
//! `B(a, b)`, and the second-kind Bessel functions `Yₙ` and `Kₙ`.
//!
//! Each result is checked against a *closed form* to many digits — digamma and
//! polygamma against `−γ`, `π²/6`, `−2ζ(3)` and friends, beta against `π` and
//! small rationals, and the Bessel values against tabulated DLMF constants and
//! their three-term recurrences — plus precision-stability (Ziv consistency: a
//! higher-precision result rounded down to a lower precision reproduces the
//! lower-precision result exactly).
#![cfg(feature = "float")]

use puremp::{Float, Int, RoundingMode};

const N: RoundingMode = RoundingMode::Nearest;

fn int(v: i64, prec: u64) -> Float {
    Float::from_int(&Int::from_i64(v), prec, N)
}

fn f64f(x: f64, prec: u64) -> Float {
    Float::from_f64(x, prec, N)
}

fn dec(x: &Float, digits: u32) -> String {
    x.to_decimal_string(digits)
}

// --- digamma ------------------------------------------------------------

#[test]
fn digamma_special_values() {
    let p = 220;
    let g = Float::euler_gamma(p, N);
    let ln2 = Float::ln2(p, N);

    // ψ(1) = −γ.
    assert_eq!(dec(&int(1, p).digamma(p, N), 50), dec(&g.neg(), 50));
    // ψ(½) = −γ − 2 ln 2.
    let half = f64f(0.5, p).digamma(p, N);
    let want_half = g.neg().sub(&ln2, p, N).sub(&ln2, p, N);
    assert_eq!(dec(&half, 50), dec(&want_half, 50));
    // ψ(2) = 1 − γ.
    assert_eq!(
        dec(&int(2, p).digamma(p, N), 50),
        dec(&int(1, p).sub(&g, p, N), 50)
    );
    // ψ(3) = 3/2 − γ.
    let three_halves = f64f(1.5, p);
    assert_eq!(
        dec(&int(3, p).digamma(p, N), 50),
        dec(&three_halves.sub(&g, p, N), 50)
    );
}

#[test]
fn digamma_negative_argument_reflection() {
    // ψ(−½) = 2 − γ − 2 ln 2 (reflection through ψ(3/2)).
    let p = 200;
    let g = Float::euler_gamma(p, N);
    let ln2 = Float::ln2(p, N);
    let want = int(2, p).sub(&g, p, N).sub(&ln2, p, N).sub(&ln2, p, N);
    assert_eq!(dec(&f64f(-0.5, p).digamma(p, N), 45), dec(&want, 45));
}

#[test]
fn digamma_poles_and_nonfinite() {
    assert!(Float::zero(64).digamma(64, N).is_nan());
    assert!(int(-1, 64).digamma(64, N).is_nan());
    assert!(int(-5, 64).digamma(64, N).is_nan());
    assert!(Float::nan(64).digamma(64, N).is_nan());
    assert!(Float::infinity(64).digamma(64, N).is_infinite());
    assert!(!Float::infinity(64).digamma(64, N).is_sign_negative());
    assert!(Float::neg_infinity(64).digamma(64, N).is_nan());
}

// --- polygamma ----------------------------------------------------------

#[test]
fn polygamma_zero_is_digamma() {
    let p = 160;
    for &x in &[0.5f64, 1.0, 2.5, 7.0] {
        let a = f64f(x, p).polygamma(0, p, N);
        let b = f64f(x, p).digamma(p, N);
        assert_eq!(a.to_exact_string(), b.to_exact_string());
    }
}

#[test]
fn polygamma_trigamma_values() {
    let p = 220;
    let pi = Float::pi(p, N);
    let pi2 = pi.mul(&pi, p, N);
    // ψ'(1) = π²/6.
    assert_eq!(
        dec(&int(1, p).polygamma(1, p, N), 50),
        dec(&pi2.div(&int(6, p), p, N), 50)
    );
    // ψ'(½) = π²/2.
    assert_eq!(
        dec(&f64f(0.5, p).polygamma(1, p, N), 50),
        dec(&pi2.div(&int(2, p), p, N), 50)
    );
}

#[test]
fn polygamma_tetragamma_value() {
    // ψ''(1) = −2 ζ(3).
    let p = 220;
    let want = int(3, p).zeta(p, N).mul(&int(-2, p), p, N);
    assert_eq!(dec(&int(1, p).polygamma(2, p, N), 50), dec(&want, 50));
}

#[test]
fn polygamma_recurrence() {
    // ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) + (−1)ⁿ n!/x^{n+1}.
    let p = 200;
    for n in 1u64..=3 {
        let x = f64f(2.5, p);
        let x1 = f64f(3.5, p);
        let lhs = x1.polygamma(n, p, N);
        // n!/x^{n+1}.
        let mut fact = 1i64;
        for k in 2..=n {
            fact *= k as i64;
        }
        let mut xp = int(1, p);
        for _ in 0..=n {
            xp = xp.mul(&x, p, N);
        }
        let mut corr = int(fact, p).div(&xp, p, N);
        if n % 2 == 1 {
            corr = corr.neg();
        }
        let rhs = x.polygamma(n, p, N).add(&corr, p, N);
        assert_eq!(dec(&lhs, 45), dec(&rhs, 45), "polygamma recurrence n={n}");
    }
}

// --- beta ---------------------------------------------------------------

#[test]
fn beta_special_values() {
    let p = 220;
    // B(1,1) = 1.
    assert_eq!(
        dec(&Float::beta(&int(1, p), &int(1, p), p, N), 50),
        dec(&int(1, p), 50)
    );
    // B(2,3) = 1/12.
    assert_eq!(
        dec(&Float::beta(&int(2, p), &int(3, p), p, N), 50),
        dec(&int(1, p).div(&int(12, p), p, N), 50)
    );
    // B(½,½) = π.
    assert_eq!(
        dec(&Float::beta(&f64f(0.5, p), &f64f(0.5, p), p, N), 50),
        dec(&Float::pi(p, N), 50)
    );
}

#[test]
fn beta_symmetry_and_negative() {
    let p = 200;
    // Symmetry B(a,b) = B(b,a).
    let ab = Float::beta(&f64f(2.5, p), &f64f(3.5, p), p, N);
    let ba = Float::beta(&f64f(3.5, p), &f64f(2.5, p), p, N);
    assert_eq!(ab.to_exact_string(), ba.to_exact_string());
    // Negative (non-integer) argument: B(−3/2, 5/2) = Γ(−3/2)Γ(5/2)/Γ(1) = π.
    assert_eq!(
        dec(&Float::beta(&f64f(-1.5, p), &f64f(2.5, p), p, N), 45),
        dec(&Float::pi(p, N), 45)
    );
}

#[test]
fn beta_poles_and_nonfinite() {
    // Non-positive-integer a or b (Γ pole in the numerator) → NaN.
    assert!(Float::beta(&int(0, 64), &int(2, 64), 64, N).is_nan());
    assert!(Float::beta(&int(-2, 64), &int(3, 64), 64, N).is_nan());
    // Only a+b a non-positive integer (Γ pole in the denominator) → 0.
    assert!(Float::beta(&f64f(-1.5, 80), &f64f(-1.5, 80), 80, N).is_zero());
    // Non-finite → NaN.
    assert!(Float::beta(&Float::nan(64), &int(1, 64), 64, N).is_nan());
    assert!(Float::beta(&Float::infinity(64), &int(1, 64), 64, N).is_nan());
}

// --- Bessel Y / K -------------------------------------------------------

#[test]
fn bessel_y_known_values() {
    let p = 200;
    // DLMF-tabulated values (15+ digits).
    assert_eq!(dec(&int(1, p).bessel_y(0, p, N), 15), "0.088256964215677");
    assert_eq!(dec(&int(1, p).bessel_y(1, p, N), 15), "-0.781212821300289");
    // 15-digit float sanity.
    assert!((int(1, p).bessel_y(0, p, N).to_f64() - 0.0882569642156769).abs() < 1e-15);
    assert!((int(1, p).bessel_y(1, p, N).to_f64() - (-0.7812128213002887)).abs() < 1e-15);
}

#[test]
fn bessel_k_known_values() {
    let p = 200;
    assert_eq!(dec(&int(1, p).bessel_k(0, p, N), 15), "0.421024438240708");
    assert_eq!(dec(&int(1, p).bessel_k(1, p, N), 15), "0.601907230197235");
    // K₋ₙ = Kₙ.
    assert_eq!(
        int(1, p).bessel_k(-1, p, N).to_exact_string(),
        int(1, p).bessel_k(1, p, N).to_exact_string()
    );
    assert_eq!(
        int(2, p).bessel_k(-2, p, N).to_exact_string(),
        int(2, p).bessel_k(2, p, N).to_exact_string()
    );
    assert!((int(1, p).bessel_k(0, p, N).to_f64() - 0.4210244382407083).abs() < 1e-15);
    assert!((int(1, p).bessel_k(1, p, N).to_f64() - 0.6019072301972346).abs() < 1e-15);
}

#[test]
fn bessel_y_recurrence() {
    // Y_{n+1}(x) = (2n/x) Yₙ(x) − Y_{n−1}(x), to many digits.
    let p = 200;
    for &x in &[1.0f64, 3.0, 6.0] {
        for n in 1i64..=4 {
            let xf = f64f(x, p);
            let ynm1 = xf.bessel_y(n - 1, p, N);
            let yn = xf.bessel_y(n, p, N);
            let lhs = xf.bessel_y(n + 1, p, N);
            let rhs = int(2 * n, p).div(&xf, p, N).mul(&yn, p, N).sub(&ynm1, p, N);
            assert_eq!(dec(&lhs, 40), dec(&rhs, 40), "Y recurrence n={n} x={x}");
        }
    }
}

#[test]
fn bessel_k_recurrence() {
    // K_{n+1}(x) = (2n/x) Kₙ(x) + K_{n−1}(x), to many digits.
    let p = 200;
    for &x in &[1.0f64, 2.0, 5.0] {
        for n in 1i64..=4 {
            let xf = f64f(x, p);
            let knm1 = xf.bessel_k(n - 1, p, N);
            let kn = xf.bessel_k(n, p, N);
            let lhs = xf.bessel_k(n + 1, p, N);
            let rhs = int(2 * n, p).div(&xf, p, N).mul(&kn, p, N).add(&knm1, p, N);
            assert_eq!(dec(&lhs, 40), dec(&rhs, 40), "K recurrence n={n} x={x}");
        }
    }
}

#[test]
fn bessel_y_k_domain() {
    // x ≤ 0 and non-finite handling.
    assert!(int(1, 64).bessel_y(0, 64, N).is_finite());
    assert!(int(-1, 64).bessel_y(0, 64, N).is_nan());
    assert!(Float::zero(64).bessel_y(0, 64, N).is_infinite());
    assert!(Float::zero(64).bessel_y(0, 64, N).is_sign_negative());
    assert!(Float::nan(64).bessel_y(1, 64, N).is_nan());

    assert!(int(-1, 64).bessel_k(0, 64, N).is_nan());
    assert!(Float::zero(64).bessel_k(0, 64, N).is_infinite());
    assert!(!Float::zero(64).bessel_k(0, 64, N).is_sign_negative());
    assert!(Float::nan(64).bessel_k(1, 64, N).is_nan());
}

// --- Ziv / precision stability ------------------------------------------

/// A value computed at high precision, rounded down to a lower precision, must
/// equal the value computed directly at the lower precision.
fn ziv_consistent<Fun: Fn(u64) -> Float>(lo: u64, hi: u64, g: Fun) {
    let low = g(lo);
    let high_rounded = g(hi).round(lo, N);
    assert_eq!(
        low.to_exact_string(),
        high_rounded.to_exact_string(),
        "Ziv inconsistency between {lo} and {hi} bits"
    );
}

#[test]
fn digamma_precision_stable() {
    ziv_consistent(80, 220, |p| f64f(2.3, p + 8).digamma(p, N));
    ziv_consistent(80, 220, |p| int(1, p + 8).digamma(p, N));
    ziv_consistent(80, 220, |p| f64f(-0.5, p + 8).digamma(p, N));
    ziv_consistent(80, 220, |p| f64f(0.5, p + 8).polygamma(1, p, N));
}

#[test]
fn beta_precision_stable() {
    ziv_consistent(80, 220, |p| {
        Float::beta(&f64f(2.5, p + 8), &f64f(3.5, p + 8), p, N)
    });
    ziv_consistent(80, 220, |p| {
        Float::beta(&int(2, p + 8), &int(3, p + 8), p, N)
    });
}

#[test]
fn bessel_precision_stable() {
    ziv_consistent(80, 200, |p| int(1, p + 8).bessel_y(0, p, N));
    ziv_consistent(80, 200, |p| int(2, p + 8).bessel_k(1, p, N));
    ziv_consistent(80, 200, |p| int(4, p + 8).bessel_y(2, p, N));
}
