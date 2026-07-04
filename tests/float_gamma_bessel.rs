//! Tests for the gamma and Bessel special functions on the arbitrary-precision
//! `Float` layer: `gamma`, `ln_gamma`, `bessel_j` (Jₙ) and `bessel_i` (Iₙ).
//!
//! Each function is checked against a *known closed form* to many digits: the
//! gamma values against `√π` and factorials (via the Stirling core and the
//! reflection formula), the Bessel values against tabulated DLMF constants and
//! the symmetries `J₋ₙ = (−1)ⁿ Jₙ`, `I₋ₙ = Iₙ`. Every function is also checked
//! for precision stability (Ziv consistency: a higher-precision result rounded
//! down reproduces the lower-precision result exactly).
#![cfg(feature = "float")]

use puremp::{Float, Int, Rational, RoundingMode};

const N: RoundingMode = RoundingMode::Nearest;

fn int(v: i64, prec: u64) -> Float {
    Float::from_int(&Int::from_i64(v), prec, N)
}

fn rat(a: i64, b: i64, prec: u64) -> Float {
    Float::from_rational(&Rational::new(Int::from_i64(a), Int::from_i64(b)), prec, N)
}

fn dec(x: &Float, digits: u32) -> String {
    x.to_decimal_string(digits)
}

// --- gamma --------------------------------------------------------------

#[test]
fn gamma_integer_values() {
    // Γ(n) = (n−1)!: Γ(1) = 1, Γ(2) = 1, Γ(5) = 24, Γ(6) = 120.
    let p = 200;
    assert_eq!(dec(&int(1, p).gamma(p, N), 6), "1.000000");
    assert_eq!(dec(&int(2, p).gamma(p, N), 6), "1.000000");
    assert_eq!(dec(&int(5, p).gamma(p, N), 6), "24.000000");
    assert_eq!(dec(&int(6, p).gamma(p, N), 6), "120.000000");
    // Γ(20) = 19! = 121645100408832000.
    let mut f = Int::ONE;
    for k in 2..=19 {
        f = f.mul(&Int::from_i64(k));
    }
    assert_eq!(
        dec(&int(20, p).gamma(p, N), 2),
        dec(&Float::from_int(&f, p, N), 2)
    );
}

#[test]
fn gamma_half_is_sqrt_pi() {
    // Γ(½) = √π, to ~40 digits, compared against Float::pi().sqrt().
    let p = 160;
    let g = rat(1, 2, p).gamma(p, N);
    let sqrt_pi = Float::pi(p + 16, N).sqrt(p, N);
    assert_eq!(dec(&g, 40), dec(&sqrt_pi, 40));
}

#[test]
fn gamma_three_halves_is_sqrt_pi_over_two() {
    // Γ(3/2) = √π / 2.
    let p = 160;
    let g = rat(3, 2, p).gamma(p, N);
    let sqrt_pi_half = Float::pi(p + 16, N).sqrt(p, N).div(&int(2, p), p, N);
    assert_eq!(dec(&g, 40), dec(&sqrt_pi_half, 40));
}

#[test]
fn gamma_negative_half_reflection() {
    // Γ(−½) = −2√π, obtained via the reflection formula.
    let p = 160;
    let g = rat(-1, 2, p).gamma(p, N);
    let m2_sqrt_pi = Float::pi(p + 16, N).sqrt(p, N).mul(&int(-2, p), p, N);
    assert_eq!(dec(&g, 40), dec(&m2_sqrt_pi, 40));
    // Γ(−3/2) = 4√π/3.
    let g = rat(-3, 2, p).gamma(p, N);
    let ref_val = Float::pi(p + 16, N)
        .sqrt(p, N)
        .mul(&int(4, p), p, N)
        .div(&int(3, p), p, N);
    assert_eq!(dec(&g, 40), dec(&ref_val, 40));
}

#[test]
fn gamma_poles_are_nan() {
    // Γ has poles at 0, −1, −2, …
    let p = 64;
    assert!(int(0, p).gamma(p, N).is_nan());
    assert!(int(-1, p).gamma(p, N).is_nan());
    assert!(int(-5, p).gamma(p, N).is_nan());
    assert!(Float::nan(p).gamma(p, N).is_nan());
    assert!(Float::neg_infinity(p).gamma(p, N).is_nan());
    // Γ(+∞) = +∞.
    let g = Float::infinity(p).gamma(p, N);
    assert!(!g.is_finite() && !g.is_sign_negative());
}

#[test]
fn ln_gamma_known_values() {
    // ln Γ(10) = ln(362880).
    let p = 200;
    let lg = int(10, p).ln_gamma(p, N);
    let ln_fact = int(362880, p).ln(p, N);
    assert_eq!(dec(&lg, 40), dec(&ln_fact, 40));
    // ln Γ(5) = ln(24) = 3.17805383034794561964694160129706…
    assert_eq!(
        dec(&int(5, p).ln_gamma(p, N), 32),
        "3.17805383034794561964694160129706"
    );
    // ln Γ(½) = ln √π = ½ ln π = 0.57236494292470008707171367567653…
    assert_eq!(
        dec(&rat(1, 2, p).ln_gamma(p, N), 32),
        "0.57236494292470008707171367567653"
    );
    // ln Γ(1) = ln Γ(2) = 0.
    assert!(int(1, p).ln_gamma(p, N).is_zero());
    assert!(int(2, p).ln_gamma(p, N).is_zero());
    // Domain edges.
    assert!(int(-3, p).ln_gamma(p, N).is_nan());
    let z = Float::zero(p).ln_gamma(p, N);
    assert!(!z.is_finite() && !z.is_sign_negative());
}

#[test]
fn gamma_ziv_precision_stability() {
    // Compute at p+64, round to p, and require equality with the direct-at-p
    // result (a correctly-rounded function is Ziv-stable). The *same* input
    // value is fed at both precisions: a fractional argument like 1/3 is not
    // representable, so we round it once at high precision and reuse it (feeding
    // `rat(1, 3, p)` at each `p` would supply a *different* number each time).
    for &(a, b) in &[(3i64, 1i64), (7, 2), (1, 3), (5, 2), (-3, 2)] {
        let x = rat(a, b, 400);
        for &prec in &[53u64, 64, 100, 150] {
            let direct = x.gamma(prec, N);
            let high = x.gamma(prec + 64, N).round(prec, N);
            assert_eq!(
                dec(&direct, 40),
                dec(&high, 40),
                "gamma({a}/{b}) unstable at p={prec}"
            );
        }
    }
}

// --- Bessel Jₙ ----------------------------------------------------------

#[test]
fn bessel_j_at_zero() {
    // J₀(0) = 1, Jₙ(0) = 0 for n > 0.
    let p = 100;
    assert_eq!(int(0, p).bessel_j(0, p, N).to_f64(), 1.0);
    assert!(int(0, p).bessel_j(1, p, N).is_zero());
    assert!(int(0, p).bessel_j(5, p, N).is_zero());
    assert!(int(0, p).bessel_j(-2, p, N).is_zero());
    assert!(Float::nan(p).bessel_j(0, p, N).is_nan());
}

#[test]
fn bessel_j_tabulated() {
    let p = 200;
    // J₀(1) = 0.76519768655796655144971752610266…
    assert_eq!(
        dec(&int(1, p).bessel_j(0, p, N), 32),
        "0.76519768655796655144971752610266"
    );
    // J₁(1) = 0.44005058574493351595968220371891…
    assert_eq!(
        dec(&int(1, p).bessel_j(1, p, N), 32),
        "0.44005058574493351595968220371891"
    );
    // J₀(10) = −0.24593576445134833519776086248533…
    assert_eq!(
        dec(&int(10, p).bessel_j(0, p, N), 32),
        "-0.24593576445134833519776086248533"
    );
    // Classic 16-digit table values.
    assert!((int(1, p).bessel_j(0, p, N).to_f64() - 0.7651976865579665).abs() < 1e-15);
    assert!((int(1, p).bessel_j(1, p, N).to_f64() - 0.4400505857449335).abs() < 1e-15);
}

#[test]
fn bessel_j_negative_order() {
    // J₋ₙ(x) = (−1)ⁿ Jₙ(x): J₋₁ = −J₁, J₋₂ = J₂, J₋₃ = −J₃.
    let p = 180;
    for &x in &[1i64, 3, 7] {
        let j1 = int(x, p).bessel_j(1, p, N);
        let jm1 = int(x, p).bessel_j(-1, p, N);
        assert_eq!(dec(&jm1, 45), dec(&j1.neg(), 45));

        let j2 = int(x, p).bessel_j(2, p, N);
        let jm2 = int(x, p).bessel_j(-2, p, N);
        assert_eq!(dec(&jm2, 45), dec(&j2, 45));

        let j3 = int(x, p).bessel_j(3, p, N);
        let jm3 = int(x, p).bessel_j(-3, p, N);
        assert_eq!(dec(&jm3, 45), dec(&j3.neg(), 45));
    }
}

#[test]
fn bessel_j_ziv_precision_stability() {
    for &(n, x) in &[(0i64, 1i64), (1, 2), (2, 7), (5, 3), (3, 10)] {
        for &prec in &[53u64, 64, 100, 150] {
            let direct = int(x, prec).bessel_j(n, prec, N);
            let high = int(x, prec + 64).bessel_j(n, prec + 64, N).round(prec, N);
            assert_eq!(
                dec(&direct, 40),
                dec(&high, 40),
                "J_{n}({x}) unstable at p={prec}"
            );
        }
    }
}

// --- Bessel Iₙ ----------------------------------------------------------

#[test]
fn bessel_i_at_zero() {
    // I₀(0) = 1, Iₙ(0) = 0 for n > 0.
    let p = 100;
    assert_eq!(int(0, p).bessel_i(0, p, N).to_f64(), 1.0);
    assert!(int(0, p).bessel_i(2, p, N).is_zero());
    assert!(int(0, p).bessel_i(-3, p, N).is_zero());
    assert!(Float::nan(p).bessel_i(0, p, N).is_nan());
}

#[test]
fn bessel_i_tabulated() {
    let p = 200;
    // I₀(1) = 1.26606587775200833559824462521472…
    assert_eq!(
        dec(&int(1, p).bessel_i(0, p, N), 32),
        "1.26606587775200833559824462521472"
    );
    // I₂(5) = 17.50561496662423601488701189518042…
    assert_eq!(
        dec(&int(5, p).bessel_i(2, p, N), 32),
        "17.50561496662423601488701189518042"
    );
    // Classic 16-digit table value.
    assert!((int(1, p).bessel_i(0, p, N).to_f64() - 1.2660658777520084).abs() < 1e-15);
}

#[test]
fn bessel_i_negative_order_symmetry() {
    // I₋ₙ(x) = Iₙ(x).
    let p = 180;
    for &x in &[1i64, 4] {
        for n in [1i64, 2, 3] {
            let pos = int(x, p).bessel_i(n, p, N);
            let neg = int(x, p).bessel_i(-n, p, N);
            assert_eq!(dec(&pos, 45), dec(&neg, 45));
        }
    }
}

#[test]
fn bessel_i_ziv_precision_stability() {
    for &(n, x) in &[(0i64, 1i64), (1, 4), (2, 5), (3, 2)] {
        for &prec in &[53u64, 64, 100, 150] {
            let direct = int(x, prec).bessel_i(n, prec, N);
            let high = int(x, prec + 64).bessel_i(n, prec + 64, N).round(prec, N);
            assert_eq!(
                dec(&direct, 40),
                dec(&high, 40),
                "I_{n}({x}) unstable at p={prec}"
            );
        }
    }
}
