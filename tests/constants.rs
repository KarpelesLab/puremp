#![cfg(feature = "float")]
//! Euler–Mascheroni γ and Catalan's constant against known digits.
use puremp::{Float, RoundingMode};
const M: RoundingMode = RoundingMode::Nearest;

// Reference digits (OEIS A001620, A006752).
const GAMMA: &str = "0.57721566490153286060651209008240243104215933593992";
const CATALAN: &str = "0.91596559417721901505460351493238411077414937428167";

fn approx_digits(f: &Float, decimals: u32) -> String {
    // format with enough decimals via the rational, then compare a prefix
    let r = f.to_rational().unwrap();
    let mut s = String::new();
    r.write_decimal(&mut s, decimals, true).unwrap();
    s
}

#[test]
fn euler_gamma_digits() {
    let g = Float::euler_gamma(400, M);
    let s = approx_digits(&g, 40u32);
    assert_eq!(&s[..42], &GAMMA[..42], "got {s}");
}

#[test]
fn catalan_digits() {
    let c = Float::catalan(400, M);
    let s = approx_digits(&c, 40u32);
    assert_eq!(&s[..42], &CATALAN[..42], "got {s}");
}

#[test]
fn gamma_stable_across_precisions() {
    // Higher-precision value, rounded down, matches lower-precision value.
    let lo = Float::euler_gamma(200, M);
    let hi = Float::euler_gamma(500, M).round(200, M);
    assert_eq!(lo, hi);
}
