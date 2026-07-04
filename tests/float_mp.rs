#![cfg(feature = "float")]
//! Multi-prime exp fast path: must match the series path (bit-for-bit) across
//! the precision band where it engages, over many arguments.

use core::str::FromStr;
use puremp::{Float, RoundingMode};

const M: RoundingMode = RoundingMode::Nearest;

// Reference: exp at a precision above the embedded range (forces the series),
// rounded down to `prec` — the correctly-rounded value.
fn reference(x: &Float, prec: u64) -> Float {
    x.exp(9600, M).round(prec, M)
}

#[test]
fn multiprime_exp_matches_series_random() {
    let mut seed = 0x0BADF00Du64;
    for &prec in &[384u64, 512, 1000, 2048, 4096, 8000, 9000] {
        for _ in 0..40 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let whole = (seed >> 40) as i64 % 200 - 100;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let frac = (seed >> 40) % 1_000_000_000;
            let x = Float::from_str(&format!("{whole}.{frac:09}")).unwrap();
            assert_eq!(x.exp(prec, M), reference(&x, prec), "exp({x:?}) @ {prec}");
        }
    }
}

#[test]
fn multiprime_exp_edge_cases() {
    for s in [
        "0.0001",
        "-0.0001",
        "1",
        "-1",
        "0.6931471805599453",
        "88.5",
        "-88.5",
        "709.78",
    ] {
        let x = Float::from_str(s).unwrap();
        for &prec in &[500u64, 1500, 4096] {
            assert_eq!(x.exp(prec, M), reference(&x, prec), "exp({s}) @ {prec}");
        }
    }
    // Directed rounding modes must also agree.
    let x = Float::from_str("3.14159").unwrap();
    for mode in [
        RoundingMode::TowardZero,
        RoundingMode::TowardPositive,
        RoundingMode::TowardNegative,
    ] {
        assert_eq!(
            x.exp(2048, mode),
            x.exp(9600, mode).round(2048, mode),
            "mode {mode:?}"
        );
    }
}
