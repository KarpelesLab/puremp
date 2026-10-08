#![cfg(feature = "float")]
//! Differential tests: the binary-split Catalan (Pilehrood series) and
//! Euler–Mascheroni (refined Brent–McMillan) constants against the original
//! term-by-term implementations, for many precisions and every rounding mode.
use puremp::{Float, RoundingMode};

const MODES: [RoundingMode; 5] = [
    RoundingMode::Nearest,
    RoundingMode::TowardZero,
    RoundingMode::TowardPositive,
    RoundingMode::TowardNegative,
    RoundingMode::AwayFromZero,
];

/// Catalan's constant to 216 decimals (OEIS A006752).
const CATALAN: &str = "0.915965594177219015054603514932384110774149374281672134266498119621763019776254769479356512926115106248574422619196199579035898803325859059431594737481158406995332028773319460519038727478164087865909024706484152163000";
/// Euler–Mascheroni γ to 50 decimals (OEIS A001620).
const GAMMA: &str = "0.57721566490153286060651209008240243104215933593992";

fn precisions(max: u64) -> impl Iterator<Item = u64> {
    // Every precision up to 160 (around the γ dispatch threshold), then a
    // coarser, irregular stride.
    (2..=160u64).chain((161..=max).step_by(151))
}

fn check(max: u64) {
    for p in precisions(max) {
        for &m in &MODES {
            assert_eq!(
                Float::catalan_bsplit(p, m),
                Float::catalan_series_reference(p, m),
                "catalan p={p} {m:?}"
            );
            assert_eq!(
                Float::euler_gamma_bsplit(p, m),
                Float::euler_gamma_series_reference(p, m),
                "gamma p={p} {m:?}"
            );
        }
    }
}

#[test]
fn bsplit_matches_series_reference() {
    check(4000);
}

#[test]
fn dispatched_matches_reference() {
    for p in [2u64, 53, 64, 96, 97, 113, 200, 1000, 2500] {
        for &m in &MODES {
            assert_eq!(Float::catalan(p, m), Float::catalan_series_reference(p, m));
            assert_eq!(
                Float::euler_gamma(p, m),
                Float::euler_gamma_series_reference(p, m)
            );
        }
    }
}

fn digits(f: &Float, decimals: u32) -> String {
    let mut s = String::new();
    f.to_rational()
        .unwrap()
        .write_decimal(&mut s, decimals, true)
        .unwrap();
    s
}

#[test]
fn known_digits() {
    // 216 decimals ≈ 718 bits; 760 bits leaves margin for the last digit.
    let c = Float::catalan(760, RoundingMode::Nearest);
    let s = digits(&c, 216);
    assert_eq!(&s[..210], &CATALAN[..210]);
    let g = Float::euler_gamma(200, RoundingMode::Nearest);
    let s = digits(&g, 50);
    assert_eq!(&s[..48], &GAMMA[..48]);
}

#[test]
#[ignore = "slow: old O(n²) reference at up to 20 kbit"]
fn bsplit_matches_series_reference_high() {
    for p in (4001..=20000u64).step_by(1009) {
        for &m in &MODES {
            assert_eq!(
                Float::catalan_bsplit(p, m),
                Float::catalan_series_reference(p, m),
                "catalan p={p} {m:?}"
            );
            assert_eq!(
                Float::euler_gamma_bsplit(p, m),
                Float::euler_gamma_series_reference(p, m),
                "gamma p={p} {m:?}"
            );
        }
    }
}

#[test]
#[ignore = "slow: 64 kbit self-consistency"]
fn bsplit_self_consistent_very_high() {
    // A 65536-bit value rounded down must equal the directly computed lower
    // precision value.
    let m = RoundingMode::Nearest;
    let hi_c = Float::catalan(65536, m);
    let hi_g = Float::euler_gamma(65536, m);
    for p in [30000u64, 50000, 65000] {
        assert_eq!(hi_c.round(p, m), Float::catalan(p, m), "catalan {p}");
        assert_eq!(hi_g.round(p, m), Float::euler_gamma(p, m), "gamma {p}");
    }
    // And the series references agree at a high precision.
    assert_eq!(
        Float::catalan(30011, m),
        Float::catalan_series_reference(30011, m)
    );
}
