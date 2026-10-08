//! Integration tests for certified ranks by 2-isogeny descent.
//!
//! Rank facts used (all classical):
//! - congruent-number curves `y² = x³ − n²x`: `n = 1, 2, 3, 4, 10, 11` are not
//!   congruent (rank 0; Fermat for `n = 1, 2`, Genocchi for primes `≡ 3 mod 8`);
//!   `n = 5, 6, 7, 13, 14` have rank 1 (explicit points + descent bound);
//! - `n = 34, 41`: the descent bound is 2 and two independent points are found,
//!   so the rank 2 is self-certified by the code (and agrees with the tables);
//! - `n = 157` (Zagier's congruent number with a huge generator) and primes
//!   `≡ 5, 7 mod 8` have rank 1 (Heegner points / Monsky); there the code uses
//!   the 2-converse path because no point is found by the small search;
//! - `y² = x³ + 17x`: the isogenous quartic `u⁴ − 17v⁴ = 2w²` (Lind–Reichardt)
//!   is everywhere locally soluble but has no rational point, so the Selmer
//!   bound exceeds the rank and the result must be `Bounds`.

use puremp::elliptic_descent::{
    certificate_from_descent, quartic_is_locally_soluble, quartic_is_real_soluble,
};
use puremp::rational::Rational;
use puremp::{EllipticCurve, Int, RankCertificate, RankMethod, TwoIsogenyCurve};

fn congruent(n: i64) -> TwoIsogenyCurve {
    TwoIsogenyCurve::new(Int::ZERO, Int::from(-n * n)).unwrap()
}

fn proven(rank: u32, method: RankMethod) -> RankCertificate {
    RankCertificate::Proven { rank, method }
}

#[test]
fn rejects_singular() {
    assert!(TwoIsogenyCurve::new(Int::from(3), Int::ZERO).is_none());
    assert!(TwoIsogenyCurve::new(Int::from(4), Int::from(4)).is_none());
    let c = TwoIsogenyCurve::new(Int::from(1), Int::from(-6)).unwrap();
    let iso = c.isogenous();
    assert_eq!(
        (iso.a().clone(), iso.b().clone()),
        (Int::from(-2), Int::from(25))
    );
}

#[test]
fn congruent_rank_zero() {
    for n in [1, 2, 3, 4, 10, 11] {
        assert_eq!(
            congruent(n).certified_rank(),
            proven(0, RankMethod::SelmerBound),
            "n = {n}"
        );
    }
}

#[test]
fn congruent_rank_one() {
    for n in [5, 6, 7, 13, 14] {
        assert_eq!(
            congruent(n).certified_rank(),
            proven(1, RankMethod::PointsMeetSelmerBound),
            "n = {n}"
        );
    }
}

#[test]
fn congruent_rank_two() {
    for n in [34, 41] {
        let d = congruent(n).two_isogeny_descent(400);
        assert_eq!((d.lower, d.upper), (2, 2), "n = {n}");
        assert_eq!(
            certificate_from_descent(&d),
            proven(2, RankMethod::PointsMeetSelmerBound)
        );
    }
}

#[test]
fn two_converse_path() {
    // n = 157: Selmer bound 1, generator far beyond the search box.
    let d = congruent(157).two_isogeny_descent(100);
    assert_eq!((d.lower, d.upper), (0, 1));
    assert!(d.curve.exact && d.isogenous.exact);
    assert_eq!(
        certificate_from_descent(&d),
        proven(1, RankMethod::TwoConverse)
    );
    // n = 5 without a point search falls back to the 2-converse too; with the
    // search the classical certificate takes precedence.
    assert_eq!(
        congruent(5).certified_rank_with_search_bound(0),
        proven(1, RankMethod::TwoConverse)
    );
    assert_eq!(
        congruent(5).certified_rank_with_search_bound(10),
        proven(1, RankMethod::PointsMeetSelmerBound)
    );
}

#[test]
fn large_prime_congruent_curves() {
    // Exercises the scan-free (Cantor–Zassenhaus / Weil) local tests.
    let p3 = Int::from(1_000_003); // ≡ 3 mod 8: not congruent
    let c3 = TwoIsogenyCurve::new(Int::ZERO, p3.square().neg()).unwrap();
    assert_eq!(
        c3.certified_rank_with_search_bound(50),
        proven(0, RankMethod::SelmerBound)
    );
    for p in [1_000_037i64, 1_000_039] {
        // ≡ 5, 7 mod 8: rank 1.
        let c = TwoIsogenyCurve::new(Int::ZERO, Int::from(p).square().neg()).unwrap();
        assert_eq!(
            c.certified_rank_with_search_bound(50).rank(),
            Some(1),
            "p = {p}"
        );
    }
}

#[test]
fn textbook_curves() {
    let curve = |a: i64, b: i64| TwoIsogenyCurve::new(Int::from(a), Int::from(b)).unwrap();
    // y² = x³ + x: E(ℚ) = {O, (0, 0)}.
    assert_eq!(
        curve(0, 1).certified_rank(),
        proven(0, RankMethod::SelmerBound)
    );
    // y² = x³ − 2x: rank 1, (−1, 1).
    assert_eq!(
        curve(0, -2).certified_rank(),
        proven(1, RankMethod::PointsMeetSelmerBound)
    );
    // y² = x³ − 17x: (−1, 4) and (−4, 2) are independent; bound 2.
    assert_eq!(
        curve(0, -17).certified_rank(),
        proven(2, RankMethod::PointsMeetSelmerBound)
    );
}

#[test]
fn sha_obstruction_gives_bounds() {
    let c = TwoIsogenyCurve::new(Int::ZERO, Int::from(17)).unwrap();
    let d = c.two_isogeny_descent(400);
    // The Lind–Reichardt class 2 is in S^(φ)(E/ℚ) (computed on E': y² = x³ − 68x)
    // but is not in the image of a found point.
    assert!(d.isogenous.selmer.contains(&Int::from(2)));
    assert!(!d.isogenous.image_basis.contains(&Int::from(2)));
    let cert = certificate_from_descent(&d);
    assert!(
        matches!(cert, RankCertificate::Bounds { lower: 0, upper: 2 }),
        "{cert:?}"
    );
    // Same for the isogenous curve.
    let iso = c.isogenous().certified_rank();
    assert!(matches!(iso, RankCertificate::Bounds { .. }), "{iso:?}");
}

#[test]
fn lind_reichardt_quartic_is_everywhere_locally_soluble() {
    // On E': 2u⁴ − 34v⁴ = w², i.e. u⁴ − 17v⁴ = 2(w/2)².
    let (d, a, e) = (Int::from(2), Int::ZERO, Int::from(-34));
    assert!(quartic_is_real_soluble(&d, &a, &e));
    for p in [2, 17] {
        assert_eq!(
            quartic_is_locally_soluble(&d, &a, &e, &Int::from(p)),
            Some(true)
        );
    }
}

fn padic_square(mut n: i128, p: i128) -> bool {
    if n == 0 {
        return true;
    }
    let mut v = 0;
    while n % p == 0 {
        n /= p;
        v += 1;
    }
    if v % 2 == 1 {
        return false;
    }
    if p == 2 {
        return n.rem_euclid(8) == 1;
    }
    let r = n.rem_euclid(p);
    (1..p).any(|x| x * x % p == r)
}

/// Some `(u, v) ∈ [0, pᵏ)²`, not both divisible by `p`, with `d u⁴ + a u²v² + e v⁴`
/// a square in `ℚ_p`.
fn brute_soluble(d: i128, a: i128, e: i128, p: i128, k: u32) -> bool {
    let m = p.pow(k);
    (0..m).any(|u| {
        (0..m).any(|v| {
            if u % p == 0 && v % p == 0 {
                return false;
            }
            let (u2, v2) = (u * u, v * v);
            padic_square(d * u2 * u2 + a * u2 * v2 + e * v2 * v2, p)
        })
    })
}

#[test]
fn local_solubility_matches_brute_force() {
    let mut cases = 0;
    for (p, k) in [(2i128, 6u32), (3, 4), (5, 3), (7, 2)] {
        for d in -6i128..=6 {
            for e in -6i128..=6 {
                for a in -3i128..=3 {
                    if d == 0 || e == 0 || a * a == 4 * d * e {
                        continue;
                    }
                    let fast = quartic_is_locally_soluble(
                        &Int::from(d as i64),
                        &Int::from(a as i64),
                        &Int::from(e as i64),
                        &Int::from(p as i64),
                    );
                    assert_eq!(
                        fast,
                        Some(brute_soluble(d, a, e, p, k)),
                        "p={p} d={d} a={a} e={e}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 3000);
}

#[test]
fn sweep_bounds_are_consistent() {
    for a in -6i64..=6 {
        for b in -12i64..=12 {
            let Some(c) = TwoIsogenyCurve::new(Int::from(a), Int::from(b)) else {
                continue;
            };
            let d = c.two_isogeny_descent(60);
            assert!(d.lower <= d.upper, "a={a} b={b}");
            assert!(d.curve.exact && d.isogenous.exact);
            // E and E' have the same rank.
            let di = c.isogenous().two_isogeny_descent(60);
            assert_eq!(d.upper, di.upper, "a={a} b={b}");
        }
    }
}

#[test]
fn short_weierstrass_conversion() {
    let q = |n: i64, d: i64| Rational::new(Int::from(n), Int::from(d));
    // y² = x³ − 25x.
    let e = EllipticCurve::new(q(-25, 1), q(0, 1)).unwrap();
    assert_eq!(
        e.certified_rank(),
        Some(proven(1, RankMethod::PointsMeetSelmerBound))
    );
    // (x − 1/2)(x² + x/2 + 3) = x³ + (11/4)x − 3/2: rational root 1/2.
    let e = EllipticCurve::new(q(11, 4), q(-3, 2)).unwrap();
    let (c, map) = TwoIsogenyCurve::from_short_weierstrass(&e).unwrap();
    assert_eq!(map.x_shift, q(1, 2));
    // The 2-torsion point (1/2, 0) maps to (0, 0); a generic point stays on the curve.
    let (x0, y0) = map.map_point(&q(1, 2), &q(0, 1));
    assert!(x0.is_zero() && y0.is_zero());
    let ca = Rational::from_integer(c.a().clone());
    let cb = Rational::from_integer(c.b().clone());
    // x = 1: y² = 1 + 11/4 − 3/2 = 9/4.
    let (x, y) = map.map_point(&q(1, 1), &q(3, 2));
    assert_eq!(
        y.mul(&y),
        x.mul(&x).mul(&x).add(&ca.mul(&x).mul(&x)).add(&cb.mul(&x))
    );
    assert!(e.certified_rank().is_some());
    // Round trip through the short model keeps the curve's rank.
    let back = congruent(6).to_short_weierstrass();
    assert_eq!(
        back.certified_rank(),
        Some(proven(1, RankMethod::PointsMeetSelmerBound))
    );
    // No rational 2-torsion.
    let e = EllipticCurve::new(q(1, 1), q(1, 1)).unwrap();
    assert!(TwoIsogenyCurve::from_short_weierstrass(&e).is_none());
    assert_eq!(e.certified_rank(), None);
}
