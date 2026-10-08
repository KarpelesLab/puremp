//! Integration tests for rational point search and sums of two rational cubes.
//!
//! Facts used (classical unless stated):
//! - `6 = (17/21)³ + (37/21)³` (Dudeney), `7 = 2³ − 1³`, `9 = 2³ + 1³`,
//!   `13 = (7/3)³ + (2/3)³`, `19 = 3³ − 2³`;
//! - `1, 2` have only the trivial representations (Euler/Fermat: `E_n` has rank 0);
//! - `3, 4, 5` are not sums of two rational cubes (Euler for 3; Sylvester and
//!   Selmer; `E_n` has rank 0), so a search must find nothing;
//! - every prime `ℓ ≡ 4, 7, 8 (mod 9)` is a sum of two rational cubes (OpenAI,
//!   *The Selmer converse for elliptic curves at every prime*, preprint
//!   September 24, 2026, §10 — not refereed).

use puremp::cubes::{
    cube_sum_curve, cube_sum_to_curve, curve_to_cube_sum, search_quartic_points, sum_of_two_cubes,
};
use puremp::rational::Rational;
use puremp::{EllipticCurve, Int};

fn q(n: i64, d: i64) -> Rational {
    Rational::new(Int::from(n), Int::from(d))
}

fn check(n: i64, effort: u64) -> Option<(Rational, Rational)> {
    let r = sum_of_two_cubes(&Int::from(n), effort);
    if let Some((x, y)) = &r {
        assert_eq!(x.pow(3) + y.pow(3), Rational::from(n), "n = {n}");
    }
    r
}

fn is_prime(n: i64) -> bool {
    n >= 2 && (2..n).take_while(|d| d * d <= n).all(|d| n % d != 0)
}

#[test]
fn known_identities() {
    for (n, x, y) in [
        (6, q(17, 21), q(37, 21)),
        (7, q(2, 1), q(-1, 1)),
        (9, q(2, 1), q(1, 1)),
        (13, q(7, 3), q(2, 3)),
        (19, q(3, 1), q(-2, 1)),
    ] {
        assert_eq!(x.pow(3) + y.pow(3), Rational::from(n));
        assert!(check(n, 1000).is_some(), "n = {n}");
    }
}

#[test]
fn trivial_and_degenerate() {
    assert_eq!(check(0, 10), Some((Rational::from(0), Rational::from(0))));
    assert_eq!(check(1, 10), Some((Rational::from(1), Rational::from(0))));
    assert_eq!(check(2, 10), Some((Rational::from(1), Rational::from(1))));
    assert_eq!(
        check(-2, 10),
        Some((Rational::from(-1), Rational::from(-1)))
    );
    // Cube factors: 16 = 2³·2, 56 = 2³·7, −189 = −3³·7, 1000 = 10³.
    assert_eq!(check(16, 10), Some((Rational::from(2), Rational::from(2))));
    assert!(check(56, 100).is_some());
    assert!(check(-189, 100).is_some());
    assert_eq!(
        check(1000, 10),
        Some((Rational::from(10), Rational::from(0)))
    );
    assert!(check(-7, 100).is_some());
}

#[test]
fn rank_zero_values_find_nothing() {
    // Consistent with (not a proof of) 3, 4, 5 not being sums of two cubes.
    for n in [3, 4, 5, -3, 24, 32] {
        assert_eq!(check(n, 512), None, "n = {n}");
    }
}

#[test]
fn small_n_in_range() {
    // Cube-free n < 50 that are sums of two rational cubes (all found quickly).
    for n in [
        6, 7, 9, 12, 13, 15, 17, 19, 20, 22, 26, 28, 30, 31, 33, 34, 35, 37, 42, 43,
    ] {
        assert!(check(n, 256).is_some(), "n = {n}");
    }
    // Several split primes: 91 = 3³ + 4³ = 7·13, 1729 = 1³ + 12³ = 9³ + 10³.
    assert!(check(91, 64).is_some());
    assert!(check(1729, 64).is_some());
    assert!(check(-1729 * 8, 64).is_some());
}

#[test]
fn sylvester_primes() {
    let mut missing = Vec::new();
    // Every prime ℓ ≡ 4, 7, 8 (mod 9) below 200 is found with |u|, |v| ≤ 256;
    // the largest solutions are for 157, 179 and 193 (numerators ~10⁸), all found
    // by the Eisenstein descent (outside the reach of a direct search).
    for l in (5..200).filter(|&l| is_prime(l) && matches!(l % 9, 4 | 7 | 8)) {
        if check(l, 256).is_none() {
            missing.push(l);
        }
    }
    assert!(missing.is_empty(), "not found: {missing:?}");
}

#[test]
fn mordell_round_trip() {
    let n = Int::from(6);
    let (x, y) = (q(17, 21), q(37, 21));
    let p = cube_sum_to_curve(&n, &x, &y).unwrap();
    assert_eq!(
        p.coordinates().unwrap(),
        (&Rational::from(28), &Rational::from(-80))
    );
    assert_eq!(curve_to_cube_sum(&n, &p).unwrap(), (x.clone(), y.clone()));
    // Group law on E_6 produces new solutions.
    for k in 2..5 {
        let pk = p.scalar_mul(&Int::from(k));
        let (u, v) = curve_to_cube_sum(&n, &pk).unwrap();
        assert_eq!(u.pow(3) + v.pow(3), Rational::from(6));
        assert_eq!(cube_sum_to_curve(&n, &u, &v).unwrap(), pk);
    }
    // Torsion for n = 1 (order 3) and n = 2 (order 2).
    let p1 = cube_sum_to_curve(&Int::from(1), &Rational::from(1), &Rational::from(0)).unwrap();
    assert!(p1.is_torsion());
    assert!(p1.scalar_mul(&Int::from(3)).is_infinity());
    let p2 = cube_sum_to_curve(&Int::from(2), &Rational::from(1), &Rational::from(1)).unwrap();
    assert!(p2.is_torsion() && p2.double().is_infinity());
    assert!(!p.is_torsion());
    // Not a solution / infinity.
    assert!(cube_sum_to_curve(&n, &x, &x).is_none());
    assert!(curve_to_cube_sum(&n, &p.curve().identity()).is_none());
    assert!(cube_sum_curve(&Int::ZERO).is_none());
}

#[test]
fn weierstrass_search() {
    // E_6 : Y² = X³ − 15552 has generator (28, 80).
    let e6 = cube_sum_curve(&Int::from(6)).unwrap();
    let pts = e6.search_points(1000);
    assert!(!pts.is_empty());
    assert_eq!(
        pts[0].coordinates().unwrap(),
        (&Rational::from(28), &Rational::from(80))
    );
    for p in &pts {
        assert!(p.is_on_curve() && !p.is_torsion());
        let (u, v) = curve_to_cube_sum(&Int::from(6), p).unwrap();
        assert_eq!(u.pow(3) + v.pow(3), Rational::from(6));
    }
    // Rank 0 curves: E_1, E_2 have only torsion; y² = x³ − 2 has (3, 5).
    assert!(
        cube_sum_curve(&Int::from(1))
            .unwrap()
            .search_points(2000)
            .is_empty()
    );
    assert!(
        cube_sum_curve(&Int::from(2))
            .unwrap()
            .search_points(2000)
            .is_empty()
    );
    let c = EllipticCurve::new(Rational::from(0), Rational::from(-2)).unwrap();
    let pts = c.search_points(2000);
    assert_eq!(
        pts[0].coordinates().unwrap(),
        (&Rational::from(3), &Rational::from(5))
    );
    // y² = x³ − 2 also has (129/100, 383/1000).
    assert!(pts.iter().any(|p| p.x() == Some(&q(129, 100))));
    // Non-integral coefficients: y² = x³ + x/16 + 1/64 — scaled by e = 16.
    let c = EllipticCurve::new(q(1, 16), q(1, 64)).unwrap();
    for p in c.search_points(5000) {
        assert!(p.is_on_curve());
    }
    // Congruent-number curve y² = x³ − 36x (n = 6, rank 1) via the generic API.
    let c = EllipticCurve::new(Rational::from(-36), Rational::from(0)).unwrap();
    let pts = c.search_points(100);
    assert!(pts.iter().any(|p| p.x() == Some(&Rational::from(-3))));
    assert!(pts.iter().all(|p| !p.is_torsion()));
}

#[test]
fn quartic_search() {
    // w² = u⁴ + v⁴ only has trivial solutions (Fermat).
    let g = [Int::ONE, Int::ZERO, Int::ZERO, Int::ZERO, Int::ONE];
    let sols = search_quartic_points(&g, 60);
    for (u, v, _) in &sols {
        assert!(u.is_zero() || v.is_zero());
    }
    // w² = 2u⁴ − v⁴ has (1, 1, 1) and (13, 1, 239).
    let g = [Int::from(-1), Int::ZERO, Int::ZERO, Int::ZERO, Int::from(2)];
    let sols = search_quartic_points(&g, 200);
    assert!(sols.contains(&(Int::ONE, Int::ONE, Int::ONE)));
    assert!(sols.contains(&(Int::from(13), Int::ONE, Int::from(239))));
    for (u, v, w) in &sols {
        let val = Int::from(2).mul(&u.pow(4)).sub(&v.pow(4));
        assert_eq!(w.square(), val);
    }
    // Coefficients too large for the i128 path: w² = u⁴ + 10⁴⁰·v⁴.
    let big = Int::from(10).pow(40);
    let g = [big, Int::ZERO, Int::ZERO, Int::ZERO, Int::ONE];
    let sols = search_quartic_points(&g, 5);
    assert!(sols.contains(&(Int::ONE, Int::ZERO, Int::ONE)));
    assert!(sols.contains(&(Int::ZERO, Int::ONE, Int::from(10).pow(20))));
    // Cubic y² = 4x³ + 1 via g = v·(4u³ + v³): contains x = 0, y = 1.
    let g = [Int::ONE, Int::ZERO, Int::ZERO, Int::from(4), Int::ZERO];
    let sols = search_quartic_points(&g, 50);
    for (u, v, w) in &sols {
        let val = v.mul(&Int::from(4).mul(&u.pow(3)).add(&v.pow(3)));
        assert_eq!(w.square(), val);
    }
    assert!(sols.contains(&(Int::from(0), Int::ONE, Int::ONE)));
}

#[test]
#[ignore]
fn explore_primes() {
    let effort: u64 = std::env::var("EFFORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);
    let bound: i64 = std::env::var("BOUND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    for l in (5..bound).filter(|&l| is_prime(l) && matches!(l % 9, 4 | 7 | 8)) {
        let t = std::time::Instant::now();
        let r = check(l, effort);
        println!(
            "{l}: {:?} {:?}",
            r.map(|(x, y)| format!("{x} {y}")),
            t.elapsed()
        );
    }
}
