//! Integration tests for elliptic curves `y² = x³ + a·x + b`.
//!
//! Over `GF(17)` the standard textbook curve `y² = x³ + 2x + 2` has group order
//! 19 (a prime, so the group is cyclic and every non-identity point generates
//! it). These tests exercise the group axioms, point orders, point counting
//! (checked against the known 19 and the Hasse bound), `is_on_curve`, the
//! singular-curve rejection, `scalar_mul`, and — over `ℚ` — the doublings of the
//! rational point `(3, 5)` on `y² = x³ − 2`.

use puremp::mod_int::ModInt;
use puremp::rational::Rational;
use puremp::{EllipticCurve, Int, Point};

/// The GF(17) curve `y² = x³ + 2x + 2`.
fn curve_gf17() -> EllipticCurve<ModInt> {
    let p = Int::from(17);
    let a = ModInt::new(Int::from(2), p.clone());
    let b = ModInt::new(Int::from(2), p);
    EllipticCurve::new(a, b).expect("non-singular")
}

fn m17(v: i64) -> ModInt {
    ModInt::new(Int::from(v), Int::from(17))
}

/// Collects every point of the curve (identity plus all affine points) by brute
/// force over the base field.
fn all_points(curve: &EllipticCurve<ModInt>) -> Vec<Point<ModInt>> {
    let mut pts = vec![curve.identity()];
    for x in 0..17i64 {
        for y in 0..17i64 {
            if let Some(p) = curve.point(m17(x), m17(y)) {
                pts.push(p);
            }
        }
    }
    pts
}

#[test]
fn identity_and_inverse() {
    let curve = curve_gf17();
    let o = curve.identity();
    assert!(o.is_infinity());

    let p = curve.point(m17(5), m17(1)).expect("(5,1) on curve");
    // P + O = P and O + P = P.
    assert_eq!(&p + &o, p);
    assert_eq!(&o + &p, p);
    // P + (-P) = O.
    let neg_p = -&p;
    assert_eq!(neg_p.coordinates().unwrap().0, &m17(5));
    assert_eq!(neg_p.coordinates().unwrap().1, &m17(16)); // -1 mod 17
    assert!((&p + &neg_p).is_infinity());
    // -O = O.
    assert!((-&o).is_infinity());
}

#[test]
fn on_curve_accepts_and_rejects() {
    let curve = curve_gf17();
    // (5,1): 1 = 125 + 10 + 2 = 137 = 8*17 + 1 ≡ 1 (mod 17). On curve.
    assert!(curve.point(m17(5), m17(1)).is_some());
    // (5,2): 4 ≠ 1, off curve.
    assert!(curve.point(m17(5), m17(2)).is_none());
    // The identity is always "on curve".
    assert!(curve.identity().is_on_curve());
}

#[test]
fn singular_curve_rejected() {
    // Δ = -16(4a³ + 27b²). For a = 0, b = 0 the discriminant vanishes.
    assert!(EllipticCurve::new(m17(0), m17(0)).is_none());
    // Sweep all (a,b) mod 17: the constructor must reject exactly the singular
    // pairs (Δ = 0) and never accept one.
    let mut singular_rejections = 0;
    for a in 0..17i64 {
        for b in 0..17i64 {
            match EllipticCurve::new(m17(a), m17(b)) {
                None => singular_rejections += 1,
                Some(c) => assert!(
                    !c.discriminant().is_zero(),
                    "accepted a singular curve (a={a}, b={b})"
                ),
            }
        }
    }
    assert!(
        singular_rejections > 0,
        "expected some singular (a,b) to be rejected"
    );
}

#[test]
fn discriminant_and_j_invariant() {
    let curve = curve_gf17();
    assert!(!curve.discriminant().is_zero());
    // j = 0 when a = 0 (curve y² = x³ + b).
    let c_a0 = EllipticCurve::new(m17(0), m17(1)).unwrap();
    assert_eq!(c_a0.j_invariant(), m17(0));
    // j = 1728 when b = 0 (curve y² = x³ + a·x). 1728 mod 17 = 1728 - 101*17
    // = 1728 - 1717 = 11.
    let c_b0 = EllipticCurve::new(m17(1), m17(0)).unwrap();
    assert_eq!(c_b0.j_invariant(), m17(11));
}

#[test]
fn group_is_commutative_and_associative() {
    let curve = curve_gf17();
    let pts = all_points(&curve);
    // Commutativity on a sample grid.
    for p in pts.iter().take(6) {
        for q in pts.iter().take(6) {
            assert_eq!(p + q, q + p, "commutativity failed");
        }
    }
    // Associativity (P+Q)+R = P+(Q+R) on a sample.
    for p in pts.iter().take(5) {
        for q in pts.iter().take(5) {
            for r in pts.iter().take(5) {
                let left = &(p + q) + r;
                let right = p + &(q + r);
                assert_eq!(left, right, "associativity failed");
            }
        }
    }
}

#[test]
fn curve_order_matches_known_and_hasse() {
    let curve = curve_gf17();
    let order = curve.curve_order();
    // Known: this curve has 19 points (incl. O).
    assert_eq!(order, Int::from(19));
    // Cross-check against the brute-force enumeration.
    assert_eq!(Int::from(all_points(&curve).len() as i64), order);
    // Hasse: |#E - (p+1)| ≤ 2√p. p = 17, p+1 = 18, |19-18| = 1 ≤ 2·√17 ≈ 8.24.
    let p = Int::from(17);
    let diff = (&order - &(&p + Int::from(1))).abs();
    // 2√p: compare diff² ≤ 4p to avoid floating point.
    assert!((&diff * &diff) <= (Int::from(4) * &p));
}

#[test]
fn point_order_divides_group_order() {
    let curve = curve_gf17();
    // The group order is the prime 19, so every non-identity point has order 19.
    let p = curve.point(m17(5), m17(1)).expect("(5,1) on curve");
    let n = curve.order_of_point(&p);
    assert_eq!(n, Int::from(19));
    // n·P = O.
    assert!(p.scalar_mul(&n).is_infinity());
    // m·P ≠ O for 0 < m < n.
    let mut m = Int::from(1);
    while m < n {
        assert!(
            !p.scalar_mul(&m).is_infinity(),
            "m·P vanished early at m = {m}"
        );
        m = &m + Int::from(1);
    }
    // The identity has order 1.
    assert_eq!(curve.order_of_point(&curve.identity()), Int::from(1));
}

#[test]
fn scalar_mul_matches_repeated_addition() {
    let curve = curve_gf17();
    let p = curve.point(m17(5), m17(1)).expect("(5,1) on curve");
    // k·P by repeated addition equals scalar_mul for small k.
    let mut acc = curve.identity();
    for k in 0..25i64 {
        assert_eq!(
            p.scalar_mul(&Int::from(k)),
            acc,
            "scalar_mul mismatch k={k}"
        );
        acc = &acc + &p;
    }
    // (-k)·P = -(k·P).
    for k in 1..10i64 {
        let kp = p.scalar_mul(&Int::from(k));
        let neg_kp = p.scalar_mul(&Int::from(-k));
        assert_eq!(neg_kp, -&kp, "(-k)P != -(kP) at k={k}");
    }
    // Distributivity k·P + m·P = (k+m)·P.
    for k in 0..8i64 {
        for m in 0..8i64 {
            let lhs = &p.scalar_mul(&Int::from(k)) + &p.scalar_mul(&Int::from(m));
            let rhs = p.scalar_mul(&Int::from(k + m));
            assert_eq!(lhs, rhs, "distributivity failed k={k} m={m}");
        }
    }
}

#[test]
fn point_from_x_recovers_y() {
    let curve = curve_gf17();
    // x = 5 gives a valid point (y² ≡ 1).
    let p = curve.point_from_x(&m17(5)).expect("x=5 recoverable");
    assert!(p.is_on_curve());
    assert_eq!(p.x().unwrap(), &m17(5));
    // The recovered point squares back correctly regardless of which root.
    let y = p.y().unwrap().clone();
    assert_eq!(y.clone() * y, m17(1));
    // Some x has no point (rhs a non-residue) — at least one must exist since
    // there are 17 x-values but only (19-1)/2 = 9 distinct x with points.
    let mut none_seen = false;
    for x in 0..17i64 {
        if curve.point_from_x(&m17(x)).is_none() {
            none_seen = true;
        }
    }
    assert!(none_seen, "expected some x with no point");
}

#[test]
fn different_curve_addition_panics() {
    let c1 = curve_gf17();
    let c2 = EllipticCurve::new(m17(1), m17(6)).unwrap();
    let p1 = c1.point(m17(5), m17(1)).unwrap();
    let p2 = c2.identity();
    let res = std::panic::catch_unwind(|| Point::add(&p1, &p2));
    assert!(
        res.is_err(),
        "adding points on different curves should panic"
    );
}

// --- ℚ curves ---

fn q(n: i64) -> Rational {
    Rational::from(n)
}

#[test]
fn rational_curve_doublings() {
    // y² = x³ - 2, with the rational point (3, 5): 25 = 27 - 2. ✓
    let curve = EllipticCurve::new(q(0), q(-2)).expect("non-singular over ℚ");
    let p = curve.point(q(3), q(5)).expect("(3,5) on curve");
    assert!(p.is_on_curve());

    // 2P: λ = 3x²/(2y) = 27/10; x3 = λ² - 2x = 729/100 - 6 = 129/100.
    let p2 = p.double();
    assert!(p2.is_on_curve());
    assert_eq!(
        p2.x().unwrap(),
        &Rational::new(Int::from(129), Int::from(100))
    );

    // 3P and 4P must also stay on the curve.
    let p3 = &p2 + &p;
    assert!(p3.is_on_curve());
    let p4 = p2.double();
    assert!(p4.is_on_curve());

    // j = 0 for a = 0.
    assert_eq!(curve.j_invariant(), q(0));

    // scalar_mul agrees with repeated addition over ℚ too.
    assert_eq!(p.scalar_mul(&Int::from(3)), p3);
    // (-2)P = -(2P).
    assert_eq!(p.scalar_mul(&Int::from(-2)), -&p2);
}

// --- Differential: Jacobian ladder vs. the affine double-and-add reference ---

/// Reference `k·P` using only the public affine [`Point::double`]/[`Point::add`]
/// (the same left-to-right binary ladder the old `scalar_mul` used), against
/// which the new inversion-free Jacobian `scalar_mul` must be bit-identical.
fn affine_scalar_mul<F: puremp::ring::Field>(p: &Point<F>, k: &Int) -> Point<F> {
    if k.is_zero() || p.is_infinity() {
        return p.curve().identity();
    }
    let mag = k.abs();
    let mut result = p.curve().identity();
    let mut i = mag.bit_len();
    while i > 0 {
        i -= 1;
        result = result.double();
        if mag.bit(i) {
            result = result.add(p);
        }
    }
    if k.is_negative() { -&result } else { result }
}

#[test]
fn jacobian_matches_affine_gf_p() {
    // A handful of small primes, each with a couple of curves and base points.
    for &pv in &[17i64, 23, 101, 1009, 7919] {
        let p = Int::from(pv);
        let mk = |v: i64| ModInt::new(Int::from(v), p.clone());
        for &(av, bv) in &[(2i64, 2i64), (0, 7), (3, 5), (1, 1), (5, 0)] {
            let curve = match EllipticCurve::new(mk(av), mk(bv)) {
                Some(c) => c,
                None => continue, // singular pair, skip
            };
            // Collect a few base points by scanning x-coordinates.
            let mut bases = vec![curve.identity()];
            for x in 0..pv.min(60) {
                if let Some(pt) = curve.point_from_x(&mk(x)) {
                    bases.push(pt);
                }
                if bases.len() >= 5 {
                    break;
                }
            }
            let order = curve.curve_order();
            for base in &bases {
                // Deterministic pseudo-random scalars plus structural values.
                let mut ks = vec![
                    Int::ZERO,
                    Int::ONE,
                    Int::from(2),
                    order.clone(),     // hits infinity for base
                    &order + Int::ONE, // ≡ base
                    -&order,           // hits infinity
                    Int::from(-1),
                    Int::from(-7),
                ];
                let mut s: u64 = 0x1234_5678 ^ (pv as u64) ^ ((av as u64) << 8);
                for _ in 0..12 {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let mut k = Int::from((s % 100_000) as i64);
                    if s & 1 == 0 {
                        k = -k;
                    }
                    ks.push(k);
                }
                for k in &ks {
                    let jac = base.scalar_mul(k);
                    let aff = affine_scalar_mul(base, k);
                    assert_eq!(
                        jac, aff,
                        "GF({pv}) a={av} b={bv} base={base} k={k}: Jacobian != affine"
                    );
                    assert!(jac.is_on_curve());
                }
            }
        }
    }
}

#[test]
fn jacobian_matches_affine_two_torsion() {
    // y² = x³ - x over GF(101) has full 2-torsion: x ∈ {0, 1, 100} give y = 0.
    let p = Int::from(101);
    let mk = |v: i64| ModInt::new(Int::from(v), p.clone());
    let curve = EllipticCurve::new(mk(-1), mk(0)).expect("non-singular");
    for &x0 in &[0i64, 1, 100] {
        let t = curve.point(mk(x0), mk(0)).expect("2-torsion point");
        assert!(t.double().is_infinity(), "2-torsion should double to O");
        for k in -6..=6i64 {
            let ki = Int::from(k);
            assert_eq!(
                t.scalar_mul(&ki),
                affine_scalar_mul(&t, &ki),
                "2-torsion x={x0} k={k}"
            );
        }
    }
}

#[test]
fn jacobian_matches_affine_rationals() {
    // y² = x³ - 2 with rational point (3, 5).
    let curve = EllipticCurve::new(q(0), q(-2)).expect("non-singular over ℚ");
    let base = curve.point(q(3), q(5)).expect("(3,5) on curve");
    for k in -12..=12i64 {
        let ki = Int::from(k);
        let jac = base.scalar_mul(&ki);
        let aff = affine_scalar_mul(&base, &ki);
        assert_eq!(jac, aff, "ℚ curve k={k}: Jacobian != affine");
        assert!(jac.is_on_curve());
    }
    // A second curve: y² = x³ + x + 1 with (0, 1).
    let curve2 = EllipticCurve::new(q(1), q(1)).expect("non-singular over ℚ");
    let base2 = curve2.point(q(0), q(1)).expect("(0,1) on curve");
    for k in -8..=8i64 {
        let ki = Int::from(k);
        assert_eq!(
            base2.scalar_mul(&ki),
            affine_scalar_mul(&base2, &ki),
            "ℚ curve2 k={k}"
        );
    }
}

#[test]
fn order_of_point_unchanged_by_jacobian() {
    // order_of_point drives its checks through scalar_mul; confirm it still
    // agrees with the group order 19 on the standard GF(17) curve.
    let curve = curve_gf17();
    for base in all_points(&curve) {
        let n = curve.order_of_point(&base);
        assert!(base.scalar_mul(&n).is_infinity(), "n·P must vanish");
        // Divisor of 19: either 1 (identity) or 19.
        assert!(n == Int::from(1) || n == Int::from(19));
    }
}

/// Timing benchmark: `EllipticCurve::<ModInt>::scalar_mul` over prime fields of
/// several sizes. With `ModInt` Montgomery-resident for odd moduli this is the
/// end-to-end effect on the group law; compare against a Barrett-only build by
/// temporarily forcing the Barrett backend in `ModInt`'s ring setup.
#[test]
#[ignore = "timing benchmark; run with --release --ignored --nocapture"]
fn scalar_mul_bench() {
    use std::hint::black_box;
    use std::time::Instant;
    // A pseudo-random prime of about `bits` bits (deterministic).
    let prime = |bits: u32, seed: u64| {
        let mut s = seed;
        let words = bits.div_ceil(64) as usize;
        let two64 = Int::from_u64(1u64 << 32) * Int::from_u64(1u64 << 32); // 2^64
        let mut n = Int::ZERO;
        for i in 0..words {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let mut w = s;
            if i == words - 1 {
                w |= 1u64 << ((bits - 1) % 64); // force ~`bits` bits (top bit set)
            }
            n = &n * &two64 + Int::from_u64(w);
        }
        n.next_prime()
    };
    println!("== GF(p) scalar_mul: affine vs Jacobian ==");
    println!(
        "{:>6} {:>8} {:>12} {:>12} {:>8}",
        "bits", "reps", "affine(ms)", "jacob(ms)", "speedup"
    );
    for &bits in &[64u32, 256, 1024] {
        let p = prime(bits, 0xDEAD_BEEF ^ bits as u64);
        let mk = |v: i64| ModInt::new(Int::from(v), p.clone());
        let curve = EllipticCurve::new(mk(2), mk(3)).expect("non-singular");
        // Find a base point by scanning x-coordinates.
        let mut point = None;
        for x in 1i64..1000 {
            if let Some(pt) = curve.point_from_x(&mk(x)) {
                point = Some(pt);
                break;
            }
        }
        let point = point.expect("a base point exists");
        // A scalar of full field size.
        let k = &p - Int::from(3);
        let reps: usize = if bits <= 64 {
            2000
        } else if bits <= 256 {
            300
        } else {
            20
        };
        // Correctness sanity + differential: both paths agree, on the curve.
        assert_eq!(point.scalar_mul(&k), affine_scalar_mul(&point, &k));
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(affine_scalar_mul(&point, &k));
        }
        let aff_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let t1 = Instant::now();
        for _ in 0..reps {
            black_box(point.scalar_mul(&k));
        }
        let jac_ms = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!(
            "{bits:>6} {reps:>8} {aff_ms:>12.4} {jac_ms:>12.4} {:>7.2}x",
            aff_ms / jac_ms
        );
    }

    // Over ℚ the height of k·P grows ~ k²·h(P) (numerators/denominators roughly
    // double in bit-length per doubling), so scalars must stay small to finish;
    // even k ≈ 200 pushes the coordinates to tens of thousands of bits.
    println!("== ℚ scalar_mul: affine vs Jacobian (y² = x³ - 2, base (3,5)) ==");
    println!(
        "{:>6} {:>8} {:>12} {:>12} {:>8}",
        "k", "reps", "affine(ms)", "jacob(ms)", "speedup"
    );
    {
        let curve = EllipticCurve::new(Rational::from(0), Rational::from(-2)).expect("ok");
        let base = curve
            .point(Rational::from(3), Rational::from(5))
            .expect("(3,5)");
        for &kv in &[50i64, 100, 200] {
            let k = Int::from(kv);
            let reps: usize = if kv <= 50 { 40 } else { 15 };
            assert_eq!(base.scalar_mul(&k), affine_scalar_mul(&base, &k));
            let t0 = Instant::now();
            for _ in 0..reps {
                black_box(affine_scalar_mul(&base, &k));
            }
            let aff_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let t1 = Instant::now();
            for _ in 0..reps {
                black_box(base.scalar_mul(&k));
            }
            let jac_ms = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;
            println!(
                "{kv:>6} {reps:>8} {aff_ms:>12.4} {jac_ms:>12.4} {:>7.2}x",
                aff_ms / jac_ms
            );
        }
    }

    println!("== end-to-end curve_order over GF(p) (drives order_of_point) ==");
    {
        // order_of_point runs a scalar_mul per prime factor of the group order,
        // so its inner loop now uses the Jacobian ladder. Its cost is still
        // dominated by curve_order's naive O(p) Legendre scan (unchanged here),
        // so use a genuinely small prime so the scan finishes; the Jacobian win
        // shows in the GF(p) scalar_mul table above. (The `prime` helper above
        // rounds up to a 64-bit word, so it cannot make a small modulus.)
        let p = Int::from(500_009).next_prime(); // ~19-bit prime
        let mk = |v: i64| ModInt::new(Int::from(v), p.clone());
        let curve = EllipticCurve::new(mk(2), mk(3)).expect("non-singular");
        let mut point = None;
        for x in 1i64..1000 {
            if let Some(pt) = curve.point_from_x(&mk(x)) {
                point = Some(pt);
                break;
            }
        }
        let point = point.expect("base point");
        let t0 = Instant::now();
        let ord = black_box(curve.order_of_point(&point));
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!("order_of_point(p≈{p}) = {ord} in {ms:.2} ms");
    }
}

// ---------------------------------------------------------------------------
// Schoof's algorithm: `schoof_point_count` must exactly match the naive scan
// wherever the scan is feasible, and satisfy Hasse + `[#E]·P = O` for large p.
// ---------------------------------------------------------------------------

fn mkp(v: i64, p: &Int) -> ModInt {
    ModInt::new(Int::from(v), p.clone())
}

/// `|#E - (p+1)| <= 2√p`, tested squared to stay in integers.
fn hasse_holds(order: &Int, p: &Int) -> bool {
    let d = order - &(p + Int::from(1));
    (&d * &d) <= (Int::from(4) * p)
}

#[test]
fn schoof_matches_naive_small_primes() {
    // Exhaustive over the tiniest primes (every non-singular a, b), where the
    // odd-ℓ special cases and division polynomials get their sharpest workout:
    // Schoof must equal the O(p) scan on every curve.
    for &pv in &[5i64, 7, 11, 13, 17, 19, 23, 29, 31] {
        let p = Int::from(pv);
        for av in 0..pv {
            for bv in 0..pv {
                let curve = match EllipticCurve::new(mkp(av, &p), mkp(bv, &p)) {
                    Some(c) => c,
                    None => continue, // singular (Δ = 0)
                };
                let naive = curve.naive_curve_order();
                let schoof = curve.schoof_point_count();
                assert_eq!(
                    naive, schoof,
                    "GF({pv}) a={av} b={bv}: naive {naive} != schoof {schoof}"
                );
                assert!(hasse_holds(&schoof, &p));
            }
        }
    }
    // A few larger small primes with a curated spread of (a, b) shapes.
    for &pv in &[37i64, 41, 43, 97, 101, 251] {
        let p = Int::from(pv);
        for &(av, bv) in &[(0i64, 1), (1, 0), (1, 1), (2, 2), (3, 5), (5, 7), (7, 11)] {
            let curve = match EllipticCurve::new(mkp(av, &p), mkp(bv, &p)) {
                Some(c) => c,
                None => continue,
            };
            assert_eq!(
                curve.naive_curve_order(),
                curve.schoof_point_count(),
                "GF({pv}) a={av} b={bv}"
            );
        }
    }
}

#[test]
fn schoof_matches_naive_medium_primes() {
    // Larger primes (still feasible for the scan), a handful of curves each,
    // including t = 0 supersingular and t = 1 anomalous shapes.
    for &pv in &[1009i64, 7919, 10007] {
        let p = Int::from(pv);
        let mut checked = 0;
        for &(av, bv) in &[(2i64, 2), (0, 7), (3, 5), (1, 6), (7, 0), (1, 0), (0, 1)] {
            let curve = match EllipticCurve::new(mkp(av, &p), mkp(bv, &p)) {
                Some(c) => c,
                None => continue,
            };
            assert_eq!(
                curve.naive_curve_order(),
                curve.schoof_point_count(),
                "GF({pv}) a={av} b={bv}"
            );
            checked += 1;
        }
        assert!(checked > 0, "no curves checked for p={pv}");
    }
}

#[test]
fn schoof_supersingular_trace_zero() {
    // For p ≡ 3 (mod 4), y² = x³ + x is supersingular: t = 0, #E = p + 1.
    for &pv in &[7i64, 11, 19, 23, 31, 43, 100003] {
        if pv % 4 != 3 {
            continue;
        }
        let p = Int::from(pv);
        let curve = EllipticCurve::new(mkp(1, &p), mkp(0, &p)).unwrap();
        assert_eq!(curve.schoof_point_count(), &p + Int::from(1));
    }
}

#[test]
fn schoof_dispatch_through_point_count() {
    // point_count() dispatches to Schoof above the size threshold; the result
    // must satisfy Hasse and kill random points.
    let p = Int::from(6_700_417); // ~23-bit prime (Fermat factor), above 2^22
    assert!(p.bit_len() >= 22);
    let curve = EllipticCurve::new(mkp(3, &p), mkp(7, &p)).unwrap();
    let n = curve.point_count();
    assert!(hasse_holds(&n, &p));
    let mut found = 0;
    let mut xi = 1i64;
    while found < 3 && xi < 500 {
        if let Some(pt) = curve.point_from_x(&mkp(xi, &p)) {
            assert!(pt.scalar_mul(&n).is_infinity(), "[#E]·P != O at x={xi}");
            found += 1;
        }
        xi += 1;
    }
    assert!(found > 0);
}

#[test]
#[ignore = "slow: naive O(p) scan at p≈10^6; run with --release --ignored"]
fn schoof_matches_naive_up_to_million() {
    // Schoof == the O(p) scan at the top of the scan's feasible range.
    for &pv in &[100_003i64, 999_983, 6_700_417] {
        let p = Int::from(pv);
        for &(av, bv) in &[(2i64, 2), (0, 7), (3, 5), (1, 1)] {
            let curve = match EllipticCurve::new(mkp(av, &p), mkp(bv, &p)) {
                Some(c) => c,
                None => continue,
            };
            assert_eq!(
                curve.naive_curve_order(),
                curve.schoof_point_count(),
                "GF({pv}) a={av} b={bv}"
            );
        }
    }
}

#[test]
#[ignore = "slow: large-p Schoof; run with --release --ignored"]
fn schoof_large_primes_hasse_and_order() {
    // Cryptographic-ish primes far beyond the O(p) scan: verify the Hasse bound
    // and that [#E]·P = O for several points — a necessary correctness check.
    for s in ["1000000007", "1000000000039", "18446744073709551557"] {
        let p = s.parse::<Int>().unwrap();
        let curve = EllipticCurve::new(mkp(3, &p), mkp(7, &p)).unwrap();
        let n = curve.schoof_point_count();
        assert!(hasse_holds(&n, &p), "Hasse failed for p={s}: #E={n}");
        let mut found = 0;
        let mut xi = 1i64;
        while found < 3 && xi < 500 {
            if let Some(pt) = curve.point_from_x(&mkp(xi, &p)) {
                assert!(pt.scalar_mul(&n).is_infinity(), "[#E]·P != O p={s} x={xi}");
                found += 1;
            }
            xi += 1;
        }
        assert!(found > 0, "no points found for p={s}");
    }
}

// ---------------------------------------------------------------------------
// SEA (Elkies) point counting: `sea_point_count` / the large-p dispatch must
// match classical Schoof exactly where both are feasible, and satisfy Hasse +
// `[#E]·P = O` far beyond the reach of the naive scan.
// ---------------------------------------------------------------------------

/// Verifies `[#E]·P = O` for a few points recovered from small x-coordinates.
fn annihilates(curve: &EllipticCurve<ModInt>, n: &Int, p: &Int) {
    let mut found = 0;
    let mut xi = 1i64;
    while found < 3 && xi < 2000 {
        if let Some(pt) = curve.point_from_x(&mkp(xi, p)) {
            assert!(pt.scalar_mul(n).is_infinity(), "[#E]·P != O at x={xi}");
            found += 1;
        }
        xi += 1;
    }
    assert!(found > 0, "no points found");
}

/// SEA must equal classical Schoof over a spread of curve shapes at a given
/// prime (both routes are exact; SEA is the faster one for large `p`).
fn assert_sea_eq_schoof(p: &Int, shapes: &[(i64, i64)]) {
    for &(av, bv) in shapes {
        let curve = match EllipticCurve::new(mkp(av, p), mkp(bv, p)) {
            Some(c) => c,
            None => continue,
        };
        let sea = curve.sea_point_count();
        let schoof = curve.schoof_point_count();
        assert_eq!(
            sea, schoof,
            "p={p} a={av} b={bv}: SEA {sea} != Schoof {schoof}"
        );
        assert!(hasse_holds(&sea, p));
    }
}

#[test]
fn sea_matches_schoof_around_threshold() {
    // Fast always-run subset: a ~24-bit prime, several Elkies-rich and Atkin-rich
    // curve shapes. SEA must equal Schoof.
    let p = "16777259".parse::<Int>().unwrap(); // ~24-bit
    assert_sea_eq_schoof(&p, &[(1, 1), (3, 5), (0, 7), (5, 0)]);
}

#[test]
#[ignore = "slow: SEA vs Schoof at ~28–34 bits; run with --release --ignored"]
fn sea_matches_schoof_larger() {
    for s in ["268435459", "4294967311", "17179869209"] {
        let p = s.parse::<Int>().unwrap();
        assert_sea_eq_schoof(
            &p,
            &[(1, 1), (2, 3), (3, 5), (0, 7), (5, 0), (7, 11), (9, 4)],
        );
    }
}

#[test]
fn sea_supersingular_trace_zero() {
    // y² = x³ + x is supersingular for p ≡ 3 (mod 4): #E = p + 1 (t = 0). Here
    // j = 1728, so every ℓ takes the Schoof fallback inside SEA — the count must
    // still be exact.
    let p = "1099511627791".parse::<Int>().unwrap(); // ~40-bit, ≡ 3 (mod 4)
    assert_eq!((&p % &Int::from(4)), Int::from(3));
    let curve = EllipticCurve::new(mkp(1, &p), mkp(0, &p)).unwrap();
    assert_eq!(curve.sea_point_count(), &p + Int::from(1));
}

#[test]
fn sea_dispatch_through_point_count() {
    // point_count() dispatches to SEA above SEA_BITS (~2^40). The result must
    // satisfy Hasse and annihilate points.
    let p = "1099511627791".parse::<Int>().unwrap(); // ~40-bit prime
    assert!(p.bit_len() >= 40);
    let curve = EllipticCurve::new(mkp(3, &p), mkp(7, &p)).unwrap();
    let n = curve.point_count();
    assert!(hasse_holds(&n, &p));
    annihilates(&curve, &n, &p);
}

#[test]
#[ignore = "slow: large-p SEA; run with --release --ignored"]
fn sea_large_primes_hasse_and_order() {
    // Primes far beyond the naive scan's reach (and where classical Schoof, while
    // possible, is markedly slower): verify Hasse + [#E]·P = O. With the ℓ ≤ 31
    // modular-polynomial table the practical ceiling is ~64 bits; above that the
    // Schoof fallback on primes ℓ > 31 begins to dominate (see the module docs).
    for s in [
        "1000000000000000003",  // ~60-bit
        "1152921504606847009",  // ~61-bit (Atkin-heavy: leans on the fallback)
        "18446744073709551557", // ~64-bit
    ] {
        let p = s.parse::<Int>().unwrap();
        for &(av, bv) in &[(3i64, 7), (2, 5)] {
            let curve = match EllipticCurve::new(mkp(av, &p), mkp(bv, &p)) {
                Some(c) => c,
                None => continue,
            };
            let n = curve.sea_point_count();
            assert!(hasse_holds(&n, &p), "Hasse failed p={s}: #E={n}");
            annihilates(&curve, &n, &p);
        }
    }
}

#[test]
#[ignore = "slow: SEA vs Schoof crossover benchmark; run with --release --ignored --nocapture"]
fn sea_vs_schoof_crossover_bench() {
    use std::time::Instant;
    // At the crossover both algorithms agree; SEA is the faster route, and it
    // keeps scaling to primes where classical Schoof grows impractical.
    for s in ["1099511627791", "1000000000000000003"] {
        let p = s.parse::<Int>().unwrap();
        let curve = EllipticCurve::new(mkp(3, &p), mkp(7, &p)).unwrap();
        let t0 = Instant::now();
        let schoof = curve.schoof_point_count();
        let dt_schoof = t0.elapsed();
        let t1 = Instant::now();
        let sea = curve.sea_point_count();
        let dt_sea = t1.elapsed();
        assert_eq!(sea, schoof, "SEA != Schoof at p={s}");
        println!("p≈2^{}: Schoof {dt_schoof:?}  SEA {dt_sea:?}", p.bit_len());
    }
}
