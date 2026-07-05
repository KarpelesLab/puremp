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
