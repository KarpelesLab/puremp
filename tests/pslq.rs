#![cfg(feature = "lattice")]
#![cfg(feature = "float")]
//! PSLQ integer-relation detection: recover known relations and verify each
//! returned vector `a` genuinely satisfies `|Σ aᵢ·xᵢ|` tiny, that rationally
//! independent inputs yield `None`, and that PSLQ agrees with the LLL-based
//! [`find_integer_relation`] where both find a relation.

use puremp::lattice::{find_integer_relation, pslq};
use puremp::{Float, Int, RoundingMode};

const PREC: u64 = 300;
const M: RoundingMode = RoundingMode::Nearest;

fn f(n: i64) -> Float {
    Float::from_int(&Int::from_i64(n), PREC, M)
}
fn sqrt(n: i64) -> Float {
    f(n).sqrt(PREC, M)
}
fn ln(n: i64) -> Float {
    f(n).ln(PREC, M)
}

/// Primitive, sign-normalized form of a relation: divide out the content (gcd)
/// and force the first nonzero entry positive, so scalar multiples compare equal.
fn canonical(v: &[Int]) -> Vec<Int> {
    let mut g = Int::ZERO;
    for c in v {
        g = g.gcd(c);
    }
    if g.is_zero() {
        return v.to_vec();
    }
    let mut out: Vec<Int> = v.iter().map(|c| c.div_floor(&g)).collect();
    if let Some(first) = out.iter().find(|c| !c.is_zero())
        && first.is_negative()
    {
        for c in &mut out {
            *c = c.neg();
        }
    }
    out
}

fn eq_upto_sign_scale(got: &[Int], want: &[i64]) -> bool {
    let w: Vec<Int> = want.iter().map(|&x| Int::from_i64(x)).collect();
    canonical(got) == canonical(&w)
}

/// Assert the relation really annihilates the inputs: |Σ aᵢ·xᵢ| below 2^-tol.
fn assert_tiny(a: &[Int], xs: &[Float], tol: u64) {
    let mut acc = Float::zero(PREC);
    for (ai, xi) in a.iter().zip(xs) {
        acc = acc.add(&Float::from_int(ai, PREC, M).mul(xi, PREC, M), PREC, M);
    }
    let bound = Float::from_int(&Int::ONE, PREC, M).div(
        &Float::from_int(&Int::ONE.mul_2k(tol as u32), PREC, M),
        PREC,
        M,
    );
    assert!(
        acc.abs() < bound,
        "residual too large: |{a:?}·x| = {}",
        acc.to_shortest_string()
    );
}

#[test]
fn golden_ratio_powers() {
    // φ = (1+√5)/2 satisfies φ² − φ − 1 = 0 → relation (−1, −1, 1).
    let phi = f(1).add(&sqrt(5), PREC, M).div(&f(2), PREC, M);
    let xs = [f(1), phi.clone(), phi.mul(&phi, PREC, M)];
    let r = pslq(&xs, PREC).expect("relation for {1, φ, φ²}");
    assert!(eq_upto_sign_scale(&r, &[-1, -1, 1]), "got {r:?}");
    assert_tiny(&r, &xs, 200);
}

#[test]
fn rational_multiples() {
    // {1.5, 2.5, 4.0}: 1.5 + 2.5 − 4.0 = 0 exactly → relation (1, 1, −1).
    let xs = [f(3).div(&f(2), PREC, M), f(5).div(&f(2), PREC, M), f(4)];
    let r = pslq(&xs, PREC).expect("relation for {1.5, 2.5, 4.0}");
    assert!(eq_upto_sign_scale(&r, &[1, 1, -1]), "got {r:?}");
    assert_tiny(&r, &xs, 200);
}

#[test]
fn logarithms() {
    // log2 + log3 − log6 = log(2·3/6) = log 1 = 0 → relation (1, 1, −1).
    let xs = [ln(2), ln(3), ln(6)];
    let r = pslq(&xs, PREC).expect("relation for {ln2, ln3, ln6}");
    assert!(eq_upto_sign_scale(&r, &[1, 1, -1]), "got {r:?}");
    assert_tiny(&r, &xs, 200);
}

#[test]
fn no_low_height_relation() {
    // {1, √2, π}: no small integer relation.
    let xs = [f(1), sqrt(2), Float::pi(PREC, M)];
    assert!(pslq(&xs, PREC).is_none());
    // {√2, √3}: Q-linearly independent.
    assert!(pslq(&[sqrt(2), sqrt(3)], PREC).is_none());
}

#[test]
fn agrees_with_lll() {
    // Inputs where the LLL-based detector also succeeds; the primitive relation
    // is unique up to sign, so PSLQ must recover the same vector.
    const SCALE: u64 = 200;
    let cases: [Vec<Float>; 3] = [
        vec![sqrt(2), sqrt(8)],    // 2√2 − √8 = 0
        vec![f(1), sqrt(2), f(2)], // 2·1 − 2 = 0
        vec![ln(2), ln(3), ln(6)], // log2 + log3 − log6 = 0
    ];
    for xs in &cases {
        let lll = find_integer_relation(xs, SCALE).expect("LLL relation");
        let ps = pslq(xs, PREC).expect("PSLQ relation");
        assert_eq!(
            canonical(&ps),
            canonical(&lll),
            "PSLQ {ps:?} vs LLL {lll:?}"
        );
    }
}

#[test]
fn two_element_relation() {
    // n = 2 path (H has a single column): 2√2 − √8 = 0.
    let xs = [sqrt(2), sqrt(8)];
    let r = pslq(&xs, PREC).expect("relation for {√2, √8}");
    assert!(eq_upto_sign_scale(&r, &[2, -1]), "got {r:?}");
    assert_tiny(&r, &xs, 200);
}
