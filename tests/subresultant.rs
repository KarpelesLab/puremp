//! Differential tests for the subresultant polynomial remainder sequence used by
//! the Sturm chains and `Poly<Rational>` GCD.
//!
//! The subresultant PRS replaces the coefficient-exploding naive rational
//! remainder sequence. Correctness requires:
//!
//!   * `subresultant_gcd` equals the naive Euclidean monic GCD, and
//!   * the subresultant Sturm chain has an **identical sign sequence** (hence
//!     identical real-root counts and isolating intervals) to the classical
//!     `p₀ = p, p₁ = p′, pᵢ = −(pᵢ₋₂ mod pᵢ₋₁)` chain.
//!
//! Each element of the subresultant chain is a *positive* rational multiple of
//! the classical Sturm polynomial, so the two sign sequences must agree at every
//! evaluation point; these tests check that over a dense grid of rationals for a
//! wide battery of polynomials (random, Wilkinson, Chebyshev, Swinnerton–Dyer,
//! clustered/rational-root, and non-squarefree inputs).
#![cfg(feature = "algebraic")]

use puremp::poly::{sturm_count, sturm_variations};
use puremp::{Algebraic, Int, Poly, Rational};

type P = Poly<Rational>;

fn q(n: i64) -> Rational {
    Rational::from(n)
}
fn qd(n: i64, d: i64) -> Rational {
    Rational::new(Int::from(n), Int::from(d))
}
fn poly(cs: &[i64]) -> P {
    Poly::new(cs.iter().map(|&c| q(c)).collect())
}

/// Reference: the *classical* naive rational Sturm chain (the pre-optimization
/// implementation). Used only to differential-test the subresultant chain.
fn naive_sturm_chain(p: &P) -> Vec<P> {
    let mut chain = vec![p.clone(), p.derivative()];
    while !chain.last().unwrap().is_zero() {
        let n = chain.len();
        let r = chain[n - 2].rem(&chain[n - 1]);
        if r.is_zero() {
            break;
        }
        chain.push(r.neg());
    }
    chain
}

/// A grid of rational evaluation points (integers and a few fractional offsets)
/// spanning a range wide enough to bracket every real root of the test inputs.
fn grid() -> Vec<Rational> {
    let mut pts = Vec::new();
    for n in -25..=25 {
        pts.push(q(n));
        pts.push(qd(2 * n + 1, 3));
        pts.push(qd(4 * n + 1, 7));
    }
    pts
}

/// Asserts the subresultant chain and the naive chain have identical sign
/// variations at every grid point (⇒ identical root counts on every interval).
fn assert_sign_sequence_equiv(p: &P) {
    let sf = p.squarefree_part();
    let new_chain = sf.sturm_chain();
    let ref_chain = naive_sturm_chain(&sf);
    for x in grid() {
        let vn = sturm_variations(&new_chain, &x);
        let vr = sturm_variations(&ref_chain, &x);
        assert_eq!(
            vn, vr,
            "sign-variation mismatch at x={x} for p={p} (new {vn} vs naive {vr})"
        );
    }
    // And the root count on every ordered pair of grid points matches.
    let pts = grid();
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len().min(i + 6) {
            let (lo, hi) = (&pts[i], &pts[j]);
            assert_eq!(
                sturm_count(&new_chain, lo, hi),
                sturm_count(&ref_chain, lo, hi),
                "root-count mismatch on ({lo}, {hi}] for p={p}"
            );
        }
    }
}

/// Simple deterministic LCG producing small integer coefficients.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> i64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as i64) % 15 - 7
    }
}

#[test]
fn subresultant_gcd_matches_naive_euclid() {
    let mut rng = Lcg(0xC0FFEE);
    for _ in 0..400 {
        let da = 1 + (rng.0 as usize % 7);
        let db = 1 + (rng.0 as usize % 6);
        let a: P = Poly::new((0..=da).map(|_| q(rng.next())).collect());
        let b: P = Poly::new((0..=db).map(|_| q(rng.next())).collect());
        if a.is_zero() || b.is_zero() {
            continue;
        }
        assert_eq!(
            a.subresultant_gcd(&b),
            a.gcd(&b),
            "gcd mismatch for a={a}, b={b}"
        );
        // Common-factor case: force a shared root by multiplying by (x-2).
        let common = poly(&[-2, 1]);
        let ac = a.mul(&common);
        let bc = b.mul(&common);
        assert_eq!(ac.subresultant_gcd(&bc), ac.gcd(&bc));
    }
}

#[test]
fn random_polynomials_sign_equivalent() {
    let mut rng = Lcg(0x1234_5678);
    for _ in 0..150 {
        let d = 1 + (rng.0 as usize % 8);
        let cs: Vec<i64> = (0..=d).map(|_| rng.next()).collect();
        let p = Poly::new(cs.iter().map(|&c| q(c)).collect::<Vec<_>>());
        if p.degree().unwrap_or(0) < 1 {
            continue;
        }
        assert_sign_sequence_equiv(&p);
    }
}

#[test]
fn wilkinson_like_sign_equivalent() {
    // (x-1)(x-2)...(x-k): tightly spaced integer roots, classic ill-conditioning.
    for k in 2..=8 {
        let mut p = poly(&[1]);
        for r in 1..=k {
            p = p.mul(&poly(&[-r, 1]));
        }
        assert_sign_sequence_equiv(&p);
        assert_eq!(p.real_root_count(), k as usize);
    }
}

#[test]
fn chebyshev_sign_equivalent() {
    // Chebyshev T_n via recurrence T_{n+1} = 2x T_n − T_{n-1}: n distinct roots
    // clustered near ±1.
    let mut tprev = poly(&[1]); // T0 = 1
    let mut cur = poly(&[0, 1]); // T1 = x
    for n in 2..=9 {
        let next = poly(&[0, 2]).mul(&cur).sub(&tprev); // 2x·T_{n-1} − T_{n-2}
        tprev = cur;
        cur = next;
        assert_sign_sequence_equiv(&cur);
        assert_eq!(cur.real_root_count(), n);
    }
}

#[test]
fn swinnerton_dyer_sign_equivalent() {
    // Minimal polynomials of √2+√3 and √2+√3+√5 — dense integer coefficients,
    // all roots real and irrational.
    // (√2+√3): x⁴ − 10x² + 1.
    let sd2 = poly(&[1, 0, -10, 0, 1]);
    assert_sign_sequence_equiv(&sd2);
    assert_eq!(sd2.real_root_count(), 4);

    // (√2+√3+√5): x⁸ − 40x⁶ + 352x⁴ − 960x² + 576.
    let sd3 = poly(&[576, 0, -960, 0, 352, 0, -40, 0, 1]);
    assert_sign_sequence_equiv(&sd3);
    assert_eq!(sd3.real_root_count(), 8);
}

#[test]
fn clustered_and_rational_roots_sign_equivalent() {
    // Clustered roots very close together: (x−1)(x−1001/1000)(x−1002/1000).
    let p = poly(&[-1, 1])
        .mul(&Poly::new(vec![qd(-1001, 1000), q(1)]))
        .mul(&Poly::new(vec![qd(-1002, 1000), q(1)]));
    assert_sign_sequence_equiv(&p);
    assert_eq!(p.real_root_count(), 3);

    // Purely rational roots with denominators.
    let r = Poly::new(vec![qd(-1, 2), q(1)])
        .mul(&Poly::new(vec![qd(-2, 3), q(1)]))
        .mul(&Poly::new(vec![qd(5, 7), q(1)]));
    assert_sign_sequence_equiv(&r);
    assert_eq!(r.real_root_count(), 3);
}

#[test]
fn non_squarefree_sign_equivalent() {
    // Repeated factors: squarefree_part must collapse multiplicity identically.
    let p = poly(&[-1, 1]) // (x-1)
        .mul(&poly(&[-1, 1])) // (x-1)^2
        .mul(&poly(&[-1, 1])) // (x-1)^3
        .mul(&poly(&[-2, 1]))
        .mul(&poly(&[-2, 1])) // (x-2)^2
        .mul(&poly(&[1, 0, 1])); // (x^2+1), no real root
    assert_sign_sequence_equiv(&p);
    assert_eq!(p.real_root_count(), 2); // distinct real roots: 1 and 2

    // (x^2 - 2)^2: one distinct positive/negative irrational root each.
    let q2 = poly(&[-2, 0, 1]).mul(&poly(&[-2, 0, 1]));
    assert_sign_sequence_equiv(&q2);
    assert_eq!(q2.real_root_count(), 2);
}

#[test]
fn isolation_intervals_identical_to_reference() {
    // isolate_real_roots is driven purely by sturm_count; verify the *intervals*
    // match those recomputed from the naive chain via the same bisection logic.
    let cases: Vec<P> = vec![
        poly(&[-6, 11, -6, 1]),       // (x-1)(x-2)(x-3)
        poly(&[1, 0, -10, 0, 1]),     // √2+√3 minimal poly
        poly(&[0, -1, 0, 1]),         // x^3 - x
        poly(&[-2, 0, 1]),            // x^2 - 2
        poly(&[24, -50, 35, -10, 1]), // (x-1)(x-2)(x-3)(x-4)
    ];
    for p in &cases {
        let got = p.isolate_real_roots();
        let want = naive_isolate(p);
        assert_eq!(got, want, "interval mismatch for p={p}");
    }
}

/// Reference isolation using the naive chain (mirrors `isolate_real_roots`).
fn naive_isolate(p: &P) -> Vec<(Rational, Rational)> {
    let sf = p.squarefree_part();
    if sf.degree().unwrap_or(0) < 1 {
        return Vec::new();
    }
    let chain = naive_sturm_chain(&sf);
    // Cauchy bound.
    let lead = sf.leading().unwrap().abs();
    let mut m = Rational::from(0);
    let deg = sf.degree().unwrap();
    for i in 0..deg {
        let r = sf.coeff(i).abs().div(&lead);
        if r > m {
            m = r;
        }
    }
    let b = m.add(&Rational::from(1));
    let two = q(2);
    let mut out = Vec::new();
    let mut stack = vec![(b.neg(), b)];
    while let Some((lo, hi)) = stack.pop() {
        let c = sturm_count(&chain, &lo, &hi);
        if c == 0 {
            continue;
        }
        if c == 1 {
            out.push((lo, hi));
            continue;
        }
        let mid = lo.add(&hi).div(&two);
        stack.push((lo, mid.clone()));
        stack.push((mid, hi));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Largest coefficient bit-length (numerator or denominator) over a chain.
fn max_coeff_bits(chain: &[P]) -> u32 {
    chain
        .iter()
        .flat_map(|p| p.coeffs())
        .map(|c| c.numerator().bit_len().max(c.denominator().bit_len()))
        .max()
        .unwrap_or(0)
}

/// Coefficient-size and timing comparison of the subresultant Sturm chain versus
/// the naive rational remainder sequence, on high-degree inputs. Run with:
/// `cargo test --all-features --test subresultant -- --ignored --nocapture bench`.
#[test]
#[ignore = "timing/coefficient-size benchmark; run explicitly with --ignored --nocapture"]
fn bench_coefficient_growth() {
    use std::time::Instant;

    // A battery of high-degree, dense-coefficient squarefree polynomials.
    let mut cases: Vec<(String, P)> = Vec::new();

    // Swinnerton–Dyer of √2+√3+√5 (degree 8).
    cases.push((
        "SD(2,3,5) deg8".into(),
        poly(&[576, 0, -960, 0, 352, 0, -40, 0, 1]),
    ));
    // Wilkinson (x-1)...(x-10), degree 10, huge integer coefficients downstream.
    let mut wilk = poly(&[1]);
    for r in 1..=10 {
        wilk = wilk.mul(&poly(&[-r, 1]));
    }
    cases.push(("Wilkinson deg10".into(), wilk));
    // Chebyshev T_12 (degree 12), roots clustered in [-1,1].
    let mut tprev = poly(&[1]);
    let mut cur = poly(&[0, 1]);
    for _ in 2..=12 {
        let next = poly(&[0, 2]).mul(&cur).sub(&tprev);
        tprev = cur;
        cur = next;
    }
    cases.push(("Chebyshev T12".into(), cur));
    // Dense rational-coefficient polynomial of degree 12.
    let mut rng = Lcg(0xBEEF);
    let dense: P = Poly::new(
        (0..=12)
            .map(|_| Rational::new(Int::from(rng.next() * 3 + 1), Int::from(2)))
            .collect(),
    );
    cases.push(("dense-rational deg12".into(), dense));

    println!(
        "\n{:<24} {:>14} {:>14} {:>10}",
        "case", "naive maxbits", "subres maxbits", "ratio"
    );
    for (name, p) in &cases {
        let sf = p.squarefree_part();

        let t0 = Instant::now();
        let naive = naive_sturm_chain(&sf);
        let naive_t = t0.elapsed();
        let nb = max_coeff_bits(&naive);

        let t1 = Instant::now();
        let sub = sf.sturm_chain();
        let sub_t = t1.elapsed();
        let sb = max_coeff_bits(&sub);

        println!(
            "{name:<24} {nb:>14} {sb:>14} {:>9.1}x   (naive {:?}, subres {:?})",
            nb as f64 / sb.max(1) as f64,
            naive_t,
            sub_t,
        );
    }

    // End-to-end Algebraic: an operation that builds a high-degree resultant and
    // isolates a root through the Sturm machinery.
    fn root_sqrt(k: i64) -> Algebraic {
        Algebraic::new(poly(&[-k, 0, 1]), q(0), q(k.max(1)))
    }
    let t = Instant::now();
    let s = root_sqrt(2)
        .add(&root_sqrt(3))
        .add(&root_sqrt(5))
        .add(&root_sqrt(7));
    println!(
        "\n√2+√3+√5+√7 = {} (degree {}) in {:?}",
        s.to_f64(),
        s.defining_polynomial().degree().unwrap(),
        t.elapsed(),
    );
}

#[test]
fn algebraic_end_to_end_known_values() {
    fn root_sqrt(k: i64) -> Algebraic {
        Algebraic::new(poly(&[-k, 0, 1]), q(0), q(k.max(1)))
    }
    let s2 = root_sqrt(2);
    let s3 = root_sqrt(3);
    let s5 = root_sqrt(5);

    // (√2)² = 2, (√2·√3)² = 6, etc.
    assert_eq!(s2.mul(&s2), Algebraic::from_int(Int::from(2)));
    assert_eq!(
        s2.mul(&s3).mul(&s2).mul(&s3),
        Algebraic::from_int(Int::from(6))
    );

    // (√2+√3) is a root of x⁴−10x²+1, and (√2+√3)·(√3−√2)=1.
    let sum = s2.add(&s3);
    let a = sum.defining_polynomial();
    assert_eq!(a.real_root_count(), 4);
    assert_eq!(s3.sub(&s2).mul(&sum), Algebraic::from_int(Int::from(1)));

    // √2 + √3 + √5 compared with itself and ordering sanity.
    let big = s2.add(&s3).add(&s5);
    assert!(big > Algebraic::from_int(Int::from(5)));
    assert!(big < Algebraic::from_int(Int::from(6)));
    assert_eq!(big.clone(), big.clone());

    // signum / comparisons.
    assert_eq!(s2.sub(&s2).signum(), 0);
    assert!(s3 > s2);
    assert!(s2.neg() < s2);

    // sqrt of an algebraic: √(√2) is a root of x⁴ − 2.
    let ss2 = s2.sqrt();
    assert_eq!(ss2.mul(&ss2), s2);
    assert_eq!(ss2.defining_polynomial().real_root_count(), 2);
}
