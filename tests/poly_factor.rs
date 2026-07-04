#![cfg(all(feature = "poly", feature = "rational"))]
//! Polynomial factorization over ℚ: reconstruction and known factorizations.

use puremp::{Int, Poly, Rational};

fn rp(coeffs: &[i64]) -> Poly<Rational> {
    Poly::new(
        coeffs
            .iter()
            .map(|&c| Rational::from_integer(Int::from_i64(c)))
            .collect(),
    )
}

fn reconstruct(factors: &[(Poly<Rational>, usize)]) -> Poly<Rational> {
    let mut prod = Poly::constant(Rational::ONE);
    for (f, m) in factors {
        for _ in 0..*m {
            prod = prod.mul(f);
        }
    }
    prod
}

#[test]
fn factors_reconstruct_and_are_monic() {
    let cases: Vec<Vec<i64>> = vec![
        vec![-1, 0, 1],        // x² − 1
        vec![-2, 0, 1],        // x² − 2 (irreducible)
        vec![-1, 0, 0, 0, 1],  // x⁴ − 1
        vec![0, -1, 0, 1],     // x³ − x
        vec![1, 2, 1],         // (x+1)²
        vec![1, 1, 2, 1, 1],   // (x²+1)(x²+x+1)
        vec![1, 5, 6],         // 6x² + 5x + 1 = (2x+1)(3x+1)
        vec![1, 0, -10, 0, 1], // x⁴ − 10x² + 1 (irreducible)
        vec![-6, 11, -6, 1],   // (x−1)(x−2)(x−3)
        vec![2, 3, 1],         // (x+1)(x+2)
    ];
    for c in &cases {
        let f = rp(c);
        let facs = f.factor();
        assert_eq!(
            reconstruct(&facs),
            f.monic(),
            "reconstruction failed for {c:?}"
        );
        for (g, m) in &facs {
            assert!(*m >= 1 && g.degree().unwrap() >= 1);
            assert_eq!(
                g.leading().unwrap(),
                &Rational::ONE,
                "factor not monic for {c:?}"
            );
        }
    }
}

#[test]
fn known_factorizations() {
    // x² − 1 → 2 linear factors.
    let f = rp(&[-1, 0, 1]).factor();
    assert_eq!(f.len(), 2);
    assert!(f.iter().all(|(g, m)| g.degree() == Some(1) && *m == 1));

    // (x+1)² → single factor with multiplicity 2.
    let f = rp(&[1, 2, 1]).factor();
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].1, 2);
    assert_eq!(f[0].0, rp(&[1, 1]));

    // x² − 2 irreducible → one degree-2 factor.
    let f = rp(&[-2, 0, 1]).factor();
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].0.degree(), Some(2));

    // x⁴ − 1 → (x−1)(x+1)(x²+1): three factors.
    let f = rp(&[-1, 0, 0, 0, 1]).factor();
    assert_eq!(f.len(), 3);
    assert_eq!(f.iter().filter(|(g, _)| g.degree() == Some(2)).count(), 1);

    // 6x² + 5x + 1 = (2x+1)(3x+1) → monic (x+1/2)(x+1/3).
    let f = rp(&[1, 5, 6]).factor();
    assert_eq!(f.len(), 2);
    assert!(f.iter().all(|(g, m)| g.degree() == Some(1) && *m == 1));
}

#[test]
fn stress_random_products() {
    // Build polynomials as known products of small random factors, then confirm
    // factor() reconstructs the monic original. This exercises multi-factor
    // recombination and Hensel lifting across many shapes.
    let mut seed = 0x00C0_FFEE_1234u64;
    let rnd = |lo: i64, hi: i64, s: &mut u64| {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        lo + ((*s >> 33) as i64).rem_euclid(hi - lo + 1)
    };
    for _ in 0..60 {
        // 2–4 random linear/quadratic factors.
        let k = 2 + (rnd(0, 2, &mut seed) as usize);
        let mut f = rp(&[1]);
        for _ in 0..k {
            let g = if rnd(0, 1, &mut seed) == 0 {
                rp(&[rnd(-5, 5, &mut seed), rnd(1, 4, &mut seed)]) // linear
            } else {
                rp(&[
                    rnd(-3, 3, &mut seed),
                    rnd(-3, 3, &mut seed),
                    rnd(1, 3, &mut seed),
                ]) // quadratic
            };
            f = f.mul(&g);
        }
        if f.degree().unwrap_or(0) == 0 {
            continue;
        }
        let facs = f.factor();
        assert_eq!(reconstruct(&facs), f.monic(), "reconstruction failed");
        // Factors must be irreducible: re-factoring each yields itself.
        for (g, _) in &facs {
            let refac = g.factor();
            assert_eq!(refac.len(), 1, "factor not irreducible: {:?}", g.coeffs());
            assert_eq!(refac[0].1, 1);
        }
    }
}
