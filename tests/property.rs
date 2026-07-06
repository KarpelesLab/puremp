//! Randomized property / differential tests for the extended numeric types.
//!
//! These check algebraic invariants (ring/field axioms, homomorphisms,
//! enclosure) and cross-check derived types against the exact core types over
//! many pseudo-random inputs. A small deterministic LCG keeps the crate free of
//! test-only RNG dependencies and makes failures reproducible.
#![cfg(feature = "rational")]

use puremp::{Int, Rational};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 1
    }
    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo) as u64) as i64
    }
    fn int(&mut self, mag: i64) -> Int {
        Int::from_i64(self.range(-mag, mag + 1))
    }
    fn rational(&mut self, mag: i64) -> Rational {
        let den = self.range(1, mag + 1);
        Rational::new(self.int(mag), Int::from_i64(den))
    }
}

#[cfg(feature = "decimal")]
#[test]
fn decimal_exact_ring_and_homomorphism() {
    use puremp::Decimal;
    let mut rng = Rng::new(0x0DEC);
    let dec = |rng: &mut Rng| Decimal::new(rng.int(100_000), rng.range(-4, 5));
    for _ in 0..500 {
        let a = dec(&mut rng);
        let b = dec(&mut rng);
        let c = dec(&mut rng);
        // ring axioms (exact)
        assert_eq!(a.add(&b), b.add(&a));
        assert_eq!(a.mul(&b), b.mul(&a));
        assert_eq!(a.add(&b).sub(&b), a);
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
        // to_rational is a ring homomorphism
        assert_eq!(
            a.add(&b).to_rational(),
            a.to_rational().add(&b.to_rational())
        );
        assert_eq!(
            a.mul(&b).to_rational(),
            a.to_rational().mul(&b.to_rational())
        );
        // Display/FromStr round-trip (value-preserving)
        let s = a.to_string();
        assert_eq!(s.parse::<Decimal>().unwrap(), a);
    }
}

#[cfg(feature = "dyadic")]
#[test]
fn dyadic_exact_ring_and_homomorphism() {
    use puremp::Dyadic;
    let mut rng = Rng::new(0x0D1A);
    let dy = |rng: &mut Rng| Dyadic::new(rng.int(100_000), rng.range(-8, 9));
    for _ in 0..500 {
        let a = dy(&mut rng);
        let b = dy(&mut rng);
        let c = dy(&mut rng);
        assert_eq!(a.add(&b).sub(&b), a);
        assert_eq!(a.mul(&b), b.mul(&a));
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
        assert_eq!(
            a.add(&b).to_rational(),
            a.to_rational().add(&b.to_rational())
        );
        // exact terminating decimal round-trips
        assert_eq!(a.to_string().parse::<Dyadic>().unwrap(), a);
    }
}

#[test]
fn rational_cross_reduction_differential() {
    // Differential check of the cross-reduced `add`/`sub`/`mul`/`div`/`addmul`
    // against the naive "full product then a single gcd" reference, which is
    // exactly what `Rational::new` does. Canonical form is unique, so a correct
    // faster path must return the *identical* (num, den) for every input.
    let mut rng = Rng::new(0x4A11);

    // Reference (naive) operations built straight from the public `Int` API.
    let ref_add = |a: &Rational, b: &Rational| {
        Rational::new(
            a.numerator()
                .mul(b.denominator())
                .add(&b.numerator().mul(a.denominator())),
            a.denominator().mul(b.denominator()),
        )
    };
    let ref_sub = |a: &Rational, b: &Rational| {
        Rational::new(
            a.numerator()
                .mul(b.denominator())
                .sub(&b.numerator().mul(a.denominator())),
            a.denominator().mul(b.denominator()),
        )
    };
    let ref_mul = |a: &Rational, b: &Rational| {
        Rational::new(
            a.numerator().mul(b.numerator()),
            a.denominator().mul(b.denominator()),
        )
    };
    let ref_div = |a: &Rational, b: &Rational| {
        Rational::new(
            a.numerator().mul(b.denominator()),
            a.denominator().mul(b.numerator()),
        )
    };

    // Build an Int of roughly `words * 64` bits with a random sign.
    let big_int = |rng: &mut Rng, words: usize| {
        let mut v = Int::ZERO;
        let shift = Int::ONE.mul_2k(64);
        for _ in 0..words.max(1) {
            v = v.mul(&shift).add(&Int::from_u64(rng.next()));
        }
        if rng.next() & 1 == 0 { v.neg() } else { v }
    };

    // A grab-bag of generators: tiny, large, unit denominators, shared factors,
    // and zeros — across all sign combinations (the sign comes from the ints).
    let make = |rng: &mut Rng, kind: u64| -> Rational {
        match kind % 6 {
            0 => Rational::ZERO,
            1 => rng.rational(50),                       // tiny fraction
            2 => Rational::from_integer(rng.int(1_000)), // unit denominator
            3 => {
                let w = (rng.next() % 20) as usize + 1;
                let n = big_int(rng, w);
                let d = big_int(rng, w).abs().add(&Int::ONE);
                Rational::new(n, d)
            }
            4 => {
                // Shared factor between numerator and denominator (canonicalised
                // away by `new`, but exercises cross-cancellation).
                let (wg, wn, wd) = (
                    (rng.next() % 8) as usize + 1,
                    (rng.next() % 8) as usize + 1,
                    (rng.next() % 8) as usize + 1,
                );
                let g = big_int(rng, wg).abs().add(&Int::ONE);
                let n = big_int(rng, wn).mul(&g);
                let d = big_int(rng, wd).abs().add(&Int::ONE).mul(&g);
                Rational::new(n, d)
            }
            _ => {
                // Numerator and denominator of very different sizes.
                Rational::new(big_int(rng, 12), big_int(rng, 1).abs().add(&Int::ONE))
            }
        }
    };

    for i in 0..4000u64 {
        let a = make(&mut rng, i);
        let b = make(&mut rng, i / 6 + 1);

        assert_eq!(a.add(&b), ref_add(&a, &b), "add mismatch: {a} + {b}");
        assert_eq!(a.sub(&b), ref_sub(&a, &b), "sub mismatch: {a} - {b}");
        assert_eq!(a.mul(&b), ref_mul(&a, &b), "mul mismatch: {a} * {b}");
        if !b.is_zero() {
            assert_eq!(a.div(&b), ref_div(&a, &b), "div mismatch: {a} / {b}");
        }
        // addmul / submul are `self ± a·b`; check against the reference chain.
        let mut fma = a.clone();
        fma.addmul(&a, &b);
        assert_eq!(fma, ref_add(&a, &ref_mul(&a, &b)));
        let mut fms = a.clone();
        fms.submul(&a, &b);
        assert_eq!(fms, ref_sub(&a, &ref_mul(&a, &b)));

        // Exact byte-for-byte numerator/denominator identity (not just value).
        let s = a.add(&b);
        let r = ref_add(&a, &b);
        assert!(s.numerator() == r.numerator() && s.denominator() == r.denominator());
    }
}

#[test]
fn mod_int_differential_vs_int() {
    use puremp::ModInt;
    let mut rng = Rng::new(0x110D);
    for _ in 0..500 {
        let m = Int::from_i64(rng.range(2, 5000));
        let a = rng.int(1_000_000);
        let b = rng.int(1_000_000);
        let (am, bm) = (
            ModInt::new(a.clone(), m.clone()),
            ModInt::new(b.clone(), m.clone()),
        );
        assert_eq!(am.add(&bm).to_int(), a.add(&b).rem_euclid(&m));
        assert_eq!(am.sub(&bm).to_int(), a.sub(&b).rem_euclid(&m));
        assert_eq!(am.mul(&bm).to_int(), a.mul(&b).rem_euclid(&m));
        assert_eq!(am.neg().to_int(), a.neg().rem_euclid(&m));
        // pow matches Int::modpow
        let e = Int::from_i64(rng.range(0, 40));
        assert_eq!(am.pow(&e).to_int(), a.modpow(&e, &m));
        // inverse identity when coprime
        if let Some(inv) = am.inv() {
            assert!(am.mul(&inv).to_int().is_one());
        }
    }
}

#[cfg(feature = "complex")]
#[test]
fn complex_gaussian_ring_axioms() {
    use puremp::Complex;
    let mut rng = Rng::new(0xC0FF);
    let c = |rng: &mut Rng| Complex::new(rng.int(1000), rng.int(1000));
    for _ in 0..500 {
        let a = c(&mut rng);
        let b = c(&mut rng);
        let d = c(&mut rng);
        assert_eq!(a.add(&b).add(&d), a.add(&b.add(&d))); // associative +
        assert_eq!(a.mul(&b), b.mul(&a)); // commutative *
        assert_eq!(a.mul(&b.add(&d)), a.mul(&b).add(&a.mul(&d))); // distributive
        assert_eq!(a.mul(&b).conj(), a.conj().mul(&b.conj())); // conj homomorphism
        // norm is multiplicative
        assert_eq!(a.mul(&b).norm_sqr(), a.norm_sqr().mul(&b.norm_sqr()));
    }
}

#[cfg(feature = "poly")]
#[test]
fn poly_eval_homomorphism_and_division() {
    use puremp::Poly;
    let mut rng = Rng::new(0xB0B0);
    let poly = |rng: &mut Rng| -> Poly<Rational> {
        let deg = rng.range(0, 5) as usize;
        Poly::new((0..=deg).map(|_| rng.rational(20)).collect())
    };
    for _ in 0..400 {
        let a = poly(&mut rng);
        let b = poly(&mut rng);
        let x = rng.rational(20);
        // eval is a ring homomorphism
        assert_eq!(a.add(&b).eval(&x), a.eval(&x).add(&b.eval(&x)));
        assert_eq!(a.mul(&b).eval(&x), a.eval(&x).mul(&b.eval(&x)));
        // division identity: a = q*b + r, deg r < deg b
        if !b.is_zero() {
            let (q, r) = a.div_rem(&b);
            assert_eq!(q.mul(&b).add(&r), a);
            if !r.is_zero() {
                assert!(r.degree().unwrap() < b.degree().unwrap());
            }
        }
    }
}

#[cfg(feature = "matrix")]
#[test]
fn matrix_determinant_and_inverse() {
    use puremp::Matrix;
    let mut rng = Rng::new(0x3A71);
    for _ in 0..80 {
        let n = rng.range(1, 4) as usize;
        let make = |rng: &mut Rng| -> Matrix<Rational> {
            Matrix::new(n, n, (0..n * n).map(|_| rng.rational(6)).collect())
        };
        let a = make(&mut rng);
        let b = make(&mut rng);
        // det is multiplicative
        assert_eq!(
            a.mul(&b).determinant(),
            a.determinant().mul(&b.determinant())
        );
        // A · A⁻¹ = I when non-singular
        if let Some(inv) = a.inverse() {
            assert_eq!(a.mul(&inv), Matrix::<Rational>::identity(n));
        }
    }
    // Bareiss integer determinant agrees with the rational determinant.
    let mut rng = Rng::new(0xBA12);
    for _ in 0..80 {
        let n = rng.range(1, 4) as usize;
        let data: Vec<Int> = (0..n * n).map(|_| rng.int(8)).collect();
        let mi = Matrix::new(n, n, data.clone());
        let mr = Matrix::new(
            n,
            n,
            data.iter()
                .map(|x| Rational::from_integer(x.clone()))
                .collect(),
        );
        assert_eq!(Rational::from_integer(mi.determinant()), mr.determinant());
    }
}

#[cfg(feature = "algebraic")]
#[test]
fn quadratic_field_axioms() {
    use puremp::Quadratic;
    let mut rng = Rng::new(0x0DAD);
    let ds = [2i64, 3, 5, 6, 7, 10];
    for _ in 0..300 {
        let d = Int::from_i64(ds[(rng.next() % ds.len() as u64) as usize]);
        let a = Quadratic::new(rng.rational(30), rng.rational(30), d.clone());
        let b = Quadratic::new(rng.rational(30), rng.rational(30), d.clone());
        let c = Quadratic::new(rng.rational(30), rng.rational(30), d);
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c))); // distributive
        assert_eq!(a.mul(&b), b.mul(&a)); // commutative
        // norm(x) = x · conj(x) (a rational)
        assert_eq!(Quadratic::rational(a.norm()), a.mul(&a.conjugate()));
        // a · a⁻¹ = 1 when a ≠ 0
        if !a.norm().is_zero() {
            assert_eq!(a.mul(&a.recip()), Quadratic::from(Int::ONE));
        }
    }
}

#[cfg(feature = "interval")]
#[test]
fn interval_enclosure_theorem() {
    use puremp::{Float, Interval, RoundingMode};
    let n = RoundingMode::Nearest;
    let mut rng = Rng::new(0x171F);
    for _ in 0..300 {
        let mk = |rng: &mut Rng| {
            let (x, y) = (rng.range(-50, 51), rng.range(-50, 51));
            let (lo, hi) = (x.min(y), x.max(y));
            (
                Interval::new(
                    Float::from_int(&Int::from_i64(lo), 53, n),
                    Float::from_int(&Int::from_i64(hi), 53, n),
                    53,
                ),
                lo,
                hi,
            )
        };
        let (ia, alo, ahi) = mk(&mut rng);
        let (ib, blo, bhi) = mk(&mut rng);
        // Sample exact integer points inside each interval.
        let px = Rational::from(rng.range(alo, ahi + 1));
        let py = Rational::from(rng.range(blo, bhi + 1));
        // The true results must lie within the interval results (enclosure).
        let inside = |iv: &Interval, v: &Rational| {
            iv.lower().to_rational().unwrap() <= *v && *v <= iv.upper().to_rational().unwrap()
        };
        assert!(inside(&ia.add(&ib), &px.add(&py)));
        assert!(inside(&ia.sub(&ib), &px.sub(&py)));
        assert!(inside(&ia.mul(&ib), &px.mul(&py)));
    }
}

#[cfg(feature = "algebraic")]
#[test]
fn algebraic_differential_vs_float() {
    use puremp::{Algebraic, RoundingMode};
    let n = RoundingMode::Nearest;
    // A few small algebraic values and their f64 references.
    let sqrt = |k: i64| {
        Algebraic::new(
            puremp::Poly::new(vec![
                Rational::from(-k),
                Rational::from(0),
                Rational::from(1),
            ]),
            Rational::from(0),
            Rational::from(k.max(1)),
        )
    };
    let vals = [
        (sqrt(2), 2.0f64.sqrt()),
        (sqrt(3), 3.0f64.sqrt()),
        (Algebraic::from_int(Int::from(2)), 2.0),
        (sqrt(5), 5.0f64.sqrt()),
    ];
    for (a, af) in &vals {
        for (b, bf) in &vals {
            // sum matches the float sum
            assert!((a.add(b).to_float(53, n).to_f64() - (af + bf)).abs() < 1e-12);
            // product matches
            assert!((a.mul(b).to_float(53, n).to_f64() - (af * bf)).abs() < 1e-12);
            // ordering matches the float ordering
            assert_eq!(a < b, af < bf);
        }
        // additive inverse and, for nonzero, multiplicative inverse
        assert!(a.add(&a.neg()).signum() == 0);
        if a.signum() != 0 {
            assert!(
                a.mul(&a.recip())
                    .sub(&Algebraic::from_int(Int::ONE))
                    .signum()
                    == 0
            );
        }
    }
}
