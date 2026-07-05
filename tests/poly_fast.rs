//! Public-API tests for the subquadratic polynomial paths: Newton fast
//! division, Half-GCD, and Kronecker multiplication. The strongest
//! differential checks (fast path vs the schoolbook/Euclid base case, called
//! directly) live as in-crate unit tests in `src/poly.rs`; these exercise the
//! same algorithms through the public surface, over degrees straddling the
//! thresholds, and add `GF(pᵏ)` coverage.
#![cfg(all(feature = "poly", feature = "rational", feature = "galois"))]

use puremp::galois::GaloisField;
use puremp::{FiniteField, Int, ModInt, Poly, Rational};

struct Lcg(core::cell::Cell<u64>);
impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg(core::cell::Cell::new(seed ^ 0x9e3779b97f4a7c15))
    }
    fn next(&self) -> u64 {
        let s = self
            .0
            .get()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0.set(s);
        s
    }
    fn range(&self, n: i64) -> i64 {
        (self.next() >> 33) as i64 % n
    }
}

fn prat(rng: &Lcg, deg: usize) -> Poly<Rational> {
    let mut v: Vec<Rational> = (0..deg)
        .map(|_| Rational::new(Int::from(rng.range(41) - 20), Int::from(rng.range(9) + 1)))
        .collect();
    v.push(Rational::from(Int::from(rng.range(20) + 1)));
    Poly::new(v)
}

fn pmod(rng: &Lcg, deg: usize, p: &Int) -> Poly<ModInt> {
    let mut v: Vec<ModInt> = (0..deg)
        .map(|_| ModInt::new(Int::from(rng.range(1 << 20)), p.clone()))
        .collect();
    v.push(ModInt::new(Int::from(rng.range(1 << 20) + 1), p.clone()));
    Poly::new(v)
}

fn pgf(rng: &Lcg, deg: usize, f: &GaloisField) -> Poly<puremp::GfElement> {
    let sample = f.one();
    let mut v: Vec<puremp::GfElement> = (0..deg)
        .map(|_| sample.from_index(&Int::from(rng.range(i64::MAX))))
        .collect();
    // ensure nonzero leading
    let mut lead = sample.from_index(&Int::from(rng.range(i64::MAX)));
    while lead.is_zero() {
        lead = sample.from_index(&Int::from(rng.range(i64::MAX)));
    }
    v.push(lead);
    Poly::new(v)
}

// ---- Newton division: the defining (q, r) identity over the threshold. ----

#[test]
fn div_rem_identity_rational() {
    let rng = Lcg::new(11);
    for _ in 0..200 {
        let na = 30 + rng.range(120) as usize;
        let nb = 1 + rng.range(na as i64) as usize;
        let a = prat(&rng, na);
        let b = prat(&rng, nb);
        let (q, r) = a.div_rem(&b);
        assert_eq!(a, &(&b * &q) + &r);
        assert!(r.is_zero() || r.degree().unwrap() < b.degree().unwrap());
    }
}

#[test]
fn div_rem_identity_gfp() {
    let p = Int::from(1_000_003);
    let rng = Lcg::new(12);
    for _ in 0..300 {
        let na = 30 + rng.range(150) as usize;
        let nb = 1 + rng.range(na as i64) as usize;
        let a = pmod(&rng, na, &p);
        let b = pmod(&rng, nb, &p);
        let (q, r) = a.div_rem(&b);
        assert_eq!(a, &(&b * &q) + &r);
        assert!(r.is_zero() || r.degree().unwrap() < b.degree().unwrap());
    }
}

// ---- Half-GCD: structured properties over GF(p) and GF(p^k). ----

fn is_monic<T: FiniteField>(g: &Poly<T>) -> bool {
    g.is_zero() || g.leading().unwrap() == &g.leading().unwrap().one()
}

#[test]
fn gcd_shared_factor_gfp() {
    let p = Int::from(1_000_003);
    let rng = Lcg::new(13);
    for _ in 0..120 {
        let g = pmod(&rng, 15 + rng.range(40) as usize, &p);
        let u = pmod(&rng, 40 + rng.range(80) as usize, &p);
        let v = pmod(&rng, 40 + rng.range(80) as usize, &p);
        let a = &g * &u;
        let b = &g * &v;
        let d = a.gcd(&b);
        assert!(is_monic(&d));
        // g divides d (d is at least g's monic), and d divides a and b
        assert!(a.rem(&d).is_zero());
        assert!(b.rem(&d).is_zero());
        assert!(d.rem(&g.monic()).is_zero());
        // idempotence / identities
        assert_eq!(a.gcd(&a), a.monic());
        assert_eq!(a.gcd(&Poly::zero()), a.monic());
    }
}

#[test]
fn gcd_one_divides_other_gfp() {
    let p = Int::from(65_537);
    let rng = Lcg::new(14);
    for _ in 0..120 {
        let q = pmod(&rng, 60 + rng.range(80) as usize, &p);
        let d = pmod(&rng, 20 + rng.range(50) as usize, &p);
        let a = &q * &d;
        assert_eq!(a.gcd(&d), d.monic());
    }
}

#[test]
fn gcd_shared_factor_gfpk() {
    // GF(3^3)
    let f = GaloisField::create(Int::from(3), 3).unwrap();
    let rng = Lcg::new(15);
    for _ in 0..40 {
        let g = pgf(&rng, 12 + rng.range(20) as usize, &f);
        let u = pgf(&rng, 40 + rng.range(50) as usize, &f);
        let v = pgf(&rng, 40 + rng.range(50) as usize, &f);
        let a = &g * &u;
        let b = &g * &v;
        let d = a.gcd(&b);
        assert!(a.rem(&d).is_zero());
        assert!(b.rem(&d).is_zero());
        assert!(d.rem(&g.monic()).is_zero());
    }
}

#[test]
fn gcd_matches_across_threshold_gfp() {
    // Same pair below and above the threshold must give the same monic gcd:
    // low degrees force Euclid, high degrees force Half-GCD.
    let p = Int::from(1_000_003);
    let rng = Lcg::new(16);
    for _ in 0..60 {
        let g = pmod(&rng, 5 + rng.range(8) as usize, &p);
        let a = &g * &pmod(&rng, 3 + rng.range(6) as usize, &p);
        let b = &g * &pmod(&rng, 3 + rng.range(6) as usize, &p);
        let small = a.gcd(&b);
        // Scale up degree by multiplying both by a common high-degree poly,
        // then divide the gcd back — the monic gcd of the scaled pair equals
        // the scaled monic gcd.
        let s = pmod(&rng, 60, &p);
        let big = (&a * &s).gcd(&(&b * &s));
        assert_eq!(big, (&small * &s).monic());
    }
}

// ---- Kronecker multiplication via public mul_kronecker. ----

#[test]
fn mul_kronecker_vs_mul_int() {
    let rng = Lcg::new(21);
    for _ in 0..200 {
        let na = 1 + rng.range(120) as usize;
        let nb = 1 + rng.range(120) as usize;
        let mk = |rng: &Lcg| {
            let mut m = Int::from(rng.range(i64::MAX));
            let extra = rng.range(4);
            for _ in 0..extra {
                m = m.mul(&Int::from(rng.range(i64::MAX)));
            }
            if rng.next() & 1 == 0 { m.neg() } else { m }
        };
        let mut av: Vec<Int> = (0..na).map(|_| mk(&rng)).collect();
        av.push(Int::from(rng.range(i64::MAX) + 1));
        let mut bv: Vec<Int> = (0..nb).map(|_| mk(&rng)).collect();
        bv.push(Int::from(rng.range(i64::MAX) + 1));
        let a = Poly::new(av);
        let b = Poly::new(bv);
        // `mul` auto-dispatches (Kronecker above threshold); `mul_kronecker`
        // always uses Kronecker. Both must equal each other.
        assert_eq!(a.mul_kronecker(&b), &a * &b);
    }
}

#[test]
fn mul_kronecker_vs_mul_rational() {
    let rng = Lcg::new(22);
    for _ in 0..150 {
        let na = 1 + rng.range(90) as usize;
        let nb = 1 + rng.range(90) as usize;
        let a = prat(&rng, na);
        let b = prat(&rng, nb);
        assert_eq!(a.mul_kronecker(&b), &a * &b);
    }
}
