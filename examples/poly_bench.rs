//! Wall-clock benchmarks for the subquadratic polynomial paths:
//! Kronecker multiplication, Newton division, and Half-GCD, each compared
//! against a local schoolbook/Euclid baseline at increasing degree so the
//! crossover and speedup are visible.
//!
//! Run with `cargo run --release --example poly_bench`. Timings are meaningful
//! only in `--release`.

use std::time::{Duration, Instant};

use puremp::Ring;
use puremp::{Int, ModInt, Poly};

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg(seed ^ 0x9e3779b97f4a7c15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn time<F: FnMut()>(mut f: F) -> Duration {
    let mut best = Duration::MAX;
    f(); // warm up
    for _ in 0..3 {
        let start = Instant::now();
        f();
        best = best.min(start.elapsed());
    }
    best
}

// ---- Int polynomials for Kronecker vs schoolbook ----

fn rand_int_poly(rng: &mut Lcg, n: usize, bits: u32) -> Poly<Int> {
    let mk = |rng: &mut Lcg| {
        let mut m = Int::from((rng.next() >> 1) as i64);
        for _ in 0..(bits / 62) {
            m = m.mul(&Int::from((rng.next() >> 1) as i64));
        }
        if rng.next() & 1 == 0 { m.neg() } else { m }
    };
    let mut v: Vec<Int> = (0..n).map(|_| mk(rng)).collect();
    v.push(Int::from(((rng.next() >> 1) as i64) | 1));
    Poly::new(v)
}

fn school_mul_int(a: &Poly<Int>, b: &Poly<Int>) -> Poly<Int> {
    let (ac, bc) = (a.coeffs(), b.coeffs());
    let mut out = vec![Int::ZERO; ac.len() + bc.len() - 1];
    for (i, x) in ac.iter().enumerate() {
        for (j, y) in bc.iter().enumerate() {
            out[i + j] = out[i + j].add(&x.mul(y));
        }
    }
    Poly::new(out)
}

fn bench_kronecker() {
    println!("== Kronecker multiplication (Poly<Int>, ~256-bit coeffs) ==");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>8}",
        "deg", "schoolbook", "kronecker", "speedup"
    );
    let mut rng = Lcg::new(1);
    for &n in &[32usize, 64, 128, 256, 512, 1024] {
        let a = rand_int_poly(&mut rng, n, 256);
        let b = rand_int_poly(&mut rng, n, 256);
        let t_school = time(|| {
            std::hint::black_box(school_mul_int(&a, &b));
        });
        let t_kron = time(|| {
            std::hint::black_box(a.mul_kronecker(&b));
        });
        assert_eq!(school_mul_int(&a, &b), a.mul_kronecker(&b));
        println!(
            "{n:>6}  {:>12?}  {:>12?}  {:>7.2}x",
            t_school,
            t_kron,
            t_school.as_secs_f64() / t_kron.as_secs_f64()
        );
    }
}

// ---- GF(p) polynomials for division and gcd ----

const P: u64 = 1_000_000_007;

fn rand_mod_poly(rng: &mut Lcg, n: usize) -> Poly<ModInt> {
    let p = Int::from_u64(P);
    let mut v: Vec<ModInt> = (0..n)
        .map(|_| ModInt::new(Int::from_u64(rng.next() % P), p.clone()))
        .collect();
    v.push(ModInt::new(
        Int::from_u64((rng.next() % (P - 1)) + 1),
        p.clone(),
    ));
    Poly::new(v)
}

fn school_divrem(a: &Poly<ModInt>, b: &Poly<ModInt>) -> (Poly<ModInt>, Poly<ModInt>) {
    let dd = b.degree().unwrap();
    let lead = b.leading().unwrap().clone();
    let inv = lead.inv().unwrap();
    let mut rem: Vec<ModInt> = a.coeffs().to_vec();
    let zero = lead.zero();
    let mut quot = vec![zero.clone(); a.coeffs().len().saturating_sub(dd)];
    while let Some(rd) = rem.iter().rposition(|c| !c.is_zero()) {
        if rd < dd {
            break;
        }
        let coef = rem[rd].clone() * inv.clone();
        let shift = rd - dd;
        for (i, dc) in b.coeffs().iter().enumerate() {
            rem[shift + i] = rem[shift + i].clone() - coef.clone() * dc.clone();
        }
        quot[shift] = coef;
    }
    (Poly::new(quot), Poly::new(rem))
}

fn euclid_gcd(a: &Poly<ModInt>, b: &Poly<ModInt>) -> Poly<ModInt> {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
        let r = school_divrem(&a, &b).1;
        a = b;
        b = r;
    }
    a.monic()
}

fn bench_division() {
    println!("\n== Newton division (Poly<GF(p)>, dividend 2·deg / divisor deg) ==");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>8}",
        "deg", "schoolbook", "newton", "speedup"
    );
    let mut rng = Lcg::new(2);
    for &d in &[64usize, 128, 256, 512, 1024, 2048] {
        let a = rand_mod_poly(&mut rng, 2 * d);
        let b = rand_mod_poly(&mut rng, d);
        let t_school = time(|| {
            std::hint::black_box(school_divrem(&a, &b));
        });
        let t_newton = time(|| {
            std::hint::black_box(a.div_rem(&b));
        });
        assert_eq!(school_divrem(&a, &b), a.div_rem(&b));
        println!(
            "{d:>6}  {:>12?}  {:>12?}  {:>7.2}x",
            t_school,
            t_newton,
            t_school.as_secs_f64() / t_newton.as_secs_f64()
        );
    }
}

fn bench_gcd() {
    println!("\n== Half-GCD (Poly<GF(p)>, coprime random pair) ==");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>8}",
        "deg", "euclid", "half-gcd", "speedup"
    );
    let mut rng = Lcg::new(3);
    for &d in &[64usize, 128, 256, 512, 1024, 2048] {
        let a = rand_mod_poly(&mut rng, d);
        let b = rand_mod_poly(&mut rng, d - 1);
        let t_euclid = time(|| {
            std::hint::black_box(euclid_gcd(&a, &b));
        });
        let t_hgcd = time(|| {
            std::hint::black_box(a.gcd(&b));
        });
        assert_eq!(euclid_gcd(&a, &b), a.gcd(&b));
        println!(
            "{d:>6}  {:>12?}  {:>12?}  {:>7.2}x",
            t_euclid,
            t_hgcd,
            t_euclid.as_secs_f64() / t_hgcd.as_secs_f64()
        );
    }
}

fn main() {
    bench_kronecker();
    bench_division();
    bench_gcd();
}
