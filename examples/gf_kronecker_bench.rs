//! Karatsuba vs nested-Kronecker `Poly<GfElement>::mul` over `GF(pᵏ)`, plus
//! end-to-end Cantor–Zassenhaus factorization. Run with
//! `cargo run --release --example gf_kronecker_bench`. Timings mean nothing in
//! debug.

use std::time::{Duration, Instant};

use puremp::{FactorOverField, GaloisField, GfElement, Int, Poly};

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
    f();
    for _ in 0..5 {
        let start = Instant::now();
        f();
        best = best.min(start.elapsed());
    }
    best
}

fn rand_elem(field: &GaloisField, k: usize, p: u64, rng: &mut Lcg) -> GfElement {
    let coeffs: Vec<Int> = (0..k).map(|_| Int::from_u64(rng.next() % p)).collect();
    field.element(&coeffs)
}

fn rand_poly(field: &GaloisField, k: usize, p: u64, n: usize, rng: &mut Lcg) -> Poly<GfElement> {
    Poly::new((0..n).map(|_| rand_elem(field, k, p, rng)).collect())
}

fn bench_mul(p: u64, k: usize, deg: usize) {
    let field = GaloisField::create(Int::from_u64(p), k).expect("field");
    let mut rng = Lcg::new(p ^ (k as u64) ^ (deg as u64));
    let a = rand_poly(&field, k, p, deg, &mut rng);
    let b = rand_poly(&field, k, p, deg, &mut rng);
    // Correctness guard.
    assert_eq!(a.mul_no_hook(&b), a.mul_nested_kronecker(&b));
    let iters = (2_000_000 / (deg * deg)).max(2);
    let kar = time(|| {
        for _ in 0..iters {
            std::hint::black_box(a.mul_no_hook(&b));
        }
    });
    let kro = time(|| {
        for _ in 0..iters {
            std::hint::black_box(a.mul_nested_kronecker(&b));
        }
    });
    let kar = kar.as_secs_f64() / iters as f64 * 1e6;
    let kro = kro.as_secs_f64() / iters as f64 * 1e6;
    println!(
        "  GF({p}^{k}) deg {deg:4}: karatsuba {kar:9.2} us  kronecker {kro:9.2} us  speedup {:.2}x",
        kar / kro
    );
}

fn bench_factor(p: u64, k: usize, deg: usize) {
    let field = GaloisField::create(Int::from_u64(p), k).expect("field");
    let mut rng = Lcg::new(0xf00d ^ p ^ (k as u64) ^ (deg as u64));
    let mut poly = Poly::constant(field.one());
    while poly.degree().unwrap_or(0) < deg {
        let fac_deg = 1 + (rng.next() as usize % 3);
        let coeffs: Vec<GfElement> = (0..=fac_deg)
            .map(|_| rand_elem(&field, k, p, &mut rng))
            .collect();
        let mut fac = Poly::new(coeffs);
        if fac.degree().unwrap_or(0) == 0 {
            continue;
        }
        fac = fac.monic();
        poly = poly.mul(&fac);
    }
    let d = time(|| {
        std::hint::black_box(poly.factor());
    });
    println!(
        "  factor GF({p}^{k}) deg {}: {:8.2} ms",
        poly.degree().unwrap_or(0),
        d.as_secs_f64() * 1e3
    );
}

fn main() {
    println!("== Poly<GfElement>::mul: Karatsuba vs nested Kronecker ==");
    for &(p, k) in &[(2u64, 4usize), (3, 4), (5, 6), (7, 3), (101, 4), (65537, 4)] {
        for &deg in &[16usize, 32, 64, 128, 256, 512] {
            bench_mul(p, k, deg);
        }
        println!();
    }

    println!("== end-to-end factorization (mul routes through the hook) ==");
    for &(p, k, deg) in &[
        (2u64, 4usize, 20usize),
        (3, 3, 16),
        (5, 4, 16),
        (7, 3, 20),
        (101, 3, 12),
        (65537, 2, 12),
    ] {
        bench_factor(p, k, deg);
    }
}
