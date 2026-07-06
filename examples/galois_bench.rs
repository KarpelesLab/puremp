//! Wall-clock benchmarks for `GF(pᵏ)` arithmetic: `GfElement::mul`, `pow`
//! (square-and-multiply), and end-to-end Cantor–Zassenhaus factorization of a
//! polynomial over `GF(pᵏ)`, across representative `(p, k)`.
//!
//! Run with `cargo run --release --example galois_bench`. Timings are
//! meaningful only in `--release`.

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
    f(); // warm up
    for _ in 0..3 {
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

fn bench_mul(p: u64, k: usize, iters: usize) {
    let field = GaloisField::create(Int::from_u64(p), k).expect("field");
    let mut rng = Lcg::new(p ^ (k as u64));
    let elems: Vec<GfElement> = (0..64).map(|_| rand_elem(&field, k, p, &mut rng)).collect();
    let d = time(|| {
        let mut acc = field.one();
        for i in 0..iters {
            acc = acc.mul(&elems[i % elems.len()]);
        }
        std::hint::black_box(&acc);
    });
    let per = d.as_nanos() as f64 / iters as f64;
    println!("  mul  GF({p}^{k}): {per:8.1} ns/op  ({iters} ops)");
}

fn bench_pow(p: u64, k: usize, iters: usize) {
    let field = GaloisField::create(Int::from_u64(p), k).expect("field");
    let mut rng = Lcg::new(p ^ (k as u64) ^ 0xabcd);
    let base = rand_elem(&field, k, p, &mut rng);
    let exp = field.order().sub(&Int::from_u64(2)); // pᵏ − 2 (inverse exponent)
    let d = time(|| {
        for _ in 0..iters {
            std::hint::black_box(base.pow(&exp));
        }
    });
    let per = d.as_nanos() as f64 / iters as f64;
    println!("  pow  GF({p}^{k}): {per:8.1} ns/op  (exp≈p^k, {iters} ops)");
}

fn bench_factor(p: u64, k: usize, deg: usize) {
    let field = GaloisField::create(Int::from_u64(p), k).expect("field");
    let mut rng = Lcg::new(0xf00d ^ p ^ (k as u64) ^ (deg as u64));
    // Build a monic polynomial of degree `deg` over GF(p^k) as a product of a
    // few random lower-degree factors so it actually splits.
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

fn rand_elem_big(field: &GaloisField, k: usize, p: &Int, rng: &mut Lcg) -> GfElement {
    let coeffs: Vec<Int> = (0..k)
        .map(|_| {
            let hi = Int::from_u64(rng.next());
            let lo = Int::from_u64(rng.next());
            hi.mul(&Int::from_u64(1 << 32)).add(&lo).rem_euclid(p)
        })
        .collect();
    field.element(&coeffs)
}

fn bench_mul_big(p: &Int, k: usize, iters: usize) {
    let field = GaloisField::create(p.clone(), k).expect("field");
    let mut rng = Lcg::new(0x1234 ^ (k as u64));
    let elems: Vec<GfElement> = (0..64)
        .map(|_| rand_elem_big(&field, k, p, &mut rng))
        .collect();
    let d = time(|| {
        let mut acc = field.one();
        for i in 0..iters {
            acc = acc.mul(&elems[i % elems.len()]);
        }
        std::hint::black_box(&acc);
    });
    let per = d.as_nanos() as f64 / iters as f64;
    println!("  mul  GF(p[{} bits]^{k}): {per:8.1} ns/op", p.bit_len());
}

fn main() {
    // A large multi-limb prime (~130 bits) to exercise the big-p reduction path.
    let big_p = Int::from_u128(170_141_183_460_469_231_731_687_303_715_884_105_727).next_prime();
    println!("== GfElement::mul (large multi-limb p) ==");
    for &k in &[2usize, 3, 4] {
        bench_mul_big(&big_p, k, 50_000);
    }

    println!("== GfElement::mul ==");
    for &(p, k) in &[
        (2u64, 4usize),
        (2, 8),
        (3, 4),
        (5, 6),
        (7, 3),
        (101, 4),
        (65537, 4),
        (1_000_003, 3),
        (2_147_483_647, 2),
    ] {
        bench_mul(p, k, 200_000);
    }

    println!("== GfElement::pow (inverse-sized exponent) ==");
    for &(p, k) in &[
        (2u64, 8usize),
        (3, 4),
        (7, 3),
        (101, 4),
        (65537, 4),
        (1_000_003, 3),
    ] {
        bench_pow(p, k, 2_000);
    }

    println!("== end-to-end factorization ==");
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
