//! Wall-clock benchmarks for the subquadratic polynomial paths:
//! Kronecker multiplication, Newton division, and Half-GCD, each compared
//! against a local schoolbook/Euclid baseline at increasing degree so the
//! crossover and speedup are visible.
//!
//! Run with `cargo run --release --example poly_bench`. Timings are meaningful
//! only in `--release`.

use std::time::{Duration, Instant};

use puremp::Ring;
use puremp::{FactorOverField, Int, ModInt, Poly};

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

// ---- Karatsuba vs Kronecker for Poly<ModInt> over GF(p) ----

fn rand_mod_poly_p(rng: &mut Lcg, n: usize, p: &Int) -> Poly<ModInt> {
    // Sample residues in [0, p) by rejection from a power-of-two-ish range built
    // out of 62-bit LCG words.
    let sample = ModInt::new(Int::ZERO, p.clone());
    let words = (p.bit_len() / 62 + 1) as usize;
    let mk = |rng: &mut Lcg| {
        let mut m = Int::from((rng.next() >> 1) as i64);
        for _ in 0..words {
            m = m
                .mul(&Int::from((rng.next() >> 1) as i64))
                .add(&Int::from((rng.next() >> 1) as i64));
        }
        sample.of(m)
    };
    let mut v: Vec<ModInt> = (0..n).map(|_| mk(rng)).collect();
    // Nonzero leading coefficient.
    let mut lead = mk(rng);
    while lead.is_zero() {
        lead = mk(rng);
    }
    v.push(lead);
    Poly::new(v)
}

/// Honest Karatsuba baseline over GF(p), independent of the `Poly::mul` hook,
/// bottoming out in schoolbook. This mirrors the pre-change `Poly<ModInt>::mul`.
fn karatsuba_mod(a: &[ModInt], b: &[ModInt]) -> Vec<ModInt> {
    let n = a.len().min(b.len());
    if n < 24 {
        let zero = a[0].zero();
        let mut out = vec![zero; a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                out[i + j] = out[i + j].clone() + x.clone() * y.clone();
            }
        }
        return out;
    }
    let m = a.len().max(b.len()) / 2;
    let split = |s: &[ModInt]| -> (Vec<ModInt>, Vec<ModInt>) {
        if m >= s.len() {
            (s.to_vec(), Vec::new())
        } else {
            (s[..m].to_vec(), s[m..].to_vec())
        }
    };
    let add = |x: &[ModInt], y: &[ModInt]| -> Vec<ModInt> {
        let mut out = x.to_vec();
        if out.len() < y.len() {
            out.resize(y.len(), a[0].zero());
        }
        for (i, c) in y.iter().enumerate() {
            out[i] = out[i].clone() + c.clone();
        }
        out
    };
    let (a0, a1) = split(a);
    let (b0, b1) = split(b);
    let z0 = karatsuba_mod(&a0, &b0);
    let use_hi = !a1.is_empty() && !b1.is_empty();
    let z2 = if use_hi {
        karatsuba_mod(&a1, &b1)
    } else {
        Vec::new()
    };
    let sa = add(&a0, &a1);
    let sb = add(&b0, &b1);
    let mid = karatsuba_mod(&sa, &sb);
    let mut out = vec![a[0].zero(); a.len() + b.len() - 1];
    for (i, c) in z0.iter().enumerate() {
        out[i] = out[i].clone() + c.clone();
    }
    for (i, c) in mid.iter().enumerate() {
        out[i + m] = out[i + m].clone() + c.clone();
    }
    for (i, c) in z0.iter().enumerate() {
        out[i + m] = out[i + m].clone() - c.clone();
    }
    for (i, c) in z2.iter().enumerate() {
        out[i + m] = out[i + m].clone() - c.clone();
        out[i + 2 * m] = out[i + 2 * m].clone() + c.clone();
    }
    out
}

fn bench_kronecker_modint() {
    for (name, p) in &[
        ("word prime 1e9+7", Int::from_u64(1_000_000_007)),
        (
            "128-bit prime",
            Int::from_u64(1_000_000_007)
                .mul(&Int::from_u64(1_000_000_009))
                .mul(&Int::from_u64(1_000_000_021))
                .mul(&Int::from_u64(9))
                .add(&Int::from_u64(1)),
        ),
    ] {
        let p = p.clone();
        println!(
            "\n== Poly<GF(p)> multiply: Karatsuba vs Kronecker ({name}, {} bits) ==",
            p.bit_len()
        );
        println!(
            "{:>6}  {:>12}  {:>12}  {:>8}",
            "deg", "karatsuba", "kronecker", "speedup"
        );
        let mut rng = Lcg::new(7);
        for &n in &[32usize, 64, 128, 256, 512, 1024, 2048] {
            let a = rand_mod_poly_p(&mut rng, n, &p);
            let b = rand_mod_poly_p(&mut rng, n, &p);
            let t_kara = time(|| {
                std::hint::black_box(karatsuba_mod(a.coeffs(), b.coeffs()));
            });
            let t_kron = time(|| {
                std::hint::black_box(a.mul_kronecker(&b));
            });
            assert_eq!(
                Poly::new(karatsuba_mod(a.coeffs(), b.coeffs())),
                a.mul_kronecker(&b)
            );
            println!(
                "{n:>6}  {:>12?}  {:>12?}  {:>7.2}x",
                t_kara,
                t_kron,
                t_kara.as_secs_f64() / t_kron.as_secs_f64()
            );
        }
    }
}

fn bench_cz_factor() {
    println!("\n== End-to-end Cantor–Zassenhaus factorization (Poly<GF(1e9+7)>) ==");
    println!("{:>6}  {:>14}", "deg", "factor()");
    let p = Int::from_u64(1_000_000_007);
    let mut rng = Lcg::new(11);
    for &deg in &[64usize, 128, 256, 512] {
        // Build a product of random low-degree factors so the polynomial really
        // factors and DDF/EDF do work.
        let mut f = Poly::constant(ModInt::new(Int::ONE, p.clone()));
        while f.degree().unwrap_or(0) < deg {
            let sz = 3 + (rng.next() as usize % 6);
            let g = rand_mod_poly_p(&mut rng, sz, &p);
            f = f.mul(&g);
        }
        let t = time(|| {
            std::hint::black_box(f.factor());
        });
        println!("{:>6}  {:>14?}", f.degree().unwrap(), t);
    }
}

fn main() {
    bench_kronecker();
    bench_division();
    bench_gcd();
    bench_kronecker_modint();
    bench_cz_factor();
}
