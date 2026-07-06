//! End-to-end throughput of the real `Rational` API: raw add/mul/div plus a
//! `Matrix<Rational>` inverse (rational Gaussian elimination, which hammers
//! Rational div/mul/sub). Run before and after a change to compare.
//!
//! Run with `cargo run --release --example rational_e2e_bench`.

use std::time::{Duration, Instant};

use puremp::{Int, Matrix, Rational};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn int(&mut self, words: usize) -> Int {
        let mut v = Int::ZERO;
        let shift = Int::ONE.mul_2k(64);
        for _ in 0..words.max(1) {
            v = v.mul(&shift).add(&Int::from_u64(self.next()));
        }
        if self.next() & 1 == 0 { v.neg() } else { v }
    }
    fn rat(&mut self, words: usize) -> Rational {
        let n = self.int(words);
        let d = self.int(words).abs().add(&Int::ONE);
        Rational::new(n, d)
    }
}

fn bench<F: FnMut() -> R, R>(label: &str, iters: u32, mut f: F) {
    let mut best = Duration::MAX;
    let _ = f();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(f());
        }
        best = best.min(start.elapsed() / iters);
    }
    println!("{label:<34} {best:?}/iter");
}

fn main() {
    let mut rng = Rng(0xDEADBEEFCAFEBABE);

    println!("== raw Rational ops ==");
    for &(label, words, iters) in &[
        ("~64b", 1usize, 100_000u32),
        ("~512b", 8, 20_000),
        ("~2kb", 32, 1_500),
    ] {
        let fs: Vec<Rational> = (0..64).map(|_| rng.rat(words)).collect();
        let len = fs.len();
        let mut c = 0usize;
        bench(&format!("add {label}"), iters, || {
            let r = fs[c % len].add(&fs[(c + 1) % len]);
            c += 1;
            r
        });
        let mut c = 0usize;
        bench(&format!("mul {label}"), iters, || {
            let r = fs[c % len].mul(&fs[(c + 1) % len]);
            c += 1;
            r
        });
        let mut c = 0usize;
        bench(&format!("div {label}"), iters, || {
            let r = fs[c % len].div(&fs[(c + 1) % len]);
            c += 1;
            r
        });
    }

    println!("\n== Matrix<Rational> inverse (rational elimination) ==");
    for &(n, iters) in &[(6usize, 400u32), (10, 60), (14, 12)] {
        // Small-integer entries so growth comes from the elimination itself.
        let data: Vec<Rational> = (0..n * n)
            .map(|_| Rational::new(Int::from_i64((rng.next() % 19) as i64 - 9), Int::ONE))
            .collect();
        let m = Matrix::<Rational>::new(n, n, data);
        // Skip singular draws.
        if m.inverse().is_none() {
            println!("inverse {n}x{n}: singular draw, skipped");
            continue;
        }
        bench(&format!("inverse {n}x{n}"), iters, || m.inverse());
    }

    println!("\n== Matrix<Rational> inverse with fractional entries ==");
    for &(n, iters) in &[(6usize, 200u32), (10, 30)] {
        let data: Vec<Rational> = (0..n * n)
            .map(|_| {
                Rational::new(
                    Int::from_i64((rng.next() % 15) as i64 - 7),
                    Int::from_i64((rng.next() % 6) as i64 + 1),
                )
            })
            .collect();
        let m = Matrix::<Rational>::new(n, n, data);
        if m.inverse().is_none() {
            println!("inverse {n}x{n} (frac): singular draw, skipped");
            continue;
        }
        bench(&format!("inverse {n}x{n} (frac)"), iters, || m.inverse());
    }
}
