//! Crossover harness for factorial/binomial: the old naive sequential loop vs
//! the shipped balanced product tree (`Int::factorial` / `Int::binomial`).
//!
//! Run with `cargo run --release --example factorial_bench`. Wall-clock timings,
//! meaningful only in `--release`; the fastest of several batches is reported.

use std::time::{Duration, Instant};

use puremp::Int;

// The sequential accumulator loop the product tree replaced — kept here as the
// differential reference and timing baseline.
fn naive_factorial(n: u64) -> Int {
    let mut acc = Int::one();
    for k in 2..=n {
        acc = acc.mul(&Int::from_i64(k as i64));
    }
    acc
}

fn naive_binomial(n: u64, k: u64) -> Int {
    if k > n {
        return Int::zero();
    }
    let k = k.min(n - k);
    let mut result = Int::one();
    for i in 1..=k {
        result = result
            .mul(&Int::from_i64((n - k + i) as i64))
            .div_exact(&Int::from_i64(i as i64));
    }
    result
}

fn bench<F, R>(label: &str, iters: u32, f: F) -> Duration
where
    F: Fn() -> R,
{
    let mut best = Duration::MAX;
    let _ = f();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(f());
        }
        best = best.min(start.elapsed() / iters);
    }
    println!("{label:<28} {best:?}/iter");
    best
}

fn main() {
    println!("== factorial: naive loop vs product tree ==");
    for &(n, iters) in &[
        (4u64, 200000u32),
        (16, 100000),
        (50, 20000),
        (100, 10000),
        (1000, 500),
        (10000, 20),
        (100000, 2),
    ] {
        assert_eq!(naive_factorial(n), Int::factorial(n));
        let a = bench(&format!("naive n={n}"), iters, || naive_factorial(n));
        let b = bench(&format!("tree  n={n}"), iters, || Int::factorial(n));
        println!(
            "  speedup n={n}: {:.2}x\n",
            a.as_secs_f64() / b.as_secs_f64()
        );
    }

    println!("== binomial: naive loop vs product tree ==");
    for &(n, k, iters) in &[
        (100u64, 50u64, 5000u32),
        (1000, 500, 500),
        (10000, 100, 200),
        (10000, 5000, 10),
        (100000, 50000, 1),
    ] {
        assert_eq!(naive_binomial(n, k), Int::binomial(n, k));
        let a = bench(&format!("naive n={n} k={k}"), iters, || {
            naive_binomial(n, k)
        });
        let b = bench(&format!("tree  n={n} k={k}"), iters, || Int::binomial(n, k));
        println!(
            "  speedup n={n} k={k}: {:.2}x\n",
            a.as_secs_f64() / b.as_secs_f64()
        );
    }
}
