//! Ad-hoc throughput harness for spotting optimization choke points.
//!
//! Run with `cargo run --release --example bench`. These are wall-clock timings,
//! not statistical benchmarks — meaningful only in `--release`.

use std::time::Instant;

use puremp::{Float, Int, Nat, Rational, RoundingMode};

fn bench<F, R>(label: &str, iters: u32, f: F) -> R
where
    F: Fn() -> R,
{
    // Warm up, then time `iters` runs and report the per-iteration average.
    let mut out = f();
    let start = Instant::now();
    for _ in 0..iters {
        out = f();
    }
    let per = start.elapsed() / iters;
    println!("{label:<34} {per:?}/iter");
    out
}

fn factorial(n: u64) -> Int {
    (2..=n).fold(Int::one(), |acc, k| acc.mul(&Int::from_i64(k as i64)))
}

fn main() {
    // A big operand well past the Karatsuba threshold (~10^5 bits).
    let big = factorial(20000);
    let big2 = factorial(20001);
    println!(
        "operands: factorial(20000) has {} bits",
        big.magnitude().bit_len()
    );

    // Multiplication (schoolbook below the threshold, Karatsuba above).
    let a1000 = Int::from_i64(7).pow(1000);
    let b1000 = Int::from_i64(3).pow(1010);
    bench("mul ~1k-bit", 200, || a1000.mul(&b1000));
    bench("mul factorial(20000)^2", 5, || big.mul(&big2));

    // Division (Knuth Algorithm D).
    let prod = big.mul(&big2);
    bench("div large/large (BZ)", 5, || prod.div_rem(&big));

    // GCD (binary), on coprime-ish large operands.
    bench("gcd ~large", 20, || big.gcd(&big2));

    // Base-10 formatting (repeated single-limb division).
    bench("to_string factorial(20000)", 10, || big.to_string().len());

    // Rational arithmetic with reduction.
    let r = Rational::new(Int::from_i64(355), Int::from_i64(113));
    bench("rational add+reduce", 5000, || r.add(&r).mul(&r));

    // Integer square root.
    bench("isqrt of factorial(20000)", 20, || big.magnitude().isqrt());

    // Float transcendentals at moderate precision.
    let n = RoundingMode::Nearest;
    bench("pi @ 1000 bits", 20, || Float::pi(1000, n));
    let two = Float::from_int(&Int::from_i64(2), 1000, n);
    bench("sqrt(2) @ 1000 bits", 200, || two.sqrt(1000, n));
    bench("exp(1) @ 1000 bits", 20, || Float::e(1000, n));

    // Sanity check: a known Nat identity so the compiler can't elide everything.
    let check = Nat::from_u64(2).pow(64);
    println!("2^64 = {check}");
}
