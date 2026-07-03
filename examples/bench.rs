//! Ad-hoc throughput harness used to spot optimization choke points while the
//! sub-quadratic algorithms in `ROADMAP.md` land.
//!
//! Run with `cargo run --release --example bench`. These are wall-clock timings,
//! not statistical benchmarks — the numbers are only meaningful in `--release`.

use std::time::Instant;

use puremp::Int;

/// Computes `n!` by an ascending product.
fn factorial(n: u64) -> Int {
    let mut acc = Int::one();
    for k in 2..=n {
        acc = acc.mul(&Int::from_i64(k as i64));
    }
    acc
}

fn bench<F, R>(label: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let out = f();
    println!("{label:<28} {:?}", start.elapsed());
    out
}

fn main() {
    let f2000 = bench("factorial(2000)", || factorial(2000));
    println!("  digits: {}", f2000.to_string().len());

    let big = bench("factorial(20000)", || factorial(20000));
    println!("  digits: {}", big.to_string().len());

    bench("2^1000000", || Int::from_i64(2).pow(1_000_000));

    // Multiply two large integers (schoolbook, for now).
    let a = factorial(5000);
    let b = factorial(5001);
    bench("mul(5000!, 5001!)", || a.mul(&b));
}
