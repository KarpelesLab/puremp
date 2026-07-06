//! Baseline sin/cos timing across precisions. `cargo run --release --example trig_bench`.
use std::time::{Duration, Instant};

use puremp::{Float, Int, RoundingMode};

fn bench<F, R>(label: &str, iters: u32, f: F)
where
    F: Fn() -> R,
{
    let mut best = Duration::MAX;
    let _ = f();
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(f());
        }
        best = best.min(start.elapsed() / iters);
    }
    println!("{label:<40} {best:?}/iter");
}

fn main() {
    let n = RoundingMode::Nearest;
    // A "small" argument near 1 and a "large" one needing big quadrant reduction.
    for &prec in &[256u64, 1000, 4000, 16000, 64000] {
        // x ~ 1.2345 (small), and x ~ 2^40 * 1.2345 (large reduction).
        let small = Float::from_int(&Int::from_i64(12345), prec + 64, n).div(
            &Float::from_int(&Int::from_i64(10000), prec + 64, n),
            prec + 64,
            n,
        );
        let big = small.mul(
            &Float::from_int(&Int::from_i64(1i64 << 40), prec + 64, n),
            prec + 64,
            n,
        );
        let iters = match prec {
            256 => 2000,
            1000 => 500,
            4000 => 100,
            16000 => 20,
            _ => 4,
        };
        bench(&format!("sin small @ {prec}"), iters, || small.sin(prec, n));
        bench(&format!("cos small @ {prec}"), iters, || small.cos(prec, n));
        bench(&format!("sin big   @ {prec}"), iters, || big.sin(prec, n));
    }
}
