//! Old (term-by-term) vs new (binary-split) Catalan and Euler–Mascheroni
//! constants across precisions. `cargo run --release --example const_bench`.
//! Pass a maximum precision in bits as the first argument to cap the sweep
//! (default 262144); the old O(n²)-ish paths get very slow at the top end.
use std::time::{Duration, Instant};

use puremp::{Float, RoundingMode};

fn time<F: Fn() -> Float>(f: F) -> Duration {
    // Best of a few runs, with the repetition count scaled to the cost.
    let start = Instant::now();
    std::hint::black_box(f());
    let first = start.elapsed();
    let reps = if first < Duration::from_millis(20) {
        (Duration::from_millis(200).as_nanos() / first.as_nanos().max(1)).clamp(1, 10_000) as u32
    } else if first < Duration::from_secs(2) {
        3
    } else {
        return first;
    };
    let mut best = first;
    for _ in 0..3 {
        let s = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(f());
        }
        best = best.min(s.elapsed() / reps);
    }
    best
}

fn main() {
    let max: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 18);
    let m = RoundingMode::Nearest;
    // Fine scan around the low-precision crossovers.
    for p in (32u64..=320).step_by(32) {
        let co = time(|| Float::catalan_series_reference(p, m));
        let cn = time(|| Float::catalan_bsplit(p, m));
        let go = time(|| Float::euler_gamma_series_reference(p, m));
        let gn = time(|| Float::euler_gamma_bsplit(p, m));
        println!(
            "scan {p:>4}: catalan {:.2}x  gamma {:.2}x",
            co.as_secs_f64() / cn.as_secs_f64(),
            go.as_secs_f64() / gn.as_secs_f64()
        );
    }
    println!(
        "{:>8} | {:>12} {:>12} {:>7} | {:>12} {:>12} {:>7}",
        "bits", "catalan old", "catalan new", "speedup", "gamma old", "gamma new", "speedup"
    );
    for &p in &[64u64, 256, 1024, 4096, 16384, 65536, 262144] {
        if p > max {
            break;
        }
        assert_eq!(
            Float::catalan_series_reference(p, m),
            Float::catalan_bsplit(p, m)
        );
        assert_eq!(
            Float::euler_gamma_series_reference(p, m),
            Float::euler_gamma_bsplit(p, m)
        );
        let co = time(|| Float::catalan_series_reference(p, m));
        let cn = time(|| Float::catalan_bsplit(p, m));
        let go = time(|| Float::euler_gamma_series_reference(p, m));
        let gn = time(|| Float::euler_gamma_bsplit(p, m));
        println!(
            "{p:>8} | {co:>12.2?} {cn:>12.2?} {:>6.2}x | {go:>12.2?} {gn:>12.2?} {:>6.2}x",
            co.as_secs_f64() / cn.as_secs_f64(),
            go.as_secs_f64() / gn.as_secs_f64()
        );
    }
}
