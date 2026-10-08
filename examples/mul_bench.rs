//! Wall-clock benchmark of large `Nat` multiplication and squaring through the
//! public dispatcher (`Nat::mul` / `Nat::square`), across sizes that straddle
//! the Toom → NTT hand-off and the power-of-two transform-length boundaries,
//! plus a few unbalanced shapes.
//!
//! Run with `cargo run --release --example mul_bench`. Timings are meaningful
//! only in `--release`; the fastest of several runs is reported.

use std::time::{Duration, Instant};

use puremp::Nat;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn rand_nat(rng: &mut Lcg, limbs: usize) -> Nat {
    let v: Vec<u64> = (0..limbs).map(|_| rng.next() | 1 << 63).collect();
    let bytes: Vec<u8> = v.iter().flat_map(|l| l.to_le_bytes()).collect();
    Nat::from_bytes_le(&bytes)
}

fn time<R>(f: impl Fn() -> R) -> Duration {
    let mut best = Duration::MAX;
    let _ = f(); // warm up
    let start = Instant::now();
    let mut runs = 0;
    // At least 3 runs, and keep going for ~0.3 s on the fast sizes.
    while runs < 3 || (start.elapsed() < Duration::from_millis(300) && runs < 50) {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed());
        drop(r);
        runs += 1;
    }
    best
}

fn main() {
    let mut rng = Lcg(0x5eed_1234_abcd_ef01);
    println!("{:>22} {:>14} {:>14}", "shape (limbs)", "mul", "square");
    for &n in &[
        2_000usize, 3_500, 5_000, 8_000, 12_000, 16_000, 24_000, 32_000, 48_000, 64_000, 100_000,
        131_000, 200_000, 262_000, 500_000, 1_000_000,
    ] {
        let a = rand_nat(&mut rng, n);
        let b = rand_nat(&mut rng, n);
        let m = time(|| a.mul(&b));
        let s = time(|| a.square());
        println!("{:>22} {:>14.3?} {:>14.3?}", format!("{n} x {n}"), m, s);
    }
    for &(la, lb) in &[
        (100_000usize, 10_000usize),
        (1_000_000, 50_000),
        (300_000, 150_000),
    ] {
        let a = rand_nat(&mut rng, la);
        let b = rand_nat(&mut rng, lb);
        let m = time(|| a.mul(&b));
        println!("{:>22} {:>14.3?}", format!("{la} x {lb}"), m);
    }
}
