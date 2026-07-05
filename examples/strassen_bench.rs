//! Strassen–Winograd vs naive matrix multiply, over `Int` and `Rational`.
//!
//! Run with `cargo run --release --example strassen_bench`. These are wall-clock
//! timings (meaningful only in `--release`), reported as the fastest of a few
//! batches. For each size the Strassen product is also checked bit-for-bit
//! against the naive one.
//!
//! [`Matrix::mul`] auto-dispatches to Strassen only for exact rings once the
//! dimension exceeds its threshold *and* the entries are large enough that
//! saving a multiply outweighs the extra additions (≈1024-bit). This harness
//! therefore sweeps entry size to expose the crossover: below it Strassen loses
//! (and `mul` correctly stays naive); above it Strassen wins, more so as the
//! entries and the dimension grow.

use std::time::Instant;

use puremp::{Int, Matrix, Rational};

/// Deterministic LCG, so runs are reproducible.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

/// Reference naive triple-loop product (the algorithm `Matrix::mul` falls back
/// to below the threshold), used here as the baseline to time against.
fn naive<T: puremp::Ring>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    let zero = a.get(0, 0).zero();
    let mut d = alloc_vec(zero, m * n);
    for i in 0..m {
        for kk in 0..k {
            let av = a.get(i, kk).clone();
            for j in 0..n {
                d[i * n + j] = d[i * n + j].clone() + av.clone() * b.get(kk, j).clone();
            }
        }
    }
    Matrix::new(m, n, d)
}

fn alloc_vec<T: Clone>(v: T, n: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(n);
    out.resize(n, v);
    out
}

/// Times `reps` runs of `f`, returning the fastest per-iteration duration.
fn time<F: FnMut()>(reps: u32, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    best
}

fn rand_int(r: &mut Lcg, big: &Int) -> Int {
    big * &Int::from_i64((r.next() % 1000) as i64 + 1)
}

fn main() {
    let mut r = Lcg(0xDEADBEEF);

    println!("== Int: sweep entry size (n = 48) ==");
    println!("  showing the crossover; `mul` engages Strassen only ≥ ~1024 bits");
    for &digits in &[60u32, 150, 300, 600, 1200] {
        let big = Int::from(10).pow(digits);
        let n = 48;
        let a = Matrix::new(n, n, (0..n * n).map(|_| rand_int(&mut r, &big)).collect());
        let b = Matrix::new(n, n, (0..n * n).map(|_| rand_int(&mut r, &big)).collect());
        assert_eq!(a.mul(&b), naive(&a, &b));
        let reps = if digits <= 300 { 4 } else { 2 };
        let tn = time(reps, || {
            std::hint::black_box(naive(&a, &b));
        });
        let ts = time(reps, || {
            std::hint::black_box(a.mul(&b));
        });
        println!(
            "  ~{:5}bit  naive {:>9.3}ms  mul {:>9.3}ms  speedup {:.2}x",
            (digits as f64 * 3.322) as u32,
            tn * 1e3,
            ts * 1e3,
            tn / ts
        );
    }

    println!("== Int: sweep dimension (~2000-bit entries) ==");
    let big = Int::from(10).pow(600);
    for &n in &[32usize, 48, 64] {
        let a = Matrix::new(n, n, (0..n * n).map(|_| rand_int(&mut r, &big)).collect());
        let b = Matrix::new(n, n, (0..n * n).map(|_| rand_int(&mut r, &big)).collect());
        assert_eq!(a.mul(&b), naive(&a, &b));
        let reps = if n <= 32 { 4 } else { 2 };
        let tn = time(reps, || {
            std::hint::black_box(naive(&a, &b));
        });
        let ts = time(reps, || {
            std::hint::black_box(a.mul(&b));
        });
        println!(
            "  n={:4}  naive {:>9.3}ms  mul {:>9.3}ms  speedup {:.2}x",
            n,
            tn * 1e3,
            ts * 1e3,
            tn / ts
        );
    }

    println!("== Rational: large reduced fractions (~1000-bit num & den) ==");
    let numbase = Int::from(10).pow(300); // ~1000-bit numerators
    let den = Int::from(2).pow(1000); // power of two ⇒ coprime with odd numerators
    for &n in &[32usize, 48] {
        let mk = |r: &mut Lcg| {
            let num = &(&numbase * &Int::from_i64((r.next() % 1000) as i64 * 2 + 1)) + &Int::ONE;
            Rational::new(num, den.clone())
        };
        let a = Matrix::new(n, n, (0..n * n).map(|_| mk(&mut r)).collect());
        let b = Matrix::new(n, n, (0..n * n).map(|_| mk(&mut r)).collect());
        assert_eq!(a.mul(&b), naive(&a, &b));
        let tn = time(2, || {
            std::hint::black_box(naive(&a, &b));
        });
        let ts = time(2, || {
            std::hint::black_box(a.mul(&b));
        });
        println!(
            "  n={:4}  naive {:>9.3}ms  mul {:>9.3}ms  speedup {:.2}x",
            n,
            tn * 1e3,
            ts * 1e3,
            tn / ts
        );
    }
}
