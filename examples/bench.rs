//! Ad-hoc throughput harness for spotting optimization choke points.
//!
//! Run with `cargo run --release --example bench`. These are wall-clock timings,
//! not statistical benchmarks — meaningful only in `--release`. Each case is run
//! in several batches and the fastest batch average is reported (min is more
//! stable against scheduler noise than mean).

use std::time::{Duration, Instant};

use puremp::{Decimal, Float, Int, Rational, RoundingMode};

/// Times `iters` runs of `f`, repeated over a few batches, printing the fastest
/// batch's per-iteration time.
fn bench<F, R>(label: &str, iters: u32, f: F)
where
    F: Fn() -> R,
{
    let mut best = Duration::MAX;
    let _ = f(); // warm up
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(f());
        }
        best = best.min(start.elapsed() / iters);
    }
    println!("{label:<36} {best:?}/iter");
}

fn factorial(n: u64) -> Int {
    (2..=n).fold(Int::one(), |acc, k| acc.mul(&Int::from_i64(k as i64)))
}

fn main() {
    let n = RoundingMode::Nearest;

    // ---- integer multiplication at a range of sizes ----
    println!("== multiplication ==");
    for &bits in &[1_000u32, 10_000, 100_000] {
        let a = Int::from_i64(7).pow(bits / 3);
        let b = Int::from_i64(5).pow(bits / 3);
        let iters = if bits <= 10_000 { 500 } else { 20 };
        bench(&format!("mul ~{}k-bit", bits / 1000), iters, || a.mul(&b));
    }

    // ---- division at a range of sizes (2n / n) ----
    println!("== division (Knuth-D / Burnikel–Ziegler) ==");
    for &bits in &[1_000u32, 10_000, 100_000] {
        let d = Int::from_i64(7).pow(bits / 3);
        let num = d
            .mul(&Int::from_i64(5).pow(bits / 3))
            .add(&Int::from_i64(123));
        let iters = if bits <= 10_000 { 300 } else { 10 };
        bench(
            &format!("div ~{}k / {}k", bits / 1000, bits / 2000),
            iters,
            || num.div_rem(&d),
        );
    }

    // ---- gcd on genuinely coprime, unstructured operands ----
    println!("== gcd / modpow ==");
    let g1 = Int::from_i64(7).pow(5000); // ~14k bits
    let g2 = Int::from_i64(11).pow(4000); // ~13.8k bits, coprime with g1
    bench("gcd ~14k-bit coprime", 200, || g1.gcd(&g2));

    // modpow: an RSA-like modular exponentiation (~1.5k-bit odd modulus).
    let modulus = Int::from_i64(3).pow(1000); // odd
    let base = Int::from_i64(5).pow(500);
    let exp = Int::from_i64(7).pow(400);
    bench("modpow ~1.5k-bit", 50, || base.modpow(&exp, &modulus));

    // ---- roots & base conversion on a big value ----
    println!("== roots / base-10 I/O ==");
    let big = factorial(20_000);
    println!("  (factorial(20000) = {} bits)", big.magnitude().bit_len());
    bench("isqrt ~257k-bit", 20, || big.magnitude().isqrt());
    bench("to_string ~257k-bit", 10, || big.to_string().len());
    let decimal = big.to_string();
    bench("from_string ~257k-bit", 10, || {
        decimal.parse::<Int>().unwrap()
    });

    // ---- rationals ----
    println!("== rational ==");
    let r = Rational::new(Int::from_i64(355), Int::from_i64(113));
    bench("rational add+reduce", 5000, || r.add(&r).mul(&r));

    // ---- float ----
    println!("== float @ 1000 bits ==");
    let two = Float::from_int(&Int::from_i64(2), 1000, n);
    bench("sqrt(2)", 500, || two.sqrt(1000, n));
    bench("pi", 20, || Float::pi(1000, n));
    bench("exp(1)", 20, || Float::e(1000, n));

    // ---- derived types ----
    println!("== derived types ==");
    let da = Decimal::new(Int::from_i64(31415926535), -10);
    let db = Decimal::new(Int::from_i64(27182818284), -10);
    bench("decimal mul (exact)", 20_000, || da.mul(&db));
    bench("decimal div @ 50 digits", 5_000, || {
        da.div(&db, 50, puremp::Rounding::HalfEven)
    });

    #[cfg(feature = "poly")]
    {
        use puremp::Poly;
        let p: Poly<Rational> = Poly::new((0..=60).map(|i| Rational::from(i as i64 + 1)).collect());
        let q: Poly<Rational> = Poly::new((0..=60).map(|i| Rational::from(i as i64 + 2)).collect());
        bench("poly mul (deg 60, Q)", 200, || p.mul(&q));
    }

    #[cfg(feature = "matrix")]
    {
        use puremp::Matrix;
        let m: Matrix<Rational> = Matrix::new(
            8,
            8,
            (0..64)
                .map(|i| Rational::from(((i * 7 + 3) % 11) as i64 + 1))
                .collect(),
        );
        bench("matrix det 8x8 (Q)", 500, || m.determinant());
    }

    #[cfg(feature = "algebraic")]
    {
        use puremp::Algebraic;
        let poly = |cs: &[i64]| puremp::Poly::new(cs.iter().map(|&c| Rational::from(c)).collect());
        let r2 = Algebraic::new(poly(&[-2, 0, 1]), Rational::from(0), Rational::from(2));
        let r3 = Algebraic::new(poly(&[-3, 0, 1]), Rational::from(0), Rational::from(2));
        bench("algebraic sqrt2+sqrt3 (deg 4)", 200, || r2.add(&r3));
    }
}
