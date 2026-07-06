//! Micro-benchmark: current vs cross-reduced `Rational` add/sub/mul/div.
//!
//! Reimplements both the CURRENT algorithm (full product then one gcd) and the
//! CROSS-REDUCED algorithm (Henrici add/sub, cross-cancel mul/div) directly on
//! `(Int, Int)` pairs via the public `Int` API, so the two can be compared
//! head-to-head before any change is wired into `Rational`.
//!
//! Run with `cargo run --release --example rational_xreduce_bench`.

use std::time::{Duration, Instant};

use puremp::Int;

// ---- tiny xorshift RNG for reproducible operands ----
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
    /// Random Int with roughly `words * 64` bits and a random sign.
    fn int(&mut self, words: usize) -> Int {
        let mut v = Int::ZERO;
        let shift = Int::ONE.mul_2k(64);
        for _ in 0..words.max(1) {
            v = v.mul(&shift).add(&Int::from_u64(self.next()));
        }
        if self.next() & 1 == 0 { v.neg() } else { v }
    }
}

/// A canonical reduced fraction (den > 0, coprime), stored as a plain pair.
#[derive(Clone)]
struct Frac {
    num: Int,
    den: Int,
}

fn reduce(mut num: Int, mut den: Int) -> Frac {
    if den.is_negative() {
        num = num.neg();
        den = den.neg();
    }
    let g = num.gcd(&den);
    if !g.is_one() {
        num = num.div_exact(&g);
        den = den.div_exact(&g);
    }
    Frac { num, den }
}

fn make_fracs(rng: &mut Rng, words: usize, n: usize, shared: bool) -> Vec<Frac> {
    (0..n)
        .map(|_| {
            let mut num = rng.int(words);
            let mut den = rng.int(words.max(1)).abs().add(&Int::ONE);
            if shared {
                // Inject a shared factor so cross-reduction has something to cancel.
                let g = rng.int((words / 2).max(1)).abs().add(&Int::ONE);
                num = num.mul(&g);
                den = den.mul(&g);
            }
            reduce(num, den)
        })
        .collect()
}

// ---- CURRENT algorithms ----
fn old_add(a: &Frac, b: &Frac) -> Frac {
    let num = a.num.mul(&b.den).add(&b.num.mul(&a.den));
    let den = a.den.mul(&b.den);
    reduce(num, den)
}
fn old_mul(a: &Frac, b: &Frac) -> Frac {
    reduce(a.num.mul(&b.num), a.den.mul(&b.den))
}
fn old_div(a: &Frac, b: &Frac) -> Frac {
    reduce(a.num.mul(&b.den), a.den.mul(&b.num))
}

// ---- CROSS-REDUCED algorithms ----
fn new_add(a: &Frac, b: &Frac) -> Frac {
    if a.num.is_zero() {
        return b.clone();
    }
    if b.num.is_zero() {
        return a.clone();
    }
    let g1 = a.den.gcd(&b.den); // gcd(b, d) > 0
    if g1.is_one() {
        let num = a.num.mul(&b.den).add(&b.num.mul(&a.den));
        let den = a.den.mul(&b.den);
        if num.is_zero() {
            return Frac {
                num: Int::ZERO,
                den: Int::ONE,
            };
        }
        return Frac { num, den };
    }
    let d_over = b.den.div_exact(&g1);
    let b_over = a.den.div_exact(&g1);
    let t = a.num.mul(&d_over).add(&b.num.mul(&b_over));
    if t.is_zero() {
        return Frac {
            num: Int::ZERO,
            den: Int::ONE,
        };
    }
    let den = a.den.mul(&d_over);
    let g2 = t.gcd(&g1);
    if g2.is_one() {
        return Frac { num: t, den };
    }
    Frac {
        num: t.div_exact(&g2),
        den: den.div_exact(&g2),
    }
}
fn new_mul(a: &Frac, b: &Frac) -> Frac {
    if a.num.is_zero() || b.num.is_zero() {
        return Frac {
            num: Int::ZERO,
            den: Int::ONE,
        };
    }
    let g1 = a.num.gcd(&b.den);
    let g2 = b.num.gcd(&a.den);
    let red = |x: &Int, g: &Int| {
        if g.is_one() {
            x.clone()
        } else {
            x.div_exact(g)
        }
    };
    let num = red(&a.num, &g1).mul(&red(&b.num, &g2));
    let den = red(&a.den, &g2).mul(&red(&b.den, &g1));
    Frac { num, den }
}
fn new_div(a: &Frac, b: &Frac) -> Frac {
    if a.num.is_zero() {
        return Frac {
            num: Int::ZERO,
            den: Int::ONE,
        };
    }
    let g1 = a.num.gcd(&b.num);
    let g2 = b.den.gcd(&a.den);
    let red = |x: &Int, g: &Int| {
        if g.is_one() {
            x.clone()
        } else {
            x.div_exact(g)
        }
    };
    let num = red(&a.num, &g1).mul(&red(&b.den, &g2));
    let den = red(&a.den, &g2).mul(&red(&b.num, &g1));
    if den.is_negative() {
        Frac {
            num: num.neg(),
            den: den.neg(),
        }
    } else {
        Frac { num, den }
    }
}

fn bench<F: FnMut() -> R, R>(label: &str, iters: u32, mut f: F) -> Duration {
    let mut best = Duration::MAX;
    let _ = f();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(f());
        }
        best = best.min(start.elapsed() / iters);
    }
    println!("{label:<40} {best:?}/iter");
    best
}

fn assert_same(
    fs: &[Frac],
    op_old: impl Fn(&Frac, &Frac) -> Frac,
    op_new: impl Fn(&Frac, &Frac) -> Frac,
) {
    for w in fs.windows(2) {
        let o = op_old(&w[0], &w[1]);
        let n = op_new(&w[0], &w[1]);
        assert!(o.num == n.num && o.den == n.den, "mismatch");
    }
}

fn main() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    // (label, words ~ bits/64, iters)
    let sizes = [
        ("tiny (~64b)", 1usize, 200_000u32),
        ("small (~256b)", 4, 60_000),
        ("medium (~1kb)", 16, 8_000),
        ("large (~4kb)", 64, 600),
    ];
    for shared in [false, true] {
        println!(
            "\n================ shared factor: {} ================",
            shared
        );
        for &(label, words, iters) in &sizes {
            let fs = make_fracs(&mut rng, words, 64, shared);
            // correctness
            assert_same(&fs, old_add, new_add);
            assert_same(&fs, old_mul, new_mul);
            assert_same(&fs, old_div, new_div);
            println!("-- {label} --");
            let len = fs.len();
            macro_rules! run {
                ($name:expr, $f:expr) => {{
                    let mut c = 0usize;
                    bench(&format!("{} {label}", $name), iters, || {
                        let a = &fs[c % len];
                        let b = &fs[(c + 1) % len];
                        c += 1;
                        $f(a, b)
                    })
                }};
            }
            let o = run!("add OLD", old_add);
            let n = run!("add NEW", new_add);
            println!("   add speedup: {:.2}x", o.as_secs_f64() / n.as_secs_f64());
            let o = run!("mul OLD", old_mul);
            let n = run!("mul NEW", new_mul);
            println!("   mul speedup: {:.2}x", o.as_secs_f64() / n.as_secs_f64());
            let o = run!("div OLD", old_div);
            let n = run!("div NEW", new_div);
            println!("   div speedup: {:.2}x", o.as_secs_f64() / n.as_secs_f64());
        }
    }
}
