//! A/B timing harness for `Algebraic` operations, used to compare the
//! subresultant Sturm/GCD machinery against the baseline. Run with
//! `cargo run --release --example alg_bench`.

use std::time::{Duration, Instant};

use puremp::{Algebraic, Int, Poly, Rational};

fn q(n: i64) -> Rational {
    Rational::from(n)
}
fn poly(cs: &[i64]) -> Poly<Rational> {
    Poly::new(cs.iter().map(|&c| q(c)).collect())
}
fn root_sqrt(k: i64) -> Algebraic {
    Algebraic::new(poly(&[-k, 0, 1]), q(0), q(k.max(1)))
}

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
    // Sum/product of several square roots: builds high-degree resultants whose
    // squarefree/Sturm processing exercises the remainder sequence.
    bench("√2+√3 (deg4)", 200, || root_sqrt(2).add(&root_sqrt(3)));
    bench("√2·√3 (deg4)", 200, || root_sqrt(2).mul(&root_sqrt(3)));
    bench("√2+√3+√5 (deg8)", 40, || {
        root_sqrt(2).add(&root_sqrt(3)).add(&root_sqrt(5))
    });
    bench("√2+√3+√5+√7 (deg16)", 8, || {
        root_sqrt(2)
            .add(&root_sqrt(3))
            .add(&root_sqrt(5))
            .add(&root_sqrt(7))
    });
    bench("(√2+√3)·(√5+√7) (deg16)", 8, || {
        root_sqrt(2)
            .add(&root_sqrt(3))
            .mul(&root_sqrt(5).add(&root_sqrt(7)))
    });

    // Isolation + comparison on a degree-8 Swinnerton–Dyer polynomial.
    let sd = poly(&[576, 0, -960, 0, 352, 0, -40, 0, 1]);
    bench("real_roots_of SD deg8", 40, || {
        Algebraic::real_roots_of(&sd)
    });

    // squarefree_part / sturm_chain on a dense degree-12 rational polynomial.
    let dense: Poly<Rational> = Poly::new(
        (0..=12)
            .map(|i| Rational::new(Int::from(((i * 7 + 3) % 11) - 5), Int::from(3)))
            .collect(),
    );
    bench("squarefree_part dense deg12", 400, || {
        dense.squarefree_part()
    });
    bench("real_root_count dense deg12", 200, || {
        dense.real_root_count()
    });

    // A high-degree comparison stress (equality detection uses the shared gcd
    // chain and refinement).
    let a = root_sqrt(2).add(&root_sqrt(3)).add(&root_sqrt(5));
    let b = root_sqrt(2).add(&root_sqrt(3)).add(&root_sqrt(5));
    bench("compare equal deg8", 40, || a == b);
}
