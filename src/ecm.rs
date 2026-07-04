//! Lenstra's elliptic-curve method (ECM) for integer factorization.
//!
//! ECM is the best method whose running time scales with the size of the
//! *factor* found rather than the number being factored, so it complements
//! Pollard's rho (good for small factors) and the quadratic sieve (good for
//! balanced semiprimes) by covering medium factors — roughly 20–40 digits —
//! that rho is too slow for and the sieve is overkill for.
//!
//! The implementation follows the standard Montgomery-curve formulation
//! (Montgomery, *Math. Comp.* 48 (1987); Brent & Zimmermann, *Modern Computer
//! Arithmetic* §6; Crandall & Pomerance, *Prime Numbers*, Alg. 7.4.4):
//!
//! * Curves are `b·y² = x³ + a·x² + x` in projective `(X : Z)` coordinates,
//!   so a point's `y` is never needed and the group law reduces to the
//!   differential operations `xDBL`/`xADD` — each a handful of multiplications
//!   modulo `n`, with no inversion.
//! * Each curve is set up by Suyama's parameterization, which produces a valid
//!   curve and non-torsion base point from a single random `σ` (the one modular
//!   inversion it needs either succeeds or, failing, hands us a factor for
//!   free).
//! * Stage 1 multiplies the base point by `k = ∏ pᵉ` over prime powers `pᵉ ≤
//!   B1`; if some prime factor `q | n` divides the curve order, the result is
//!   the identity modulo `q`, i.e. its `Z` coordinate is `≡ 0 (mod q)`, and
//!   `gcd(Z, n)` exposes `q`.
//! * Stage 2 extends the search to a single additional large prime in
//!   `(B1, B2]` by a baby-step/giant-step continuation, accumulating a product
//!   of coordinate cross-differences reduced by one final `gcd`.
//!
//! All arithmetic runs through a Barrett [`Reciprocal`], reusing the core's
//! modular reduction; nothing here is `unsafe` and the module is
//! dependency-free like the rest of the crate.

use alloc::vec::Vec;

use crate::int::Int;
use crate::nat::{Nat, Reciprocal};

/// Modular arithmetic context for a fixed modulus `n`, backed by a Barrett
/// reciprocal. Every operand passed to [`ModCtx::mul`] must already be reduced
/// (`< n`), which is the invariant the curve operations maintain.
struct ModCtx {
    n: Nat,
    recip: Reciprocal,
}

impl ModCtx {
    fn new(n: &Nat) -> ModCtx {
        ModCtx {
            n: n.clone(),
            recip: Reciprocal::new(n),
        }
    }

    /// `k mod n` for a machine-word constant.
    fn small(&self, k: u64) -> Nat {
        let v = Nat::from_u64(k);
        if v >= self.n {
            v.div_rem(&self.n).expect("n != 0").1
        } else {
            v
        }
    }

    /// `(a + b) mod n` for reduced `a, b`.
    fn add(&self, a: &Nat, b: &Nat) -> Nat {
        let s = a.add(b);
        if s >= self.n {
            s.checked_sub(&self.n).expect("s >= n")
        } else {
            s
        }
    }

    /// `(a − b) mod n` for reduced `a, b`.
    fn sub(&self, a: &Nat, b: &Nat) -> Nat {
        if a >= b {
            a.checked_sub(b).expect("a >= b")
        } else {
            // n − (b − a), all non-negative.
            self.n
                .checked_sub(&b.checked_sub(a).expect("b > a"))
                .expect("difference < n")
        }
    }

    /// `(a · b) mod n` for reduced `a, b` (their product is `< n²`, the range
    /// the Barrett reduction accepts).
    fn mul(&self, a: &Nat, b: &Nat) -> Nat {
        self.recip.reduce(&a.mul(b))
    }

    /// `a² mod n` for reduced `a`.
    fn sqr(&self, a: &Nat) -> Nat {
        self.recip.reduce(&a.square())
    }
}

/// A projective point `(X : Z)` on the current Montgomery curve. The `Z = 0`
/// case represents the identity (point at infinity).
#[derive(Clone)]
struct Point {
    x: Nat,
    z: Nat,
}

/// Point doubling `2·P` on the curve with constant `a24 = (a + 2)/4`
/// (Montgomery's `xDBL`).
fn x_dbl(ctx: &ModCtx, p: &Point, a24: &Nat) -> Point {
    let u = ctx.add(&p.x, &p.z); // X + Z
    let uu = ctx.sqr(&u); // (X + Z)²
    let v = ctx.sub(&p.x, &p.z); // X − Z
    let vv = ctx.sqr(&v); // (X − Z)²
    let diff = ctx.sub(&uu, &vv); // 4·X·Z
    let x = ctx.mul(&uu, &vv);
    let t = ctx.add(&vv, &ctx.mul(a24, &diff)); // (X−Z)² + a24·4XZ
    let z = ctx.mul(&diff, &t);
    Point { x, z }
}

/// Differential addition `P + Q` given the difference point `d = P − Q`
/// (Montgomery's `xADD`). Only the `(X : Z)` coordinates of `d` are used, so a
/// point and its negation are interchangeable as the difference.
fn x_add(ctx: &ModCtx, p: &Point, q: &Point, d: &Point) -> Point {
    let t1 = ctx.mul(&ctx.sub(&p.x, &p.z), &ctx.add(&q.x, &q.z)); // (X₁−Z₁)(X₂+Z₂)
    let t2 = ctx.mul(&ctx.add(&p.x, &p.z), &ctx.sub(&q.x, &q.z)); // (X₁+Z₁)(X₂−Z₂)
    let x = ctx.mul(&d.z, &ctx.sqr(&ctx.add(&t1, &t2)));
    let z = ctx.mul(&d.x, &ctx.sqr(&ctx.sub(&t1, &t2)));
    Point { x, z }
}

/// The Montgomery ladder: `[k]·P` for `k ≥ 1`, maintaining the invariant that
/// the two working points differ by exactly `P`, so every step is one `xADD`
/// (difference `P`) plus one `xDBL`.
fn ladder(ctx: &ModCtx, k: &Nat, p: &Point, a24: &Nat) -> Point {
    let mut r0 = p.clone(); // [m]·P
    let mut r1 = x_dbl(ctx, p, a24); // [m+1]·P, starting m = 1
    // Process the bits below the leading 1, most-significant first.
    let bits = k.bit_len();
    for i in (0..bits - 1).rev() {
        if k.bit(i) {
            r0 = x_add(ctx, &r0, &r1, p);
            r1 = x_dbl(ctx, &r1, a24);
        } else {
            r1 = x_add(ctx, &r0, &r1, p);
            r0 = x_dbl(ctx, &r0, a24);
        }
    }
    r0
}

/// Outcome of setting up one curve: either a base point and curve constant, or
/// a factor stumbled upon while inverting the Suyama denominator.
enum Curve {
    Ready {
        a24: Nat,
        base: Point,
    },
    Factor(Nat),
    /// The chosen `σ` was degenerate (a zero denominator mod `n`); try another.
    Retry,
}

/// Suyama's parameterization: from a seed `σ`, build a Montgomery curve and a
/// guaranteed non-torsion base point. Requires one inversion of `16·u³·v`
/// modulo `n`; if that is not invertible, `gcd` yields a factor directly.
fn suyama_curve(ctx: &ModCtx, sigma: &Nat) -> Curve {
    let five = ctx.small(5);
    let four = ctx.small(4);
    let three = ctx.small(3);
    let sixteen = ctx.small(16);

    let s2 = ctx.sqr(sigma);
    let u = ctx.sub(&s2, &five); // σ² − 5
    let v = ctx.mul(&four, sigma); // 4σ
    let u2 = ctx.sqr(&u);
    let u3 = ctx.mul(&u2, &u); // u³
    let v3 = ctx.mul(&ctx.sqr(&v), &v); // v³
    let base = Point {
        x: u3.clone(),
        z: v3,
    };

    // a24 = (v − u)³·(3u + v) / (16·u³·v)   [ = (A + 2)/4 ]
    let vmu = ctx.sub(&v, &u);
    let vmu3 = ctx.mul(&ctx.sqr(&vmu), &vmu);
    let num = ctx.mul(&vmu3, &ctx.add(&ctx.mul(&three, &u), &v));
    let den = ctx.mul(&sixteen, &ctx.mul(&u3, &v));

    if den.is_zero() {
        return Curve::Retry;
    }
    // Invert the denominator; a non-trivial gcd is a factor of n.
    match Int::from(den.clone()).modinv(&Int::from(ctx.n.clone())) {
        Some(inv) => {
            let a24 = ctx.mul(&num, &inv.magnitude());
            Curve::Ready { a24, base }
        }
        None => {
            let g = den.gcd(&ctx.n);
            if g.is_one() || g == ctx.n {
                Curve::Retry
            } else {
                Curve::Factor(g)
            }
        }
    }
}

/// Sieve of Eratosthenes: all primes `≤ limit`.
fn primes_up_to(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let n = limit as usize + 1;
    let mut sieve = alloc::vec![true; n];
    sieve[0] = false;
    sieve[1] = false;
    let mut i = 2usize;
    while i * i < n {
        if sieve[i] {
            let mut j = i * i;
            while j < n {
                sieve[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    (2..n).filter(|&i| sieve[i]).map(|i| i as u64).collect()
}

/// Stage-1 scalar `k = ∏ pᵉ` over prime powers `pᵉ ≤ b1` (`p ≤ b1`). Small
/// factors are batched into a `u64` and folded into the big accumulator only
/// when the next factor would overflow, keeping the big multiplications rare.
fn stage1_scalar(primes: &[u64], b1: u64) -> Nat {
    let mut k = Nat::one();
    let mut batch: u128 = 1;
    for &p in primes {
        if p > b1 {
            break;
        }
        // Largest power of p not exceeding b1.
        let mut pe = p;
        while let Some(next) = pe.checked_mul(p) {
            if next > b1 {
                break;
            }
            pe = next;
        }
        if batch.saturating_mul(pe as u128) > u64::MAX as u128 {
            k = k.mul(&Nat::from_u64(batch as u64));
            batch = pe as u128;
        } else {
            batch *= pe as u128;
        }
    }
    if batch > 1 {
        k = k.mul(&Nat::from_u64(batch as u64));
    }
    k
}

/// A non-trivial `gcd(z, n)` (i.e. strictly between `1` and `n`), or `None`.
fn nontrivial_gcd(z: &Nat, n: &Nat) -> Option<Nat> {
    if z.is_zero() {
        return None;
    }
    let g = z.gcd(n);
    if g.is_one() || &g == n { None } else { Some(g) }
}

/// Stage 2: a baby-step/giant-step continuation searching for a single prime
/// `p ∈ (b1, b2]` dividing the curve order. With `q` the stage-1 result,
/// tabulate baby steps `[j]·q` for `1 ≤ j ≤ D/2` and giant steps `[i·D]·q`,
/// then for each prime `p = i·D ± j` accumulate the coordinate cross-difference
/// `X_{iD}·Z_j − X_j·Z_{iD}`, which vanishes modulo a factor exactly when
/// `[p]·q` is the identity there. One final `gcd` reveals such a factor.
fn stage2(ctx: &ModCtx, q: &Point, a24: &Nat, primes: &[u64], b1: u64, b2: u64) -> Option<Nat> {
    if b2 <= b1 {
        return None;
    }
    // Giant-step size D ≈ √b2, even so that i·D ± j covers every residue.
    let mut d = (b2 as f64).sqrt() as u64;
    d = d.max(2) & !1; // even, ≥ 2
    let half = d / 2;

    // Baby steps [j]·q for j = 1..=half, built by a differential chain.
    // baby[j] holds [j]·q; index 0 is unused.
    let mut baby: Vec<Point> = Vec::with_capacity(half as usize + 1);
    baby.push(Point {
        x: Nat::one(),
        z: Nat::zero(),
    }); // placeholder [0]
    baby.push(q.clone()); // [1]·q
    if half >= 2 {
        baby.push(x_dbl(ctx, q, a24)); // [2]·q
        for j in 3..=half as usize {
            let next = x_add(ctx, &baby[j - 1], &baby[1], &baby[j - 2]);
            baby.push(next);
        }
    }

    // Giant steps [i·D]·q for i covering (b1, b2]. Start near b1.
    let i_min = b1 / d;
    let i_max = b2 / d + 1;
    if i_max < i_min {
        return None;
    }
    let step = ladder(ctx, &Nat::from_u64(d), q, a24); // [D]·q
    // giant[i - i_min] = [i·D]·q, built by a differential chain from two seeds.
    let mut giant: Vec<Point> = Vec::with_capacity((i_max - i_min + 1) as usize);
    giant.push(ladder(ctx, &Nat::from_u64(i_min * d), q, a24)); // [i_min·D]·q
    if i_max > i_min {
        giant.push(ladder(ctx, &Nat::from_u64((i_min + 1) * d), q, a24)); // [(i_min+1)·D]·q
        for idx in 2..=(i_max - i_min) as usize {
            let next = x_add(ctx, &giant[idx - 1], &step, &giant[idx - 2]);
            giant.push(next);
        }
    }

    // Accumulate cross-differences over primes in (b1, b2].
    let mut acc = Nat::one();
    let mut progressed = false;
    for &p in primes {
        if p <= b1 {
            continue;
        }
        if p > b2 {
            break;
        }
        // Nearest giant index; residue j = |p − i·D| ≤ D/2.
        let i = (p + half) / d;
        if i < i_min || i > i_max {
            continue;
        }
        let id = i * d;
        let j = p.abs_diff(id);
        if j == 0 || j > half {
            continue;
        }
        let g = &giant[(i - i_min) as usize];
        let b = &baby[j as usize];
        let cross = ctx.sub(&ctx.mul(&g.x, &b.z), &ctx.mul(&b.x, &g.z));
        if !cross.is_zero() {
            acc = ctx.mul(&acc, &cross);
            progressed = true;
        }
    }
    if progressed {
        nontrivial_gcd(&acc, &ctx.n)
    } else {
        None
    }
}

/// A tiny deterministic SplitMix64 generator, so `factorize` stays
/// reproducible without threading an RNG through the public API. Seeded from
/// the number being factored.
struct SplitMix64(u64);

impl SplitMix64 {
    fn seeded(n: &Nat) -> SplitMix64 {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for limb in n.as_limbs() {
            s = s.wrapping_add(*limb).wrapping_mul(0xD1B5_4A32_D192_ED03);
            s ^= s >> 31;
        }
        SplitMix64(s ^ 0xA0761D6478BD642F)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// One ECM attempt with the given bounds and curve count. Returns a
/// non-trivial factor of `n` (guaranteed `1 < f < n`) or `None`.
///
/// Preconditions: `n` is an odd composite `> 5` not divisible by `2` or `3`
/// (the caller removes those), so the small curve constants stay reduced.
fn ecm_attempt(n: &Nat, b1: u64, b2: u64, curves: u32, rng: &mut SplitMix64) -> Option<Nat> {
    let ctx = ModCtx::new(n);
    let primes = primes_up_to(b2.max(b1));
    let k = stage1_scalar(&primes, b1);

    for _ in 0..curves {
        // Random σ in [6, n): reject the degenerate small values 0..=5.
        let sigma = loop {
            let bits = n.bit_len();
            let mut limbs = Vec::with_capacity(n.as_limbs().len());
            for _ in 0..n.as_limbs().len() {
                limbs.push(rng.next());
            }
            let mut cand = Nat::from_limbs(&limbs);
            // Trim to at most bit_len bits, then reduce into range.
            cand = cand.low_bits(bits);
            if cand >= *n {
                cand = cand.div_rem(n).expect("n != 0").1;
            }
            if cand > Nat::from_u64(5) {
                break cand;
            }
        };

        let (a24, base) = match suyama_curve(&ctx, &sigma) {
            Curve::Ready { a24, base } => (a24, base),
            Curve::Factor(f) => return Some(f),
            Curve::Retry => continue,
        };

        // Stage 1.
        let q = ladder(&ctx, &k, &base, &a24);
        if let Some(f) = nontrivial_gcd(&q.z, n) {
            return Some(f);
        }
        // Stage 2.
        if let Some(f) = stage2(&ctx, &q, &a24, &primes, b1, b2) {
            return Some(f);
        }
    }
    None
}

/// The escalating `(B1, B2, curves)` schedule tried by [`ecm_factor`]. The
/// bounds climb so cheap curves clear medium factors before the expensive
/// deep runs; each level's `B1` targets roughly 15, 20, 25, 30, 35-digit
/// factors (GMP-ECM's tabulated optima, trimmed for a general-purpose default).
const ECM_SCHEDULE: &[(u64, u64, u32)] = &[
    (2_000, 200_000, 25),
    (11_000, 1_100_000, 90),
    (50_000, 5_000_000, 300),
    (250_000, 25_000_000, 700),
];

/// Attempts to split the odd composite `n` (with no factor of 2 or 3) by ECM,
/// escalating through [`ECM_SCHEDULE`]. Returns a non-trivial factor or `None`
/// if every level is exhausted without success.
pub(crate) fn ecm_factor(n: &Nat) -> Option<Nat> {
    let mut rng = SplitMix64::seeded(n);
    for &(b1, b2, curves) in ECM_SCHEDULE {
        if let Some(f) = ecm_attempt(n, b1, b2, curves, &mut rng) {
            return Some(f);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smallest prime `≥ start` via the crate's exact (`< 2^64`) BPSW test.
    fn prime_at_least(start: u64) -> Nat {
        let mut c = start | 1;
        loop {
            let v = Nat::from_u64(c);
            if v.is_prime_bpsw() {
                return v;
            }
            c += 2;
        }
    }

    fn assert_splits(p: &Nat, q: &Nat) {
        let composite = p.mul(q);
        let f = ecm_factor(&composite).expect("ECM finds a factor");
        assert!(f == *p || f == *q, "factor {f:?} is one of the two primes");
        let (cof, r) = composite.div_rem(&f).expect("f != 0");
        assert!(r.is_zero(), "factor divides n");
        assert!(cof == *p || cof == *q, "cofactor is the other prime");
    }

    #[test]
    fn splits_semiprimes_beyond_rho() {
        // Balanced semiprimes whose factors are past Pollard rho's fast range
        // but squarely in ECM's stage-1 reach. Both primes ~10-11 digits, so
        // the product is a ~21-digit hard semiprime.
        assert_splits(
            &prime_at_least(9_999_999_001),
            &prime_at_least(8_888_888_881),
        );
        assert_splits(
            &prime_at_least(50_000_000_021),
            &prime_at_least(70_000_000_027),
        );
    }

    #[test]
    fn suyama_denominator_hit_is_a_factor() {
        // The parameterization's own inversion, when it fails, must still yield
        // a genuine factor — exercised indirectly by a product with a modest
        // prime the early curves resolve quickly.
        let p = prime_at_least(1_000_003);
        let q = prime_at_least(2_000_003);
        assert_splits(&p, &q);
    }
}
