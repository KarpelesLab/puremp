//! The quadratic sieve for integer factorization — both the single-polynomial
//! variant and the self-initializing multiple-polynomial variant (SIQS).
//!
//! Where Lenstra's [ECM](crate::ecm) wins when `n` has a comparatively *small*
//! factor, the quadratic sieve's cost depends only on the size of `n` itself,
//! making it the method of choice for *balanced* semiprimes — two large,
//! comparable prime factors — in roughly the 40–100 digit range that ECM would
//! have to get lucky to reach.
//!
//! The method builds a congruence of squares `X² ≡ Y² (mod n)` from which
//! `gcd(X − Y, n)` is (usually) a proper factor. It does so in three phases
//! (Crandall & Pomerance, *Prime Numbers*, §6.1; Brent & Zimmermann, *Modern
//! Computer Arithmetic* §6; Pomerance, *A Tale of Two Sieves*):
//!
//! 1. **Factor base.** Collect the primes `p ≤ B` for which `n` is a quadratic
//!    residue (only those can divide `Q(x) = (⌈√n⌉ + x)² − n`), and precompute
//!    the two roots of `n` modulo each.
//! 2. **Sieve.** Over `x ∈ [−M, M]`, add `⌊log₂ p⌋` at every position where
//!    `p | Q(x)`; positions whose accumulated logarithm nears `log₂|Q(x)|` are
//!    smooth candidates, confirmed and fully factored by trial division over
//!    the base. Each smooth `Q(x)` yields a relation `(⌈√n⌉ + x)² ≡ Q(x)`.
//! 3. **Linear algebra.** Over `GF(2)`, find a subset of relations whose
//!    exponent vectors sum to zero — a product that is a perfect square on both
//!    sides — by Gaussian elimination with the combination history tracked, and
//!    turn each dependency into a candidate congruence of squares.
//!
//! The description above is the classic single-polynomial sieve. Beyond it,
//! this module also implements **SIQS** — the self-initializing
//! multiple-polynomial variant (see the block comment further down) — which
//! replaces the one wide interval by many short ones under a family of
//! polynomials `Q(x) = (a·x + b)² − n`, lifting the interval/memory limit and
//! pushing the practical range into the ~50-digit region. Both are
//! dependency-free and `unsafe`-free like the rest of the crate.

use alloc::vec::Vec;

use crate::Int;
use crate::nat::Nat;

/// Signed `Int` from an `i64` (there is no direct constructor for negatives).
#[inline]
fn int_from_i64(x: i64) -> Int {
    if x >= 0 {
        Int::from_u64(x as u64)
    } else {
        Int::from_u64(x.unsigned_abs()).neg()
    }
}

/// `n mod p` for a single-limb `p`, by Horner's rule over the limbs (each step
/// a `u128 % u64`), avoiding a full big-integer division per factor-base prime.
fn mod_u64(n: &Nat, p: u64) -> u64 {
    let mut r: u128 = 0;
    for &limb in n.as_limbs().iter().rev() {
        r = ((r << 64) | limb as u128) % p as u128;
    }
    r as u64
}

/// A modular square root of `a` modulo the odd prime `p` (both `< 2^64`) via
/// Tonelli–Shanks, or `None` if `a` is a non-residue. Runs entirely in machine
/// words — the factor-base primes are small, so the big-integer `sqrt_mod` is
/// unnecessary here.
fn sqrt_mod_u64(a: u64, p: u64) -> Option<u64> {
    let a = a % p;
    if a == 0 {
        return Some(0);
    }
    if p == 2 {
        return Some(a & 1);
    }
    if pow_mod_u64(a, (p - 1) / 2, p) != 1 {
        return None; // non-residue (Euler's criterion)
    }
    if p % 4 == 3 {
        return Some(pow_mod_u64(a, (p + 1) / 4, p));
    }
    // Factor p − 1 = q·2^s with q odd.
    let mut q = p - 1;
    let mut s = 0u32;
    while q & 1 == 0 {
        q >>= 1;
        s += 1;
    }
    // A quadratic non-residue z.
    let mut z = 2u64;
    while pow_mod_u64(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
    }
    let mut m = s;
    let mut c = pow_mod_u64(z, q, p);
    let mut t = pow_mod_u64(a, q, p);
    let mut r = pow_mod_u64(a, q.div_ceil(2), p);
    while t != 1 {
        // Least i with t^(2^i) = 1.
        let mut i = 0u32;
        let mut t2 = t;
        while t2 != 1 {
            t2 = mul_mod_u64(t2, t2, p);
            i += 1;
        }
        let b = pow_mod_u64(c, 1u64 << (m - i - 1), p);
        m = i;
        c = mul_mod_u64(b, b, p);
        t = mul_mod_u64(t, c, p);
        r = mul_mod_u64(r, b, p);
    }
    Some(r)
}

#[inline]
fn mul_mod_u64(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn pow_mod_u64(mut base: u64, mut exp: u64, p: u64) -> u64 {
    let mut r = 1u64;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            r = mul_mod_u64(r, base, p);
        }
        base = mul_mod_u64(base, base, p);
        exp >>= 1;
    }
    r
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

/// A factor-base prime with the two sieve roots of `Q` (positions `x` where
/// `p | Q(x)`), the second unused (`= p`) when `p | n`.
struct FbPrime {
    p: u64,
    logp: u8,
    root1: u64,
    root2: u64,
}

/// A smooth relation `(a + x)² ≡ Q(x) (mod n)`: the base value `a + x`, the
/// sign (`Q < 0`) and prime exponents of `|Q(x)|` (sparse, factor-base index →
/// exponent), and the `GF(2)` parity vector packed into `u64` words.
struct Relation {
    base: Nat,
    exps: Vec<(usize, u32)>,
    parity: Vec<u64>,
}

/// Tunable parameters chosen from the decimal size of `n`.
struct Params {
    /// Smoothness bound (largest factor-base prime).
    bound: u64,
    /// Half-width of the sieve interval `[−M, M]`.
    m: u64,
    /// Slack (in bits) subtracted from the smoothness threshold, allowing a
    /// small number of missed prime-power logs before trial division confirms.
    fudge: u32,
}

/// Picks sieve parameters for a modulus of `digits` decimal digits, from the
/// standard QS tuning tables (interpolated and clamped for a general default).
fn params_for(digits: usize) -> Params {
    // Smoothness bound and starting interval grow sub-exponentially with size.
    // The single-polynomial sieve's interval (and thus its memory) is the
    // limiting factor; these buckets keep it practical through the mid-40-digit
    // range, past which the SIQS path below takes over.
    let (bound, m): (u64, u64) = match digits {
        0..=24 => (3_000, 300_000),
        25..=29 => (10_000, 4_000_000),
        30..=34 => (25_000, 10_000_000),
        35..=39 => (45_000, 20_000_000),
        40..=42 => (75_000, 40_000_000),
        43..=45 => (120_000, 70_000_000),
        _ => (250_000, 120_000_000),
    };
    // Threshold slack over the exact log target: enough to keep smooth values
    // whose repeated small-prime powers were under-counted by the single sieve
    // pass, but tight enough to reject most non-smooth positions before the
    // (bignum) trial division. Empirically ≈ log₂(largest prime) works well.
    let fudge = (64 - bound.leading_zeros()) + 2;
    Params { bound, m, fudge }
}

/// Builds the factor base: `−1` (index 0, sign) then the primes `p ≤ bound`
/// with `(n/p) = 1`, each carrying `⌊log₂ p⌋` and its two sieve roots.
fn build_factor_base(n: &Nat, bound: u64) -> Vec<FbPrime> {
    let mut fb = Vec::new();
    for p in primes_up_to(bound) {
        let a = mod_u64(n, p);
        if p == 2 {
            // n is odd, so n ≡ 1 (mod 2): the single root is 1.
            fb.push(FbPrime {
                p: 2,
                logp: 1,
                root1: 1,
                root2: 1,
            });
            continue;
        }
        if a == 0 {
            // p | n: p is itself a factor (caught by the caller's gcd checks),
            // include it with a single root so relations stay consistent.
            fb.push(FbPrime {
                p,
                logp: 64 - p.leading_zeros() as u8 - 1,
                root1: 0,
                root2: p,
            });
            continue;
        }
        if let Some(r) = sqrt_mod_u64(a, p) {
            fb.push(FbPrime {
                p,
                logp: (64 - p.leading_zeros() - 1) as u8,
                root1: r,
                root2: p - r,
            });
        }
    }
    fb
}

/// Runs the sieve, trial-divides the candidates, and returns smooth relations
/// (at least `fb.len() + margin`, or as many as the interval yields).
fn collect_relations(
    n: &Nat,
    a: &Nat,
    fb: &[FbPrime],
    params: &Params,
    m: u64,
    want: usize,
) -> Vec<Relation> {
    let width = (2 * m + 1) as usize;
    let mut logs = alloc::vec![0u8; width];

    // Accumulate ⌊log₂ p⌋ at every position divisible by each factor-base
    // prime. Index i corresponds to x = i − M.
    for fp in fb {
        if fp.p == 2 {
            // 2 | Q(x) ⇔ (a + x) odd ⇔ (a_par + i + M) odd (since x = i − M).
            let a_par = mod_u64(a, 2) as i64;
            let mut i = (1 - a_par - m as i64).rem_euclid(2) as usize;
            while i < width {
                logs[i] = logs[i].saturating_add(1);
                i += 2;
            }
            continue;
        }
        let a_mod = mod_u64(a, fp.p) as i64;
        let single = fp.root1 == fp.root2;
        for (k, &root) in [fp.root1, fp.root2].iter().enumerate() {
            if k == 1 && single {
                break; // one distinct root only (p | n)
            }
            // p | Q(x) ⇔ x ≡ root − a (mod p) ⇔ i ≡ root − a + M (mod p).
            let start = (root as i64 - a_mod + m as i64).rem_euclid(fp.p as i64) as usize;
            let mut i = start;
            while i < width {
                logs[i] = logs[i].saturating_add(fp.logp);
                i += fp.p as usize;
            }
        }
    }

    let half_log2n = (n.bit_len() / 2) as u32;
    let mut relations = Vec::new();
    #[allow(clippy::needless_range_loop)] // i drives both logs[i] and x = i − M
    for i in 0..width {
        // Per-position target ≈ log₂|Q(x)| ≈ ½·log₂ n + log₂|x|.
        let x = i as i64 - m as i64;
        let xlog = if x == 0 {
            0
        } else {
            63 - (x.unsigned_abs()).leading_zeros()
        };
        let target = half_log2n.saturating_add(xlog);
        if (logs[i] as u32) + params.fudge < target {
            continue;
        }
        if let Some(rel) = try_relation(n, a, fb, x) {
            relations.push(rel);
            if relations.len() >= want {
                break;
            }
        }
    }
    relations
}

/// Attempts to build a relation at offset `x`: computes `Q(x) = (a + x)² − n`,
/// trial-divides it over the factor base, and returns the relation iff it is
/// fully smooth.
fn try_relation(n: &Nat, a: &Nat, fb: &[FbPrime], x: i64) -> Option<Relation> {
    // base = a + x  (a ≈ √n ≫ |x|, so a + x > 0).
    let base = if x >= 0 {
        a.add(&Nat::from_u64(x as u64))
    } else {
        a.checked_sub(&Nat::from_u64((-x) as u64))?
    };
    relation_from_base(n, base, fb)
}

/// Given the already-computed base value `g` (with `g² ≡ Q (mod n)`), trial-
/// divides `Q = g² − n` over the factor base and returns the relation iff `Q`
/// is fully smooth. Shared by the single-polynomial sieve (`g = ⌈√n⌉ + x`) and
/// SIQS (`g = a·x + b`, reduced to its magnitude modulo `n`).
fn relation_from_base(n: &Nat, base: Nat, fb: &[FbPrime]) -> Option<Relation> {
    let sq = base.square();
    // Q = base² − n, tracked with its sign.
    let (mut mag, neg) = if sq >= *n {
        (sq.checked_sub(n).unwrap(), false)
    } else {
        (n.checked_sub(&sq).unwrap(), true)
    };
    if mag.is_zero() {
        return None;
    }
    let mut exps: Vec<(usize, u32)> = Vec::new();
    if neg {
        exps.push((0, 1)); // the −1 column
    }
    // Trial-divide by each factor-base prime.
    for (idx, fp) in fb.iter().enumerate().skip(1) {
        let pn = Nat::from_u64(fp.p);
        let mut e = 0u32;
        loop {
            let (q, r) = mag.div_rem(&pn).unwrap();
            if !r.is_zero() {
                break;
            }
            mag = q;
            e += 1;
        }
        if e > 0 {
            exps.push((idx, e));
        }
        if mag.is_one() {
            break;
        }
    }
    if !mag.is_one() {
        return None; // a leftover cofactor: not smooth over the base
    }
    // Pack the exponent parities into GF(2) words.
    let words = fb.len().div_ceil(64);
    let mut parity = alloc::vec![0u64; words];
    for &(idx, e) in &exps {
        if e & 1 == 1 {
            parity[idx / 64] ^= 1u64 << (idx % 64);
        }
    }
    Some(Relation { base, exps, parity })
}

/// Finds `GF(2)` linear dependencies among the relations' parity vectors by
/// Gaussian elimination, tracking each row's combination history. Returns the
/// dependencies as subsets of relation indices (each a product that is a square
/// on both sides).
fn find_dependencies(relations: &[Relation], cols: usize) -> Vec<Vec<usize>> {
    let m = relations.len();
    let pword = cols.div_ceil(64);
    let hword = m.div_ceil(64);
    // Working rows: (parity, history), history[i] a bitset of relation indices.
    let mut par: Vec<Vec<u64>> = relations.iter().map(|r| r.parity.clone()).collect();
    for row in &mut par {
        row.resize(pword, 0);
    }
    let mut hist: Vec<Vec<u64>> = (0..m)
        .map(|i| {
            let mut h = alloc::vec![0u64; hword];
            h[i / 64] |= 1u64 << (i % 64);
            h
        })
        .collect();

    // pivot_row[c] = index of the row that owns column c, if any.
    let mut pivot_row: Vec<Option<usize>> = alloc::vec![None; cols];
    let mut deps = Vec::new();
    for i in 0..m {
        // Reduce row i against existing pivots, low column to high.
        for c in 0..cols {
            if par[i][c / 64] & (1u64 << (c % 64)) == 0 {
                continue;
            }
            if let Some(pr) = pivot_row[c] {
                // pr < i (pivots come from earlier rows), so split the borrow.
                let (lo, hi) = par.split_at_mut(i);
                for (a, b) in hi[0].iter_mut().zip(&lo[pr]) {
                    *a ^= *b;
                }
                let (lo, hi) = hist.split_at_mut(i);
                for (a, b) in hi[0].iter_mut().zip(&lo[pr]) {
                    *a ^= *b;
                }
            }
        }
        // Lowest set column becomes this row's pivot; an all-zero row is a
        // dependency.
        let mut pivot = None;
        for c in 0..cols {
            if par[i][c / 64] & (1u64 << (c % 64)) != 0 {
                pivot = Some(c);
                break;
            }
        }
        match pivot {
            Some(c) => pivot_row[c] = Some(i),
            None => {
                let subset: Vec<usize> = (0..m)
                    .filter(|&j| hist[i][j / 64] & (1u64 << (j % 64)) != 0)
                    .collect();
                if !subset.is_empty() {
                    deps.push(subset);
                }
            }
        }
    }
    deps
}

/// Turns each square congruence into a `gcd` attempt, returning the first
/// non-trivial factor found.
fn factor_from_dependencies(
    n: &Nat,
    fb: &[FbPrime],
    relations: &[Relation],
    deps: &[Vec<usize>],
) -> Option<Nat> {
    for subset in deps {
        // X = ∏ (a + x_i) mod n.
        let mut x = Nat::one();
        // Sum full exponents over the subset (all even for a dependency).
        let mut total: Vec<u32> = alloc::vec![0u32; fb.len()];
        for &i in subset {
            let r = &relations[i];
            x = r.base.mul(&x).div_rem(n).unwrap().1;
            for &(idx, e) in &r.exps {
                total[idx] += e;
            }
        }
        // Y = ∏ p^(e/2) mod n (index 0 is the −1 sign; its even exponent means
        // the product is a positive square, so Y is real).
        let mut y = Nat::one();
        for (idx, &e) in total.iter().enumerate().skip(1) {
            if e == 0 {
                continue;
            }
            debug_assert_eq!(e & 1, 0, "dependency exponent must be even");
            let pe = Nat::from_u64(fb[idx].p).modpow(&Nat::from_u64((e / 2) as u64), n);
            y = y.mul(&pe).div_rem(n).unwrap().1;
        }
        // gcd(X − Y, n): a proper factor unless X ≡ ±Y.
        let diff = if x >= y {
            x.checked_sub(&y).unwrap()
        } else {
            n.checked_sub(&y.checked_sub(&x).unwrap()).unwrap()
        };
        if !diff.is_zero() {
            let g = diff.gcd(n);
            if !g.is_one() && &g != n {
                return Some(g);
            }
        }
        // Also try X + Y.
        let sum = x.add(&y).div_rem(n).unwrap().1;
        if !sum.is_zero() {
            let g = sum.gcd(n);
            if !g.is_one() && &g != n {
                return Some(g);
            }
        }
    }
    None
}

/// Attempts to split the odd composite `n` with the quadratic sieve. Returns a
/// non-trivial factor, or `None` if the sieve interval did not yield enough
/// smooth relations or every dependency was trivial (the caller then falls
/// back). `n` must not be a perfect square (the caller handles perfect powers).
pub(crate) fn qs_factor(n: &Nat) -> Option<Nat> {
    // Decimal size, estimated from the bit length (log₁₀ 2 ≈ 0.30103) — good
    // enough to pick the parameter bucket without a full base-10 conversion.
    let digits = (n.bit_len() * 30103 / 100000 + 1) as usize;
    let params = params_for(digits);

    let fb = build_factor_base(n, params.bound);
    if fb.len() < 3 {
        return None;
    }
    // a = ⌈√n⌉.
    let root = n.isqrt();
    let a = if root.square() == *n {
        return Some(root); // perfect square: √n is a factor
    } else {
        root.add(&Nat::one())
    };

    // Aim for a comfortable surplus of relations over the factor-base size so a
    // dependency is essentially guaranteed.
    let want = fb.len() + 16 + fb.len() / 20;

    // Sieve, extending the interval if the yield falls short (balanced factors
    // give a narrow smooth region, so the first estimate can undershoot). The
    // cap bounds total work; beyond it the caller falls back.
    // Cap the interval so the sieve array (2·M bytes) stays within a few
    // hundred MB even when the yield forces extension; beyond it the caller
    // falls back rather than exhausting memory.
    const M_CAP: u64 = 300_000_000;
    let mut m = params.m.min(M_CAP);
    let mut relations = collect_relations(n, &a, &fb, &params, m, want);
    while relations.len() < want && m < M_CAP {
        m = m.saturating_mul(2).min(M_CAP);
        relations = collect_relations(n, &a, &fb, &params, m, want);
    }
    if relations.len() <= fb.len() {
        return None; // not enough relations for a guaranteed dependency
    }

    let deps = find_dependencies(&relations, fb.len());
    factor_from_dependencies(n, &fb, &relations, &deps)
}

// ===========================================================================
// Self-initializing multiple-polynomial quadratic sieve (SIQS)
// ===========================================================================
//
// The single-polynomial sieve above uses one polynomial `Q(x) = (⌈√n⌉ + x)² − n`
// whose values grow linearly with `|x|`; the interval `[−M, M]` must therefore
// be huge to gather enough smooth relations, and its memory dominates. SIQS
// (Contini's thesis; Crandall & Pomerance §6.1.3) instead uses *many* short
// polynomials
//
// ```text
//   Q_{a,b}(x) = (a·x + b)² − n,   b² ≡ n (mod a),   a ≈ √(2n) / M,
// ```
//
// with `a` a product of `s` factor-base primes. Because `b² ≡ n (mod a)`, every
// value `Q_{a,b}(x)` is divisible by `a`, and `|Q/a| ≈ M·√(n/2)` stays small
// across a *short* interval. Each distinct `a` admits `2^{s−1}` residues `b`,
// and — the "self-initializing" trick — successive `b`'s (walked in Gray-code
// order) update every prime's sieve roots by a single precomputed increment
// `±B_k·a⁻¹` rather than a fresh modular inversion. The smooth relations feed
// the *same* GF(2) linear algebra as the single-polynomial sieve: `a`'s primes
// are ordinary factor-base primes (they always divide `Q`), so the exponent
// vectors, dependency search and square-congruence extraction are reused as-is.

/// Modular inverse of `a` modulo the odd prime `p` (or `p = 2`), via Fermat:
/// `a^(p−2) mod p`. `a` must be non-zero modulo `p`.
#[inline]
fn inv_mod_u64(a: u64, p: u64) -> u64 {
    if p == 2 {
        return 1; // odd a
    }
    pow_mod_u64(a % p, p - 2, p)
}

/// A tiny deterministic linear-congruential generator, seeded from `n`, used
/// only to vary the `a`-coefficient's prime selection between polynomials. Its
/// statistical quality is irrelevant — it merely spreads the choices so that
/// distinct `a`'s are produced.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        // Numerical Recipes' 64-bit LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next() >> 33) as usize % bound
        }
    }
}

/// SIQS tuning: smoothness bound, half-interval `M`, and the number `s` of
/// factor-base primes composing the `a`-coefficient.
struct SiqsParams {
    bound: u64,
    m: u64,
    s: usize,
}

/// Picks SIQS parameters for a modulus of `digits` decimal digits. The interval
/// is deliberately short (SIQS's advantage), the smoothness bound moderate, and
/// `s` grows so each `a`-prime stays a comfortable size (`≈ (√(2n)/M)^{1/s}`).
fn siqs_params_for(digits: usize) -> SiqsParams {
    let (bound, m, s): (u64, u64, usize) = match digits {
        0..=32 => (3_000, 60_000, 3),
        33..=37 => (7_000, 65_000, 4),
        38..=42 => (18_000, 65_000, 5),
        43..=46 => (40_000, 100_000, 6),
        47..=50 => (80_000, 120_000, 7),
        _ => (150_000, 160_000, 8),
    };
    SiqsParams { bound, m, s }
}

/// The precomputed data for one `a`-coefficient: the value `a`, the factor-base
/// indices of its prime factors, the `B_k` half-terms whose signed sum forms
/// `b`, and — per active factor-base prime — `a⁻¹ mod p`, the current sieve
/// roots for the polynomial in hand, and the Gray-code root increments
/// `2·B_k·a⁻¹ mod p`.
struct ACoeff {
    a: Nat,
    b_terms: Vec<Nat>,
    /// Whether this factor-base prime participates in the sieve (`p ∤ a`, `p ≠ −1`).
    active: Vec<bool>,
    /// Current first/second sieve roots (`x` with `p | Q(x)`), per prime.
    soln1: Vec<u64>,
    soln2: Vec<u64>,
    /// `bainv2[idx*s + k] = 2·B_k·a⁻¹ mod p`, the Gray-code root increments.
    bainv2: Vec<u64>,
}

/// Selects `s` distinct factor-base primes whose product `a` is close to the
/// target `√(2n)/M`, drawing from a pool of primes near `target^{1/s}` and
/// choosing the final prime to best hit the target. Returns `None` if the pool
/// is too small (the input is below SIQS's useful range).
fn choose_a(
    n: &Nat,
    fb: &[FbPrime],
    params: &SiqsParams,
    rng: &mut Lcg,
) -> Option<(Nat, Vec<usize>)> {
    let two_n = n.add(n);
    let target = two_n.isqrt().div_rem(&Nat::from_u64(params.m)).unwrap().0;
    if target.bit_len() < 10 {
        return None;
    }
    let per_prime = target.nth_root_floor(params.s as u32).to_u64().unwrap_or(0);
    if per_prime < 3 {
        return None;
    }
    // Build a pool of the factor-base primes nearest `per_prime` (skipping −1,
    // the prime 2, and any prime dividing n).
    let mut pool: Vec<usize> = (1..fb.len())
        .filter(|&i| fb[i].p > 2 && fb[i].root1 != 0)
        .collect();
    pool.sort_by_key(|&i| fb[i].p.abs_diff(per_prime));
    let pool_len = pool.len().min((params.s * 6).max(30));
    pool.truncate(pool_len);
    if pool.len() < params.s {
        return None;
    }
    // Pick s−1 random distinct pool primes, then the last to best match target.
    let mut chosen: Vec<usize> = Vec::with_capacity(params.s);
    let mut prod = Nat::one();
    while chosen.len() + 1 < params.s {
        let cand = pool[rng.below(pool.len())];
        if chosen.contains(&cand) {
            continue;
        }
        chosen.push(cand);
        prod = prod.mul(&Nat::from_u64(fb[cand].p));
    }
    // Ideal final prime ≈ target / prod; pick the closest unused pool prime.
    let want_last = target
        .div_rem(&prod)
        .unwrap()
        .0
        .to_u64()
        .unwrap_or(u64::MAX);
    let mut best: Option<usize> = None;
    for &i in &pool {
        if chosen.contains(&i) {
            continue;
        }
        let better = match best {
            None => true,
            Some(b) => fb[i].p.abs_diff(want_last) < fb[b].p.abs_diff(want_last),
        };
        if better {
            best = Some(i);
        }
    }
    let last = best?;
    chosen.push(last);
    prod = prod.mul(&Nat::from_u64(fb[last].p));
    chosen.sort_unstable();
    Some((prod, chosen))
}

/// Builds the self-initialization data for a fresh `a`: the `B_k` half-terms,
/// per-prime `a⁻¹` and Gray-code increments, and the sieve roots for the first
/// polynomial (`b = Σ B_k`).
fn init_a(fb: &[FbPrime], m: u64, a: Nat, a_idx: Vec<usize>) -> ACoeff {
    let s = a_idx.len();
    // B_k = (a / q_k) · [ (a / q_k)⁻¹ · t_{q_k} mod q_k ], so B_k² ≡ n (mod q_k)
    // and B_k ≡ 0 (mod q_j) for j ≠ k. Then b = Σ B_k satisfies b² ≡ n (mod a).
    let mut b_terms: Vec<Nat> = Vec::with_capacity(s);
    for &k in &a_idx {
        let q = fb[k].p;
        let a_over_q = a.div_rem(&Nat::from_u64(q)).unwrap().0;
        let inv = inv_mod_u64(mod_u64(&a_over_q, q), q);
        let gamma = mul_mod_u64(fb[k].root1, inv, q);
        // Reduce to the smaller representative to keep |b| small.
        let gamma = gamma.min(q - gamma);
        b_terms.push(a_over_q.mul(&Nat::from_u64(gamma)));
    }
    let mut b = Nat::zero();
    for t in &b_terms {
        b = b.add(t);
    }

    let mut active = alloc::vec![false; fb.len()];
    let mut soln1 = alloc::vec![0u64; fb.len()];
    let mut soln2 = alloc::vec![0u64; fb.len()];
    let mut bainv2 = alloc::vec![0u64; fb.len() * s];
    let mmod = m; // interval offset added when converting x → array index

    for (idx, fp) in fb.iter().enumerate().skip(1) {
        let p = fp.p;
        if a.div_rem(&Nat::from_u64(p)).unwrap().1.is_zero() {
            continue; // p | a: no linear root, handled by trial division
        }
        active[idx] = true;
        let ai = inv_mod_u64(mod_u64(&a, p), p);
        for (k, term) in b_terms.iter().enumerate() {
            bainv2[idx * s + k] = mul_mod_u64(mul_mod_u64(2 % p, mod_u64(term, p), p), ai, p);
        }
        let t = fp.root1;
        let bmod = mod_u64(&b, p) as i128;
        // x ≡ a⁻¹·(±t − b) (mod p); array index i = x + M.
        let r1 = (ai as i128 * ((t as i128) - bmod)).rem_euclid(p as i128) as u64;
        let r2 = (ai as i128 * ((p as i128 - t as i128) - bmod)).rem_euclid(p as i128) as u64;
        soln1[idx] = ((r1 as i128 + mmod as i128) % p as i128) as u64;
        soln2[idx] = ((r2 as i128 + mmod as i128) % p as i128) as u64;
    }

    ACoeff {
        a,
        b_terms,
        active,
        soln1,
        soln2,
        bainv2,
    }
}

/// Sieves the current polynomial `Q(x) = (a·x + b)² − n` over `[−M, M]`,
/// collects the smooth relations, and appends them to `relations`.
#[allow(clippy::too_many_arguments)]
fn sieve_poly(
    n: &Nat,
    ac: &ACoeff,
    b: &Int,
    fb: &[FbPrime],
    m: u64,
    target: u32,
    fudge: u32,
    logs: &mut [u8],
    relations: &mut Vec<Relation>,
    want: usize,
) {
    let width = (2 * m + 1) as usize;
    for v in logs.iter_mut() {
        *v = 0;
    }
    for (idx, fp) in fb.iter().enumerate().skip(1) {
        if !ac.active[idx] {
            continue;
        }
        let p = fp.p as usize;
        let single = fp.root1 == fp.root2;
        for (k, &root) in [ac.soln1[idx], ac.soln2[idx]].iter().enumerate() {
            if k == 1 && single {
                break;
            }
            let mut i = root as usize;
            while i < width {
                logs[i] = logs[i].saturating_add(fp.logp);
                i += p;
            }
        }
    }
    let a_int = Int::from(ac.a.clone());
    let threshold = target.saturating_sub(fudge);
    #[allow(clippy::needless_range_loop)] // i drives both logs[i] and x = i − M
    for i in 0..width {
        if (logs[i] as u32) < threshold {
            continue;
        }
        let x = i as i64 - m as i64;
        // g = a·x + b; base = |g| (< n, since |g| ≤ a·M + |b| ≈ √(2n)).
        let g = a_int.mul(&int_from_i64(x)).add(b);
        let base = g.magnitude();
        if base.is_zero() {
            continue;
        }
        if let Some(rel) = siqs_relation(n, ac, base, fb, i) {
            relations.push(rel);
            if relations.len() >= want {
                return;
            }
        }
    }
}

/// Confirms and factors a SIQS candidate at array index `i`. Unlike the
/// single-polynomial `relation_from_base` (which trial-divides by every
/// factor-base prime), this uses the sieve roots: a prime `p ∤ a` divides
/// `Q(x)` exactly when `i` lands on one of its roots, so only the primes that
/// actually hit `i` — plus the primes dividing `a`, which divide every `Q` —
/// are divided out. This turns thousands of big-integer divisions per candidate
/// into a handful.
fn siqs_relation(n: &Nat, ac: &ACoeff, base: Nat, fb: &[FbPrime], i: usize) -> Option<Relation> {
    let sq = base.square();
    let (mut mag, neg) = if sq >= *n {
        (sq.checked_sub(n).unwrap(), false)
    } else {
        (n.checked_sub(&sq).unwrap(), true)
    };
    if mag.is_zero() {
        return None;
    }
    let mut exps: Vec<(usize, u32)> = Vec::new();
    if neg {
        exps.push((0, 1));
    }
    for (idx, fp) in fb.iter().enumerate().skip(1) {
        let p = fp.p;
        // `p | Q(x)` iff `p | a` (then it divides every value) or `i` is a root.
        let hits = if ac.active[idx] {
            let r = (i as u64) % p;
            r == ac.soln1[idx] || r == ac.soln2[idx]
        } else {
            true
        };
        if !hits {
            continue;
        }
        let pn = Nat::from_u64(p);
        let mut e = 0u32;
        loop {
            let (q, r) = mag.div_rem(&pn).unwrap();
            if !r.is_zero() {
                break;
            }
            mag = q;
            e += 1;
        }
        if e > 0 {
            exps.push((idx, e));
        }
        if mag.is_one() {
            break;
        }
    }
    if !mag.is_one() {
        return None; // leftover cofactor: not smooth over the base
    }
    let words = fb.len().div_ceil(64);
    let mut parity = alloc::vec![0u64; words];
    for &(idx, e) in &exps {
        if e & 1 == 1 {
            parity[idx / 64] ^= 1u64 << (idx % 64);
        }
    }
    Some(Relation { base, exps, parity })
}

/// Attempts to split the odd composite `n` with the self-initializing MPQS.
/// Returns a non-trivial factor, or `None` if it could not build a usable
/// `a`-coefficient, gather enough relations within the polynomial budget, or
/// turn any dependency into a proper factor (the caller then falls back).
/// `n` must not be a perfect square (the caller handles perfect powers).
pub(crate) fn siqs_factor(n: &Nat) -> Option<Nat> {
    let digits = (n.bit_len() * 30103 / 100000 + 1) as usize;
    let params = siqs_params_for(digits);
    let fb = build_factor_base(n, params.bound);
    if fb.len() < 10 {
        return None;
    }
    let root = n.isqrt();
    if root.square() == *n {
        return Some(root); // perfect square
    }

    let target = (n.bit_len() as u32).div_ceil(2) + (64 - params.m.leading_zeros());
    let fudge = (64 - params.bound.leading_zeros()) + 10;
    let want = fb.len() + 16 + fb.len() / 20;

    let mut rng = Lcg(n.as_limbs().first().copied().unwrap_or(1) | 1);
    let mut relations: Vec<Relation> = Vec::new();
    let mut logs = alloc::vec![0u8; (2 * params.m + 1) as usize];

    // Bound the total polynomial work so a hard input escalates rather than
    // grinding; each `a` yields 2^{s−1} polynomials.
    let max_polys: u64 = 400_000;
    let mut polys = 0u64;
    'outer: while relations.len() < want && polys < max_polys {
        let (a, a_idx) = choose_a(n, &fb, &params, &mut rng)?;
        let s = a_idx.len();
        let mut ac = init_a(&fb, params.m, a, a_idx);

        // Gray-code walk over the 2^{s−1} residues b (fixing the top sign to
        // avoid the ±b duplicate). `signs[k] = +1/−1` tracks the current sign
        // of B_k in b = Σ signs[k]·B_k.
        let mut signs = alloc::vec![1i64; s];
        let mut b = {
            let mut acc = Int::ZERO;
            for t in &ac.b_terms {
                acc = acc.add(&Int::from(t.clone()));
            }
            acc
        };
        let polys_this_a = 1u64 << (s - 1);
        for iter in 0..polys_this_a {
            if iter > 0 {
                // Gray code: bit `j = trailing_zeros(iter)` flips.
                let j = iter.trailing_zeros() as usize;
                let ds = signs[j];
                signs[j] = -ds;
                // b ← b − 2·ds·B_j.
                let two_bj = Int::from(ac.b_terms[j].clone()).mul(&Int::from_u64(2));
                b = if ds > 0 {
                    b.sub(&two_bj)
                } else {
                    b.add(&two_bj)
                };
                // Root increment: soln ← soln + ds·bainv2[·][j] (mod p).
                for (idx, fp) in fb.iter().enumerate().skip(1) {
                    if !ac.active[idx] {
                        continue;
                    }
                    let p = fp.p as i128;
                    let inc = ds as i128 * ac.bainv2[idx * s + j] as i128;
                    ac.soln1[idx] = ((ac.soln1[idx] as i128 + inc).rem_euclid(p)) as u64;
                    ac.soln2[idx] = ((ac.soln2[idx] as i128 + inc).rem_euclid(p)) as u64;
                }
            }
            sieve_poly(
                n,
                &ac,
                &b,
                &fb,
                params.m,
                target,
                fudge,
                &mut logs,
                &mut relations,
                want,
            );
            polys += 1;
            if relations.len() >= want {
                break 'outer;
            }
        }
    }

    if relations.len() <= fb.len() {
        return None;
    }
    let deps = find_dependencies(&relations, fb.len());
    factor_from_dependencies(n, &fb, &relations, &deps)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let f = qs_factor(&composite).expect("QS finds a factor");
        assert!(f == *p || f == *q, "factor {f:?} is one of the primes");
        let (cof, r) = composite.div_rem(&f).unwrap();
        assert!(r.is_zero());
        assert!(cof == *p || cof == *q);
    }

    #[test]
    fn splits_balanced_semiprimes() {
        // Two ~10-digit primes (~20-digit n).
        assert_splits(
            &prime_at_least(3_000_000_019),
            &prime_at_least(4_000_000_007),
        );
        // Two ~13-digit primes (~26-digit n).
        assert_splits(
            &prime_at_least(5_000_000_000_021),
            &prime_at_least(9_000_000_000_011),
        );
    }

    /// Smallest prime `≥ start`, over big integers (for SIQS-range inputs).
    fn big_prime_at_least(start: &Nat) -> Nat {
        let mut c = if start.is_even() {
            start.add(&Nat::one())
        } else {
            start.clone()
        };
        loop {
            if c.is_prime_bpsw() {
                return c;
            }
            c = c.add(&Nat::from_u64(2));
        }
    }

    fn assert_siqs_splits(p: &Nat, q: &Nat) {
        let n = p.mul(q);
        let f = siqs_factor(&n).expect("SIQS finds a factor");
        assert!(f == *p || f == *q, "factor {f:?} is one of the primes");
        let (cof, r) = n.div_rem(&f).unwrap();
        assert!(r.is_zero());
        assert!(cof == *p || cof == *q);
    }

    #[test]
    fn siqs_splits_balanced_semiprimes() {
        // A balanced ~40-digit semiprime (two ~20-digit primes).
        assert_siqs_splits(
            &big_prime_at_least(&Nat::from_u128(3_000_000_000_000_000_000_007u128)),
            &big_prime_at_least(&Nat::from_u128(5_000_000_000_000_000_000_003u128)),
        );
        // A balanced ~45-digit semiprime (two ~23-digit primes) — well beyond
        // the comfortable single-polynomial range.
        assert_siqs_splits(
            &big_prime_at_least(&Nat::from_u128(200_000_000_000_000_000_000_003u128)),
            &big_prime_at_least(&Nat::from_u128(700_000_000_000_000_000_000_001u128)),
        );
    }

    #[test]
    fn siqs_handles_perfect_square() {
        // A perfect square is split by the isqrt shortcut, not the sieve.
        let p = big_prime_at_least(&Nat::from_u128(40_000_000_000_000_000_019u128));
        let n = p.square();
        assert_eq!(siqs_factor(&n), Some(p));
    }
}
