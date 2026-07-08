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

use alloc::collections::BTreeMap;
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

// ===========================================================================
// Block Lanczos over GF(2) (Montgomery, Eurocrypt '95)
// ===========================================================================
//
// Dense Gaussian elimination on the parity matrix is O(n^3) and caps the usable
// factor-base size well below what a ~60-digit input needs. Montgomery's block
// Lanczos finds the same GF(2) dependencies in O(n * (cost of a sparse
// matrix-vector product)) -- a handful of passes over the (very sparse) relation
// matrix -- letting the factor base grow much larger.
//
// Setup. Let `R` be the `m x cols` relation/parity matrix (row = one relation's
// exponent-parities). A dependency is a set of relations whose parities sum to
// zero, i.e. a vector `z` in GF(2)^m with `R^T z = 0`. We work with the
// symmetric operator `A = R R^T` (`m x m`), applied as `A z = R (R^T z)` -- two
// sparse passes, never materialised. A vector with `A z = 0` has `R^T z` in the
// null space of `R`; the final extraction pins down the combinations with
// `R^T z = 0` exactly, which are the true dependencies.
//
// The iteration (Montgomery [1995], in the simplified single-history form given
// by Thome, "A modified block Lanczos algorithm with fewer vectors", 2016)
// keeps block-vectors of width `N = 64` (one machine word per row). Writing
// `d_i` for the diagonal 0/1 projector selecting the columns kept at step `i`
// (its complement `1 - d_i` carries the rest forward), and `w_i^inv` for the
// symmetric inverse of `v_i^T A v_i` on those columns:
//
//   c_{i+1,i} = w_i^inv (v_i^T A^2 v_i d_i + v_i^T A v_i (1 - d_i))
//   v_{i+1}   = A v_i d_i + v_i (1 - d_i) - v_i c_{i+1,i} - p_i (v_i^T A v_i d_i)
//   p_{i+1}   = v_i w_i^inv + p_i (1 - d_i)                     (p_0 = 0)
//
// The single running block `p_i` folds the entire tail of the classical
// three-term history, so only `v_i` and `p_i` need be kept. `d_i` is chosen
// (via [`make_winv`]) as a maximal set of independent columns of `v_i^T A v_i`,
// giving priority to columns not selected last step, which guarantees the
// selections cover everything and bounds the run to ~m/(64-0.76) steps. The
// solution `x = sum_i v_i w_i^inv (v_i^T v_0)` accumulates online; with
// `v_0 = A Y` (random `Y`), `x + Y` lands in (near) `ker A`. Iteration stops
// when `d_i = 0`, i.e. `v_i^T A v_i = 0`.
//
// A rank drop at the final step can leave a small residual outside `ker A`, so
// the extraction re-derives valid vectors directly and *verifies* each
// dependency (`R^T z = 0`). A short or failed run therefore just yields fewer
// dependencies and the caller falls back to dense Gaussian -- correctness never
// depends on the iteration being numerically exact.

/// One column-selected inverse for block Lanczos. Given the symmetric 64x64
/// `vav = v_i^T A v_i`, returns `(winv, sel)`: `sel` is a bitmask of a maximal
/// set of *independent columns* of `vav` (columns not in `prev` are tried first,
/// so two consecutive selections cover all 64 coordinates), and `winv` is the
/// inverse of the principal submatrix `vav[sel,sel]` scattered back onto
/// `sel x sel` (zero elsewhere). By Montgomery's Lemma 1 the principal minor on
/// any maximal independent column set is non-singular, so the submatrix is
/// invertible. The result satisfies `winv = winv·d = d·winv` and
/// `winv·vav·d = d` where `d = diag(sel)`, and `winv` is symmetric.
fn make_winv(vav: &[u64; 64], prev: u64) -> ([u64; 64], u64) {
    // Priority order: coordinates not selected last step first.
    let mut order = [0u8; 64];
    let mut k = 0;
    for c in 0..64 {
        if prev & (1u64 << c) == 0 {
            order[k] = c as u8;
            k += 1;
        }
    }
    for c in 0..64 {
        if prev & (1u64 << c) != 0 {
            order[k] = c as u8;
            k += 1;
        }
    }
    // Select a maximal independent set of columns of `vav` (which, being
    // symmetric, equals the set of independent rows `vav[c]`), in priority order.
    let mut basis: Vec<(usize, u64)> = Vec::new(); // (pivot bit, reduced row)
    let mut sel = 0u64;
    for &cc in &order {
        let c = cc as usize;
        let mut row = vav[c];
        for &(pb, br) in &basis {
            if (row >> pb) & 1 == 1 {
                row ^= br;
            }
        }
        if row != 0 {
            let pb = row.trailing_zeros() as usize;
            basis.push((pb, row));
            sel |= 1u64 << c;
        }
    }
    // Invert the principal submatrix vav[sel,sel] by Gauss-Jordan, then scatter.
    let idx: Vec<usize> = (0..64).filter(|&c| sel & (1u64 << c) != 0).collect();
    let ks = idx.len();
    let mut left = alloc::vec![0u64; ks];
    let mut right = alloc::vec![0u64; ks];
    for (r, &ir) in idx.iter().enumerate() {
        right[r] = 1u64 << r;
        let mut row = 0u64;
        for (cpos, &ic) in idx.iter().enumerate() {
            if (vav[ir] >> ic) & 1 == 1 {
                row |= 1u64 << cpos;
            }
        }
        left[r] = row;
    }
    for col in 0..ks {
        let mut p = col;
        while p < ks && (left[p] >> col) & 1 == 0 {
            p += 1;
        }
        debug_assert!(p < ks, "principal submatrix must be invertible");
        left.swap(col, p);
        right.swap(col, p);
        for r in 0..ks {
            if r != col && (left[r] >> col) & 1 == 1 {
                left[r] ^= left[col];
                right[r] ^= right[col];
            }
        }
    }
    let mut winv = [0u64; 64];
    for (r, &ir) in idx.iter().enumerate() {
        let mut w = 0u64;
        let mut bits = right[r];
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            w |= 1u64 << idx[b];
            bits &= bits - 1;
        }
        winv[ir] = w;
    }
    (winv, sel)
}

/// 64x64 GF(2) product `x*y` (row `a` of the result is the XOR of the rows of
/// `y` selected by the set bits of row `a` of `x`).
fn mat_mul(x: &[u64; 64], y: &[u64; 64]) -> [u64; 64] {
    let mut z = [0u64; 64];
    for a in 0..64 {
        let mut r = 0u64;
        let mut bits = x[a];
        while bits != 0 {
            let k = bits.trailing_zeros() as usize;
            r ^= y[k];
            bits &= bits - 1;
        }
        z[a] = r;
    }
    z
}

/// 64x64 GF(2) sum (XOR).
#[inline]
fn mat_add(x: &[u64; 64], y: &[u64; 64]) -> [u64; 64] {
    let mut z = [0u64; 64];
    for i in 0..64 {
        z[i] = x[i] ^ y[i];
    }
    z
}

/// Right-multiplies a 64x64 matrix by the column projector `diag(mask)` -- keeps
/// only the columns in `mask` (each row ANDed with `mask`).
#[inline]
fn mat_proj(x: &[u64; 64], mask: u64) -> [u64; 64] {
    let mut z = [0u64; 64];
    for i in 0..64 {
        z[i] = x[i] & mask;
    }
    z
}

/// `V^T W` for two `rows x 64` block-vectors: a 64x64 matrix whose row `a` is the
/// XOR of the `W`-rows where `V` has a 1 in column `a`.
fn block_tmul(v: &[u64], w: &[u64]) -> [u64; 64] {
    let mut t = [0u64; 64];
    for (&vi, &wi) in v.iter().zip(w) {
        let mut bits = vi;
        while bits != 0 {
            let a = bits.trailing_zeros() as usize;
            t[a] ^= wi;
            bits &= bits - 1;
        }
    }
    t
}

/// `V * U` for a `rows x 64` block-vector and a 64x64 matrix (row `i` = XOR of
/// the `U`-rows selected by the bits of `V`'s row `i`).
fn block_mul(v: &[u64], u: &[u64; 64]) -> Vec<u64> {
    v.iter()
        .map(|&vi| {
            let mut r = 0u64;
            let mut bits = vi;
            while bits != 0 {
                let k = bits.trailing_zeros() as usize;
                r ^= u[k];
                bits &= bits - 1;
            }
            r
        })
        .collect()
}

/// GF(2) dependency search via Montgomery block Lanczos over the sparse relation
/// matrix `R` (row `i` = the set factor-base columns of relation `i`). Returns
/// the same kind of dependency subsets as [`find_dependencies`], or an empty
/// vector if the iteration produced none (the caller then falls back to dense).
/// Each returned subset is verified to satisfy `R^T z = 0` exactly.
fn block_lanczos_deps(relations: &[Relation], cols: usize) -> Vec<Vec<usize>> {
    let m = relations.len();
    if m == 0 || cols == 0 {
        return Vec::new();
    }
    // Sparse R: column indices per relation (the parity set bits).
    let rows: Vec<Vec<u32>> = relations
        .iter()
        .map(|r| {
            let mut idx = Vec::new();
            for (w, &word) in r.parity.iter().enumerate() {
                let mut b = word;
                while b != 0 {
                    let c = w * 64 + b.trailing_zeros() as usize;
                    if c < cols {
                        idx.push(c as u32);
                    }
                    b &= b - 1;
                }
            }
            idx
        })
        .collect();

    // R^T applied to an `m x 64` block -> `cols x 64`.
    let apply_rt = |x: &[u64]| -> Vec<u64> {
        let mut t = alloc::vec![0u64; cols];
        for (i, xi) in x.iter().enumerate() {
            for &j in &rows[i] {
                t[j as usize] ^= *xi;
            }
        }
        t
    };
    // A = R R^T applied to an `m x 64` block -> `m x 64`.
    let apply_a = |x: &[u64]| -> Vec<u64> {
        let t = apply_rt(x);
        rows.iter()
            .map(|ridx| {
                let mut acc = 0u64;
                for &j in ridx {
                    acc ^= t[j as usize];
                }
                acc
            })
            .collect()
    };

    // Random Y (deterministic, seeded from the relations so runs reproduce).
    let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ (m as u64).wrapping_mul(0xff51_afd7_ed55_8ccd);
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let y: Vec<u64> = (0..m).map(|_| rng()).collect();

    let v0 = apply_a(&y);
    let mut v = v0.clone();
    let mut p = alloc::vec![0u64; m]; // history accumulator p_i (p_0 = 0)
    let mut x = alloc::vec![0u64; m]; // solution accumulator
    let mut sel_prev = 0u64;

    let max_iter = m / 63 + 16;
    for _ in 0..max_iter {
        let av = apply_a(&v);
        let vav = block_tmul(&v, &av); // v_i^T A v_i
        if vav.iter().all(|&w| w == 0) {
            break; // d_i = 0: terminate.
        }
        let (winv, sel) = make_winv(&vav, sel_prev);
        if sel == 0 {
            break;
        }
        let notsel = !sel;
        let vaav = block_tmul(&av, &av); // v_i^T A^2 v_i

        // Accumulate x += v * winv * (v^T v0).
        let vtv0 = block_tmul(&v, &v0);
        let xadd = block_mul(&v, &mat_mul(&winv, &vtv0));
        for (xi, ai) in x.iter_mut().zip(&xadd) {
            *xi ^= *ai;
        }

        // c = winv (vaav·d + vav·(1-d)).
        let c = mat_mul(
            &winv,
            &mat_add(&mat_proj(&vaav, sel), &mat_proj(&vav, notsel)),
        );
        // v_{i+1} = A v·d + v·(1-d) - v·c - p·(vav·d).
        let vc = block_mul(&v, &c);
        let pv = block_mul(&p, &mat_proj(&vav, sel));
        let mut v_next = alloc::vec![0u64; m];
        for i in 0..m {
            v_next[i] = (av[i] & sel) ^ (v[i] & notsel) ^ vc[i] ^ pv[i];
        }
        // p_{i+1} = v·winv + p·(1-d).
        let vw = block_mul(&v, &winv);
        let mut p_next = alloc::vec![0u64; m];
        for i in 0..m {
            p_next[i] = vw[i] ^ (p[i] & notsel);
        }

        v = v_next;
        p = p_next;
        sel_prev = sel;
    }

    // Candidate block C = X + Y. Its columns should lie (near) ker A. Extract the
    // combinations of columns c with R^T c = 0 -- the exact dependencies.
    let cand: Vec<u64> = x.iter().zip(&y).map(|(&a, &b)| a ^ b).collect();
    let rtc = apply_rt(&cand); // cols x 64: column j's bits over the 64 candidates.
    let words = cols.div_ceil(64);
    // colvec[c] = the length-`cols` vector R^T(candidate column c), bit-packed.
    let mut colvec = alloc::vec![alloc::vec![0u64; words]; 64];
    for (j, &rj) in rtc.iter().enumerate() {
        let mut bits = rj;
        while bits != 0 {
            let c = bits.trailing_zeros() as usize;
            colvec[c][j / 64] |= 1u64 << (j % 64);
            bits &= bits - 1;
        }
    }
    // Null space among the 64 column-vectors: combos summing to zero over GF(2).
    let mut rowbits = colvec;
    let mut hist: Vec<u64> = (0..64u32).map(|c| 1u64 << c).collect();
    let mut pivots: Vec<(usize, usize)> = Vec::new();
    let mut combos: Vec<u64> = Vec::new();
    for r in 0..64 {
        // pivots only reference earlier rows (pr < r), so split the borrow.
        let (lo, hi) = rowbits.split_at_mut(r);
        let cur = &mut hi[0];
        for &(pb, pr) in &pivots {
            if (cur[pb / 64] >> (pb % 64)) & 1 == 1 {
                for (a, b) in cur.iter_mut().zip(&lo[pr]) {
                    *a ^= *b;
                }
                hist[r] ^= hist[pr];
            }
        }
        let piv = cur
            .iter()
            .enumerate()
            .find(|&(_, &x)| x != 0)
            .map(|(w, &x)| w * 64 + x.trailing_zeros() as usize);
        match piv {
            Some(b) => pivots.push((b, r)),
            None => combos.push(hist[r]),
        }
    }

    // Each zero-combo K over the candidate columns gives z = xor_{c in K} cand[:,c];
    // its set rows are the dependency's relation indices. Verify R^T z = 0.
    let mut deps = Vec::new();
    for &kmask in &combos {
        let subset: Vec<usize> = (0..m)
            .filter(|&i| (cand[i] & kmask).count_ones() & 1 == 1)
            .collect();
        if subset.is_empty() {
            continue;
        }
        let mut acc = alloc::vec![0u64; words];
        for &i in &subset {
            for (w, &word) in relations[i].parity.iter().enumerate() {
                acc[w] ^= word;
            }
        }
        if acc.iter().all(|&w| w == 0) {
            deps.push(subset);
        }
    }
    deps
}

/// Finds GF(2) dependencies among the relations, choosing the solver by size:
/// dense Gaussian elimination for small matrices (also the differential
/// reference), block Lanczos above a threshold -- falling back to dense if the
/// Lanczos run yields no dependency. Both always produce *valid* dependencies;
/// the caller verifies every candidate congruence by `gcd`, so the choice can
/// never affect correctness.
fn solve_dependencies(relations: &[Relation], cols: usize) -> Vec<Vec<usize>> {
    // Dense elimination is O(n^3); past a few thousand relations block Lanczos'
    // O(n*nnz) wins decisively and keeps the memory linear in the (sparse) matrix.
    const DENSE_MAX: usize = 2500;
    if relations.len() <= DENSE_MAX {
        return find_dependencies(relations, cols);
    }
    let deps = block_lanczos_deps(relations, cols);
    if !deps.is_empty() {
        deps
    } else {
        find_dependencies(relations, cols)
    }
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

    let deps = solve_dependencies(&relations, fb.len());
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
        51..=54 => (150_000, 160_000, 8),
        55..=57 => (300_000, 200_000, 9),
        _ => (600_000, 250_000, 10),
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

/// Accumulated large-prime state across a whole `siqs_factor` run, plus yield
/// counters for reporting. In **single-LP** mode `reps` maps each seen large
/// prime to a representative partial (its base and factor-base exponents), and
/// two partials sharing a prime combine directly. In **double-LP** mode the
/// `graph` holds a union-find forest over the large primes (Lenstra–Manasse
/// cycle counting, see [`LpGraph`]): each partial is an edge, and every graph
/// cycle yields one combined full relation.
struct LpStore {
    /// large prime → (base, factor-base exponents) of its representative partial
    /// (single-LP mode only).
    reps: BTreeMap<u64, (Nat, Vec<(usize, u32)>)>,
    /// Large-prime graph for the double-LP variation (`Some` iff double mode).
    graph: Option<LpGraph>,
    /// Relations found directly smooth (classic full relations). Read only by
    /// the yield benchmark; incremented always so the count is exact.
    #[cfg_attr(not(test), allow(dead_code))]
    direct: usize,
    /// Full relations synthesised by combining matched partials (single-LP) or
    /// closing a large-prime cycle (double-LP).
    #[cfg_attr(not(test), allow(dead_code))]
    combined: usize,
    /// Total partial relations seen (matched or not).
    #[cfg_attr(not(test), allow(dead_code))]
    partials_seen: usize,
}

impl LpStore {
    /// Single-large-prime store.
    fn new() -> Self {
        LpStore {
            reps: BTreeMap::new(),
            graph: None,
            direct: 0,
            combined: 0,
            partials_seen: 0,
        }
    }

    /// Double-large-prime store (partials tracked in a large-prime graph).
    fn new_double() -> Self {
        let mut s = LpStore::new();
        s.graph = Some(LpGraph::new());
        s
    }

    #[inline]
    fn is_double(&self) -> bool {
        self.graph.is_some()
    }

    /// Absorbs one partial relation (large primes `lp1`, and `lp2 = 1` for a
    /// single-LP partial or a second prime for a double). Returns a synthesised
    /// full relation when this partial completes a combination — a matched pair
    /// in single-LP mode, or a closed cycle in double-LP mode — else `None`.
    fn absorb(
        &mut self,
        n: &Nat,
        lp1: u64,
        lp2: u64,
        base: Nat,
        exps: Vec<(usize, u32)>,
        cols: usize,
    ) -> Option<Relation> {
        self.partials_seen += 1;
        if let Some(g) = self.graph.as_mut() {
            let rel = g.add_edge(n, lp1, lp2, base, exps, cols);
            if rel.is_some() {
                self.combined += 1;
            }
            rel
        } else {
            debug_assert_eq!(lp2, 1, "single-LP store received a double partial");
            match self.reps.get(&lp1) {
                Some((b0, e0)) => {
                    let rel = combine_partials(n, lp1, b0, e0, &base, &exps, cols);
                    if rel.is_some() {
                        self.combined += 1;
                    }
                    rel
                }
                None => {
                    self.reps.insert(lp1, (base, exps));
                    None
                }
            }
        }
    }
}

/// The large-prime graph for the double-large-prime variation
/// (Lenstra–Manasse / Pomerance–Smith cycle counting). Vertices are the large
/// primes; vertex `0` is a sentinel standing for "no prime" (the second slot of
/// a single-LP partial). Each partial relation is an **edge** joining the
/// vertices of its (one or two) large primes, carrying the relation's base value
/// and factor-base exponents. A partial that closes a **cycle** — its endpoints
/// already connected — combines the whole fundamental cycle into a full relation:
/// around a cycle each large prime is incident to exactly two edges, so every
/// large prime occurs squared in the product of the cycle's `Q`-values and
/// cancels, leaving a value smooth over the factor base.
///
/// The forest is a genuine spanning forest with parent pointers (no path
/// compression, so cycles can be traced); components are merged by re-rooting one
/// endpoint's tree and linking it under the other.
struct LpGraph {
    /// large prime → vertex index (`> 0`).
    id: BTreeMap<u64, usize>,
    /// vertex → its large prime (`0` for the sentinel vertex).
    prime_of: Vec<u64>,
    /// forest parent (a root points at itself).
    parent: Vec<usize>,
    /// base value of the partial on the edge to this vertex's parent (dummy at a
    /// root).
    edge_base: Vec<Nat>,
    /// factor-base exponents of that partial (dummy at a root).
    edge_exps: Vec<Vec<(usize, u32)>>,
}

impl LpGraph {
    fn new() -> Self {
        // Vertex 0 is the sentinel "no large prime".
        LpGraph {
            id: BTreeMap::new(),
            prime_of: alloc::vec![0],
            parent: alloc::vec![0],
            edge_base: alloc::vec![Nat::zero()],
            edge_exps: alloc::vec![Vec::new()],
        }
    }

    /// Vertex index for a large prime (`≤ 1` maps to the sentinel), creating it
    /// on first sight.
    fn vertex(&mut self, prime: u64) -> usize {
        if prime <= 1 {
            return 0;
        }
        if let Some(&v) = self.id.get(&prime) {
            return v;
        }
        let v = self.prime_of.len();
        self.id.insert(prime, v);
        self.prime_of.push(prime);
        self.parent.push(v);
        self.edge_base.push(Nat::zero());
        self.edge_exps.push(Vec::new());
        v
    }

    /// Root of the tree containing `x` (plain walk — the forest keeps its shape
    /// for cycle tracing, so no path compression).
    fn root(&self, mut x: usize) -> usize {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }

    /// Re-roots `v`'s tree at `v` by reversing the parent chain from `v` up to
    /// the old root, moving each edge's stored partial with it.
    fn reroot(&mut self, v: usize) {
        if self.parent[v] == v {
            return;
        }
        let mut prev = v;
        let mut cur = self.parent[v];
        let mut carry_base = core::mem::replace(&mut self.edge_base[v], Nat::zero());
        let mut carry_exps = core::mem::take(&mut self.edge_exps[v]);
        self.parent[v] = v;
        loop {
            let cur_was_root = self.parent[cur] == cur;
            let next = self.parent[cur];
            let next_base = core::mem::replace(&mut self.edge_base[cur], Nat::zero());
            let next_exps = core::mem::take(&mut self.edge_exps[cur]);
            self.parent[cur] = prev;
            self.edge_base[cur] = carry_base;
            self.edge_exps[cur] = carry_exps;
            if cur_was_root {
                break;
            }
            prev = cur;
            cur = next;
            carry_base = next_base;
            carry_exps = next_exps;
        }
    }

    /// Adds the partial relation for large primes `p1`, `p2` (`p2 = 1` for a
    /// single-LP partial) as an edge. If it closes a cycle, returns the combined
    /// full relation; otherwise links the two trees and returns `None`.
    fn add_edge(
        &mut self,
        n: &Nat,
        p1: u64,
        p2: u64,
        base: Nat,
        exps: Vec<(usize, u32)>,
        cols: usize,
    ) -> Option<Relation> {
        let u = self.vertex(p1);
        let v = self.vertex(p2);
        if u == v {
            // Self-loop (`p1 == p2`, i.e. `Q` had a squared large prime): a
            // one-edge cycle. The single large prime, squared, cancels.
            return build_cycle_relation(n, &[&base], &[&exps[..]], &[self.prime_of[u]], cols);
        }
        if self.root(u) == self.root(v) {
            // Both endpoints already connected: this edge closes a fundamental
            // cycle. Trace the two tree paths to their common ancestor.
            let mut au = Vec::new();
            {
                let mut x = u;
                loop {
                    au.push(x);
                    if self.parent[x] == x {
                        break;
                    }
                    x = self.parent[x];
                }
            }
            let au_pos: BTreeMap<usize, usize> =
                au.iter().enumerate().map(|(i, &x)| (x, i)).collect();
            let mut av = Vec::new();
            let mut lca_idx = au.len() - 1;
            {
                let mut y = v;
                loop {
                    av.push(y);
                    if let Some(&i) = au_pos.get(&y) {
                        lca_idx = i;
                        break;
                    }
                    if self.parent[y] == y {
                        break;
                    }
                    y = self.parent[y];
                }
            }
            // Cycle edges: u→LCA, v→LCA, and this closing edge.
            let mut bases: Vec<&Nat> = Vec::new();
            let mut exps_list: Vec<&[(usize, u32)]> = Vec::new();
            let mut primes: Vec<u64> = Vec::new();
            for &w in &au[..lca_idx] {
                bases.push(&self.edge_base[w]);
                exps_list.push(&self.edge_exps[w]);
            }
            for &w in &av[..av.len() - 1] {
                bases.push(&self.edge_base[w]);
                exps_list.push(&self.edge_exps[w]);
            }
            bases.push(&base);
            exps_list.push(&exps);
            for &w in &au[..=lca_idx] {
                primes.push(self.prime_of[w]);
            }
            for &w in &av[..av.len() - 1] {
                primes.push(self.prime_of[w]);
            }
            build_cycle_relation(n, &bases, &exps_list, &primes, cols)
        } else {
            // Distinct trees: merge by re-rooting v's tree and hanging it under u.
            self.reroot(v);
            self.parent[v] = u;
            self.edge_base[v] = base;
            self.edge_exps[v] = exps;
            None
        }
    }
}

/// Builds a full relation from the partials around one large-prime cycle: the
/// base is `∏ bᵢ · L⁻¹ (mod n)` where `L` is the product of the distinct large
/// primes on the cycle (each occurs squared in `∏ Qᵢ` and so halves into `L`),
/// and the exponent vector is the merge of every partial's factor-base
/// exponents. Returns `None` only if `L` is not invertible modulo `n` (it shares
/// a factor with `n` — a lucky split the dependency search will also find) or the
/// base collapses to zero.
fn build_cycle_relation(
    n: &Nat,
    bases: &[&Nat],
    exps_list: &[&[(usize, u32)]],
    cycle_primes: &[u64],
    cols: usize,
) -> Option<Relation> {
    let mut base = Nat::one();
    for b in bases {
        base = base.mul(b).div_rem(n).unwrap().1;
    }
    let mut l = Nat::one();
    for &p in cycle_primes {
        if p > 1 {
            l = l.mul(&Nat::from_u64(p)).div_rem(n).unwrap().1;
        }
    }
    let inv = Int::from(l).modinv(&Int::from(n.clone()))?.magnitude();
    base = base.mul(&inv).div_rem(n).unwrap().1;
    if base.is_zero() {
        return None;
    }
    let mut merged: Vec<(usize, u32)> = Vec::new();
    for e in exps_list {
        merged = merge_exps(&merged, e);
    }
    let parity = parity_of(&merged, cols);
    Some(Relation {
        base,
        exps: merged,
        parity,
    })
}

/// Sieves the current polynomial `Q(x) = (a·x + b)² − n` over `[−M, M]`,
/// collects the smooth relations, and appends them to `relations`. Partial
/// relations (a single leftover large prime `≤ lp_bound`) are routed through
/// `lp`: the first with a given large prime is stored, and each later one
/// sharing it is combined with the stored representative into a full relation,
/// itself appended to `relations`.
#[allow(clippy::too_many_arguments)]
fn sieve_poly(
    n: &Nat,
    ac: &ACoeff,
    b: &Int,
    fb: &[FbPrime],
    m: u64,
    target: u32,
    fudge: u32,
    lp_bound: u64,
    logs: &mut [u8],
    relations: &mut Vec<Relation>,
    lp: &mut LpStore,
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
    // The double-large-prime cofactor window: a composite leftover up to
    // `lp_bound²` (capped so the word-sized rho split stays cheap) whose two
    // prime factors both lie within the single-LP bound. Zero in single/off mode.
    let lp2_bound = if lp.is_double() {
        lp_bound.saturating_mul(lp_bound).min(1u64 << 58)
    } else {
        0
    };
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
        match siqs_relation(n, ac, base, fb, i, lp_bound, lp2_bound) {
            Cand::Full(rel) => {
                lp.direct += 1;
                relations.push(rel);
            }
            Cand::Partial {
                lp1,
                lp2,
                base,
                exps,
            } => {
                if let Some(rel) = lp.absorb(n, lp1, lp2, base, exps, fb.len()) {
                    relations.push(rel);
                }
            }
            Cand::None => {}
        }
        if relations.len() >= want {
            return;
        }
    }
}

/// The outcome of trial-dividing one SIQS sieve candidate over the factor base.
enum Cand {
    /// `Q(x)` factored completely over the base — a directly-smooth full relation.
    Full(Relation),
    /// `Q(x)` reduced to one or two leftover large primes — a *partial* relation,
    /// carrying its base value and factor-base exponents. A single large prime
    /// sets `lp2 = 1` (the sentinel "no second prime"); two large primes give a
    /// double partial `lp1·lp2`. Partials sharing a large prime combine — a pair
    /// sharing one (single-LP), or a whole cycle in the large-prime graph
    /// (double-LP, see [`LpGraph`]) — into a full relation.
    Partial {
        lp1: u64,
        lp2: u64,
        base: Nat,
        exps: Vec<(usize, u32)>,
    },
    /// Not usable: `Q(x)` retained a composite cofactor (or a prime above `LP`).
    None,
}

/// Packs a sparse exponent list into `GF(2)` parity words over `cols` columns.
fn parity_of(exps: &[(usize, u32)], cols: usize) -> Vec<u64> {
    let mut parity = alloc::vec![0u64; cols.div_ceil(64)];
    for &(idx, e) in exps {
        if e & 1 == 1 {
            parity[idx / 64] ^= 1u64 << (idx % 64);
        }
    }
    parity
}

/// Confirms and factors a SIQS candidate at array index `i`. Unlike the
/// single-polynomial `relation_from_base` (which trial-divides by every
/// factor-base prime), this uses the sieve roots: a prime `p ∤ a` divides
/// `Q(x)` exactly when `i` lands on one of its roots, so only the primes that
/// actually hit `i` — plus the primes dividing `a`, which divide every `Q` —
/// are divided out. This turns thousands of big-integer divisions per candidate
/// into a handful.
///
/// Because every factor-base prime dividing `Q(x)` is divided out, the residue
/// `mag` left over is coprime to the whole base. If it is `1` the candidate is
/// a directly-smooth full relation; if it is a single prime `≤ lp1_bound` the
/// candidate is kept as a *partial* relation tagged by that large prime (the
/// single-large-prime variation). When `lp2_bound > 0` the double-large-prime
/// variation is also active: a composite leftover `≤ lp2_bound` that splits into
/// exactly two primes, each in `(bound, lp1_bound]`, is kept as a double partial
/// (both large primes recorded). `lp1_bound = 0` disables partials (the classic
/// full-only behaviour). Anything else — a larger composite cofactor, or a prime
/// above the bound — is discarded.
fn siqs_relation(
    n: &Nat,
    ac: &ACoeff,
    base: Nat,
    fb: &[FbPrime],
    i: usize,
    lp1_bound: u64,
    lp2_bound: u64,
) -> Cand {
    let sq = base.square();
    let (mut mag, neg) = if sq >= *n {
        (sq.checked_sub(n).unwrap(), false)
    } else {
        (n.checked_sub(&sq).unwrap(), true)
    };
    if mag.is_zero() {
        return Cand::None;
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
    if mag.is_one() {
        let parity = parity_of(&exps, fb.len());
        return Cand::Full(Relation { base, exps, parity });
    }
    // A leftover cofactor, coprime to the whole factor base. Keep it iff it is a
    // single prime `≤ lp1_bound`, or (double-LP) a semiprime `≤ lp2_bound` whose
    // two prime factors both lie in `(bound, lp1_bound]`.
    if lp1_bound > 0
        && mag.bit_len() <= 62
        && let Some(v) = mag.to_u64()
    {
        if v <= lp1_bound && mag.is_prime_bpsw() {
            return Cand::Partial {
                lp1: v,
                lp2: 1,
                base,
                exps,
            };
        }
        if lp2_bound > 0
            && v <= lp2_bound
            && let Some((p, q)) = factor_two_primes(v, lp1_bound)
        {
            return Cand::Partial {
                lp1: p,
                lp2: q,
                base,
                exps,
            };
        }
    }
    Cand::None // composite (or too-large) leftover: not usable
}

/// Binary GCD on machine words.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Pollard's rho on a word-sized composite `n` (odd, `< 2⁶²`), returning a
/// non-trivial factor. All arithmetic stays in `u128` intermediates so the
/// modular multiply never overflows. Used only to split the small semiprime
/// cofactors of double-large-prime partials.
fn rho_u64(n: u64) -> Option<u64> {
    if n.is_multiple_of(2) {
        return Some(2);
    }
    let mut c = 1u64;
    while c < 32 {
        let f = |x: u64| ((x as u128 * x as u128 + c as u128) % n as u128) as u64;
        let (mut x, mut y) = (2u64, 2u64);
        let mut d = 1u64;
        while d == 1 {
            x = f(x);
            y = f(f(y));
            d = gcd_u64(x.abs_diff(y), n);
        }
        if d != n {
            return Some(d);
        }
        c += 1;
    }
    None
}

/// Splits a word-sized cofactor `m` into two primes both in `(fb-bound, max_p]`,
/// or returns `None` if `m` is prime, not a semiprime, or a factor falls outside
/// the large-prime window. The leftover of a fully-trial-divided candidate is
/// coprime to the whole factor base, so any prime factor already exceeds the
/// smoothness bound; only the upper bound `max_p` needs checking here.
fn factor_two_primes(m: u64, max_p: u64) -> Option<(u64, u64)> {
    if Nat::from_u64(m).is_prime_bpsw() {
        return None; // a single prime — handled by the single-LP branch
    }
    let d = rho_u64(m)?;
    let e = m / d;
    if d * e != m {
        return None;
    }
    if d <= max_p
        && e <= max_p
        && Nat::from_u64(d).is_prime_bpsw()
        && Nat::from_u64(e).is_prime_bpsw()
    {
        Some((d.min(e), d.max(e)))
    } else {
        None
    }
}

/// Merges two ascending sparse exponent lists, summing shared indices.
fn merge_exps(a: &[(usize, u32)], b: &[(usize, u32)]) -> Vec<(usize, u32)> {
    let mut out: Vec<(usize, u32)> = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(&(ia, ea)), Some(&(ib, eb))) => {
                if ia < ib {
                    out.push((ia, ea));
                    i += 1;
                } else if ib < ia {
                    out.push((ib, eb));
                    j += 1;
                } else {
                    out.push((ia, ea + eb));
                    i += 1;
                    j += 1;
                }
            }
            (Some(&e), None) => {
                out.push(e);
                i += 1;
            }
            (None, Some(&e)) => {
                out.push(e);
                j += 1;
            }
            (None, None) => break,
        }
    }
    out
}

/// Combines two partial relations sharing the large prime `lp` into a full one.
///
/// With `bᵢ² ≡ Qᵢ = lp·∏ p^{eᵢ} (mod n)`, set `base = b₁·b₂·lp⁻¹ (mod n)`; then
/// `base² ≡ Q₁·Q₂·lp⁻² = ∏ p^{e₁+e₂} (mod n)` is smooth over the factor base —
/// the large prime, squared, has dropped out. The result is an ordinary full
/// relation feeding the existing dependency search unchanged. Returns `None`
/// only in the degenerate cases (`lp` not invertible mod `n`, i.e. a factor of
/// `n`, or a zero base), which the caller simply skips.
fn combine_partials(
    n: &Nat,
    lp: u64,
    b1: &Nat,
    e1: &[(usize, u32)],
    b2: &Nat,
    e2: &[(usize, u32)],
    cols: usize,
) -> Option<Relation> {
    let inv = Int::from_u64(lp).modinv(&Int::from(n.clone()))?.magnitude();
    let base = b1
        .mul(b2)
        .div_rem(n)
        .unwrap()
        .1
        .mul(&inv)
        .div_rem(n)
        .unwrap()
        .1;
    if base.is_zero() {
        return None;
    }
    let exps = merge_exps(e1, e2);
    let parity = parity_of(&exps, cols);
    Some(Relation { base, exps, parity })
}

/// Attempts to split the odd composite `n` with the self-initializing MPQS,
/// using the single-large-prime variation (see [`siqs_run`]).
/// Returns a non-trivial factor, or `None` if it could not build a usable
/// `a`-coefficient, gather enough relations within the polynomial budget, or
/// turn any dependency into a proper factor (the caller then falls back).
/// `n` must not be a perfect square (the caller handles perfect powers).
pub(crate) fn siqs_factor(n: &Nat) -> Option<Nat> {
    // The double-large-prime variation (cycle counting) yields materially more
    // relations per polynomial in the larger range, where its extra cofactor
    // work pays off; below that the single-LP variation's lower per-candidate
    // overhead wins.
    let digits = (n.bit_len() * 30103 / 100000 + 1) as usize;
    let policy = if digits >= 45 {
        LargePrime::Double
    } else {
        LargePrime::Single
    };
    siqs_run(n, policy).0
}

/// The large-prime variation to apply while collecting SIQS relations.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LargePrime {
    /// Full relations only (the classic sieve) — used as the benchmark baseline.
    #[cfg_attr(not(test), allow(dead_code))]
    Off,
    /// Keep a single leftover prime in `(bound, bound·2⁷]` as a partial relation
    /// and combine matched partials into fulls.
    Single,
    /// Keep up to two leftover primes and combine via large-prime cycles
    /// (Lenstra–Manasse), extending relation yield further.
    Double,
}

/// Runs SIQS with the chosen large-prime policy, returning the factor (if any)
/// together with the relation-yield statistics ([`LpStore`]) for reporting and
/// benchmarking. Factored out of [`siqs_factor`] so the with/without-partials
/// comparison exercises the identical sieve.
///
/// The large-prime bound is `LP = bound·2⁷`. It sits comfortably inside the
/// existing log-threshold slack (`fudge ≈ log₂ bound + 11`), so a candidate
/// whose only missing factor is a prime `≤ LP` already clears the threshold and
/// reaches trial division — the variation merely *keeps* such a candidate (as a
/// partial) instead of discarding it, adding relations without widening the
/// sieve or admitting extra non-smooth candidates.
fn siqs_run(n: &Nat, policy: LargePrime) -> (Option<Nat>, LpStore) {
    let digits = (n.bit_len() * 30103 / 100000 + 1) as usize;
    let params = siqs_params_for(digits);
    let mut lp = if policy == LargePrime::Double {
        LpStore::new_double()
    } else {
        LpStore::new()
    };
    let fb = build_factor_base(n, params.bound);
    if fb.len() < 10 {
        return (None, lp);
    }
    let root = n.isqrt();
    if root.square() == *n {
        return (Some(root), lp); // perfect square
    }

    let target = (n.bit_len() as u32).div_ceil(2) + (64 - params.m.leading_zeros());
    let fudge = (64 - params.bound.leading_zeros()) + 10;
    let lp_bound = match policy {
        LargePrime::Off => 0,
        LargePrime::Single | LargePrime::Double => params.bound.saturating_mul(128),
    };
    // Combined-partial relations are individually valid but more linearly
    // correlated than directly-smooth ones (many share a representative partial,
    // or a cycle's edges), so more of the resulting dependencies can be trivial.
    // Carry a larger surplus under the large-prime policies to keep the
    // factor-yield at least as reliable as the full-only sieve while still
    // finishing in fewer polynomials.
    let margin = match policy {
        LargePrime::Off => 16 + fb.len() / 20,
        LargePrime::Single => 48 + fb.len() / 8,
        LargePrime::Double => 64 + fb.len() / 6,
    };
    let want = fb.len() + margin;

    let mut rng = Lcg(n.as_limbs().first().copied().unwrap_or(1) | 1);
    let mut relations: Vec<Relation> = Vec::new();
    let mut logs = alloc::vec![0u8; (2 * params.m + 1) as usize];

    // Bound the total polynomial work so a hard input escalates rather than
    // grinding; each `a` yields 2^{s−1} polynomials.
    let max_polys: u64 = 400_000;
    let mut polys = 0u64;
    'outer: while relations.len() < want && polys < max_polys {
        let Some((a, a_idx)) = choose_a(n, &fb, &params, &mut rng) else {
            return (None, lp);
        };
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
                lp_bound,
                &mut logs,
                &mut relations,
                &mut lp,
                want,
            );
            polys += 1;
            if relations.len() >= want {
                break 'outer;
            }
        }
    }

    if relations.len() <= fb.len() {
        return (None, lp);
    }
    let deps = solve_dependencies(&relations, fb.len());
    (factor_from_dependencies(n, &fb, &relations, &deps), lp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a synthetic relation carrying only a GF(2) parity vector (the base
    /// and exponents are irrelevant to the pure linear-algebra solvers).
    fn parity_relation(set_cols: &[usize], cols: usize) -> Relation {
        let mut parity = alloc::vec![0u64; cols.div_ceil(64)];
        for &c in set_cols {
            parity[c / 64] ^= 1u64 << (c % 64);
        }
        Relation {
            base: Nat::zero(),
            exps: Vec::new(),
            parity,
        }
    }

    /// A subset of relations is a valid dependency iff its parity vectors XOR to
    /// zero (and it is non-empty).
    fn is_valid_dep(relations: &[Relation], cols: usize, subset: &[usize]) -> bool {
        if subset.is_empty() {
            return false;
        }
        let mut acc = alloc::vec![0u64; cols.div_ceil(64)];
        for &i in subset {
            for (w, &word) in relations[i].parity.iter().enumerate() {
                acc[w] ^= word;
            }
        }
        acc.iter().all(|&w| w == 0)
    }

    /// Block Lanczos finds valid GF(2) dependencies, matching the dense solver in
    /// validity (not identity) across many random sparse matrices with more rows
    /// than columns. Exercises the solver directly on the parity matrix.
    #[test]
    fn block_lanczos_matches_dense_on_random() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for &(cols, extra) in &[
            (64usize, 40usize),
            (150, 60),
            (300, 80),
            (500, 90),
            (1200, 120),
        ] {
            let m = cols + extra; // more relations than columns => dependencies exist
            let weight = 6usize; // sparse rows
            let relations: Vec<Relation> = (0..m)
                .map(|_| {
                    let set: Vec<usize> = (0..weight).map(|_| (rng() as usize) % cols).collect();
                    parity_relation(&set, cols)
                })
                .collect();

            let dense = find_dependencies(&relations, cols);
            let lanczos = block_lanczos_deps(&relations, cols);

            assert!(
                !dense.is_empty(),
                "dense found no dependency (cols={cols}, m={m})"
            );
            assert!(
                !lanczos.is_empty(),
                "block Lanczos found no dependency (cols={cols}, m={m})"
            );
            for dep in &lanczos {
                assert!(
                    is_valid_dep(&relations, cols, dep),
                    "block Lanczos returned an invalid dependency (cols={cols})"
                );
            }
            for dep in &dense {
                assert!(is_valid_dep(&relations, cols, dep));
            }
        }
    }

    /// The 64x64 GF(2) principal-submatrix inverse produced by `make_winv` really
    /// inverts `vav` on the selected columns: `vav * winv * vav == vav`
    /// restricted to `sel`, and `sel` is non-empty for a non-zero `vav`.
    #[test]
    fn make_winv_inverts_on_selected() {
        let mut seed = 0xdead_beef_cafe_babeu64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..200 {
            // Build a random *symmetric* 64x64 matrix.
            let mut vav = [0u64; 64];
            for a in 0..64 {
                for b in a..64 {
                    if rng() & 1 == 1 {
                        vav[a] |= 1u64 << b;
                        vav[b] |= 1u64 << a;
                    }
                }
            }
            let (winv, sel) = make_winv(&vav, 0);
            // P = diag(sel). Check P*vav*winv*vav*P == P*vav*P.
            let prod = mat_mul(&vav, &mat_mul(&winv, &vav));
            for a in 0..64 {
                if sel & (1u64 << a) == 0 {
                    continue;
                }
                let lhs = prod[a] & sel;
                let rhs = vav[a] & sel;
                assert_eq!(lhs, rhs, "winv fails to invert vav on selected columns");
            }
        }
    }

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

    /// Checks the relation invariant `base² ≡ (−1)^{e₀}·∏ p^{e} (mod n)` — i.e.
    /// that the stored exponent factorization really is `Q = base² − n` up to
    /// multiples of `n`. Holds for directly-smooth *and* combined-partial fulls.
    fn check_relation(n: &Nat, fb: &[FbPrime], rel: &Relation) -> bool {
        let lhs = rel.base.square().div_rem(n).unwrap().1;
        let mut neg = false;
        let mut rhs = Nat::one();
        for &(idx, e) in &rel.exps {
            if idx == 0 {
                neg = e & 1 == 1;
                continue;
            }
            let pe = Nat::from_u64(fb[idx].p).modpow(&Nat::from_u64(e as u64), n);
            rhs = rhs.mul(&pe).div_rem(n).unwrap().1;
        }
        let rhs = if neg {
            n.checked_sub(&rhs).unwrap()
        } else {
            rhs
        };
        lhs == rhs
    }

    /// Combined-partial relations are algebraically valid: gather real relations
    /// (including many synthesised from matched partials) and confirm every one
    /// satisfies the square-congruence invariant. Heavy — run in release.
    #[test]
    #[ignore = "heavy: release-only relation-invariant check"]
    fn combined_partial_relations_are_valid() {
        let (n, _p, _q) = balanced_semiprime(38, 3);
        let params = siqs_params_for((n.bit_len() * 30103 / 100000 + 1) as usize);
        let fb = build_factor_base(&n, params.bound);
        let target = (n.bit_len() as u32).div_ceil(2) + (64 - params.m.leading_zeros());
        let fudge = (64 - params.bound.leading_zeros()) + 10;
        let lp_bound = params.bound.saturating_mul(128);
        let m = params.m;
        let mut logs = alloc::vec![0u8; (2 * m + 1) as usize];
        let mut rng = Lcg(n.as_limbs().first().copied().unwrap_or(1) | 1);
        let mut relations: Vec<Relation> = Vec::new();
        let mut lp = LpStore::new();

        // Sieve the full Gray-code family of a few `a`-coefficients so combined
        // relations actually arise.
        for _ in 0..8 {
            let Some((a, a_idx)) = choose_a(&n, &fb, &params, &mut rng) else {
                break;
            };
            let s = a_idx.len();
            let mut ac = init_a(&fb, m, a, a_idx);
            let mut signs = alloc::vec![1i64; s];
            let mut b = {
                let mut acc = Int::ZERO;
                for t in &ac.b_terms {
                    acc = acc.add(&Int::from(t.clone()));
                }
                acc
            };
            for iter in 0..(1u64 << (s - 1)) {
                if iter > 0 {
                    let j = iter.trailing_zeros() as usize;
                    let ds = signs[j];
                    signs[j] = -ds;
                    let two_bj = Int::from(ac.b_terms[j].clone()).mul(&Int::from_u64(2));
                    b = if ds > 0 {
                        b.sub(&two_bj)
                    } else {
                        b.add(&two_bj)
                    };
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
                    &n,
                    &ac,
                    &b,
                    &fb,
                    m,
                    target,
                    fudge,
                    lp_bound,
                    &mut logs,
                    &mut relations,
                    &mut lp,
                    usize::MAX,
                );
            }
        }

        assert!(lp.combined > 0, "expected some combined-partial relations");
        for rel in &relations {
            assert!(
                check_relation(&n, &fb, rel),
                "invalid relation base={:?}",
                rel.base
            );
        }
    }

    /// Decimal 10^k, for building semiprimes larger than `u128`.
    fn ten_pow(k: u32) -> Nat {
        let ten = Nat::from_u64(10);
        let mut r = Nat::one();
        for _ in 0..k {
            r = r.mul(&ten);
        }
        r
    }

    /// A balanced `d`-digit semiprime: two distinct primes of about `d/2` digits.
    fn balanced_semiprime(d: u32, seed: u64) -> (Nat, Nat, Nat) {
        let base = ten_pow(d / 2);
        let p = big_prime_at_least(&base.add(&Nat::from_u64(seed.wrapping_mul(2) + 1)));
        let q = big_prime_at_least(&base.add(&Nat::from_u64(seed.wrapping_mul(1000) + 12345)));
        let n = p.mul(&q);
        (n, p, q)
    }

    /// Correctness across the SIQS range with the large-prime variation on:
    /// every factor SIQS returns is an exact prime divisor, and the vast
    /// majority of cases are solved. (SIQS may decline on a few inputs — it is
    /// documented to return `None`, whereupon the production caller falls back —
    /// so this tolerates a small miss rate but demands every *returned* factor
    /// be correct.) Heavy — run in release:
    /// `cargo test --release siqs_lp_correctness -- --ignored`.
    #[test]
    #[ignore = "heavy: release-only SIQS batch"]
    fn siqs_lp_correctness_batch() {
        let (mut solved, mut total) = (0u32, 0u32);
        for &d in &[30u32, 34, 38, 42, 46] {
            for seed in 0..4u64 {
                let (n, p, q) = balanced_semiprime(d, seed);
                if p == q {
                    continue;
                }
                total += 1;
                let Some(f) = siqs_factor(&n) else { continue };
                solved += 1;
                assert!(f == p || f == q, "d={d} seed={seed}: bad factor {f:?}");
                let (cof, r) = n.div_rem(&f).unwrap();
                assert!(r.is_zero(), "d={d} seed={seed}: not a divisor");
                assert!(f.is_prime_bpsw() && cof.is_prime_bpsw());
                assert_eq!(f.mul(&cof), n, "d={d} seed={seed}: product mismatch");
            }
        }
        // Every returned factor was verified above; require a high solve rate so
        // the batch is a meaningful exercise of the large-prime path.
        assert!(
            solved * 10 >= total * 9,
            "solved {solved}/{total}: SIQS+LP solve rate too low"
        );
    }

    /// Relation-yield / timing benchmark: SIQS with vs without the single
    /// large-prime variation over representative balanced semiprimes. Prints the
    /// speedup and the share of relations that came from combined partials.
    /// Run: `cargo test --release siqs_lp_benchmark -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark: release-only, prints timings"]
    fn siqs_lp_benchmark() {
        use std::time::Instant;
        std::println!(
            "\n{:>4} {:>10} {:>10} {:>8} {:>9} {:>9} {:>7}",
            "dig",
            "off (ms)",
            "lp (ms)",
            "speedup",
            "combined",
            "partials",
            "share",
        );
        // Fastest of several runs per policy — min is more stable than mean
        // against scheduler noise (same convention as `examples/bench.rs`).
        let best = |n: &Nat, pol: LargePrime| -> (f64, LpStore) {
            let mut ms = f64::MAX;
            let mut store = LpStore::new();
            for _ in 0..3 {
                let t = Instant::now();
                let (f, s) = siqs_run(n, pol);
                ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
                assert!(f.is_some());
                store = s;
            }
            (ms, store)
        };
        for &d in &[38u32, 42, 44, 46, 48] {
            let (n, _p, _q) = balanced_semiprime(d, 1);
            let (off_ms, _soff) = best(&n, LargePrime::Off);
            let (lp_ms, slp) = best(&n, LargePrime::Single);

            let total = slp.direct + slp.combined;
            let share = if total > 0 {
                100.0 * slp.combined as f64 / total as f64
            } else {
                0.0
            };
            std::println!(
                "{d:>4} {off_ms:>10.1} {lp_ms:>10.1} {:>7.2}x {:>9} {:>9} {:>6.1}%",
                off_ms / lp_ms,
                slp.combined,
                slp.partials_seen,
                share,
            );
        }
    }

    /// Collects real SIQS relations for a balanced `d`-digit semiprime, sieving
    /// the Gray-code family of `a`-coefficients until at least `want` relations
    /// accumulate. Returns the modulus, the relations, and the factor base —
    /// shared feed for the differential dense-vs-Lanczos check below.
    fn gather_relations(d: u32, seed: u64, want: usize) -> (Nat, Vec<Relation>, Vec<FbPrime>) {
        let (n, _p, _q) = balanced_semiprime(d, seed);
        let digits = (n.bit_len() * 30103 / 100000 + 1) as usize;
        let params = siqs_params_for(digits);
        let fb = build_factor_base(&n, params.bound);
        let target = (n.bit_len() as u32).div_ceil(2) + (64 - params.m.leading_zeros());
        let fudge = (64 - params.bound.leading_zeros()) + 10;
        let lp_bound = params.bound.saturating_mul(128);
        let m = params.m;
        let mut logs = alloc::vec![0u8; (2 * m + 1) as usize];
        let mut rng = Lcg(n.as_limbs().first().copied().unwrap_or(1) | 1);
        let mut relations: Vec<Relation> = Vec::new();
        let mut lp = LpStore::new_double();
        while relations.len() < want {
            let Some((a, a_idx)) = choose_a(&n, &fb, &params, &mut rng) else {
                break;
            };
            let s = a_idx.len();
            let mut ac = init_a(&fb, m, a, a_idx);
            let mut signs = alloc::vec![1i64; s];
            let mut b = {
                let mut acc = Int::ZERO;
                for t in &ac.b_terms {
                    acc = acc.add(&Int::from(t.clone()));
                }
                acc
            };
            for iter in 0..(1u64 << (s - 1)) {
                if iter > 0 {
                    let j = iter.trailing_zeros() as usize;
                    let ds = signs[j];
                    signs[j] = -ds;
                    let two_bj = Int::from(ac.b_terms[j].clone()).mul(&Int::from_u64(2));
                    b = if ds > 0 {
                        b.sub(&two_bj)
                    } else {
                        b.add(&two_bj)
                    };
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
                    &n,
                    &ac,
                    &b,
                    &fb,
                    m,
                    target,
                    fudge,
                    lp_bound,
                    &mut logs,
                    &mut relations,
                    &mut lp,
                    want,
                );
                if relations.len() >= want {
                    break;
                }
            }
        }
        (n, relations, fb)
    }

    /// Differential check on a *real* SIQS relation matrix large enough that
    /// block Lanczos is the production solver: both dense Gaussian and block
    /// Lanczos find valid GF(2) dependencies, every Lanczos dependency is valid,
    /// and each solver's dependencies split the modulus. Heavy — run in release:
    /// `cargo test --release block_lanczos_real -- --ignored`.
    #[test]
    #[ignore = "heavy: release-only real relation matrix"]
    fn block_lanczos_real_relations() {
        let (n, relations, fb) = gather_relations(48, 1, 4200);
        assert!(
            relations.len() > 2500,
            "need a matrix past the dense threshold, got {}",
            relations.len()
        );

        let dense = find_dependencies(&relations, fb.len());
        let lanczos = block_lanczos_deps(&relations, fb.len());

        assert!(!dense.is_empty(), "dense found no dependency");
        assert!(!lanczos.is_empty(), "block Lanczos found no dependency");
        for dep in &lanczos {
            assert!(
                is_valid_dep(&relations, fb.len(), dep),
                "block Lanczos returned an invalid dependency"
            );
        }

        // Both solvers must be able to split n.
        let fd = factor_from_dependencies(&n, &fb, &relations, &dense).expect("dense splits");
        let fl = factor_from_dependencies(&n, &fb, &relations, &lanczos).expect("Lanczos splits");
        for f in [fd, fl] {
            assert!(!f.is_one() && f != n);
            let (cof, r) = n.div_rem(&f).unwrap();
            assert!(r.is_zero() && f.is_prime_bpsw() && cof.is_prime_bpsw());
        }
    }

    /// Correct, complete prime factorizations across the scaled SIQS range with
    /// block Lanczos + double large primes, at 45/50/55/60 digits. Prints the
    /// per-size timing. Heavy — run in release:
    /// `cargo test --release siqs_scaled_batch -- --ignored --nocapture`.
    #[test]
    #[ignore = "heavy: release-only scaled SIQS batch, prints timings"]
    fn siqs_scaled_batch() {
        use std::time::Instant;
        std::println!("\n{:>4} {:>10} {:>8}", "dig", "time (s)", "status");
        for &d in &[45u32, 50, 55, 60] {
            for seed in 0..2u64 {
                let (n, p, q) = balanced_semiprime(d, seed);
                if p == q {
                    continue;
                }
                let t = Instant::now();
                let f = siqs_factor(&n);
                let secs = t.elapsed().as_secs_f64();
                match f {
                    Some(f) => {
                        assert!(f == p || f == q, "d={d} seed={seed}: bad factor");
                        let (cof, r) = n.div_rem(&f).unwrap();
                        assert!(r.is_zero(), "d={d} seed={seed}: not a divisor");
                        assert!(
                            f.is_prime_bpsw() && cof.is_prime_bpsw() && f.mul(&cof) == n,
                            "d={d} seed={seed}: not a complete prime factorization"
                        );
                        std::println!("{d:>4} {secs:>10.2} {:>8}", "ok");
                    }
                    None => std::println!("{d:>4} {secs:>10.2} {:>8}", "none"),
                }
            }
        }
    }
}
