//! Atkin–Morain **ECPP** (Elliptic Curve Primality Proving) — the CM approach.
//!
//! This is the engine behind the `Ecpp` arm of [`PrimalityCertificate`]. Given a
//! probable prime `n` (already Baillie–PSW-passed) that the `n∓1` methods in
//! [`crate::primality`] could not settle, it produces one **downrun step**: an
//! elliptic curve `E/(ℤ/nℤ)` and a point `P` witnessing that the primality of `n`
//! reduces to the primality of a strictly smaller prime `q`, which is then proved
//! recursively.
//!
//! # The theorem (Goldwasser–Kilian / Atkin–Morain)
//!
//! Let `n > 1` with `gcd(n, 6) = 1`, let `E: y² = x³ + a·x + b` be an elliptic
//! curve over `ℤ/nℤ` (so `gcd(4a³ + 27b², n) = 1`), and let `m`, `q` be integers
//! with `q` prime, `q ∣ m`, and `q > (n^{1/4} + 1)²`. If there is a point
//! `P ∈ E(ℤ/nℤ)` with `[m]P = O` and `[m/q]P ≠ O`, then `n` is prime.
//!
//! *Proof sketch.* If `n` were composite it would have a prime factor `p ≤ √n`.
//! Reducing `E`, `P` mod `p`, the point `[m/q]P` is non-zero while `[m]P = O`, so
//! the order of `P` in `E(𝔽_p)` is a multiple of `q`; hence `q ∣ #E(𝔽_p)`. But by
//! Hasse `#E(𝔽_p) ≤ (√p + 1)² ≤ (n^{1/4} + 1)² < q`, and a non-trivial multiple of
//! `q` cannot be smaller than `q` — contradiction. (Atkin & Morain, *Math. Comp.*
//! **61** (1993); Cohen, *A Course in Computational Algebraic Number Theory*
//! §8.6; Crandall & Pomerance, *Prime Numbers*, §7.6.)
//!
//! The **soundness** of the whole module rests on this theorem alone: the
//! certificate records `(a, b, m, q, k = m/q, P)` and a recursive proof of `q`,
//! and the verifier re-checks `[m/q]P ≠ O` and `[m]P = O` *directly* on the curve.
//! Everything used to *find* the curve — the discriminant table, the Hilbert
//! class polynomials, Cornacchia, the candidate curve orders — only affects
//! whether a step is *found*; a wrong table entry can make a proof *fail*, never
//! make it *wrong*.
//!
//! # How a step is built
//!
//! 1. For fundamental discriminants `D < 0` (increasing `|D|`, from
//!    [`DISCRIMINANTS`]) with `(D/n) = 1`, solve `4n = t² + |D|v²` by **Cornacchia**
//!    (Cohen, Algorithm 1.5.3).
//! 2. The candidate curve orders are `m = n + 1 ∓ a` for the CM traces `a`
//!    (`±t`, plus the extra units for `D = −3, −4`). Accept an `m` that factors as
//!    `m = k·q` with `k` smooth and `q` a probable prime with `q > (n^{1/4}+1)²`.
//! 3. Build the CM curve: find a root `j` of the Hilbert class polynomial `H_D`
//!    modulo `n` (trivial for the nine class-number-one `D`, where `H_D = x − j`;
//!    otherwise by modular root-finding), derive `(a, b)` from `j`, and pick the
//!    correct quadratic twist by testing a random point.
//! 4. Emit `(a, b, m, q, k, P)`; the caller proves `q` recursively.
//!
//! # Clean-room provenance
//!
//! Every algorithm is from the open literature cited above (plus Schertz for the
//! class-polynomial construction and the standard `j`-invariant values of the
//! nine class-number-one imaginary quadratic fields). No third-party source code
//! (PARI/GP, mpz_aprcl, …) was consulted.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::int::Int;
use crate::mod_int::ModInt;
use crate::poly::Poly;
use crate::primality::{EcppCert, Primality, prove_prime};

/// Trial-division bound for peeling the smooth part `k` of a candidate order.
const SMOOTH_BOUND: u64 = 1 << 16;

/// How many distinct base-point `x`-coordinates to try when hunting for a witness
/// point on a fixed curve.
const POINT_TRIES: u64 = 60;

/// How many curves with a fixed `j`-invariant `0` or `1728` to try (their sextic
/// / quartic twists realise the several candidate orders).
const TWIST_TRIES: u64 = 40;

/// One imaginary quadratic discriminant and its Hilbert class polynomial `H_D`,
/// stored as integer coefficients low-to-high (monic, so the leading `1` is
/// implicit-checked). For the nine class-number-one discriminants `H_D = x − j`.
struct Disc {
    /// The (negative, fundamental) discriminant `D`.
    d: i64,
    /// Coefficients of `H_D`, low-to-high, monic (`h(D) + 1` of them).
    poly: &'static [i128],
}

/// Discriminant table, smallest `|D|` first. The nine class-number-one entries
/// carry `H_D = x − j` with the classical integral `j`-invariants; a couple of
/// class-number-two entries carry the full quadratic `H_D` (exercised by the
/// modular root-finder). Extending this table is a data-only change — soundness
/// does not depend on it.
const DISCRIMINANTS: &[Disc] = &[
    // --- class number 1: H_D(x) = x - j ---
    Disc {
        d: -3,
        poly: &[0, 1],
    },
    Disc {
        d: -4,
        poly: &[-1728, 1],
    },
    Disc {
        d: -7,
        poly: &[3375, 1],
    },
    Disc {
        d: -8,
        poly: &[-8000, 1],
    },
    Disc {
        d: -11,
        poly: &[32768, 1],
    },
    Disc {
        d: -19,
        poly: &[884736, 1],
    },
    Disc {
        d: -43,
        poly: &[884736000, 1],
    },
    Disc {
        d: -67,
        poly: &[147197952000, 1],
    },
    Disc {
        d: -163,
        poly: &[262537412640768000, 1],
    },
    // --- class number 2: H_D(x) = x² - (j1+j2) x + j1 j2 ---
    // D = -15: roots (-191025 ± 85995√5)/2.
    Disc {
        d: -15,
        poly: &[-121287375, 191025, 1],
    },
];

/// An affine point on `E/(ℤ/nℤ)`, or the point at infinity (`None`).
type Pt = Option<(ModInt, ModInt)>;

/// Outcome of a group operation over `ℤ/nℤ`: either a point, or a failed
/// inversion (which exposes a factor of `n`, so `n` is composite).
type EcResult = Result<Pt, ()>;

/// `2·P` on `y² = x³ + a·x + b` over `ℤ/nℤ`. `Err(())` if the required inversion
/// fails (a factor of `n` was hit — `n` is composite).
fn ec_double(a: &ModInt, p: &Pt) -> EcResult {
    let (x, y) = match p {
        None => return Ok(None),
        Some(pt) => pt,
    };
    if y.is_zero() {
        return Ok(None); // 2-torsion: vertical tangent
    }
    let three = y.of(Int::from(3));
    let two = y.of(Int::from(2));
    let num = three.mul(&x.mul(x)).add(a);
    let den = two.mul(y);
    let inv = den.inv().ok_or(())?;
    let lam = num.mul(&inv);
    let x3 = lam.mul(&lam).sub(&x.mul(&two));
    let y3 = lam.mul(&x.sub(&x3)).sub(y);
    Ok(Some((x3, y3)))
}

/// `P + Q` on `y² = x³ + a·x + b` over `ℤ/nℤ`. `Err(())` on a failed inversion.
fn ec_add(a: &ModInt, p: &Pt, q: &Pt) -> EcResult {
    let (x1, y1) = match p {
        None => return Ok(q.clone()),
        Some(pt) => pt,
    };
    let (x2, y2) = match q {
        None => return Ok(p.clone()),
        Some(pt) => pt,
    };
    if x1 == x2 {
        if y1 == y2 {
            return ec_double(a, p);
        }
        return Ok(None); // P = -Q
    }
    let den = x2.sub(x1);
    let inv = den.inv().ok_or(())?;
    let lam = y2.sub(y1).mul(&inv);
    let x3 = lam.mul(&lam).sub(x1).sub(x2);
    let y3 = lam.mul(&x1.sub(&x3)).sub(y1);
    Ok(Some((x3, y3)))
}

/// `[scalar]·P` by double-and-add (`scalar ≥ 0`). `Err(())` on a failed inversion.
fn ec_mul(a: &ModInt, p: &Pt, scalar: &Int) -> EcResult {
    if scalar.is_zero() || p.is_none() {
        return Ok(None);
    }
    let mut acc: Pt = None;
    let mut i = scalar.bit_len();
    while i > 0 {
        i -= 1;
        acc = ec_double(a, &acc)?;
        if scalar.bit(i) {
            acc = ec_add(a, &acc, p)?;
        }
    }
    Ok(acc)
}

/// `⌊√x⌋` for `x ≥ 0`.
fn isqrt(x: &Int) -> Int {
    Int::from(x.magnitude().isqrt())
}

/// Tests `q > (n^{1/4} + 1)²` with pure integer arithmetic (no roots).
///
/// Writing `w = q − 1`, the real condition `q > (n^{1/4}+1)²` is equivalent, for
/// `w > √n`, to `(w² + n)² > 4n·(w + 2)²`; both are required (the first also
/// rules out the tiny-`q` branch where the squaring would be invalid).
fn q_large_enough(n: &Int, q: &Int) -> bool {
    let w = q.sub(&Int::ONE);
    let w2 = w.mul(&w);
    if w2 <= *n {
        return false;
    }
    let lhs = w2.add(n);
    let lhs = lhs.mul(&lhs);
    let wp2 = w.add(&Int::from(2));
    let rhs = Int::from(4).mul(n).mul(&wp2).mul(&wp2);
    lhs > rhs
}

/// Modified Cornacchia (Cohen, Algorithm 1.5.3): solves `t² + |D|·v² = 4n` for a
/// prime `n` and a negative discriminant `D ≡ 0, 1 (mod 4)` with `|D| < 4n`.
/// Returns `(t, v)` with `t ≥ 0`, or `None` when no solution exists.
fn cornacchia(n: &Int, d: i64) -> Option<(Int, Int)> {
    debug_assert!(d < 0);
    let di = Int::from(d);
    let abs_d = Int::from(-d);
    // (D/n) must be a square for a solution to exist.
    if di.jacobi(n) != 1 {
        return None;
    }
    // x0 = √D mod n, adjusted so x0 ≡ D (mod 2).
    let mut x0 = di.sqrt_mod(n)?;
    let d_odd = di.rem_euclid(&Int::from(2)).is_one();
    if x0.is_odd() != d_odd {
        x0 = n.sub(&x0);
    }
    // Euclidean descent from (2n, x0) until the remainder drops to ≤ ⌊2√n⌋.
    let four_n = Int::from(4).mul(n);
    let limit = isqrt(&four_n); // ⌊√(4n)⌋ = ⌊2√n⌋
    let mut a = Int::from(2).mul(n);
    let mut b = x0;
    while b > limit {
        if b.is_zero() {
            return None;
        }
        let r = a.div_rem_trunc(&b).1;
        a = b;
        b = r;
    }
    // t = b; check 4n − t² = |D|·v² with v² a perfect square.
    let rem = four_n.sub(&b.mul(&b));
    if rem.is_negative() {
        return None;
    }
    let (c, r) = rem.div_rem_trunc(&abs_d);
    if !r.is_zero() {
        return None;
    }
    let v = c.sqrt_exact()?;
    Some((b, v))
}

/// The candidate curve orders `m = n + 1 − a` for the CM traces `a` of
/// discriminant `D` given a Cornacchia solution `(t, v)`. Always includes `a = ±t`;
/// for `D = −4` also `a = ±2v`, and for `D = −3` the two extra unit multiples
/// `a = (t ± 3v)/2` (when integral). Extra entries only widen the search — an
/// order no curve realises is silently discarded downstream.
fn candidate_orders(n: &Int, d: i64, t: &Int, v: &Int) -> Vec<Int> {
    let np1 = n.add(&Int::ONE);
    let mut traces: Vec<Int> = alloc::vec![t.clone(), t.neg()];
    match d {
        -4 => {
            let two_v = Int::from(2).mul(v);
            traces.push(two_v.clone());
            traces.push(two_v.neg());
        }
        -3 => {
            let three_v = Int::from(3).mul(v);
            for s in [t.add(&three_v), t.sub(&three_v)] {
                if s.is_even() {
                    let a = s.div_trunc(&Int::from(2));
                    traces.push(a.clone());
                    traces.push(a.neg());
                }
            }
        }
        _ => {}
    }
    let mut orders: Vec<Int> = Vec::new();
    for a in traces {
        let m = np1.sub(&a);
        if m > Int::ONE && !orders.contains(&m) {
            orders.push(m);
        }
    }
    orders
}

/// Peels the smooth part of `m` (primes `≤ SMOOTH_BOUND`) and returns `(q, k)`
/// with `m = k·q` when the residual cofactor `q` is a probable prime satisfying
/// `q < n` (so the downrun makes progress) and `q > (n^{1/4}+1)²`. Otherwise
/// `None`.
fn split_order(m: &Int, n: &Int) -> Option<(Int, Int)> {
    let mut c = m.clone();
    let two = Int::from(2);
    while c.is_even() {
        c = c.div_trunc(&two);
    }
    let mut d = 3u64;
    while d <= SMOOTH_BOUND {
        let dd = Int::from(d);
        if dd.mul(&dd) > c {
            break;
        }
        loop {
            let (quot, rem) = c.div_rem_trunc(&dd);
            if rem.is_zero() {
                c = quot;
            } else {
                break;
            }
        }
        d += 2;
    }
    if c <= Int::ONE || &c >= n {
        return None;
    }
    if !q_large_enough(n, &c) {
        return None;
    }
    if !c.is_prime_bpsw() {
        return None;
    }
    let k = m.div_exact(&c);
    Some((c, k))
}

/// Finds a root of the Hilbert class polynomial `entry` modulo `n`, returned as a
/// `ModInt` `j`-invariant, or `None` if it has no root mod `n` (which, for a
/// genuine prime `n` and a valid table, means this `D` is unusable — the caller
/// moves on). `sample` seeds the shared ring `ℤ/nℤ`.
fn class_poly_root(entry: &Disc, sample: &ModInt) -> Option<ModInt> {
    // Degree 1 (class number one): H_D = x - j, so the root is the constant j.
    if entry.poly.len() == 2 {
        return Some(sample.of(Int::from(entry.poly[0])).neg());
    }
    let coeffs: Vec<ModInt> = entry
        .poly
        .iter()
        .map(|&c| sample.of(Int::from(c)))
        .collect();
    let f = Poly::new(coeffs);
    poly_any_root(&f, sample)
}

/// `x` as the degree-1 polynomial over the shared ring.
fn poly_x(sample: &ModInt) -> Poly<ModInt> {
    Poly::new(alloc::vec![sample.of(Int::ZERO), sample.of(Int::ONE)])
}

/// `base^e mod modulus` in `(ℤ/nℤ)[x]` by square-and-multiply.
fn poly_powmod(
    base: &Poly<ModInt>,
    e: &Int,
    modulus: &Poly<ModInt>,
    sample: &ModInt,
) -> Poly<ModInt> {
    let mut result = Poly::constant(sample.of(Int::ONE));
    let mut b = base.rem(modulus);
    let mut i = 0u32;
    let bits = e.bit_len();
    while i < bits {
        if e.bit(i) {
            result = result.mul(&b).rem(modulus);
        }
        i += 1;
        if i < bits {
            b = b.mul(&b).rem(modulus);
        }
    }
    result
}

/// Returns some root of `f` in `ℤ/nℤ` (`n` the shared ring's prime modulus), or
/// `None` if `f` has no root there. Uses the standard root-extraction: intersect
/// `f` with `x^n − x` (the product of all linear factors), then split off a
/// linear factor by Cantor–Zassenhaus equal-degree splitting.
fn poly_any_root(f: &Poly<ModInt>, sample: &ModInt) -> Option<ModInt> {
    let n = sample.modulus();
    let x = poly_x(sample);
    // g = gcd(f, x^n - x) collects exactly the roots living in 𝔽_n.
    let xn = poly_powmod(&x, &n, f, sample);
    let g = f.gcd(&xn.sub(&x));
    match g.degree() {
        None | Some(0) => None,
        _ => split_off_root(&g, sample, &n),
    }
}

/// One root of a nonzero polynomial `g` that is known to split into distinct
/// linear factors over `𝔽_n`. Cantor–Zassenhaus with successive shifts.
fn split_off_root(g: &Poly<ModInt>, sample: &ModInt, n: &Int) -> Option<ModInt> {
    if g.degree() == Some(1) {
        // g = c1·x + c0 (monic after gcd): root = -c0/c1.
        let c0 = g.coeff(0);
        let c1 = g.coeff(1);
        return Some(c0.neg().div(&c1));
    }
    let half = n.sub(&Int::ONE).div_trunc(&Int::from(2)); // (n-1)/2
    let one = Poly::constant(sample.of(Int::ONE));
    let mut delta = 1u64;
    loop {
        // (x + delta)^{(n-1)/2} - 1 shares a nontrivial gcd with g roughly half
        // the time, splitting g into two proper factors.
        let shifted = Poly::new(alloc::vec![
            sample.of(Int::from(delta)),
            sample.of(Int::ONE)
        ]);
        let powered = poly_powmod(&shifted, &half, g, sample);
        let cand = g.gcd(&powered.sub(&one));
        match cand.degree() {
            Some(dg) if dg >= 1 && dg < g.degree().unwrap() => {
                let (quot, _) = g.div_rem(&cand);
                let smaller = if cand.degree() <= quot.degree() {
                    cand
                } else {
                    quot
                };
                return split_off_root(&smaller, sample, n);
            }
            _ => {}
        }
        delta += 1;
        if delta > 4 * n.bit_len() as u64 + 64 {
            return None; // give up (should not happen for a genuine prime n)
        }
    }
}

/// Smallest quadratic non-residue mod the prime `n`, as an `Int` in `[2, n)`.
fn nonresidue(n: &Int) -> Int {
    let mut u = 2u64;
    loop {
        let ui = Int::from(u);
        if ui.jacobi(n) == -1 {
            return ui;
        }
        u += 1;
    }
}

/// A found curve+point realising a target order: curve `(a, b)` and a witness
/// point `P = (px, py)` on it.
struct CurveWitness {
    a: ModInt,
    b: ModInt,
    px: ModInt,
    py: ModInt,
}

/// `4a³ + 27b²` (the discriminant numerator, up to the constant `−16`).
fn disc_num(a: &ModInt, b: &ModInt) -> ModInt {
    let four = a.of(Int::from(4));
    let twenty_seven = a.of(Int::from(27));
    four.mul(&a.mul(a).mul(a)).add(&twenty_seven.mul(&b.mul(b)))
}

/// Tries to find, on the curve `y² = x³ + a·x + b` over `ℤ/nℤ`, a point `P` with
/// `[k]P ≠ O` and `[q]([k]P) = O` — i.e. a witness that the curve order is the
/// target `m = k·q`. Returns:
/// - `Ok(Some(P))` on success;
/// - `Ok(None)` if this curve does *not* have order `m` (wrong twist), so the
///   caller should try another curve;
/// - `Err(())` if the curve is singular mod `n` or an inversion failed (`n`
///   composite).
fn witness_on_curve(a: &ModInt, b: &ModInt, k: &Int, q: &Int, sample: &ModInt) -> Result<Pt, ()> {
    // Reject a singular curve (would not be an elliptic curve over ℤ/nℤ).
    let dnum = disc_num(a, b);
    if dnum.is_zero() || dnum.inv().is_none() {
        return Err(());
    }
    let n = sample.modulus();
    let mut tried = 0u64;
    let mut xv = 1u64;
    while tried < POINT_TRIES {
        let x = sample.of(Int::from(xv));
        xv += 1;
        // rhs = x³ + a·x + b; need it to be a nonzero quadratic residue.
        let rhs = x.mul(&x).mul(&x).add(&a.mul(&x)).add(b);
        if rhs.is_zero() {
            continue;
        }
        let rhs_int = rhs.to_int();
        if rhs_int.jacobi(&n) != 1 {
            continue;
        }
        tried += 1;
        let y = match rhs_int.sqrt_mod(&n) {
            Some(y) => sample.of(y),
            None => continue,
        };
        let p: Pt = Some((x, y));
        let qk = ec_mul(a, &p, k)?; // [k]P
        if qk.is_none() {
            continue; // P killed by k; try another point
        }
        let top = ec_mul(a, &qk, q)?; // [q]([k]P) = [m]P
        if top.is_none() {
            return Ok(qk); // success: [k]P ≠ O and [m]P = O
        }
        // [m]P ≠ O ⇒ curve order ≠ m: wrong twist, stop scanning points.
        return Ok(None);
    }
    Ok(None)
}

/// Builds a curve with `j`-invariant `j` realising the target order `m = k·q`,
/// and a witness point. Handles the special `j = 0` / `j = 1728` families by
/// scanning twists; the generic case tries the standard curve and its quadratic
/// twist. Returns `None` if no realisation is found.
fn build_curve(j: &ModInt, k: &Int, q: &Int, sample: &ModInt, n: &Int) -> Option<CurveWitness> {
    let zero = sample.of(Int::ZERO);
    let j1728 = sample.of(Int::from(1728));
    if j.is_zero() {
        // j = 0 (D = -3): y² = x³ + b, scan b over the sextic twists.
        for bb in 1..=TWIST_TRIES {
            let a = zero.clone();
            let b = sample.of(Int::from(bb));
            if let Ok(Some((px, py))) = witness_on_curve(&a, &b, k, q, sample) {
                return Some(CurveWitness { a, b, px, py });
            }
        }
        return None;
    }
    if *j == j1728 {
        // j = 1728 (D = -4): y² = x³ + a·x, scan a over the quartic twists.
        for aa in 1..=TWIST_TRIES {
            let a = sample.of(Int::from(aa));
            let b = zero.clone();
            if let Ok(Some((px, py))) = witness_on_curve(&a, &b, k, q, sample) {
                return Some(CurveWitness { a, b, px, py });
            }
        }
        return None;
    }
    // Generic j: k_j = j/(1728 - j); E: y² = x³ + 3·k_j·x + 2·k_j.
    let denom = j1728.sub(j);
    let inv = denom.inv()?;
    let kj = j.mul(&inv);
    let a0 = sample.of(Int::from(3)).mul(&kj);
    let b0 = sample.of(Int::from(2)).mul(&kj);
    // Try the curve and its single quadratic twist (a·u², b·u³, u a non-residue).
    let u = sample.of(nonresidue(n));
    let u2 = u.mul(&u);
    let u3 = u2.mul(&u);
    for (a, b) in [(a0.clone(), b0.clone()), (a0.mul(&u2), b0.mul(&u3))] {
        if let Ok(Some((px, py))) = witness_on_curve(&a, &b, k, q, sample) {
            return Some(CurveWitness { a, b, px, py });
        }
    }
    None
}

/// Attempts to build one ECPP downrun step for `n` and recursively prove the
/// resulting prime `q`, returning the assembled certificate arm. `n` must be a
/// probable prime with `gcd(n, 6) = 1` and `n` large enough that
/// `(n^{1/4}+1)²`-sized primes below it exist (the caller guarantees this by
/// only invoking ECPP for `n ≥ 2⁶⁴`).
pub(crate) fn prove_ecpp(n: &Int) -> Option<EcppCert> {
    let sample = ModInt::new(Int::ZERO, n.clone());
    for entry in DISCRIMINANTS {
        let (t, v) = match cornacchia(n, entry.d) {
            Some(sol) => sol,
            None => continue,
        };
        let orders = candidate_orders(n, entry.d, &t, &v);
        // Any order must factor as k·q before the (possibly costly) root search.
        let mut splits: Vec<(Int, Int, Int)> = Vec::new(); // (m, q, k)
        for m in &orders {
            if let Some((q, k)) = split_order(m, n) {
                splits.push((m.clone(), q, k));
            }
        }
        if splits.is_empty() {
            continue;
        }
        let j = match class_poly_root(entry, &sample) {
            Some(j) => j,
            None => continue, // no root: this D is unusable for n
        };
        for (m, q, k) in splits {
            let cw = match build_curve(&j, &k, &q, &sample, n) {
                Some(cw) => cw,
                None => continue,
            };
            // Prove q recursively; on failure, try the next candidate/discriminant.
            match prove_prime(&q) {
                Primality::Prime(child) => {
                    return Some(EcppCert {
                        a: cw.a.to_int(),
                        b: cw.b.to_int(),
                        m,
                        q,
                        k,
                        px: cw.px.to_int(),
                        py: cw.py.to_int(),
                        q_proof: Box::new(child),
                    });
                }
                _ => continue,
            }
        }
    }
    None
}

/// Independently re-checks one ECPP certificate arm for `n`: re-derives the curve
/// and point, re-runs the two scalar multiplications, and re-checks the
/// Goldwasser–Kilian side conditions and the recursive proof of `q`. Shares no
/// state with [`prove_ecpp`]; any tampered field makes it return `false`.
pub(crate) fn verify_ecpp(n: &Int, cert: &EcppCert) -> bool {
    // n must be > 3 and coprime to 6.
    if n <= &Int::from(3) || n.is_even() || n.div_rem_trunc(&Int::from(3)).1.is_zero() {
        return false;
    }
    // m = k·q, k ≥ 1, q < n.
    if cert.m != cert.k.mul(&cert.q) {
        return false;
    }
    if cert.k < Int::ONE || cert.q >= *n {
        return false;
    }
    // q > (n^{1/4}+1)².
    if !q_large_enough(n, &cert.q) {
        return false;
    }
    // q is prime: recursively verified (or small-BPSW inside its own cert).
    if cert.q_proof.n() != &cert.q || !cert.q_proof.verify(&cert.q) {
        return false;
    }
    let sample = ModInt::new(Int::ZERO, n.clone());
    let a = sample.of(cert.a.clone());
    let b = sample.of(cert.b.clone());
    // Non-singular curve over ℤ/nℤ: 4a³ + 27b² invertible mod n.
    let dnum = disc_num(&a, &b);
    if dnum.is_zero() || dnum.inv().is_none() {
        return false;
    }
    // P must lie on the curve.
    let px = sample.of(cert.px.clone());
    let py = sample.of(cert.py.clone());
    let lhs = py.mul(&py);
    let rhs = px.mul(&px).mul(&px).add(&a.mul(&px)).add(&b);
    if lhs != rhs {
        return false;
    }
    let p: Pt = Some((px, py));
    // Q = [k]P must be finite (≠ O)...
    let qk = match ec_mul(&a, &p, &cert.k) {
        Ok(pt) => pt,
        Err(()) => return false, // inversion failed ⇒ n composite
    };
    if qk.is_none() {
        return false;
    }
    // ...and [q]Q = [m]P must be O.
    matches!(ec_mul(&a, &qk, &cert.q), Ok(None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primality::{Primality, prove_prime};

    /// `q_large_enough` must exactly implement `q > (n^{1/4}+1)²`. Checked at the
    /// boundary for `n` a perfect fourth power (where the bound is the integer
    /// `(r+1)²`) and for the worked example `n = 100`.
    #[test]
    fn q_bound_is_exact() {
        // n = r^4 ⇒ n^{1/4} = r ⇒ bound = (r+1)², integer.
        for r in [2u64, 3, 10, 100, 1000] {
            let n = Int::from(r).pow(4);
            let thresh = (r + 1) * (r + 1); // (r+1)^2
            assert!(
                !q_large_enough(&n, &Int::from(thresh)),
                "q=(r+1)² must fail"
            );
            assert!(
                q_large_enough(&n, &Int::from(thresh + 1)),
                "q=(r+1)²+1 must pass"
            );
        }
        // n = 100: (100^{1/4}+1)² ≈ 17.32, so 18 passes and 17 fails.
        let n = Int::from(100);
        assert!(q_large_enough(&n, &Int::from(18)));
        assert!(!q_large_enough(&n, &Int::from(17)));
    }

    /// Cornacchia returns a genuine solution of `t² + |D|v² = 4n` when it
    /// succeeds, and only when `(D/n) = 1`.
    #[test]
    fn cornacchia_identity_holds() {
        let n = Int::from(1000003);
        for d in [-3i64, -4, -7, -8, -11, -19, -43, -67, -163] {
            match cornacchia(&n, d) {
                Some((t, v)) => {
                    let lhs = t.mul(&t).add(&Int::from(-d).mul(&v.mul(&v)));
                    assert_eq!(lhs, Int::from(4).mul(&n), "D={d}");
                    assert_eq!(Int::from(d).jacobi(&n), 1);
                }
                None => assert_ne!(Int::from(d).jacobi(&n), 1, "D={d} missed a solution"),
            }
        }
    }

    /// Modular root-finding recovers a root of the class-number-two H_{-15}.
    #[test]
    fn class_poly_root_degree_two() {
        // A prime where (-15/n) = 1 so H_{-15} splits mod n.
        let n = Int::from(2_000_000_089u64);
        let sample = ModInt::new(Int::ZERO, n.clone());
        let entry = DISCRIMINANTS.iter().find(|e| e.d == -15).unwrap();
        if Int::from(-15).jacobi(&n) == 1 {
            let j = class_poly_root(entry, &sample).expect("root exists");
            // H_{-15}(j) ≡ 0 mod n.
            let jj = j.clone();
            let val = jj
                .mul(&jj)
                .add(&sample.of(Int::from(191025)).mul(&jj))
                .sub(&sample.of(Int::from(121287375)));
            assert!(val.is_zero(), "j must be a root of H_-15 mod n");
        }
    }

    /// A produced ECPP step verifies, its `q` is a smaller prime, and the whole
    /// recursive chain re-checks. Cross-checked against Baillie–PSW.
    #[test]
    fn ecpp_step_proves_and_verifies() {
        let primes = [
            "18446744073709551629",                    // > 2^64
            "170141183460469231731687303715884105727", // 2^127 − 1
            "340282366920938463463374607431768211507", // > 2^128
        ];
        for ps in primes {
            let n = Int::from_str_radix(ps, 10).unwrap();
            let cert = prove_ecpp(&n).expect("ECPP should find a step");
            assert!(verify_ecpp(&n, &cert), "cert must verify: {ps}");
            assert!(cert.q < n, "q must be smaller than n");
            assert!(cert.q.is_prime_bpsw());
            assert_eq!(cert.m, cert.k.mul(&cert.q));
        }
    }

    /// Every single-field mutation of a valid ECPP certificate is rejected.
    #[test]
    fn tampered_ecpp_is_rejected() {
        let n = Int::from_str_radix("170141183460469231731687303715884105727", 10).unwrap();
        let cert = prove_ecpp(&n).expect("step");
        assert!(verify_ecpp(&n, &cert));

        let bump = |x: &Int| x.add(&Int::ONE);

        let mut c = cert.clone();
        c.a = bump(&c.a);
        assert!(!verify_ecpp(&n, &c), "tampered a");

        let mut c = cert.clone();
        c.b = bump(&c.b);
        assert!(!verify_ecpp(&n, &c), "tampered b");

        let mut c = cert.clone();
        c.m = bump(&c.m);
        assert!(!verify_ecpp(&n, &c), "tampered m (breaks m=k·q)");

        let mut c = cert.clone();
        c.q = bump(&c.q);
        assert!(!verify_ecpp(&n, &c), "tampered q");

        let mut c = cert.clone();
        c.k = bump(&c.k);
        assert!(!verify_ecpp(&n, &c), "tampered k");

        let mut c = cert.clone();
        c.px = bump(&c.px);
        assert!(!verify_ecpp(&n, &c), "tampered px (P off curve)");

        let mut c = cert.clone();
        c.py = bump(&c.py);
        assert!(!verify_ecpp(&n, &c), "tampered py (P off curve)");

        // A forged small q that is prime but violates the size bound must fail.
        let mut c = cert.clone();
        c.q = Int::from(7);
        c.k = c.m.div_trunc(&Int::from(7));
        assert!(!verify_ecpp(&n, &c), "q too small must fail the bound");
    }

    /// ECPP must never manufacture a certificate for a composite: fed a composite
    /// coprime to 6 directly (bypassing Baillie–PSW), `prove_ecpp` returns `None`.
    #[test]
    fn ecpp_never_certifies_composite() {
        // Products of two primes, both > 2^32, coprime to 6.
        let composites = [
            Int::from(4294967311u64).mul(&Int::from(4294967357u64)),
            Int::from(10000000019u64).mul(&Int::from(10000000033u64)),
        ];
        for c in composites {
            assert!(!c.is_prime_bpsw());
            assert!(prove_ecpp(&c).is_none(), "no ECPP cert for composite {c}");
        }
    }

    /// A larger prime whose whole downrun is exercised (slower).
    #[test]
    #[ignore = "slow: full ECPP downrun of a ~200-bit prime"]
    fn ecpp_large_downrun() {
        // (2^198 + 15) is prime.
        let n = Int::from(2).pow(198).add(&Int::from(15));
        assert!(n.is_prime_bpsw());
        let cert = prove_ecpp(&n).expect("ECPP downrun");
        assert!(verify_ecpp(&n, &cert));
    }

    /// The public `prove_prime` routes to ECPP for an `n−1`-hard prime and the
    /// unified certificate verifies.
    #[test]
    fn public_prove_prime_uses_ecpp() {
        // n = 2·p·q + 1 (156-bit); n−1's cofactor is a hard ~147-bit semiprime,
        // so the n−1 methods give up and ECPP takes over.
        let n = Int::from_str_radix("60881034391988601177056022264639143542433476079", 10).unwrap();
        assert!(n.is_prime_bpsw());
        match prove_prime(&n) {
            Primality::Prime(cert) => {
                assert_eq!(cert.bound(), None, "should be an ECPP (not n−1) proof");
                assert!(cert.verify(&n));
                // Tampering the unified certificate's target n is rejected.
                assert!(!cert.verify(&n.add(&Int::from(2))));
            }
            other => panic!("expected ECPP proof, got {other:?}"),
        }
    }
}
