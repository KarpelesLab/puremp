//! The General Number Field Sieve (GNFS) integer factorization.
//!
//! This is a clean-room, **correctness-first** end-to-end GNFS pipeline. The
//! goal is a *correct* split of a composite `N` into two nontrivial factors, not
//! record performance: every stage favours provable correctness over speed, and
//! the public entry point [`Int::factor_gnfs`] can only ever return a *true*
//! factorization (it verifies the split divides `N`) or `None`.
//!
//! # Pipeline
//!
//! 1. **Polynomial selection (base-`m`).** Pick a degree `d` (3–5 by the size of
//!    `N`), set `m = ⌊N^{1/d}⌋`, and expand `N` in base `m` to obtain a monic
//!    `f(x) = x^d + c_{d-1}x^{d-1} + … + c_0` with `f(m) = N` (so `f(m) ≡ 0 mod
//!    N`) and `g(x) = x − m`. `θ` denotes a root of `f`; `K = ℚ[x]/(f)`.
//! 2. **Factor bases.** A rational base (primes `p ≤ B_r`, dividing `a − bm`),
//!    an algebraic base (pairs `(p, r)` with `f(r) ≡ 0 mod p`, controlling
//!    `N(a − bθ) = Σ c_i a^i b^{d-i}`), and a quadratic-character base (pairs
//!    `(q, s)`) forcing a genuine square on the algebraic side.
//! 3. **Sieving.** Line sieving over coprime `(a, b)`: find `(a, b)` with `a − bm`
//!    smooth over the rational base *and* `N(a − bθ)` smooth over the algebraic
//!    base.
//! 4. **Linear algebra.** A GF(2) matrix over {rational sign, algebraic-norm
//!    sign, rational-prime parities, algebraic-prime parities, quadratic
//!    characters}; dependencies are found with the crate's block-Lanczos solver
//!    (reused from the quadratic sieve).
//! 5. **Square roots.** Rational: `∏(a − bm)` has even prime exponents, so its
//!    square root mod `N` is read straight off the summed exponents. Algebraic:
//!    with `γ = f'(θ)^2 · ∏(a − bθ)` (an element of `ℤ[θ]` that is a perfect
//!    square), compute `δ = √γ ∈ ℤ[θ]` by a **single-prime Hensel lift** — a
//!    finite-field square root in `𝔽_p[x]/(f)` (Tonelli–Shanks) lifted by a
//!    coupled Newton iteration to `ℤ/p^kℤ`, then reconstructed with balanced
//!    residues and **verified exactly** (`δ·δ == γ`).
//! 6. **Combine.** The ring map `φ: ℤ[θ] → ℤ/Nℤ`, `θ ↦ m`, sends `δ ↦ X`; with
//!    the rational root `Y = f'(m)·√∏(a−bm)` one has `X² ≡ Y² (mod N)`, so
//!    `gcd(X − Y, N)` (which always divides `N`) yields a factor. Other
//!    dependencies are tried if a split is trivial.
//!
//! The size range actually handled is documented on [`Int::factor_gnfs`].

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::int::Int;
use crate::mod_int::ModInt;
use crate::poly::Poly;
use crate::poly_finite_field::FactorOverField;

// ===========================================================================
// Small helpers
// ===========================================================================

/// Sieve of Eratosthenes: all primes `≤ limit`.
fn primes_up_to(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let n = limit as usize;
    let mut sieve = vec![true; n + 1];
    sieve[0] = false;
    sieve[1] = false;
    let mut p = 2usize;
    while p * p <= n {
        if sieve[p] {
            let mut q = p * p;
            while q <= n {
                sieve[q] = false;
                q += p;
            }
        }
        p += 1;
    }
    (2..=n).filter(|&i| sieve[i]).map(|i| i as u64).collect()
}

/// The roots in `[0, p)` of `f (mod p)` (coefficients low-to-high), brute force.
fn roots_mod_p(f: &[Int], p: u64) -> Vec<u64> {
    let mut coeffs: Vec<u64> = f
        .iter()
        .map(|c| {
            let r = c.rem_euclid(&Int::from_u64(p));
            r.to_u64().unwrap_or(0)
        })
        .collect();
    while coeffs.len() > 1 && *coeffs.last().unwrap() == 0 {
        coeffs.pop();
    }
    let mut out = Vec::new();
    for x in 0..p {
        // Horner mod p in u128.
        let mut acc: u128 = 0;
        for &c in coeffs.iter().rev() {
            acc = (acc * x as u128 + c as u128) % p as u128;
        }
        if acc == 0 {
            out.push(x);
        }
    }
    out
}

/// Modular inverse of `a mod p` for small moduli via extended Euclid on i128.
fn u64_modinv(a: u64, p: u64) -> Option<u64> {
    let (mut old_r, mut r) = (a as i128 % p as i128, p as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        let nr = old_r - q * r;
        old_r = r;
        r = nr;
        let ns = old_s - q * s;
        old_s = s;
        s = ns;
    }
    if old_r != 1 {
        return None;
    }
    Some(old_s.rem_euclid(p as i128) as u64)
}

// ===========================================================================
// ℤ[θ]/(f) element arithmetic, f monic of degree d (power basis 1,θ,…,θ^{d-1})
// ===========================================================================

/// Multiply two elements (each a length-`d` coefficient vector) modulo `f`,
/// optionally reducing each coefficient modulo `modulus`. `f` is monic degree
/// `d`, given low-to-high as `f[0..=d]` with `f[d] == 1`.
fn el_mul(a: &[Int], b: &[Int], f: &[Int], d: usize, modulus: Option<&Int>) -> Vec<Int> {
    let mut t = vec![Int::zero(); 2 * d - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            if bj.is_zero() {
                continue;
            }
            let prod = ai.mul(bj);
            t[i + j] = t[i + j].add(&prod);
        }
    }
    // Reduce degrees >= d using θ^d = -(f[0] + f[1]θ + … + f[d-1]θ^{d-1}).
    for k in (d..2 * d - 1).rev() {
        if t[k].is_zero() {
            continue;
        }
        let coef = t[k].clone();
        t[k] = Int::zero();
        for (i, fi) in f.iter().enumerate().take(d) {
            // t[k-d+i] -= f[i] * coef
            let sub = fi.mul(&coef);
            let idx = k - d + i;
            t[idx] = t[idx].sub(&sub);
        }
    }
    t.truncate(d);
    t.resize(d, Int::zero());
    if let Some(m) = modulus {
        for c in &mut t {
            *c = c.rem_euclid(m);
        }
    }
    t
}

/// Reduce an element's coefficients modulo `m` into `[0, m)`.
fn el_reduce(a: &[Int], d: usize, m: &Int) -> Vec<Int> {
    let mut v: Vec<Int> = a.iter().map(|c| c.rem_euclid(m)).collect();
    v.resize(d, Int::zero());
    v
}

/// Element `[c, 0, …]` (a constant), reduced modulo `m` when given.
fn el_const(c: i64, d: usize, m: Option<&Int>) -> Vec<Int> {
    let mut v = vec![Int::zero(); d];
    let ci = Int::from_i64(c);
    v[0] = match m {
        Some(m) => ci.rem_euclid(m),
        None => ci,
    };
    v
}

fn el_sub(a: &[Int], b: &[Int], m: &Int) -> Vec<Int> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.sub(y).rem_euclid(m))
        .collect()
}

fn el_eq(a: &[Int], b: &[Int]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

// ---- finite field 𝔽_p[x]/(f) ≅ 𝔽_{p^d} (f irreducible mod p) ----

fn ff_mul(a: &[Int], b: &[Int], f: &[Int], d: usize, p: &Int) -> Vec<Int> {
    el_mul(a, b, f, d, Some(p))
}

fn ff_pow(a: &[Int], e: &Int, f: &[Int], d: usize, p: &Int) -> Vec<Int> {
    let mut result = el_const(1, d, Some(p));
    let mut base = a.to_vec();
    let bits = e.bit_len();
    for i in 0..bits {
        if e.bit(i) {
            result = ff_mul(&result, &base, f, d, p);
        }
        if i + 1 < bits {
            base = ff_mul(&base, &base, f, d, p);
        }
    }
    result
}

/// Square root in `𝔽_{p^d}` (Tonelli–Shanks). Returns `None` if the element is a
/// non-residue.
fn ff_sqrt(a: &[Int], f: &[Int], d: usize, p: &Int) -> Option<Vec<Int>> {
    let one = el_const(1, d, Some(p));
    let minus_one = {
        let mut v = vec![Int::zero(); d];
        v[0] = p.sub(&Int::one());
        v
    };
    // q = p^d
    let q = p.pow(d as u32);
    let q_minus_1 = q.sub(&Int::one());
    // Legendre: a^{(q-1)/2}
    let half = q_minus_1.div_exact(&Int::from_i64(2));
    let ls = ff_pow(a, &half, f, d, p);
    if el_eq(&ls, &minus_one) {
        return None; // non-residue
    }
    if !el_eq(&ls, &one) {
        return None; // a == 0 or malformed; caller should avoid
    }
    // Write q-1 = 2^S * Q, Q odd.
    let mut s_exp = 0u32;
    let mut qodd = q_minus_1.clone();
    let two = Int::from_i64(2);
    while qodd.is_even() {
        qodd = qodd.div_exact(&two);
        s_exp += 1;
    }
    if s_exp == 1 {
        // q ≡ 3 (mod 4): sqrt = a^{(q+1)/4}
        let e = q.add(&Int::one()).div_exact(&Int::from_i64(4));
        let r = ff_pow(a, &e, f, d, p);
        if el_eq(&ff_mul(&r, &r, f, d, p), &el_reduce(a, d, p)) {
            return Some(r);
        }
        return None;
    }
    // Find a non-residue z by enumerating field elements.
    let mut z = None;
    let mut counter: u64 = 2;
    let pv = p.to_u64().unwrap_or(0);
    while z.is_none() && counter < 1_000_000 {
        // element from base-p digits of `counter`
        let mut v = vec![Int::zero(); d];
        let mut c = counter;
        let mut idx = 0;
        while c > 0 && idx < d {
            v[idx] = Int::from_u64(c % pv);
            c /= pv;
            idx += 1;
        }
        let chk = ff_pow(&v, &half, f, d, p);
        if el_eq(&chk, &minus_one) {
            z = Some(v);
        }
        counter += 1;
    }
    let z = z?;
    let mut m_exp = s_exp;
    let mut c = ff_pow(&z, &qodd, f, d, p);
    let qp1_half = qodd.add(&Int::one()).div_exact(&two);
    let mut t = ff_pow(a, &qodd, f, d, p);
    let mut r = ff_pow(a, &qp1_half, f, d, p);
    loop {
        if el_eq(&t, &one) {
            return Some(r);
        }
        // least i in [1, m_exp) with t^{2^i} == 1
        let mut i = 0u32;
        let mut t2 = t.clone();
        while !el_eq(&t2, &one) {
            t2 = ff_mul(&t2, &t2, f, d, p);
            i += 1;
            if i >= m_exp {
                return None;
            }
        }
        // b = c^{2^{m_exp - i - 1}}
        let mut b = c.clone();
        for _ in 0..(m_exp - i - 1) {
            b = ff_mul(&b, &b, f, d, p);
        }
        m_exp = i;
        c = ff_mul(&b, &b, f, d, p);
        t = ff_mul(&t, &c, f, d, p);
        r = ff_mul(&r, &b, f, d, p);
    }
}

// ===========================================================================
// Stage 1: polynomial selection (base-m)
// ===========================================================================

/// Base-`m` selection for a chosen degree `d`. Returns the monic `f` coefficients
/// (low-to-high, length `d+1`, `f[d] == 1`) and `m`, or `None` if the leading
/// digit is not 1 (i.e. `f` would be non-monic for this `d`).
fn select_poly(n: &Int, d: usize) -> Option<(Vec<Int>, Int)> {
    let nnat = n.magnitude();
    let m = Int::from(nnat.nth_root_floor(d as u32));
    if m < Int::from_i64(2) {
        return None;
    }
    // Base-m digits of N: c_0 + c_1 m + … .
    let mut digits = Vec::new();
    let mut rem = n.clone();
    while !rem.is_zero() {
        let (qq, rr) = rem.div_rem_floor(&m);
        digits.push(rr);
        rem = qq;
    }
    if digits.len() != d + 1 {
        return None;
    }
    if digits[d] != Int::one() {
        return None; // not monic
    }
    Some((digits, m))
}

// ===========================================================================
// Factor bases
// ===========================================================================

struct FactorBases {
    /// rational primes p ≤ B_r
    rat_primes: Vec<u64>,
    /// algebraic (p, r) pairs with f(r) ≡ 0 mod p, p ≤ B_a
    alg_pairs: Vec<(u64, u64)>,
    /// (p, r) -> column offset within the algebraic block
    alg_index: BTreeMap<(u64, u64), usize>,
    /// distinct algebraic primes (for trial division)
    alg_primes: Vec<u64>,
    /// quadratic-character pairs (q, s), q > B_a
    qc_pairs: Vec<(u64, u64)>,
}

fn build_factor_bases(f: &[Int], d: usize, br: u64, ba: u64, num_qc: usize) -> FactorBases {
    let rat_primes = primes_up_to(br);
    let alg_prime_candidates = primes_up_to(ba);
    let mut alg_pairs = Vec::new();
    let mut alg_index = BTreeMap::new();
    let mut alg_primes = Vec::new();
    for &p in &alg_prime_candidates {
        let roots = roots_mod_p(f, p);
        if !roots.is_empty() {
            alg_primes.push(p);
        }
        for r in roots {
            alg_index.insert((p, r), alg_pairs.len());
            alg_pairs.push((p, r));
        }
    }
    // Quadratic characters: primes just above B_a with a simple root of f.
    let mut qc_pairs = Vec::new();
    let mut cand = Int::from_u64(ba).next_prime();
    let fprime_c = derivative_coeffs(f, d);
    while qc_pairs.len() < num_qc {
        let p = match cand.to_u64() {
            Some(v) => v,
            None => break,
        };
        if p > 1u64 << 40 {
            break;
        }
        for r in roots_mod_p(f, p) {
            // require f'(r) != 0 mod p (simple root) so the character is well-defined
            let fp = eval_mod_p(&fprime_c, r, p);
            if fp != 0 {
                qc_pairs.push((p, r));
                if qc_pairs.len() >= num_qc {
                    break;
                }
            }
        }
        cand = cand.next_prime();
    }
    FactorBases {
        rat_primes,
        alg_pairs,
        alg_index,
        alg_primes,
        qc_pairs,
    }
}

/// Coefficients (low-to-high) of `f'` given `f` monic degree `d`.
fn derivative_coeffs(f: &[Int], d: usize) -> Vec<Int> {
    // f'(x) = Σ_{i=1}^{d} i c_i x^{i-1}
    let mut out = vec![Int::zero(); d];
    for i in 1..=d {
        out[i - 1] = f[i].mul(&Int::from_u64(i as u64));
    }
    out
}

fn eval_mod_p(coeffs: &[Int], x: u64, p: u64) -> u64 {
    let mut acc: u128 = 0;
    for c in coeffs.iter().rev() {
        let cr = c.rem_euclid(&Int::from_u64(p)).to_u64().unwrap_or(0) as u128;
        acc = (acc * x as u128 + cr) % p as u128;
    }
    acc as u64
}

/// `F(a, b) = Σ_{i=0}^{d} c_i a^i b^{d-i}` (the norm of `a − bθ`, up to sign).
fn norm_ab(f: &[Int], d: usize, a: &Int, b: &Int) -> Int {
    let mut acc = Int::zero();
    // powers of a and b
    let mut apow = vec![Int::one(); d + 1];
    let mut bpow = vec![Int::one(); d + 1];
    for i in 1..=d {
        apow[i] = apow[i - 1].mul(a);
        bpow[i] = bpow[i - 1].mul(b);
    }
    for i in 0..=d {
        let term = f[i].mul(&apow[i]).mul(&bpow[d - i]);
        acc = acc.add(&term);
    }
    acc
}

// ===========================================================================
// Relations
// ===========================================================================

struct Relation {
    a: Int,
    b: Int,
    rat_exps: Vec<(usize, u32)>,
    parity: Vec<u64>,
}

/// Try to factor `value` completely over the given prime list; returns the
/// sparse exponent vector `(index, exp)` if fully smooth (cofactor 1), else
/// `None`.
fn factor_smooth(value: &Int, primes: &[u64]) -> Option<Vec<(usize, u32)>> {
    let mut m = value.abs();
    if m.is_zero() {
        return None;
    }
    let mut exps = Vec::new();
    for (idx, &p) in primes.iter().enumerate() {
        if m.is_one() {
            break;
        }
        let pint = Int::from_u64(p);
        let mut e = 0u32;
        loop {
            let (q, r) = m.div_rem_trunc(&pint);
            if r.is_zero() {
                m = q;
                e += 1;
            } else {
                break;
            }
        }
        if e > 0 {
            exps.push((idx, e));
        }
    }
    if m.is_one() { Some(exps) } else { None }
}

fn set_bit(parity: &mut [u64], col: usize) {
    parity[col / 64] ^= 1u64 << (col % 64);
}

#[allow(clippy::too_many_arguments)]
fn make_relation(
    f: &[Int],
    d: usize,
    m: &Int,
    fb: &FactorBases,
    a: &Int,
    b: &Int,
    cols: usize,
    col_rat: usize,
    col_alg: usize,
    col_qc: usize,
) -> Option<Relation> {
    // Rational side: value = a - b*m.
    let rat_val = a.sub(&b.mul(m));
    if rat_val.is_zero() {
        return None;
    }
    let rat_exps = factor_smooth(&rat_val, &fb.rat_primes)?;
    // Algebraic side: norm F(a,b).
    let norm = norm_ab(f, d, a, b);
    if norm.is_zero() {
        return None;
    }
    let alg_exps = factor_smooth(&norm, &fb.alg_primes)?;

    let mut parity = vec![0u64; cols.div_ceil(64)];
    // sign columns
    if rat_val.is_negative() {
        set_bit(&mut parity, 0);
    }
    if norm.is_negative() {
        set_bit(&mut parity, 1);
    }
    // rational prime parities
    for &(idx, e) in &rat_exps {
        if e & 1 == 1 {
            set_bit(&mut parity, col_rat + idx);
        }
    }
    // algebraic prime parities: attribute exp of p to (p, r0), r0 = a/b mod p.
    for &(pidx, e) in &alg_exps {
        if e & 1 == 0 {
            continue;
        }
        let p = fb.alg_primes[pidx];
        // p | b cannot happen for monic f (then p ∤ norm), but stay safe.
        let binv = u64_modinv(b.rem_euclid(&Int::from_u64(p)).to_u64().unwrap_or(0), p)?;
        let ar = a.rem_euclid(&Int::from_u64(p)).to_u64().unwrap_or(0);
        let r0 = ((ar as u128 * binv as u128) % p as u128) as u64;
        let col = *fb.alg_index.get(&(p, r0))?;
        set_bit(&mut parity, col_alg + col);
    }
    // quadratic characters
    for (k, &(q, s)) in fb.qc_pairs.iter().enumerate() {
        let qi = Int::from_u64(q);
        let val = a.sub(&b.mul(&Int::from_u64(s))).rem_euclid(&qi);
        if val.is_zero() {
            return None; // character undefined; drop this relation
        }
        if val.legendre(&qi) == -1 {
            set_bit(&mut parity, col_qc + k);
        }
    }
    Some(Relation {
        a: a.clone(),
        b: b.clone(),
        rat_exps,
        parity,
    })
}

// ===========================================================================
// Sieving (line sieve with a bit-length prefilter, then exact verification)
// ===========================================================================

fn bits_u64(p: u64) -> u32 {
    64 - p.leading_zeros()
}

#[allow(clippy::too_many_arguments)]
fn collect_relations(
    f: &[Int],
    d: usize,
    m: &Int,
    fb: &FactorBases,
    cols: usize,
    col_rat: usize,
    col_alg: usize,
    col_qc: usize,
    a_max: i64,
    b_max: i64,
    needed: usize,
    budget: u64,
) -> Vec<Relation> {
    let mut relations = Vec::new();
    let mut tried: u64 = 0;

    // prime bit-weights for the prefilter
    let rat_bits: Vec<u32> = fb.rat_primes.iter().map(|&p| bits_u64(p)).collect();
    let max_rat_bits = rat_bits.iter().copied().max().unwrap_or(1);
    let max_alg_bits = fb
        .alg_primes
        .iter()
        .map(|&p| bits_u64(p))
        .max()
        .unwrap_or(1);
    let rslack = 2 * max_rat_bits + 8;
    let aslack = 2 * max_alg_bits + 8;

    let width = (2 * a_max + 1) as usize;

    for b in 1..=b_max {
        if relations.len() >= needed || tried >= budget {
            break;
        }
        let bint = Int::from_i64(b);
        // accumulators over a in [-a_max, a_max]
        let mut racc = vec![0u32; width];
        let mut aacc = vec![0u32; width];
        let bm = m.mul(&bint); // b*m

        // rational sieve: a ≡ b*m (mod p)
        for (&p, &pb) in fb.rat_primes.iter().zip(&rat_bits) {
            let pl = p as i64;
            let start = (bm.rem_euclid(&Int::from_u64(p)).to_u64().unwrap_or(0)) as i64;
            // smallest a >= -a_max with a ≡ start (mod p)
            let mut a = start;
            // shift into range
            a -= ((a + a_max) / pl) * pl;
            while a < -a_max {
                a += pl;
            }
            while a <= a_max {
                let idx = (a + a_max) as usize;
                racc[idx] += pb;
                a += pl;
            }
        }
        // algebraic sieve: a ≡ b*r (mod p) for each (p, r)
        for &(p, r) in &fb.alg_pairs {
            let pl = p as i64;
            let br = (Int::from_u64(r)
                .mul(&bint)
                .rem_euclid(&Int::from_u64(p))
                .to_u64()
                .unwrap_or(0)) as i64;
            let pb = bits_u64(p);
            let mut a = br;
            a -= ((a + a_max) / pl) * pl;
            while a < -a_max {
                a += pl;
            }
            while a <= a_max {
                let idx = (a + a_max) as usize;
                aacc[idx] += pb;
                a += pl;
            }
        }

        for ai in -a_max..=a_max {
            if relations.len() >= needed || tried >= budget {
                break;
            }
            let idx = (ai + a_max) as usize;
            let aint = Int::from_i64(ai);
            // coprimality
            if aint.gcd(&bint) != Int::one() {
                continue;
            }
            tried += 1;
            // rational target: |a - b*m|
            let rv = aint.sub(&bm);
            if rv.is_zero() {
                continue;
            }
            let rtarget = rv.bit_len();
            if racc[idx] + rslack < rtarget {
                continue;
            }
            // algebraic target: |F(a,b)|
            let nv = norm_ab(f, d, &aint, &bint);
            if nv.is_zero() {
                continue;
            }
            let atarget = nv.bit_len();
            if aacc[idx] + aslack < atarget {
                continue;
            }
            if let Some(rel) =
                make_relation(f, d, m, fb, &aint, &bint, cols, col_rat, col_alg, col_qc)
            {
                relations.push(rel);
            }
        }
    }
    relations
}

// ===========================================================================
// Square roots + combine
// ===========================================================================

/// The algebraic square root `δ = √γ ∈ ℤ[θ]` via single-prime Hensel lifting.
/// `gamma` is the exact element `f'(θ)^2 · ∏(a − bθ)` (length `d`). Returns `δ`
/// (length `d`) with `δ·δ == gamma` verified exactly, or `None`.
fn algebraic_sqrt(gamma: &[Int], f: &[Int], d: usize) -> Option<Vec<Int>> {
    // bound on |coeff| of gamma
    let gamma_bits = gamma.iter().map(|c| c.bit_len()).max().unwrap_or(0);
    // Try several inert primes.
    let mut p = Int::from_i64(3);
    let mut attempts = 0;
    while attempts < 200 {
        p = p.next_prime();
        attempts += 1;
        let pu = match p.to_u64() {
            Some(v) if v > 2 => v,
            _ => continue,
        };
        // f must be irreducible mod p (so 𝔽_p[x]/(f) is a field).
        if !is_irreducible_mod_p(f, d, &p) {
            continue;
        }
        // gamma must be nonzero mod p.
        let gp = el_reduce(gamma, d, &p);
        if gp.iter().all(|c| c.is_zero()) {
            continue;
        }
        // finite-field square root
        let beta0 = match ff_sqrt(&gp, f, d, &p) {
            Some(b) => b,
            None => continue, // non-residue: try another prime (or bad dependency)
        };
        // seed inverse s0 = (2*beta0)^{-1} in the field, via Fermat.
        let two = el_const(2, d, Some(&p));
        let twobeta = ff_mul(&two, &beta0, f, d, &p);
        let q = p.pow(d as u32);
        let qm2 = q.sub(&Int::from_i64(2));
        let s0 = ff_pow(&twobeta, &qm2, f, d, &p);

        // Hensel lift target: p^k > 2 * 2^{ceil(gamma_bits/2)} * safety.
        let mut target_bits = gamma_bits / 2 + 32 + 8 * (d as u32);

        for _ in 0..4 {
            if let Some(delta) = hensel_lift(&beta0, &s0, gamma, f, d, &p, pu, target_bits) {
                // verify exactly
                let sq = el_mul(&delta, &delta, f, d, None);
                if sq.len() == gamma.len() && el_eq(&sq, gamma) {
                    return Some(delta);
                }
            }
            target_bits *= 2;
        }
        // If verification failed for this prime, the dependency may not be a
        // square; try another prime a couple of times, then give up.
        if attempts > 8 {
            return None;
        }
    }
    None
}

fn is_irreducible_mod_p(f: &[Int], d: usize, p: &Int) -> bool {
    let coeffs: Vec<ModInt> = f
        .iter()
        .take(d + 1)
        .map(|c| ModInt::new(c.clone(), p.clone()))
        .collect();
    let poly = Poly::new(coeffs);
    if poly.degree() != Some(d) {
        return false;
    }
    poly.is_irreducible()
}

/// Coupled Newton Hensel lift of `(β, s)` (root and inverse-of-2root) to modulus
/// `p^k` with `p^k > 2·2^{target_bits}`, returning the balanced representative.
#[allow(clippy::too_many_arguments)]
fn hensel_lift(
    beta0: &[Int],
    s0: &[Int],
    gamma: &[Int],
    f: &[Int],
    d: usize,
    p: &Int,
    _pu: u64,
    target_bits: u32,
) -> Option<Vec<Int>> {
    let mut modulus = p.clone(); // current p^k
    let mut beta = el_reduce(beta0, d, &modulus);
    let mut s = el_reduce(s0, d, &modulus);
    let two = Int::from_i64(2);

    // lift until modulus has more than target_bits + 2 bits
    while modulus.bit_len() <= target_bits + 2 {
        let newmod = modulus.mul(&modulus); // p^{2k}
        let const2 = el_const(2, d, Some(&newmod));
        let gred = el_reduce(gamma, d, &newmod);
        // β' = β - (β² - γ)·s
        let b2 = el_mul(&beta, &beta, f, d, Some(&newmod));
        let diff = el_sub(&b2, &gred, &newmod);
        let corr = el_mul(&diff, &s, f, d, Some(&newmod));
        let beta_new = el_sub(&beta, &corr, &newmod);
        // s' = s·(2 - 2β'·s)
        let bs: Vec<Int> = beta_new
            .iter()
            .map(|x| x.mul(&two).rem_euclid(&newmod))
            .collect();
        let abs_ = el_mul(&bs, &s, f, d, Some(&newmod));
        let tm = el_sub(&const2, &abs_, &newmod);
        let s_new = el_mul(&s, &tm, f, d, Some(&newmod));
        beta = beta_new;
        s = s_new;
        modulus = newmod;
    }
    // balanced representative
    let half = modulus.div_trunc(&two);
    let delta: Vec<Int> = beta
        .iter()
        .map(|c| {
            if *c > half {
                c.sub(&modulus)
            } else {
                c.clone()
            }
        })
        .collect();
    Some(delta)
}

// ===========================================================================
// Driver
// ===========================================================================

struct Params {
    d: usize,
    br: u64,
    ba: u64,
    num_qc: usize,
    a_max: i64,
    b_max: i64,
    extra: usize,
    budget: u64,
}

fn choose_params(n: &Int) -> Params {
    let l = n.bit_len();
    let d = if l < 100 {
        3
    } else if l < 170 {
        4
    } else {
        5
    };
    // Factor-base bound and sieve extent, by size. Correctness-first: generous
    // and adaptive rather than optimal.
    let (br, a_max, b_max) = if l <= 30 {
        (400u64, 2_000i64, 400i64)
    } else if l <= 45 {
        (1_200, 8_000, 1_200)
    } else if l <= 60 {
        (3_000, 24_000, 2_500)
    } else if l <= 80 {
        (8_000, 60_000, 5_000)
    } else if l <= 110 {
        (40_000, 300_000, 12_000)
    } else if l <= 140 {
        (150_000, 1_500_000, 30_000)
    } else {
        (600_000, 8_000_000, 120_000)
    };
    Params {
        d,
        br,
        ba: br,
        num_qc: 40,
        a_max,
        b_max,
        extra: 32,
        budget: 40_000_000,
    }
}

/// Full GNFS attempt for one parameter set. Returns a nontrivial factor of `n`
/// or `None`.
fn gnfs_attempt(n: &Int, params: &Params) -> Option<Int> {
    let d = params.d;
    let (f, m) = select_poly(n, d)?;

    let fb = build_factor_bases(&f, d, params.br, params.ba, params.num_qc);
    let r = fb.rat_primes.len();
    let ap = fb.alg_pairs.len();
    let qc = fb.qc_pairs.len();
    let col_rat = 2;
    let col_alg = 2 + r;
    let col_qc = 2 + r + ap;
    let cols = 2 + r + ap + qc;
    if cols == 0 {
        return None;
    }
    let needed = cols + params.extra;

    let relations = collect_relations(
        &f,
        d,
        &m,
        &fb,
        cols,
        col_rat,
        col_alg,
        col_qc,
        params.a_max,
        params.b_max,
        needed,
        params.budget,
    );
    if relations.len() < cols + 1 {
        return None;
    }

    // Dependencies via block Lanczos (reused from the quadratic sieve).
    let parity: Vec<Vec<u64>> = relations.iter().map(|r| r.parity.clone()).collect();
    let deps = crate::qsieve::gf2_dependencies(&parity, cols);
    if deps.is_empty() {
        return None;
    }

    let fprime = derivative_coeffs(&f, d);
    let fprime2 = el_mul(&fprime, &fprime, &f, d, None);
    let fpm = eval_int(&fprime, &m); // f'(m)

    for dep in &deps {
        if dep.is_empty() {
            continue;
        }
        if let Some(factor) = try_dependency(n, dep, &relations, &fb, &f, d, &m, &fprime2, &fpm) {
            return Some(factor);
        }
    }
    None
}

fn eval_int(coeffs: &[Int], x: &Int) -> Int {
    let mut acc = Int::zero();
    for c in coeffs.iter().rev() {
        acc = acc.mul(x).add(c);
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn try_dependency(
    n: &Int,
    dep: &[usize],
    relations: &[Relation],
    fb: &FactorBases,
    f: &[Int],
    d: usize,
    m: &Int,
    fprime2: &[Int],
    fpm: &Int,
) -> Option<Int> {
    // Rational square root u = √∏(a-bm) mod N.
    let mut sum = vec![0u64; fb.rat_primes.len()];
    for &ri in dep {
        for &(idx, e) in &relations[ri].rat_exps {
            sum[idx] += e as u64;
        }
    }
    let mut u = Int::one();
    for (idx, &s) in sum.iter().enumerate() {
        if s == 0 {
            continue;
        }
        if s & 1 == 1 {
            return None;
        }
        let base = Int::from_u64(fb.rat_primes[idx]);
        let e = Int::from_u64(s / 2);
        u = u.mul(&base.modpow(&e, n)).rem_euclid(n);
    }

    // γ = f'(θ)² · ∏(a − bθ), exact.
    let mut gamma = fprime2.to_vec();
    for &ri in dep {
        let rel = &relations[ri];
        // (a - bθ) = [a, -b, 0, …]
        let mut factor = vec![Int::zero(); d];
        factor[0] = rel.a.clone();
        if d >= 2 {
            factor[1] = rel.b.neg();
        }
        gamma = el_mul(&gamma, &factor, f, d, None);
    }

    // δ = √γ in ℤ[θ], verified exactly.
    let delta = algebraic_sqrt(&gamma, f, d)?;

    // X = φ(δ) = δ(m) mod N ; Y = f'(m)·u mod N.
    let x = eval_int(&delta, m).rem_euclid(n);
    let y = fpm.mul(&u).rem_euclid(n);

    for cand in [x.sub(&y), x.add(&y)] {
        let g = cand.gcd(n);
        if g > Int::one() && &g < n {
            return Some(g);
        }
    }
    None
}

impl Int {
    /// Factor this integer with the **General Number Field Sieve** (GNFS).
    ///
    /// Returns a pair of nontrivial factors `(p, q)` with `p·q == |self|`, or
    /// `None` if the pipeline does not succeed. The result is always *correct*:
    /// the returned factors are verified to multiply back to `|self|`, and the
    /// method can never return a wrong split (a failed attempt yields `None`).
    ///
    /// This is a clean-room, correctness-first implementation (plain base-`m`
    /// polynomial selection, line sieving, block-Lanczos linear algebra, and a
    /// Hensel-lifted, exactly-verified algebraic square root). It is intended to
    /// demonstrate a *complete* GNFS; it is **not** tuned for speed and does not
    /// implement lattice sieving or advanced polynomial selection.
    ///
    /// # Practical range
    ///
    /// It reliably factors small-to-modest composites — semiprimes up to roughly
    /// the low tens of decimal digits — within reasonable time. For anything the
    /// GNFS is not well suited to (prime inputs, prime powers, or inputs below
    /// the crossover where trial methods are trivial) it simply returns `None`;
    /// callers that want a guaranteed factorization should use
    /// [`Int::factorize`](Int::factorize).
    ///
    /// # Examples
    ///
    /// ```
    /// use puremp::Int;
    /// let n = Int::from(1_022_117); // 1009 × 1013
    /// if let Some((p, q)) = n.factor_gnfs() {
    ///     assert_eq!(p.mul(&q), n);
    ///     assert!(p > Int::from(1) && q > Int::from(1));
    /// }
    /// ```
    pub fn factor_gnfs(&self) -> Option<(Int, Int)> {
        let n = self.abs();
        if n < Int::from_i64(4) {
            return None;
        }
        if n.is_even() {
            let two = Int::from_i64(2);
            return Some((two.clone(), n.div_exact(&two)));
        }
        // Perfect power? nth roots give an easy split.
        if let Some(r) = n.sqrt_exact() {
            return Some((r.clone(), r));
        }
        if n.is_prime_bpsw() {
            return None;
        }

        let params = choose_params(&n);
        let g = gnfs_attempt(&n, &params)?;
        if g > Int::one() && g < n {
            let other = n.div_exact(&g);
            if other.mul(&g) == n {
                return Some((g, other));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert a returned split is a *correct*, nontrivial factorization of `n`,
    /// with both factors prime (balanced semiprime inputs).
    fn assert_correct_split(n: &Int, p: &Int, q: &Int) {
        assert_eq!(p.mul(q), *n, "factors do not multiply to N");
        assert!(*p > Int::one() && *q > Int::one(), "trivial factor");
        assert!(*p < *n && *q < *n, "factor equals N");
        assert!(p.is_prime_bpsw(), "factor {p} not prime");
        assert!(q.is_prime_bpsw(), "factor {q} not prime");
    }

    // ---- stage-level unit checks ----

    #[test]
    fn poly_selection_is_monic_and_evaluates_to_n() {
        let n = Int::from_i64(1_022_117);
        let (f, m) = select_poly(&n, 3).expect("base-m selection");
        assert_eq!(*f.last().unwrap(), Int::one(), "leading coeff must be 1");
        assert_eq!(eval_int(&f, &m), n, "f(m) must equal N");
    }

    #[test]
    fn norm_matches_homogeneous_form() {
        // f = x^3 + 2x^2 + 21x + 17
        let f = [
            Int::from_i64(17),
            Int::from_i64(21),
            Int::from_i64(2),
            Int::from_i64(1),
        ];
        // N(a-bθ) = a^3 + 2a^2 b + 21 a b^2 + 17 b^3
        let a = Int::from_i64(5);
        let b = Int::from_i64(3);
        let got = norm_ab(&f, 3, &a, &b);
        let want = Int::from_i64(125 + 2 * 25 * 3 + 21 * 5 * 9 + 17 * 27);
        assert_eq!(got, want);
    }

    #[test]
    fn finite_field_sqrt_roundtrips() {
        // 𝔽_p[x]/(f), f = x^3 + 2x^2 + 21x + 17, choose an inert prime.
        let f = [
            Int::from_i64(17),
            Int::from_i64(21),
            Int::from_i64(2),
            Int::from_i64(1),
        ];
        let d = 3;
        let mut p = Int::from_i64(3);
        for _ in 0..50 {
            p = p.next_prime();
            if is_irreducible_mod_p(&f, d, &p) {
                // a = (θ+1)^2, then sqrt(a) must square back to a
                let base = {
                    let mut v = vec![Int::zero(); d];
                    v[0] = Int::one();
                    v[1] = Int::one();
                    v
                };
                let a = ff_mul(&base, &base, &f, d, &p);
                let r = ff_sqrt(&a, &f, d, &p).expect("a is a square");
                assert!(el_eq(&ff_mul(&r, &r, &f, d, &p), &a));
                return;
            }
        }
        panic!("no inert prime found");
    }

    // ---- full end-to-end (fast, always run) ----

    #[test]
    fn factors_small_semiprime_end_to_end() {
        // 1009 × 1013 — exercises every stage: poly select → factor bases →
        // sieve → block-Lanczos dependency → rational + algebraic square root →
        // combine.
        let n = Int::from_i64(1_022_117);
        let (p, q) = n.factor_gnfs().expect("GNFS must factor 1009*1013");
        assert_correct_split(&n, &p, &q);
        // cross-check against the general factorizer
        let mut fac: Vec<Int> = n.factorize();
        fac.sort();
        assert_eq!(fac, vec![Int::from_i64(1009), Int::from_i64(1013)]);
    }

    #[test]
    fn never_returns_wrong_split_for_prime() {
        // A prime input must yield None, never a bogus split.
        let prime = Int::from_i64(1_000_003);
        assert!(prime.is_prime_bpsw());
        assert!(prime.factor_gnfs().is_none());
    }

    #[test]
    fn handles_even_and_perfect_square() {
        let even = Int::from_i64(2 * 999_983);
        let (a, b) = even.factor_gnfs().expect("even split");
        assert_eq!(a.mul(&b), even);
        let sq = Int::from_i64(1013).mul(&Int::from_i64(1013));
        let (a, b) = sq.factor_gnfs().expect("square split");
        assert_eq!(a.mul(&b), sq);
    }

    // ---- heavier end-to-end cases (release-only, ignored) ----
    //
    // Run with: cargo test --release --all-features gnfs_ -- --ignored

    fn end_to_end(a: i64, b: i64) {
        let (pa, pb) = (Int::from_i64(a), Int::from_i64(b));
        let n = pa.mul(&pb);
        let (p, q) = n
            .factor_gnfs()
            .unwrap_or_else(|| panic!("GNFS failed on {a}*{b}"));
        assert_correct_split(&n, &p, &q);
        assert!((p == pa && q == pb) || (p == pb && q == pa));
    }

    #[test]
    #[ignore = "slow; run with --release"]
    fn gnfs_11_digit() {
        end_to_end(100_003, 100_019);
    }

    #[test]
    #[ignore = "slow; run with --release"]
    fn gnfs_15_digit() {
        end_to_end(10_000_019, 10_000_079);
    }

    #[test]
    #[ignore = "slow; run with --release"]
    fn gnfs_19_digit() {
        end_to_end(1_000_000_007, 1_000_000_009);
    }
}
