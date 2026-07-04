//! Univariate polynomial factorization over ℚ.
//!
//! [`Poly::factor`](crate::poly::Poly::factor) factors a rational polynomial into
//! irreducible factors over ℚ with multiplicities, by the classical
//! Berlekamp–Zassenhaus pipeline: square-free decomposition (Yun), factorization
//! modulo a well-chosen prime (deterministic Cantor–Zassenhaus), Hensel lifting
//! to a modulus exceeding the Mignotte coefficient bound, and recombination of
//! the lifted factors into true integer factors.
//!
//! All modular work uses machine-word `𝔽ₚ` arithmetic; the recombination and the
//! final results are exact `Int`/`Rational`. References: Knuth, *TAOCP* Vol. 2
//! §4.6.2; Brent & Zimmermann, *Modern Computer Arithmetic* §3.

use alloc::vec;
use alloc::vec::Vec;

use crate::int::Int;

// ---------------------------------------------------------------------------
// 𝔽ₚ polynomial arithmetic (coefficients are `u64 < p`, low-to-high, trimmed).
// ---------------------------------------------------------------------------

#[inline]
fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

#[inline]
fn addmod(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b;
    if s >= p { s - p } else { s }
}

#[inline]
fn submod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { a + p - b }
}

/// `a^e mod p` (fast exponentiation).
fn powmod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64 % p;
    a %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, p);
        }
        a = mulmod(a, a, p);
        e >>= 1;
    }
    r
}

/// Modular inverse of `a` (nonzero) mod prime `p`, by Fermat.
fn invmod(a: u64, p: u64) -> u64 {
    powmod(a, p - 2, p)
}

/// Drops trailing (high-degree) zero coefficients.
fn trim(mut a: Vec<u64>) -> Vec<u64> {
    while a.last() == Some(&0) {
        a.pop();
    }
    a
}

fn deg(a: &[u64]) -> isize {
    a.len() as isize - 1
}

fn fp_sub(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut r = vec![0u64; n];
    for (i, slot) in r.iter_mut().enumerate() {
        let x = *a.get(i).unwrap_or(&0);
        let y = *b.get(i).unwrap_or(&0);
        *slot = submod(x, y, p);
    }
    trim(r)
}

fn fp_mul(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = vec![0u64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            r[i + j] = addmod(r[i + j], mulmod(ai, bj, p), p);
        }
    }
    trim(r)
}

fn fp_scale(a: &[u64], s: u64, p: u64) -> Vec<u64> {
    trim(a.iter().map(|&x| mulmod(x, s, p)).collect())
}

/// Multiplies a monic-normalizing scalar so the leading coefficient becomes 1.
fn fp_monic(a: &[u64], p: u64) -> Vec<u64> {
    match a.last() {
        None => Vec::new(),
        Some(&lc) => {
            if lc == 1 {
                a.to_vec()
            } else {
                fp_scale(a, invmod(lc, p), p)
            }
        }
    }
}

/// Polynomial division: returns `(quotient, remainder)` with `a = q·b + r`,
/// `deg r < deg b`. `b` must be nonzero.
fn fp_divmod(a: &[u64], b: &[u64], p: u64) -> (Vec<u64>, Vec<u64>) {
    let mut r = a.to_vec();
    let db = deg(b);
    if deg(&r) < db {
        return (Vec::new(), trim(r));
    }
    let inv_lc = invmod(*b.last().unwrap(), p);
    let mut q = vec![0u64; (deg(&r) - db + 1) as usize];
    while deg(&r) >= db && !r.is_empty() {
        let d = (deg(&r) - db) as usize;
        let coef = mulmod(*r.last().unwrap(), inv_lc, p);
        q[d] = coef;
        // r -= coef · x^d · b
        for (j, &bj) in b.iter().enumerate() {
            r[d + j] = submod(r[d + j], mulmod(coef, bj, p), p);
        }
        r = trim(r);
    }
    (trim(q), r)
}

fn fp_rem(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    fp_divmod(a, b, p).1
}

fn fp_gcd(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let mut a = trim(a.to_vec());
    let mut b = trim(b.to_vec());
    while !b.is_empty() {
        let r = fp_rem(&a, &b, p);
        a = b;
        b = r;
    }
    fp_monic(&a, p)
}

/// `base^e mod modulus` in `𝔽ₚ[x]`.
fn fp_powmod(base: &[u64], mut e: u64, modulus: &[u64], p: u64) -> Vec<u64> {
    let mut r = vec![1u64 % p];
    let mut b = fp_rem(base, modulus, p);
    while e > 0 {
        if e & 1 == 1 {
            r = fp_rem(&fp_mul(&r, &b, p), modulus, p);
        }
        b = fp_rem(&fp_mul(&b, &b, p), modulus, p);
        e >>= 1;
    }
    trim(r)
}

fn fp_derivative(a: &[u64], p: u64) -> Vec<u64> {
    if a.len() <= 1 {
        return Vec::new();
    }
    let mut r = vec![0u64; a.len() - 1];
    for i in 1..a.len() {
        r[i - 1] = mulmod(a[i], (i as u64) % p, p);
    }
    trim(r)
}

// ---------------------------------------------------------------------------
// Cantor–Zassenhaus: factor a square-free monic `f` in 𝔽ₚ[x] into irreducibles.
// ---------------------------------------------------------------------------

/// Distinct-degree factorization: splits square-free monic `f` into `(d, g_d)`
/// where `g_d` is the product of all monic irreducible factors of degree `d`.
fn distinct_degree(f: &[u64], p: u64) -> Vec<(usize, Vec<u64>)> {
    let mut out = Vec::new();
    let mut fstar = f.to_vec();
    let mut d = 1usize;
    // x^(p^d) mod f, maintained by repeated p-power Frobenius.
    let mut xp = fp_powmod(&[0, 1], p, &fstar, p); // x^p mod f
    while deg(&fstar) >= 2 * d as isize {
        // gcd(f*, x^(p^d) - x)
        let g = fp_gcd(&fstar, &fp_sub(&xp, &[0, 1], p), p);
        if deg(&g) > 0 {
            out.push((d, g.clone()));
            fstar = fp_divmod(&fstar, &g, p).0;
        }
        d += 1;
        if deg(&fstar) >= 2 * d as isize {
            xp = fp_powmod(&xp, p, &fstar, p); // (x^(p^{d-1}))^p = x^(p^d)
        }
    }
    if deg(&fstar) > 0 {
        out.push((deg(&fstar) as usize, fstar));
    }
    out
}

/// Equal-degree factorization: `f` is a product of monic irreducibles each of
/// degree `d`; split it fully. Deterministic — tries a reproducible sequence of
/// splitting polynomials seeded from `f`.
fn equal_degree(f: &[u64], d: usize, p: u64) -> Vec<Vec<u64>> {
    let r = (deg(f) / d as isize) as usize; // number of irreducible factors
    if r == 1 {
        return vec![fp_monic(f, p)];
    }
    let mut factors = vec![fp_monic(f, p)];
    // Deterministic pseudo-random splitting polynomials via an LCG seeded from f.
    let mut seed = f.iter().fold(0x9E37_79B9_7F4A_7C15u64, |s, &c| {
        s.wrapping_mul(6364136223846793005).wrapping_add(c | 1)
    });
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };
    let pd_half = pd_minus_one_over_two(p, d); // (p^d − 1)/2

    while factors.len() < r {
        // Build a random polynomial a of degree < deg(f).
        let n = f.len() - 1;
        let a: Vec<u64> = (0..n).map(|_| next() % p).collect();
        let a = trim(a);
        if a.is_empty() {
            continue;
        }
        // b = a^((p^d - 1)/2) - 1 mod f
        let b = fp_sub(&fp_powmod(&a, pd_half, f, p), &[1], p);
        let mut new_factors = Vec::with_capacity(factors.len());
        for h in factors.into_iter() {
            if deg(&h) == d as isize {
                new_factors.push(h);
                continue;
            }
            let g = fp_gcd(&h, &b, p);
            if deg(&g) > 0 && deg(&g) < deg(&h) {
                let other = fp_divmod(&h, &g, p).0;
                new_factors.push(fp_monic(&g, p));
                new_factors.push(fp_monic(&other, p));
            } else {
                new_factors.push(h);
            }
        }
        factors = new_factors;
    }
    factors
}

/// Exact `(p^d - 1)/2` as a `u64`, for the modest `p^d` that arises here.
fn pd_minus_one_over_two(p: u64, d: usize) -> u64 {
    let mut pd = 1u128;
    for _ in 0..d {
        pd *= p as u128;
    }
    ((pd - 1) / 2) as u64
}

/// Full factorization of a square-free monic `f` in 𝔽ₚ[x] into monic irreducibles.
fn factor_mod_p(f: &[u64], p: u64) -> Vec<Vec<u64>> {
    let mut out = Vec::new();
    for (d, g) in distinct_degree(f, p) {
        out.extend(equal_degree(&g, d, p));
    }
    out
}

/// Extended Euclid in 𝔽ₚ[x]: returns `(s, t)` with `s·a + t·b ≡ gcd ≡ 1` when
/// `a, b` are coprime (the only case used here).
fn fp_bezout(a: &[u64], b: &[u64], p: u64) -> (Vec<u64>, Vec<u64>) {
    let (mut r0, mut r1) = (a.to_vec(), b.to_vec());
    let (mut s0, mut s1) = (vec![1u64 % p], Vec::new());
    let (mut t0, mut t1) = (Vec::new(), vec![1u64 % p]);
    while !r1.is_empty() {
        let (q, r) = fp_divmod(&r0, &r1, p);
        r0 = r1;
        r1 = r;
        let s = fp_sub(&s0, &fp_mul(&q, &s1, p), p);
        s0 = s1;
        s1 = s;
        let t = fp_sub(&t0, &fp_mul(&q, &t1, p), p);
        t0 = t1;
        t1 = t;
    }
    // Normalize so gcd is 1 (r0 is a nonzero constant here).
    let inv = invmod(*r0.last().unwrap(), p);
    (fp_scale(&s0, inv, p), fp_scale(&t0, inv, p))
}

// ---------------------------------------------------------------------------
// ℤ[x] arithmetic (coefficients `Int`, low-to-high, trimmed).
// ---------------------------------------------------------------------------

fn ip_trim(mut a: Vec<Int>) -> Vec<Int> {
    while a.last().is_some_and(Int::is_zero) {
        a.pop();
    }
    a
}

fn ip_mul(a: &[Int], b: &[Int]) -> Vec<Int> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = vec![Int::ZERO; a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            r[i + j] = r[i + j].add(&ai.mul(bj));
        }
    }
    ip_trim(r)
}

fn ip_sub(a: &[Int], b: &[Int]) -> Vec<Int> {
    let n = a.len().max(b.len());
    let mut r = vec![Int::ZERO; n];
    for (i, slot) in r.iter_mut().enumerate() {
        let x = a.get(i).cloned().unwrap_or(Int::ZERO);
        let y = b.get(i).cloned().unwrap_or(Int::ZERO);
        *slot = x.sub(&y);
    }
    ip_trim(r)
}

/// Polynomial division by a **monic** divisor `b` over ℤ (exact integer arithmetic).
fn ip_divmod_monic(a: &[Int], b: &[Int]) -> (Vec<Int>, Vec<Int>) {
    debug_assert!(
        b.last().map(|c| c == &Int::ONE).unwrap_or(false),
        "divisor must be monic"
    );
    let db = b.len() as isize - 1;
    let mut r = a.to_vec();
    if (r.len() as isize - 1) < db {
        return (Vec::new(), ip_trim(r));
    }
    let mut q = vec![Int::ZERO; (r.len() as isize - db) as usize];
    while (r.len() as isize - 1) >= db && !r.is_empty() {
        let d = (r.len() as isize - 1 - db) as usize;
        let coef = r.last().unwrap().clone();
        for (j, bj) in b.iter().enumerate() {
            r[d + j] = r[d + j].sub(&coef.mul(bj));
        }
        q[d] = coef;
        r = ip_trim(r);
    }
    (ip_trim(q), ip_trim(r))
}

/// Reduces every coefficient of `a` into the symmetric range `(-m/2, m/2]`.
fn ip_sym_mod(a: &[Int], m: &Int) -> Vec<Int> {
    let half = m.div_floor(&Int::from_i64(2));
    ip_trim(
        a.iter()
            .map(|c| {
                let r = c.rem_euclid(m);
                if r.cmp(&half) == core::cmp::Ordering::Greater {
                    r.sub(m)
                } else {
                    r
                }
            })
            .collect(),
    )
}

fn fp_to_ip(a: &[u64]) -> Vec<Int> {
    a.iter().map(|&c| Int::from_u64(c)).collect()
}

fn ip_content(a: &[Int]) -> Int {
    a.iter().fold(Int::ZERO, |g, c| g.gcd(c))
}

fn ip_primitive(a: &[Int]) -> Vec<Int> {
    let c = ip_content(a);
    if c.is_zero() || c == Int::ONE {
        return a.to_vec();
    }
    a.iter().map(|x| x.div_exact(&c)).collect()
}

// ---------------------------------------------------------------------------
// Hensel lifting and recombination (monic case).
// ---------------------------------------------------------------------------

/// Linear Hensel lift of a coprime monic pair. Given monic `g0·h0 ≡ f (mod p)`
/// with `g0, h0` coprime in 𝔽ₚ, returns monic `(G, H)` over ℤ with `f ≡ G·H
/// (mod p^exp)`, `G ≡ g0`, `H ≡ h0 (mod p)`. Lifts one power at a time using the
/// fixed mod-`p` Bézout coefficients (`s·g0 + t·h0 ≡ 1`).
fn hensel_lift_two(f: &[Int], g0: &[u64], h0: &[u64], p: u64, exp: u32) -> (Vec<Int>, Vec<Int>) {
    let (s, _t) = fp_bezout(g0, h0, p);
    let mut g = fp_to_ip(g0);
    let mut h = fp_to_ip(h0);
    let mut pk = Int::from_u64(p);
    for _ in 1..exp {
        // e = f − g·h ≡ 0 (mod pk); ē = (e / pk) reduced mod p.
        let e = ip_sub(f, &ip_mul(&g, &h));
        let ebar_fp: Vec<u64> = trim(
            e.iter()
                .map(|c| {
                    let q = c.div_exact(&pk); // exact: e ≡ 0 (mod pk)
                    q.rem_euclid(&Int::from_u64(p)).to_u64().unwrap_or(0)
                })
                .collect(),
        );
        // Solve g0·u + h0·v = ē in 𝔽ₚ with deg u < deg h0, deg v < deg g0.
        let u = fp_rem(&fp_mul(&s, &ebar_fp, p), h0, p);
        let v = fp_divmod(&fp_sub(&ebar_fp, &fp_mul(g0, &u, p), p), h0, p).0;
        // g += pk·v ; h += pk·u  (corrections lift the factors by one power).
        g = ip_add_scaled(&g, &v, &pk);
        h = ip_add_scaled(&h, &u, &pk);
        pk = pk.mul(&Int::from_u64(p));
    }
    (g, h)
}

/// `a + scale·(b as ℤ-poly)`.
fn ip_add_scaled(a: &[Int], b: &[u64], scale: &Int) -> Vec<Int> {
    let n = a.len().max(b.len());
    let mut r = vec![Int::ZERO; n];
    for (i, slot) in r.iter_mut().enumerate() {
        let x = a.get(i).cloned().unwrap_or(Int::ZERO);
        let add = b
            .get(i)
            .map(|&c| Int::from_u64(c).mul(scale))
            .unwrap_or(Int::ZERO);
        *slot = x.add(&add);
    }
    ip_trim(r)
}

/// Lifts the full mod-`p` factorization of monic `f` to monic factors mod
/// `p^exp`, by a recursive subproduct split.
fn lift_all(f: &[Int], modfacs: &[Vec<u64>], p: u64, exp: u32, m: &Int) -> Vec<Vec<Int>> {
    if modfacs.len() == 1 {
        return vec![ip_sym_mod(f, m)];
    }
    let mid = modfacs.len() / 2;
    let gl = modfacs[..mid]
        .iter()
        .fold(vec![1u64 % p], |acc, g| fp_mul(&acc, g, p));
    let hr = modfacs[mid..]
        .iter()
        .fold(vec![1u64 % p], |acc, g| fp_mul(&acc, g, p));
    let (g, h) = hensel_lift_two(f, &gl, &hr, p, exp);
    let mut out = lift_all(&ip_sym_mod(&g, m), &modfacs[..mid], p, exp, m);
    out.extend(lift_all(&ip_sym_mod(&h, m), &modfacs[mid..], p, exp, m));
    out
}

/// Recombines lifted monic factors (mod `m`) into the true monic integer factors
/// of monic `f` (whose content is 1). Trial recombination over subsets of
/// increasing size — exponential in the number of modular factors in the worst
/// case, but correct.
fn recombine(f: &[Int], lifted: &[Vec<Int>], m: &Int) -> Vec<Vec<Int>> {
    let mut remaining = f.to_vec();
    let mut pool: Vec<Vec<Int>> = lifted.to_vec();
    let mut out = Vec::new();
    let mut size = 1;
    while size <= pool.len() {
        let mut found = None;
        for combo in subsets(pool.len(), size) {
            // Candidate = symmetric residue of the subset product (monic).
            let mut cand = vec![Int::ONE];
            for &i in &combo {
                cand = ip_sym_mod(&ip_mul(&cand, &pool[i]), m);
            }
            if (remaining.len() as isize - 1) < (cand.len() as isize - 1) {
                continue;
            }
            let (q, r) = ip_divmod_monic(&remaining, &cand);
            if r.is_empty() {
                out.push(ip_primitive(&cand));
                remaining = q;
                found = Some(combo);
                break;
            }
        }
        match found {
            Some(combo) => {
                // Remove the used factors (high indices first) and restart at size 1.
                let mut idx = combo;
                idx.sort_unstable_by(|a, b| b.cmp(a));
                for i in idx {
                    pool.remove(i);
                }
                size = 1;
            }
            None => size += 1,
        }
    }
    if (remaining.len() as isize - 1) > 0 {
        out.push(ip_primitive(&remaining));
    }
    out
}

// ---------------------------------------------------------------------------
// Drivers.
// ---------------------------------------------------------------------------

/// Reduces an ℤ-polynomial mod `p` into `𝔽ₚ` (coefficients in `[0, p)`).
fn reduce_mod_p(f: &[Int], p: u64) -> Vec<u64> {
    let pp = Int::from_u64(p);
    trim(
        f.iter()
            .map(|c| c.rem_euclid(&pp).to_u64().unwrap_or(0))
            .collect(),
    )
}

/// Factors a **monic square-free** `f ∈ ℤ[x]` (degree ≥ 1) into monic irreducible
/// integer factors, by Berlekamp–Zassenhaus.
fn factor_monic_squarefree(f: &[Int]) -> Vec<Vec<Int>> {
    let n = f.len() - 1;
    if n <= 1 {
        return alloc::vec![f.to_vec()];
    }
    // Pick a prime keeping f square-free mod p (f is monic, so p never divides lc).
    const PRIMES: [u64; 24] = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    ];
    let (p, fp) = PRIMES
        .iter()
        .find_map(|&p| {
            let fp = reduce_mod_p(f, p);
            if deg(&fp) != n as isize {
                return None;
            }
            (deg(&fp_gcd(&fp, &fp_derivative(&fp, p), p)) == 0).then_some((p, fp))
        })
        .expect("a small prime keeps a square-free polynomial square-free mod p");

    let modfacs = factor_mod_p(&fp, p);
    if modfacs.len() == 1 {
        return alloc::vec![f.to_vec()]; // irreducible over ℤ
    }
    // Lift to a modulus exceeding twice the Mignotte bound 2^n·‖f‖₂ ≤ 2^n·‖f‖₁.
    let norm1 = f.iter().fold(Int::ZERO, |a, c| a.add(&c.abs()));
    let bound = Int::ONE.mul_2k(n as u32 + 1).mul(&norm1);
    let mut m = Int::from_u64(p);
    let mut exp = 1u32;
    while m.cmp(&bound) != core::cmp::Ordering::Greater {
        m = m.mul(&Int::from_u64(p));
        exp += 1;
    }
    let lifted = lift_all(f, &modfacs, p, exp, &m);
    recombine(f, &lifted, &m)
}

/// Factors a **primitive square-free** `f ∈ ℤ[x]` (possibly non-monic) into
/// primitive irreducible integer factors, via the monic-associate substitution
/// `F(x) = l^{n-1}·f(x/l)` (`l = lc f`), factoring the monic `F`, then mapping
/// each factor `G` back as `primitive_part(G(l·x))`.
fn factor_primitive_squarefree(f: &[Int]) -> Vec<Vec<Int>> {
    let n = f.len() - 1;
    if n <= 1 {
        return alloc::vec![f.to_vec()];
    }
    let l = f.last().unwrap().clone();
    if l == Int::ONE {
        return factor_monic_squarefree(f);
    }
    if l == Int::from_i64(-1) {
        let neg: Vec<Int> = f.iter().map(Int::neg).collect();
        return factor_monic_squarefree(&neg);
    }
    // Monic associate F: F_k = f_k·l^{n-1-k} for k < n, F_n = 1.
    let mut bigf = alloc::vec![Int::ZERO; n + 1];
    bigf[n] = Int::ONE;
    for (k, slot) in bigf.iter_mut().enumerate().take(n) {
        *slot = f[k].mul(&l.pow((n - 1 - k) as u32));
    }
    factor_monic_squarefree(&bigf)
        .iter()
        .map(|g| {
            // g(l·x) then primitive part.
            let sub: Vec<Int> = g
                .iter()
                .enumerate()
                .map(|(i, c)| c.mul(&l.pow(i as u32)))
                .collect();
            ip_primitive(&ip_trim(sub))
        })
        .collect()
}

/// All size-`k` subsets of `0..n` (indices), as index vectors.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut combo: Vec<usize> = (0..k).collect();
    if k == 0 || k > n {
        return out;
    }
    loop {
        out.push(combo.clone());
        // Advance to the next combination in lexicographic order.
        let mut i = k;
        while i > 0 {
            i -= 1;
            if combo[i] != i + n - k {
                combo[i] += 1;
                for j in i + 1..k {
                    combo[j] = combo[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return out;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Top level: square-free decomposition (Yun) + Poly<Rational> entry point.
// ---------------------------------------------------------------------------

use crate::poly::Poly;
use crate::rational::Rational;

fn is_constant(p: &Poly<Rational>) -> bool {
    p.degree().is_none_or(|d| d == 0)
}

/// Yun's square-free decomposition of a monic `f`: returns `(sᵢ, i)` with
/// `f = ∏ sᵢ^i`, each `sᵢ` monic and square-free.
fn yun(f: &Poly<Rational>) -> alloc::vec::Vec<(Poly<Rational>, usize)> {
    let fp = f.derivative();
    let a0 = f.gcd(&fp);
    let mut b = f.div_rem(&a0).0;
    let mut c = fp.div_rem(&a0).0;
    let mut d = c.sub(&b.derivative());
    let mut out = alloc::vec::Vec::new();
    let mut i = 1;
    while !is_constant(&b) {
        let a = b.gcd(&d);
        if !is_constant(&a) {
            out.push((a.monic(), i));
        }
        let b_next = b.div_rem(&a).0;
        c = d.div_rem(&a).0;
        d = c.sub(&b_next.derivative());
        b = b_next;
        i += 1;
    }
    out
}

/// Converts a rational polynomial to its primitive integer form (clear
/// denominators, then divide out the integer content).
fn to_primitive_int(p: &Poly<Rational>) -> Vec<Int> {
    let mut lcm = Int::ONE;
    for c in p.coeffs() {
        lcm = lcm.lcm(c.denominator());
    }
    let ints: Vec<Int> = p
        .coeffs()
        .iter()
        .map(|c| c.numerator().mul(&lcm.div_exact(c.denominator())))
        .collect();
    ip_primitive(&ip_trim(ints))
}

/// Converts an integer polynomial to a monic rational polynomial (divide by lc).
fn to_monic_rat(g: &[Int]) -> Poly<Rational> {
    let lc = g.last().unwrap();
    Poly::new(
        g.iter()
            .map(|c| Rational::new(c.clone(), lc.clone()))
            .collect(),
    )
}

/// Factors a rational polynomial into monic irreducible factors over ℚ with
/// multiplicities. Constants (and the zero polynomial) yield an empty list.
pub(crate) fn factor_rational(f: &Poly<Rational>) -> alloc::vec::Vec<(Poly<Rational>, usize)> {
    if is_constant(f) {
        return alloc::vec::Vec::new();
    }
    let mut out = alloc::vec::Vec::new();
    for (sqfree, mult) in yun(&f.monic()) {
        for g in factor_primitive_squarefree(&to_primitive_int(&sqfree)) {
            out.push((to_monic_rat(&g), mult));
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;

    fn prod(fs: &[Vec<u64>], p: u64) -> Vec<u64> {
        fs.iter().fold(vec![1u64], |acc, g| fp_mul(&acc, g, p))
    }

    #[test]
    fn cantor_zassenhaus_reconstructs() {
        let p = 13;
        // (x+1)(x+2)(x²+2): two linear + one irreducible quadratic (−2 is a
        // non-residue mod 13).
        let f = fp_mul(&fp_mul(&[1, 1], &[2, 1], p), &[2, 0, 1], p);
        let fm = fp_monic(&f, p);
        let facs = factor_mod_p(&fm, p);
        assert_eq!(fp_monic(&prod(&facs, p), p), fm, "product mismatch");
        assert_eq!(facs.len(), 3, "expected 3 irreducible factors: {facs:?}");
        assert_eq!(facs.iter().filter(|g| deg(g) == 2).count(), 1);
    }

    #[test]
    fn fully_split_linear() {
        let p = 13;
        // (x−1)(x−2)(x−3)(x−4)(x−5): five distinct roots.
        let mut f = vec![1u64];
        for r in 1..=5u64 {
            f = fp_mul(&f, &[p - r, 1], p);
        }
        let facs = factor_mod_p(&fp_monic(&f, p), p);
        assert_eq!(facs.len(), 5);
        assert!(facs.iter().all(|g| deg(g) == 1));
        assert_eq!(fp_monic(&prod(&facs, p), p), fp_monic(&f, p));
    }
}
