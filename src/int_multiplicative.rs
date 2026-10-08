//! Multiplicative-group number theory on [`Int`]: the structure of the unit
//! group `(ℤ/nℤ)*` and the inverse of Euler's totient.
//!
//! - [`Int::multiplicative_order`] — Mathematica's `MultiplicativeOrder[k, n]`.
//! - [`Int::carmichael_lambda`] — `CarmichaelLambda[n]`, the exponent of
//!   `(ℤ/nℤ)*`.
//! - [`Int::primitive_root`] / [`Int::primitive_root_list`] — `PrimitiveRoot[n]`
//!   and `PrimitiveRootList[n]`.
//! - [`Int::inverse_totient`] / [`Int::is_totient`] — every `n` with
//!   `φ(n) = m`, and whether one exists.
//!
//! All of these take the modulus up to sign (`|n|`), like [`Int::euler_phi`].

use alloc::vec;
use alloc::vec::Vec;

use crate::int::Int;

impl Int {
    /// Multiplicative order of `self` modulo `n` — Mathematica's
    /// `MultiplicativeOrder[k, n]`: the least `m ≥ 1` with `selfᵐ ≡ 1 (mod |n|)`.
    ///
    /// Returns `None` when no such `m` exists, i.e. when `gcd(self, n) ≠ 1`.
    /// Conventions: the sign of `n` is ignored and `self` may be negative (it is
    /// reduced mod `|n|`); for `|n| = 1` every `k` has order `1`; for `n = 0`
    /// the congruence is equality in `ℤ`, so only `1` (order `1`) and `−1`
    /// (order `2`) have an order.
    ///
    /// Algorithm: the order divides the Carmichael function `λ(|n|)`, so start
    /// from `λ`, factor it, and for each prime `q | λ` strip factors of `q`
    /// while `self^{order/q} ≡ 1` — `O(Σ eᵢ)` modular exponentiations rather
    /// than a linear search.
    pub fn multiplicative_order(&self, n: &Int) -> Option<Int> {
        let m = n.abs();
        if m.is_zero() {
            return if self.is_one() {
                Some(Int::ONE)
            } else if self.is_minus_one() {
                Some(Int::from(2))
            } else {
                None
            };
        }
        if m.is_one() {
            return Some(Int::ONE);
        }
        if !self.gcd(&m).is_one() {
            return None;
        }
        let mut order = m.carmichael_lambda();
        for (q, e) in order.factor_exponents() {
            for _ in 0..e {
                let cand = order.div_exact(&q);
                if self.modpow(&cand, &m).is_one() {
                    order = cand;
                } else {
                    break;
                }
            }
        }
        Some(order)
    }

    /// Carmichael's function `λ(|self|)` — Mathematica's `CarmichaelLambda[n]`:
    /// the exponent of `(ℤ/nℤ)*`, i.e. the least `m ≥ 1` with `aᵐ ≡ 1 (mod n)`
    /// for every `a` coprime to `n`. `λ(1) = 1`; `λ(0) = 0` by convention.
    ///
    /// Computed as `lcm` over the prime powers `pᵉ ∥ n` of `λ(pᵉ)`, where
    /// `λ(pᵉ) = pᵉ⁻¹(p−1)` for odd `p`, `λ(2) = 1`, `λ(4) = 2` and
    /// `λ(2ᵉ) = 2ᵉ⁻²` for `e ≥ 3`.
    pub fn carmichael_lambda(&self) -> Int {
        if self.is_zero() {
            return Int::ZERO;
        }
        let mut lambda = Int::ONE;
        for (p, e) in self.factor_exponents() {
            let l = if p == Int::from(2) {
                match e {
                    1 => Int::ONE,
                    2 => Int::from(2),
                    _ => Int::ONE.mul_2k(e - 2),
                }
            } else {
                p.pow(e - 1).mul(&p.sub(&Int::ONE))
            };
            lambda = lambda.lcm(&l);
        }
        lambda
    }

    /// The smallest primitive root modulo `|self|` — Mathematica's
    /// `PrimitiveRoot[n]`: a generator of the cyclic group `(ℤ/nℤ)*`.
    ///
    /// A primitive root exists iff `n ∈ {1, 2, 4, pᵏ, 2pᵏ}` (`p` an odd prime);
    /// otherwise (and for `n = 0`) this returns `None`. For `n = 1` the single
    /// residue `0` generates the trivial group, so the result is `Some(0)`.
    ///
    /// Algorithm: with `φ = φ(n)` factored, candidates `g = 2, 3, …` coprime to
    /// `n` are tested by `g^{φ/q} ≢ 1 (mod n)` for every prime `q | φ`. The least
    /// primitive root is small in practice (`O(log⁶ p)` under GRH, Shoup), so
    /// this is fast even for large prime moduli.
    pub fn primitive_root(&self) -> Option<Int> {
        least_primitive_root(&self.abs()).map(|(g, _)| g)
    }

    /// All primitive roots modulo `|self|`, sorted ascending — Mathematica's
    /// `PrimitiveRootList[n]`. Empty when none exist (see
    /// [`Int::primitive_root`]); `[0]` for `n = 1`.
    ///
    /// There are `φ(φ(n))` of them, derived from the least root `g` as
    /// `gᵏ mod n` for `1 ≤ k ≤ φ(n)` with `gcd(k, φ(n)) = 1`. The walk takes
    /// `φ(n)` modular multiplications, so this is only sensible for small `n`.
    pub fn primitive_root_list(&self) -> Vec<Int> {
        let n = self.abs();
        let Some((g, phi)) = least_primitive_root(&n) else {
            return Vec::new();
        };
        if n.is_one() || n == Int::from(2) {
            return vec![g];
        }
        let mut out = Vec::new();
        let mut k = Int::ONE;
        let mut x = g.clone();
        while k <= phi {
            if k.gcd(&phi).is_one() {
                out.push(x.clone());
            }
            x = x.mul(&g).rem_euclid(&n);
            k = k.add(&Int::ONE);
        }
        out.sort();
        out
    }

    /// Every `n ≥ 1` with Euler totient `φ(n) = self`, sorted ascending (the
    /// inverse of [`Int::euler_phi`]; empty when `self` is not a totient, which
    /// includes every `self ≤ 0` and every odd `self > 1`). `φ⁻¹(1) = [1, 2]`.
    ///
    /// Algorithm (dynamic programming over the divisors of `m`, after
    /// Alekseyev, *Computing the inverses, their power sums, and extrema for
    /// Euler's totient and other multiplicative functions*, 2016): any solution
    /// is a product of prime powers `pᵏ` with `φ(pᵏ) = pᵏ⁻¹(p−1) | m`, so only
    /// the primes `p = d + 1` for divisors `d | m` can occur. Divisors are
    /// represented by their exponent vectors over the factorization of `m`;
    /// with the candidate primes sorted `p₀ < p₁ < …`, a forward pass records
    /// which divisors are reachable as `φ` of a product of the first `j`
    /// primes. Enumeration then walks primes in descending order and only
    /// descends into reachable states, so no work is spent on dead ends and the
    /// cost is linear in the output (times the recursion depth). Practical for
    /// highly composite `m` around `10¹²` and beyond.
    pub fn inverse_totient(&self) -> Vec<Int> {
        let Some(t) = TotientTable::new(self) else {
            return Vec::new();
        };
        let full = t.cands.len();
        let top = t.div_count - 1; // index of m itself
        if !t.reachable(full, top) {
            return Vec::new();
        }
        let mut out = Vec::new();
        t.enumerate(top, full, &Int::ONE, &mut out);
        out.sort();
        out
    }

    /// Whether `self` is a value of Euler's totient, i.e. whether
    /// [`Int::inverse_totient`] is non-empty — without enumerating the
    /// preimages. `false` for `self ≤ 0` and odd `self > 1`.
    pub fn is_totient(&self) -> bool {
        if !self.is_positive() {
            return false;
        }
        if self.is_one() {
            return true;
        }
        if self.is_odd() {
            return false;
        }
        match TotientTable::new(self) {
            Some(t) => t.reachable(t.cands.len(), t.div_count - 1),
            None => false,
        }
    }
}

/// For `n ≥ 0`, the least primitive root modulo `n` together with `φ(n)`, or
/// `None` when `n` has no primitive root (`n ∉ {1, 2, 4, pᵏ, 2pᵏ}`). `n` is
/// factored once; `φ(n)`'s primes come from that factorization and `p − 1`.
fn least_primitive_root(n: &Int) -> Option<(Int, Int)> {
    let two = Int::from(2);
    if n.is_zero() {
        return None;
    }
    if n.is_one() {
        return Some((Int::ZERO, Int::ONE));
    }
    if *n == two {
        return Some((Int::ONE, Int::ONE));
    }
    if *n == Int::from(4) {
        return Some((Int::from(3), two));
    }
    let fe = n.factor_exponents();
    let (p, e) = match fe.as_slice() {
        [(p, e)] if *p != two => (p.clone(), *e),
        [(t, 1), (p, e)] if *t == two => (p.clone(), *e),
        _ => return None,
    };
    // φ(pᵉ) = φ(2pᵉ) = pᵉ⁻¹·(p−1): its primes are those of p−1, plus p if e ≥ 2.
    let pm1 = p.sub(&Int::ONE);
    let phi = p.pow(e - 1).mul(&pm1);
    let mut qs: Vec<Int> = pm1.factor_exponents().into_iter().map(|(q, _)| q).collect();
    if e >= 2 {
        qs.push(p);
    }
    let exps: Vec<Int> = qs.iter().map(|q| phi.div_exact(q)).collect();
    let mut g = two;
    loop {
        if g.gcd(n).is_one() && exps.iter().all(|x| !g.modpow(x, n).is_one()) {
            return Some((g, phi));
        }
        g = g.add(&Int::ONE);
    }
}

/// A candidate prime for [`Int::inverse_totient`] with the exponent-vector data
/// of each admissible power `pᵏ` (those with `φ(pᵏ) | m`).
struct TotientCand {
    p: Int,
    /// `(k, divisor-index offset of φ(pᵏ), exponent vector of φ(pᵏ))`.
    powers: Vec<(u32, usize, Vec<u32>)>,
}

/// Divisor lattice of `m` plus the reachability table driving
/// [`Int::inverse_totient`] / [`Int::is_totient`].
struct TotientTable {
    /// Distinct primes of `m` with their exponents.
    primes: Vec<(Int, u32)>,
    /// Number of divisors; divisor `∏ qₜ^{aₜ}` has the mixed-radix index
    /// `Σ aₜ·∏_{s<t}(eₛ+1)`.
    div_count: usize,
    /// Candidate primes in ascending order.
    cands: Vec<TotientCand>,
    /// `reach[j]` (bitset over divisor indices): divisors that are `φ` of some
    /// product of powers of `cands[..j]`.
    reach: Vec<Vec<u64>>,
}

impl TotientTable {
    /// Builds the table for `m`, or `None` when `m` is trivially not a totient
    /// (`m ≤ 0`, or `m` odd and `> 1`).
    fn new(m: &Int) -> Option<TotientTable> {
        if !m.is_positive() || (m.is_odd() && !m.is_one()) {
            return None;
        }
        let primes = m.factor_exponents();
        let mut strides = Vec::with_capacity(primes.len());
        let mut div_count = 1usize;
        for (_, e) in &primes {
            strides.push(div_count);
            div_count = div_count
                .checked_mul(*e as usize + 1)
                .expect("inverse_totient: too many divisors");
        }

        // Enumerate the divisors d | m (with exponent vectors) and keep the
        // primes p = d + 1.
        let mut cands = Vec::new();
        let mut vecv = vec![0u32; primes.len()];
        for idx in 0..div_count {
            let mut rest = idx;
            let mut d = Int::ONE;
            for (t, (q, e)) in primes.iter().enumerate() {
                let a = (rest % (*e as usize + 1)) as u32;
                rest /= *e as usize + 1;
                vecv[t] = a;
                d = d.mul(&q.pow(a));
            }
            let p = d.add(&Int::ONE);
            if !p.is_prime_bpsw() {
                continue;
            }
            // φ(pᵏ) = pᵏ⁻¹·d: extra powers of p only if p | m.
            let pos = primes.iter().position(|(q, _)| *q == p);
            let mut powers = vec![(1u32, idx, vecv.clone())];
            if let Some(t) = pos {
                let mut v = vecv.clone();
                let mut k = 1u32;
                while v[t] < primes[t].1 {
                    v[t] += 1;
                    k += 1;
                    let off: usize = v.iter().zip(&strides).map(|(a, s)| *a as usize * s).sum();
                    powers.push((k, off, v.clone()));
                }
            }
            cands.push(TotientCand { p, powers });
        }
        cands.sort_by(|a, b| a.p.cmp(&b.p));

        // Forward reachability pass.
        let words = div_count.div_ceil(64);
        let mut reach = Vec::with_capacity(cands.len() + 1);
        let mut cur = vec![0u64; words];
        cur[0] = 1; // φ(1) = 1 ↦ index 0
        reach.push(cur.clone());
        let mut digits = vec![0u32; primes.len()];
        for c in &cands {
            let prev = reach.last().expect("non-empty");
            for idx in 0..div_count {
                if prev[idx / 64] >> (idx % 64) & 1 == 0 {
                    continue;
                }
                decode(idx, &primes, &mut digits);
                for (_, off, v) in &c.powers {
                    if digits
                        .iter()
                        .zip(v)
                        .zip(&primes)
                        .all(|((a, b), (_, e))| a + b <= *e)
                    {
                        let j = idx + off;
                        cur[j / 64] |= 1 << (j % 64);
                    }
                }
            }
            reach.push(cur.clone());
        }
        Some(TotientTable {
            primes,
            div_count,
            cands,
            reach,
        })
    }

    fn reachable(&self, j: usize, idx: usize) -> bool {
        self.reach[j][idx / 64] >> (idx % 64) & 1 == 1
    }

    /// Pushes `acc · x` for every `x` built from `cands[..j]` with `φ(x)` equal
    /// to divisor `idx`. Only called on reachable `(j, idx)`.
    fn enumerate(&self, idx: usize, j: usize, acc: &Int, out: &mut Vec<Int>) {
        if idx == 0 {
            out.push(acc.clone());
        }
        let mut here = vec![0u32; self.primes.len()];
        decode(idx, &self.primes, &mut here);
        for i in (0..j).rev() {
            let c = &self.cands[i];
            for (k, off, v) in &c.powers {
                if !here.iter().zip(v).all(|(a, b)| a >= b) {
                    continue;
                }
                let rest = idx - off;
                if self.reachable(i, rest) {
                    let next = acc.mul(&c.p.pow(*k));
                    self.enumerate(rest, i, &next, out);
                }
            }
        }
    }
}

/// Decodes a divisor index into its exponent vector.
fn decode(mut idx: usize, primes: &[(Int, u32)], digits: &mut [u32]) {
    for (t, (_, e)) in primes.iter().enumerate() {
        digits[t] = (idx % (*e as usize + 1)) as u32;
        idx /= *e as usize + 1;
    }
}
