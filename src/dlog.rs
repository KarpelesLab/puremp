//! Discrete logarithms — solving `g^x ≡ h (mod n)` for the exponent `x`.
//!
//! Given a base `g`, a target `h`, a modulus `n`, and the order of the group
//! (or an upper bound on the order of `g`), [`discrete_log`] returns the least
//! non-negative `x` with `g^x ≡ h (mod n)`, or `None` when no such `x` exists.
//!
//! Two classical square-root algorithms are provided (both run in `O(√order)`
//! group operations):
//!
//! - [`bsgs`] — Shanks' *baby-step giant-step* (HAC §3.6.2). Deterministic, and
//!   it always returns the least solution, but it needs `O(√order)` memory for
//!   the baby-step table.
//! - [`pollard_rho`] — *Pollard's rho for logarithms* (HAC §3.6.3; Crandall &
//!   Pomerance §5.2). Uses only `O(1)` memory via Floyd cycle detection over the
//!   partition-into-three-sets iteration, at the cost of being randomized.
//!
//! When the group `order` factors as `∏ pᵢ^eᵢ`, [`pohlig_hellman`] (HAC §3.6.4)
//! reduces the problem to a discrete logarithm in each prime-order subgroup and
//! recombines the results with the Chinese Remainder Theorem. Its cost is
//! `Σ eᵢ·√pᵢ` group operations — dramatically cheaper than the `√order` of the
//! square-root methods whenever `order` is *smooth* (all prime factors small).
//!
//! [`discrete_log`] dispatches between all three: [`pohlig_hellman`] when the
//! order is composite (so the per-subgroup work is much smaller), otherwise
//! baby-step giant-step for small orders (where the table fits comfortably in
//! memory) and Pollard's rho for large ones, falling back to baby-step
//! giant-step if rho fails to converge.
//!
//! The base is assumed to be a unit modulo `n` (i.e. `gcd(g, n) == 1`), the
//! usual setting for discrete logarithms; a non-invertible base yields `None`.
//!
//! # Example
//!
//! ```
//! use puremp::{Int, dlog::discrete_log};
//!
//! // 2 is a generator of (ℤ/101ℤ)*, whose order is 100.
//! let (g, n, order) = (Int::from(2), Int::from(101), Int::from(100));
//! let h = g.modpow(&Int::from(37), &n); // h = 2^37 mod 101
//! assert_eq!(discrete_log(&g, &h, &n, &order), Some(Int::from(37)));
//! ```

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::int::Int;
use crate::mod_int::ModInt;
use crate::nat::Nat;

/// Orders with a bit length up to this threshold use [`bsgs`] inside
/// [`discrete_log`]; larger ones use [`pollard_rho`]. A 40-bit order needs a
/// baby-step table of at most `2^20` entries.
const BSGS_MAX_ORDER_BITS: u64 = 40;

/// Number of independent Pollard-rho walks [`discrete_log`] tries before giving
/// up on rho and falling back to [`bsgs`].
const RHO_ATTEMPTS: u64 = 24;

/// Reduces `value` into the canonical residue `[0, modulus)`.
#[inline]
fn reduce(value: &Int, modulus: &Int) -> Int {
    value.rem_euclid(modulus)
}

/// `(a · b) mod modulus`, with both factors and the result in `[0, modulus)`.
#[inline]
fn mul_mod(a: &Int, b: &Int, modulus: &Int) -> Int {
    a.mul(b).rem_euclid(modulus)
}

/// Handles the degenerate inputs shared by every algorithm. Returns
/// `Err(result)` when the answer is already determined (`result` may itself be
/// `None` for "provably no solution"), or `Ok((g, h, m))` with the reduced
/// modulus `m = |modulus|` and the base/target in `[0, m)` for the general
/// search.
fn prepare(
    base: &Int,
    target: &Int,
    modulus: &Int,
    order: &Int,
) -> core::result::Result<(Int, Int, Int), Option<Int>> {
    let m = modulus.abs();
    // Modulo 0 or ±1 every value collapses to 0, so x = 0 always works.
    if m <= Int::ONE {
        return Err(Some(Int::ZERO));
    }
    // A non-positive order bounds no exponent range: nothing to search.
    if !order.is_positive() {
        return Err(None);
    }
    let g = reduce(base, &m);
    let h = reduce(target, &m);
    // g^0 = 1, and it is the least non-negative solution when it applies.
    if h.is_one() {
        return Err(Some(Int::ZERO));
    }
    // A base ≡ 0 only ever produces 0 (for x ≥ 1) or 1 (for x = 0, handled above).
    if g.is_zero() {
        return Err(if h.is_zero() { Some(Int::ONE) } else { None });
    }
    Ok((g, h, m))
}

/// Ceiling of the integer square root of `order`.
fn isqrt_ceil(order: &Nat) -> Nat {
    let s = order.isqrt();
    if &s.mul(&s) < order {
        s.add(&Nat::one())
    } else {
        s
    }
}

/// Baby-step giant-step (Shanks): finds the least non-negative `x` with
/// `base^x ≡ target (mod modulus)`, searching the range `[0, order)`.
///
/// `order` must be the order of the group (or any upper bound on the order of
/// `base`); the algorithm runs in `O(√order)` time and space, storing a
/// baby-step table of about `√order` residues. Returns `None` when no such `x`
/// exists in `[0, order)`, or when `base` is not invertible modulo `modulus`.
///
/// See HAC §3.6.2.
///
/// # Example
///
/// ```
/// use puremp::{Int, dlog::bsgs};
///
/// let (g, n, order) = (Int::from(2), Int::from(101), Int::from(100));
/// let h = g.modpow(&Int::from(50), &n);
/// assert_eq!(bsgs(&g, &h, &n, &order), Some(Int::from(50)));
/// ```
pub fn bsgs(base: &Int, target: &Int, modulus: &Int, order: &Int) -> Option<Int> {
    let (g, h, m) = match prepare(base, target, modulus, order) {
        Ok(v) => v,
        Err(done) => return done,
    };

    // m_steps = ⌈√order⌉; both the baby- and giant-step loops run m_steps times.
    let m_steps = isqrt_ceil(&order.magnitude());
    let m_steps = m_steps.to_u64().expect("bsgs: order is too large");
    let m_step_int = Int::from(m_steps);

    // Baby steps: table maps g^j -> j for j in [0, m_steps). Insert the smallest
    // j per residue so the recovered x is the least in its class.
    let mut table: BTreeMap<Nat, u64> = BTreeMap::new();
    let mut power = Int::ONE; // g^0
    for j in 0..m_steps {
        table.entry(power.magnitude()).or_insert(j);
        power = mul_mod(&power, &g, &m);
    }

    // Giant stride: factor = g^{-m_steps}. Requires g to be a unit mod m.
    let g_inv = g.modinv(&m)?;
    let factor = g_inv.modpow(&m_step_int, &m);

    // Giant steps: gamma = h · factor^i = h · g^{-i·m_steps}. A hit gamma = g^j
    // means h = g^{i·m_steps + j}. Scanning i upward yields the least x.
    let mut gamma = h;
    for i in 0..m_steps {
        if let Some(&j) = table.get(&gamma.magnitude()) {
            let x = Int::from(i).mul(&m_step_int).add(&Int::from(j));
            // x may exceed the searched [0, order) window only if the same
            // residue recurs; guard to honour the documented range.
            if &x < order {
                return Some(x);
            }
        }
        gamma = mul_mod(&gamma, &factor, &m);
    }
    None
}

/// One state of a Pollard-rho walk: the current group element `x = g^a · h^b`
/// together with the exponents `a`, `b` (kept reduced modulo `order`).
#[derive(Clone)]
struct Walk {
    x: Int,
    a: Int,
    b: Int,
}

/// Advances a rho walk by one step of the partition-into-three-sets iteration
/// (partitioning by `x mod 3`; the identity `1` lands in the multiply-by-`g`
/// set, never the squaring set, as HAC requires).
fn rho_step(w: &Walk, g: &Int, h: &Int, m: &Int, order: &Int) -> Walk {
    let part = w.x.rem_euclid(&Int::from(3u64));
    if part.is_zero() {
        // Squaring set: x -> x², (a, b) -> (2a, 2b).
        Walk {
            x: mul_mod(&w.x, &w.x, m),
            a: w.a.add(&w.a).rem_euclid(order),
            b: w.b.add(&w.b).rem_euclid(order),
        }
    } else if part.is_one() {
        // Multiply-by-g set: x -> g·x, a -> a + 1.
        Walk {
            x: mul_mod(&w.x, g, m),
            a: w.a.add(&Int::ONE).rem_euclid(order),
            b: w.b.clone(),
        }
    } else {
        // Multiply-by-h set: x -> h·x, b -> b + 1.
        Walk {
            x: mul_mod(&w.x, h, m),
            a: w.a.clone(),
            b: w.b.add(&Int::ONE).rem_euclid(order),
        }
    }
}

/// Solves the linear congruence `coeff · x ≡ rhs (mod order)` and returns the
/// least non-negative candidate `x` that also verifies `g^x ≡ h (mod m)`, or
/// `None` if the congruence is unsolvable or no candidate verifies.
fn solve_and_verify(coeff: &Int, rhs: &Int, order: &Int, g: &Int, h: &Int, m: &Int) -> Option<Int> {
    let coeff = coeff.rem_euclid(order);
    let rhs = rhs.rem_euclid(order);
    if coeff.is_zero() {
        // 0·x ≡ rhs carries no exponent information.
        return None;
    }
    let d = coeff.gcd(order);
    if !d.divides(&rhs) {
        return None;
    }
    let coeff_d = coeff.div_exact(&d);
    let rhs_d = rhs.div_exact(&d);
    let order_d = order.div_exact(&d);
    let inv = coeff_d.modinv(&order_d)?;
    let x0 = mul_mod(&rhs_d, &inv, &order_d);
    // The d solutions in [0, order) are x0 + t·order_d, t = 0..d. Return the
    // least that actually satisfies g^x ≡ h.
    let dd = d.to_u64().unwrap_or(u64::MAX);
    let mut cand = x0;
    for _ in 0..dd {
        if &g.modpow(&cand, m) == h {
            return Some(cand);
        }
        cand = cand.add(&order_d);
        if &cand >= order {
            break;
        }
    }
    None
}

/// Pollard's rho for discrete logarithms: finds an `x` in `[0, order)` with
/// `base^x ≡ target (mod modulus)` using `O(1)` memory (Floyd cycle detection),
/// or `None` if this randomized walk fails to yield a solution.
///
/// `order` must be the order of the group (or an upper bound on the order of
/// `base`); it is best used when `order` is prime. `seed` selects the walk's
/// starting exponents, so distinct seeds explore independent walks — retry with
/// a fresh seed on `None`. The returned `x` is verified to satisfy the
/// congruence but is not guaranteed to be the least such `x` when the true order
/// of `base` is a proper divisor of `order` (use [`bsgs`] for the least).
///
/// See HAC §3.6.3 and Crandall & Pomerance §5.2.
///
/// # Example
///
/// ```
/// use puremp::{Int, dlog::pollard_rho};
///
/// // 3 generates (ℤ/1019ℤ)*, order 1018 = 2·509.
/// let (g, n, order) = (Int::from(3), Int::from(1019), Int::from(1018));
/// let h = g.modpow(&Int::from(222), &n);
/// // Retry across seeds since any single walk is randomized.
/// let x = (0..8).find_map(|s| pollard_rho(&g, &h, &n, &order, s)).unwrap();
/// assert_eq!(g.modpow(&x, &n), h);
/// ```
pub fn pollard_rho(base: &Int, target: &Int, modulus: &Int, order: &Int, seed: u64) -> Option<Int> {
    let (g, h, m) = match prepare(base, target, modulus, order) {
        Ok(v) => v,
        Err(done) => return done,
    };

    // Start at x = g^a0 · h^b0 for seed-dependent (a0, b0), so different seeds
    // give independent walks. b0 ≥ 1 keeps h present in the relation.
    let a0 = Int::from(seed).rem_euclid(order);
    let b0 = Int::from(seed / 2 + 1).rem_euclid(order);
    let start = Walk {
        x: mul_mod(&g.modpow(&a0, &m), &h.modpow(&b0, &m), &m),
        a: a0,
        b: b0,
    };

    // Floyd: tortoise advances one step, hare two, until their x collide. The
    // walk enters a cycle within O(√order) steps; cap generously to bail out.
    let steps = isqrt_ceil(&order.magnitude());
    let cap = steps.to_u64().unwrap_or(u64::MAX).saturating_mul(8).max(64);

    let mut tortoise = start.clone();
    let mut hare = start;
    for _ in 0..cap {
        tortoise = rho_step(&tortoise, &g, &h, &m, order);
        hare = rho_step(&rho_step(&hare, &g, &h, &m, order), &g, &h, &m, order);
        if tortoise.x == hare.x {
            // g^{a_t} h^{b_t} = g^{a_h} h^{b_h}  ⇒  g^{a_t-a_h} = h^{b_h-b_t}
            // ⇒  x·(b_h - b_t) ≡ (a_t - a_h)  (mod ord(g)).
            let coeff = hare.b.sub(&tortoise.b);
            let rhs = tortoise.a.sub(&hare.a);
            return solve_and_verify(&coeff, &rhs, order, &g, &h, &m);
        }
    }
    None
}

/// Pohlig–Hellman discrete logarithm (HAC §3.6.4): finds an `x` in `[0, order)`
/// with `base^x ≡ target (mod modulus)` by solving one small logarithm per prime
/// factor of `order` and recombining the answers with the CRT.
///
/// `order` must be the (multiplicative) order of `base` modulo `modulus`. It is
/// factored as `∏ pᵢ^eᵢ`; for each prime power the exponent `x mod pᵢ^eᵢ` is
/// recovered digit-by-digit in base `pᵢ` (each digit is a logarithm in the
/// order-`pᵢ` subgroup, solved by [`discrete_log`], i.e. [`bsgs`]/[`pollard_rho`]).
/// This costs `Σ eᵢ·√pᵢ` group operations — far cheaper than `√order` when the
/// order is smooth. Returns `None` when no solution exists (some subgroup
/// logarithm has none) or when `base` is not a unit modulo `modulus`.
///
/// The returned `x` is the unique representative in `[0, order)`; it is verified
/// to satisfy the congruence, guarding against an incorrect `order`.
///
/// # Example
///
/// ```
/// use puremp::{Int, dlog::pohlig_hellman};
///
/// // 11 generates (ℤ/1009ℤ)*, whose order 1008 = 2^4·3^2·7 is smooth.
/// let (g, n, order) = (Int::from(11), Int::from(1009), Int::from(1008));
/// let h = g.modpow(&Int::from(555), &n);
/// assert_eq!(pohlig_hellman(&g, &h, &n, &order), Some(Int::from(555)));
/// ```
pub fn pohlig_hellman(base: &Int, target: &Int, modulus: &Int, order: &Int) -> Option<Int> {
    let (g, h, m) = match prepare(base, target, modulus, order) {
        Ok(v) => v,
        Err(done) => return done,
    };

    // g^{-1} mod m — used to strip the digits already recovered.
    let g_inv = g.modinv(&m)?;

    let factors = order.factor_exponents();
    let mut residues: Vec<Int> = Vec::with_capacity(factors.len());
    let mut moduli: Vec<Int> = Vec::with_capacity(factors.len());

    for (p, e) in &factors {
        // Generator of the order-`p` subgroup: γ = g^{order/p}.
        let gamma = g.modpow(&order.div_exact(p), &m);

        // Recover xᵢ = x mod p^e, one base-p digit a_j at a time.
        //   xᵢ = a₀ + a₁·p + … + a_{e-1}·p^{e-1}.
        let mut xi = Int::ZERO; // Σ recovered digits so far (x mod p^j).
        let mut pj = Int::ONE; // p^j.
        for j in 0..*e {
            // β_j = (h · g^{-xi})^{order / p^{j+1}} = γ^{a_j} lives in ⟨γ⟩.
            let stripped = mul_mod(&h, &g_inv.modpow(&xi, &m), &m);
            let beta = stripped.modpow(&order.div_exact(&p.pow(j + 1)), &m);
            // Solve the order-`p` sub-instance for the j-th digit a_j ∈ [0, p).
            let digit = discrete_log(&gamma, &beta, &m, p)?;
            xi = xi.add(&digit.mul(&pj));
            pj = pj.mul(p); // advance to p^{j+1}; ends at p^e.
        }
        residues.push(xi);
        moduli.push(pj);
    }

    // CRT-combine the xᵢ over the pairwise-coprime prime powers p^e.
    let x = Int::crt(&residues, &moduli)?;
    // Only return a genuinely valid logarithm (a wrong `order` can mislead the
    // per-subgroup lifts into an inconsistent, non-verifying candidate).
    if g.modpow(&x, &m) == h { Some(x) } else { None }
}

/// Solves the discrete logarithm `base^x ≡ target (mod modulus)`, returning the
/// least non-negative `x` in `[0, order)`, or `None` when no solution exists.
///
/// `order` must be the order of the multiplicative group modulo `modulus` (or
/// any upper bound on the order of `base`). The base is assumed to be a unit
/// modulo `modulus`; a non-invertible base yields `None`.
///
/// Dispatch, cheapest applicable method first:
///
/// - When `order` is **composite** (has more than one prime factor, counting
///   multiplicity) the problem is smooth-friendly, so [`pohlig_hellman`] handles
///   it — its cost `Σ eᵢ·√pᵢ` beats `√order`, decisively so for smooth orders.
/// - Otherwise (`order` prime, or the Pohlig–Hellman attempt did not verify)
///   small orders fall to deterministic [`bsgs`] (which returns the least `x`)
///   and large ones to [`pollard_rho`] with several independent walks, backing
///   off to [`bsgs`] if none converge.
///
/// See HAC §3.6. Note that when `order` is a *strict* multiple of the true order
/// of `base`, the Pohlig–Hellman path returns the unique representative in
/// `[0, order)` rather than necessarily the least solution; pass the exact order
/// of `base` for the canonical least `x`.
///
/// # Example
///
/// ```
/// use puremp::{Int, dlog::discrete_log};
///
/// let (g, n, order) = (Int::from(2), Int::from(101), Int::from(100));
/// let h = g.modpow(&Int::from(64), &n);
/// assert_eq!(discrete_log(&g, &h, &n, &order), Some(Int::from(64)));
///
/// // No power of 2 is ≡ 0 mod 101, so there is no solution.
/// assert_eq!(discrete_log(&g, &Int::ZERO, &n, &order), None);
/// ```
pub fn discrete_log(base: &Int, target: &Int, modulus: &Int, order: &Int) -> Option<Int> {
    if order.is_positive() {
        let factors = order.factor_exponents();
        // Composite order (≥ 2 prime factors with multiplicity) ⇒ Pohlig–Hellman,
        // whose per-subgroup cost Σ eᵢ·√pᵢ beats the √order of the generic search.
        let composite = factors.len() > 1 || factors.first().is_some_and(|(_, e)| *e > 1);
        if composite && let Some(x) = pohlig_hellman(base, target, modulus, order) {
            return Some(x);
        }
        // If PH is skipped (prime order) or misses (a randomized subgroup walk
        // failed to converge), fall through to the generic square-root search.
    }
    if order.is_positive() && order.magnitude().bit_len() <= BSGS_MAX_ORDER_BITS {
        return bsgs(base, target, modulus, order);
    }
    for seed in 0..RHO_ATTEMPTS {
        if let Some(x) = pollard_rho(base, target, modulus, order, seed) {
            return Some(x);
        }
    }
    // Rho did not converge; the deterministic method still settles it.
    bsgs(base, target, modulus, order)
}

impl ModInt {
    /// Solves `self^x ≡ target (mod m)` for the exponent `x`, given the group
    /// `order` (or an upper bound on the order of `self`). Returns the least
    /// non-negative `x`, or `None` when no solution exists.
    ///
    /// This is the [`discrete_log`] free function specialised to `self` as the
    /// base and its modulus; see it for the algorithm details and assumptions.
    ///
    /// # Example
    ///
    /// ```
    /// use puremp::{Int, ModInt};
    ///
    /// let g = ModInt::new(Int::from(2), Int::from(101));
    /// let h = g.pow(&Int::from(73));
    /// assert_eq!(g.discrete_log(&h, &Int::from(100)), Some(Int::from(73)));
    /// ```
    pub fn discrete_log(&self, target: &ModInt, order: &Int) -> Option<Int> {
        discrete_log(&self.to_int(), &target.to_int(), &self.modulus(), order)
    }
}
