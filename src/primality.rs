//! Certificate-based primality **proving** — deterministic proofs, not just
//! probable-prime tests.
//!
//! A Baillie–PSW test (see [`Int::is_prime_bpsw`](crate::Int::is_prime_bpsw))
//! is fast and has no known counterexample, but for a value above `2^64` it is
//! only a *probable*-prime test: it can, in principle, be fooled. This module
//! instead produces a short, independently **checkable certificate** that a
//! number is prime, following the classical `n−1` methods.
//!
//! # Methods
//!
//! - **Pocklington–Lehmer (`n−1`).** Factor `n−1 = F·R` with `F` fully factored
//!   and `gcd(F, R) = 1`. If, for every prime `q | F`, there is a base `a_q`
//!   with `a_q^{n−1} ≡ 1 (mod n)` and `gcd(a_q^{(n−1)/q} − 1, n) = 1`, then every
//!   prime factor `p` of `n` satisfies `p ≡ 1 (mod F)`. When `F > √n` this forces
//!   `n` to be prime. (Brillhart, Lehmer & Selfridge, *Math. Comp.* **29** (1975);
//!   Crandall & Pomerance §4.1; HAC §4.6.)
//! - **BLS `n^{1/3}` refinement.** The same witnesses only need `F > n^{1/3}`:
//!   if `n` were composite it would be a product of exactly two primes
//!   `p_i = a_i·F + 1`, and the pair `(a_1, a_2)` would be the integer roots of
//!   `x² − t·x + s` where `s = ⌊R/F⌋`, `t = R mod F`. If that quadratic's
//!   discriminant `t² − 4s` is not a perfect square (or the implied factor does
//!   not divide `n`), no such factorization exists and `n` is prime. This lets a
//!   proof succeed with only a third of `n−1` factored.
//!
//! - **Atkin–Morain [ECPP].** When `n∓1` cannot be factored far enough, the
//!   elliptic-curve method takes over: it builds a CM elliptic curve
//!   `E/(ℤ/nℤ)` whose order has a large prime factor `q`, reducing the primality
//!   of `n` to that of the smaller `q` (proved recursively). This depends only on
//!   `n` being represented by a small imaginary-quadratic discriminant, not on
//!   any factorisation of `n∓1`. The
//!   Goldwasser–Kilian/Atkin–Morain theorem and construction are documented in
//!   the (private) `ecpp` module.
//!
//! All three share one recursive [`PrimalityCertificate`]: each `n − 1` proof
//! records the [`Bound`] (√ or ∛) it relied on, and every large prime it defers
//! to (a factor `q | F`, or the ECPP prime `q`) carries its own nested
//! certificate, so [`PrimalityCertificate::verify`] re-derives the whole proof
//! from the ground up, bottoming out at values below `2^64` where Baillie–PSW is
//! deterministic.
//!
//! APR-CL — the other general-purpose proof — is out of scope; ECPP is only
//! attempted when the `n−1` factoring cannot reach the `n^{1/3}` bound, and if
//! ECPP also finds no usable discriminant [`prove_prime`] reports
//! [`Primality::Unproven`].
//!
//! [ECPP]: https://en.wikipedia.org/wiki/Elliptic_curve_primality
//!
//! # Example
//!
//! ```
//! use puremp::Int;
//! use puremp::primality::{prove_prime, Primality};
//!
//! // The Mersenne prime 2^89 − 1; n − 1 factors into small primes.
//! let n = Int::from(2).pow(89).sub(&Int::ONE);
//! match prove_prime(&n) {
//!     Primality::Prime(cert) => assert!(cert.verify(&n)),
//!     _ => panic!("2^89 − 1 is prime"),
//! }
//!
//! // A composite is rejected with certainty.
//! assert!(matches!(prove_prime(&Int::from(561)), Primality::Composite));
//! ```

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::int::Int;

/// Trial-division bound used to peel the small-prime part of `n−1`.
const TRIAL_BOUND: u64 = 1 << 16;

/// Largest base tried when searching for a Pocklington witness `a_q`. Real
/// primes almost always yield a witness at a tiny base, so this is generous.
const WITNESS_LIMIT: u64 = 4096;

/// A composite cofactor of `n−1` this many bits or smaller may be handed to the
/// full [`Int::factorize`] escalation (trial → rho → ECM/QS) to try to reach the
/// proving bound. Larger hard cofactors are left unfactored (→ `Unproven`) so a
/// proof attempt never runs away.
const FACTOR_CAP_BITS: u32 = 128;

/// The outcome of a primality **proof** attempt (see [`prove_prime`]).
#[derive(Clone, Debug)]
pub enum Primality {
    /// `n` is prime, witnessed by a checkable [`PrimalityCertificate`].
    Prime(PrimalityCertificate),
    /// `n` is *certainly* composite (a factor, a Fermat witness, or a
    /// deterministic Baillie–PSW rejection was found).
    Composite,
    /// `n` is a probable prime, but neither the `n∓1` methods nor ECPP (over the
    /// built-in discriminant table) could build a certificate. A different
    /// general-purpose proof (APR-CL), or a larger discriminant table, would be
    /// needed.
    Unproven,
}

/// How much of `n−1` had to be factored for the proof to go through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// Classic Pocklington–Lehmer: the factored part `F` exceeds `√n`.
    Sqrt,
    /// BLS refinement: `F` only exceeds `n^{1/3}`, closed by the discriminant
    /// test on the residual cofactor.
    Cbrt,
}

/// A proof that one prime factor `q` of `F` is itself prime.
#[derive(Clone, Debug)]
enum PrimeProof {
    /// `q < 2^64`, where Baillie–PSW is a proven, deterministic test.
    Small,
    /// `q ≥ 2^64`, proved recursively by its own certificate.
    Recursive(Box<PrimalityCertificate>),
}

/// One Atkin–Morain **ECPP** downrun step: an elliptic curve
/// `E: y² = x³ + a·x + b` over `ℤ/nℤ`, a witness point `P = (px, py)`, the target
/// order `m = k·q` with `q` a large prime, and a recursive proof that `q` is
/// prime. The Goldwasser–Kilian theorem this witnesses is documented in the
/// (private) `ecpp` module. All ring elements
/// are stored as canonical residues in `[0, n)`.
#[derive(Clone, Debug)]
pub(crate) struct EcppCert {
    /// Curve coefficient `a`.
    pub(crate) a: Int,
    /// Curve coefficient `b`.
    pub(crate) b: Int,
    /// The curve order used, `m = k·q`.
    pub(crate) m: Int,
    /// The large prime factor `q` of `m`, with `q > (n^{1/4}+1)²`.
    pub(crate) q: Int,
    /// The smooth cofactor `k = m/q`.
    pub(crate) k: Int,
    /// `x`-coordinate of the witness point `P`.
    pub(crate) px: Int,
    /// `y`-coordinate of the witness point `P`.
    pub(crate) py: Int,
    /// Recursive proof that `q` is prime.
    pub(crate) q_proof: Box<PrimalityCertificate>,
}

/// One prime power `q^e ‖ F`, its Pocklington witness, and the proof that `q`
/// is prime.
#[derive(Clone, Debug)]
struct FactorWitness {
    /// The prime `q`.
    prime: Int,
    /// Its multiplicity `e` in `F` (so `q^e` divides `n − 1`).
    exp: u32,
    /// A base `a` with `a^{n−1} ≡ 1` and `gcd(a^{(n−1)/q} − 1, n) = 1 (mod n)`.
    witness: Int,
    /// Proof that `prime` is prime.
    proof: PrimeProof,
}

/// The concrete proof carried by a [`PrimalityCertificate`].
#[derive(Clone, Debug)]
enum Kind {
    /// `n < 2^64`: settled directly by deterministic Baillie–PSW.
    SmallBpsw,
    /// An `n − 1` (Pocklington / BLS) proof.
    NMinusOne {
        /// The fully-factored part `F = ∏ prime^exp` of `n − 1`.
        factors: Vec<FactorWitness>,
        /// The unfactored cofactor `R = (n − 1) / F`, coprime to `F`.
        cofactor: Int,
        /// Which size bound on `F` the proof relied on.
        bound: Bound,
    },
    /// An Atkin–Morain ECPP proof: primality of `n` reduced to that of a smaller
    /// prime `q` via an elliptic curve. Boxed (the arm is much larger than the
    /// others). See [`EcppCert`].
    Ecpp(Box<EcppCert>),
}

/// A checkable proof that a specific number is prime.
///
/// Produced by [`prove_prime`] and re-checked, independently of how it was
/// built, by [`PrimalityCertificate::verify`]. The certificate is *recursive*:
/// verifying it re-derives every step, including the primality of each prime
/// factor of `F`, so a passing [`verify`](PrimalityCertificate::verify) is a
/// self-contained proof.
#[derive(Clone, Debug)]
pub struct PrimalityCertificate {
    /// The number this certificate is about.
    n: Int,
    /// The proof method and its data.
    kind: Kind,
}

impl PrimalityCertificate {
    /// The number this certificate proves prime.
    pub fn n(&self) -> &Int {
        &self.n
    }

    /// The size bound on the factored part `F` this proof relied on, or `None`
    /// for a small (`< 2^64`, Baillie–PSW) certificate.
    pub fn bound(&self) -> Option<Bound> {
        match &self.kind {
            Kind::SmallBpsw | Kind::Ecpp(_) => None,
            Kind::NMinusOne { bound, .. } => Some(*bound),
        }
    }

    /// Re-checks the certificate from scratch, returning `true` only if it is a
    /// valid proof that `n` (which must equal the number the certificate was
    /// built for) is prime.
    ///
    /// This shares no state with [`prove_prime`]: it recomputes `F`, re-verifies
    /// every witness congruence, recursively re-checks each prime factor of `F`,
    /// and (for a BLS proof) re-runs the discriminant test. Tampering with any
    /// field makes it return `false`.
    pub fn verify(&self, n: &Int) -> bool {
        if &self.n != n {
            return false;
        }
        verify_cert(self)
    }
}

/// Attempts to **prove** whether `n` is prime.
///
/// Returns:
/// - [`Primality::Prime`] with a certificate whose
///   [`verify`](PrimalityCertificate::verify) succeeds, when a Pocklington / BLS
///   `n − 1` proof is found (values below `2^64` are settled directly by
///   deterministic Baillie–PSW);
/// - [`Primality::Composite`] — with certainty — for composites (including
///   Carmichael numbers and products of large primes);
/// - [`Primality::Unproven`] when `n` is a probable prime that neither the `n−1`
///   proof nor ECPP could settle.
pub fn prove_prime(n: &Int) -> Primality {
    // Units and negatives are not prime; report them as (not-prime) composite.
    if n < &Int::from(2) {
        return Primality::Composite;
    }
    // Below 2^64 Baillie–PSW is a proven, deterministic test.
    if n.to_u64().is_some() {
        return if n.is_prime_bpsw() {
            Primality::Prime(PrimalityCertificate {
                n: n.clone(),
                kind: Kind::SmallBpsw,
            })
        } else {
            Primality::Composite
        };
    }
    // A Baillie–PSW *rejection* is always correct: n is certainly composite.
    if !n.is_prime_bpsw() {
        return Primality::Composite;
    }
    // n is a probable prime; try to build an n − 1 proof first (cheap when it
    // works), then fall back to Atkin–Morain ECPP.
    match prove_n_minus_1(n) {
        Attempt::Proved(cert) => Primality::Prime(cert),
        Attempt::Composite => Primality::Composite,
        Attempt::Insufficient => match crate::ecpp::prove_ecpp(n) {
            Some(cert) => Primality::Prime(PrimalityCertificate {
                n: n.clone(),
                kind: Kind::Ecpp(Box::new(cert)),
            }),
            None => Primality::Unproven,
        },
    }
}

/// The internal result of the `n − 1` proving attempt.
enum Attempt {
    /// A finished certificate.
    Proved(PrimalityCertificate),
    /// `n` was shown composite along the way (a factor or Fermat witness).
    Composite,
    /// Could not factor `n − 1` far enough, or no witness was found.
    Insufficient,
}

/// Runs the Pocklington / BLS `n − 1` proof for a probable prime `n ≥ 2^64`.
fn prove_n_minus_1(n: &Int) -> Attempt {
    let m = n.sub(&Int::ONE); // n − 1
    let (mut primes, mut cofactor) = peel_smooth(&m);

    // If the residual cofactor is itself prime, it fully factors n − 1.
    if cofactor > Int::ONE && cofactor.is_prime_bpsw() {
        primes.push((cofactor.clone(), 1));
        cofactor = Int::ONE;
    }

    let mut factored = product_pow(&primes);

    // Not enough yet? If the leftover is a small-enough composite, spend the
    // full factoring escalation on it to try to reach the bound.
    if cube_le(&factored, n) && cofactor > Int::ONE && cofactor.bit_len() <= FACTOR_CAP_BITS {
        for p in cofactor.factorize() {
            match primes.iter_mut().find(|(q, _)| *q == p) {
                Some((_, e)) => *e += 1,
                None => primes.push((p, 1)),
            }
        }
        primes.sort_by(|a, b| a.0.cmp(&b.0));
        factored = product_pow(&primes);
        cofactor = m.div_trunc(&factored);
    }

    // Decide which bound (if any) F reaches.
    let bound = if factored.mul(&factored) > *n {
        Bound::Sqrt
    } else if !cube_le(&factored, n) {
        Bound::Cbrt
    } else {
        return Attempt::Insufficient;
    };

    assemble(n, &m, primes, cofactor, bound)
}

/// `true` iff `f³ ≤ n`.
fn cube_le(f: &Int, n: &Int) -> bool {
    f.mul(f).mul(f) <= *n
}

/// Finds a witness for each prime and, on success, builds the certificate.
fn assemble(n: &Int, m: &Int, primes: Vec<(Int, u32)>, cofactor: Int, bound: Bound) -> Attempt {
    let mut factors = Vec::with_capacity(primes.len());
    for (q, e) in primes {
        let witness = match find_witness(n, m, &q) {
            Witness::Found(a) => a,
            Witness::Composite => return Attempt::Composite,
            Witness::NotFound => return Attempt::Insufficient,
        };
        // Prove q itself prime (recursively for large q).
        let proof = if q.to_u64().is_some() {
            PrimeProof::Small
        } else {
            match prove_prime(&q) {
                Primality::Prime(sub) => PrimeProof::Recursive(Box::new(sub)),
                _ => return Attempt::Insufficient,
            }
        };
        factors.push(FactorWitness {
            prime: q,
            exp: e,
            witness,
            proof,
        });
    }
    // For a BLS (∛) proof, the discriminant test must actually close the case.
    if bound == Bound::Cbrt {
        let f = product_pow_from(&factors);
        if matches!(bls_discriminant(n, &f, &cofactor), Disc::Composite) {
            return Attempt::Composite;
        }
    }
    Attempt::Proved(PrimalityCertificate {
        n: n.clone(),
        kind: Kind::NMinusOne {
            factors,
            cofactor,
            bound,
        },
    })
}

/// The result of searching for a Pocklington witness for one prime `q`.
enum Witness {
    /// A base satisfying both congruences.
    Found(Int),
    /// A proof (Fermat failure or a proper factor) that `n` is composite.
    Composite,
    /// No base up to [`WITNESS_LIMIT`] worked.
    NotFound,
}

/// Searches bases `a = 2, 3, …` for one with `a^{n−1} ≡ 1 (mod n)` and
/// `gcd(a^{(n−1)/q} − 1, n) = 1`.
fn find_witness(n: &Int, m: &Int, q: &Int) -> Witness {
    let e = m.div_trunc(q); // (n − 1) / q, exact
    let mut a = 2u64;
    while a <= WITNESS_LIMIT {
        let base = Int::from(a);
        // a^{n−1} must be 1 mod n, otherwise n is composite (Fermat).
        if base.modpow(m, n) != Int::ONE {
            return Witness::Composite;
        }
        let x = base.modpow(&e, n);
        let g = x.sub(&Int::ONE).gcd(n);
        if g == Int::ONE {
            return Witness::Found(base);
        }
        if g != *n {
            // 1 < gcd < n is a proper factor: n is composite.
            return Witness::Composite;
        }
        // g == n means a^{(n−1)/q} ≡ 1; try the next base.
        a += 1;
    }
    Witness::NotFound
}

/// Peels the small-prime part of `m` by trial division up to [`TRIAL_BOUND`],
/// returning the prime powers found (each fully divided out) and the residual
/// cofactor coprime to every peeled prime.
fn peel_smooth(m: &Int) -> (Vec<(Int, u32)>, Int) {
    let mut primes: Vec<(Int, u32)> = Vec::new();
    let mut c = m.clone();

    // Factor of two.
    let two = Int::from(2);
    let mut e = 0u32;
    while c > Int::ZERO && c.is_even() {
        c = c.div_trunc(&two);
        e += 1;
    }
    if e > 0 {
        primes.push((two, e));
    }

    // Odd trial divisors.
    let mut d = 3u64;
    while d <= TRIAL_BOUND {
        let dn = Int::from(d);
        // Stop early once the divisor squared passes the cofactor.
        if dn.mul(&dn) > c {
            break;
        }
        let mut e = 0u32;
        loop {
            let (q, r) = c.div_rem_trunc(&dn);
            if r.is_zero() {
                c = q;
                e += 1;
            } else {
                break;
            }
        }
        if e > 0 {
            primes.push((dn, e));
        }
        d += 2;
    }

    (primes, c)
}

/// `∏ prime^exp` for a `(prime, exp)` list.
fn product_pow(primes: &[(Int, u32)]) -> Int {
    let mut f = Int::ONE;
    for (p, e) in primes {
        f = f.mul(&p.pow(*e));
    }
    f
}

/// `∏ prime^exp` for a witness list.
fn product_pow_from(factors: &[FactorWitness]) -> Int {
    let mut f = Int::ONE;
    for fw in factors {
        f = f.mul(&fw.prime.pow(fw.exp));
    }
    f
}

/// Outcome of the BLS discriminant test.
enum Disc {
    /// No composite factorization `n = (a₁F+1)(a₂F+1)` exists: `n` is prime.
    Prime,
    /// Such a factorization exists and its factor divides `n`: `n` is composite.
    Composite,
}

/// The BLS `n^{1/3}` discriminant test. Assumes every prime factor of `n` is
/// `≡ 1 (mod F)` (established by the Pocklington witnesses) and `F³ > n`, so any
/// composite `n` is a product of exactly two primes `a_i·F + 1`. The pair
/// `(a₁, a₂)` are the integer roots of `x² − t·x + s`, with `s = ⌊R/F⌋`,
/// `t = R mod F`; if the discriminant is not a perfect square (or the implied
/// factor does not divide `n`), no such factorization exists.
fn bls_discriminant(n: &Int, f: &Int, r: &Int) -> Disc {
    let (s, t) = r.div_rem_trunc(f); // R = s·F + t, 0 ≤ t < F
    let disc = t.mul(&t).sub(&s.mul(&Int::from(4)));
    if disc.is_negative() {
        return Disc::Prime; // no real roots
    }
    let root = match disc.sqrt_exact() {
        Some(root) => root,
        None => return Disc::Prime, // not a perfect square
    };
    let num = t.sub(&root); // 2·a₁
    if num.is_odd() || num.is_negative() {
        return Disc::Prime; // no non-negative integer root
    }
    let a1 = num.div_trunc(&Int::from(2));
    if a1.is_zero() {
        return Disc::Prime; // factor would be 1, not proper
    }
    let cand = a1.mul(f).add(&Int::ONE); // a₁·F + 1
    if cand > Int::ONE && &cand < n && cand.divides(n) {
        return Disc::Composite;
    }
    Disc::Prime
}

// --- verification (independent re-check) ---------------------------------

/// Independently re-checks a certificate. See [`PrimalityCertificate::verify`].
fn verify_cert(cert: &PrimalityCertificate) -> bool {
    let n = &cert.n;
    if n < &Int::from(2) {
        return false;
    }
    match &cert.kind {
        Kind::SmallBpsw => n.to_u64().is_some() && n.is_prime_bpsw(),
        Kind::NMinusOne {
            factors,
            cofactor,
            bound,
        } => verify_n_minus_1(n, factors, cofactor, *bound),
        Kind::Ecpp(ecpp) => crate::ecpp::verify_ecpp(n, ecpp),
    }
}

/// Re-checks an `n − 1` (Pocklington / BLS) certificate from scratch.
fn verify_n_minus_1(n: &Int, factors: &[FactorWitness], cofactor: &Int, bound: Bound) -> bool {
    if factors.is_empty() || cofactor < &Int::ONE {
        return false;
    }
    let m = n.sub(&Int::ONE);
    let f = product_pow_from(factors);

    // F · R must equal n − 1, and F and R must be coprime.
    if f.mul(cofactor) != m {
        return false;
    }
    if f.gcd(cofactor) != Int::ONE {
        return false;
    }

    // The bound on F must hold, and match the recorded kind.
    let f2 = f.mul(&f);
    match bound {
        Bound::Sqrt => {
            if f2 <= *n {
                return false;
            }
        }
        Bound::Cbrt => {
            if f2.mul(&f) <= *n {
                return false;
            }
        }
    }

    // Every prime power and its witness.
    for fw in factors {
        // q must be prime (directly for small q, recursively otherwise).
        match &fw.proof {
            PrimeProof::Small => {
                if fw.prime.to_u64().is_none() || !fw.prime.is_prime_bpsw() {
                    return false;
                }
            }
            PrimeProof::Recursive(sub) => {
                if sub.n() != &fw.prime || !sub.verify(&fw.prime) {
                    return false;
                }
            }
        }
        // q^exp must actually divide n − 1 to the claimed multiplicity.
        let qe = fw.prime.pow(fw.exp);
        if !qe.divides(&m) {
            return false;
        }
        // Witness congruences: a^{n−1} ≡ 1 and gcd(a^{(n−1)/q} − 1, n) = 1.
        if fw.witness < Int::from(2) {
            return false;
        }
        if fw.witness.modpow(&m, n) != Int::ONE {
            return false;
        }
        let e = m.div_trunc(&fw.prime);
        let x = fw.witness.modpow(&e, n);
        if x.sub(&Int::ONE).gcd(n) != Int::ONE {
            return false;
        }
    }

    // A BLS proof additionally needs the discriminant test to close the case.
    if bound == Bound::Cbrt && matches!(bls_discriminant(n, &f, cofactor), Disc::Composite) {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tampered certificate — any field mutated — must fail `verify`.
    #[test]
    fn tampered_certificate_is_rejected() {
        // 2^89 − 1 is a Pocklington-provable prime.
        let n = Int::from(2).pow(89).sub(&Int::ONE);
        let cert = match prove_prime(&n) {
            Primality::Prime(c) => c,
            other => panic!("expected a proof, got {other:?}"),
        };
        assert!(cert.verify(&n));

        // Corrupt a witness base.
        let mut bad = cert.clone();
        if let Kind::NMinusOne { factors, .. } = &mut bad.kind {
            factors[0].witness = factors[0].witness.add(&Int::ONE);
        }
        assert!(!bad.verify(&n), "bumped witness must fail");

        // Corrupt a prime-power exponent.
        let mut bad = cert.clone();
        if let Kind::NMinusOne { factors, .. } = &mut bad.kind {
            factors[0].exp += 1;
        }
        assert!(!bad.verify(&n), "wrong exponent must fail");

        // Corrupt the cofactor.
        let mut bad = cert.clone();
        if let Kind::NMinusOne { cofactor, .. } = &mut bad.kind {
            *cofactor = cofactor.add(&Int::ONE);
        }
        assert!(!bad.verify(&n), "wrong cofactor must fail");

        // Drop a factor from F (so the bound no longer holds).
        let mut bad = cert.clone();
        if let Kind::NMinusOne { factors, .. } = &mut bad.kind {
            factors.pop();
        }
        assert!(!bad.verify(&n), "shrunken F must fail");

        // Claim the certificate is about a different number.
        assert!(!cert.verify(&n.add(&Int::from(2))), "wrong n must fail");
    }

    /// Falsely relabelling a genuine BLS (∛) proof as a plain Pocklington (√)
    /// one must be rejected, since `F² ≤ n` there.
    #[test]
    fn forged_bound_label_is_rejected() {
        let n = Int::from(2).pow(127).sub(&Int::ONE); // needs the BLS bound
        let cert = match prove_prime(&n) {
            Primality::Prime(c) => c,
            other => panic!("expected a proof, got {other:?}"),
        };
        assert_eq!(cert.bound(), Some(Bound::Cbrt));
        let mut bad = cert.clone();
        if let Kind::NMinusOne { bound, .. } = &mut bad.kind {
            *bound = Bound::Sqrt;
        }
        assert!(!bad.verify(&n), "F² ≤ n, so the √ label is a lie");
    }
}
