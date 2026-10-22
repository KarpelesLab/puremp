//! Integer roots and perfect-power detection on [`Int`].
//!
//! - [`Int::isqrt`] / [`Int::isqrt_rem`] — `⌊√x⌋` and the remainder.
//! - [`Int::root`] / [`Int::root_rem`] — the `n`th root truncated toward
//!   zero (odd roots of negatives allowed) and the remainder.
//! - [`Int::is_perfect_square`] / [`Int::is_perfect_power`] — GMP's
//!   `mpz_perfect_square_p` / `mpz_perfect_power_p`.

use alloc::vec::Vec;

use crate::int::{Int, Sign};
use crate::nat::Nat;

/// Trial-division bound for [`Int::is_perfect_power`]: a value with no prime
/// factor below it can only be a `k`th power for `k ≤ bits / log2(BOUND)`.
const TRIAL_BOUND: u64 = 1024;

/// `SQUARE_MOD_256[r]` is true iff `r` is a square modulo 256.
const SQUARE_MOD_256: [bool; 256] = {
    let mut t = [false; 256];
    let mut i = 0;
    while i < 256 {
        t[(i * i) % 256] = true;
        i += 1;
    }
    t
};

/// Primes `≤ n` (ascending), by the sieve of Eratosthenes.
fn primes_up_to(n: u64) -> Vec<u64> {
    let n = n as usize;
    let mut composite = alloc::vec![false; n + 1];
    let mut out = Vec::new();
    for i in 2..=n {
        if !composite[i] {
            out.push(i as u64);
            for j in (i * i..=n).step_by(i) {
                composite[j] = true;
            }
        }
    }
    out
}

/// Whether `m` is an exact `k`th power, for `k ≥ 2`.
fn is_kth_power(m: &Nat, k: u64) -> bool {
    let k = u32::try_from(k).expect("root degree below the bit length");
    m.nth_root_floor(k).pow(k) == *m
}

/// `⌊m^(1/n)⌋` for `n ≥ 1`, cutting large degrees short (`m < 2^n` ⇒ the root
/// is `0` or `1`) so the degree may be any `u64`.
fn nat_root(m: &Nat, n: u64) -> Nat {
    if n >= m.bit_len() {
        return if m.is_zero() { Nat::zero() } else { Nat::one() };
    }
    m.nth_root_floor(u32::try_from(n).expect("n < bit length"))
}

impl Int {
    /// Returns `⌊√self⌋`.
    ///
    /// # Panics
    ///
    /// If `self` is negative.
    pub fn isqrt(&self) -> Int {
        assert!(!self.is_negative(), "isqrt of a negative integer");
        Int::from(self.magnitude_ref().isqrt())
    }

    /// Returns `(r, self − r²)` with `r = ⌊√self⌋`.
    ///
    /// # Panics
    ///
    /// If `self` is negative.
    pub fn isqrt_rem(&self) -> (Int, Int) {
        let r = self.isqrt();
        let rem = self.sub(&r.square());
        (r, rem)
    }

    /// Returns the `n`th root truncated toward zero, `sgn(self)·⌊|self|^(1/n)⌋`
    /// (GMP's `mpz_root`).
    ///
    /// # Panics
    ///
    /// If `n == 0`, or `n` is even and `self` is negative.
    pub fn root(&self, n: u64) -> Int {
        assert!(n != 0, "root: degree must be non-zero");
        assert!(
            !(self.is_negative() && n.is_multiple_of(2)),
            "root: even root of a negative integer"
        );
        Int::from_sign_magnitude(self.sign(), nat_root(&self.magnitude_ref(), n))
    }

    /// Returns `(r, self − rⁿ)` with `r = self.root(n)`; the remainder has the
    /// sign of `self` (GMP's `mpz_rootrem`).
    ///
    /// # Panics
    ///
    /// As [`Int::root`].
    pub fn root_rem(&self, n: u64) -> (Int, Int) {
        let r = self.root(n);
        let rem = self.sub(&r.pow(n));
        (r, rem)
    }

    /// Returns `true` if `self` is the square of an integer (`0` and `1`
    /// included; negatives never are).
    pub fn is_perfect_square(&self) -> bool {
        if self.is_negative() {
            return false;
        }
        // Squares occupy 44 of the 256 residues mod 256: reject most inputs
        // without a square root.
        SQUARE_MOD_256[(self.least_significant_limb() & 0xff) as usize]
            && self.sqrt_exact().is_some()
    }

    /// Returns `true` if `self = rᵏ` for some integer `r` and `k ≥ 2`. `0`,
    /// `1` and `-1` are perfect powers; a negative value is one only through
    /// an odd exponent (so `-8` is but `-4` is not), as in GMP's
    /// `mpz_perfect_power_p`.
    pub fn is_perfect_power(&self) -> bool {
        let m = self.magnitude_ref();
        if m.bit_len() <= 1 {
            return true;
        }
        let neg = self.sign() == Sign::Negative;
        if !neg && self.is_perfect_square() {
            return true;
        }
        // Any exponent k must divide the multiplicity e of every prime factor;
        // a prime q | k gives r^k = (r^(k/q))^q, so testing the prime divisors
        // of e suffices. Find the smallest prime factor below TRIAL_BOUND.
        let small = primes_up_to(TRIAL_BOUND);
        let mut found = None;
        for &p in &small {
            let (mut q, r) = m.divmod_small(p);
            if r != 0 {
                continue;
            }
            let mut e = 1u64;
            loop {
                let (q2, r2) = q.divmod_small(p);
                if r2 != 0 {
                    break;
                }
                q = q2;
                e += 1;
            }
            found = Some(e);
            break;
        }
        let odd_only = |q: &u64| !neg || q % 2 == 1;
        match found {
            Some(e) => primes_up_to(e)
                .into_iter()
                .filter(|q| e.is_multiple_of(*q))
                .filter(odd_only)
                .any(|q| is_kth_power(&m, q)),
            None => {
                // Every prime factor is ≥ TRIAL_BOUND, so r ≥ TRIAL_BOUND and
                // k ≤ bits / log2(TRIAL_BOUND).
                let kmax = m.bit_len() / (TRIAL_BOUND.ilog2() as u64);
                primes_up_to(kmax)
                    .into_iter()
                    .filter(odd_only)
                    .any(|q| is_kth_power(&m, q))
            }
        }
    }
}
