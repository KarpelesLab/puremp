//! Context-cached elementary functions.
//!
//! [`FloatContext`] caches the reusable constants that the stateless
//! [`Float`] transcendentals recompute on every call — chiefly `ln 2` — so that
//! repeated [`exp`](FloatContext::exp) evaluations at a
//! given precision are meaningfully faster. The cache is held by the caller, so
//! the library stays free of global state (`no_std`, dependency-free, no
//! `unsafe`).
//!
//! The reduction is the standard `x = k·ln2 + r` with a `√`-precision second
//! stage (`exp(r) = exp(r/2ʲ)^(2ʲ)`), evaluated in scaled-integer arithmetic.
//! **Multi-prime argument reduction** (Johansson) — reducing by a combination of
//! several prime logarithms to shorten the series further — is a natural
//! extension of this cache but needs a high-precision Babai reduction over many
//! primes to beat the tuned `√`-precision method; it is left as future work.

use alloc::vec::Vec;

use crate::float::{Float, RoundingMode};
use crate::int::Int;
use crate::rational::Rational;

const NEAR: RoundingMode = RoundingMode::Nearest;

/// A reusable cache of precomputed constants that accelerates repeated
/// [`exp`](Self::exp) calls (create once, reuse across calls at
/// the same or growing precision).
///
/// ```
/// # use puremp::{FloatContext, Float, RoundingMode};
/// let mut ctx = FloatContext::new();
/// let x = Float::from_int(&3i64.into(), 256, RoundingMode::Nearest);
/// let y = ctx.exp(&x, 256, RoundingMode::Nearest); // ≈ e³, faster on repeat calls
/// ```
#[derive(Clone, Default)]
pub struct FloatContext {
    /// `(prime, precision, ln(prime))`, one entry per prime (highest precision).
    logs: Vec<(u64, u64, Float)>,
}

impl FloatContext {
    /// Creates an empty context.
    pub fn new() -> FloatContext {
        FloatContext::default()
    }

    /// Returns `ln(p)` accurate to at least `w` bits, computing and caching it (or
    /// a higher-precision value) as needed.
    fn ln_prime(&mut self, p: u64, w: u64) -> Float {
        if let Some((_, _, f)) = self.logs.iter().find(|(pp, pw, _)| *pp == p && *pw >= w) {
            return f.clone();
        }
        let f = Float::from_int(&Int::from_u64(p), w, NEAR).ln(w, NEAR);
        self.logs.retain(|(pp, _, _)| *pp != p);
        self.logs.push((p, w, f.clone()));
        f
    }

    /// `e^x`, correctly rounded to `precision` bits.
    ///
    /// Reduces `x = k·ln2 + r` (with cached `ln2`) then sums the Taylor series for
    /// `exp(r/2ʲ)` in scaled integers and squares `j ≈ √precision` times.
    pub fn exp(&mut self, x: &Float, precision: u64, mode: RoundingMode) -> Float {
        if x.is_zero() {
            return Float::from_int(&Int::ONE, precision, mode);
        }
        // Bulk 2-exponent k first, so the cancellation guard tracks k's size.
        let w0 = precision + 64;
        let ln2_0 = self.ln_prime(2, w0);
        let k = round_to_int(&x.div(&ln2_0, w0, NEAR));
        let j = precision.isqrt().max(1);
        let n = precision + j + k.bit_len() as u64 + 16;

        let ln2 = self.ln_prime(2, n);
        let r = x.sub(&Float::from_int(&k, n, NEAR).mul(&ln2, n, NEAR), n, NEAR);
        // big_r = ⌊(r/2ʲ)·2ⁿ⌋ = ⌊r·2^(n−j)⌋.
        let big_r = scaled_trunc(&r, n as i64 - j as i64);
        let sum = exp_series_scaled(&big_r, n, j);

        // exp(x) = 2ᵏ · (sum / 2ⁿ) = sum · 2^(k−n).
        scaled_to_float(&sum, k.to_i64().unwrap_or(0) - n as i64, precision, mode)
    }
}

/// Evaluates `exp(R/2ⁿ)·2ⁿ` (with `R/2ⁿ` tiny) as a scaled integer, then undoes
/// `j` argument halvings by squaring.
fn exp_series_scaled(big_r: &Int, n: u64, j: u64) -> Int {
    let mut sum = Int::ONE.mul_2k(n as u32);
    let mut term = sum.clone();
    let mut kk = 1i64;
    loop {
        term = term
            .mul(big_r)
            .div_2k_trunc(n as u32)
            .div_trunc(&Int::from_i64(kk));
        if term.is_zero() {
            break;
        }
        sum = sum.add(&term);
        kk += 1;
    }
    for _ in 0..j {
        sum = sum.square().div_2k_trunc(n as u32);
    }
    sum
}

/// Nearest integer to a finite float (ties toward +∞).
fn round_to_int(f: &Float) -> Int {
    match f.to_rational() {
        Some(r) => round_scaled_rat(&r, 0),
        None => Int::ZERO,
    }
}

/// `⌊r·2ⁿ⌉` (round to nearest, ties toward +∞) for a rational `r`.
fn round_scaled_rat(r: &Rational, n: u64) -> Int {
    let num = r.numerator().mul_2k(n as u32);
    let den = r.denominator();
    let two = Int::from_i64(2);
    num.mul(&two).add(den).div_floor(&den.mul(&two))
}

/// `⌊f·2^m⌋`, truncated toward zero (`m` may be negative).
fn scaled_trunc(f: &Float, m: i64) -> Int {
    match f.to_rational() {
        Some(r) => {
            let (num, den) = (r.numerator(), r.denominator());
            if m >= 0 {
                num.mul_2k(m as u32).div_trunc(den)
            } else {
                num.div_trunc(&den.mul_2k((-m) as u32))
            }
        }
        None => Int::ZERO,
    }
}

/// Rounds `sum·2^exp2` to a `precision`-bit float.
fn scaled_to_float(sum: &Int, exp2: i64, precision: u64, mode: RoundingMode) -> Float {
    let val = if exp2 >= 0 {
        Rational::from_integer(sum.mul_2k(exp2 as u32))
    } else {
        Rational::new(sum.clone(), Int::ONE.mul_2k((-exp2) as u32))
    };
    Float::from_rational(&val, precision, mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    fn ff(s: &str) -> Float {
        Float::from_str(s).unwrap()
    }

    #[test]
    fn exp_matches_random() {
        // ctx.exp must agree bit-for-bit with the stateless Float::exp across many
        // random arguments and two precisions.
        let mut seed = 0x00E7_51EDu64;
        let mut ctx = FloatContext::new();
        for &prec in &[120u64, 400] {
            for _ in 0..200 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let whole = (seed >> 40) as i64 % 60 - 30;
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let frac = (seed >> 40) % 1_000_000;
                let x = Float::from_str(&alloc::format!("{whole}.{frac:06}")).unwrap();
                assert_eq!(
                    ctx.exp(&x, prec, NEAR),
                    x.exp(prec, NEAR),
                    "exp mismatch at {x:?}"
                );
            }
        }
    }

    #[test]
    fn exp_matches_float_exp() {
        let prec = 200u64;
        let mut ctx = FloatContext::new();
        for s in [
            "0.5", "-0.5", "1", "-1", "3.25", "-4.75", "12.5", "-0.001", "50",
        ] {
            assert_eq!(
                ctx.exp(&ff(s), prec, NEAR),
                ff(s).exp(prec, NEAR),
                "exp({s})"
            );
        }
    }
}
