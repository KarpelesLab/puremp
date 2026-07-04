//! Multi-prime argument reduction for `exp` (fast path).
//!
//! Reduces `x = k·ln2 + r`, then reduces the small `r` further by a combination
//! of prime logarithms `Σ eᵢ·ln pᵢ` found with an f64 Babai nearest-plane on the
//! precomputed LLL-reduced lattice ([`float_mp_consts`]). The recovered factor
//! `∏ pᵢ^eᵢ` is a cheap smooth integer, and the leftover `t` is tiny — so the
//! Taylor series needs *no* argument-halving squarings (the `√`-precision
//! method's dominant cost). Roughly 1.5–2× faster than [`Float::exp`] in the
//! ~1k–8k-bit range.
//!
//! Correctness is not assumed: the series truncation error is bounded, and the
//! result is only returned when both ends of that error interval round to the
//! same `precision`-bit float (otherwise `None` → the caller uses the trusted
//! `√`-precision path). Reference: F. Johansson, *Computing elementary functions
//! using multi-prime argument reduction* (2022).

use crate::float::{Float, RoundingMode};
use crate::float_mp_consts as c;
use crate::int::Int;
use crate::rational::Rational;

const NEAR: RoundingMode = RoundingMode::Nearest;
const PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// `exp(x)` via multi-prime reduction, correctly rounded — or `None` when out of
/// the supported band (precision or argument), so the caller falls back.
pub(crate) fn exp_mp(x: &Float, precision: u64, mode: RoundingMode) -> Option<Float> {
    // Worthwhile band; also keep within the embedded prime-log length.
    if precision < 384 || precision + 192 > c::PLOG_BITS {
        return None;
    }
    let n = precision + precision.isqrt() + 48; // working precision (guard bits)

    // Bulk reduction x = k·ln2 + r, |r| < ln2/2 (keeps the Babai target small so
    // f64 stays accurate even for large |x|).
    let ln2 = crate::float::ln2_embedded(n)?;
    let k = round_to_int(&x.div(&ln2, n, NEAR));
    let r = x.sub(&Float::from_int(&k, n, NEAR).mul(&ln2, n, NEAR), n, NEAR);

    // Fine reduction of r by the smooth reducers (f64 Babai on the constant reduced
    // basis — no prime logs needed here).
    let r_scaled = round_scaled(&r, c::MP_SCALE);
    let cf = babai_f64(&r_scaled)?;
    // Prime exponents e[j] = Σ cf[i]·BASIS[i][j].
    let mut e = [0i64; 12];
    for (i, &ci) in cf.iter().enumerate() {
        for (j, ej) in e.iter_mut().enumerate() {
            *ej += ci * c::MP_BASIS[i][j];
        }
    }
    // Guard: exponents must stay tiny (else f64 misbehaved) so ∏p^e is cheap.
    if e.iter().any(|&v| v.unsigned_abs() > 1 << 20) {
        return None;
    }

    // Exact residual t = r − Σ eⱼ·ln pⱼ, computing only the logs actually needed.
    const SIGS: [&[u64]; 11] = [
        &c::LN_3,
        &c::LN_5,
        &c::LN_7,
        &c::LN_11,
        &c::LN_13,
        &c::LN_17,
        &c::LN_19,
        &c::LN_23,
        &c::LN_29,
        &c::LN_31,
        &c::LN_37,
    ];
    let mut sum = Float::from_int(&Int::ZERO, n, NEAR);
    for (j, &ej) in e.iter().enumerate() {
        if ej == 0 {
            continue;
        }
        let l = if j == 0 {
            ln2.clone()
        } else {
            crate::float::round_const_bits(SIGS[j - 1], c::PLOG_BITS, n)
        };
        sum = sum.add(
            &Float::from_int(&Int::from_i64(ej), n, NEAR).mul(&l, n, NEAR),
            n,
            NEAR,
        );
    }
    let t = r.sub(&sum, n, NEAR);

    // exp(t)·2ⁿ as a scaled integer (no halvings — t is tiny), with a term count.
    let big_t = scaled_trunc(&t, n as i64);
    let (expt, terms) = exp_series(&big_t, n);

    // Smooth factor ∏_{i≥1} pᵢ^{eᵢ} = num/den; the 2-exponent folds k + e₀.
    let (mut num, mut den) = (Int::ONE, Int::ONE);
    for (j, &ej) in e.iter().enumerate().skip(1) {
        if ej > 0 {
            num = num.mul(&Int::from_u64(PRIMES[j]).pow(ej as u32));
        } else if ej < 0 {
            den = den.mul(&Int::from_u64(PRIMES[j]).pow((-ej) as u32));
        }
    }
    // value = (num/den)·(expt/2ⁿ)·2^{k+e₀}. Assemble num/den·sum then adjust 2^exp.
    let two_exp = k.to_i64().unwrap_or(0) + e[0] - n as i64;
    let scaled_num = expt.mul(&num);

    // Correct-rounding interval: the scaled sum is within ±E of the true value.
    // E covers the series truncation (< terms ulps) with margin, propagated
    // through the exact num/den multiply.
    let err = Int::from_i64(terms as i64 + 4).mul(&num);
    let lo = crate::float::round_ratio(&scaled_num.sub(&err), &den, two_exp, precision, mode);
    let hi = crate::float::round_ratio(&scaled_num.add(&err), &den, two_exp, precision, mode);
    (lo == hi).then_some(lo)
}

/// Round an `f64` to the nearest integer without `libm` (bare `no_std` has no
/// `f64::round`): `(x ± 0.5) as i64` truncates toward zero.
#[inline]
fn round_f64(x: f64) -> f64 {
    (x + if x >= 0.0 { 0.5 } else { -0.5 }) as i64 as f64
}

/// f64 Gram–Schmidt of the (constant) reduced basis, then Babai nearest-plane for
/// the target `(0,…,0, r_scaled)`. Returns the integer coefficients, or `None` if
/// `r_scaled` is too large for f64 to place accurately.
#[allow(clippy::needless_range_loop)] // the index drives bstar[i][t] / b[j][t] / bv[t] together
fn babai_f64(r_scaled: &Int) -> Option<[i64; 12]> {
    let dim = 13;
    let rt = r_scaled.to_i64()? as f64; // r is small; must fit f64 exactly-ish
    let b: [[f64; 13]; 12] =
        core::array::from_fn(|i| core::array::from_fn(|j| c::MP_BASIS[i][j] as f64));
    // Gram–Schmidt.
    let mut bstar = b;
    let dotf = |a: &[f64; 13], c: &[f64; 13]| a.iter().zip(c).map(|(x, y)| x * y).sum::<f64>();
    for i in 0..12 {
        for j in 0..i {
            let mu = dotf(&b[i], &bstar[j]) / dotf(&bstar[j], &bstar[j]);
            for t in 0..dim {
                bstar[i][t] -= mu * bstar[j][t];
            }
        }
    }
    let norm2: [f64; 12] = core::array::from_fn(|i| dotf(&bstar[i], &bstar[i]));
    // Nearest-plane on target = [0,…,0, rt].
    let mut bv = [0.0f64; 13];
    bv[12] = rt;
    let mut coeffs = [0i64; 12];
    for i in (0..12).rev() {
        let ci = round_f64(dotf(&bv, &bstar[i]) / norm2[i]);
        for t in 0..dim {
            bv[t] -= ci * b[i][t];
        }
        coeffs[i] = ci as i64;
    }
    Some(coeffs)
}

/// `exp(T/2ⁿ)·2ⁿ` (T/2ⁿ tiny) as a scaled integer, plus the number of terms.
fn exp_series(big_t: &Int, n: u64) -> (Int, u64) {
    let mut sum = Int::ONE.mul_2k(n as u32);
    let mut term = sum.clone();
    let mut kk = 1i64;
    loop {
        term = term
            .mul(big_t)
            .div_2k_trunc(n as u32)
            .div_trunc(&Int::from_i64(kk));
        if term.is_zero() {
            break;
        }
        sum = sum.add(&term);
        kk += 1;
    }
    (sum, kk as u64)
}

fn round_to_int(f: &Float) -> Int {
    match f.to_rational() {
        Some(r) => round_scaled_rat(&r, 0),
        None => Int::ZERO,
    }
}

fn round_scaled(f: &Float, s: u64) -> Int {
    match f.to_rational() {
        Some(r) => round_scaled_rat(&r, s),
        None => Int::ZERO,
    }
}

fn round_scaled_rat(r: &Rational, s: u64) -> Int {
    let num = r.numerator().mul_2k(s as u32);
    let den = r.denominator();
    let two = Int::from_i64(2);
    num.mul(&two).add(den).div_floor(&den.mul(&two))
}

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
