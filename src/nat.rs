//! Arbitrary-precision natural numbers (unsigned integers).
//!
//! [`Nat`] stores magnitude as a little-endian `Vec` of 64-bit limbs,
//! kept in a canonical form with no trailing zero limbs (so the value zero is
//! the empty vector). That canonical form makes equality and ordering cheap and
//! lets the derived [`PartialEq`]/[`Eq`] be correct.
//!
//! This is the layer that carries the heavy limb-level algorithms: addition,
//! subtraction, multiplication (schoolbook → Karatsuba → Toom-3 → NTT),
//! squaring, division (single-limb, Knuth Algorithm D, and Burnikel–Ziegler),
//! shifts, GCD (binary → Lehmer), roots, and sub-quadratic radix I/O.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::limb::{LIMB_BITS, Limb, adc, mac, sbb};

// Multiplication crossovers, tuned from `measure_mul_crossovers` (see the test
// module) with the addmul_2 schoolbook loop. The faster basecase pushes every
// crossover up: Karatsuba from ~128 limbs, Toom-3 from ~1.4k, Toom-4 from ~6k.
// Even with the division-free Goldilocks reduction, the single-prime NTT's
// power-of-two transform steps keep it slower than the smoothly-scaling Toom
// ladder until ~11k limbs. Re-measure per platform to retune.

/// Operands with fewer than this many limbs use schoolbook multiplication.
const KARATSUBA_THRESHOLD: usize = 128;

/// Operands with at least this many limbs use Toom-3 (above Karatsuba).
const TOOM3_THRESHOLD: usize = 1400;

/// Operands with at least this many limbs use Toom-4 (above Toom-3).
const TOOM4_THRESHOLD: usize = 6000;

/// GCD switches from Stein's binary algorithm to Lehmer's above this many limbs.
const LEHMER_THRESHOLD: usize = 16;

/// Operands with at least this many limbs use NTT multiplication (above Toom-4).
const NTT_THRESHOLD: usize = 11000;

// --- Number-theoretic transform over the Goldilocks field 2^64 − 2^32 + 1 ---
//
// This prime has `p − 1 = 2^32·(2^32 − 1)`, so it supports NTTs of any power-of-
// two length up to 2^32, and 7 is a primitive root. Modular reduction is
// division-free, exploiting `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1 (mod p)`.

/// The Goldilocks prime `2^64 − 2^32 + 1`.
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;
/// A primitive root of the Goldilocks multiplicative group.
const GOLDILOCKS_ROOT: u64 = 7;
/// `2^64 mod p = 2^32 − 1`.
const GF_EPSILON: u128 = 0xFFFF_FFFF;

/// Reduces a 128-bit value modulo the Goldilocks prime without any division,
/// using `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1 (mod p)`. Returns a canonical result
/// in `[0, p)`.
#[inline]
fn gf_reduce128(x: u128) -> u64 {
    let lo = (x as u64) as u128;
    let hi = (x >> 64) as u64;
    let hi_hi = (hi >> 32) as u128; // top 32 bits contribute ·2^96 ≡ −1
    let hi_lo = (hi & 0xFFFF_FFFF) as u128; // next 32 bits contribute ·2^64 ≡ ε
    // acc ≡ x (mod p); adding one p keeps the `− hi_hi` non-negative. acc < 2^66.
    let acc = lo + hi_lo * GF_EPSILON + GOLDILOCKS as u128 - hi_hi;
    // Fold the ≤ 2 high bits back in (value·2^64 ≡ value·ε). folded < 2^64 + 2^34.
    let folded = (acc & u64::MAX as u128) + (acc >> 64) * GF_EPSILON;
    let mut r = folded as u64;
    if (folded >> 64) != 0 {
        // One more 2^64 to fold; `s + ε` cannot overflow (s < ε here).
        let (s, c) = r.overflowing_add(GF_EPSILON as u64);
        r = if c { s + GF_EPSILON as u64 } else { s };
    }
    if r >= GOLDILOCKS { r - GOLDILOCKS } else { r }
}

#[inline]
fn gf_mul(a: u64, b: u64) -> u64 {
    gf_reduce128(a as u128 * b as u128)
}

#[inline]
fn gf_add(a: u64, b: u64) -> u64 {
    let s = a as u128 + b as u128;
    (if s >= GOLDILOCKS as u128 {
        s - GOLDILOCKS as u128
    } else {
        s
    }) as u64
}

#[inline]
fn gf_sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        (a as u128 + GOLDILOCKS as u128 - b as u128) as u64
    }
}

fn gf_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut r = 1u64;
    base %= GOLDILOCKS;
    while exp > 0 {
        if exp & 1 == 1 {
            r = gf_mul(r, base);
        }
        base = gf_mul(base, base);
        exp >>= 1;
    }
    r
}

/// In-place iterative NTT (or its inverse) over the Goldilocks field.
fn ntt(a: &mut [u64], inverse: bool) {
    let n = a.len();
    // Bit-reversal permutation.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    // Twiddle scratch, reused across stages (largest stage needs n/2 entries).
    let mut tw: Vec<u64> = Vec::with_capacity(n / 2);
    let mut len = 2;
    while len <= n {
        let mut wlen = gf_pow(GOLDILOCKS_ROOT, (GOLDILOCKS - 1) / len as u64);
        if inverse {
            wlen = gf_pow(wlen, GOLDILOCKS - 2);
        }
        let half = len / 2;
        // Precompute this stage's twiddles once, then reuse across every block.
        tw.clear();
        let mut w = 1u64;
        for _ in 0..half {
            tw.push(w);
            w = gf_mul(w, wlen);
        }
        let mut i = 0;
        while i < n {
            for k in 0..half {
                let u = a[i + k];
                let v = gf_mul(a[i + k + half], tw[k]);
                a[i + k] = gf_add(u, v);
                a[i + k + half] = gf_sub(u, v);
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let n_inv = gf_pow(n as u64, GOLDILOCKS - 2);
        for x in a.iter_mut() {
            *x = gf_mul(*x, n_inv);
        }
    }
}

/// Splits `x` into little-endian digits of `bpd` bytes each (at least one).
fn to_digits(x: &Nat, bpd: usize) -> Vec<u64> {
    let bytes = x.to_bytes_le();
    let mut d = Vec::with_capacity(bytes.len() / bpd + 1);
    for chunk in bytes.chunks(bpd) {
        let mut digit = 0u64;
        for (i, &b) in chunk.iter().enumerate() {
            digit |= (b as u64) << (8 * i);
        }
        d.push(digit);
    }
    if d.is_empty() {
        d.push(0);
    }
    d
}

/// NTT-based multiplication over a single Goldilocks prime.
///
/// The digit width adapts to the operand size so the convolution coefficients
/// (`≈ n · 2^(16·bpd)`) always stay below the prime: 2 bytes/digit for typical
/// inputs, shrinking to 1 (then falling back to Toom-4 only for astronomically
/// large operands), so no multi-prime CRT is needed in practice.
fn mul_ntt(a: &Nat, b: &Nat) -> Nat {
    // Rough transform length with 2-byte digits (4 per limb).
    let approx = (a.limbs.len() + b.limbs.len()) * 4 + 2;
    let mut est = 1usize;
    while est < approx {
        est <<= 1;
    }
    // Pick the largest digit width whose coefficients stay below the prime.
    let bpd = if (est as u128) << 32 < GOLDILOCKS as u128 {
        2
    } else {
        1
    };

    let da = to_digits(a, bpd);
    let db = to_digits(b, bpd);
    let need = da.len() + db.len();
    let mut n = 1usize;
    while n < need {
        n <<= 1;
    }
    // Coefficient bound: n · (2^(8·bpd))². Fall back only if even 1-byte digits
    // would overflow (operands beyond ~2^51 bits).
    if (n as u128) << (16 * bpd as u32) >= GOLDILOCKS as u128 {
        return a.mul_toom4(b);
    }

    let mut fa = alloc::vec![0u64; n];
    let mut fb = alloc::vec![0u64; n];
    fa[..da.len()].copy_from_slice(&da);
    fb[..db.len()].copy_from_slice(&db);
    ntt(&mut fa, false);
    ntt(&mut fb, false);
    for (x, y) in fa.iter_mut().zip(&fb) {
        *x = gf_mul(*x, *y);
    }
    ntt(&mut fa, true);

    // Carry-propagate the coefficients in base 2^(8·bpd).
    let mut bytes: Vec<u8> = Vec::with_capacity(bpd * n + 8);
    let mut carry: u128 = 0;
    for &coef in &fa {
        carry += coef as u128;
        for _ in 0..bpd {
            bytes.push((carry & 0xFF) as u8);
            carry >>= 8;
        }
    }
    while carry != 0 {
        bytes.push((carry & 0xFF) as u8);
        carry >>= 8;
    }
    Nat::from_bytes_le(&bytes)
}

/// Divisors with at least this many limbs use Burnikel–Ziegler recursive
/// division; smaller ones use Knuth Algorithm D directly.
const BZ_THRESHOLD: usize = 256;

/// Recursion base case (in half-block limbs) for Burnikel–Ziegler, tuned with
/// the addmul_2 basecase (larger Knuth blocks now beat recursion overhead).
const BZ_BASE: usize = 96;

/// Extracts block `i` (limbs `[i·n, (i+1)·n)`) of `x` as a [`Nat`].
fn bz_block(x: &Nat, i: usize, n: usize) -> Nat {
    let lo = i * n;
    let l = x.limbs.len();
    if lo >= l {
        Nat::zero()
    } else {
        Nat::from_limbs(&x.limbs[lo..(lo + n).min(l)])
    }
}

/// Burnikel–Ziegler top level: normalize the divisor, then process the dividend
/// in `n`-limb blocks from the top, dividing each `≤ 2n`-limb window. Requires
/// `a > b` and `b.limbs.len() >= 2`.
fn bz_div_rem(a: &Nat, b: &Nat) -> (Nat, Nat) {
    let n = b.limbs.len();
    // Use a power-of-two block size so every recursive split stays even and the
    // recursion never bails to quadratic Knuth. Padding the divisor to `n2`
    // limbs by an extra left shift is value-preserving: shifting both operands
    // left by the same amount leaves the quotient unchanged and only scales the
    // remainder, which the final `shr` undoes.
    let n2 = n.next_power_of_two();
    let s = b.limbs[n - 1].leading_zeros() as u64;
    let shift = s + (n2 - n) as u64 * LIMB_BITS as u64;
    let bn = b.shl(shift); // exactly n2 limbs, top bit set
    let an = a.shl(shift);
    let nbits = n2 as u64 * LIMB_BITS as u64;
    let t = an.limbs.len().div_ceil(n2).max(2);

    let mut r = Nat::zero();
    let mut parts: Vec<Nat> = Vec::with_capacity(t);
    for i in (0..t).rev() {
        let cur = r.shl(nbits).add(&bz_block(&an, i, n2));
        let (qi, ri) = bz_div_2n_1n(&cur, &bn, n2);
        parts.push(qi);
        r = ri;
    }
    let mut q = Nat::zero();
    for (j, part) in parts.into_iter().enumerate() {
        q = q.add(&part.shl((t - 1 - j) as u64 * nbits));
    }
    (q, r.shr(shift))
}

/// Divide a `≤ 2n`-limb value by the `n`-limb normalized divisor `b`
/// (`quotient < 2^(64n)`).
fn bz_div_2n_1n(a: &Nat, b: &Nat, n: usize) -> (Nat, Nat) {
    if a.cmp_ref(b) == Ordering::Less {
        return (Nat::zero(), a.clone());
    }
    if n < BZ_BASE || n % 2 == 1 {
        if a.cmp_ref(b) == Ordering::Equal {
            return (Nat::one(), Nat::zero());
        }
        if b.limbs.len() == 1 {
            let (q, rr) = a.divmod_small(b.limbs[0]);
            return (q, Nat::from_u64(rr));
        }
        return a.div_rem_knuth(b);
    }
    let half = n / 2;
    let hbits = half as u64 * LIMB_BITS as u64;
    let (q1, r1) = bz_div_3n_2n(&a.shr(hbits), b, half);
    let (q2, r2) = bz_div_3n_2n(&r1.shl(hbits).add(&a.low_bits(hbits)), b, half);
    (q1.shl(hbits).add(&q2), r2)
}

/// Divide a `≤ 3·half`-limb value by the `2·half`-limb normalized divisor `b`.
fn bz_div_3n_2n(a: &Nat, b: &Nat, half: usize) -> (Nat, Nat) {
    use crate::int::Int;
    let hbits = half as u64 * LIMB_BITS as u64;
    let b1 = b.shr(hbits);
    let b2 = b.low_bits(hbits);
    let a12 = a.shr(hbits);
    let a3 = a.low_bits(hbits);

    let (q_nat, r_pre): (Nat, Int) = if a12.shr(hbits).cmp_ref(&b1) == Ordering::Less {
        let (q, r) = bz_div_2n_1n(&a12, &b1, half);
        (q, Int::from(r))
    } else {
        // q = 2^(64·half) − 1; R = A12 − q·B1.
        let q = Nat::one()
            .shl(hbits)
            .checked_sub(&Nat::one())
            .expect("2^k >= 1");
        let r = Int::from(a12).sub(&Int::from(q.mul(&b1)));
        (q, r)
    };

    // R = R·2^(64·half) + A3 − q·B2, corrected to be non-negative.
    let mut r_int = r_pre
        .mul_2k(hbits as u32)
        .add(&Int::from(a3))
        .sub(&Int::from(q_nat.mul(&b2)));
    let mut q_int = Int::from(q_nat);
    // The estimate is at most two too large, so this correction almost never
    // runs — only clone the (full-width) divisor into an `Int` when it does.
    if r_int.is_negative() {
        let b_int = Int::from(b.clone());
        while r_int.is_negative() {
            q_int = q_int.sub(&Int::ONE);
            r_int = r_int.add(&b_int);
        }
    }
    (q_int.magnitude(), r_int.magnitude())
}

/// Recursive square root with remainder (Zimmermann's "Karatsuba square root"):
/// returns `(s, r)` with `s = ⌊√a⌋` and `r = a − s²`.
///
/// Precondition: `a` is normalized so its top limb is `≥ 2^62` (ensured by an
/// even shift in [`Nat::isqrt`]). With `l = ⌊n/4⌋` that guarantees the root of
/// the high part satisfies `2·s₁ ≥ 2^(64l)`, so the quotient below fits `l`
/// limbs (+1) and the correction loops run O(1) times.
fn sqrt_rem(a: &Nat) -> (Nat, Nat) {
    let n = a.limbs.len();
    if n <= 2 {
        let v = a.to_u128().expect("<= 2 limbs");
        let s = isqrt_u128(v);
        return (Nat::from_u128(s), Nat::from_u128(v - s * s));
    }
    if n == 3 {
        // Too small for an l ≥ 1, n ≥ 4l split: one guarded Newton step from a
        // 128-bit seed, then exact ±1 adjustment (cheap at this size).
        let b = a.bit_len();
        let c = b / 4;
        let seed = a.shr(2 * c).to_u128().expect("~b/2 <= 96 bits");
        let s0 = Nat::from_u128(isqrt_u128(seed)).shl(c);
        let q = a.div_rem(&s0).expect("s0 > 0").0;
        let mut x = s0.add(&q).shr(1);
        while x.square().cmp_ref(a) == Ordering::Greater {
            x = x.checked_sub(&Nat::one()).expect("x >= 1");
        }
        loop {
            let x1 = x.add(&Nat::one());
            if x1.square().cmp_ref(a) != Ordering::Greater {
                x = x1;
            } else {
                break;
            }
        }
        let r = a.checked_sub(&x.square()).expect("x = floor(sqrt(a))");
        return (x, r);
    }

    // a = H·B^(2l) + a₁·B^l + a₀ with B = 2^64; recurse on the (normalized)
    // high part, then one division computes the low half of the root.
    let l = n / 4;
    let lbits = l as u64 * LIMB_BITS as u64;
    let high = Nat::from_limbs(&a.limbs[2 * l..]);
    let (s1, r1) = sqrt_rem(&high);
    let a1 = Nat::from_limbs(&a.limbs[l..2 * l]);
    let a0 = Nat::from_limbs(&a.limbs[..l]);
    // (q, u) = (r₁·B^l + a₁) divmod 2·s₁; root estimate s = s₁·B^l + q.
    let (q, u) = r1.shl(lbits).add(&a1).div_rem(&s1.shl(1)).expect("s1 > 0");
    let mut s = s1.shl(lbits).add(&q);
    // r = u·B^l + a₀ − q², fixed up so 0 ≤ r ≤ 2s. Both loops run O(1) times
    // and cost only additions of half-size values — never full-width squarings.
    let t = u.shl(lbits).add(&a0);
    let q2 = q.square();
    let one = Nat::one();
    match t.checked_sub(&q2) {
        Some(mut r) => {
            // Rarely the estimate is one too small: while r > 2s, step s up
            // (r −= 2s+1 with the old s). Exiting at the first r ≤ 2s makes s
            // the floor, since r = a − s² is maintained exactly.
            loop {
                let d = s.shl(1);
                if r.cmp_ref(&d) != Ordering::Greater {
                    return (s, r);
                }
                r = r
                    .checked_sub(&d)
                    .and_then(|x| x.checked_sub(&one))
                    .expect("r > 2s in the up-adjustment");
                s = s.add(&one);
            }
        }
        None => {
            // Estimate too large (r < 0): while r < 0, step s down (r += 2s−1
            // with the old s), tracked as the positive deficit q² − t.
            let mut deficit = q2.checked_sub(&t).expect("t < q2");
            loop {
                let d = s
                    .shl(1)
                    .checked_sub(&one)
                    .expect("s >= 1 in the down-adjustment"); // 2s − 1
                s = s.checked_sub(&one).expect("s >= 1");
                match d.checked_sub(&deficit) {
                    Some(r) => return (s, r),
                    None => {
                        deficit = deficit.checked_sub(&d).expect("still negative");
                    }
                }
            }
        }
    }
}

/// Integer square root of a `u128` (base case of [`Nat::isqrt`]).
fn isqrt_u128(v: u128) -> u128 {
    if v == 0 {
        return 0;
    }
    let bits = 128 - v.leading_zeros();
    // Seed ≥ √v; Newton from above descends to ⌊√v⌋. `x + v/x` stays near
    // `2·√v ≤ 2^65`, so it never overflows.
    let mut x = 1u128 << bits.div_ceil(2);
    loop {
        let y = (x + v / x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Left-to-right fixed-window modular exponentiation in an abstract domain.
///
/// `base` and `one` are already in the working domain (e.g. Montgomery form),
/// and `mulmod` multiplies within it. Processing the exponent `w` bits at a time
/// replaces most of the per-bit multiplications with a small precomputed table,
/// cutting the multiply count relative to binary square-and-multiply while the
/// squaring count is unchanged. (Squaring stays fast because `Nat::mul`
/// dispatches equal operands to `square`.)
fn modpow_windowed(base: Nat, one: Nat, exp: &Nat, mulmod: impl Fn(&Nat, &Nat) -> Nat) -> Nat {
    let bits = exp.bit_len();
    if bits == 0 {
        return one; // exp == 0
    }
    // Window width scaled to the exponent size (table costs 2^w multiplies).
    let w: u64 = match bits {
        0..=32 => 2,
        33..=128 => 3,
        129..=512 => 4,
        513..=2048 => 5,
        _ => 6,
    };
    // Precompute base^0 .. base^(2^w − 1).
    let size = 1usize << w;
    let mut table = Vec::with_capacity(size);
    table.push(one);
    table.push(base.clone());
    for i in 2..size {
        table.push(mulmod(&table[i - 1], &base));
    }

    let mut result: Option<Nat> = None;
    let mut idx = bits;
    while idx > 0 {
        let take = idx.min(w);
        let shift = idx - take;
        let mut window = 0usize;
        for j in 0..take {
            if exp.bit(shift + j) {
                window |= 1 << j;
            }
        }
        result = Some(match result {
            None => table[window].clone(), // first (top) window: no squaring yet
            Some(mut r) => {
                for _ in 0..take {
                    r = mulmod(&r, &r); // square `take` times
                }
                if window != 0 {
                    r = mulmod(&r, &table[window]);
                }
                r
            }
        });
        idx = shift;
    }
    result.expect("bits > 0 guarantees at least one window")
}

/// Recombines Toom coefficients `Σ cᵢ·2^(64·k·i)` into a single [`Nat`], writing
/// each (non-negative) coefficient at whole-limb offset `k·i` with carry
/// propagation. Cheaper than a chain of `mul_2k`/`add`.
fn recombine_coeffs(product_limbs: usize, k: usize, coeffs: &[crate::int::Int]) -> Nat {
    let mut out = alloc::vec![0 as Limb; product_limbs + 2];
    for (i, c) in coeffs.iter().enumerate() {
        debug_assert!(!c.is_negative(), "toom coefficient is negative");
        let mag = c.magnitude();
        if !mag.is_zero() {
            add_at(&mut out, i * k, mag.as_limbs());
        }
    }
    let mut n = Nat { limbs: out };
    n.normalize();
    n
}

/// Modular inverse of an odd `x` modulo `2^64`, by Newton's iteration
/// (`y ← y·(2 − x·y)`), which doubles the number of correct low bits each step.
#[inline]
fn inv_mod_2_64(x: Limb) -> Limb {
    debug_assert!(x & 1 == 1, "inverse mod 2^64 requires an odd input");
    // Seed correct to 5 bits, then 5 → 10 → 20 → 40 → 80 (≥ 64) bits.
    let mut y = x.wrapping_mul(3) ^ 2;
    for _ in 0..4 {
        y = y.wrapping_mul(2u64.wrapping_sub(x.wrapping_mul(y)));
    }
    y
}

/// Montgomery multiplication by the CIOS (Coarsely Integrated Operand Scanning)
/// method: returns `a·b·R⁻¹ mod m` in `[0, m)`, where `R = 2^(64·m.len())` and
/// `n0inv = −m⁻¹ mod 2^64`. Multiply and reduction are interleaved word-by-word
/// through a single `s+2`-word accumulator, so there is no full-width product or
/// per-step allocation. Requires an odd modulus with a non-zero top limb, and
/// `a, b < m` (shorter inputs are zero-extended).
#[allow(clippy::needless_range_loop)] // index drives t[j], t[j-1] and m[j] together
fn mont_mul_cios(a: &Nat, b: &Nat, m: &[Limb], n0inv: Limb) -> Nat {
    let s = m.len();
    // Zero-pad `b` to exactly s limbs once, so the hot inner loop is a plain
    // bounds-check-free zip instead of a checked `get` per limb product.
    let mut bb: Vec<Limb> = Vec::with_capacity(s);
    bb.extend_from_slice(&b.limbs);
    bb.resize(s, 0);
    let mut t = alloc::vec![0u64; s + 2];
    for i in 0..s {
        // t += a[i]·b
        let ai = a.limbs.get(i).copied().unwrap_or(0) as u128;
        let mut carry: u128 = 0;
        for (tj, &bj) in t[..s].iter_mut().zip(&bb) {
            let sum = *tj as u128 + ai * bj as u128 + carry;
            *tj = sum as Limb;
            carry = sum >> LIMB_BITS;
        }
        let sum = t[s] as u128 + carry;
        t[s] = sum as Limb;
        t[s + 1] = (sum >> LIMB_BITS) as Limb;

        // Reduce one word: t += (t[0]·n0inv)·m, then shift right by one word.
        let mi = t[0].wrapping_mul(n0inv) as u128;
        let mut carry = (t[0] as u128 + mi * m[0] as u128) >> LIMB_BITS; // low word → 0
        for j in 1..s {
            let sum = t[j] as u128 + mi * m[j] as u128 + carry;
            t[j - 1] = sum as Limb;
            carry = sum >> LIMB_BITS;
        }
        let sum = t[s] as u128 + carry;
        t[s - 1] = sum as Limb;
        t[s] = t[s + 1].wrapping_add((sum >> LIMB_BITS) as Limb);
    }
    // t holds an `s+1`-word value < 2m; one conditional subtraction canonicalizes.
    let mut result = Nat {
        limbs: t[..=s].to_vec(),
    };
    result.normalize();
    let m_nat = Nat { limbs: m.to_vec() };
    if result.cmp_ref(&m_nat) != Ordering::Less {
        result = result.checked_sub(&m_nat).expect("result < 2m");
    }
    result
}

/// Schoolbook squaring of `a` into the zeroed buffer `t` (`t.len() >= 2·a.len()`):
/// accumulate the strictly-upper triangle of cross products, then double it and
/// add the diagonal `aᵢ²` terms in one fused carry pass.
///
/// Triangle rows are processed two at a time (the same `addmul_2` shape as
/// [`Nat::mul_schoolbook`]): rows `i` and `i+1` share the tail `a[i+2..]`, so
/// `t += (aᵢ + aᵢ₊₁·2^64)·a[i+2..]` plus the lone `aᵢ·aᵢ₊₁` product.
fn sqr_into(a: &[Limb], t: &mut [Limb]) {
    use crate::limb::DLimb;
    let n = a.len();
    if n == 0 {
        return;
    }
    let mut i = 0;
    while i + 2 <= n {
        let (a0, a1) = (a[i], a[i + 1]);
        // The lone cross product aᵢ·aᵢ₊₁ sits at position 2i+1.
        let p = a0 as DLimb * a1 as DLimb;
        add_at(t, 2 * i + 1, &[p as Limb, (p >> LIMB_BITS) as Limb]);
        // Both rows' shared tail a[i+2..] lands at base position 2i+2.
        let b = &a[i + 2..];
        let rn = b.len();
        let mut ph0: Limb = 0;
        let mut pl1: Limb = 0;
        let mut ph1: Limb = 0;
        let mut ph1p: Limb = 0;
        let mut carry: Limb = 0;
        let row = &mut t[2 * i + 2..i + n + 2];
        for (o, &bj) in row.iter_mut().zip(b) {
            let p0 = a0 as DLimb * bj as DLimb;
            let p1 = a1 as DLimb * bj as DLimb;
            let acc = *o as DLimb
                + (p0 as Limb) as DLimb
                + ph0 as DLimb
                + pl1 as DLimb
                + ph1p as DLimb
                + carry as DLimb;
            *o = acc as Limb;
            carry = (acc >> LIMB_BITS) as Limb;
            ph0 = (p0 >> LIMB_BITS) as Limb;
            ph1p = ph1;
            pl1 = p1 as Limb;
            ph1 = (p1 >> LIMB_BITS) as Limb;
        }
        if rn > 0 {
            let acc =
                row[rn] as DLimb + ph0 as DLimb + pl1 as DLimb + ph1p as DLimb + carry as DLimb;
            row[rn] = acc as Limb;
            let top = row[rn + 1] as DLimb + ph1 as DLimb + (acc >> LIMB_BITS);
            row[rn + 1] = top as Limb;
            debug_assert_eq!(top >> LIMB_BITS, 0, "square top carry escaped");
        }
        i += 2;
    }
    // Fused finish: t[k] = 2·cross[k] + diagonal, one carry pass over 2n limbs.
    let mut hi_bit: Limb = 0;
    let mut carry: DLimb = 0;
    for (k, tk) in t[..2 * n].iter_mut().enumerate() {
        let c = *tk;
        let doubled = (c << 1) | hi_bit;
        hi_bit = c >> (LIMB_BITS - 1);
        // Diagonal aᵢ² contributes its low limb at 2i, high limb at 2i+1.
        let ai = a[k / 2];
        let sq = ai as DLimb * ai as DLimb;
        let add = if k & 1 == 0 {
            sq as Limb
        } else {
            (sq >> LIMB_BITS) as Limb
        };
        let sum = doubled as DLimb + add as DLimb + carry;
        *tk = sum as Limb;
        carry = sum >> LIMB_BITS;
    }
    debug_assert_eq!(carry, 0, "square carry escaped the buffer");
    debug_assert_eq!(hi_bit, 0, "square doubling bit escaped the buffer");
}

/// Montgomery squaring: returns `a²·R⁻¹ mod m` in `[0, m)` (same contract as
/// [`mont_mul_cios`] with both operands equal). Computes the full double-width
/// square with the fast symmetric squaring, then reduces it in place with `s`
/// word-level REDC steps — about 25% fewer limb multiplications than the
/// general interleaved CIOS product.
fn mont_sqr(a: &Nat, m: &[Limb], n0inv: Limb) -> Nat {
    let s = m.len();
    let mut t = alloc::vec![0 as Limb; 2 * s + 1];
    if a.limbs.len() < KARATSUBA_THRESHOLD {
        sqr_into(&a.limbs, &mut t);
    } else {
        let sq = a.square();
        t[..sq.limbs.len()].copy_from_slice(&sq.limbs);
    }
    for i in 0..s {
        // Zero word i: t += (t[i]·n0inv)·m·B^i.
        let mi = t[i].wrapping_mul(n0inv) as u128;
        let mut carry: u128 = 0;
        let (row, rest) = t[i..].split_at_mut(s);
        for (tj, &mj) in row.iter_mut().zip(m) {
            let sum = *tj as u128 + mi * mj as u128 + carry;
            *tj = sum as Limb;
            carry = sum >> LIMB_BITS;
        }
        for tj in rest.iter_mut() {
            if carry == 0 {
                break;
            }
            let sum = *tj as u128 + carry;
            *tj = sum as Limb;
            carry = sum >> LIMB_BITS;
        }
        debug_assert_eq!(carry, 0, "REDC carry escaped the buffer");
    }
    // t / B^s is an (s+1)-word value < 2m; one conditional subtraction.
    let mut result = Nat {
        limbs: t[s..].to_vec(),
    };
    result.normalize();
    let m_nat = Nat { limbs: m.to_vec() };
    if result.cmp_ref(&m_nat) != Ordering::Less {
        result = result.checked_sub(&m_nat).expect("result < 2m");
    }
    result
}

/// Adds the limbs of `val` into `out` starting at limb `offset`, propagating the
/// carry. `out` must be large enough to hold the result (including any carry).
#[inline]
fn add_at(out: &mut [Limb], offset: usize, val: &[Limb]) {
    let mut carry = 0u128;
    let dst = &mut out[offset..offset + val.len()];
    for (o, &v) in dst.iter_mut().zip(val) {
        let s = *o as u128 + v as u128 + carry;
        *o = s as Limb;
        carry = s >> LIMB_BITS;
    }
    let mut i = offset + val.len();
    while carry != 0 {
        let s = out[i] as u128 + carry;
        out[i] = s as Limb;
        carry = s >> LIMB_BITS;
        i += 1;
    }
}

/// Computes `a·u + b·v` (guaranteed non-negative by the Lehmer invariant) on
/// naturals with explicit signs, in one fused pass with a signed 128-bit
/// carry — no intermediate products or sign case allocations.
/// Requires `|a|, |b| < 2^63` (they are i64 cofactors at every call site).
fn lincomb_pos(a: i128, u: &Nat, b: i128, v: &Nat) -> Nat {
    let (am, bm) = (a.unsigned_abs() as u64, b.unsigned_abs() as u64);
    debug_assert!(a.unsigned_abs() >> 63 == 0 && b.unsigned_abs() >> 63 == 0);
    let n = u.limbs.len().max(v.limbs.len()) + 2;
    let ul = |i: usize| u.limbs.get(i).copied().unwrap_or(0);
    let vl = |i: usize| v.limbs.get(i).copied().unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    if a >= 0 && b >= 0 {
        // Both terms add: unsigned accumulate (products < 2^127 each).
        let mut carry: u128 = 0;
        for i in 0..n {
            let acc = am as u128 * ul(i) as u128 + bm as u128 * vl(i) as u128 + carry;
            out.push(acc as Limb);
            carry = acc >> LIMB_BITS;
        }
        debug_assert_eq!(carry, 0, "lincomb carry escaped");
    } else {
        // Exactly one term is negative; orient as m₁·w₁ − m₂·w₂ ≥ 0 and let a
        // signed carry ripple (|product| < 2^127, so the i128 never overflows).
        let (m1, m2, flip) = if a >= 0 {
            (am, bm, false)
        } else {
            (bm, am, true)
        };
        let mut carry: i128 = 0;
        for i in 0..n {
            let (w1, w2) = if flip { (vl(i), ul(i)) } else { (ul(i), vl(i)) };
            let p = m1 as i128 * w1 as i128 - m2 as i128 * w2 as i128 + carry;
            out.push(p as Limb);
            carry = p >> LIMB_BITS; // arithmetic: sign-extends the borrow
        }
        assert_eq!(
            carry, 0,
            "lincomb result is non-negative by the Lehmer invariant"
        );
    }
    let mut r = Nat { limbs: out };
    r.normalize();
    r
}

/// An arbitrary-precision natural number (a non-negative integer).
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Nat {
    /// Little-endian limbs, normalized so the most-significant limb is non-zero.
    /// The value zero is represented by an empty vector.
    limbs: Vec<Limb>,
}

impl Nat {
    /// Returns the natural number zero.
    #[inline]
    pub fn zero() -> Self {
        Nat { limbs: Vec::new() }
    }

    /// Returns the natural number one.
    #[inline]
    pub fn one() -> Self {
        Nat::from_u64(1)
    }

    /// Builds a [`Nat`] from a `u64`.
    #[inline]
    pub fn from_u64(v: u64) -> Self {
        let mut n = Nat {
            limbs: if v == 0 { Vec::new() } else { alloc::vec![v] },
        };
        n.normalize();
        n
    }

    /// Builds a [`Nat`] from a `u128`.
    pub fn from_u128(v: u128) -> Self {
        let lo = v as Limb;
        let hi = (v >> LIMB_BITS) as Limb;
        let mut n = Nat {
            limbs: alloc::vec![lo, hi],
        };
        n.normalize();
        n
    }

    /// Returns `true` if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Returns `true` if this value is even (including zero).
    #[inline]
    pub fn is_even(&self) -> bool {
        self.limbs.first().is_none_or(|&l| l & 1 == 0)
    }

    /// Returns the number of significant bits (the position of the highest set
    /// bit plus one); zero has a bit length of zero.
    pub fn bit_len(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(&top) => {
                (self.limbs.len() as u64 - 1) * LIMB_BITS as u64
                    + (LIMB_BITS - top.leading_zeros()) as u64
            }
        }
    }

    /// Returns the number of trailing zero bits, i.e. the largest `k` such that
    /// `2^k` divides this value. Returns zero for the value zero.
    pub fn trailing_zeros(&self) -> u64 {
        for (i, &l) in self.limbs.iter().enumerate() {
            if l != 0 {
                return i as u64 * LIMB_BITS as u64 + l.trailing_zeros() as u64;
            }
        }
        0
    }

    /// Drops any trailing zero limbs, restoring the canonical form.
    fn normalize(&mut self) {
        while matches!(self.limbs.last(), Some(&0)) {
            self.limbs.pop();
        }
    }

    /// Compares two naturals.
    fn cmp_ref(&self, other: &Nat) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
        for (a, b) in self.limbs.iter().rev().zip(other.limbs.iter().rev()) {
            match a.cmp(b) {
                Ordering::Equal => continue,
                non_eq => return non_eq,
            }
        }
        Ordering::Equal
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Nat) -> Nat {
        let (long, short) = if self.limbs.len() >= rhs.limbs.len() {
            (self, rhs)
        } else {
            (rhs, self)
        };
        let sl = short.limbs.len();
        let mut out = Vec::with_capacity(long.limbs.len() + 1);
        let mut carry = 0;
        // Overlapping low limbs (bounds-check-free zip).
        for (&a, &b) in long.limbs[..sl].iter().zip(&short.limbs) {
            let (s, c) = adc(a, b, carry);
            out.push(s);
            carry = c;
        }
        // High limbs of `long`: propagate the carry, then bulk-copy the rest.
        let tail = &long.limbs[sl..];
        let mut i = 0;
        while carry != 0 && i < tail.len() {
            let (s, c) = adc(tail[i], 0, carry);
            out.push(s);
            carry = c;
            i += 1;
        }
        out.extend_from_slice(&tail[i..]);
        if carry != 0 {
            out.push(carry);
        }
        // Already normalized: the top limb is non-zero (or both inputs were zero).
        Nat { limbs: out }
    }

    /// Returns `self - rhs`, or `None` if `rhs > self` (naturals cannot be
    /// negative).
    pub fn checked_sub(&self, rhs: &Nat) -> Option<Nat> {
        if self.cmp_ref(rhs) == Ordering::Less {
            return None;
        }
        let rl = rhs.limbs.len();
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0;
        // Overlapping low limbs (self is at least as long, given self >= rhs).
        for (&a, &b) in self.limbs[..rl].iter().zip(&rhs.limbs) {
            let (d, bb) = sbb(a, b, borrow);
            out.push(d);
            borrow = bb;
        }
        // High limbs of `self`: propagate the borrow, then bulk-copy the rest.
        let tail = &self.limbs[rl..];
        let mut i = 0;
        while borrow != 0 && i < tail.len() {
            let (d, bb) = sbb(tail[i], 0, borrow);
            out.push(d);
            borrow = bb;
            i += 1;
        }
        out.extend_from_slice(&tail[i..]);
        debug_assert_eq!(borrow, 0, "checked_sub borrow escaped after a >= b check");
        let mut n = Nat { limbs: out };
        n.normalize();
        Some(n)
    }

    /// Returns `self · rhs`, dispatching by operand size along a
    /// schoolbook → Karatsuba → Toom-3 → Toom-4 → NTT ladder.
    pub fn mul(&self, rhs: &Nat) -> Nat {
        if self.is_zero() || rhs.is_zero() {
            return Nat::zero();
        }
        if self.is_one() {
            return rhs.clone();
        }
        if rhs.is_one() {
            return self.clone();
        }
        if self.limbs == rhs.limbs {
            return self.square();
        }
        let min_len = self.limbs.len().min(rhs.limbs.len());
        if min_len < KARATSUBA_THRESHOLD {
            self.mul_schoolbook(rhs)
        } else if min_len < TOOM3_THRESHOLD {
            self.mul_karatsuba(rhs)
        } else if min_len < TOOM4_THRESHOLD {
            self.mul_toom3(rhs)
        } else if min_len < NTT_THRESHOLD {
            self.mul_toom4(rhs)
        } else {
            mul_ntt(self, rhs)
        }
    }

    /// Toom-4 multiplication: seven quarter-size products evaluated at
    /// {0, 1, −1, 2, −2, 3, ∞} and interpolated. Asymptotically `O(n^1.404)`.
    fn mul_toom4(&self, rhs: &Nat) -> Nat {
        use crate::int::Int;

        let n = self.limbs.len().max(rhs.limbs.len());
        let k = n.div_ceil(4);
        let part = |x: &Nat, i: usize| -> Int {
            let l = x.limbs.len();
            let lo = i * k;
            if lo >= l {
                Int::ZERO
            } else {
                Int::from(Nat::from_limbs(&x.limbs[lo..(lo + k).min(l)]))
            }
        };
        let three = Int::from_i64(3);
        let nine = Int::from_i64(9);
        let twenty_seven = Int::from_i64(27);

        // Evaluate a polynomial's four digits at the seven points.
        let eval = |x: &Nat| -> [Int; 7] {
            let (d0, d1, d2, d3) = (part(x, 0), part(x, 1), part(x, 2), part(x, 3));
            let even1 = d0.add(&d2); // d0 + d2
            let odd1 = d1.add(&d3); // d1 + d3
            let p1 = even1.add(&odd1); // x(1)
            let pm1 = even1.sub(&odd1); // x(-1)
            let even2 = d0.add(&d2.mul_2k(2)); // d0 + 4 d2
            let odd2 = d1.mul_2k(1).add(&d3.mul_2k(3)); // 2 d1 + 8 d3
            let p2 = even2.add(&odd2); // x(2)
            let pm2 = even2.sub(&odd2); // x(-2)
            let p3 = d0
                .add(&d1.mul(&three))
                .add(&d2.mul(&nine))
                .add(&d3.mul(&twenty_seven)); // x(3)
            [d0, p1, pm1, p2, pm2, p3, d3]
        };
        let ea = eval(self);
        let eb = eval(rhs);
        // Pointwise products at 0, 1, −1, 2, −2, 3, ∞.
        let v: [Int; 7] = core::array::from_fn(|i| ea[i].mul(&eb[i]));
        let (v0, v1, vm1, v2, vm2, v3, vinf) = (&v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]);

        // Interpolate c0..c6 (all divisions exact).
        let two = Int::from_i64(2);
        let c0 = v0.clone();
        let c6 = vinf.clone();
        let e1 = v1.add(vm1).div_exact(&two).sub(&c0).sub(&c6); // c2 + c4
        let o1 = v1.sub(vm1).div_exact(&two); // c1 + c3 + c5
        let e2 = v2
            .add(vm2)
            .div_exact(&two)
            .sub(&c0)
            .sub(&c6.mul(&Int::from_i64(64))); // 4c2 + 16c4
        let o2h = v2.sub(vm2).div_exact(&Int::from_i64(4)); // c1 + 4c3 + 16c5
        let c4 = e2
            .sub(&e1.mul(&Int::from_i64(4)))
            .div_exact(&Int::from_i64(12));
        let c2 = e1.sub(&c4);
        let f = o2h.sub(&o1).div_exact(&three); // c3 + 5c5
        let g = v3
            .sub(&c0)
            .sub(&c2.mul(&nine))
            .sub(&c4.mul(&Int::from_i64(81)))
            .sub(&c6.mul(&Int::from_i64(729)))
            .div_exact(&three); // c1 + 9c3 + 81c5
        let h = g.sub(&o1).div_exact(&Int::from_i64(8)); // c3 + 10c5
        let c5 = h.sub(&f).div_exact(&Int::from_i64(5));
        let c3 = f.sub(&c5.mul(&Int::from_i64(5)));
        let c1 = o1.sub(&c3).sub(&c5);

        recombine_coeffs(
            self.limbs.len() + rhs.limbs.len(),
            k,
            &[c0, c1, c2, c3, c4, c5, c6],
        )
    }

    /// Toom-3 multiplication: five half-third-size products, evaluated at the
    /// points {0, 1, −1, 2, ∞} and interpolated (signed intermediates use
    /// [`Int`]). Asymptotically `O(n^1.465)`.
    fn mul_toom3(&self, rhs: &Nat) -> Nat {
        use crate::int::Int;

        let n = self.limbs.len().max(rhs.limbs.len());
        let k = n.div_ceil(3);

        // Split a value into its base-2^(64k) digits a0 + a1·B + a2·B², as Int.
        let part = |x: &Nat, lo: usize, hi: usize| -> Int {
            let l = x.limbs.len();
            if lo >= l {
                Int::ZERO
            } else {
                Int::from(Nat::from_limbs(&x.limbs[lo..hi.min(l)]))
            }
        };
        let (a0, a1, a2) = (
            part(self, 0, k),
            part(self, k, 2 * k),
            part(self, 2 * k, 3 * k),
        );
        let (b0, b1, b2) = (
            part(rhs, 0, k),
            part(rhs, k, 2 * k),
            part(rhs, 2 * k, 3 * k),
        );

        // Evaluate a(x), b(x) at 1, −1, 2 (0 and ∞ are a0/a2 directly).
        let pa = a0.add(&a2);
        let (pm1, p1) = (pa.sub(&a1), pa.add(&a1));
        let p2 = p1.add(&a2).mul_2k(1).sub(&a0);
        let qb = b0.add(&b2);
        let (qm1, q1) = (qb.sub(&b1), qb.add(&b1));
        let q2 = q1.add(&b2).mul_2k(1).sub(&b0);

        // Pointwise products (these recurse through the dispatcher).
        let r0 = a0.mul(&b0);
        let r1 = p1.mul(&q1);
        let rm1 = pm1.mul(&qm1);
        let r2 = p2.mul(&q2);
        let rinf = a2.mul(&b2);

        // Interpolate the coefficients c0..c4 (exact divisions by 2 and 6).
        let two = Int::from_i64(2);
        let c0 = r0;
        let c4 = rinf;
        let c2 = r1.add(&rm1).div_exact(&two).sub(&c0).sub(&c4);
        let s = r1.sub(&rm1).div_exact(&two);
        let t = r2
            .sub(&c0)
            .sub(&c2.mul(&Int::from_i64(4)))
            .sub(&c4.mul(&Int::from_i64(16)))
            .sub(&s.mul(&two));
        let c3 = t.div_exact(&Int::from_i64(6));
        let c1 = s.sub(&c3);

        // The final coefficients are the (non-negative) product-polynomial
        // coefficients; recombine them at whole-limb offsets `i·k` directly.
        recombine_coeffs(self.limbs.len() + rhs.limbs.len(), k, &[c0, c1, c2, c3, c4])
    }

    /// Quadratic schoolbook (long) multiplication (see [`mul_into_schoolbook`]).
    fn mul_schoolbook(&self, rhs: &Nat) -> Nat {
        let mut out = alloc::vec![0 as Limb; self.limbs.len() + rhs.limbs.len()];
        mul_into_schoolbook(&self.limbs, &rhs.limbs, &mut out);
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }
}

/// Quadratic schoolbook (long) multiplication of `a·b` into the zeroed buffer
/// `out` (`out.len() == a.len() + b.len()`); accumulates on top of whatever is
/// already there (the buffer being zeroed makes it a plain product).
///
/// The main loop is an `addmul_2`: each pass adds two multiplier limbs'
/// worth of products (`out += (a₀ + a₁·2^64)·b`), halving the passes over
/// `out` and keeping two independent limb products in flight per iteration
/// instead of one serial multiply-accumulate chain.
fn mul_into_schoolbook(a: &[Limb], b: &[Limb], out: &mut [Limb]) {
    use crate::limb::DLimb;
    debug_assert_eq!(out.len(), a.len() + b.len());
    let rn = b.len();
    if a.is_empty() || rn == 0 {
        return;
    }
    let mut i = 0;
    while i + 2 <= a.len() {
        let (a0, a1) = (a[i], a[i + 1]);
        // Position i+j collects lo(a0·b[j]) + hi(a0·b[j−1]) + lo(a1·b[j−1])
        // + hi(a1·b[j−2]) + carry: five limb-sized terms, so the u128
        // accumulator cannot overflow and the carry out is at most 4.
        let mut ph0: Limb = 0; // hi(a0·b[j−1])
        let mut pl1: Limb = 0; // lo(a1·b[j−1])
        let mut ph1: Limb = 0; // hi(a1·b[j−1])
        let mut ph1p: Limb = 0; // hi(a1·b[j−2])
        let mut carry: Limb = 0;
        let row = &mut out[i..i + rn + 2];
        for (o, &bj) in row.iter_mut().zip(b) {
            let p0 = a0 as DLimb * bj as DLimb;
            let p1 = a1 as DLimb * bj as DLimb;
            let acc = *o as DLimb
                + (p0 as Limb) as DLimb
                + ph0 as DLimb
                + pl1 as DLimb
                + ph1p as DLimb
                + carry as DLimb;
            *o = acc as Limb;
            carry = (acc >> LIMB_BITS) as Limb;
            ph0 = (p0 >> LIMB_BITS) as Limb;
            ph1p = ph1;
            pl1 = p1 as Limb;
            ph1 = (p1 >> LIMB_BITS) as Limb;
        }
        // Flush the pipeline into the two limbs above the row.
        let acc = row[rn] as DLimb + ph0 as DLimb + pl1 as DLimb + ph1p as DLimb + carry as DLimb;
        row[rn] = acc as Limb;
        let top = row[rn + 1] as DLimb + ph1 as DLimb + (acc >> LIMB_BITS);
        row[rn + 1] = top as Limb;
        debug_assert_eq!(top >> LIMB_BITS, 0, "schoolbook top carry escaped");
        i += 2;
    }
    if i < a.len() {
        // Odd tail: one classic addmul_1 row.
        let ai = a[i];
        let mut carry = 0;
        let row = &mut out[i..i + rn];
        for (o, &bj) in row.iter_mut().zip(b) {
            let (lo, hi) = mac(*o, ai, bj, carry);
            *o = lo;
            carry = hi;
        }
        out[i + rn] = carry;
    }
}

/// Sums `a + b` into `out`, which must be exactly `max(len) + 1` limbs; every
/// limb of `out` is written (the top limb is the 0/1 carry).
fn add_full(a: &[Limb], b: &[Limb], out: &mut [Limb]) {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    debug_assert_eq!(out.len(), long.len() + 1);
    let mut carry = 0;
    let (head, tail) = out.split_at_mut(short.len());
    for ((o, &x), &y) in head.iter_mut().zip(long).zip(short) {
        let (s, c) = adc(x, y, carry);
        *o = s;
        carry = c;
    }
    let (mid, last) = tail.split_at_mut(long.len() - short.len());
    for (o, &x) in mid.iter_mut().zip(&long[short.len()..]) {
        let (s, c) = adc(x, 0, carry);
        *o = s;
        carry = c;
    }
    last[0] = carry;
}

/// Subtracts `src` from `dst` in place (`dst -= src`); requires `dst >= src`
/// as numbers and `dst.len() >= src.len()`.
fn sub_in_place(dst: &mut [Limb], src: &[Limb]) {
    let mut borrow = 0;
    let (head, tail) = dst.split_at_mut(src.len());
    for (d, &s) in head.iter_mut().zip(src) {
        let (r, b) = sbb(*d, s, borrow);
        *d = r;
        borrow = b;
    }
    let mut it = tail.iter_mut();
    while borrow != 0 {
        let d = it.next().expect("sub_in_place borrow escaped: dst < src");
        let (r, b) = sbb(*d, 0, borrow);
        *d = r;
        borrow = b;
    }
}

/// Karatsuba multiplication of `a·b` into the zeroed buffer `out`
/// (`out.len() == a.len() + b.len()`), using `scratch` for every intermediate —
/// no per-node allocation. `z0` and `z2` land directly in the disjoint halves
/// of `out`; only the middle term needs scratch space.
fn kara_into(a: &[Limb], b: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    debug_assert_eq!(out.len(), a.len() + b.len());
    if a.len().min(b.len()) < KARATSUBA_THRESHOLD {
        mul_into_schoolbook(a, b, out);
        return;
    }
    let h = a.len().max(b.len()).div_ceil(2);
    let (a0, a1) = a.split_at(a.len().min(h));
    let (b0, b1) = b.split_at(b.len().min(h));
    // z0 = a0·b0 at offset 0, z2 = a1·b1 at offset 2h (disjoint slices).
    let z0_len = a0.len() + b0.len();
    let have_z2 = !a1.is_empty() && !b1.is_empty();
    kara_into(a0, b0, &mut out[..z0_len], scratch);
    if have_z2 {
        kara_into(a1, b1, &mut out[2 * h..], scratch);
    }
    // zm = (a0+a1)·(b0+b1) − z0 − z2, added at offset h.
    let sa_len = a0.len().max(a1.len()) + 1;
    let sb_len = b0.len().max(b1.len()) + 1;
    let (sa, rest) = scratch.split_at_mut(sa_len);
    let (sb, rest) = rest.split_at_mut(sb_len);
    let (zm, rest) = rest.split_at_mut(sa_len + sb_len);
    add_full(a0, a1, sa);
    add_full(b0, b1, sb);
    zm.fill(0);
    kara_into(sa, sb, zm, rest);
    sub_in_place(zm, &out[..z0_len]);
    if have_z2 {
        sub_in_place(zm, &out[2 * h..]);
    }
    // Trim zm's zero top limbs: the value always fits, the buffer may not.
    let zm_len = zm.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
    add_at(out, h, &zm[..zm_len]);
}

/// Karatsuba squaring of `a` into the zeroed buffer `out`
/// (`out.len() == 2·a.len()`), sharing the same scratch discipline as
/// [`kara_into`].
fn kara_sqr_into(a: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    debug_assert_eq!(out.len(), 2 * a.len());
    if a.len() < KARATSUBA_THRESHOLD {
        sqr_into(a, out);
        return;
    }
    let h = a.len().div_ceil(2);
    let (a0, a1) = a.split_at(h);
    kara_sqr_into(a0, &mut out[..2 * h], scratch);
    kara_sqr_into(a1, &mut out[2 * h..], scratch);
    let (sa, rest) = scratch.split_at_mut(h + 1);
    let (zm, rest) = rest.split_at_mut(2 * (h + 1));
    add_full(a0, a1, sa);
    zm.fill(0);
    kara_sqr_into(sa, zm, rest);
    sub_in_place(zm, &out[..2 * h]);
    sub_in_place(zm, &out[2 * h..]);
    let zm_len = zm.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
    add_at(out, h, &zm[..zm_len]);
}

/// Scratch size sufficient for [`kara_into`]/[`kara_sqr_into`] on operands of
/// at most `n` limbs each: each level consumes `≤ 2n + 6` limbs and recurses on
/// `≤ n/2 + 2`.
fn kara_scratch_len(n: usize) -> usize {
    let mut need = 0;
    let mut m = n;
    while m >= KARATSUBA_THRESHOLD {
        need += 2 * m + 6;
        m = m / 2 + 2;
    }
    need
}

impl Nat {
    /// Returns `self²`, using a symmetric schoolbook or Karatsuba squaring
    /// (roughly half the limb multiplications of the general `mul`).
    pub fn square(&self) -> Nat {
        if self.is_zero() {
            return Nat::zero();
        }
        if self.limbs.len() < KARATSUBA_THRESHOLD {
            self.square_schoolbook()
        } else {
            self.square_karatsuba()
        }
    }

    /// Symmetric schoolbook squaring (see [`sqr_into`]).
    fn square_schoolbook(&self) -> Nat {
        let mut out = alloc::vec![0 as Limb; 2 * self.limbs.len()];
        sqr_into(&self.limbs, &mut out);
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }

    /// Karatsuba squaring: three half-size squarings, run as a slice recursion
    /// into one output buffer with one shared scratch allocation.
    fn square_karatsuba(&self) -> Nat {
        let n = self.limbs.len();
        if n < KARATSUBA_THRESHOLD {
            return self.square_schoolbook();
        }
        let mut out = alloc::vec![0 as Limb; 2 * n];
        let mut scratch = alloc::vec![0 as Limb; kara_scratch_len(n)];
        kara_sqr_into(&self.limbs, &mut out, &mut scratch);
        let mut r = Nat { limbs: out };
        r.normalize();
        r
    }

    /// Karatsuba multiplication: three half-size products instead of four, run
    /// as a slice recursion into one output buffer with one shared scratch
    /// allocation (see [`kara_into`]) instead of ~9 `Vec`s per node.
    fn mul_karatsuba(&self, rhs: &Nat) -> Nat {
        if self.limbs.len().min(rhs.limbs.len()) < KARATSUBA_THRESHOLD {
            return self.mul_schoolbook(rhs);
        }
        let mut out = alloc::vec![0 as Limb; self.limbs.len() + rhs.limbs.len()];
        let mut scratch =
            alloc::vec![0 as Limb; kara_scratch_len(self.limbs.len().max(rhs.limbs.len()))];
        kara_into(&self.limbs, &rhs.limbs, &mut out, &mut scratch);
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }

    /// Returns `self << bits`.
    pub fn shl(&self, bits: u64) -> Nat {
        if self.is_zero() || bits == 0 {
            return self.clone();
        }
        let limb_shift = (bits / LIMB_BITS as u64) as usize;
        let bit_shift = (bits % LIMB_BITS as u64) as u32;
        let mut out = alloc::vec![0 as Limb; limb_shift];
        if bit_shift == 0 {
            out.extend_from_slice(&self.limbs);
        } else {
            let mut carry = 0;
            for &l in &self.limbs {
                out.push((l << bit_shift) | carry);
                carry = l >> (LIMB_BITS - bit_shift);
            }
            if carry != 0 {
                out.push(carry);
            }
        }
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }

    /// Returns `self >> bits` (floor division by `2^bits`).
    pub fn shr(&self, bits: u64) -> Nat {
        if self.is_zero() || bits == 0 {
            return self.clone();
        }
        let limb_shift = (bits / LIMB_BITS as u64) as usize;
        let bit_shift = (bits % LIMB_BITS as u64) as u32;
        if limb_shift >= self.limbs.len() {
            return Nat::zero();
        }
        let src = &self.limbs[limb_shift..];
        let mut out = Vec::with_capacity(src.len());
        if bit_shift == 0 {
            out.extend_from_slice(src);
        } else {
            for i in 0..src.len() {
                let lo = src[i] >> bit_shift;
                let hi = src
                    .get(i + 1)
                    .map(|&h| h << (LIMB_BITS - bit_shift))
                    .unwrap_or(0);
                out.push(lo | hi);
            }
        }
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }

    /// Returns the greatest common divisor of `self` and `rhs`.
    ///
    /// Small operands use Stein's binary GCD; large ones use Lehmer's algorithm,
    /// which advances several Euclidean steps per multi-precision operation.
    /// `gcd(0, n) == gcd(n, 0) == n`, and `gcd(0, 0) == 0`.
    pub fn gcd(&self, rhs: &Nat) -> Nat {
        if self.is_zero() {
            return rhs.clone();
        }
        if rhs.is_zero() {
            return self.clone();
        }
        // gcd(1, n) == 1 without touching the operands (dividing by a unit is
        // the common case when normalizing integer-valued rationals).
        if self.is_one() || rhs.is_one() {
            return Nat::one();
        }
        // Machine-word operands: allocation-free binary GCD (this is the hot
        // path of every small-`Rational` reduction).
        if self.limbs.len() <= 2 && rhs.limbs.len() <= 2 {
            return Nat::from_u128(u128_gcd(
                self.to_u128().expect("<= 2 limbs"),
                rhs.to_u128().expect("<= 2 limbs"),
            ));
        }
        if self.limbs.len().max(rhs.limbs.len()) < LEHMER_THRESHOLD {
            self.gcd_binary(rhs)
        } else {
            self.gcd_lehmer(rhs)
        }
    }

    /// Stein's binary GCD (no division). Precondition: both operands non-zero.
    fn gcd_binary(&self, rhs: &Nat) -> Nat {
        let mut u = self.clone();
        let mut v = rhs.clone();
        let shift = u.trailing_zeros().min(v.trailing_zeros());
        u = u.shr(u.trailing_zeros());
        v = v.shr(v.trailing_zeros());
        loop {
            v = v.shr(v.trailing_zeros());
            if u.cmp_ref(&v) == Ordering::Greater {
                core::mem::swap(&mut u, &mut v);
            }
            v = v
                .checked_sub(&u)
                .expect("binary gcd: v >= u by construction");
            if v.is_zero() {
                break;
            }
        }
        u.shl(shift)
    }

    /// Lehmer's GCD (Knuth TAOCP §4.5.2, Algorithm L): use the leading words to
    /// derive a 2×2 cofactor matrix in single precision, then apply it to the
    /// full operands, doing far fewer multi-precision divisions than plain
    /// Euclid. Precondition: both operands non-zero.
    fn gcd_lehmer(&self, rhs: &Nat) -> Nat {
        let mut u = self.clone();
        let mut v = rhs.clone();
        if u.cmp_ref(&v) == Ordering::Less {
            core::mem::swap(&mut u, &mut v);
        }
        while v.limbs.len() > 1 {
            // Leading ~63 bits of u, and of v at the same alignment.
            let shift = u.bit_len().saturating_sub(63);
            let mut x = u.shr(shift).to_u64().unwrap_or(0);
            let mut y = v.shr(shift).to_u64().unwrap_or(0);

            // Single-precision partial Euclid, accumulating [[a,b],[c,d]] in
            // machine words: an i128 quotient lowers to a slow division
            // libcall, while Lehmer's invariants keep every quantity inside a
            // u64/i64. Any sign or overflow violation simply breaks to the
            // (always correct) multi-precision fallback below.
            let (mut a, mut b, mut c, mut d) = (1i64, 0i64, 0i64, 1i64);
            loop {
                let (yc, yd) = (y as i128 + c as i128, y as i128 + d as i128);
                if yc <= 0 || yd <= 0 {
                    break;
                }
                // x < 2^63 and |a| ≤ i64::MAX, so x+a fits a u64 when ≥ 0.
                let (xa, xb) = (x as i128 + a as i128, x as i128 + b as i128);
                if xa < 0 || xb < 0 {
                    break;
                }
                let q = (xa as u64) / (yc as u64);
                if q != (xb as u64) / (yd as u64) {
                    break; // Lehmer's exactness test failed
                }
                let Ok(qi) = i64::try_from(q) else { break };
                let (Some(nc), Some(nd)) = (
                    qi.checked_mul(c).and_then(|t| a.checked_sub(t)),
                    qi.checked_mul(d).and_then(|t| b.checked_sub(t)),
                ) else {
                    break;
                };
                (a, b) = (c, d);
                (c, d) = (nc, nd);
                // ny = x − q·y, non-negative by construction (same bits as the
                // former i128 computation).
                let ny = (x as u128).wrapping_sub(q as u128 * y as u128) as u64;
                x = y;
                y = ny;
            }

            if b == 0 {
                // No single-precision progress: one full division step.
                let (_, r) = u.div_rem(&v).expect("v is non-zero");
                u = core::mem::replace(&mut v, r);
            } else {
                // Apply the matrix to the full operands (result stays positive),
                // borrowing `u`/`v` rather than cloning them into `Int`.
                let nu = lincomb_pos(a as i128, &u, b as i128, &v);
                let nv = lincomb_pos(c as i128, &u, d as i128, &v);
                u = nu;
                v = nv;
                if u.cmp_ref(&v) == Ordering::Less {
                    core::mem::swap(&mut u, &mut v);
                }
            }
        }
        // v now fits a single limb: finish in machine words.
        if v.is_zero() {
            return u;
        }
        let vr = v.limbs[0];
        let ur = u.divmod_small(vr).1;
        Nat::from_u64(u64_gcd(vr, ur))
    }

    /// Returns bit `i` (0 = least significant), or `false` past the top.
    #[inline]
    pub fn bit(&self, i: u64) -> bool {
        let limb = (i / LIMB_BITS as u64) as usize;
        match self.limbs.get(limb) {
            Some(&l) => (l >> (i % LIMB_BITS as u64)) & 1 == 1,
            None => false,
        }
    }

    /// Divides by `rhs`, returning `(quotient, remainder)` with
    /// `self == quotient·rhs + remainder` and `remainder < rhs`, or `None` if
    /// `rhs` is zero.
    ///
    /// Dispatches to single-limb division, Knuth's Algorithm D (TAOCP Vol. 2
    /// §4.3.1), or sub-quadratic Burnikel–Ziegler recursive division by size.
    pub fn div_rem(&self, rhs: &Nat) -> Option<(Nat, Nat)> {
        if rhs.is_zero() {
            return None;
        }
        match self.cmp_ref(rhs) {
            Ordering::Less => return Some((Nat::zero(), self.clone())),
            Ordering::Equal => return Some((Nat::one(), Nat::zero())),
            Ordering::Greater => {}
        }
        if rhs.limbs.len() == 1 {
            let (q, r) = self.divmod_small(rhs.limbs[0]);
            return Some((q, Nat::from_u64(r)));
        }
        if rhs.limbs.len() >= BZ_THRESHOLD {
            return Some(bz_div_rem(self, rhs));
        }
        Some(self.div_rem_knuth(rhs))
    }

    /// Knuth Algorithm D: schoolbook long division in base `2^64`, with a
    /// normalized divisor and the 2-by-1 limb quotient estimate. Precondition:
    /// `rhs` has ≥ 2 limbs and `self > rhs`.
    fn div_rem_knuth(&self, rhs: &Nat) -> (Nat, Nat) {
        const B: u128 = 1 << LIMB_BITS;
        let n = rhs.limbs.len();
        let m = self.limbs.len() - n;

        // Normalize so the divisor's top limb has its high bit set.
        let shift = rhs.limbs[n - 1].leading_zeros();
        let vn = rhs.shl(shift as u64);
        let vv = &vn.limbs;
        debug_assert_eq!(vv.len(), n);
        let un = self.shl(shift as u64);
        let mut u = un.limbs; // move: `un` is a fresh local used only here
        u.resize(self.limbs.len() + 1, 0); // exactly m + n + 1 limbs

        let (b1, b2) = (vv[n - 1] as u128, vv[n - 2] as u128);
        let mut q = alloc::vec![0 as Limb; m + 1];

        for j in (0..=m).rev() {
            // Estimate the quotient limb from the top two dividend limbs.
            let num = ((u[j + n] as u128) << LIMB_BITS) | u[j + n - 1] as u128;
            let mut qhat = num / b1;
            let mut rhat = num % b1;
            while qhat >= B || qhat * b2 > ((rhat << LIMB_BITS) | u[j + n - 2] as u128) {
                qhat -= 1;
                rhat += b1;
                if rhat >= B {
                    break;
                }
            }

            // Multiply and subtract: u[j..=j+n] -= qhat · vv.
            let mut carry: u128 = 0;
            let mut borrow: i64 = 0;
            for i in 0..n {
                let p = qhat * vv[i] as u128 + carry;
                carry = p >> LIMB_BITS;
                let d = (u[j + i] as i128) - ((p as u64) as i128) - (borrow as i128);
                u[j + i] = d as u64;
                borrow = if d < 0 { 1 } else { 0 };
            }
            let d = (u[j + n] as i128) - (carry as i128) - (borrow as i128);
            u[j + n] = d as u64;

            q[j] = qhat as Limb;
            if d < 0 {
                // qhat was one too large: add the divisor back.
                q[j] -= 1;
                let mut add_carry: u128 = 0;
                for i in 0..n {
                    let s = u[j + i] as u128 + vv[i] as u128 + add_carry;
                    u[j + i] = s as u64;
                    add_carry = s >> LIMB_BITS;
                }
                u[j + n] = (u[j + n] as u128 + add_carry) as u64;
            }
        }

        let mut quotient = Nat { limbs: q };
        quotient.normalize();
        // Denormalize the remainder (the low n limbs of u), undoing the shift.
        let remainder = Nat::from_limbs(&u[..n]).shr(shift as u64);
        (quotient, remainder)
    }

    /// Divides by a single-limb value, returning `(quotient, remainder)`.
    ///
    /// The divisor must be non-zero. This is the primitive behind decimal
    /// formatting; full multi-limb division is a later milestone.
    fn divmod_small(&self, d: Limb) -> (Nat, Limb) {
        debug_assert!(d != 0, "divmod_small by zero");
        let dd = d as u128;
        let mut rem: u128 = 0;
        let mut q = alloc::vec![0 as Limb; self.limbs.len()];
        for i in (0..self.limbs.len()).rev() {
            let cur = (rem << LIMB_BITS) | self.limbs[i] as u128;
            q[i] = (cur / dd) as Limb;
            rem = cur % dd;
        }
        let mut n = Nat { limbs: q };
        n.normalize();
        (n, rem as Limb)
    }
}

impl Nat {
    /// Returns the value as a `u64` if it fits in a single limb.
    pub fn to_u64(&self) -> Option<u64> {
        match self.limbs.as_slice() {
            [] => Some(0),
            &[only] => Some(only),
            _ => None,
        }
    }

    /// Returns the value as a `u128` if it fits (at most two limbs).
    pub fn to_u128(&self) -> Option<u128> {
        match self.limbs.as_slice() {
            [] => Some(0),
            &[lo] => Some(lo as u128),
            &[lo, hi] => Some(((hi as u128) << 64) | lo as u128),
            _ => None,
        }
    }

    /// Returns `true` if this value is one.
    #[inline]
    pub fn is_one(&self) -> bool {
        self.limbs.as_slice() == [1]
    }

    /// Returns the little-endian limb slice of the magnitude, normalized so the
    /// most-significant limb is non-zero (empty for zero).
    #[inline]
    pub fn as_limbs(&self) -> &[Limb] {
        &self.limbs
    }

    /// Builds a natural from little-endian limbs (any trailing zeros are
    /// stripped).
    pub fn from_limbs(limbs: &[Limb]) -> Nat {
        let mut n = Nat {
            limbs: limbs.to_vec(),
        };
        n.normalize();
        n
    }

    /// Builds a natural from little-endian bytes.
    pub fn from_bytes_le(bytes: &[u8]) -> Nat {
        let mut limbs = Vec::with_capacity(bytes.len() / 8 + 1);
        for chunk in bytes.chunks(8) {
            let mut limb: Limb = 0;
            for (i, &b) in chunk.iter().enumerate() {
                limb |= (b as Limb) << (8 * i);
            }
            limbs.push(limb);
        }
        let mut n = Nat { limbs };
        n.normalize();
        n
    }

    /// Returns the magnitude as little-endian bytes (no trailing zero bytes;
    /// empty for zero).
    pub fn to_bytes_le(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.limbs.len() * 8);
        for &limb in &self.limbs {
            out.extend_from_slice(&limb.to_le_bytes());
        }
        while matches!(out.last(), Some(&0)) {
            out.pop();
        }
        out
    }

    /// Returns the low `k` bits of this value, i.e. `self mod 2^k`.
    pub fn low_bits(&self, k: u64) -> Nat {
        if k == 0 {
            return Nat::zero();
        }
        let full = (k / LIMB_BITS as u64) as usize;
        let rem = (k % LIMB_BITS as u64) as u32;
        let take = full.min(self.limbs.len());
        let mut out: Vec<Limb> = self.limbs[..take].to_vec();
        if rem > 0 && full < self.limbs.len() {
            while out.len() < full {
                out.push(0);
            }
            out.push(self.limbs[full] & ((1u64 << rem) - 1));
        }
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }

    /// Returns `self` raised to `exp` (`self^0 == 1`), by square-and-multiply.
    pub fn pow(&self, exp: u32) -> Nat {
        let mut result = Nat::one();
        let mut base = self.clone();
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul(&base);
            }
            e >>= 1;
            if e > 0 {
                base = base.square();
            }
        }
        result
    }

    /// Returns the floor of the square root, `⌊√self⌋`.
    ///
    /// Zimmermann's "Karatsuba square root" ([`sqrt_rem`]): the divisions it
    /// performs are half-size-by-quarter-size (rather than full-width), and the
    /// remainder is maintained algebraically instead of via full-width squarings,
    /// for a total cost of `O(M(n))` with small constants.
    pub fn isqrt(&self) -> Nat {
        let b = self.bit_len();
        if b <= 128 {
            return Nat::from_u128(isqrt_u128(self.to_u128().expect("<= 128 bits")));
        }
        // Normalize with an even left shift so the top limb is ≥ 2^62 (the
        // recursion's precondition); `⌊√(v·4^t)⌋ >> t == ⌊√v⌋` undoes it.
        let top = *self.limbs.last().expect("non-zero");
        let sh = (top.leading_zeros() & !1) as u64;
        if sh == 0 {
            sqrt_rem(self).0
        } else {
            sqrt_rem(&self.shl(sh)).0.shr(sh / 2)
        }
    }

    /// Returns the floor of the `k`th root, `⌊self^(1/k)⌋`, for `k >= 1`, by
    /// bitwise binary search.
    pub fn nth_root_floor(&self, k: u32) -> Nat {
        assert!(k >= 1, "nth_root_floor: k must be >= 1");
        if k == 1 || self.is_zero() || self.is_one() {
            return self.clone();
        }
        if k == 2 {
            return self.isqrt();
        }
        let hb = self.bit_len().div_ceil(k as u64);
        let mut root = Nat::zero();
        for bit in (0..=hb).rev() {
            let cand = root.add(&Nat::one().shl(bit));
            if cand.pow(k).cmp_ref(self) != Ordering::Greater {
                root = cand;
            }
        }
        root
    }

    /// Writes the magnitude in the given `radix` (2–36) to `out`.
    pub fn write_radix(&self, out: &mut impl fmt::Write, radix: u32) -> fmt::Result {
        assert!((2..=36).contains(&radix), "radix must be in 2..=36");
        if self.is_zero() {
            return out.write_str("0");
        }
        out.write_str(&self.to_radix_string(radix))
    }

    /// Returns the minimal (no leading zeros) base-`radix` digit string, using
    /// divide-and-conquer: split off a divisor `radix^len ≈ √self`, recurse on
    /// the quotient and remainder, and zero-pad the remainder to `len` digits.
    /// With sub-quadratic multiplication/division this is `O(M(n)·log n)`.
    fn to_radix_string(&self, radix: u32) -> String {
        // Base case: a few limbs go straight through single-limb division.
        if self.limbs.len() <= RADIX_RECURSION_LIMBS {
            return simple_radix_string(self, radix);
        }
        // Build the ladder `[radix^1, radix^2, radix^4, …]` once (largest entry
        // ≤ self), then share it across the whole recursion instead of
        // re-squaring it at every node.
        let mut powers = alloc::vec![Nat::from_u64(radix as u64)];
        let bits = self.bit_len();
        loop {
            let last = powers.last().unwrap();
            // `bit_len(x²) ≥ 2·bit_len(x) − 1`: skip the (large, discarded)
            // final squaring when it certainly exceeds `self`.
            if 2 * last.bit_len() - 1 > bits {
                break;
            }
            let sq = last.square();
            if sq.cmp_ref(self) == Ordering::Greater {
                break;
            }
            powers.push(sq);
        }
        to_radix_recursive(self, &powers, radix)
    }
}

/// Recursive base-`radix` conversion sharing a precomputed power ladder
/// (`powers[k] == radix^(2^k)`, ascending).
fn to_radix_recursive(v: &Nat, powers: &[Nat], radix: u32) -> String {
    if v.limbs.len() <= RADIX_RECURSION_LIMBS {
        return simple_radix_string(v, radix);
    }
    // Split by the largest ladder entry `p = radix^(2^k) ≤ v`.
    let k = powers
        .iter()
        .rposition(|p| p.cmp_ref(v) != Ordering::Greater)
        .expect("v is large, so radix <= v");
    let len = 1usize << k;
    let (q, r) = v.div_rem(&powers[k]).expect("p is non-zero");
    let mut s = to_radix_recursive(&q, powers, radix);
    let r_str = if r.is_zero() {
        String::new()
    } else {
        to_radix_recursive(&r, powers, radix)
    };
    // Zero-pad the low part to exactly `len` digits.
    for _ in 0..len - r_str.len() {
        s.push('0');
    }
    s.push_str(&r_str);
    s
}

/// Number of limbs at or below which radix conversion uses the simple
/// single-limb-division loop rather than recursing.
const RADIX_RECURSION_LIMBS: usize = 10;

/// Minimal base-`radix` digit string via repeated single-limb division (for
/// small values / the recursion base case). Each multi-limb division peels a
/// whole limb's worth of digits (`radix^chunk`, the largest power fitting a
/// `u64`), which are then split out with cheap machine-word arithmetic.
fn simple_radix_string(n: &Nat, radix: u32) -> String {
    if n.is_zero() {
        return String::new();
    }
    // Largest `chunk` with `radix^chunk` fitting a u64, and that power itself.
    let (chunk, base) = {
        let (mut d, mut p) = (0u32, 1u64);
        while let Some(next) = p.checked_mul(radix as u64) {
            p = next;
            d += 1;
        }
        (d, p)
    };
    let mut n = n.clone();
    let mut buf = Vec::new();
    while !n.is_zero() {
        let (q, mut r) = n.divmod_small(base);
        n = q;
        if n.is_zero() {
            // Most significant chunk: stop at its leading digit.
            while r != 0 {
                buf.push(digit_char((r % radix as u64) as u32));
                r /= radix as u64;
            }
        } else {
            for _ in 0..chunk {
                buf.push(digit_char((r % radix as u64) as u32));
                r /= radix as u64;
            }
        }
    }
    buf.reverse();
    String::from_utf8(buf).unwrap_or_default()
}

/// Maps a digit value `0..36` to its ASCII character (`0-9`, then `a-z`).
#[inline]
fn digit_char(d: u32) -> u8 {
    if d < 10 {
        b'0' + d as u8
    } else {
        b'a' + (d - 10) as u8
    }
}

/// Parses an unsigned integer in the given `radix` (2–36).
pub(crate) fn parse_radix(s: &str, radix: u32) -> Result<Nat> {
    if !(2..=36).contains(&radix) || s.is_empty() {
        return Err(Error::Parse);
    }
    // Largest `d` with `radix^d` fitting a `u64`, and that base `B = radix^d`.
    let (chunk, base) = {
        let (mut d, mut p) = (0u32, 1u64);
        while let Some(next) = p.checked_mul(radix as u64) {
            p = next;
            d += 1;
        }
        (d as usize, p)
    };

    // Validate and collect the digit values (big-endian).
    let digits: Vec<u32> = s
        .chars()
        .map(|c| c.to_digit(radix).ok_or(Error::Parse))
        .collect::<Result<_>>()?;

    // Fast path for small inputs: a single base-`B` limb.
    if digits.len() <= chunk {
        let mut val: u64 = 0;
        for &dg in &digits {
            val = val * radix as u64 + dg as u64;
        }
        return Ok(Nat::from_u64(val));
    }

    // Pack `chunk` digits at a time into base-`B` limbs, least-significant first.
    let mut level: Vec<Nat> = Vec::with_capacity(digits.len() / chunk + 1);
    let mut end = digits.len();
    while end > 0 {
        let start = end.saturating_sub(chunk);
        let mut val: u64 = 0;
        for &dg in &digits[start..end] {
            val = val * radix as u64 + dg as u64;
        }
        level.push(Nat::from_u64(val));
        end = start;
    }

    // Merge adjacent limbs up a balanced tree: `pair.0 + pair.1 · B^(2^k)`.
    // `power` starts at `B` and squares each level, so the work is
    // `O(M(n)·log n)` with sub-quadratic multiplication.
    let mut power = Nat::from_u64(base);
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                next.push(pair[0].add(&pair[1].mul(&power)));
            } else {
                next.push(pair[0].clone());
            }
        }
        level = next;
        if level.len() > 1 {
            // The last level's power is never used: skip that (full-size) square.
            power = power.mul(&power);
        }
    }
    Ok(level.pop().unwrap_or_else(Nat::zero))
}

impl Nat {
    /// Returns `self^exp mod modulus`. Panics if `modulus` is zero.
    ///
    /// Odd moduli use Montgomery reduction; others fall back to
    /// square-and-multiply with a division-based reduction.
    pub fn modpow(&self, exp: &Nat, modulus: &Nat) -> Nat {
        assert!(!modulus.is_zero(), "modpow: zero modulus");
        if modulus.is_one() {
            return Nat::zero();
        }
        if modulus.limbs.len() < 2 {
            self.modpow_simple(exp, modulus)
        } else if !modulus.is_even() {
            self.modpow_montgomery(exp, modulus)
        } else {
            self.modpow_barrett(exp, modulus)
        }
    }

    /// Square-and-multiply using a precomputed Barrett [`Reciprocal`] (works for
    /// any multi-limb modulus; used for even moduli, where Montgomery does not
    /// apply).
    fn modpow_barrett(&self, exp: &Nat, modulus: &Nat) -> Nat {
        let recip = Reciprocal::new(modulus);
        let base = self.div_rem(modulus).expect("non-zero").1;
        modpow_windowed(base, Nat::one(), exp, |a, b| recip.reduce(&a.mul(b)))
    }

    /// Square-and-multiply with a division-based reduction after each step.
    fn modpow_simple(&self, exp: &Nat, modulus: &Nat) -> Nat {
        let base = self.div_rem(modulus).expect("non-zero modulus").1;
        modpow_windowed(base, Nat::one(), exp, |a, b| {
            a.mul(b).div_rem(modulus).expect("non-zero").1
        })
    }

    /// Montgomery-reduction modpow for an odd `modulus > 1`.
    fn modpow_montgomery(&self, exp: &Nat, modulus: &Nat) -> Nat {
        let k = modulus.limbs.len();
        let m = modulus.limbs.as_slice(); // exactly k words, top non-zero, odd
        let n0inv = inv_mod_2_64(m[0]).wrapping_neg(); // −m⁻¹ mod 2^64

        // R = 2^(64k); R² mod m converts a residue into Montgomery form via one
        // CIOS multiply. `1` in Montgomery form is R mod m = CIOS(1, R²).
        let r = Nat::one().shl(k as u64 * LIMB_BITS as u64);
        let r2 = r.mul(&r).div_rem(modulus).expect("non-zero").1;

        let base_mod = self.div_rem(modulus).expect("non-zero").1;
        let base = mont_mul_cios(&base_mod, &r2, m, n0inv);
        let one_mont = mont_mul_cios(&Nat::one(), &r2, m, n0inv);
        // The windowed ladder squares via `mulmod(&r, &r)`; detect that aliasing
        // and take the cheaper dedicated Montgomery squaring.
        let result = modpow_windowed(base, one_mont, exp, |a, b| {
            if core::ptr::eq(a, b) {
                mont_sqr(a, m, n0inv)
            } else {
                mont_mul_cios(a, b, m, n0inv)
            }
        });
        // Back out of Montgomery form: value = CIOS(result, 1).
        mont_mul_cios(&result, &Nat::one(), m, n0inv)
    }

    /// Returns the smallest prime strictly greater than `self`, found by
    /// scanning odd candidates with the Miller–Rabin test.
    pub fn next_prime(&self, rng: &mut impl crate::random::RandomSource) -> Nat {
        let two = Nat::from_u64(2);
        if self.cmp_ref(&two) == Ordering::Less {
            return two; // next prime after 0 or 1
        }
        let mut c = self.add(&Nat::one());
        if c.is_even() {
            c = c.add(&Nat::one()); // start at an odd candidate ≥ 3
        }
        loop {
            if c.is_probable_prime(40, rng) {
                return c;
            }
            c = c.add(&two);
        }
    }

    /// Returns the largest prime strictly less than `self`, or `None` if there
    /// is none (`self <= 2`).
    pub fn prev_prime(&self, rng: &mut impl crate::random::RandomSource) -> Option<Nat> {
        let two = Nat::from_u64(2);
        if self.cmp_ref(&two) != Ordering::Greater {
            return None;
        }
        if self.cmp_ref(&Nat::from_u64(3)) == Ordering::Equal {
            return Some(two);
        }
        let mut c = self.checked_sub(&Nat::one()).unwrap();
        if c.is_even() {
            c = c.checked_sub(&Nat::one()).unwrap();
        }
        loop {
            if c.cmp_ref(&two) == Ordering::Less {
                return Some(two);
            }
            if c.is_probable_prime(40, rng) {
                return Some(c);
            }
            c = c.checked_sub(&two).unwrap_or_else(Nat::zero);
        }
    }

    /// Deterministic Baillie–PSW probable-primality test (no RNG needed): a
    /// strong Miller–Rabin test to base 2 plus a strong Lucas test. There are no
    /// known counterexamples, and it is exact for all `self < 2^64`.
    pub fn is_prime_bpsw(&self) -> bool {
        let two = Nat::from_u64(2);
        let three = Nat::from_u64(3);
        if self.cmp_ref(&two) == Ordering::Less {
            return false;
        }
        if self.cmp_ref(&three) != Ordering::Greater {
            return true; // 2 or 3
        }
        if self.is_even() {
            return false;
        }
        // A perfect square is composite and would break the Lucas D search.
        let r = self.isqrt();
        if r.square() == *self {
            return false;
        }
        miller_rabin_witness(&two, self) && lucas_strong(self)
    }

    /// Returns the prime factorization of `self` as a sorted list of prime
    /// factors *with multiplicity* (empty for `0` and `1`).
    ///
    /// Small factors are removed by trial division; the rest are split with
    /// Pollard's rho and confirmed prime with Baillie–PSW. Practical for numbers
    /// with factors up to ~20 digits; genuinely hard semiprimes are, as always,
    /// hard.
    pub fn factorize(&self) -> Vec<Nat> {
        let mut factors = Vec::new();
        if self.is_zero() || self.is_one() {
            return factors;
        }
        let mut n = self.clone();
        // Factor out 2, then odd trial divisors up to a small bound.
        while n.is_even() {
            factors.push(Nat::from_u64(2));
            n = n.shr(1);
        }
        let mut d = 3u64;
        while d <= 4096 {
            let dn = Nat::from_u64(d);
            if dn.mul(&dn).cmp_ref(&n) == Ordering::Greater {
                break;
            }
            loop {
                let (q, r) = n.div_rem(&dn).expect("non-zero");
                if r.is_zero() {
                    factors.push(dn.clone());
                    n = q;
                } else {
                    break;
                }
            }
            d += 2;
        }
        // Split whatever remains with Pollard's rho.
        let mut stack = Vec::new();
        if !n.is_one() {
            stack.push(n);
        }
        while let Some(m) = stack.pop() {
            if m.is_prime_bpsw() {
                factors.push(m);
                continue;
            }
            let factor = pollard_rho(&m);
            let cofactor = m.div_rem(&factor).expect("non-zero").0;
            stack.push(factor);
            stack.push(cofactor);
        }
        factors.sort();
        factors
    }

    /// Miller–Rabin probable-primality test with `rounds` random witnesses.
    ///
    /// Deterministic for the tiny cases; for larger `self` the probability of a
    /// composite passing is at most `4^-rounds`.
    pub fn is_probable_prime(
        &self,
        rounds: u32,
        rng: &mut impl crate::random::RandomSource,
    ) -> bool {
        let two = Nat::from_u64(2);
        let three = Nat::from_u64(3);
        if self.cmp_ref(&two) == Ordering::Less {
            return false;
        }
        if self.cmp_ref(&three) != Ordering::Greater {
            return true; // 2 or 3
        }
        if self.is_even() {
            return false;
        }
        let one = Nat::one();
        let n1 = self.checked_sub(&one).expect("self >= 1");
        let s = n1.trailing_zeros();
        let d = n1.shr(s);
        let n3 = self.checked_sub(&three).expect("self >= 3");

        'witness: for _ in 0..rounds {
            let a = two.add(&Nat::random_below(&n3, rng).unwrap_or_else(Nat::zero));
            let mut x = a.modpow(&d, self);
            if x == one || x == n1 {
                continue;
            }
            for _ in 1..s {
                x = x.square().div_rem(self).expect("non-zero").1;
                if x == n1 {
                    continue 'witness;
                }
            }
            return false; // definitely composite
        }
        true
    }
}

/// A precomputed reciprocal of a fixed modulus for fast repeated reduction
/// (Barrett's method / Möller–Granlund division by an invariant).
///
/// Building one costs a division; each [`Reciprocal::reduce`] then costs a
/// couple of multiplications, so it pays off when reducing many values modulo
/// the same modulus (e.g. modular exponentiation, hashing).
#[derive(Clone, Debug)]
pub struct Reciprocal {
    modulus: Nat,
    /// `μ = ⌊2^(128k) / modulus⌋`, where `k = modulus.limbs.len()`.
    mu: Nat,
    kbits: u64,
}

impl Reciprocal {
    /// Precomputes the reciprocal of `modulus`. Panics if `modulus` is zero.
    pub fn new(modulus: &Nat) -> Reciprocal {
        assert!(!modulus.is_zero(), "Reciprocal: zero modulus");
        let kbits = modulus.limbs.len() as u64 * LIMB_BITS as u64;
        let mu = Nat::one()
            .shl(2 * kbits)
            .div_rem(modulus)
            .expect("non-zero")
            .0;
        Reciprocal {
            modulus: modulus.clone(),
            mu,
            kbits,
        }
    }

    /// Returns the modulus.
    #[inline]
    pub fn modulus(&self) -> &Nat {
        &self.modulus
    }

    /// Returns `x mod modulus`.
    ///
    /// Requires `x < modulus²` — the range that arises when reducing a product
    /// of two already-reduced values. For a general dividend, use
    /// [`Nat::div_rem`].
    pub fn reduce(&self, x: &Nat) -> Nat {
        let m = &self.modulus;
        let kbits = self.kbits;
        let q3 = x
            .shr(kbits - LIMB_BITS as u64)
            .mul(&self.mu)
            .shr(kbits + LIMB_BITS as u64);
        let mask = kbits + LIMB_BITS as u64;
        let r1 = x.low_bits(mask);
        let r2 = q3.mul(m).low_bits(mask);
        let mut r = if r1.cmp_ref(&r2) != Ordering::Less {
            r1.checked_sub(&r2).unwrap()
        } else {
            r1.add(&Nat::one().shl(mask)).checked_sub(&r2).unwrap()
        };
        while r.cmp_ref(m) != Ordering::Less {
            r = r.checked_sub(m).unwrap();
        }
        r
    }
}

/// Pollard's rho: returns a non-trivial factor of the odd composite `n > 1`
/// (Floyd cycle detection over `f(x) = x² + c mod n`, retrying with larger `c`).
fn pollard_rho(n: &Nat) -> Nat {
    if n.is_even() {
        return Nat::from_u64(2);
    }
    let one = Nat::one();
    let recip = Reciprocal::new(n);
    let mut c = 1u64;
    loop {
        let f = |x: &Nat| recip.reduce(&x.square().add(&Nat::from_u64(c)));
        let (mut x, mut y) = (Nat::from_u64(2), Nat::from_u64(2));
        let mut d = one.clone();
        while d == one {
            x = f(&x);
            y = f(&f(&y));
            let diff = if x.cmp_ref(&y) != Ordering::Less {
                x.checked_sub(&y).unwrap()
            } else {
                y.checked_sub(&x).unwrap()
            };
            d = if diff.is_zero() {
                n.clone()
            } else {
                diff.gcd(n)
            };
        }
        if d != *n {
            return d;
        }
        c += 1; // cycle without a factor: try a different polynomial
    }
}

/// Strong Miller–Rabin test to a single witness `a` for odd `n > 2`; returns
/// `true` if `a` is not a witness to compositeness.
fn miller_rabin_witness(a: &Nat, n: &Nat) -> bool {
    let one = Nat::one();
    let n1 = n.checked_sub(&one).expect("n >= 1");
    let s = n1.trailing_zeros();
    let d = n1.shr(s);
    let mut x = a.modpow(&d, n);
    if x == one || x == n1 {
        return true;
    }
    for _ in 1..s {
        x = x.square().div_rem(n).expect("non-zero").1;
        if x == n1 {
            return true;
        }
    }
    false
}

/// Jacobi symbol `(d/n)` for odd `n > 0`.
pub(crate) fn jacobi(d: &crate::int::Int, n: &Nat) -> i32 {
    let mut a = d.rem_euclid(&crate::int::Int::from(n.clone())).magnitude();
    let mut m = n.clone();
    let mut result = 1i32;
    let lo = |x: &Nat| x.limbs.first().copied().unwrap_or(0);
    while !a.is_zero() {
        while a.is_even() {
            a = a.shr(1);
            let r = lo(&m) & 7;
            if r == 3 || r == 5 {
                result = -result;
            }
        }
        core::mem::swap(&mut a, &mut m);
        if lo(&a) & 3 == 3 && lo(&m) & 3 == 3 {
            result = -result;
        }
        a = a.div_rem(&m).expect("m non-zero").1;
    }
    if m.is_one() { result } else { 0 }
}

/// Strong Lucas probable-primality test with Selfridge parameters, for odd
/// `n > 3` that is not a perfect square.
fn lucas_strong(n: &Nat) -> bool {
    use crate::int::Int;

    // Selfridge D: first of 5, −7, 9, −11, … with (D/n) == −1.
    let mut d_val: i64 = 5;
    loop {
        let j = jacobi(&Int::from_i64(d_val), n);
        if j == -1 {
            break;
        }
        if j == 0 {
            // gcd(D, n) > 1: a proper factor means composite, but if n divides D
            // (only possible for tiny n) skip to the next candidate.
            let g = Nat::from_u64(d_val.unsigned_abs()).gcd(n);
            if g.cmp_ref(n) != Ordering::Equal {
                return false;
            }
        }
        d_val = if d_val > 0 { -(d_val + 2) } else { -d_val + 2 };
    }
    let d = Int::from_i64(d_val);
    let p = Int::ONE;
    let q = Int::ONE.sub(&d).div_trunc(&Int::from_i64(4)); // (1 − D)/4, exact

    let modn = Int::from(n.clone());
    let two = Int::from_i64(2);
    let half_mod = |x: &Int| -> Int {
        let xm = x.rem_euclid(&modn).magnitude();
        if xm.is_even() {
            Int::from(xm.shr(1))
        } else {
            Int::from(xm.add(n).shr(1))
        }
    };

    // n + 1 = dd · 2^s, dd odd.
    let np1 = n.add(&Nat::one());
    let s = np1.trailing_zeros();
    let dd = np1.shr(s);

    // Build U_dd, V_dd via the Lucas doubling chain over the bits of dd.
    let mut u = Int::ONE;
    let mut v = p.clone();
    let mut qk = q.rem_euclid(&modn);
    for i in (0..dd.bit_len().saturating_sub(1)).rev() {
        u = u.mul(&v).rem_euclid(&modn);
        v = v.mul(&v).sub(&two.mul(&qk)).rem_euclid(&modn);
        qk = qk.mul(&qk).rem_euclid(&modn);
        if dd.bit(i) {
            let u_new = half_mod(&p.mul(&u).add(&v));
            let v_new = half_mod(&d.mul(&u).add(&v));
            u = u_new.rem_euclid(&modn);
            v = v_new.rem_euclid(&modn);
            qk = qk.mul(&q).rem_euclid(&modn);
        }
    }

    if u.is_zero() || v.is_zero() {
        return true;
    }
    for _ in 1..s {
        v = v.mul(&v).sub(&two.mul(&qk)).rem_euclid(&modn);
        qk = qk.mul(&qk).rem_euclid(&modn);
        if v.is_zero() {
            return true;
        }
    }
    false
}

/// Binary GCD on two double words (internal helper for the small-operand path).
fn u128_gcd(mut u: u128, mut v: u128) -> u128 {
    if u == 0 {
        return v;
    }
    if v == 0 {
        return u;
    }
    let shift = (u | v).trailing_zeros();
    u >>= u.trailing_zeros();
    loop {
        v >>= v.trailing_zeros();
        if u > v {
            core::mem::swap(&mut u, &mut v);
        }
        v -= u;
        if v == 0 {
            break;
        }
    }
    u << shift
}

/// Binary GCD on two machine words.
pub fn u64_gcd(mut u: u64, mut v: u64) -> u64 {
    if u == 0 {
        return v;
    }
    if v == 0 {
        return u;
    }
    let shift = (u | v).trailing_zeros();
    u >>= u.trailing_zeros();
    loop {
        v >>= v.trailing_zeros();
        if u > v {
            core::mem::swap(&mut u, &mut v);
        }
        v -= u;
        if v == 0 {
            break;
        }
    }
    u << shift
}

/// Binary GCD on two 32-bit machine words.
#[inline]
pub fn u_gcd(u: u32, v: u32) -> u32 {
    u64_gcd(u as u64, v as u64) as u32
}

impl PartialOrd for Nat {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Nat {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_ref(other)
    }
}

macro_rules! nat_from_small_unsigned {
    ($($t:ty)*) => {$(
        impl From<$t> for Nat {
            #[inline]
            fn from(v: $t) -> Self { Nat::from_u64(v as u64) }
        }
    )*};
}
nat_from_small_unsigned!(u8 u16 u32 u64 usize);

impl From<u128> for Nat {
    #[inline]
    fn from(v: u128) -> Self {
        Nat::from_u128(v)
    }
}

impl FromStr for Nat {
    type Err = Error;

    /// Parses a non-negative decimal integer. An empty string, or any character
    /// that is not an ASCII digit, is a [`Error::Parse`].
    fn from_str(s: &str) -> Result<Self> {
        // Reject a leading sign here (base-10 naturals only), then use the
        // shared sub-quadratic radix parser.
        if s.starts_with(['+', '-']) {
            return Err(Error::Parse);
        }
        parse_radix(s, 10)
    }
}

impl fmt::Display for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        f.write_str(&self.to_radix_string(10))
    }
}

impl fmt::LowerHex for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let mut it = self.limbs.iter().rev();
        write!(f, "{:x}", it.next().expect("non-empty checked above"))?;
        for limb in it {
            write!(f, "{limb:016x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Nat({self})")
    }
}

impl core::ops::Add for &Nat {
    type Output = Nat;
    #[inline]
    fn add(self, rhs: &Nat) -> Nat {
        Nat::add(self, rhs)
    }
}

impl core::ops::Mul for &Nat {
    type Output = Nat;
    #[inline]
    fn mul(self, rhs: &Nat) -> Nat {
        Nat::mul(self, rhs)
    }
}

// Owned `+`/`*` (and assign forms). Subtraction is intentionally absent —
// naturals are not closed under it; use `checked_sub`.
impl core::ops::Add for Nat {
    type Output = Nat;
    #[inline]
    fn add(self, rhs: Nat) -> Nat {
        Nat::add(&self, &rhs)
    }
}

impl core::ops::Mul for Nat {
    type Output = Nat;
    #[inline]
    fn mul(self, rhs: Nat) -> Nat {
        Nat::mul(&self, &rhs)
    }
}

impl core::ops::AddAssign for Nat {
    #[inline]
    fn add_assign(&mut self, rhs: Nat) {
        *self = Nat::add(self, &rhs);
    }
}

impl core::ops::MulAssign for Nat {
    #[inline]
    fn mul_assign(&mut self, rhs: Nat) {
        *self = Nat::mul(self, &rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    #[test]
    fn inv_mod_2_64_is_correct() {
        let mut x = 1u64;
        for _ in 0..100_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1) | 1; // odd
            assert_eq!(x.wrapping_mul(inv_mod_2_64(x)), 1, "inverse of {x}");
        }
        assert_eq!(inv_mod_2_64(1), 1);
        assert_eq!(3u64.wrapping_mul(inv_mod_2_64(3)), 1);
    }

    #[test]
    fn goldilocks_reduce_matches_modulo() {
        let p = GOLDILOCKS as u128;
        // Edge and structured values around the reduction's fold boundaries.
        let edges: &[u64] = &[
            0,
            1,
            GOLDILOCKS - 1,
            GOLDILOCKS,
            0xFFFF_FFFF,
            0x1_0000_0000,
            0xFFFF_FFFF_0000_0000,
            u64::MAX,
            0x1234_5678_9ABC_DEF0,
        ];
        for &a in edges {
            for &b in edges {
                let x = a as u128 * b as u128;
                assert_eq!(gf_reduce128(x), (x % p) as u64, "reduce({a}·{b})");
            }
        }
        // Pseudo-random coverage across the full 128-bit product range.
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            s
        };
        for _ in 0..200_000 {
            let (a, b) = (next() % GOLDILOCKS, next() % GOLDILOCKS);
            let x = a as u128 * b as u128;
            assert_eq!(gf_reduce128(x), (x % p) as u64);
            // Also full-width u128 inputs (products can be up to (p-1)^2 < 2^128).
            let y = ((next() as u128) << 64) | next() as u128;
            assert_eq!(gf_reduce128(y), (y % p) as u64);
        }
    }

    /// Reference bit-at-a-time long division, kept only for differential testing
    /// against the production Algorithm-D path.
    fn div_rem_binary(a: &Nat, b: &Nat) -> (Nat, Nat) {
        assert!(!b.is_zero());
        if a.cmp_ref(b) == Ordering::Less {
            return (Nat::zero(), a.clone());
        }
        let one = Nat::one();
        let mut q = Nat::zero();
        let mut r = Nat::zero();
        for i in (0..a.bit_len()).rev() {
            r = r.shl(1);
            if a.bit(i) {
                r = r.add(&one);
            }
            q = q.shl(1);
            if r.cmp_ref(b) != Ordering::Less {
                r = r.checked_sub(b).unwrap();
                q = q.add(&one);
            }
        }
        (q, r)
    }

    fn n(s: &str) -> Nat {
        Nat::from_str(s).unwrap()
    }

    #[test]
    fn knuth_matches_binary_reference() {
        // A spread of dividend/divisor sizes, including multi-limb divisors,
        // exact multiples, and near-boundary values.
        let cases = [
            (
                "340282366920938463463374607431768211456",
                "18446744073709551616",
            ),
            (
                "123456789012345678901234567890123456789",
                "98765432109876543210",
            ),
            ("100000000000000000000000000000000000000", "3"),
            (
                "18446744073709551617000000000000000000000",
                "18446744073709551617",
            ),
            (
                "999999999999999999999999999999999999999999",
                "1000000000000000000001",
            ),
        ];
        for (a_s, b_s) in cases.iter() {
            let (a, b) = (n(a_s), n(b_s));
            let (q, r) = a.div_rem(&b).unwrap();
            let (rq, rr) = div_rem_binary(&a, &b);
            assert_eq!(q, rq, "quotient {a_s}/{b_s}");
            assert_eq!(r, rr, "remainder {a_s}/{b_s}");
            // Reconstruction and range.
            assert_eq!(q.mul(&b).add(&r), a);
            assert!(r.cmp_ref(&b) == Ordering::Less);
        }
    }

    #[test]
    fn ntt_matches_toom3() {
        // NTT multiplication must agree with the (verified) Toom-3 path, and
        // with a value computed a different way.
        let p = Nat::from_u64(10).pow(4000); // ~13k bits, ~208 limbs
        let q = Nat::from_u64(10).pow(4100);
        let mut expected = String::from("1");
        expected.push_str(&"0".repeat(8100));
        assert_eq!(mul_ntt(&p, &q), Nat::from_str(&expected).unwrap());

        let mut state = 0x0f0f_1234_dead_beefu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let build = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let bytes: Vec<u8> = (0..cnt * 8).map(|_| f() as u8).collect();
            Nat::from_bytes_le(&bytes)
        };
        for _ in 0..8 {
            let a = build(200 + (next() % 400) as usize, &mut next);
            let b = build(200 + (next() % 400) as usize, &mut next);
            assert_eq!(mul_ntt(&a, &b), a.mul_toom3(&b), "NTT vs Toom-3 mismatch");
        }
    }

    #[test]
    fn burnikel_ziegler_matches_knuth() {
        // Differential: BZ recursive division must match Knuth Algorithm D over
        // random large operands, and satisfy a == q·b + r with r < b.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let build = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let bytes: Vec<u8> = (0..cnt * 8).map(|_| f() as u8).collect();
            Nat::from_bytes_le(&bytes)
        };
        for _ in 0..25 {
            // Divisor 70–110 limbs (crosses BZ recursion), dividend larger.
            let b = build(70 + (next() % 40) as usize, &mut next);
            let extra = build(30 + (next() % 90) as usize, &mut next);
            let a = b.mul(&extra).add(&build(40, &mut next));
            if b.is_zero() || a.cmp_ref(&b) != Ordering::Greater {
                continue;
            }
            let (q_bz, r_bz) = bz_div_rem(&a, &b);
            let (q_kn, r_kn) = a.div_rem_knuth(&b);
            assert_eq!(q_bz, q_kn, "BZ quotient mismatch");
            assert_eq!(r_bz, r_kn, "BZ remainder mismatch");
            assert_eq!(q_bz.mul(&b).add(&r_bz), a);
            assert!(r_bz.cmp_ref(&b) == Ordering::Less);
        }
    }

    #[test]
    #[ignore = "measurement only: cargo test -- --ignored --nocapture measure_mul"]
    fn measure_mul_crossovers() {
        use std::time::Instant;
        let mkbig = |limbs: usize| -> Nat {
            let bytes: Vec<u8> = (0..limbs * 8)
                .map(|i| (i * 2654435761usize) as u8)
                .collect();
            Nat::from_bytes_le(&bytes)
        };
        let bench = |f: &dyn Fn() -> Nat| {
            let mut best = core::time::Duration::MAX;
            let _ = f();
            for _ in 0..6 {
                let t = Instant::now();
                let mut r = f();
                for _ in 0..7 {
                    r = f();
                }
                let _ = r.limbs.len();
                best = best.min(t.elapsed() / 8);
            }
            best
        };
        for &sz in &[
            48usize, 96, 112, 128, 160, 224, 320, 448, 640, 800, 1024, 1600, 2400, 3200, 4000,
            8000, 16000,
        ] {
            let a = mkbig(sz);
            let b = mkbig(sz + 1);
            let school = if sz <= 2000 {
                bench(&|| a.mul_schoolbook(&b))
            } else {
                Default::default()
            };
            let kara = bench(&|| a.mul_karatsuba(&b));
            let t3 = bench(&|| a.mul_toom3(&b));
            let t4 = bench(&|| a.mul_toom4(&b));
            let ntt = bench(&|| mul_ntt(&a, &b));
            std::println!(
                "sz={sz:<6} school={school:>11?} kara={kara:>11?} toom3={t3:>11?} toom4={t4:>11?} ntt={ntt:>11?}"
            );
        }
    }

    #[test]
    fn toom_direct_matches_schoolbook() {
        // Exercise the Toom-3 and Toom-4 code paths directly (independent of the
        // dispatch thresholds), differentially against schoolbook.
        let mk = |limbs: usize, seed: u64| {
            let mut s = seed;
            let bytes: Vec<u8> = (0..limbs * 8)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    s as u8
                })
                .collect();
            Nat::from_bytes_le(&bytes)
        };
        let (a3, b3) = (mk(300, 1), mk(280, 2));
        assert_eq!(a3.mul_toom3(&b3), a3.mul_schoolbook(&b3));
        let (a4, b4) = (mk(500, 3), mk(470, 4));
        assert_eq!(a4.mul_toom4(&b4), a4.mul_schoolbook(&b4));
        // Odd/unbalanced sizes.
        let (a5, b5) = (mk(457, 5), mk(451, 6));
        assert_eq!(a5.mul_toom4(&b5), a5.mul_schoolbook(&b5));
        assert_eq!(a5.mul_toom3(&b5), a5.mul_schoolbook(&b5));
    }

    #[test]
    fn bpsw_matches_trial_division() {
        fn trial(n: u64) -> bool {
            if n < 2 {
                return false;
            }
            let mut i = 2u64;
            while i * i <= n {
                if n.is_multiple_of(i) {
                    return false;
                }
                i += 1;
            }
            true
        }
        for n in 0u64..3000 {
            assert_eq!(Nat::from_u64(n).is_prime_bpsw(), trial(n), "bpsw {n}");
        }
        // Large primes, a Mersenne prime, composites, and Carmichael numbers.
        assert!(n("1000000007").is_prime_bpsw());
        assert!(n("170141183460469231731687303715884105727").is_prime_bpsw()); // 2^127 − 1
        assert!(!n("1000000005").is_prime_bpsw());
        for c in ["561", "1105", "1729", "2465", "2821", "6601", "62745"] {
            assert!(!n(c).is_prime_bpsw(), "carmichael {c}");
        }
    }

    #[test]
    fn montgomery_matches_simple_modpow() {
        // Montgomery-reduction modpow must match the division-based version for
        // random bases/exponents and odd moduli of assorted sizes.
        let mut state = 0xabcd_1234_5678_9999u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let build = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let bytes: Vec<u8> = (0..cnt * 8).map(|_| f() as u8).collect();
            Nat::from_bytes_le(&bytes)
        };
        for _ in 0..20 {
            let base = build(2 + (next() % 8) as usize, &mut next);
            let exp = build(1 + (next() % 4) as usize, &mut next);
            let mut m = build(2 + (next() % 6) as usize, &mut next);
            if m.is_even() {
                m = m.add(&Nat::one()); // make odd
            }
            if m.limbs.len() < 2 {
                continue;
            }
            assert_eq!(
                base.modpow_montgomery(&exp, &m),
                base.modpow_simple(&exp, &m),
                "montgomery vs simple modpow"
            );
            // Barrett works for any multi-limb modulus (test the even case).
            let m_even = m.add(&Nat::one());
            if m_even.limbs.len() >= 2 {
                assert_eq!(
                    base.modpow_barrett(&exp, &m_even),
                    base.modpow_simple(&exp, &m_even),
                    "barrett vs simple modpow (even modulus)"
                );
            }
        }
    }

    #[test]
    fn lehmer_matches_binary_gcd() {
        // Deterministic pseudo-random large pairs; Lehmer must match binary GCD.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..40 {
            // Build multi-limb operands (20–40 limbs) so Lehmer is exercised.
            let build = |cnt: usize, f: &mut dyn FnMut() -> u64| {
                let bytes: Vec<u8> = (0..cnt * 8).map(|_| f() as u8).collect();
                Nat::from_bytes_le(&bytes)
            };
            let a = build(20 + (next() % 20) as usize, &mut next);
            let b = build(20 + (next() % 20) as usize, &mut next);
            if a.is_zero() || b.is_zero() {
                continue;
            }
            let g_lehmer = a.gcd_lehmer(&b);
            let g_binary = a.gcd_binary(&b);
            assert_eq!(g_lehmer, g_binary, "gcd mismatch");
            // g divides both.
            assert!(a.div_rem(&g_lehmer).unwrap().1.is_zero());
            assert!(b.div_rem(&g_lehmer).unwrap().1.is_zero());
        }
        // A case with a large known common factor.
        let common = Nat::from_u64(10).pow(50);
        let a = common.mul(&Nat::from_u64(7).pow(30));
        let b = common.mul(&Nat::from_u64(11).pow(25));
        assert_eq!(a.gcd_lehmer(&b), common);
    }

    #[test]
    fn karatsuba_matches_schoolbook() {
        // The slice-recursion Karatsuba (mul and square) must agree with
        // schoolbook across balanced, unbalanced, and threshold-edge sizes.
        let mk = |limbs: usize, seed: u64| {
            let mut s = seed;
            let bytes: Vec<u8> = (0..limbs * 8)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    s as u8
                })
                .collect();
            Nat::from_bytes_le(&bytes)
        };
        let sizes: &[(usize, usize)] = &[
            (128, 128),
            (128, 129),
            (129, 257),
            (200, 400),
            (130, 1000),
            (333, 334),
            (512, 512),
            (150, 900),
        ];
        for &(x, y) in sizes {
            let (a, b) = (mk(x, x as u64 + 1), mk(y, y as u64 + 7));
            assert_eq!(
                a.mul_karatsuba(&b),
                a.mul_schoolbook(&b),
                "kara mul {x}x{y}"
            );
            assert_eq!(
                b.mul_karatsuba(&a),
                b.mul_schoolbook(&a),
                "kara mul {y}x{x}"
            );
        }
        for &limbs in &[128usize, 129, 255, 256, 300, 511] {
            let a = mk(limbs, limbs as u64 * 31 + 5);
            assert_eq!(
                a.square_karatsuba(),
                a.mul_schoolbook(&a.clone()),
                "kara square {limbs}"
            );
        }
    }

    #[test]
    fn sqrt_rem_stress() {
        // Floor property and exact remainder across limb counts (all n mod 4
        // residues), random values, perfect squares, and off-by-one edges.
        let mut state = 0x5eed_5eed_5eed_5eedu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let check = |v: &Nat| {
            let s = v.isqrt();
            let s2 = s.square();
            assert!(s2.cmp_ref(v) != Ordering::Greater, "s² <= v for {v:?}");
            let next_sq = s.add(&Nat::one()).square();
            assert!(
                next_sq.cmp_ref(v) == Ordering::Greater,
                "(s+1)² > v for {v:?}"
            );
        };
        for limbs in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 15, 33, 64, 130] {
            for _ in 0..8 {
                let bytes: Vec<u8> = (0..limbs * 8).map(|_| next() as u8).collect();
                let v = Nat::from_bytes_le(&bytes);
                if v.is_zero() {
                    continue;
                }
                check(&v);
                // Perfect square and its neighbours.
                let sq = v.square();
                assert_eq!(sq.isqrt(), v, "isqrt of a perfect square");
                check(&sq.add(&Nat::one()));
                if let Some(m) = sq.checked_sub(&Nat::one()) {
                    check(&m);
                }
            }
        }
    }

    #[test]
    fn knuth_stress_products() {
        // Build large values and divide, checking the identity and the
        // multi-limb divisor path (10^k has many limbs).
        let ten_k = Nat::from_u64(10).pow(60); // ~200 bits, several limbs
        let big = Nat::from_u64(7).pow(200);
        let (q, r) = big.div_rem(&ten_k).unwrap();
        assert_eq!(q.mul(&ten_k).add(&r), big);
        assert!(r.cmp_ref(&ten_k) == Ordering::Less);

        // Exact division: (a*b)/b == a, remainder 0.
        let a = Nat::from_u64(3).pow(150);
        let b = Nat::from_u64(11).pow(80);
        let prod = a.mul(&b);
        let (q2, r2) = prod.div_rem(&b).unwrap();
        assert_eq!(q2, a);
        assert!(r2.is_zero());
    }
}
