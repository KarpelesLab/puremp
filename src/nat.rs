//! Arbitrary-precision natural numbers (unsigned integers).
//!
//! [`Nat`] stores magnitude as a little-endian `Vec` of 64-bit limbs,
//! kept in a canonical form with no trailing zero limbs (so the value zero is
//! the empty vector). That canonical form makes equality and ordering cheap and
//! lets the derived [`PartialEq`]/[`Eq`] be correct.
//!
//! This is the layer that carries the heavy limb-level algorithms: addition,
//! subtraction, multiplication (schoolbook → Karatsuba → Toom-3/4 → three-prime
//! NTT), squaring, division (single-limb, Knuth Algorithm D, and Burnikel–Ziegler),
//! shifts, GCD (binary → Lehmer), roots, and sub-quadratic radix I/O.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::int::{Int, Sign};
use crate::limb::{LIMB_BITS, Limb, adc, mac, sbb};

// Multiplication crossovers, tuned on x86-64 (Zen 5) from
// `measure_mul_crossovers` (see the test module) and a random-operand product
// benchmark, with the interleaved addmul_2 basecase and the allocation-free
// slice Karatsuba/Toom ladder: Karatsuba from ~28 limbs, Toom-3 from ~96,
// Toom-4 from ~280; squaring keeps its half-product schoolbook up to ~100
// limbs, then Toom-3 (~100) and Toom-4 (~300) squaring.
// The NTT hand-off is decided per-shape by `ntt_worthwhile` rather than a
// single size threshold (see `measure_ntt_paths`/`measure_ntt_unbalanced`);
// it takes over from roughly 1800–2000 limbs. Re-measure per platform to
// retune.

/// Operands with fewer than this many limbs use schoolbook multiplication.
const KARATSUBA_THRESHOLD: usize = 28;

/// Squarings of fewer than this many limbs use the symmetric schoolbook
/// square. It roughly halves the limb products, so it holds out against
/// Karatsuba much longer than the general product does. Must stay
/// `≥ KARATSUBA_THRESHOLD` (the shared scratch sizing assumes it).
const KARATSUBA_SQR_THRESHOLD: usize = 128;

/// Operands with at least this many limbs use Toom-3 (above Karatsuba).
const TOOM3_THRESHOLD: usize = 96;

/// Operands with at least this many limbs use Toom-4 (above Toom-3).
const TOOM4_THRESHOLD: usize = 280;

/// Squarings of at least this many limbs use Toom-3 squaring.
const TOOM3_SQR_THRESHOLD: usize = 100;

/// Squarings of at least this many limbs use Toom-4 squaring.
const TOOM4_SQR_THRESHOLD: usize = 300;

/// Products whose shorter operand has fewer limbs never use the NTT.
const NTT_MIN_LIMBS: usize = 500;

/// Number of limbs below which the Half-GCD recursion stops and reduces the
/// pair with the in-place double-word Lehmer base case (`hgcd_base`). The
/// allocation-free Lehmer pass is cheap enough that recursing below a few
/// hundred limbs (where the matrix products are small Karatsuba/schoolbook
/// ones) no longer pays.
const HGCD_RECURSION_BASE_LIMBS: usize = 400;

/// Production recursion base (in limbs). A test-only seam (see the `tests`
/// module's `hgcd_with_test_base`) can shrink this so small inputs still drive
/// the full recursion — `hgcd_make_valid`, `hgcd_straddle`, `hgcd_apply_split`
/// and the make-valid back-ups — millions of times cheaply.
#[cfg(not(test))]
#[inline(always)]
fn hgcd_recursion_base_limbs() -> usize {
    HGCD_RECURSION_BASE_LIMBS
}

#[cfg(test)]
std::thread_local! {
    /// Test-only override for the Half-GCD recursion base (per-thread, so
    /// parallel tests stay isolated).
    static HGCD_TEST_BASE: core::cell::Cell<usize> =
        const { core::cell::Cell::new(HGCD_RECURSION_BASE_LIMBS) };
}

#[cfg(test)]
fn hgcd_recursion_base_limbs() -> usize {
    HGCD_TEST_BASE.with(|c| c.get())
}

/// Runs `f` with the Half-GCD recursion base shrunk to `base` limbs (>= 2),
/// restoring the previous value afterwards. Lets the differential tests drive
/// the full recursion — make-valid back-ups, straddle, apply-split — on tiny
/// operands, so millions of pairs cost little yet exercise the same code paths
/// as production's 128-limb base.
#[cfg(test)]
fn hgcd_with_test_base<R>(base: usize, f: impl FnOnce() -> R) -> R {
    let prev = HGCD_TEST_BASE.with(|c| c.replace(base.max(2)));
    let out = f();
    HGCD_TEST_BASE.with(|c| c.set(prev));
    out
}

/// Smallest operand (in limbs) for which [`Nat::gcd`] engages the subquadratic
/// Half-GCD driver; smaller inputs stay on the (faster there) in-place Lehmer
/// path. Measured crossover: ~2200–2800 limbs (the two are within noise
/// there).
const HGCD_GCD_THRESHOLD: usize = 2400;

/// Smallest operand (in limbs, the smaller of the two) at which the extended
/// GCD ([`Int::extended_gcd`], [`Int::modinv`]) engages the Half-GCD cofactor
/// driver; below it the in-place extended Lehmer loop wins (measured
/// crossover ~1000 limbs).
pub(crate) const HGCD_MODINV_THRESHOLD: usize = 1000;

/// Limb floor for the extended-GCD Half-GCD driver: the reductions run down
/// to about this size, then the residual is finished by the extended Lehmer
/// loop (tracking both cofactors), which is faster below it.
const HGCD_EXTGCD_FLOOR: usize = 800;

/// The drivers run each Half-GCD reduction on the top `1 − NUM/DEN` of the
/// larger operand ([`hgcd_reduce_top`] with `k = bit_len·NUM/DEN` low bits
/// split off): measured best among 1/4 … 1/2.
const HGCD_SPLIT_NUM: u64 = 1;
const HGCD_SPLIT_DEN: u64 = 3;

// --- Three-prime NTT with full 64-bit limb coefficients ---
//
// Pollard's multi-modular convolution (Pollard 1971; Brent & Zimmermann, MCA
// §2.3): the limbs themselves are the polynomial coefficients, the cyclic
// convolution is computed modulo three primes `p = c·2^42 + 1` just below 2^62,
// and each product coefficient (`< min(la, lb)·2^128 < p₁p₂p₃ ≈ 2^186`) is
// recovered exactly by Garner's CRT. This replaced a single-prime transform over
// the Goldilocks field 2^64 − 2^32 + 1, which had to keep each coefficient
// below the prime and so could carry only 16–24 bits per point (and at most
// 2^32 points): three transforms of whole limbs measured ~5× faster than one
// transform of 2–3-byte digits, and the length limit rises to 2^42 points.
//
// The butterflies use lazy reduction in the style of Harvey ("Faster arithmetic
// for number-theoretic transforms", J. Symbolic Comput. 2014): values live in
// `[0, 2p)` or `[0, 4p)` (hence `p < 2^62`) and are only made canonical at the
// end. Products use Montgomery's REDC with twiddles stored as `w·2^64 mod p`, so
// one REDC yields `x·w` directly. The forward transform is decimation-in-
// frequency (natural order in, bit-reversed out) and the inverse is
// decimation-in-time (bit-reversed in, natural out), so no bit-reversal
// permutation is ever needed: the pointwise product does not care about order.

/// One NTT prime `p < 2^62` with its Montgomery constants.
struct MpPrime {
    /// The prime.
    p: u64,
    /// `p^-1 mod 2^64`.
    pinv: u64,
    /// `2^128 mod p` (converts into Montgomery form).
    r2: u64,
    /// A primitive root modulo `p`.
    g: u64,
}

/// Builds an [`MpPrime`], deriving the Montgomery constants at compile time.
const fn mp_prime(p: u64, g: u64) -> MpPrime {
    // Newton's iteration for the inverse mod 2^64 (each step doubles the bits).
    let mut pinv = p; // correct to 3 bits for odd p
    let mut i = 0;
    while i < 5 {
        pinv = pinv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(pinv)));
        i += 1;
    }
    let r = ((1u128 << 64) % p as u128) as u64;
    let r2 = ((r as u128 * r as u128) % p as u128) as u64;
    MpPrime { p, pinv, r2, g }
}

/// The three NTT primes `c·2^42 + 1` (each with `3 | c`), in Garner order.
const MP_PRIMES: [MpPrime; 3] = [
    mp_prime(0x3FFF_8400_0000_0001, 19),
    mp_prime(0x3FFF_5400_0000_0001, 5),
    mp_prime(0x3FFE_0400_0000_0001, 5),
];

/// Base-2 logarithm of the longest transform all three primes support.
const MP_MAX_LOG2: u32 = 42;

/// Transforms at or below this many points run every remaining stage on one
/// cache-resident block before moving on (depth-first), instead of sweeping
/// the whole array once per stage.
const MP_BLOCK: usize = 1 << 12;

/// Montgomery product `a·b·2^-64 mod p`, lazily reduced into `[1, 2p)`.
/// Requires `a·b < p·2^64` (e.g. `a < 4p, b < p` or `a, b < 2p`).
#[inline(always)]
fn mp_mul(a: u64, b: u64, p: u64, pinv: u64) -> u64 {
    let t = a as u128 * b as u128;
    let m = (t as u64).wrapping_mul(pinv);
    let mp = ((m as u128 * p as u128) >> 64) as u64;
    // t − m·p is an exact multiple of 2^64 (the low halves cancel), and both
    // high halves are below p, so the difference lies in (−p, p).
    ((t >> 64) as u64) + p - mp
}

/// Subtracts `m` once if `x ≥ m`.
#[inline(always)]
fn mp_csub(x: u64, m: u64) -> u64 {
    if x >= m { x - m } else { x }
}

/// `b^e mod p` by plain square-and-multiply (setup only).
fn mp_pow_slow(mut b: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = ((r as u128 * b as u128) % p as u128) as u64;
        }
        b = ((b as u128 * b as u128) % p as u128) as u64;
        e >>= 1;
    }
    r
}

/// `x·2^64 mod p` (canonical) for `x < p`.
#[inline]
fn mp_to_mont(x: u64, pr: &MpPrime) -> u64 {
    mp_csub(mp_mul(x, pr.r2, pr.p, pr.pinv), pr.p)
}

/// Fills `out[k] = w^k` (Montgomery form, canonical) for `w` in Montgomery
/// form. Only the first 64 powers form a serial chain; every later entry steps
/// from the one 64 places back by `w^64`, so 64 independent multiplication
/// chains overlap instead of one latency-bound chain.
fn mp_fill_powers(out: &mut [u64], w: u64, pr: &MpPrime) {
    const CHAINS: usize = 64;
    let (p, pinv) = (pr.p, pr.pinv);
    let mut x = mp_to_mont(1, pr);
    let head = out.len().min(CHAINS);
    for slot in &mut out[..head] {
        *slot = x;
        x = mp_csub(mp_mul(x, w, p, pinv), p);
    }
    // x = w^CHAINS whenever the table extends past the head.
    for k in CHAINS..out.len() {
        out[k] = mp_csub(mp_mul(out[k - CHAINS], x, p, pinv), p);
    }
}

/// Twiddle table for an `n`-point transform: for every stage of half-length
/// `h` (a power of two below `n`), `tw[h + k] = ω_{2h}^k·2^64 mod p` for
/// `0 ≤ k < h`, canonical. The largest stage is built by [`mp_fill_powers`]
/// and each smaller one takes every other entry of the next.
fn mp_twiddles(n: usize, pr: &MpPrime) -> Vec<u64> {
    let p = pr.p;
    let mut tw = alloc::vec![0u64; n.max(2)];
    let half = n / 2;
    if half == 0 {
        return tw;
    }
    let w = mp_to_mont(mp_pow_slow(pr.g, (p - 1) / n as u64, p), pr);
    mp_fill_powers(&mut tw[half..n], w, pr);
    let mut h = half / 2;
    while h >= 1 {
        for k in 0..h {
            tw[h + k] = tw[2 * h + 2 * k];
        }
        h /= 2;
    }
    tw
}

/// One decimation-in-frequency stage over every `len`-point block of `a`.
/// Inputs and outputs in `[0, 2p)`.
#[inline]
fn mp_dif_stage(a: &mut [u64], len: usize, tw: &[u64], p: u64, pinv: u64) {
    let half = len / 2;
    let p2 = 2 * p;
    if half == 1 {
        for pair in a.as_chunks_mut::<2>().0 {
            let (u, v) = (pair[0], pair[1]);
            pair[0] = mp_csub(u + v, p2);
            pair[1] = mp_csub(u + p2 - v, p2);
        }
        return;
    }
    let w = &tw[half..len];
    for blk in a.chunks_exact_mut(len) {
        let (lo, hi) = blk.split_at_mut(half);
        for ((x, y), &w) in lo.iter_mut().zip(hi.iter_mut()).zip(w) {
            let (u, v) = (*x, *y);
            *x = mp_csub(u + v, p2);
            *y = mp_mul(u + p2 - v, w, p, pinv);
        }
    }
}

/// One decimation-in-time stage of the inverse transform over every
/// `len`-point block of `a`. Inputs and outputs in `[0, 4p)`. The inverse
/// twiddle `ω^-k = −ω^(h−k)` (as `ω^h = −1`) is read from the forward table
/// with the butterfly's two outputs swapped, so no second table is needed.
#[inline]
fn mp_dit_stage(a: &mut [u64], len: usize, tw: &[u64], p: u64, pinv: u64) {
    let half = len / 2;
    let p2 = 2 * p;
    for blk in a.chunks_exact_mut(len) {
        let (lo, hi) = blk.split_at_mut(half);
        // k = 0: unit twiddle.
        let u = mp_csub(lo[0], p2);
        let q = mp_csub(hi[0], p2);
        lo[0] = u + q;
        hi[0] = u + p2 - q;
        // tw[len − k] for k = 1, 2, …, h − 1: the stage's table read backwards.
        let wr = tw[half + 1..len].iter().rev();
        for ((x, y), &w) in lo[1..].iter_mut().zip(hi[1..].iter_mut()).zip(wr) {
            let u = mp_csub(*x, p2);
            let q = mp_mul(*y, w, p, pinv); // y·ω^(h−k) = −y·ω^-k
            *x = u + p2 - q;
            *y = u + q;
        }
    }
}

/// Two consecutive decimation-in-frequency stages (`len`, then `len/2`) over
/// every `len`-point block of `a`, fused into one pass: each group of four
/// points `j, j + q, j + 2q, j + 3q` (`q = len/4`) goes through both
/// stages' butterflies in registers. The arithmetic is exactly that of two
/// [`mp_dif_stage`] calls (same twiddles, same lazy bounds), with half the
/// loads and stores. Inputs and outputs in `[0, 2p)`.
#[inline]
fn mp_dif_stage4(a: &mut [u64], len: usize, tw: &[u64], p: u64, pinv: u64) {
    let q = len / 4;
    let p2 = 2 * p;
    if q == 1 {
        // Stages 4 and 2: twiddles ω_4^0 = 1, ω_4^1 = tw[3], ω_2^0 = 1.
        let w3 = tw[3];
        for blk in a.as_chunks_mut::<4>().0 {
            let [x0, x1, x2, x3] = *blk;
            let y0 = mp_csub(x0 + x2, p2);
            let y2 = mp_csub(x0 + p2 - x2, p2);
            let y1 = mp_csub(x1 + x3, p2);
            let y3 = mp_mul(x1 + p2 - x3, w3, p, pinv);
            *blk = [
                mp_csub(y0 + y1, p2),
                mp_csub(y0 + p2 - y1, p2),
                mp_csub(y2 + y3, p2),
                mp_csub(y2 + p2 - y3, p2),
            ];
        }
        return;
    }
    let (w1, w2) = (&tw[q..2 * q], &tw[2 * q..len]);
    let (w2a, w2b) = w2.split_at(q);
    for blk in a.chunks_exact_mut(len) {
        let (b01, b23) = blk.split_at_mut(2 * q);
        let (b0, b1) = b01.split_at_mut(q);
        let (b2, b3) = b23.split_at_mut(q);
        let rows = b0
            .iter_mut()
            .zip(b1.iter_mut())
            .zip(b2.iter_mut())
            .zip(b3.iter_mut());
        let tws = w1.iter().zip(w2a).zip(w2b);
        for ((((x0, x1), x2), x3), ((&t1, &ta), &tb)) in rows.zip(tws) {
            let (u0, u1, u2, u3) = (*x0, *x1, *x2, *x3);
            // Stage len: (j, j + 2q) by ω^j, (j + q, j + 3q) by ω^(j+q).
            let y0 = mp_csub(u0 + u2, p2);
            let y2 = mp_mul(u0 + p2 - u2, ta, p, pinv);
            let y1 = mp_csub(u1 + u3, p2);
            let y3 = mp_mul(u1 + p2 - u3, tb, p, pinv);
            // Stage len/2: (j, j + q) and (j + 2q, j + 3q), both by ω_{len/2}^j.
            *x0 = mp_csub(y0 + y1, p2);
            *x1 = mp_mul(y0 + p2 - y1, t1, p, pinv);
            *x2 = mp_csub(y2 + y3, p2);
            *x3 = mp_mul(y2 + p2 - y3, t1, p, pinv);
        }
    }
}

/// Two consecutive decimation-in-time stages (`len/2`, then `len`) over every
/// `len`-point block of `a`, fused into one pass (the mirror image of
/// [`mp_dif_stage4`]): exactly the arithmetic of two [`mp_dit_stage`] calls.
/// Inputs and outputs in `[0, 4p)`.
#[inline]
fn mp_dit_stage4(a: &mut [u64], len: usize, tw: &[u64], p: u64, pinv: u64) {
    let q = len / 4;
    let p2 = 2 * p;
    // One DIT butterfly with twiddle w (y·w = −y·ω^-k, as in mp_dit_stage).
    let bfly = |x: u64, y: u64, w: u64| {
        let u = mp_csub(x, p2);
        let t = mp_mul(y, w, p, pinv);
        (u + p2 - t, u + t)
    };
    // Unit-twiddle butterfly (k = 0).
    let bfly1 = |x: u64, y: u64| {
        let u = mp_csub(x, p2);
        let t = mp_csub(y, p2);
        (u + t, u + p2 - t)
    };
    // Stage len/2 reads tw[len/2 − j] (j ≥ 1), stage len reads tw[len − k].
    let (wh, wl) = (&tw[q + 1..2 * q], &tw[2 * q + 1..len]);
    for blk in a.chunks_exact_mut(len) {
        let (b01, b23) = blk.split_at_mut(2 * q);
        let (b0, b1) = b01.split_at_mut(q);
        let (b2, b3) = b23.split_at_mut(q);
        // j = 0: unit twiddles in stage len/2 and for the (0, 2q) pair.
        {
            let (y0, y1) = bfly1(b0[0], b1[0]);
            let (y2, y3) = bfly1(b2[0], b3[0]);
            let (z0, z2) = bfly1(y0, y2);
            // Pair (q, 3q) at stage len: k = q, twiddle tw[len − q].
            let (z1, z3) = bfly(y1, y3, tw[len - q]);
            b0[0] = z0;
            b1[0] = z1;
            b2[0] = z2;
            b3[0] = z3;
        }
        let rows = b0[1..]
            .iter_mut()
            .zip(b1[1..].iter_mut())
            .zip(b2[1..].iter_mut())
            .zip(b3[1..].iter_mut());
        // For j = 1..q: stage len/2 twiddle tw[2q − j] (wh reversed); stage
        // len twiddles tw[4q − j] (wl reversed, first q − 1 entries from the
        // top) and tw[3q − j].
        let tws = wh
            .iter()
            .rev()
            .zip(wl[q..].iter().rev())
            .zip(wl[..q - 1].iter().rev());
        for ((((x0, x1), x2), x3), ((&th, &ta), &tb)) in rows.zip(tws) {
            let (y0, y1) = bfly(*x0, *x1, th);
            let (y2, y3) = bfly(*x2, *x3, th);
            let (z0, z2) = bfly(y0, y2, ta);
            let (z1, z3) = bfly(y1, y3, tb);
            *x0 = z0;
            *x1 = z1;
            *x2 = z2;
            *x3 = z3;
        }
    }
}
/// Runs the DIF stages `len, len/2, …, 2` over every `len`-point block of `a`:
/// breadth-first while blocks exceed [`MP_BLOCK`], then depth-first per block,
/// two stages per pass ([`mp_dif_stage4`]) wherever two remain.
fn mp_dif(a: &mut [u64], mut len: usize, tw: &[u64], p: u64, pinv: u64) {
    while len > MP_BLOCK {
        if len / 2 > MP_BLOCK {
            mp_dif_stage4(a, len, tw, p, pinv);
            len /= 4;
        } else {
            mp_dif_stage(a, len, tw, p, pinv);
            len /= 2;
        }
    }
    if len >= 2 {
        for blk in a.chunks_exact_mut(len) {
            let mut l = len;
            while l >= 4 {
                mp_dif_stage4(blk, l, tw, p, pinv);
                l /= 4;
            }
            if l == 2 {
                mp_dif_stage(blk, 2, tw, p, pinv);
            }
        }
    }
}

/// Runs the DIT stages `2, 4, …, len` over every `len`-point block of `a`
/// (the mirror image of [`mp_dif`]).
fn mp_dit(a: &mut [u64], len: usize, tw: &[u64], p: u64, pinv: u64) {
    let blk_len = len.min(MP_BLOCK);
    if blk_len >= 2 {
        for blk in a.chunks_exact_mut(blk_len) {
            // `done`: the block size the stages so far have covered.
            let mut done = 1;
            if blk_len.trailing_zeros() % 2 == 1 {
                mp_dit_stage(blk, 2, tw, p, pinv);
                done = 2;
            }
            while done * 4 <= blk_len {
                mp_dit_stage4(blk, done * 4, tw, p, pinv);
                done *= 4;
            }
        }
    }
    let mut done = blk_len;
    while done * 2 <= len {
        if done * 4 <= len {
            mp_dit_stage4(a, done * 4, tw, p, pinv);
            done *= 4;
        } else {
            mp_dit_stage(a, done * 2, tw, p, pinv);
            done *= 2;
        }
    }
}
/// Powers `ω_n^k·2^64 mod p` (`0 ≤ k < n`) of a primitive `n`-th root, for
/// the radix-3 stage of an `n = 3m`-point transform: it reads `ω^j`, `ω^2j`
/// and their inverses `ω^(n−j)`, `ω^(n−2j)`, and the cube roots of unity
/// `ω^m`, `ω^2m`.
fn mp_powers(n: usize, pr: &MpPrime) -> Vec<u64> {
    let p = pr.p;
    let w = mp_to_mont(mp_pow_slow(pr.g, (p - 1) / n as u64, p), pr);
    let mut out = alloc::vec![0u64; n];
    mp_fill_powers(&mut out, w, pr);
    out
}

/// Forward transform of `src` zero-padded to `n` points (values in `[0, 2p)`,
/// in an order only [`mp_inverse`] needs to understand). For `n = 2^k` this is
/// a plain DIF pass; when `src` fits in the low half its first stage is fused
/// into the load (the butterflies see a zero upper input). For `n = 3m` a
/// radix-3 DIF stage splits the input into three twisted `m`-point
/// sub-transforms: `y_r[j] = ω^(rj)·Σ_s x[j + sm]·ω₃^(rs)`, whose DFTs give
/// the outputs `X[3k + r]`.
fn mp_forward(src: &[Limb], n: usize, tw: &[u64], pow3: &[u64], pr: &MpPrime) -> Vec<u64> {
    let (p, pinv) = (pr.p, pr.pinv);
    let (p2, p4) = (2 * p, 4 * p);
    // A limb is below 2^64 < 5p; two conditional subtractions land in [0, 2p).
    let reduce = |x: u64| mp_csub(mp_csub(x, p4), p2);
    let mut a = alloc::vec![0u64; n];
    if !pow3.is_empty() {
        let m = n / 3;
        let (y0, rest) = a.split_at_mut(m);
        let (y1, y2) = rest.split_at_mut(m);
        if src.len() <= m {
            // x1 = x2 = 0: every output is x0 times a twiddle.
            for (j, &s) in src.iter().enumerate() {
                let x0 = reduce(s);
                y0[j] = x0;
                y1[j] = mp_mul(x0, pow3[j], p, pinv);
                y2[j] = mp_mul(x0, pow3[2 * j], p, pinv);
            }
        } else {
            // ω₃² = −1 − ω₃, so one product t = ω₃·(x1 − x2) serves both
            // y1 = x0 − x2 + t and y2 = x0 − x1 − t. With the inputs in
            // [0, p) and t in [1, 2p), every sum below stays under 4p < 2^64.
            let w3 = pow3[m];
            let at = |i: usize| src.get(i).map_or(0, |&s| mp_csub(reduce(s), p));
            for j in 0..m {
                let (x0, x1, x2) = (at(j), at(j + m), at(j + 2 * m));
                let t = mp_mul(x1 + p - x2, w3, p, pinv);
                y0[j] = mp_csub(x0 + x1 + x2, p2);
                y1[j] = mp_mul(x0 + p - x2 + t, pow3[j], p, pinv);
                y2[j] = mp_mul(x0 + 3 * p - x1 - t, pow3[2 * j], p, pinv);
            }
        }
        mp_dif(&mut a, m, tw, p, pinv);
        return a;
    }
    let half = n / 2;
    let mut len = n;
    if half >= 1 && src.len() <= half {
        let w = &tw[half..n];
        let (lo, hi) = a.split_at_mut(half);
        for (((x, y), &s), &w) in lo.iter_mut().zip(hi.iter_mut()).zip(src).zip(w) {
            let s = reduce(s);
            *x = s;
            *y = mp_mul(s, w, p, pinv);
        }
        len = half;
    } else {
        for (x, &s) in a.iter_mut().zip(src) {
            *x = reduce(s);
        }
    }
    mp_dif(&mut a, len, tw, p, pinv);
    a
}

/// Inverse of [`mp_forward`], unscaled: the result is `n` times the true
/// inverse, in natural order. Values in `[0, 4p)`. For `n = 3m` the three
/// `m`-point blocks are inverted first, then untwisted by `ω^-(rj)` and
/// recombined by the inverse 3-point DFT (with `ω₃^-1 = ω₃²`).
fn mp_inverse(a: &mut [u64], tw: &[u64], pow3: &[u64], pr: &MpPrime) {
    let (p, pinv) = (pr.p, pr.pinv);
    let n = a.len();
    if pow3.is_empty() {
        mp_dit(a, n, tw, p, pinv);
        return;
    }
    let m = n / 3;
    mp_dit(a, m, tw, p, pinv);
    let w3_inv = pow3[2 * m];
    let (y0, rest) = a.split_at_mut(m);
    let (y1, y2) = rest.split_at_mut(m);
    // As in the forward stage: canonical inputs keep every sum under 4p.
    // Untwisting by ω^-j and ω^-2j reads pow3[n − j] and pow3[n − 2j]
    // (pow3[0] for j = 0), walked downwards.
    let inv1 = core::iter::once(&pow3[0]).chain(pow3[n - m + 1..].iter().rev());
    let inv2 = core::iter::once(&pow3[0]).chain(pow3[n - 2 * m + 2..n - 1].iter().rev().step_by(2));
    let rows = y0.iter_mut().zip(y1.iter_mut()).zip(y2.iter_mut());
    for (((u0, u1), u2), (&w1, &w2)) in rows.zip(inv1.zip(inv2)) {
        let z0 = mp_csub(mp_csub(*u0, 2 * p), p);
        let z1 = mp_csub(mp_mul(*u1, w1, p, pinv), p);
        let z2 = mp_csub(mp_mul(*u2, w2, p, pinv), p);
        let t = mp_mul(z1 + p - z2, w3_inv, p, pinv);
        *u0 = z0 + z1 + z2;
        *u1 = z0 + p - z2 + t;
        *u2 = z0 + 3 * p - z1 - t;
    }
}

/// Transform length for a three-prime product with `len` coefficients: the
/// smaller of the next power of two and the next `3·2^k`, which caps the
/// zero-padding at a third instead of a half. `None` past the primes'
/// `2^42`-point (power-of-two part) limit.
fn mp_shape(len: usize) -> Option<usize> {
    let pow2 = len.checked_next_power_of_two()?.max(2);
    let n = match len.div_ceil(3).checked_next_power_of_two() {
        Some(m) if m >= 2 && m.checked_mul(3).is_some_and(|t| t < pow2) => 3 * m,
        _ => pow2,
    };
    let pow2_part = if n % 3 == 0 { n / 3 } else { n };
    ((pow2_part as u64) <= 1u64 << MP_MAX_LOG2).then_some(n)
}

/// Exact number of limb products [`mp_wrapped_coeffs`] spends on the top `s`
/// coefficients of an `la × lb` product (the `u`-th from the top has
/// `min(u + 1, la, lb)` terms).
fn mp_wrap_products(s: usize, la: usize, lb: usize) -> usize {
    let m = la.min(lb);
    if s <= m {
        s * (s + 1) / 2
    } else {
        m * (m + 1) / 2 + (s - m) * m
    }
}

/// Chooses the transform length for an `la × lb` product (`len = la + lb − 1`
/// coefficients): either the [`mp_shape`] length that holds every
/// coefficient, or a shorter `2^k`/`3·2^k` length `n ≥ max(la, lb)` whose
/// cyclic convolution wraps the top `len − n` coefficients onto the bottom
/// ones. Those few top coefficients involve only the top limbs of each
/// operand, so they are computed exactly by a small schoolbook
/// ([`mp_wrapped_coeffs`]) and subtracted back out — far cheaper than a
/// transform 1.33–1.5× longer when `len` lands just past a size step. The
/// choice minimizes a measured cycle estimate (see below). Returns
/// `(n, wrapped)` with `wrapped = len − n` or 0. A `square` runs two thirds
/// of the transforms, so the wrap correction must be correspondingly smaller
/// to pay off.
fn mp_plan(la: usize, lb: usize, square: bool) -> Option<(usize, usize)> {
    let len = la + lb - 1;
    let full = mp_shape(len)?;
    // Measured (x86-64, Zen 5): about 23 cycles per n·log₂n for the whole
    // three-prime product (about two thirds of that for a square), 12% more
    // for 3·2^k lengths, and about 2.6 cycles per wrapped limb product.
    let per = if square { 155 } else { 230 };
    let cost = |n: usize| {
        let c = per * n as u64 * u64::from(usize::BITS - 1 - n.leading_zeros());
        if n.is_multiple_of(3) {
            c * 112 / 100
        } else {
            c
        }
    };
    let mut best = (full, 0usize);
    let mut best_cost = cost(full);
    let floor = la.max(lb).max(len / 2 + 1);
    let mut cand = full;
    loop {
        // Next smaller length of the 2^k / 3·2^k family.
        cand = if cand % 3 == 0 {
            cand / 3 * 2
        } else if cand >= 4 {
            cand / 4 * 3
        } else {
            break;
        };
        // (Radix-3 lengths need a power-of-two part of at least 2.)
        if cand < floor || cand < 4 {
            break;
        }
        let s = len - cand;
        let c = cost(cand) + 26 * mp_wrap_products(s, la, lb) as u64;
        if c < best_cost {
            best = (cand, s);
            best_cost = c;
        }
    }
    Some(best)
}

/// The top `s` coefficients `c_{len−s} … c_{len−1}` of the limb convolution of
/// `a` and `b` (`len = a.len() + b.len() − 1`), exactly, as 192-bit values
/// `(low 128 bits, high 64 bits)`.
fn mp_wrapped_coeffs(a: &[Limb], b: &[Limb], s: usize) -> Vec<(u128, u64)> {
    let len = a.len() + b.len() - 1;
    (len - s..len)
        .map(|k| {
            // a's indices i with 0 ≤ k − i < b.len().
            let i0 = (k + 1).saturating_sub(b.len());
            let i1 = (k + 1).min(a.len());
            // Low and high product halves summed separately: each sum is a
            // plain add/adc pair and cannot overflow 128 bits.
            let (mut lo, mut hi) = (0u128, 0u128);
            for (&x, &y) in a[i0..i1].iter().zip(b[k + 1 - i1..=k - i0].iter().rev()) {
                let p = x as u128 * y as u128;
                lo += (p as u64) as u128;
                hi += p >> 64;
            }
            let (v, o) = lo.overflowing_add(hi << 64);
            (v, (hi >> 64) as u64 + u64::from(o))
        })
        .collect()
}

/// Three-prime NTT multiplication (see the section comment above): the
/// convolution of the limb vectors modulo each prime, recombined by Garner's
/// CRT and carry-propagated straight into the result limbs.
fn mul_ntt(a: &Nat, b: &Nat) -> Nat {
    let (la, lb) = (a.limbs.len(), b.limbs.len());
    if la == 0 || lb == 0 {
        return Nat::zero();
    }
    let out_len = la + lb;
    // `la + lb − 1` coefficients; the top `wrap` of them fold onto the bottom
    // of the cyclic convolution and are corrected during recombination.
    let square = core::ptr::eq(a, b) || a.limbs == b.limbs;
    let Some((n, wrap)) = mp_plan(la, lb, square) else {
        return a.mul_toom4(b);
    };
    let wrapped = mp_wrapped_coeffs(&a.limbs, &b.limbs, wrap);
    // Power-of-two sub-transform length (n itself, or n/3 under a radix-3 stage).
    let radix3 = n % 3 == 0;
    let m = if radix3 { n / 3 } else { n };

    // Per prime: forward transforms, pointwise product (in Montgomery form,
    // which contributes a factor 2^-64), unscaled inverse.
    let mut res: [Vec<u64>; 3] = Default::default();
    for (slot, pr) in res.iter_mut().zip(&MP_PRIMES) {
        let (p, pinv) = (pr.p, pr.pinv);
        let tw = mp_twiddles(m, pr);
        let pow3 = if radix3 { mp_powers(n, pr) } else { Vec::new() };
        let mut fa = mp_forward(&a.limbs, n, &tw, &pow3, pr);
        if square {
            for x in fa.iter_mut() {
                *x = mp_mul(*x, *x, p, pinv);
            }
        } else {
            let fb = mp_forward(&b.limbs, n, &tw, &pow3, pr);
            for (x, &y) in fa.iter_mut().zip(&fb) {
                *x = mp_mul(*x, y, p, pinv);
            }
        }
        mp_inverse(&mut fa, &tw, &pow3, pr);
        *slot = fa;
    }

    // Undoing both the transform's factor n and the pointwise 2^-64 takes a
    // Montgomery multiply by n^-1·2^128 (n | p − 1 — including the factor 3 of
    // a radix-3 length — so n^-1 = p − (p − 1)/n).
    let scale = |pr: &MpPrime| {
        let n_inv = pr.p - (pr.p - 1) / n as u64;
        ((n_inv as u128 * pr.r2 as u128) % pr.p as u128) as u64
    };
    let [q1, q2, q3] = &MP_PRIMES;
    let (p1, p2, p3) = (q1.p, q2.p, q3.p);
    let (s1, s2, s3) = (scale(q1), scale(q2), scale(q3));
    // Garner constants, in Montgomery form for their respective primes.
    let inv_p1_mod_p2 = mp_to_mont(mp_pow_slow(p1 % p2, p2 - 2, p2), q2);
    let p1_mod_p3 = mp_to_mont(p1 % p3, q3);
    let p12_mod_p3 = ((p1 as u128 * p2 as u128) % p3 as u128) as u64;
    let inv_p12_mod_p3 = mp_to_mont(mp_pow_slow(p12_mod_p3, p3 - 2, p3), q3);
    let p12 = p1 as u128 * p2 as u128;
    let (p12_lo, p12_hi) = (p12 as u64, (p12 >> 64) as u64);

    let mut out: Vec<Limb> = Vec::with_capacity(out_len);
    // Running 192-bit carry: each coefficient is < 2^186 and the carried-over
    // high part < 2^128, so three limbs never overflow.
    let (mut c0, mut c1) = (0u64, 0u64);
    let [res1, res2, res3] = &res;
    // `h mod p` for a wrapped 192-bit coefficient.
    let rem = |&(lo, hi): &(u128, u64), p: u64| -> u64 {
        let p = p as u128;
        let r = (((hi as u128 % p) << 64) | (lo >> 64)) % p;
        (((r << 64) | (lo as u64) as u128) % p) as u64
    };
    let residues = |k: usize| {
        (
            mp_csub(mp_mul(res1[k], s1, p1, q1.pinv), p1),
            mp_csub(mp_mul(res2[k], s2, p2, q2.pinv), p2),
            mp_csub(mp_mul(res3[k], s3, p3, q3.pinv), p3),
        )
    };
    // Residue k < wrap also holds the wrapped c_{n+k}; take it back out.
    let fixed: Vec<(u64, u64, u64)> = wrapped
        .iter()
        .enumerate()
        .map(|(k, h)| {
            let (r1, r2, r3) = residues(k);
            (
                mp_csub(r1 + p1 - rem(h, p1), p1),
                mp_csub(r2 + p2 - rem(h, p2), p2),
                mp_csub(r3 + p3 - rem(h, p3), p3),
            )
        })
        .collect();
    let plain = (wrapped.len()..n.min(out_len - 1)).map(residues);
    for (r1, r2, r3) in fixed.into_iter().chain(plain) {
        // x ≡ r1 (p1): x12 = r1 + p1·v2 with v2 = (r2 − r1)/p1 mod p2.
        // (r1 < p1 < 2·p2, so r2 + 2·p2 − r1 > 0 and < 4·p2.)
        let v2 = mp_csub(mp_mul(r2 + 2 * p2 - r1, inv_p1_mod_p2, p2, q2.pinv), p2);
        // x12 mod p3 = (r1 + p1·v2) mod p3, with r1 < p1 < 2·p3.
        let t = mp_csub(mp_mul(v2, p1_mod_p3, p3, q3.pinv), p3) + r1;
        let t = mp_csub(mp_csub(t, 2 * p3), p3);
        let v3 = mp_csub(mp_mul(r3 + p3 - t, inv_p12_mod_p3, p3, q3.pinv), p3);
        // x = x12 + p1·p2·v3 (< p1·p2·p3 < 2^186), added into the carry.
        let x12 = r1 as u128 + p1 as u128 * v2 as u128;
        let lo = p12_lo as u128 * v3 as u128;
        let hi = p12_hi as u128 * v3 as u128 + (lo >> 64);
        let s = c0 as u128 + (x12 as u64) as u128 + (lo as u64) as u128;
        let m = c1 as u128 + (x12 >> 64) + (hi as u64) as u128 + (s >> 64);
        out.push(s as u64);
        c0 = m as u64;
        c1 = ((hi >> 64) + (m >> 64)) as u64;
    }
    // The wrapped top coefficients, already exact.
    for &(lo, hi) in &wrapped {
        let s = c0 as u128 + (lo as u64) as u128;
        let m = c1 as u128 + (lo >> 64) + (s >> 64);
        out.push(s as u64);
        c0 = m as u64;
        c1 = hi + (m >> 64) as u64;
    }
    out.push(c0);
    debug_assert_eq!(c1, 0, "three-prime product overflowed");
    let mut r = Nat { limbs: out };
    r.normalize();
    r
}

/// Decides whether the three-prime NTT beats the Karatsuba/Toom ladder for a
/// product of the given operand sizes (in limbs). The transform length (see
/// [`mp_plan`]) steps through `2^k` and `3·2^k`, so the NTT's cost is a step
/// function of the input size: on short transforms it wins only while the
/// transform is well filled (and not by too large a wrap correction), and
/// from 4096 points on it wins outright. A small operand keeps the Toom
/// ladder's cost below the transform's `(la + lb)·log(la + lb)`, hence the
/// floor on the shorter side, raised for very long transforms. (Cutoffs
/// measured on x86-64 (Zen 5) with `measure_ntt_paths` /
/// `measure_ntt_unbalanced`; re-measure per platform to retune.)
fn ntt_worthwhile(la: usize, lb: usize, square: bool) -> bool {
    let min = la.min(lb);
    if min < NTT_MIN_LIMBS {
        return false;
    }
    let Some((n, _)) = mp_plan(la, lb, square) else {
        return false;
    };
    if min < 1000 && n > 1 << 16 {
        return false;
    }
    // Coefficients per transform point, in percent: above 100 when the plan
    // wraps (the excess is computed by the quadratic wrap correction).
    let fill = ((la + lb - 1) as u64 * 100 / n as u64) as u32;
    match n {
        ..=3072 => false,
        // A short side below ~700 limbs keeps the block-split Toom ahead
        // until the transform is long.
        _ if min < 700 && n < 8192 => false,
        // ... and while a wrap correction stays modest.
        4096 => fill >= if square { 85 } else { 93 } && fill <= 112,
        6144 => fill >= if square { 85 } else { 88 } && fill <= 112,
        _ => true,
    }
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
    // Use a block size that stays even through every recursive split so the
    // recursion never bails to quadratic Knuth early. Padding the divisor to `n2`
    // limbs by an extra left shift is value-preserving: shifting both operands
    // left by the same amount leaves the quotient unchanged and only scales the
    // remainder, which the final `shr` undoes.
    // Smallest block n2 ≥ n that halves evenly k times down to the base case
    // (n2 = ⌈n/2^k⌉·2^k with ⌈n/2^k⌉ < BZ_BASE): pads by < 2^k limbs rather
    // than up to 2× for a next-power-of-two block.
    let mut k = 0;
    while n.div_ceil(1 << k) >= BZ_BASE {
        k += 1;
    }
    let n2 = n.div_ceil(1 << k) << k;
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

// --- Newton-reciprocal division (Brent & Zimmermann, MCA §3.4–3.5) ---------
//
// For large operands, where multiplication is NTT-based and Burnikel–Ziegler
// costs `O(M(n)·log n)`, divide through an approximate reciprocal instead:
// `V ≈ β^(2n)/d` by Newton's iteration with doubling precision (`O(M(n))`),
// then each `n`-limb block of the quotient is one high product with `V` and
// one back-multiplication, corrected exactly against the true remainder.

/// Whether Newton-reciprocal division beats Burnikel–Ziegler for an `n`-limb
/// divisor and an `m`-limb quotient. BZ costs ~5 products per further `n`-limb
/// quotient block, Newton ~2 after a one-time ~3-product reciprocal, so Newton
/// wins on long quotients from moderate sizes, and on balanced ones only once
/// BZ's `log n` factor dominates. (Measured on x86-64; ratios to `M(n)`.)
fn newton_div_worthwhile(n: usize, m: usize) -> bool {
    (n >= 768 && m >= 4 * n) || (n >= 2048 && m >= 2 * n) || (n >= 16384 && m >= n)
}

/// Precision (limbs) at or below which the reciprocal is computed by one exact
/// division instead of a further Newton level.
const NEWTON_RECIP_BASE: usize = 192;

/// `⌊x / β^k⌋` (drops the low `k` limbs).
fn limbs_shr(x: &Nat, k: usize) -> Nat {
    if k >= x.limbs.len() {
        Nat::zero()
    } else {
        Nat::from_limbs(&x.limbs[k..])
    }
}

/// `x · β^k`.
fn limbs_shl(x: &Nat, k: usize) -> Nat {
    if x.is_zero() {
        return Nat::zero();
    }
    let mut v = Vec::with_capacity(x.limbs.len() + k);
    v.resize(k, 0);
    v.extend_from_slice(&x.limbs);
    Nat { limbs: v }
}

/// Approximate reciprocal of the normalized `n`-limb `d` (top bit set):
/// returns `V` with `|V − β^(2n)/d| ≤ 4`.
///
/// One Newton step from the reciprocal `Vh` of the top `h = ⌈n/2⌉ + 1` limbs
/// `dh`: with `e = β^(n+h) − d·Vh` (`|e| ≲ 6·β^n`),
/// `V = Vh·β^(n−h) + ⌊Vh·e / β^(2h)⌋`, whose relative error is the square of
/// `Vh`'s (≲ (6β^−h)²), so the absolute error is `O(β^(n−2h)) < 1` plus
/// rounding.
fn newton_recip(d: &Nat) -> Nat {
    let n = d.limbs.len();
    if n <= NEWTON_RECIP_BASE {
        return limbs_shl(&Nat::one(), 2 * n).div_rem(d).expect("d > 0").0;
    }
    let h = n / 2 + 2;
    let dh = limbs_shr(d, n - h);
    let vh = newton_recip(&dh);
    let p = d.mul(&vh);
    let top = limbs_shl(&Nat::one(), n + h);
    let base = limbs_shl(&vh, n - h);
    // Truncate e to its top limbs: dropping h − 1 low limbs perturbs
    // Vh·e/β^(2h) by < Vh·β^(h−1)/β^(2h) < 1.
    if p.cmp_ref(&top) != Ordering::Greater {
        let e = top.checked_sub(&p).expect("p <= top");
        let corr = limbs_shr(&vh.mul(&limbs_shr(&e, h - 1)), h + 1);
        base.add(&corr)
    } else {
        let e = p.checked_sub(&top).expect("p > top");
        let corr = limbs_shr(&vh.mul(&limbs_shr(&e, h - 1)), h + 1);
        base.checked_sub(&corr.add_u64(1))
            .expect("correction is far below Vh·β^(n−h)")
    }
}

/// Turns a quotient estimate `q0` (within a few units of `⌊a/d⌋`) into the
/// exact `(q, r)` by stepping against the true remainder; `None` if the
/// estimate was off by more than a small bound (the caller then falls back).
fn correct_quotient(a: &Nat, d: &Nat, q0: Nat) -> Option<(Nat, Nat)> {
    const MAX_STEPS: usize = 16;
    let qd = q0.mul(d);
    let mut q = q0;
    let mut r = if qd.cmp_ref(a) == Ordering::Greater {
        // Too large: step down until q·d ≤ a, tracking the deficit q·d − a.
        let mut deficit = qd;
        deficit.sub_assign_limbs(&a.limbs);
        let mut steps = 0;
        loop {
            steps += 1;
            if steps > MAX_STEPS || q.is_zero() {
                return None;
            }
            q.sub_assign_u64_unchecked(1);
            if deficit.cmp_ref(d) != Ordering::Greater {
                let mut r = d.clone();
                r.sub_assign_limbs(&deficit.limbs);
                break r;
            }
            deficit.sub_assign_limbs(&d.limbs);
        }
    } else {
        let mut r = a.clone();
        r.sub_assign_limbs(&qd.limbs);
        r
    };
    let mut steps = 0;
    while r.cmp_ref(d) != Ordering::Less {
        steps += 1;
        if steps > MAX_STEPS {
            return None;
        }
        r.sub_assign_limbs(&d.limbs);
        q.add_assign_u64(1);
    }
    Some((q, r))
}

/// Divides `a < d·β^(n−2)` by the normalized `n`-limb `d` using its
/// approximate reciprocal `v ≈ β^(2n)/d` (from [`newton_recip`]).
///
/// `q₀ = ⌊⌊a/β^(n−2)⌋·⌊v/β⌋ / β^(n+1)⌋`: dropping `n − 2` limbs of `a` costs
/// `< 2β^(−2)`, dropping one limb of `v` costs `< β^(−1)`, so `q₀` is within a
/// few units of `⌊a/d⌋`. Both factors have at most `n` limbs, keeping the
/// product at a clean `n × n` size (the quotient block is `n − 2` limbs).
fn newton_div_block(a: &Nat, d: &Nat, v: &Nat, n: usize) -> (Nat, Nat) {
    let q0 = limbs_shr(&limbs_shr(a, n - 2).mul(&limbs_shr(v, 1)), n + 1);
    correct_quotient(a, d, q0).unwrap_or_else(|| div_rem_classic(a, d))
}

/// `a divmod d` (`d` ≥ 2 limbs) by Burnikel–Ziegler / Knuth only — never the
/// Newton path, so the Newton code can fall back on it without recursion.
fn div_rem_classic(a: &Nat, d: &Nat) -> (Nat, Nat) {
    match a.cmp_ref(d) {
        Ordering::Less => (Nat::zero(), a.clone()),
        Ordering::Equal => (Nat::one(), Nat::zero()),
        Ordering::Greater => {
            let q = a.limbs.len() - d.limbs.len();
            if d.limbs.len() >= BZ_THRESHOLD && q >= BZ_THRESHOLD {
                bz_div_rem(a, d)
            } else {
                a.div_rem_knuth(d)
            }
        }
    }
}

/// Division with a short quotient: `a < d·β^f` for the normalized `n`-limb
/// `d`, with `f + 2 < n`. Only the top `f + 2` limbs of `d` (and the matching
/// limbs of `a`) determine the quotient up to a small error, so divide those
/// (a balanced `≈ 2f`-by-`f` problem), then correct against the full remainder
/// with one `f`-by-`n` product.
fn div_short_quotient(a: &Nat, d: &Nat, f: usize) -> (Nat, Nat) {
    let n = d.limbs.len();
    debug_assert!(f + 2 < n);
    let k = n - (f + 2);
    let at = limbs_shr(a, k);
    let dt = limbs_shr(d, k);
    let q0 = if at.cmp_ref(&dt) == Ordering::Less {
        Nat::zero()
    } else {
        at.div_rem(&dt).expect("dt > 0").0
    };
    correct_quotient(a, d, q0).unwrap_or_else(|| div_rem_classic(a, d))
}

/// A divisor prepared for repeated Newton-reciprocal division: normalized
/// (top bit set) with its approximate reciprocal `v ≈ β^(2n)/d`.
pub(crate) struct PreInv {
    d: Nat,
    shift: u64,
    v: Nat,
}

impl PreInv {
    /// Prepares `b` (≥ 3 limbs).
    pub(crate) fn new(b: &Nat) -> PreInv {
        let n = b.limbs.len();
        debug_assert!(n >= 3);
        let shift = b.limbs[n - 1].leading_zeros() as u64;
        let d = b.shl(shift);
        let v = newton_recip(&d);
        PreInv { d, shift, v }
    }

    /// `a divmod b` for any `a`.
    pub(crate) fn div_rem(&self, a: &Nat) -> (Nat, Nat) {
        let d = &self.d;
        let n = d.limbs.len();
        if a.limbs.len() < n {
            return (Nat::zero(), a.clone());
        }
        let an = a.shl(self.shift);
        // Quotient limbs m (q < β^m); blocks of w = n − 2 quotient limbs from
        // the top, the first one taking the remainder m mod w.
        let m = an.limbs.len() + 1 - n;
        let w = n - 2;
        let first = match m % w {
            0 => w,
            f => f,
        };
        let mut q_limbs: Vec<Limb> = alloc::vec![0; m];
        let mut r = Nat::zero();
        let mut pos = m; // quotient limbs still to produce, below `pos`
        let mut width = first;
        while pos > 0 {
            let lo = pos - width;
            // Window W = R·β^width + an[lo..pos) < d·β^width, where R < d is
            // the running remainder; initially R is an's top n − 1 limbs.
            let cur = if pos == m {
                Nat::from_limbs(&an.limbs[lo..])
            } else {
                let mut cur = Vec::with_capacity(width + r.limbs.len());
                cur.extend_from_slice(&an.limbs[lo..pos]);
                cur.extend_from_slice(&r.limbs);
                Nat::from_limb_vec(cur)
            };
            let (qi, ri) = if width == w {
                newton_div_block(&cur, d, &self.v, n)
            } else {
                div_short_quotient(&cur, d, width)
            };
            q_limbs[lo..lo + qi.limbs.len()].copy_from_slice(&qi.limbs);
            r = ri;
            pos = lo;
            width = w;
        }
        (Nat::from_limb_vec(q_limbs), r.shr(self.shift))
    }
}

/// Newton-reciprocal division. Precondition: `a > b`, `b` has ≥ 3 limbs.
fn newton_div_rem(a: &Nat, b: &Nat) -> (Nat, Nat) {
    let n = b.limbs.len();
    // A quotient shorter than the block width needs no reciprocal.
    let shift = b.limbs[n - 1].leading_zeros() as u64;
    let an = a.shl(shift);
    let m = an.limbs.len() + 1 - n;
    if m + 2 < n {
        let (q, r) = div_short_quotient(&an, &b.shl(shift), m);
        return (q, r.shr(shift));
    }
    PreInv::new(b).div_rem(a)
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
        .mul_2k(hbits)
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

/// Largest root size (limbs) handled by the allocation-free slice square
/// root [`sqrtrem_slice`]; above it the `Nat`-based recursion (whose divisions
/// and squarings use the sub-quadratic kernels) takes over.
const SQRT_SLICE_LIMBS: usize = 96;

/// Allocation-free Karatsuba square root (Zimmermann, "Karatsuba Square
/// Root", INRIA RR-3805) on limb slices.
///
/// Preconditions: `a.len() == 2n`, `s.len() == r.len() == n`, `n ≥ 1`, and the
/// top limb of `a` is `≥ 2^62`. Writes `s = ⌊√a⌋` and the low `n` limbs of
/// `a − s²` into `r`, returning the remainder's top limb (`0` or `1`, since
/// the remainder is `≤ 2s`). `scratch` needs `3n + 8` limbs; the recursive call
/// on the top half reuses it before this level touches it.
fn sqrtrem_slice(s: &mut [Limb], r: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) -> Limb {
    let n = s.len();
    debug_assert!(n >= 1 && a.len() == 2 * n && r.len() == n);
    debug_assert!(a[2 * n - 1] >> 62 != 0);
    if n == 1 {
        let v = ((a[1] as u128) << 64) | a[0] as u128;
        let x = isqrt_u128(v);
        let rem = v - x * x; // ≤ 2x < 2^65
        s[0] = x as Limb;
        r[0] = rem as Limb;
        return (rem >> 64) as Limb;
    }
    let l = n / 2;
    let h = n - l;
    // Top half: s₁ = ⌊√(a div β^(2l))⌋ into s[l..], R₁ into r[l..] (+ c1·β^h).
    let c1 = {
        let (_, s_hi) = s.split_at_mut(l);
        let (_, r_hi) = r.split_at_mut(l);
        sqrtrem_slice(s_hi, r_hi, &a[2 * l..], scratch)
    };
    let (nb, rest) = scratch.split_at_mut(n + 1);
    let (qq, rest) = rest.split_at_mut(l + 1);
    let (sq, _) = rest.split_at_mut(2 * l + 2);
    // N = R₁·β^l + a₁ (n + 1 limbs), divided by s₁ (normalized: top bit set).
    nb[..l].copy_from_slice(&a[l..2 * l]);
    nb[l..n].copy_from_slice(&r[l..]);
    nb[n] = c1;
    {
        let s1 = &s[l..];
        if h == 1 {
            let d = s1[0];
            let v = recip_2by1(d);
            let rem = divrem_1_normalized_in_place(nb, d, 0, v);
            qq.copy_from_slice(&nb[..l + 1]);
            nb[0] = rem;
        } else {
            // The top h limbs of N are R₁ div β < s₁, as the kernel requires.
            div_schoolbook_normalized(nb, s1, qq);
        }
    }
    // (q, u) = N divmod 2s₁: halve the quotient, and fold its low bit back
    // into the remainder as one more s₁.
    let odd = qq[0] & 1;
    for i in 0..l {
        qq[i] = (qq[i] >> 1) | (qq[i + 1] << 63);
    }
    qq[l] >>= 1;
    // u (h limbs + carry `cu`) lives in nb[..h].
    let mut cu: Limb = 0;
    if odd == 1 {
        cu += add_n_in_place(&mut nb[..h], &s[l..]);
    }
    // q ≤ β^l; cap q = β^l to β^l − 1, which moves 2s₁ into u.
    if qq[l] != 0 {
        qq[l] = 0;
        for x in qq[..l].iter_mut() {
            *x = Limb::MAX;
        }
        cu += add_n_in_place(&mut nb[..h], &s[l..]);
        cu += add_n_in_place(&mut nb[..h], &s[l..]);
    }
    // s = s₁·β^l + q.
    s[..l].copy_from_slice(&qq[..l]);
    // R = u·β^l + a₀ − q², tracked as r (n limbs) plus a signed top `carry`.
    sq[..2 * l].fill(0);
    sqr_into(&qq[..l], &mut sq[..2 * l]);
    r[..l].copy_from_slice(&a[..l]);
    r[l..].copy_from_slice(&nb[..h]);
    let mut carry = cu as i64;
    let mut borrow = 0;
    for (ri, &qi) in r.iter_mut().zip(&sq[..2 * l]) {
        let (d, b) = sbb(*ri, qi, borrow);
        *ri = d;
        borrow = b;
    }
    for ri in r[2 * l..].iter_mut() {
        let (d, b) = sbb(*ri, 0, borrow);
        *ri = d;
        borrow = b;
    }
    carry -= borrow as i64;
    // s one too large: R += 2s − 1 (old s), then s −= 1.
    while carry < 0 {
        let c = add_n_in_place(r, s) + add_n_in_place(r, s);
        carry += c as i64;
        // R −= 1
        let mut b = 1;
        for ri in r.iter_mut() {
            let (d, bb) = sbb(*ri, 0, b);
            *ri = d;
            b = bb;
            if b == 0 {
                break;
            }
        }
        carry -= b as i64;
        // s −= 1 (s ≥ 1 here)
        for si in s.iter_mut() {
            let (d, bb) = sbb(*si, 0, 1);
            *si = d;
            if bb == 0 {
                break;
            }
        }
    }
    // s one too small (only after capping q): while R > 2s, R −= 2s + 1, s += 1.
    loop {
        // Compare R (carry, r) with 2s (top bit of s, s << 1).
        let top2 = (s[n - 1] >> 63) as i64;
        let ord = match carry.cmp(&top2) {
            Ordering::Equal => {
                let mut o = Ordering::Equal;
                for i in (0..n).rev() {
                    let twice = (s[i] << 1) | if i > 0 { s[i - 1] >> 63 } else { 0 };
                    if r[i] != twice {
                        o = r[i].cmp(&twice);
                        break;
                    }
                }
                o
            }
            o => o,
        };
        if ord != Ordering::Greater {
            break;
        }
        // R −= 2s + 1, then s += 1.
        let b = submul_1(r, s, 2);
        carry -= b as i64;
        let mut bb = 1;
        for ri in r.iter_mut() {
            let (d, b2) = sbb(*ri, 0, bb);
            *ri = d;
            bb = b2;
            if bb == 0 {
                break;
            }
        }
        carry -= bb as i64;
        for si in s.iter_mut() {
            let (d, c) = adc(*si, 0, 1);
            *si = d;
            if c == 0 {
                break;
            }
        }
    }
    debug_assert!((0..=1).contains(&carry));
    carry as Limb
}

/// `(⌊√a⌋, a − ⌊√a⌋²)` via [`sqrtrem_slice`] for a normalized `a` (top limb
/// `≥ 2^62`) of at most `2·SQRT_SLICE_LIMBS` limbs. An odd limb count is
/// handled as `a·β` (root `s'`, `s = s' >> 32`, remainder recovered exactly).
fn sqrt_rem_small(a: &Nat) -> (Nat, Nat) {
    let len = a.limbs.len();
    let odd = len % 2 == 1;
    let mut buf: Vec<Limb> = Vec::with_capacity(len + 1);
    if odd {
        buf.push(0);
    }
    buf.extend_from_slice(&a.limbs);
    let n = buf.len() / 2;
    let mut s = alloc::vec![0 as Limb; n];
    let mut r = alloc::vec![0 as Limb; n + 1];
    let mut scratch = alloc::vec![0 as Limb; 3 * n + 8];
    let c = sqrtrem_slice(&mut s, &mut r[..n], &buf, &mut scratch);
    r[n] = c;
    let s = Nat::from_limb_vec(s);
    let r = Nat::from_limb_vec(r);
    if !odd {
        return (s, r);
    }
    // a·β = s'² + r', s' = s·2^32 + t  ⇒  a − s² = (2·s·t·2^32 + t² + r') / β.
    let t = s.limbs[0] & 0xffff_ffff;
    let root = s.shr(32);
    let num = root.mul_u64(t).shl(33).add_u64(t * t).add(&r);
    (root, num.shr(64))
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
    if n > 2 && n <= 2 * SQRT_SLICE_LIMBS {
        return sqrt_rem_small(a);
    }
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
    // `lo ‖ hi` = lo + hi·B^l for an l-limb `lo`, built in one allocation.
    let concat = |lo: &[Limb], hi: &Nat| -> Nat {
        let mut v = Vec::with_capacity(l + hi.limbs.len() + 1);
        v.extend_from_slice(lo);
        v.extend_from_slice(&hi.limbs);
        Nat::from_limb_vec(v)
    };
    let (s1, r1) = sqrt_rem(&Nat::from_limbs(&a.limbs[2 * l..]));
    // (q, u) = (r₁·B^l + a₁) divmod 2·s₁; root estimate s = s₁·B^l + q.
    let mut s1x2 = s1.clone();
    s1x2.add_assign_limbs(&s1.limbs);
    let (q, u) = concat(&a.limbs[l..2 * l], &r1)
        .div_rem(&s1x2)
        .expect("s1 > 0");
    let mut s = Nat {
        limbs: Vec::with_capacity(l + s1.limbs.len() + 1),
    };
    s.limbs.resize(l, 0);
    s.limbs.extend_from_slice(&s1.limbs);
    s.add_assign_limbs(&q.limbs);
    // r = u·B^l + a₀ − q², fixed up so 0 ≤ r ≤ 2s. Both loops run O(1) times
    // and cost only additions of half-size values — never full-width squarings.
    let t = concat(&a.limbs[..l], &u);
    let q2 = q.square();
    let one = Nat::one();
    let diff = if t.cmp_ref(&q2) == Ordering::Less {
        Err((t, q2))
    } else {
        let mut r = t;
        r.sub_assign_limbs(&q2.limbs);
        Ok(r)
    };
    match diff {
        Ok(mut r) => {
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
        Err((t, q2)) => {
            // Estimate too large (r < 0): while r < 0, step s down (r += 2s−1
            // with the old s), tracked as the positive deficit q² − t.
            let mut deficit = q2;
            deficit.sub_assign_limbs(&t.limbs);
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
    if v >> 64 == 0 {
        return isqrt_u64(v as u64) as u128;
    }
    // Seed from the top 62–64 bits (an even shift 2k): with t = ⌊√(v >> 2k)⌋ ≥ 2^31,
    // x₀ = (t + 1)·2^k exceeds √v by less than 2^k, so one Newton step leaves
    // an error below 2^(2k)/2^(k+32) ≤ 1 (k ≤ 32); then fix up exactly.
    let bits = 128 - v.leading_zeros();
    let k = (bits - 64).div_ceil(2);
    let top = isqrt_u64((v >> (2 * k)) as u64) as u128;
    let x0 = (top + 1) << k; // > √v, and ≤ 2^64
    let mut x = (x0 + v / x0) / 2; // ≥ ⌊√v⌋ (AM-GM), and close to it
    // x < 2^64 + small; x·x may need checking without overflow.
    while x.checked_mul(x).is_none_or(|sq| sq > v) {
        x -= 1;
    }
    while (x + 1).checked_mul(x + 1).is_some_and(|sq| sq <= v) {
        x += 1;
    }
    x
}

/// `⌊√v⌋` for a `u64`: the hardware `f64` square root (with `std`) or an
/// integer Newton iteration (bare `no_std`, where `core` has no `sqrt`), plus
/// an exact fix-up.
fn isqrt_u64(v: u64) -> u64 {
    #[cfg(feature = "std")]
    let mut x = (v as f64).sqrt() as u64; // within ±1 of the true root
    #[cfg(not(feature = "std"))]
    let mut x = {
        if v == 0 {
            return 0;
        }
        // Newton from above (2^⌈bits/2⌉ ≥ √v) descends to ⌊√v⌋.
        let bits = 64 - v.leading_zeros();
        let mut x = 1u64 << bits.div_ceil(2);
        loop {
            let y = (x + v / x) / 2;
            if y >= x {
                break x;
            }
            x = y;
        }
    };
    while x.checked_mul(x).is_none_or(|sq| sq > v) {
        x -= 1;
    }
    while (x + 1).checked_mul(x + 1).is_some_and(|sq| sq <= v) {
        x += 1;
    }
    x
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

/// `x`'s limbs zero-padded to exactly `n` (`x` must fit).
fn padded(x: &Nat, n: usize) -> Vec<Limb> {
    debug_assert!(
        x.limbs.len() <= n,
        "Montgomery operand wider than the modulus"
    );
    let mut v = Vec::with_capacity(n);
    v.extend_from_slice(&x.limbs[..x.limbs.len().min(n)]);
    v.resize(n, 0);
    v
}

/// Montgomery multiplication: returns `a·b·R⁻¹ mod m` in `[0, m)`, where
/// `R = 2^(64·m.len())`. Requires an odd modulus with a non-zero top limb and
/// `a, b < m`.
fn mont_mul(a: &Nat, b: &Nat, m: &[Limb]) -> Nat {
    let mut out = padded(b, m.len());
    let bp = out.clone();
    crate::mont::Mont::new(m).mul(&mut out, &padded(a, m.len()), &bp);
    let mut r = Nat { limbs: out };
    r.normalize();
    r
}

/// Montgomery squaring: returns `a²·R⁻¹ mod m` in `[0, m)` (same contract as
/// [`mont_mul`] with both operands equal).
fn mont_sqr(a: &Nat, m: &[Limb]) -> Nat {
    let mut out = padded(a, m.len());
    let ap = out.clone();
    crate::mont::Mont::new(m).sqr(&mut out, &ap);
    let mut r = Nat { limbs: out };
    r.normalize();
    r
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
        if i + 2 < n {
            addmul2_row(&mut t[2 * i + 2..i + n + 2], &a[i + 2..], a0, a1);
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

/// Montgomery arithmetic context for an odd modulus `m > 1` (any limb count).
///
/// Encapsulates the one-time setup that [`Nat::modpow_montgomery`] performs
/// inline — `R² mod m`, with `R = 2^(64·k)` — so hot
/// loops (Pollard rho, ECM) can convert their operands into Montgomery form once
/// and run every multiply and squaring through [`mont_mul`]/[`mont_sqr`] without
/// a per-step division-based reduction.
///
/// Addition and subtraction need no conversion: the Montgomery map `x ↦ x·R` is
/// linear, so a mod-`m` add/sub of two Montgomery representatives is the
/// Montgomery representative of the sum/difference. Likewise `gcd(a, m)` is
/// invariant under the `R` factor (because `gcd(R, m) = 1` for odd `m`), so a
/// factor can be extracted from a Montgomery-form value without converting out.
pub(crate) struct MontCtx {
    /// Modulus limbs (exactly `k`, top non-zero, odd).
    m: Vec<Limb>,
    /// Modulus as a `Nat`, for reductions and comparisons.
    modulus: Nat,
    /// `R² mod m`; converts a residue into Montgomery form via one `mont_mul`.
    r2: Nat,
    /// `R³ mod m`; turns the integer inverse of a Montgomery representative back
    /// into Montgomery form with a single `mont_mul` (see [`MontCtx::inv`]).
    r3: Nat,
    /// `R mod m` — the Montgomery form of `1`.
    one: Nat,
}

impl MontCtx {
    /// Builds the context for an odd `modulus > 1`.
    pub(crate) fn new(modulus: &Nat) -> MontCtx {
        debug_assert!(!modulus.is_even(), "MontCtx requires an odd modulus");
        let k = modulus.limbs.len();
        let m = modulus.limbs.clone();
        let r = Nat::one().shl(k as u64 * LIMB_BITS as u64);
        let r2 = r.mul(&r).div_rem(modulus).expect("non-zero").1;
        let r3 = r2.mul(&r).div_rem(modulus).expect("non-zero").1;
        let one = mont_mul(&Nat::one(), &r2, &m);
        MontCtx {
            m,
            modulus: modulus.clone(),
            r2,
            r3,
            one,
        }
    }

    /// The modulus `m`.
    #[inline]
    pub(crate) fn modulus(&self) -> &Nat {
        &self.modulus
    }

    /// The Montgomery form of `1` (`R mod m`).
    #[inline]
    pub(crate) fn one(&self) -> &Nat {
        &self.one
    }

    /// Converts a residue into Montgomery form: `a·R mod m`. Reduces `a` first if
    /// it is not already `< m`.
    pub(crate) fn to_mont(&self, a: &Nat) -> Nat {
        let reduced;
        let a = if a.cmp_ref(&self.modulus) != Ordering::Less {
            reduced = a.div_rem(&self.modulus).expect("non-zero").1;
            &reduced
        } else {
            a
        };
        mont_mul(a, &self.r2, &self.m)
    }

    /// Converts out of Montgomery form: `a·R⁻¹ mod m` (the true residue).
    pub(crate) fn to_residue(&self, a: &Nat) -> Nat {
        mont_mul(a, &Nat::one(), &self.m)
    }

    /// `a·b·R⁻¹ mod m`: the Montgomery product of two Montgomery-form operands
    /// (itself in Montgomery form). Requires `a, b < m`.
    #[inline]
    pub(crate) fn mul(&self, a: &Nat, b: &Nat) -> Nat {
        mont_mul(a, b, &self.m)
    }

    /// `a²·R⁻¹ mod m`: the Montgomery squaring of a Montgomery-form operand.
    #[inline]
    pub(crate) fn sqr(&self, a: &Nat) -> Nat {
        mont_sqr(a, &self.m)
    }

    /// `(a + b) mod m` for reduced `a, b` (identical whether the operands are
    /// true residues or Montgomery representatives — the map is linear).
    pub(crate) fn add(&self, a: &Nat, b: &Nat) -> Nat {
        let s = a.add(b);
        if s.cmp_ref(&self.modulus) != Ordering::Less {
            s.checked_sub(&self.modulus).expect("s >= m")
        } else {
            s
        }
    }

    /// `(a − b) mod m` for reduced `a, b`, returned as the canonical
    /// representative in `[0, m)`.
    pub(crate) fn sub(&self, a: &Nat, b: &Nat) -> Nat {
        if a.cmp_ref(b) != Ordering::Less {
            a.checked_sub(b).expect("a >= b")
        } else {
            self.modulus
                .checked_sub(&b.checked_sub(a).expect("b > a"))
                .expect("difference < m")
        }
    }

    /// `base^exp` (Montgomery form → Montgomery form): the Montgomery
    /// representative raised to `exp`, via a windowed square-and-multiply that
    /// stays entirely in Montgomery form. `base` must be a reduced Montgomery
    /// representative (`< m`).
    pub(crate) fn pow(&self, base: &Nat, exp: &Nat) -> Nat {
        modpow_windowed(base.clone(), self.one.clone(), exp, |a, b| {
            if core::ptr::eq(a, b) {
                mont_sqr(a, &self.m)
            } else {
                mont_mul(a, b, &self.m)
            }
        })
    }

    /// Montgomery-domain inverse: given `u = a·R mod m`, returns `a⁻¹·R mod m`
    /// (the Montgomery form of the modular inverse), or `None` when
    /// `gcd(a, m) ≠ 1`.
    ///
    /// The integer inverse of the representative is `(a·R)⁻¹ = a⁻¹·R⁻¹`; one
    /// `mont_mul` by `R³` maps it back into Montgomery form
    /// (`a⁻¹R⁻¹ · R³ · R⁻¹ = a⁻¹R`), avoiding a separate convert-out/convert-in.
    pub(crate) fn inv(&self, u: &Nat) -> Option<Nat> {
        let inv = Int::from(u.clone()).modinv(&Int::from(self.modulus.clone()))?;
        Some(mont_mul(&inv.magnitude(), &self.r3, &self.m))
    }
}

/// Adds the limbs of `val` into `out` starting at limb `offset`, propagating the
/// carry. `out` must be large enough to hold the result (including any carry).
#[inline]
fn add_at(out: &mut [Limb], offset: usize, val: &[Limb]) {
    add_in_place(&mut out[offset..], val);
}

/// `x + y + carry` on limbs with a boolean carry in and out. Chained, these
/// paired overflow flags lower to a single `adc` chain.
#[inline(always)]
fn adc_b(x: Limb, y: Limb, carry: bool) -> (Limb, bool) {
    let (r, c1) = x.overflowing_add(y);
    let (r, c2) = r.overflowing_add(Limb::from(carry));
    (r, c1 | c2)
}

/// `x − y − borrow` on limbs with a boolean borrow in and out (lowers to
/// an `sbb` chain).
#[inline(always)]
fn sbb_b(x: Limb, y: Limb, borrow: bool) -> (Limb, bool) {
    let (r, b1) = x.overflowing_sub(y);
    let (r, b2) = r.overflowing_sub(Limb::from(borrow));
    (r, b1 | b2)
}

/// `out[i] = f(x[i], y[i], flag)` over equal-length slices with a carried
/// flag, four limbs per iteration so the flag stays in the CPU's carry bit
/// inside each group (it is only materialized once per four limbs).
#[inline(always)]
fn carry_chain3(
    out: &mut [Limb],
    x: &[Limb],
    y: &[Limb],
    f: impl Fn(Limb, Limb, bool) -> (Limb, bool),
) -> bool {
    debug_assert!(out.len() == x.len() && x.len() == y.len());
    let mut c = false;
    let (o4, ot) = out.as_chunks_mut::<4>();
    let (x4, xt) = x.as_chunks::<4>();
    let (y4, yt) = y.as_chunks::<4>();
    for ((o, a), b) in o4.iter_mut().zip(x4).zip(y4) {
        let (r0, c0) = f(a[0], b[0], c);
        let (r1, c1) = f(a[1], b[1], c0);
        let (r2, c2) = f(a[2], b[2], c1);
        let (r3, c3) = f(a[3], b[3], c2);
        *o = [r0, r1, r2, r3];
        c = c3;
    }
    for ((o, &a), &b) in ot.iter_mut().zip(xt).zip(yt) {
        let (r, c1) = f(a, b, c);
        *o = r;
        c = c1;
    }
    c
}

/// In-place form of [`carry_chain3`]: `dst[i] = f(dst[i], src[i], flag)`
/// over `src.len()` limbs.
#[inline(always)]
fn carry_chain2(
    dst: &mut [Limb],
    src: &[Limb],
    f: impl Fn(Limb, Limb, bool) -> (Limb, bool),
) -> bool {
    let dst = &mut dst[..src.len()];
    let mut c = false;
    let (d4, dt) = dst.as_chunks_mut::<4>();
    let (s4, st) = src.as_chunks::<4>();
    for (d, s) in d4.iter_mut().zip(s4) {
        let (r0, c0) = f(d[0], s[0], c);
        let (r1, c1) = f(d[1], s[1], c0);
        let (r2, c2) = f(d[2], s[2], c1);
        let (r3, c3) = f(d[3], s[3], c2);
        *d = [r0, r1, r2, r3];
        c = c3;
    }
    for (d, &s) in dt.iter_mut().zip(st) {
        let (r, c1) = f(*d, s, c);
        *d = r;
        c = c1;
    }
    c
}

/// `dst[..src.len()] += src`, returning the carry out (`dst.len() ≥
/// src.len()`).
#[inline]
fn add_n(dst: &mut [Limb], src: &[Limb]) -> bool {
    carry_chain2(dst, src, adc_b)
}

/// `dst[..src.len()] −= src`, returning the borrow out (`dst.len() ≥
/// src.len()`).
#[inline]
fn sub_n(dst: &mut [Limb], src: &[Limb]) -> bool {
    carry_chain2(dst, src, sbb_b)
}

/// `out = x − y` for equal-length slices, returning the borrow out.
#[inline]
fn sub_n3(out: &mut [Limb], x: &[Limb], y: &[Limb]) -> bool {
    carry_chain3(out, x, y, sbb_b)
}

/// `out = x + y` for equal-length slices, returning the carry out.
#[inline]
fn add_n3(out: &mut [Limb], x: &[Limb], y: &[Limb]) -> bool {
    carry_chain3(out, x, y, adc_b)
}
/// Propagates a carry of 1 into `dst`, returning whether it ran off the end.
#[inline]
fn incr(dst: &mut [Limb]) -> bool {
    for d in dst.iter_mut() {
        let (r, c) = d.overflowing_add(1);
        *d = r;
        if !c {
            return false;
        }
    }
    true
}

/// Propagates a borrow of 1 out of `dst`, returning whether it ran off the end.
#[inline]
fn decr(dst: &mut [Limb]) -> bool {
    for d in dst.iter_mut() {
        let (r, b) = d.overflowing_sub(1);
        *d = r;
        if !b {
            return false;
        }
    }
    true
}

/// `dst += src` (`dst.len() ≥ src.len()`); the sum must fit in `dst`.
#[inline]
fn add_in_place(dst: &mut [Limb], src: &[Limb]) {
    let carry = add_n(dst, src) && incr(&mut dst[src.len()..]);
    debug_assert!(!carry, "add_in_place carry escaped");
}

/// Subtracts `src` from `dst` in place (`dst -= src`); requires `dst >= src`
/// as numbers and `dst.len() >= src.len()`.
#[inline]
fn sub_in_place(dst: &mut [Limb], src: &[Limb]) {
    let borrow = sub_n(dst, src) && decr(&mut dst[src.len()..]);
    debug_assert!(!borrow, "sub_in_place borrow escaped: dst < src");
}

// ---------------------------------------------------------------------------
// Subquadratic Half-GCD (Schönhage; Möller, *Math. Comp.* 77 (2008);
// Lichtblau, "Half-GCD, Fast Rational Recovery, and Planar Lattice Reduction",
// ISSAC 2005). See Brent & Zimmermann, *Modern Computer Arithmetic* §1.6.
//
// `hgcd(m, n)` (m > n ≥ 0) returns a 2×2 cofactor matrix `R` — a product of
// elementary Euclidean quotient matrices `E(q) = [[0,1],[1,-q]]`, so `det R =
// ±1` — together with the reduced pair `(va, vb) = R·(m, n)`, a *consecutive
// pair in the Euclidean remainder sequence of `(m, n)`* that straddles the
// half-length point: `bit_len(va) > h ≥ bit_len(vb)` with `h = ⌈bit_len(m)/2⌉`.
//
// Correctness rests on Lichtblau's Lemma 3 (a.k.a. Fact 1 in Möller): *any*
// product of elementary matrices applied to `(m, n)` that yields an ordered
// positive pair `u > v > 0` is a genuine consecutive remainder pair of
// `(m, n)`. A recursion working on the top halves of the limbs may overshoot
// the true quotient sequence by a bounded amount; `hgcd_make_valid` backs up
// with reverse Euclidean steps (recovering the last quotient from the matrix
// entries) until the pair is ordered again — Möller's left-to-right stop
// condition guarantees this needs only a bounded number of steps, so the
// asymptotic cost stays `O(M(n)·log n)`. Because every `R` is unimodular, the
// pair-gcd is preserved unconditionally: `gcd(va, vb) = gcd(m, n)` even in the
// (tested-against) event of an off-by-one reduction, which is the safety net
// that makes the driver's output bit-identical to plain Euclid/Lehmer.
// ---------------------------------------------------------------------------

/// A 2×2 integer matrix `[[a, b], [c, d]]` accumulating Euclidean cofactors,
/// maintained as a product of elementary matrices (determinant `±1`).
#[derive(Clone)]
struct HgcdMat {
    a: Int,
    b: Int,
    c: Int,
    d: Int,
}

impl HgcdMat {
    fn identity() -> Self {
        HgcdMat {
            a: Int::from(1i64),
            b: Int::from(0i64),
            c: Int::from(0i64),
            d: Int::from(1i64),
        }
    }

    fn is_identity(&self) -> bool {
        self.b.is_zero() && self.c.is_zero() && self.a.is_one() && self.d.is_one()
    }

    /// Matrix product `self · other`.
    fn mul(&self, o: &HgcdMat) -> HgcdMat {
        HgcdMat {
            a: self.a.mul(&o.a).add(&self.b.mul(&o.c)),
            b: self.a.mul(&o.b).add(&self.b.mul(&o.d)),
            c: self.c.mul(&o.a).add(&self.d.mul(&o.c)),
            d: self.c.mul(&o.b).add(&self.d.mul(&o.d)),
        }
    }

    /// Matrix–vector product `self · (x, y)`.
    fn mul_vec(&self, x: &Int, y: &Int) -> (Int, Int) {
        (
            self.a.mul(x).add(&self.b.mul(y)),
            self.c.mul(x).add(&self.d.mul(y)),
        )
    }

    /// The last Euclidean quotient baked into `self` (which must be a nontrivial
    /// product of elementary matrices): `q = ⌊|c|/|a|⌋`, or `⌊|d|/|b|⌋` when
    /// `a = 0` (the single-factor case). Reversing this quotient strips the last
    /// elementary factor, taking `R_j` back to `R_{j-1}`.
    fn last_quotient(&self) -> Nat {
        let (num, den) = if !self.a.is_zero() {
            (self.c.magnitude(), self.a.magnitude())
        } else {
            (self.d.magnitude(), self.b.magnitude())
        };
        num.div_rem(&den).map(|(q, _)| q).unwrap_or_else(Nat::zero)
    }
}

/// One forward Euclidean step on an ordered positive pair (`va > vb > 0`):
/// premultiply `r` by `E(q) = [[0,1],[1,-q]]` and advance
/// `(va, vb) → (vb, va mod vb)`.
fn hgcd_forward(r: &mut HgcdMat, va: &mut Int, vb: &mut Int) {
    let (q, rem) = va
        .magnitude()
        .div_rem(&vb.magnitude())
        .expect("hgcd_forward: vb > 0");
    let qi = Int::from(q);
    // R ← [[0,1],[1,-q]] · R  =  [[c, d], [a - q·c, b - q·d]]
    let nc = r.a.sub(&qi.mul(&r.c));
    let nd = r.b.sub(&qi.mul(&r.d));
    r.a = core::mem::replace(&mut r.c, nc);
    r.b = core::mem::replace(&mut r.d, nd);
    // (va, vb) ← (vb, rem)
    *va = core::mem::replace(vb, Int::from(rem));
}

/// One reverse Euclidean step: strip the last elementary factor of `r`
/// (premultiply by `[[q,1],[1,0]]`) and map `(va, vb) → (q·va + vb, va)`.
fn hgcd_reverse(r: &mut HgcdMat, va: &mut Int, vb: &mut Int, q: &Nat) {
    let qi = Int::from(q.clone());
    // R ← [[q,1],[1,0]] · R  =  [[q·a + c, q·b + d], [a, b]]
    let na = qi.mul(&r.a).add(&r.c);
    let nb = qi.mul(&r.b).add(&r.d);
    r.c = core::mem::replace(&mut r.a, na);
    r.d = core::mem::replace(&mut r.b, nb);
    // (va, vb) ← (q·va + vb, va)
    let nva = qi.mul(va).add(vb);
    *vb = core::mem::replace(va, nva);
}

/// Back up (reverse Euclidean steps) until `(va, vb)` is an ordered pair
/// `va > vb ≥ 0` with `va > 0`. By Lichtblau's Lemma 3 the resulting pair is a
/// genuine consecutive remainder pair of the original `(m, n)`.
fn hgcd_make_valid(r: &mut HgcdMat, va: &mut Int, vb: &mut Int) {
    while !(va.is_positive() && !vb.is_negative() && Ord::cmp(&*va, &*vb) == Ordering::Greater) {
        if r.is_identity() {
            break;
        }
        let q = r.last_quotient();
        hgcd_reverse(r, va, vb, &q);
    }
}

/// From an ordered genuine pair, walk forward/backward to the pair straddling
/// bit length `h`: `bit_len(va) > h ≥ bit_len(vb)`.
fn hgcd_straddle(r: &mut HgcdMat, va: &mut Int, vb: &mut Int, h: u64) {
    loop {
        if vb.is_zero() {
            break;
        }
        if vb.bit_len() as u64 > h {
            hgcd_forward(r, va, vb);
            continue;
        }
        if (va.bit_len() as u64) <= h && !r.is_identity() {
            let q = r.last_quotient();
            hgcd_reverse(r, va, vb, &q);
            continue;
        }
        break;
    }
}

/// Base case: reduce `(m, n)` to the pair straddling bit length `h`. Runs the
/// in-place double-word Lehmer passes of [`Nat::gcd_lehmer`] (each certified
/// to keep the full remainder above `2^h`, so the straddle is never
/// overshot), with exact single division steps where no pass certifies. The
/// cofactor matrix is accumulated as non-negative magnitudes plus a parity
/// bit (products of elementary matrices have a checkerboard sign pattern),
/// column by column with [`lehmer_cofactor_apply`].
fn hgcd_base(m: &Nat, n: &Nat, h: u64) -> (HgcdMat, Nat, Nat) {
    let mut u = m.limbs.clone();
    let mut v = Vec::with_capacity(u.len());
    v.extend_from_slice(&n.limbs);
    v.resize(u.len(), 0);
    let mut nl = u.len();
    let cap = nl + 3;
    let unit = |bit: bool| {
        let mut w = Vec::with_capacity(cap);
        w.push(Limb::from(bit));
        w
    };
    // Columns (r11, r21) and (r12, r22) of R = identity.
    let (mut r11, mut r21, mut r12, mut r22) = (unit(true), unit(false), unit(false), unit(true));
    let mut odd = false;
    loop {
        let vn = sig_len(&v, nl);
        let vbits = if vn == 0 {
            0
        } else {
            vn as u64 * LIMB_BITS as u64 - v[vn - 1].leading_zeros() as u64
        };
        if vbits <= h {
            break;
        }
        let mut mat = None;
        if nl >= 3 && vn + 1 >= nl {
            let lz = u[nl - 1].leading_zeros();
            let p = nl as u64 * LIMB_BITS as u64 - lz as u64 - 128;
            // Keep every certified remainder ≥ 2^h: y_k ≥ |neg| + 2^(h−p).
            let lo = if h < p {
                Some(1)
            } else if h - p < 127 {
                Some(1u128 << (h - p))
            } else {
                None
            };
            if let Some(lo) = lo {
                mat = lehmer_mat(top128(&u, nl, lz), top128(&v, nl, lz), lo);
            }
        }
        if let Some(mat) = mat {
            lehmer_apply(&mut u[..nl], &mut v[..nl], &mat);
            lehmer_cofactor_apply(&mut r11, &mut r21, &mat);
            lehmer_cofactor_apply(&mut r12, &mut r22, &mat);
            odd ^= mat.odd;
            nl = sig_len(&u, nl);
        } else {
            let (nn, q) = lehmer_div_step(&mut u, &mut v, nl, vn);
            nl = nn;
            lehmer_cofactor_div(&mut r11, &mut r21, &q);
            lehmer_cofactor_div(&mut r12, &mut r22, &q);
            odd = !odd;
        }
    }
    let int = |mag: Vec<Limb>, neg: bool| {
        let mut x = Nat { limbs: mag };
        x.normalize();
        Int::from_sign_magnitude(if neg { Sign::Negative } else { Sign::Positive }, x)
    };
    // Signed R: [[+r11, −r12], [−r21, +r22]] for an even factor count, the
    // negation for an odd one.
    let r = HgcdMat {
        a: int(r11, odd),
        b: int(r12, !odd),
        c: int(r21, !odd),
        d: int(r22, odd),
    };
    u.truncate(nl);
    let mut vv = v;
    vv.truncate(nl);
    let mut vv = Nat { limbs: vv };
    vv.normalize();
    (r, Nat { limbs: u }, vv)
}

/// Applies a child cofactor matrix to a full-precision pair using the known
/// high part: given `child·(hi_x, hi_y) = (px, py)` (the child's reduced pair)
/// and the low `k`-bit remainders `(xl, yl)`, returns
/// `child·(x, y) = 2^k·(px, py) + child·(xl, yl)`. Only the low-bit product is
/// computed, halving the multiplication size versus a full matrix–vector apply.
fn hgcd_apply_split(child: &HgcdMat, px: &Nat, py: &Nat, xl: &Int, yl: &Int, k: u64) -> (Int, Int) {
    let (lo0, lo1) = child.mul_vec(xl, yl);
    let hi0 = Int::from(px.clone()).mul_2k(k);
    let hi1 = Int::from(py.clone()).mul_2k(k);
    (hi0.add(&lo0), hi1.add(&lo1))
}

/// Exact correctness check for one Half-GCD reduction: returns `true` iff
/// `det R = ±1` and `(va, vb) = R·(m, n)` exactly. Those two facts together
/// *prove* `gcd(va, vb) = gcd(m, n)` — a unimodular map preserves the gcd.
/// Costs a matrix–vector product plus a 2×2 determinant (`O(M(n))`), so it is
/// asserted on every [`hgcd`] return in debug builds and used by the tests;
/// the release drivers use the `O(n)` residue check
/// [`hgcd_reduction_ok_fast`].
#[cfg(any(test, debug_assertions))]
fn hgcd_reduction_ok(r: &HgcdMat, m: &Nat, n: &Nat, va: &Nat, vb: &Nat) -> bool {
    if !r.a.mul(&r.d).sub(&r.b.mul(&r.c)).abs().is_one() {
        return false;
    }
    let (cva, cvb) = r.mul_vec(&Int::from(m.clone()), &Int::from(n.clone()));
    cva == Int::from(va.clone()) && cvb == Int::from(vb.clone())
}

/// Mersenne prime modulus of the cheap Half-GCD driver guard.
const GUARD_P: u64 = (1 << 61) - 1;

/// `x mod (2^61 − 1)` for `x < 2^125`.
#[inline(always)]
fn guard_reduce(x: u128) -> u64 {
    let p = GUARD_P as u128;
    let x = (x & p) + (x >> 61);
    let x = ((x & p) + (x >> 61)) as u64;
    if x >= GUARD_P { x - GUARD_P } else { x }
}

/// `(x mod 2^64, x mod (2^61 − 1))` of a natural (one linear pass).
fn guard_residues_nat(x: &Nat) -> (u64, u64) {
    let lo = x.limbs.first().copied().unwrap_or(0);
    let mp = x.limbs.iter().rev().fold(0u64, |acc, &l| {
        guard_reduce(((acc as u128) << 64) | l as u128)
    });
    (lo, mp)
}

/// Signed residues of an [`Int`], as for [`guard_residues_nat`].
fn guard_residues_int(x: &Int) -> (u64, u64) {
    let (lo, mp) = guard_residues_nat(&x.magnitude_ref());
    if x.is_negative() {
        (lo.wrapping_neg(), if mp == 0 { 0 } else { GUARD_P - mp })
    } else {
        (lo, mp)
    }
}

/// A value known only by its residues modulo `2^64` and `2^61 − 1` (the pair
/// [`guard_residues_nat`] produces), with the ring operations the extended
/// driver needs to track `rhs`'s cofactor without computing it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct GuardRes(u64, u64);

impl GuardRes {
    const ZERO: GuardRes = GuardRes(0, 0);
    const ONE: GuardRes = GuardRes(1, 1);

    fn of_int(x: &Int) -> GuardRes {
        let (a, b) = guard_residues_int(x);
        GuardRes(a, b)
    }

    fn of_nat(x: &Nat) -> GuardRes {
        let (a, b) = guard_residues_nat(x);
        GuardRes(a, b)
    }

    fn add(self, o: GuardRes) -> GuardRes {
        GuardRes(
            self.0.wrapping_add(o.0),
            guard_reduce(self.1 as u128 + o.1 as u128),
        )
    }

    fn sub(self, o: GuardRes) -> GuardRes {
        GuardRes(
            self.0.wrapping_sub(o.0),
            guard_reduce(self.1 as u128 + (GUARD_P - o.1) as u128),
        )
    }

    fn mul(self, o: GuardRes) -> GuardRes {
        GuardRes(
            self.0.wrapping_mul(o.0),
            guard_reduce(self.1 as u128 * o.1 as u128),
        )
    }
}

/// Cheap always-on guard for one Half-GCD reduction in the drivers: checks
/// `(va, vb) = R·(m, n)` and `det R = ±1` modulo `2^64` and modulo the
/// Mersenne prime `2^61 − 1`, in `O(n)` — instead of the exact `O(M(n))`
/// [`hgcd_reduction_ok`] (still asserted on every [`hgcd`] return in debug
/// builds and by the tests). Any genuine slip in the reduction — a wrong
/// matrix entry or reduced value — shows up in these residues with
/// overwhelming probability, and the drivers then fall back to the trusted
/// Lehmer/Euclid loops; the gcd driver additionally checks exactly that its
/// result divides both inputs, and the extended driver re-verifies its final
/// Bézout identity exactly.
fn hgcd_reduction_ok_fast(r: &HgcdMat, m: &Nat, n: &Nat, va: &Nat, vb: &Nat) -> bool {
    let (a, b, c, d) = (
        guard_residues_int(&r.a),
        guard_residues_int(&r.b),
        guard_residues_int(&r.c),
        guard_residues_int(&r.d),
    );
    let (m, n, va, vb) = (
        guard_residues_nat(m),
        guard_residues_nat(n),
        guard_residues_nat(va),
        guard_residues_nat(vb),
    );
    // Modulo 2^64 (wrapping arithmetic).
    let w = |x: u64, y: u64| x.wrapping_mul(y);
    let det64 = w(a.0, d.0).wrapping_sub(w(b.0, c.0));
    if det64 != 1 && det64 != u64::MAX {
        return false;
    }
    if w(a.0, m.0).wrapping_add(w(b.0, n.0)) != va.0
        || w(c.0, m.0).wrapping_add(w(d.0, n.0)) != vb.0
    {
        return false;
    }
    // Modulo 2^61 − 1.
    let mm = |x: u64, y: u64| guard_reduce(x as u128 * y as u128);
    let add = |x: u64, y: u64| guard_reduce(x as u128 + y as u128);
    let detp = add(mm(a.1, d.1), GUARD_P - mm(b.1, c.1));
    let want = if det64 == 1 { 1 } else { GUARD_P - 1 };
    detp == want
        && add(mm(a.1, m.1), mm(b.1, n.1)) == va.1
        && add(mm(c.1, m.1), mm(d.1, n.1)) == vb.1
}

/// Recursive Half-GCD. Precondition: `m > n ≥ 0`. Returns the cofactor matrix
/// `R` and reduced pair `(va, vb) = R·(m, n)` straddling `2^{⌈bit_len(m)/2⌉}`.
///
/// This wrapper checks the gcd-preservation invariants on *every* return path
/// (including the base case and both early returns of [`hgcd_inner`]) in debug
/// builds, so any drift between `R` and `(va, vb)` trips immediately in tests.
fn hgcd(m: &Nat, n: &Nat) -> (HgcdMat, Nat, Nat) {
    let (r, va, vb) = hgcd_inner(m, n);
    #[cfg(debug_assertions)]
    {
        debug_assert!(
            hgcd_reduction_ok(&r, m, n, &va, &vb),
            "hgcd invariant (va,vb)=R·(m,n) with det R=±1 violated"
        );
    }
    (r, va, vb)
}

/// The low `k` bits of `x` (`x mod 2^k`), copied straight from the limbs.
fn low_bits(x: &Nat, k: u64) -> Nat {
    let full = (k / LIMB_BITS as u64) as usize;
    let rem = (k % LIMB_BITS as u64) as u32;
    let take = (full + usize::from(rem != 0)).min(x.limbs.len());
    let mut limbs = x.limbs[..take].to_vec();
    if rem != 0 && take == full + 1 {
        limbs[full] &= (1 << rem) - 1;
    }
    let mut out = Nat { limbs };
    out.normalize();
    out
}

/// One Half-GCD reduction step of the drivers, on the top part only: runs
/// [`hgcd`] on `(a ≫ k, b ≫ k)` and applies the resulting matrix to the full
/// pair through [`hgcd_apply_split`] (only the low `k` bits are multiplied),
/// backing up with [`hgcd_make_valid`] if the top-part quotients overshoot.
/// Returns `(R, va, vb)` with `(va, vb) = R·(a, b)` a genuine consecutive
/// remainder pair (`va > vb ≥ 0`), or `None` when no reduction was made.
/// Cheaper per stripped bit than a full-size [`hgcd`] because neither the
/// second recursive phase nor the final matrix product is needed.
/// Precondition: `a > b`.
fn hgcd_reduce_top(a: &Nat, b: &Nat, k: u64) -> Option<(HgcdMat, Nat, Nat)> {
    let f0 = a.shr(k);
    let g0 = b.shr(k);
    if f0.cmp_ref(&g0) != Ordering::Greater {
        return None;
    }
    let (rc, ri, rip1) = hgcd(&f0, &g0);
    if rc.is_identity() {
        return None;
    }
    let (al, bl) = (Int::from(low_bits(a, k)), Int::from(low_bits(b, k)));
    let (mut va, mut vb) = hgcd_apply_split(&rc, &ri, &rip1, &al, &bl, k);
    let mut r = rc;
    hgcd_make_valid(&mut r, &mut va, &mut vb);
    if r.is_identity() || va.is_negative() || vb.is_negative() {
        return None;
    }
    Some((r, va.magnitude(), vb.magnitude()))
}

/// Recursion body for [`hgcd`]; see that wrapper for the checked contract.
fn hgcd_inner(m: &Nat, n: &Nat) -> (HgcdMat, Nat, Nat) {
    debug_assert!(m.cmp_ref(n) == Ordering::Greater, "hgcd requires m > n");
    let bm = m.bit_len();
    let h = bm.div_ceil(2);
    // Already straddled: nothing to reduce.
    if n.is_zero() || n.bit_len() <= h {
        return (HgcdMat::identity(), m.clone(), n.clone());
    }
    if m.limbs.len() <= hgcd_recursion_base_limbs() {
        return hgcd_base(m, n, h);
    }

    // Phase 1: recurse on the high half (strip the low k ≈ bit_len(m)/2 bits),
    // then apply the resulting cofactor matrix to the full pair. Only the low
    // k bits need the (expensive) matrix–vector product: the high part of
    // rc1·(m, n) is already the child's reduced pair, since
    //   rc1·(m, n) = 2^k·rc1·(m≫k, n≫k) + rc1·(m mod 2^k, n mod 2^k).
    let k = (bm - 1) / 2;
    let f0 = m.shr(k);
    let g0 = n.shr(k);
    let f1 = m.checked_sub(&f0.shl(k)).expect("low bits of m");
    let g1 = n.checked_sub(&g0.shl(k)).expect("low bits of n");
    let (rc1, ri, rip1) = if f0.cmp_ref(&g0) == Ordering::Greater {
        hgcd(&f0, &g0)
    } else {
        (HgcdMat::identity(), f0.clone(), g0.clone())
    };
    let (mut va, mut vb) = hgcd_apply_split(&rc1, &ri, &rip1, &Int::from(f1), &Int::from(g1), k);
    let mut r = rc1;
    hgcd_make_valid(&mut r, &mut va, &mut vb);

    // Phase 2: recurse on the high part of the partially reduced pair to strip
    // the second quarter, reaching the half-length straddle point.
    if !vb.is_zero() && vb.bit_len() as u64 > h {
        let bva = va.bit_len() as u64;
        let k2 = (2 * h).saturating_sub(bva);
        if k2 > 0 {
            let vam = va.magnitude();
            let vbm = vb.magnitude();
            let f2 = vam.shr(k2);
            let g2 = vbm.shr(k2);
            if f2.cmp_ref(&g2) == Ordering::Greater {
                let f2l = vam.checked_sub(&f2.shl(k2)).expect("low bits of va");
                let g2l = vbm.checked_sub(&g2.shl(k2)).expect("low bits of vb");
                let (rc2, sj, sjp1) = hgcd(&f2, &g2);
                (va, vb) = hgcd_apply_split(&rc2, &sj, &sjp1, &Int::from(f2l), &Int::from(g2l), k2);
                r = rc2.mul(&r);
                hgcd_make_valid(&mut r, &mut va, &mut vb);
            }
        }
    }

    // Land exactly on the straddle pair (a bounded number of adjustment steps).
    hgcd_straddle(&mut r, &mut va, &mut vb, h);
    debug_assert!(
        !va.is_negative() && !vb.is_negative() && va.cmp(&vb) == Ordering::Greater,
        "hgcd must return an ordered non-negative pair"
    );
    // The gcd-preservation invariants — (va, vb) = R·(m, n) and det R = ±1 — are
    // checked on every return path by the [`hgcd`] wrapper.
    (r, va.magnitude(), vb.magnitude())
}

// ---------------------------------------------------------------------------
// Allocation-free double-word Lehmer (Jebelean's exact-quotient condition).
//
// For `A = x·2^p + A₀`, `B = y·2^p + B₀` (`0 ≤ A₀, B₀ < 2^p`), run Euclid on the
// leading words `(x, y)`, accumulating the product `L` of elementary matrices
// `E(q) = [[0,1],[1,−q]]`. Its rows have opposite checkerboard sign patterns,
// so `L·(A, B) = 2^p·(x_k, y_k) + L·(A₀, B₀)` where each low-part term lies
// strictly between `−2^p·(negative coefficient)` and `+2^p·(positive one)`.
// Hence (Jebelean, "Improving the multiprecision Euclidean algorithm", 1993)
//   y_k ≥ |negative entry of row 2|   and   x_k − y_k ≥ |row-1 − row-2 entry|
// guarantee `L·(A, B) = (A', B')` with `A' > B' > 0`; by Lichtblau's Lemma 3
// (see the Half-GCD notes below) such a pair is a genuine consecutive
// remainder pair, so `L` is exactly a prefix of the Euclidean quotient
// sequence. Only one (usually subtraction-only) quotient is needed per step,
// and on a 128-bit window the entries stay below 2^64: each pass strips about
// 64 bits with single-limb multipliers.
// ---------------------------------------------------------------------------

/// A Lehmer cofactor matrix stored as non-negative magnitudes `[[a1, b1],
/// [a2, b2]]` plus the parity of its elementary-factor count: for an even count
/// the signed matrix is `[[a1, −b1], [−a2, b2]]`, for an odd one
/// `[[−a1, b1], [a2, −b2]]`.
#[derive(Clone, Copy, Debug)]
struct LehmerMat {
    a1: u64,
    b1: u64,
    a2: u64,
    b2: u64,
    odd: bool,
}

/// `(⌊x/y⌋, x mod y)` for `x ≥ y > 0`, trying the (by far most common, Gauss–
/// Kuzmin) quotients 1–3 by subtraction before paying for a division.
#[inline(always)]
fn lehmer_divrem(x: u128, y: u128) -> (u128, u128) {
    let r = x - y;
    if r < y {
        return (1, r);
    }
    let r = r - y;
    if r < y {
        return (2, r);
    }
    let r = r - y;
    if r < y {
        return (3, r);
    }
    if x >> 64 == 0 {
        let (x, y) = (x as u64, y as u64);
        let q = x / y;
        return (q as u128, (x - q * y) as u128);
    }
    // Estimate from the leading 64 bits with one hardware word division (a
    // u128 division is a software routine). `⌊x_t/(y_t+1)⌋ ≤ ⌊x/y⌋`, and with
    // y_t ≥ 2^32 it undershoots by at most ~2, fixed by subtraction.
    let s = x.leading_zeros();
    let xt = ((x << s) >> 64) as u64;
    let yt = ((y << s) >> 64) as u64;
    if yt >> 32 != 0 {
        let mut q = (xt / yt.saturating_add(1)) as u128;
        let mut r = x - q * y;
        while r >= y {
            r -= y;
            q += 1;
        }
        return (q, r);
    }
    let q = x / y;
    (q, x - q * y)
}

/// One double-word Lehmer pass on aligned leading windows `x ≥ y` of `A ≥ B`
/// (see the section comment). Returns `None` when not even one quotient can be
/// certified (the caller then takes one full division step).
///
/// With `lo > 0` each accepted step additionally guarantees the new full
/// remainder exceeds `2^p·lo` (requiring `y_k ≥ |neg| + lo`), which lets the
/// Half-GCD base case stop exactly at its straddle point.
fn lehmer_mat(mut x: u128, mut y: u128, lo: u128) -> Option<LehmerMat> {
    let (mut a1, mut b1, mut a2, mut b2) = (1u64, 0u64, 0u64, 1u64);
    let mut odd = false;
    let mut any = false;
    while y != 0 {
        let (q, r) = lehmer_divrem(x, y);
        if q >> 64 != 0 {
            break;
        }
        // Magnitudes add (the rows have opposite sign patterns); the products
        // fit u128 since q, a2, b2 < 2^64.
        let q = q as u64 as u128;
        let na2 = a1 as u128 + q * a2 as u128;
        let nb2 = b1 as u128 + q * b2 as u128;
        if (na2 | nb2) >> 64 != 0 {
            break;
        }
        // Jebelean's condition for the new pair (y, r) and new parity.
        let ok = if odd {
            r >= na2 + lo && y - r >= b2 as u128 + nb2
        } else {
            r >= nb2 + lo && y - r >= a2 as u128 + na2
        };
        if !ok {
            break;
        }
        (a1, b1, a2, b2) = (a2, b2, na2 as u64, nb2 as u64);
        (x, y) = (y, r);
        odd = !odd;
        any = true;
    }
    any.then_some(LehmerMat {
        a1,
        b1,
        a2,
        b2,
        odd,
    })
}

/// In place: `(u, v) ← (p·X − q·Y, r·Y − s·X)` with `(X, Y) = (u, v)`, or
/// `(v, u)` when `SWAP`. Both results must be non-negative and fit the slice
/// (guaranteed for a certified [`LehmerMat`]). Two carry chains per output,
/// the borrow folded into the subtrahend's carry.
#[inline(always)]
fn lehmer_kernel<const SWAP: bool>(u: &mut [Limb], v: &mut [Limb], p: u64, q: u64, r: u64, s: u64) {
    let (mut cp1, mut cn1, mut cp2, mut cn2) = (0u128, 0u128, 0u128, 0u128);
    let (p, q, r, s) = (p as u128, q as u128, r as u128, s as u128);
    for (ui, vi) in u.iter_mut().zip(v.iter_mut()) {
        let (x, y) = if SWAP { (*vi, *ui) } else { (*ui, *vi) };
        let (x, y) = (x as u128, y as u128);
        let t = p * x + cp1;
        let w = q * y + cn1;
        let (o1, b1) = (t as Limb).overflowing_sub(w as Limb);
        cp1 = t >> LIMB_BITS;
        cn1 = (w >> LIMB_BITS) + b1 as u128;
        let t = r * y + cp2;
        let w = s * x + cn2;
        let (o2, b2) = (t as Limb).overflowing_sub(w as Limb);
        cp2 = t >> LIMB_BITS;
        cn2 = (w >> LIMB_BITS) + b2 as u128;
        *ui = o1;
        *vi = o2;
    }
    debug_assert!(cp1 == cn1 && cp2 == cn2, "Lehmer matrix left a carry");
}

/// Applies a certified [`LehmerMat`] to the operand pair in place:
/// `(u, v) ← L·(u, v)` (equal-length slices, `v` zero-padded).
fn lehmer_apply(u: &mut [Limb], v: &mut [Limb], m: &LehmerMat) {
    if m.odd {
        // (−a1·u + b1·v, a2·u − b2·v)
        lehmer_kernel::<true>(u, v, m.b1, m.a1, m.a2, m.b2);
    } else {
        // (a1·u − b1·v, −a2·u + b2·v)
        lehmer_kernel::<false>(u, v, m.a1, m.b1, m.b2, m.a2);
    }
}

/// Cofactor magnitudes update `(s0, s1) ← (a1·s0 + b1·s1, a2·s0 + b2·s1)` in
/// place (the signed cofactors alternate in sign exactly like the matrix rows,
/// so the magnitudes only ever add). Grows both vectors by the final carry.
fn lehmer_cofactor_apply(s0: &mut Vec<Limb>, s1: &mut Vec<Limb>, m: &LehmerMat) {
    let (a1, b1, a2, b2) = (m.a1 as u128, m.b1 as u128, m.a2 as u128, m.b2 as u128);
    let (mut c0, mut c1) = (0u128, 0u128);
    for (x, y) in s0.iter_mut().zip(s1.iter_mut()) {
        let (xv, yv) = (*x as u128, *y as u128);
        // a·x + carry < 2^128, then + b·y on the low word only: the carry stays
        // below 2^65, keeping every sum inside u128.
        let t = a1 * xv + c0;
        let w = b1 * yv + (t as Limb as u128);
        *x = w as Limb;
        c0 = (t >> LIMB_BITS) + (w >> LIMB_BITS);
        let t = a2 * xv + c1;
        let w = b2 * yv + (t as Limb as u128);
        *y = w as Limb;
        c1 = (t >> LIMB_BITS) + (w >> LIMB_BITS);
    }
    while c0 != 0 || c1 != 0 {
        s0.push(c0 as Limb);
        s1.push(c1 as Limb);
        c0 >>= LIMB_BITS;
        c1 >>= LIMB_BITS;
    }
}

/// The top 128 bits of the `n`-limb value `w` (`n ≥ 3`), shifted left by `s`
/// (the leading-zero count of the larger operand's top limb, so that operand's
/// window has its top bit at position 127; the other operand is read at the
/// same alignment).
#[inline(always)]
fn top128(w: &[Limb], n: usize, s: u32) -> u128 {
    let hi = ((w[n - 1] as u128) << LIMB_BITS) | w[n - 2] as u128;
    if s == 0 {
        hi
    } else {
        (hi << s) | (w[n - 3] >> (LIMB_BITS - s)) as u128
    }
}

/// Significant length of `w[..n]` (drops zero high limbs).
#[inline(always)]
fn sig_len(w: &[Limb], mut n: usize) -> usize {
    while n > 0 && w[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// Reads limbs as a `u128` (`w.len() ≤ 2`).
#[inline(always)]
fn limbs_u128(w: &[Limb]) -> u128 {
    w.iter()
        .rev()
        .fold(0u128, |acc, &l| (acc << LIMB_BITS) | l as u128)
}

/// Exact Euclid on two words `x ≥ y`, accumulating a [`LehmerMat`] for as long
/// as its entries fit a limb. Returns the matrix (or `None` if the very first
/// quotient is already ≥ 2^64) and the reduced pair.
fn euclid_mat_u128(mut x: u128, mut y: u128) -> (Option<LehmerMat>, u128, u128) {
    let (mut a1, mut b1, mut a2, mut b2) = (1u64, 0u64, 0u64, 1u64);
    let mut odd = false;
    let mut any = false;
    while y != 0 {
        let (q, r) = lehmer_divrem(x, y);
        if q >> 64 != 0 {
            break;
        }
        let q = q as u64 as u128;
        let na2 = a1 as u128 + q * a2 as u128;
        let nb2 = b1 as u128 + q * b2 as u128;
        if (na2 | nb2) >> 64 != 0 {
            break;
        }
        (a1, b1, a2, b2) = (a2, b2, na2 as u64, nb2 as u64);
        (x, y) = (y, r);
        odd = !odd;
        any = true;
    }
    let m = any.then_some(LehmerMat {
        a1,
        b1,
        a2,
        b2,
        odd,
    });
    (m, x, y)
}

/// One full Euclidean division step on the in-place Lehmer buffers:
/// `(u, v) ← (v, u mod v)` (`u[..n] ≥ v[..n]`, `v` has `vn` significant limbs).
/// Returns the new length (`vn`) and the quotient. Used when a Lehmer pass
/// cannot certify a quotient (one too large for a word, or a lopsided pair).
fn lehmer_div_step(u: &mut [Limb], v: &mut [Limb], n: usize, vn: usize) -> (usize, Nat) {
    let mut a = Nat {
        limbs: u[..n].to_vec(),
    };
    a.normalize();
    let b = Nat {
        limbs: v[..vn].to_vec(),
    };
    let (q, r) = a.div_rem(&b).expect("v is non-zero");
    u[..vn].copy_from_slice(&v[..vn]);
    u[vn..n].fill(0);
    v[..n].fill(0);
    v[..r.limbs.len()].copy_from_slice(&r.limbs);
    (vn, q)
}

/// Cofactor magnitudes for one explicit quotient `q`:
/// `(s0, s1) ← (s1, s0 + q·s1)`, keeping both vectors the same length.
fn lehmer_cofactor_div(s0: &mut Vec<Limb>, s1: &mut Vec<Limb>, q: &Nat) {
    let mut a = Nat {
        limbs: core::mem::take(s0),
    };
    a.normalize();
    let mut b = Nat {
        limbs: core::mem::take(s1),
    };
    b.normalize();
    let ns1 = a.add(&q.mul(&b));
    *s0 = b.limbs;
    *s1 = ns1.limbs;
    let len = s0.len().max(s1.len());
    s0.resize(len, 0);
    s1.resize(len, 0);
}

/// Euclid's gcd on two words (`u128` Lehmer-free: subtraction-first quotients,
/// then a single-word binary GCD once both fit a limb).
fn u128_gcd_euclid(mut x: u128, mut y: u128) -> u128 {
    if x < y {
        core::mem::swap(&mut x, &mut y);
    }
    while y >> LIMB_BITS != 0 {
        let (_, r) = lehmer_divrem(x, y);
        (x, y) = (y, r);
    }
    if y == 0 {
        return x;
    }
    let r = (x % y) as u64;
    u64_gcd(y as u64, r) as u128
}

/// Compares two normalized little-endian limb slices as naturals.
pub(crate) fn cmp_limbs(a: &[Limb], b: &[Limb]) -> Ordering {
    match a.len().cmp(&b.len()) {
        Ordering::Equal => {}
        non_eq => return non_eq,
    }
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        match x.cmp(y) {
            Ordering::Equal => continue,
            non_eq => return non_eq,
        }
    }
    Ordering::Equal
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

    /// Number of significant base-`2^64` limbs (`0` for zero).
    #[cfg(test)]
    #[inline]
    pub(crate) fn limb_count(&self) -> usize {
        self.limbs.len()
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
    #[inline]
    fn cmp_ref(&self, other: &Nat) -> Ordering {
        cmp_limbs(&self.limbs, &other.limbs)
    }

    /// A natural holding a copy of the (normalized) `limbs`, with room for
    /// `capacity` limbs so a following in-place update need not reallocate.
    pub(crate) fn from_limbs_with_capacity(limbs: &[Limb], capacity: usize) -> Nat {
        let mut v = Vec::with_capacity(capacity.max(limbs.len()));
        v.extend_from_slice(limbs);
        Nat { limbs: v }
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

    /// Returns `self · rhs`, dispatching by operand shape along a
    /// schoolbook → Karatsuba → Toom-3 (or Toom-3.2 / block split for
    /// unbalanced operands) → Toom-4 → NTT ladder.
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
        } else if ntt_worthwhile(self.limbs.len(), rhs.limbs.len(), false) {
            mul_ntt(self, rhs)
        } else {
            let (a, b) = order_by_len(&self.limbs, &rhs.limbs);
            product_with(a, b, |out, scratch| mul_slices(a, b, out, scratch))
        }
    }

    /// Toom-4 multiplication at the top level (seven quarter-size products,
    /// see [`toom4_into`]), falling back to the general slice dispatcher when
    /// the operands are too unbalanced for a four-way split. Kept callable
    /// directly so tests and crossover measurements can force the path.
    fn mul_toom4(&self, rhs: &Nat) -> Nat {
        let (a, b) = order_by_len(&self.limbs, &rhs.limbs);
        product_with(a, b, |out, scratch| {
            if b.len() > 3 * a.len().div_ceil(4) {
                toom4_into(a, b, out, scratch);
            } else {
                mul_slices(a, b, out, scratch);
            }
        })
    }

    /// Toom-3 multiplication at the top level (five third-size products, see
    /// [`toom3_into`]), falling back to the general slice dispatcher when the
    /// operands are too unbalanced for a three-way split.
    #[cfg(test)]
    fn mul_toom3(&self, rhs: &Nat) -> Nat {
        let (a, b) = order_by_len(&self.limbs, &rhs.limbs);
        product_with(a, b, |out, scratch| {
            if b.len() > 2 * a.len().div_ceil(3) {
                toom3_into(a, b, out, scratch);
            } else {
                mul_slices(a, b, out, scratch);
            }
        })
    }
    /// Quadratic schoolbook (long) multiplication (see [`mul_into_schoolbook`]).
    fn mul_schoolbook(&self, rhs: &Nat) -> Nat {
        let mut out = alloc::vec![0 as Limb; self.limbs.len() + rhs.limbs.len()];
        // Already zeroed: rows over the longer operand directly.
        let (long, short) = order_by_len(&self.limbs, &rhs.limbs);
        mul_into_schoolbook(short, long, &mut out);
        let mut n = Nat { limbs: out };
        n.normalize();
        n
    }
}

/// Quadratic schoolbook (long) multiplication of `a·b` into the zeroed buffer
/// `out` (`out.len() == a.len() + b.len()`).
///
/// The main loop is an `addmul_2` ([`addmul2_row`]): each pass adds two
/// multiplier limbs' worth of products (`out += (a₀ + a₁·2^64)·b`), halving
/// the passes over `out` and keeping two independent limb products in flight
/// per iteration instead of one serial multiply-accumulate chain.
fn mul_into_schoolbook(a: &[Limb], b: &[Limb], out: &mut [Limb]) {
    debug_assert_eq!(out.len(), a.len() + b.len());
    let rn = b.len();
    if a.is_empty() || rn == 0 {
        return;
    }
    let mut i = 0;
    while i + 2 <= a.len() {
        addmul2_row(&mut out[i..i + rn + 2], b, a[i], a[i + 1]);
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

/// Basecase product: writes `a·b` into `out` (`out.len() == a.len() +
/// b.len()`, previous contents ignored), as `addmul_2` rows over the longer
/// operand (fewer, longer rows amortize the per-row setup and flush).
fn mul_basecase(a: &[Limb], b: &[Limb], out: &mut [Limb]) {
    out.fill(0);
    let (long, short) = order_by_len(a, b);
    mul_into_schoolbook(short, long, out);
}
/// The `addmul_2` kernel: `row += (a0 + a1·2^64)·b` with
/// `row.len() == b.len() + 2` (the sum must fit in `row`).
///
/// Two interleaved `addmul_1` rows with a one-limb lag: at position `j` the
/// first row adds `a0·b[j]` (plus its carry `c0`), and the second row adds
/// `a1·b[j−1]` (plus its carry `c1`) on top of that result before it is
/// stored. Each row's carry is its own two-instruction loop-carried chain
/// (the incoming carry is added last, after the carry-independent
/// `product + limb` sum), so the two chains overlap, and each product costs
/// one multiply and two add-with-carry pairs.
#[inline(always)]
fn addmul2_row(row: &mut [Limb], b: &[Limb], a0: Limb, a1: Limb) {
    use crate::limb::DLimb;
    let rn = b.len();
    debug_assert_eq!(row.len(), rn + 2);
    let (mut c0, mut c1, mut bprev): (Limb, Limb, Limb) = (0, 0, 0);
    // `p·q + x < 2^128` for limbs p, q, x, so the products-plus-limb sums
    // cannot overflow; the carries are added with explicit overflow flags so
    // LLVM keeps them at the end of each sum.
    let mut step = |o: &mut Limb, bj: Limb| {
        let w0 = a0 as DLimb * bj as DLimb + *o as DLimb;
        let (s0, k0) = (w0 as Limb).overflowing_add(c0);
        c0 = (w0 >> LIMB_BITS) as Limb + Limb::from(k0);
        let w1 = a1 as DLimb * bprev as DLimb + s0 as DLimb;
        let (s1, k1) = (w1 as Limb).overflowing_add(c1);
        c1 = (w1 >> LIMB_BITS) as Limb + Limb::from(k1);
        *o = s1;
        bprev = bj;
    };
    // Two positions per iteration (halves the loop overhead).
    let (rows2, row_tail) = row[..rn].as_chunks_mut::<2>();
    let (bs2, b_tail) = b.as_chunks::<2>();
    for (o, bj) in rows2.iter_mut().zip(bs2) {
        let [o0, o1] = o;
        step(o0, bj[0]);
        step(o1, bj[1]);
    }
    if let (Some(o), Some(&bj)) = (row_tail.first_mut(), b_tail.first()) {
        step(o, bj);
    }
    // Position rn: the first row's carry, the second row's last product and
    // carry; position rn + 1: what overflows from those.
    let t0 = row[rn] as DLimb + c0 as DLimb;
    let t1 = a1 as DLimb * bprev as DLimb + (t0 as Limb) as DLimb + c1 as DLimb;
    row[rn] = t1 as Limb;
    let top = row[rn + 1] as DLimb + (t0 >> LIMB_BITS) + (t1 >> LIMB_BITS);
    row[rn + 1] = top as Limb;
    debug_assert_eq!(top >> LIMB_BITS, 0, "addmul_2 top carry escaped");
}
/// Sums `a + b` into `out`, which must be exactly `max(len) + 1` limbs; every
/// limb of `out` is written (the top limb is the 0/1 carry).
fn add_full(a: &[Limb], b: &[Limb], out: &mut [Limb]) {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    debug_assert_eq!(out.len(), long.len() + 1);
    let (head, tail) = out.split_at_mut(short.len());
    let mut carry = add_n3(head, &long[..short.len()], short);
    let (mid, last) = tail.split_at_mut(long.len() - short.len());
    mid.copy_from_slice(&long[short.len()..]);
    if carry {
        carry = incr(mid);
    }
    last[0] = Limb::from(carry);
}
/// Karatsuba multiplication of `a·b` into `out` (`out.len() == a.len() +
/// b.len()`, previous contents ignored), using `scratch` for every intermediate —
/// no per-node allocation. `z0` and `z2` land directly in the disjoint halves
/// of `out`; only the middle term needs scratch space.
///
/// This is the subtractive form: the middle term is
/// `z0 + z2 − (a0 − a1)·(b0 − b1)`, so the third product works on two
/// `h`-limb differences (tracked with their signs) rather than on `h + 1`-limb
/// sums, keeping every recursive product exactly half size.
fn kara_into(a: &[Limb], b: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    debug_assert_eq!(out.len(), a.len() + b.len());
    if a.len().min(b.len()) < KARATSUBA_THRESHOLD {
        mul_basecase(a, b, out);
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
        out[z0_len..2 * h].fill(0);
        kara_into(a1, b1, &mut out[2 * h..], scratch);
    } else {
        out[z0_len..].fill(0);
    }
    // da = |a0 − a1|, db = |b0 − b1|, zm = da·db.
    let (da, rest) = scratch.split_at_mut(a0.len());
    let (db, rest) = rest.split_at_mut(b0.len());
    let neg = abs_diff_into(a0, a1, da) != abs_diff_into(b0, b1, db);
    let (zm, rest) = rest.split_at_mut(z0_len);
    let (t, rest) = rest.split_at_mut(z0_len + 1);
    kara_into(da, db, zm, rest);
    // t = z0 + z2 ∓ zm = a0·b1 + a1·b0 (non-negative, fits z0_len + 1 limbs).
    let z2_len = if have_z2 { a1.len() + b1.len() } else { 0 };
    add_full(&out[..z0_len], &out[2 * h..2 * h + z2_len], t);
    if neg {
        // (a0 − a1)·(b0 − b1) < 0: the middle term is z0 + z2 + |zm|.
        add_at(t, 0, zm);
    } else {
        sub_in_place(t, zm);
    }
    // Trim t's zero top limbs: the value always fits, the buffer may not.
    let t_len = t.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
    add_at(out, h, &t[..t_len]);
}

/// Writes `|x − y|` into `out` (`out.len() == x.len() ≥ y.len()`), returning
/// whether `x < y`.
fn abs_diff_into(x: &[Limb], y: &[Limb], out: &mut [Limb]) -> bool {
    debug_assert!(out.len() == x.len() && x.len() >= y.len());
    // x < y only if x has nothing above y's length and loses at the highest
    // differing limb.
    let (xl, xh) = x.split_at(y.len());
    let x_less = xh.iter().all(|&l| l == 0) && padded_less(xl, y);
    let (ol, oh) = out.split_at_mut(y.len());
    if x_less {
        // x's high part is zero; so is the result's.
        sub_n3(ol, y, xl);
        oh.fill(0);
    } else {
        let borrow = sub_n3(ol, xl, y);
        oh.copy_from_slice(xh);
        let escaped = borrow && decr(oh);
        debug_assert!(!escaped);
    }
    x_less
}

/// Whether `x < y` for equal-length slices (most significant limb last).
#[inline]
fn padded_less(x: &[Limb], y: &[Limb]) -> bool {
    debug_assert_eq!(x.len(), y.len());
    x.iter()
        .rev()
        .zip(y.iter().rev())
        .find(|(p, q)| p != q)
        .is_some_and(|(p, q)| p < q)
}

/// Karatsuba squaring of `a` into the zeroed buffer `out`
/// (`out.len() == 2·a.len()`), sharing the same scratch discipline as
/// [`kara_into`].
fn kara_sqr_into(a: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    debug_assert_eq!(out.len(), 2 * a.len());
    if a.len() < KARATSUBA_SQR_THRESHOLD {
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
/// at most `n` limbs each: each level consumes `≤ 3n + 6` limbs and recurses on
/// `≤ n/2 + 2`.
#[cfg(test)]
fn kara_scratch_len(n: usize) -> usize {
    let mut need = 0;
    let mut m = n;
    while m >= KARATSUBA_THRESHOLD {
        need += 3 * m + 6;
        m = m / 2 + 2;
    }
    need
}

// --- Slice-based Toom-Cook ---
//
// Every Toom node evaluates its operands, recurses on the pointwise products
// through [`mul_slices`]/[`sqr_slices`] and interpolates, all inside the
// caller's output buffer plus one shared scratch slice: no allocation below
// the top level. The interpolation sequences follow Bodrato's approach
// ("Towards optimal Toom-Cook multiplication", WAIFI 2007; Brent &
// Zimmermann, MCA §1.3.3): exact divisions by 2 (shifts) and by 3 or 5
// (Hensel division by a single-limb inverse), every intermediate a
// non-negative integer, only the values at negative points carrying a sign.

/// Scratch size sufficient for [`mul_slices`]/[`sqr_slices`] on operands of
/// at most `n` limbs: every node (Karatsuba, Toom-3/3.2/4 or the unbalanced
/// block split) uses at most `6m + 64` limbs at size `m` and recurses at most
/// to `m/2 + 2`.
fn toom_scratch_len(n: usize) -> usize {
    let mut need = 0;
    let mut m = n;
    while m >= KARATSUBA_THRESHOLD {
        need += 6 * m + 64;
        m = m / 2 + 2;
    }
    need
}

/// Orders two limb slices as `(longer, shorter)`.
fn order_by_len<'a>(a: &'a [Limb], b: &'a [Limb]) -> (&'a [Limb], &'a [Limb]) {
    if a.len() >= b.len() { (a, b) } else { (b, a) }
}

/// Runs a slice product `f(out, scratch)` into a fresh `a.len() + b.len()`
/// limb buffer with a [`toom_scratch_len`] scratch, returning the result.
fn product_with(a: &[Limb], b: &[Limb], f: impl FnOnce(&mut [Limb], &mut [Limb])) -> Nat {
    let mut out = alloc::vec![0 as Limb; a.len() + b.len()];
    let mut scratch = alloc::vec![0 as Limb; toom_scratch_len(a.len().max(b.len()))];
    f(&mut out, &mut scratch);
    let mut n = Nat { limbs: out };
    n.normalize();
    n
}

/// Writes `a·b` into `out` (`out.len() == a.len() + b.len()`, previous
/// contents ignored), choosing schoolbook, Karatsuba, Toom-3, Toom-3.2,
/// Toom-4 or a block split of the longer operand by shape. `scratch` must
/// hold [`toom_scratch_len`]`(max(len))` limbs.
fn mul_slices(a: &[Limb], b: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    let (a, b) = order_by_len(a, b);
    let (la, lb) = (a.len(), b.len());
    debug_assert_eq!(out.len(), la + lb);
    if lb < KARATSUBA_THRESHOLD {
        mul_basecase(a, b, out);
    } else if la >= 2 * lb {
        mul_unbalanced_into(a, b, out, scratch);
    } else if lb < TOOM3_THRESHOLD {
        kara_into(a, b, out, scratch);
    } else if lb >= TOOM4_THRESHOLD && lb > 3 * la.div_ceil(4) {
        toom4_into(a, b, out, scratch);
    } else if lb > 2 * la.div_ceil(3) {
        toom3_into(a, b, out, scratch);
    } else {
        toom32_into(a, b, out, scratch);
    }
}

/// Writes `a²` into `out` (`out.len() == 2·a.len()`, previous contents
/// ignored), by symmetric schoolbook, Karatsuba, Toom-3 or Toom-4 squaring.
fn sqr_slices(a: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    let n = a.len();
    debug_assert_eq!(out.len(), 2 * n);
    // Checked from the top: with the current tuning Toom-3 squaring takes
    // over straight from the symmetric schoolbook, so Karatsuba squaring
    // only serves the Montgomery path.
    if n >= TOOM4_SQR_THRESHOLD {
        toom4_sqr_into(a, out, scratch);
    } else if n >= TOOM3_SQR_THRESHOLD {
        toom3_sqr_into(a, out, scratch);
    } else if n >= KARATSUBA_SQR_THRESHOLD {
        out.fill(0);
        kara_sqr_into(a, out, scratch);
    } else {
        out.fill(0);
        sqr_into(a, out);
    }
}

/// `a·b` for `a.len() ≥ 2·b.len()`: `a` is cut into `b.len()`-limb blocks,
/// each multiplied by `b` as a balanced product and added in at its offset
/// (a trailing short block recurses as the transposed unbalanced product).
fn mul_unbalanced_into(a: &[Limb], b: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    let lb = b.len();
    let (first, mut rest) = a.split_at(lb);
    mul_slices(first, b, &mut out[..2 * lb], scratch);
    out[2 * lb..].fill(0);
    let mut off = lb;
    while !rest.is_empty() {
        let (chunk, tail) = rest.split_at(rest.len().min(lb));
        let (tmp, sub) = scratch.split_at_mut(chunk.len() + lb);
        mul_slices(chunk, b, tmp, sub);
        add_at(out, off, trim_limbs(tmp));
        off += chunk.len();
        rest = tail;
    }
}

/// `x` without its zero high limbs.
fn trim_limbs(x: &[Limb]) -> &[Limb] {
    &x[..x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1)]
}

/// `dst = src − dst` for equal-length slices with `src ≥ dst`.
fn rsub_in_place(dst: &mut [Limb], src: &[Limb]) {
    debug_assert_eq!(dst.len(), src.len());
    let borrow = carry_chain2(dst, src, |d, s, b| sbb_b(s, d, b));
    debug_assert!(!borrow, "rsub_in_place: src < dst");
}

/// `x = |x − y|` (`x.len() ≥ y.len()`, values compared as unnormalized
/// little-endian numbers), returning whether `x < y` held.
fn abs_diff_in_place(x: &mut [Limb], y: &[Limb]) -> bool {
    let (xl, xh) = x.split_at_mut(y.len());
    let x_less = xh.iter().all(|&l| l == 0) && padded_less(xl, y);
    if x_less {
        rsub_in_place(xl, y);
    } else {
        sub_in_place(x, y);
    }
    x_less
}

/// `dst[i] = f(dst[i], (src·2^s)[i], flag)` for `i ≤ src.len()` (the last
/// position takes the bits shifted out of `src`, when `dst` has room for
/// them), four limbs per iteration; returns the outgoing flag. `0 < s < 64`.
#[inline(always)]
fn lsh_chain(
    dst: &mut [Limb],
    src: &[Limb],
    s: u32,
    f: impl Fn(Limb, Limb, bool) -> (Limb, bool),
) -> bool {
    debug_assert!(s > 0 && s < LIMB_BITS);
    let n = src.len();
    let r = LIMB_BITS - s;
    let mut prev: Limb = 0;
    let mut c = false;
    let (d4, dt) = dst[..n].as_chunks_mut::<4>();
    let (s4, st) = src.as_chunks::<4>();
    for (d, x) in d4.iter_mut().zip(s4) {
        let v0 = (x[0] << s) | (prev >> r);
        let v1 = (x[1] << s) | (x[0] >> r);
        let v2 = (x[2] << s) | (x[1] >> r);
        let v3 = (x[3] << s) | (x[2] >> r);
        prev = x[3];
        let (r0, c0) = f(d[0], v0, c);
        let (r1, c1) = f(d[1], v1, c0);
        let (r2, c2) = f(d[2], v2, c1);
        let (r3, c3) = f(d[3], v3, c2);
        *d = [r0, r1, r2, r3];
        c = c3;
    }
    for (d, &x) in dt.iter_mut().zip(st) {
        let (rr, c1) = f(*d, (x << s) | (prev >> r), c);
        prev = x;
        *d = rr;
        c = c1;
    }
    let spill = prev >> r;
    match dst.get_mut(n) {
        Some(d) => {
            let (rr, c1) = f(*d, spill, c);
            *d = rr;
            c1
        }
        None => {
            debug_assert_eq!(spill, 0);
            c
        }
    }
}

/// `dst += src·2^s` for `0 < s < 64`; the sum must fit in `dst`.
fn add_lsh_in_place(dst: &mut [Limb], src: &[Limb], s: u32) {
    let n = src.len();
    let carry = lsh_chain(dst, src, s, adc_b) && incr(dst.get_mut(n + 1..).unwrap_or_default());
    debug_assert!(!carry, "add_lsh_in_place carry escaped");
}

/// `dst −= src·2^s` for `0 < s < 64`; requires `dst ≥ src·2^s`.
fn sub_lsh_in_place(dst: &mut [Limb], src: &[Limb], s: u32) {
    let n = src.len();
    let borrow = lsh_chain(dst, src, s, sbb_b) && decr(dst.get_mut(n + 1..).unwrap_or_default());
    debug_assert!(!borrow, "sub_lsh_in_place: dst < src·2^s");
}

/// `x <<= s` in place for `0 < s < 64`; the bits shifted out must be zero.
fn shl_in_place(x: &mut [Limb], s: u32) {
    debug_assert!(s > 0 && s < LIMB_BITS);
    let r = LIMB_BITS - s;
    let mut carry: Limb = 0;
    let (x4, xt) = x.as_chunks_mut::<4>();
    for l in x4.iter_mut() {
        let [a, b, c, d] = *l;
        *l = [
            (a << s) | carry,
            (b << s) | (a >> r),
            (c << s) | (b >> r),
            (d << s) | (c >> r),
        ];
        carry = d >> r;
    }
    for l in xt.iter_mut() {
        let v = *l;
        *l = (v << s) | carry;
        carry = v >> r;
    }
    debug_assert_eq!(carry, 0, "shl_in_place dropped bits");
}

/// `x >>= 1` in place; the low bit must be zero (exact halving).
fn shr1_in_place(x: &mut [Limb]) {
    debug_assert!(x.first().is_none_or(|&l| l & 1 == 0), "inexact halving");
    let n = x.len();
    for i in 0..n.saturating_sub(1) {
        x[i] = (x[i] >> 1) | (x[i + 1] << (LIMB_BITS - 1));
    }
    if let Some(last) = x.last_mut() {
        *last >>= 1;
    }
}
/// `d^-1 mod 2^64` for odd `d`, by Newton's iteration (compile-time).
const fn inv_limb_const(d: Limb) -> Limb {
    let mut inv = d; // correct to 3 bits for odd d
    let mut i = 0;
    while i < 5 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(inv)));
        i += 1;
    }
    inv
}

/// An exact small odd divisor with its inverse modulo `2^64`.
#[derive(Clone, Copy)]
struct ExactDivisor {
    d: Limb,
    inv: Limb,
}

impl ExactDivisor {
    const fn new(d: Limb) -> Self {
        ExactDivisor {
            d,
            inv: inv_limb_const(d),
        }
    }

    /// One Hensel step: the quotient limb for `x` given the incoming borrow,
    /// and the outgoing borrow.
    #[inline(always)]
    fn step(self, x: Limb, borrow: Limb) -> (Limb, Limb) {
        let (s, b1) = x.overflowing_sub(borrow);
        let q = s.wrapping_mul(self.inv);
        let hi = ((q as u128 * self.d as u128) >> LIMB_BITS) as Limb;
        (q, hi + Limb::from(b1))
    }
}

const DIV3: ExactDivisor = ExactDivisor::new(3);
const DIV9: ExactDivisor = ExactDivisor::new(9);
const DIV15: ExactDivisor = ExactDivisor::new(15);

/// `x /= d` in place for a small odd `d` dividing `x` exactly: Hensel
/// (low-to-high) exact division, one multiply per limb and no trial
/// quotients (MCA §1.4.7).
fn divexact_in_place(x: &mut [Limb], d: ExactDivisor) {
    let mut borrow: Limb = 0;
    for l in x.iter_mut() {
        let (q, b) = d.step(*l, borrow);
        *l = q;
        borrow = b;
    }
    debug_assert_eq!(borrow, 0, "inexact small division");
}

/// Two independent exact divisions `x /= dx`, `y /= dy` (equal lengths) in
/// one loop, so their latency-bound borrow chains overlap.
fn divexact_pair(x: &mut [Limb], dx: ExactDivisor, y: &mut [Limb], dy: ExactDivisor) {
    debug_assert_eq!(x.len(), y.len());
    let (mut bx, mut by): (Limb, Limb) = (0, 0);
    for (u, v) in x.iter_mut().zip(y.iter_mut()) {
        let (q, b) = dx.step(*u, bx);
        *u = q;
        bx = b;
        let (q, b) = dy.step(*v, by);
        *v = q;
        by = b;
    }
    debug_assert!(bx == 0 && by == 0, "inexact small division");
}
/// `x *= m` in place for a single-limb `m`; the product must fit.
fn mul_small_in_place(x: &mut [Limb], m: Limb) {
    let mut carry = 0;
    for l in x.iter_mut() {
        let (lo, hi) = mac(0, *l, m, carry);
        *l = lo;
        carry = hi;
    }
    debug_assert_eq!(carry, 0, "mul_small_in_place overflow");
}

/// Copies `src` into the low limbs of `dst` and zeroes the rest.
fn copy_padded(dst: &mut [Limb], src: &[Limb]) {
    let (lo, hi) = dst.split_at_mut(src.len());
    lo.copy_from_slice(src);
    hi.fill(0);
}

/// Replaces `x` by `(x + y)/2` and `y` by `(x − y)/2` (equal lengths,
/// `x ≥ y`, both sums even). Used to split the values at `±t` into their
/// even and odd parts.
fn half_sum_diff(x: &mut [Limb], y: &mut [Limb]) {
    rsub_in_place(y, x); // y = x − y
    shl_in_place(x, 1);
    sub_in_place(x, y); // x = 2x − (x − y) = x + y
    shr1_in_place(x);
    shr1_in_place(y);
}

/// Toom-3 evaluation of `x0 + x1·t + x2·t²` (`x0`, `x1` of `k` limbs, `x2`
/// of at most `k`) into `k + 1`-limb buffers: `e1 = x(1)`, `em1 = |x(−1)|`
/// and, when given, `e2 = x(2)`. Returns whether `x(−1) < 0`.
fn toom3_eval(
    x0: &[Limb],
    x1: &[Limb],
    x2: &[Limb],
    e1: &mut [Limb],
    em1: &mut [Limb],
    e2: Option<&mut [Limb]>,
) -> bool {
    add_full(x0, x2, em1); // x0 + x2
    e1.copy_from_slice(em1);
    add_in_place(e1, x1);
    let neg = abs_diff_in_place(em1, x1);
    if let Some(e2) = e2 {
        // x(2) = 2·(x(1) + x2) − x0.
        e2.copy_from_slice(e1);
        add_in_place(e2, x2);
        shl_in_place(e2, 1);
        sub_in_place(e2, x0);
    }
    neg
}

/// Toom-3 interpolation. On entry `out[..2k]` holds `v0`, `out[4k..]` holds
/// `v∞`, and the `2k + 2`-limb buffers hold `w1 = v(1)`, `wm1 = |v(−1)|`
/// (negative iff `neg`) and `w2 = v(2)`. Leaves the full product in `out`.
fn toom3_interp(
    out: &mut [Limb],
    k: usize,
    w1: &mut [Limb],
    wm1: &mut [Limb],
    neg: bool,
    w2: &mut [Limb],
) {
    // w2 = (v(2) − v(−1))/3 = c1 + c2 + 3c3 + 5c4.
    if neg {
        add_in_place(w2, wm1);
    } else {
        sub_in_place(w2, wm1);
    }
    divexact_in_place(w2, DIV3);
    // wm1 = (v(1) − v(−1))/2 = c1 + c3.
    if neg {
        add_in_place(wm1, w1);
    } else {
        rsub_in_place(wm1, w1);
    }
    shr1_in_place(wm1);
    let (lo, vinf) = out.split_at_mut(4 * k);
    let v0 = &lo[..2 * k];
    // w1 = v(1) − c0 = c1 + c2 + c3 + c4.
    sub_in_place(w1, v0);
    // w2 = (w2 − w1)/2 − 2c4 = c3.
    sub_in_place(w2, w1);
    shr1_in_place(w2);
    sub_lsh_in_place(w2, vinf, 1);
    // w1 = w1 − (c1 + c3) − c4 = c2.
    sub_in_place(w1, wm1);
    sub_in_place(w1, vinf);
    // wm1 = (c1 + c3) − c3 = c1.
    sub_in_place(wm1, w2);
    lo[2 * k..].fill(0);
    add_at(out, k, trim_limbs(wm1));
    add_at(out, 2 * k, trim_limbs(w1));
    add_at(out, 3 * k, trim_limbs(w2));
}

/// Toom-3 product `a·b` into `out` for `a.len() ≥ b.len() > 2·⌈a.len()/3⌉`:
/// five products of about a third of the size, at the points
/// `0, 1, −1, 2, ∞`.
fn toom3_into(a: &[Limb], b: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    let k = a.len().div_ceil(3);
    debug_assert!(a.len() >= b.len() && b.len() > 2 * k);
    let (a0, a1, a2) = (&a[..k], &a[k..2 * k], &a[2 * k..]);
    let (b0, b1, b2) = (&b[..k], &b[k..2 * k], &b[2 * k..]);
    let (ea1, s) = scratch.split_at_mut(k + 1);
    let (eam1, s) = s.split_at_mut(k + 1);
    let (ea2, s) = s.split_at_mut(k + 1);
    let (eb1, s) = s.split_at_mut(k + 1);
    let (ebm1, s) = s.split_at_mut(k + 1);
    let (eb2, s) = s.split_at_mut(k + 1);
    let (w1, s) = s.split_at_mut(2 * k + 2);
    let (wm1, s) = s.split_at_mut(2 * k + 2);
    let (w2, rest) = s.split_at_mut(2 * k + 2);
    let na = toom3_eval(a0, a1, a2, ea1, eam1, Some(&mut *ea2));
    let nb = toom3_eval(b0, b1, b2, eb1, ebm1, Some(&mut *eb2));
    mul_slices(ea1, eb1, w1, rest);
    mul_slices(eam1, ebm1, wm1, rest);
    mul_slices(ea2, eb2, w2, rest);
    let (lo, hi) = out.split_at_mut(4 * k);
    mul_slices(a0, b0, &mut lo[..2 * k], rest);
    mul_slices(a2, b2, hi, rest);
    toom3_interp(out, k, w1, wm1, na != nb, w2);
}

/// Toom-3 squaring of `a` into `out` (`a.len() ≥ 3`).
fn toom3_sqr_into(a: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    let k = a.len().div_ceil(3);
    let (a0, a1, a2) = (&a[..k], &a[k..2 * k], &a[2 * k..]);
    let (e1, s) = scratch.split_at_mut(k + 1);
    let (em1, s) = s.split_at_mut(k + 1);
    let (e2, s) = s.split_at_mut(k + 1);
    let (w1, s) = s.split_at_mut(2 * k + 2);
    let (wm1, s) = s.split_at_mut(2 * k + 2);
    let (w2, rest) = s.split_at_mut(2 * k + 2);
    toom3_eval(a0, a1, a2, e1, em1, Some(&mut *e2));
    sqr_slices(e1, w1, rest);
    sqr_slices(em1, wm1, rest);
    sqr_slices(e2, w2, rest);
    let (lo, hi) = out.split_at_mut(4 * k);
    sqr_slices(a0, &mut lo[..2 * k], rest);
    sqr_slices(a2, hi, rest);
    toom3_interp(out, k, w1, wm1, false, w2);
}

/// Toom-3.2 product `a·b` into `out` for `⌈a.len()/3⌉ < b.len() ≤
/// 2·⌈a.len()/3⌉`: `a` in three parts and `b` in two, four products at the
/// points `0, 1, −1, ∞`.
fn toom32_into(a: &[Limb], b: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    let k = a.len().div_ceil(3);
    debug_assert!(b.len() > k && b.len() <= 2 * k);
    let (a0, a1, a2) = (&a[..k], &a[k..2 * k], &a[2 * k..]);
    let (b0, b1) = b.split_at(k);
    let (ea1, s) = scratch.split_at_mut(k + 1);
    let (eam1, s) = s.split_at_mut(k + 1);
    let (eb1, s) = s.split_at_mut(k + 1);
    let (ebm1, s) = s.split_at_mut(k + 1);
    let (w1, s) = s.split_at_mut(2 * k + 2);
    let (wm1, rest) = s.split_at_mut(2 * k + 2);
    let na = toom3_eval(a0, a1, a2, ea1, eam1, None);
    add_full(b0, b1, eb1);
    copy_padded(ebm1, b0);
    let nb = abs_diff_in_place(ebm1, b1);
    mul_slices(ea1, eb1, w1, rest);
    mul_slices(eam1, ebm1, wm1, rest);
    let (lo, vinf) = out.split_at_mut(3 * k);
    mul_slices(a0, b0, &mut lo[..2 * k], rest);
    mul_slices(a2, b1, vinf, rest);
    // (w1, wm1) = (c0 + c2, c1 + c3) from v(±1).
    half_sum_diff(w1, wm1);
    let (even, odd) = if na != nb { (wm1, w1) } else { (w1, wm1) };
    sub_in_place(even, &lo[..2 * k]);
    sub_in_place(odd, vinf);
    lo[2 * k..].fill(0);
    add_at(out, k, trim_limbs(odd));
    add_at(out, 2 * k, trim_limbs(even));
}

/// Toom-4 evaluation of `x0 + x1·t + x2·t² + x3·t³` (`x0..x2` of `k` limbs,
/// `x3` of at most `k`) into `k + 1`-limb buffers `e = [x(1), |x(−1)|, x(2),
/// |x(−2)|, 8·x(1/2)]`, using the two `k + 1`-limb temporaries `t`, `u`.
/// Returns the signs of `x(−1)` and `x(−2)`.
fn toom4_eval(
    x: [&[Limb]; 4],
    e: [&mut [Limb]; 5],
    t: &mut [Limb],
    u: &mut [Limb],
) -> (bool, bool) {
    let [x0, x1, x2, x3] = x;
    let [e1, em1, e2, em2, eh] = e;
    // ±1: (x0 + x2) ± (x1 + x3).
    add_full(x0, x2, t);
    add_full(x1, x3, u);
    e1.copy_from_slice(t);
    add_in_place(e1, u);
    em1.copy_from_slice(t);
    let n1 = abs_diff_in_place(em1, u);
    // ±2: (x0 + 4·x2) ± 2·(x1 + 4·x3).
    copy_padded(t, x0);
    add_lsh_in_place(t, x2, 2);
    copy_padded(u, x1);
    add_lsh_in_place(u, x3, 2);
    shl_in_place(u, 1);
    e2.copy_from_slice(t);
    add_in_place(e2, u);
    em2.copy_from_slice(t);
    let n2 = abs_diff_in_place(em2, u);
    // 8·x(1/2) = ((2·x0 + x1)·2 + x2)·2 + x3.
    copy_padded(eh, x0);
    shl_in_place(eh, 1);
    add_in_place(eh, x1);
    shl_in_place(eh, 1);
    add_in_place(eh, x2);
    shl_in_place(eh, 1);
    add_in_place(eh, x3);
    (n1, n2)
}

/// Toom-4 interpolation. On entry `out[..2k]` holds `v0`, `out[6k..]` holds
/// `v∞`, and the `2k + 2`-limb buffers hold `w = [v(1), |v(−1)|, v(2),
/// |v(−2)|, 64·v(1/2)]` with the signs `neg1`, `neg2` of `v(−1)`, `v(−2)`.
/// Leaves the full product in `out`.
fn toom4_interp(out: &mut [Limb], k: usize, w: [&mut [Limb]; 5], neg1: bool, neg2: bool) {
    let [w1, wm1, w2, wm2, wh] = w;
    let (lo, vinf) = out.split_at_mut(6 * k);
    let v0 = &lo[..2 * k];
    // Even/odd parts at 1: e1 = c0 + c2 + c4 + c6, o1 = c1 + c3 + c5.
    half_sum_diff(w1, wm1);
    let (e1, o1) = if neg1 { (wm1, w1) } else { (w1, wm1) };
    // At 2: e2 = c0 + 4c2 + 16c4 + 64c6, o2 = 2·(c1 + 4c3 + 16c5).
    half_sum_diff(w2, wm2);
    let (e2, o2) = if neg2 { (wm2, w2) } else { (w2, wm2) };
    shr1_in_place(o2);
    // e1 = c2 + c4; e2 = (e2 − c0 − 64c6)/4 − e1 = 3c4.
    sub_in_place(e1, v0);
    sub_in_place(e1, vinf);
    sub_in_place(e2, v0);
    sub_lsh_in_place(e2, vinf, 6);
    shr1_in_place(e2);
    shr1_in_place(e2);
    sub_in_place(e2, e1);
    // wh = (64·v(1/2) − 64c0 − c6 + 4·(3c4) − 16·(c2 + c4))/2
    //    = (64·v(1/2) − 64c0 − 16c2 − 4c4 − c6)/2 = 16c1 + 4c3 + c5,
    // using 3c4 before its division (each step stays non-negative).
    sub_lsh_in_place(wh, v0, 6);
    sub_in_place(wh, vinf);
    add_lsh_in_place(wh, e2, 2);
    sub_lsh_in_place(wh, e1, 4);
    shr1_in_place(wh);
    // o2 = F = 3c3 + 15c5, wh = G = 15c1 + 3c3, o1 = 15·o1 − F − G = 9c3.
    sub_in_place(o2, o1);
    sub_in_place(wh, o1);
    mul_small_in_place(o1, 15);
    sub_in_place(o1, o2);
    sub_in_place(o1, wh);
    // c4 = 3c4/3 and c3 = 9c3/9, as one interleaved pass; then c2.
    divexact_pair(e2, DIV3, o1, DIV9);
    sub_in_place(e1, e2);
    // c5 = (F − 3c3)/15, c1 = (G − 3c3)/15.
    sub_lsh_in_place(o2, o1, 1);
    sub_in_place(o2, o1);
    sub_lsh_in_place(wh, o1, 1);
    sub_in_place(wh, o1);
    divexact_pair(o2, DIV15, wh, DIV15);
    let (c1, c2, c3, c4, c5) = (&*wh, &*e1, &*o1, &*e2, &*o2);
    lo[2 * k..].fill(0);
    add_at(out, k, trim_limbs(c1));
    add_at(out, 2 * k, trim_limbs(c2));
    add_at(out, 3 * k, trim_limbs(c3));
    add_at(out, 4 * k, trim_limbs(c4));
    add_at(out, 5 * k, trim_limbs(c5));
}

/// Splits `x` into four `k`-limb parts (the last one shorter).
fn quarters(x: &[Limb], k: usize) -> [&[Limb]; 4] {
    [&x[..k], &x[k..2 * k], &x[2 * k..3 * k], &x[3 * k..]]
}

/// Toom-4 product `a·b` into `out` for `a.len() ≥ b.len() > 3·⌈a.len()/4⌉`:
/// seven products of about a quarter of the size, at the points
/// `0, ±1, ±2, 1/2, ∞`.
fn toom4_into(a: &[Limb], b: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    let k = a.len().div_ceil(4);
    debug_assert!(a.len() >= b.len() && b.len() > 3 * k);
    let (xa, xb) = (quarters(a, k), quarters(b, k));
    let (ea, s) = scratch.split_at_mut(5 * (k + 1));
    let (eb, s) = s.split_at_mut(5 * (k + 1));
    let (t, s) = s.split_at_mut(k + 1);
    let (u, s) = s.split_at_mut(k + 1);
    let (w, rest) = s.split_at_mut(5 * (2 * k + 2));
    let mut ea: [&mut [Limb]; 5] = split5(ea);
    let mut eb: [&mut [Limb]; 5] = split5(eb);
    let ea_r = ea.each_mut().map(|x| &mut **x);
    let (na1, na2) = toom4_eval(xa, ea_r, t, u);
    let eb_r = eb.each_mut().map(|x| &mut **x);
    let (nb1, nb2) = toom4_eval(xb, eb_r, t, u);
    let mut w: [&mut [Limb]; 5] = split5(w);
    for i in 0..5 {
        mul_slices(ea[i], eb[i], &mut *w[i], rest);
    }
    let (lo, hi) = out.split_at_mut(6 * k);
    mul_slices(xa[0], xb[0], &mut lo[..2 * k], rest);
    mul_slices(xa[3], xb[3], hi, rest);
    let w_r = w.each_mut().map(|x| &mut **x);
    toom4_interp(out, k, w_r, na1 != nb1, na2 != nb2);
}

/// Toom-4 squaring of `a` into `out` (`a.len() ≥ 4`).
fn toom4_sqr_into(a: &[Limb], out: &mut [Limb], scratch: &mut [Limb]) {
    let k = a.len().div_ceil(4);
    let xa = quarters(a, k);
    let (ea, s) = scratch.split_at_mut(5 * (k + 1));
    let (t, s) = s.split_at_mut(k + 1);
    let (u, s) = s.split_at_mut(k + 1);
    let (w, rest) = s.split_at_mut(5 * (2 * k + 2));
    let mut ea: [&mut [Limb]; 5] = split5(ea);
    let ea_r = ea.each_mut().map(|x| &mut **x);
    toom4_eval(xa, ea_r, t, u);
    let mut w: [&mut [Limb]; 5] = split5(w);
    for i in 0..5 {
        sqr_slices(ea[i], &mut *w[i], rest);
    }
    let (lo, hi) = out.split_at_mut(6 * k);
    sqr_slices(xa[0], &mut lo[..2 * k], rest);
    sqr_slices(xa[3], hi, rest);
    let w_r = w.each_mut().map(|x| &mut **x);
    toom4_interp(out, k, w_r, false, false);
}

/// Splits `x` into five equal consecutive parts.
fn split5(x: &mut [Limb]) -> [&mut [Limb]; 5] {
    let m = x.len() / 5;
    let (p0, x) = x.split_at_mut(m);
    let (p1, x) = x.split_at_mut(m);
    let (p2, x) = x.split_at_mut(m);
    let (p3, p4) = x.split_at_mut(m);
    [p0, p1, p2, p3, p4]
}

impl Nat {
    /// Returns `self²`, using a symmetric schoolbook squaring (roughly half
    /// the limb multiplications of the general `mul`), Toom-3/Toom-4
    /// squaring, or the NTT where its two-transform squaring wins (per the
    /// internal `ntt_worthwhile` shape heuristic).
    pub fn square(&self) -> Nat {
        let n = self.limbs.len();
        if n == 0 {
            return Nat::zero();
        }
        if n < KARATSUBA_SQR_THRESHOLD.min(TOOM3_SQR_THRESHOLD) {
            self.square_schoolbook()
        } else if ntt_worthwhile(n, n, true) {
            mul_ntt(self, self)
        } else {
            let a = &self.limbs[..];
            product_with(a, a, |out, scratch| sqr_slices(a, out, scratch))
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
    #[cfg(test)]
    fn square_karatsuba(&self) -> Nat {
        let n = self.limbs.len();
        if n < KARATSUBA_SQR_THRESHOLD {
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
    #[cfg(test)]
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
    /// Word-size operands use a binary GCD, larger ones Lehmer's algorithm
    /// (~64 bits per in-place pass), and very large ones the subquadratic
    /// Half-GCD driver.
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
        // Large operands: subquadratic Half-GCD driver. The smaller operand
        // bounds the work, so gate on it.
        if self.limbs.len().min(rhs.limbs.len()) >= HGCD_GCD_THRESHOLD {
            return self.gcd_hgcd(rhs);
        }
        self.gcd_lehmer(rhs)
    }

    /// Subquadratic GCD: a Lehmer-style loop driven by the recursive Half-GCD.
    /// Each iteration reduces `(a, b)` by a Half-GCD of its top part
    /// ([`hgcd_reduce_top`]) to a consecutive remainder pair a constant
    /// fraction shorter, in `O(M(n)·log n)`, so the whole GCD costs
    /// `O(M(n)·log² n)` at worst. The residual small pair is finished with
    /// [`Nat::gcd_lehmer`]. Precondition: both non-zero.
    fn gcd_hgcd(&self, rhs: &Nat) -> Nat {
        // The Half-GCD path carries an always-on correctness guard. If it ever
        // trips (a latent reduction bug), fall back to the trusted Lehmer GCD so
        // a wrong result can never escape into `Rational` reduction.
        self.gcd_hgcd_floor(rhs, HGCD_GCD_THRESHOLD)
            .unwrap_or_else(|| self.gcd_lehmer(rhs))
    }

    /// [`Nat::gcd_hgcd`] with an explicit limb `floor`: once the smaller operand
    /// drops below `floor` limbs the residual is finished with Lehmer. Split out
    /// so the differential tests can force the Half-GCD path (and its full
    /// recursion) on medium-sized inputs where an independent oracle is cheap.
    fn gcd_hgcd_floor(&self, rhs: &Nat, floor: usize) -> Option<Nat> {
        let floor = floor.max(3);
        let mut a = self.clone();
        let mut b = rhs.clone();
        if a.cmp_ref(&b) == Ordering::Less {
            core::mem::swap(&mut a, &mut b);
        }
        while b.limbs.len() >= floor {
            let h = a.bit_len().div_ceil(2);
            // Unbalanced (Half-GCD a no-op) or equal operands (`hgcd` needs
            // `m > n` strictly): a single full division makes progress.
            if b.bit_len() <= h || a.cmp_ref(&b) != Ordering::Greater {
                let (_, r) = a.div_rem(&b).expect("b is non-zero");
                a = core::mem::replace(&mut b, r);
                continue;
            }
            let k = a.bit_len() * HGCD_SPLIT_NUM / HGCD_SPLIT_DEN;
            let Some((r, va, vb)) = hgcd_reduce_top(&a, &b, k) else {
                let (_, r) = a.div_rem(&b).expect("b is non-zero");
                a = core::mem::replace(&mut b, r);
                continue;
            };
            // Always-on guard: `det R = ±1` and `(va, vb) = R·(a, b)` (checked
            // on residues) mean this reduction preserved the gcd. If it fails,
            // bail so the caller falls back to the trusted Lehmer GCD.
            if !hgcd_reduction_ok_fast(&r, &a, &b, &va, &vb) {
                return None;
            }
            if vb.bit_len() >= b.bit_len() {
                // Defensive: force progress if a level somehow made none.
                let (_, r) = a.div_rem(&b).expect("b is non-zero");
                a = core::mem::replace(&mut b, r);
            } else {
                a = va;
                b = vb;
            }
        }
        let g = if b.is_zero() { a } else { a.gcd_lehmer(&b) };
        // Belt-and-suspenders: the result must divide both original operands.
        // (One division of each — O(M(n)) once — catches any driver bookkeeping
        // slip the per-reduction guard would not.)
        if g.is_zero() {
            return None;
        }
        let divides = |x: &Nat| x.div_rem(&g).map(|(_, r)| r.is_zero()).unwrap_or(false);
        if !divides(self) || !divides(rhs) {
            return None;
        }
        Some(g)
    }

    /// Extended GCD via the Half-GCD driver, on non-negative operands. Returns
    /// `(g, x)` with `g = gcd(self, rhs) ≥ 0` and `x·self ≡ g (mod rhs)`.
    ///
    /// Uses the production floor ([`HGCD_EXTGCD_FLOOR`]): the Half-GCD reductions
    /// run until the smaller operand drops below it, then the residual is
    /// finished by the extended Lehmer loop. Returns `None` if a guard trips,
    /// so the caller can fall back to the trusted Lehmer loop.
    ///
    /// The cofactor `x` is *a* valid Bézout coefficient — when `g = 1` it is the
    /// unique modular inverse mod `rhs`, which is all [`Int::modinv`] needs. It
    /// is **not** guaranteed to be the minimal one plain extended Euclid
    /// returns, so [`Int::extended_gcd`] reduces it to the canonical cofactor.
    pub(crate) fn extgcd_hgcd(&self, rhs: &Nat) -> Option<(Nat, Int)> {
        self.extgcd_hgcd_floor(rhs, HGCD_EXTGCD_FLOOR)
    }

    /// [`Nat::extgcd_hgcd`] with an explicit limb `floor`. Only `self`'s
    /// cofactor column is tracked: with `(u, v) ≡ (s0, s1)·self (mod rhs)`,
    /// each reduction `(u, v) ← R·(u, v)` maps `(s0, s1) ← R·(s0, s1)` (four
    /// products instead of a full matrix product). The `floor` seam lets the
    /// differential tests force the Half-GCD path (and its full recursion) on
    /// medium operands where a plain-Euclid oracle is cheap. Handles zero,
    /// equal and unordered operands.
    pub(crate) fn extgcd_hgcd_floor(&self, rhs: &Nat, floor: usize) -> Option<(Nat, Int)> {
        if rhs.is_zero() {
            return Some((self.clone(), Int::ONE));
        }
        if self.is_zero() {
            return Some((rhs.clone(), Int::ZERO));
        }
        let floor = floor.max(3);
        let mut u = self.clone();
        let mut v = rhs.clone();
        // Invariant: (u, v) = (s0, s1)·self + (t0, t1)·rhs. The `t` column is
        // tracked only by residues ([`GuardRes`]), enough to verify the final
        // Bézout identity in O(n) instead of a full product and division.
        let (mut s0, mut s1) = (Int::ONE, Int::ZERO);
        let (mut t0, mut t1) = (GuardRes::ZERO, GuardRes::ONE);
        if u.cmp_ref(&v) == Ordering::Less {
            core::mem::swap(&mut u, &mut v);
            core::mem::swap(&mut s0, &mut s1);
            core::mem::swap(&mut t0, &mut t1);
        }
        while v.limbs.len() >= floor {
            let h = u.bit_len().div_ceil(2);
            let reduced = if v.bit_len() <= h || u.cmp_ref(&v) != Ordering::Greater {
                None
            } else {
                hgcd_reduce_top(&u, &v, u.bit_len() * HGCD_SPLIT_NUM / HGCD_SPLIT_DEN)
            };
            let Some((rc, va, vb)) = reduced else {
                // Unbalanced, equal, or no top-part progress: one exact
                // Euclidean step.
                let (q, r) = u.div_rem(&v).expect("v is non-zero");
                u = core::mem::replace(&mut v, r);
                let qr = GuardRes::of_nat(&q);
                let ns1 = s0.sub(&s1.mul(&Int::from(q)));
                s0 = core::mem::replace(&mut s1, ns1);
                let nt1 = t0.sub(t1.mul(qr));
                t0 = core::mem::replace(&mut t1, nt1);
                continue;
            };
            // Always-on guard (residues of `det rc = ±1`, `(va, vb) = rc·(u, v)`):
            // bail to the trusted Lehmer loop if it trips.
            if !hgcd_reduction_ok_fast(&rc, &u, &v, &va, &vb) {
                return None;
            }
            (s0, s1) = rc.mul_vec(&s0, &s1);
            let (ra, rb, rcc, rd) = (
                GuardRes::of_int(&rc.a),
                GuardRes::of_int(&rc.b),
                GuardRes::of_int(&rc.c),
                GuardRes::of_int(&rc.d),
            );
            (t0, t1) = (ra.mul(t0).add(rb.mul(t1)), rcc.mul(t0).add(rd.mul(t1)));
            u = va;
            v = vb;
        }
        let (g, x, t) = if v.is_zero() {
            (u, s0, t0)
        } else {
            // g = x'·u + y'·v  ⇒  g = (x'·s0 + y'·s1)·self + (x'·t0 + y'·t1)·rhs.
            let (g, xp, yp) = u.extgcd_lehmer_full(&v);
            let t = GuardRes::of_int(&xp)
                .mul(t0)
                .add(GuardRes::of_int(&yp).mul(t1));
            (g, xp.mul(&s0).add(&yp.mul(&s1)), t)
        };
        // Always-on final check of the Bézout identity g = x·self + t·rhs on
        // residues (a slip in the cofactor bookkeeping cannot pass unnoticed).
        let lhs = GuardRes::of_int(&x)
            .mul(GuardRes::of_nat(self))
            .add(t.mul(GuardRes::of_nat(rhs)));
        if lhs != GuardRes::of_nat(&g) {
            return None;
        }
        Some((g, x))
    }

    /// Stein's binary GCD (no division). Precondition: both operands non-zero.
    /// No longer on the dispatch path (Lehmer wins from 3 limbs up, and the
    /// u128 fast path covers everything below), but kept as the independent
    /// reference oracle for the gcd tests.
    #[cfg(test)]
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

    /// Lehmer's GCD (Knuth TAOCP §4.5.2, Algorithm L) with Jebelean's exact
    /// quotient condition: the leading 128 bits yield a single-limb cofactor
    /// matrix ([`lehmer_mat`], ~64 bits per pass), applied to both operands in
    /// place ([`lehmer_apply`]) — no allocation per pass. A pass that cannot
    /// certify a quotient (a huge quotient) takes one full division instead.
    /// Precondition: both operands non-zero.
    fn gcd_lehmer(&self, rhs: &Nat) -> Nat {
        let (big, small) = if self.cmp_ref(rhs) == Ordering::Less {
            (rhs, self)
        } else {
            (self, rhs)
        };
        let mut u = big.limbs.clone();
        let mut v = Vec::with_capacity(u.len());
        v.extend_from_slice(&small.limbs);
        v.resize(u.len(), 0);
        let mut n = u.len();
        loop {
            // Invariant: u[..n] ≥ v[..n], u[n-1] ≠ 0.
            let vn = sig_len(&v, n);
            if vn == 0 {
                u.truncate(n);
                return Nat { limbs: u };
            }
            if n <= 2 {
                return Nat::from_u128(u128_gcd_euclid(limbs_u128(&u[..n]), limbs_u128(&v[..n])));
            }
            let m = if vn + 1 >= n {
                let s = u[n - 1].leading_zeros();
                lehmer_mat(top128(&u, n, s), top128(&v, n, s), 0)
            } else {
                None
            };
            if let Some(m) = m {
                lehmer_apply(&mut u[..n], &mut v[..n], &m);
                n = sig_len(&u, n);
                debug_assert!(cmp_limbs(&u[..n], &v[..sig_len(&v, n)]) == Ordering::Greater);
            } else {
                n = lehmer_div_step(&mut u, &mut v, n, vn).0;
            }
        }
    }

    /// Returns `(g, x)` with `g = gcd(self, rhs)` and `x·self ≡ g (mod rhs)`
    /// (*a* Bézout cofactor, not normalized), by the Half-GCD driver on large
    /// operands and Lehmer's algorithm otherwise. Precondition: both non-zero.
    pub(crate) fn extgcd_cofactor(&self, rhs: &Nat) -> (Nat, Int) {
        if self.limbs.len().min(rhs.limbs.len()) >= HGCD_MODINV_THRESHOLD
            && let Some((g, x)) = self.extgcd_hgcd(rhs)
        {
            return (g, x);
        }
        self.extgcd_lehmer(rhs)
    }

    /// Extended Lehmer GCD: returns `(g, x)` with `g = gcd(self, rhs)` and
    /// `x·self ≡ g (mod rhs)` — *a* Bézout cofactor of `self`, not normalized.
    /// Only `self`'s cofactor is tracked (the caller recovers the other with
    /// one exact division). The cofactor pair alternates in sign exactly like
    /// the Euclidean matrix rows, so only its magnitudes are stored (plus a
    /// parity bit) and every Lehmer matrix is applied to them as a non-negative
    /// matrix, in place. Precondition: both non-zero.
    pub(crate) fn extgcd_lehmer(&self, rhs: &Nat) -> (Nat, Int) {
        let (g, x, _) = self.extgcd_lehmer_impl::<false>(rhs);
        (g, x)
    }

    /// Extended Lehmer GCD tracking both cofactors: `(g, x, y)` with
    /// `g = x·self + y·rhs`, the cofactors being exactly those of the plain
    /// extended Euclidean algorithm (Lehmer's certified matrices reproduce its
    /// quotient sequence). Precondition: both non-zero.
    pub(crate) fn extgcd_lehmer_full(&self, rhs: &Nat) -> (Nat, Int, Int) {
        self.extgcd_lehmer_impl::<true>(rhs)
    }

    /// Body of [`Nat::extgcd_lehmer`] / [`Nat::extgcd_lehmer_full`]; `rhs`'s
    /// cofactor is tracked (and returned, else zero) only when `BOTH`.
    fn extgcd_lehmer_impl<const BOTH: bool>(&self, rhs: &Nat) -> (Nat, Int, Int) {
        debug_assert!(!self.is_zero() && !rhs.is_zero());
        // Invariants: u = ±(s0·self − t0·rhs), v = ∓(s1·self − t1·rhs) — the
        // signed cofactor matrix [[s0, t0], [s1, t1]] has a checkerboard sign
        // pattern: s0 ≥ 0, s1 ≤ 0 when `odd` is false, the reverse when true,
        // and each t opposite to its s. Only magnitudes are stored.
        let swapped = self.cmp_ref(rhs) == Ordering::Less;
        let (big, small) = if swapped { (rhs, self) } else { (self, rhs) };
        let mut u = big.limbs.clone();
        let mut v = Vec::with_capacity(u.len());
        v.extend_from_slice(&small.limbs);
        v.resize(u.len(), 0);
        let mut n = u.len();
        // Initial matrix: identity, or after Euclid's q = 0 swap step
        // [[0, 1], [1, 0]] with odd parity.
        // Cofactor magnitudes never exceed the smaller operand (plus the
        // transient carry limbs), so one up-front capacity avoids regrowth.
        let cap = small.limbs.len() + 3;
        let cof = |bit: bool| {
            let mut w = Vec::with_capacity(cap);
            w.push(Limb::from(bit));
            w
        };
        let mut s0 = cof(!swapped);
        let mut s1 = cof(swapped);
        let (mut t0, mut t1) = if BOTH {
            (cof(swapped), cof(!swapped))
        } else {
            (Vec::new(), Vec::new())
        };
        let mut odd = swapped;
        loop {
            let vn = sig_len(&v, n);
            if vn == 0 {
                break;
            }
            if n <= 2 {
                // Exact word-size Euclid, folded into the cofactors in
                // single-limb matrix chunks.
                let (mut x, mut y) = (limbs_u128(&u[..n]), limbs_u128(&v[..n]));
                while y != 0 {
                    let (m, nx, ny) = euclid_mat_u128(x, y);
                    if let Some(m) = m {
                        lehmer_cofactor_apply(&mut s0, &mut s1, &m);
                        if BOTH {
                            lehmer_cofactor_apply(&mut t0, &mut t1, &m);
                        }
                        odd ^= m.odd;
                        (x, y) = (nx, ny);
                    } else {
                        let q = x / y;
                        (x, y) = (y, x - q * y);
                        let q = Nat::from_u128(q);
                        lehmer_cofactor_div(&mut s0, &mut s1, &q);
                        if BOTH {
                            lehmer_cofactor_div(&mut t0, &mut t1, &q);
                        }
                        odd = !odd;
                    }
                }
                u.clear();
                u.extend_from_slice(&Nat::from_u128(x).limbs);
                n = u.len();
                break;
            }
            let m = if vn + 1 >= n {
                let s = u[n - 1].leading_zeros();
                lehmer_mat(top128(&u, n, s), top128(&v, n, s), 0)
            } else {
                None
            };
            if let Some(m) = m {
                lehmer_apply(&mut u[..n], &mut v[..n], &m);
                lehmer_cofactor_apply(&mut s0, &mut s1, &m);
                if BOTH {
                    lehmer_cofactor_apply(&mut t0, &mut t1, &m);
                }
                odd ^= m.odd;
                n = sig_len(&u, n);
            } else {
                let (nn, q) = lehmer_div_step(&mut u, &mut v, n, vn);
                n = nn;
                lehmer_cofactor_div(&mut s0, &mut s1, &q);
                if BOTH {
                    lehmer_cofactor_div(&mut t0, &mut t1, &q);
                }
                odd = !odd;
            }
        }
        u.truncate(n);
        let signed = |mag: Vec<Limb>, neg: bool| {
            let mut x = Nat { limbs: mag };
            x.normalize();
            Int::from_sign_magnitude(if neg { Sign::Negative } else { Sign::Positive }, x)
        };
        let x = signed(s0, odd);
        let y = if BOTH { signed(t0, !odd) } else { Int::ZERO };
        (Nat { limbs: u }, x, y)
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
        // Burnikel–Ziegler only pays off when both the divisor and the
        // quotient are long: schoolbook costs `O(n·m)` for an `m`-limb
        // quotient, which beats any recursion when `m` is small.
        let quot_limbs = self.limbs.len() - rhs.limbs.len();
        if newton_div_worthwhile(rhs.limbs.len(), quot_limbs) {
            return Some(newton_div_rem(self, rhs));
        }
        if rhs.limbs.len() >= BZ_THRESHOLD && quot_limbs >= BZ_THRESHOLD {
            return Some(bz_div_rem(self, rhs));
        }
        Some(self.div_rem_knuth(rhs))
    }

    /// Knuth Algorithm D: schoolbook long division in base `2^64` (the inner
    /// loop is [`div_schoolbook_normalized`]). Precondition: `rhs` has ≥ 2
    /// limbs and `self > rhs`.
    fn div_rem_knuth(&self, rhs: &Nat) -> (Nat, Nat) {
        let n = rhs.limbs.len();
        let m = self.limbs.len() - n;

        // Normalize so the divisor's top limb has its high bit set; the
        // dividend gets one extra top limb to catch the shifted-out bits.
        let shift = rhs.limbs[n - 1].leading_zeros();
        let mut d = alloc::vec![0 as Limb; n];
        shl_limbs_into(&rhs.limbs, shift, &mut d);
        let mut u = alloc::vec![0 as Limb; self.limbs.len() + 1];
        shl_limbs_into(&self.limbs, shift, &mut u);
        let mut q = alloc::vec![0 as Limb; m + 1];
        div_schoolbook_normalized(&mut u, &d, &mut q);

        let mut quotient = Nat { limbs: q };
        quotient.normalize();
        // Denormalize the remainder (the low n limbs of u), undoing the shift;
        // `u`'s buffer is reused for it.
        u.truncate(n);
        shr_limbs_in_place(&mut u, shift);
        let mut remainder = Nat { limbs: u };
        remainder.normalize();
        (quotient, remainder)
    }

    /// Divides by a single-limb value, returning `(quotient, remainder)`.
    ///
    /// The divisor must be non-zero. Uses a precomputed Möller–Granlund
    /// reciprocal so the per-limb step is two multiplications instead of a
    /// (libcall) 128-by-64-bit hardware division.
    pub(crate) fn divmod_small(&self, d: Limb) -> (Nat, Limb) {
        debug_assert!(d != 0, "divmod_small by zero");
        let n = self.limbs.len();
        if n == 0 {
            return (Nat::zero(), 0);
        }
        // Normalize the divisor (top bit set); the dividend is scaled by the
        // same 2^s on the fly, which scales the remainder and leaves the
        // quotient unchanged.
        let s = d.leading_zeros();
        let dn = d << s;
        let v = recip_2by1(dn);
        let mut q = alloc::vec![0 as Limb; n];
        // Running remainder of the scaled dividend; the initial value is the
        // top limb's overflow bits (< 2^s ≤ 2^63 ≤ dn).
        let mut rem: Limb = if s == 0 {
            0
        } else {
            self.limbs[n - 1] >> (LIMB_BITS - s)
        };
        for i in (0..n).rev() {
            let lo = if s == 0 || i == 0 {
                0
            } else {
                self.limbs[i - 1] >> (LIMB_BITS - s)
            };
            let cur = (self.limbs[i] << s) | lo;
            let (qi, r) = div_2by1(rem, cur, dn, v);
            q[i] = qi;
            rem = r;
        }
        let mut nq = Nat { limbs: q };
        nq.normalize();
        (nq, rem >> s)
    }

    /// Single-limb remainder `self mod d` (`d != 0`) without materializing the
    /// quotient. A Horner pass over the limbs using the host's 128-by-64 divide;
    /// cheaper than [`Nat::divmod_small`] (no quotient allocation) and used by
    /// the small-prime trial-division prefilter, where only divisibility matters.
    #[inline]
    fn rem_small(&self, d: Limb) -> Limb {
        debug_assert!(d != 0, "rem_small by zero");
        let mut rem: Limb = 0;
        for &limb in self.limbs.iter().rev() {
            rem = ((((rem as u128) << LIMB_BITS) | limb as u128) % d as u128) as Limb;
        }
        rem
    }

    /// Returns the value as a `u64` if it fits in a single limb, else `None`.
    /// A cheap, allocation-free way to compare a `Nat` against a small constant.
    #[inline]
    fn as_u64(&self) -> Option<u64> {
        match self.limbs.len() {
            0 => Some(0),
            1 => Some(self.limbs[0]),
            _ => None,
        }
    }
}

/// Odd primes below 1000, used by the Baillie–PSW trial-division prefilter to
/// reject the overwhelming majority of composites with a handful of single-limb
/// remainders instead of a full Miller–Rabin + strong-Lucas test. The list is
/// grouped on the fly into blocks whose product fits in a `u64`, so one
/// big-integer remainder per block serves all its primes (since `p | M` implies
/// `n mod p == (n mod M) mod p`). The 1000 bound was chosen empirically (see the
/// `measure_prime_prefilter` bench); it clears ~84% of random odd composites,
/// within ~6 points of the asymptotic ceiling, while a 10× larger bound costs
/// far more per survivor for only a few points more.
static SMALL_PRIMES: &[u32] = &[
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547,
    557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797,
    809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929,
    937, 941, 947, 953, 967, 971, 977, 983, 991, 997,
];

/// Trial-division prefilter for odd `n > 3`: probes divisibility by every prime
/// in [`SMALL_PRIMES`]. Returns `Some(verdict)` when some small prime `p`
/// divides `n` — `Some(true)` if `n == p` (so `n` is that prime), `Some(false)`
/// otherwise (a proper factor, so composite). Returns `None` when no small prime
/// divides `n`, in which case the caller must run the full test.
///
/// Primes are consumed in greedy blocks whose product stays within a `u64`, so
/// only one big-integer remainder is taken per block.
fn small_prime_verdict(n: &Nat) -> Option<bool> {
    let mut i = 0;
    while i < SMALL_PRIMES.len() {
        // Grow a block product `m` (starts at 1) until the next prime would
        // overflow a limb.
        let start = i;
        let mut m: Limb = 1;
        while i < SMALL_PRIMES.len() {
            match m.checked_mul(SMALL_PRIMES[i] as Limb) {
                Some(v) => {
                    m = v;
                    i += 1;
                }
                None => break,
            }
        }
        // n mod m; then n mod p = (n mod m) mod p for each p in the block.
        let r = n.rem_small(m);
        for &p in &SMALL_PRIMES[start..i] {
            if r.is_multiple_of(p as Limb) {
                return Some(n.as_u64() == Some(p as u64));
            }
        }
    }
    None
}

/// Möller–Granlund reciprocal of a normalized (top bit set) divisor:
/// `v = ⌊(2^128 − 1)/d⌋ − 2^64`. The one hardware 128-bit division here is
/// amortized over every limb of the dividend.
#[inline]
fn recip_2by1(d: Limb) -> Limb {
    debug_assert!(
        d >> (LIMB_BITS - 1) == 1,
        "reciprocal needs a normalized divisor"
    );
    // The quotient is in [2^64, 2^65), so truncating to u64 subtracts 2^64.
    (u128::MAX / d as u128) as Limb
}

/// Divides `hi·2^64 + lo` by the normalized divisor `d` given its reciprocal
/// `v` (see [`recip_2by1`]), returning `(quotient, remainder)`. Requires
/// `hi < d`, so the quotient fits one limb (Möller–Granlund Algorithm 4).
#[inline]
fn div_2by1(hi: Limb, lo: Limb, d: Limb, v: Limb) -> (Limb, Limb) {
    debug_assert!(hi < d, "2-by-1 quotient must fit a limb");
    // Candidate quotient q1 = high limb of v·hi + (hi, lo), plus one; it is
    // never more than one too large after the first correction below.
    let q = (v as u128) * (hi as u128) + (((hi as u128) << LIMB_BITS) | lo as u128);
    let mut q1 = ((q >> LIMB_BITS) as Limb).wrapping_add(1);
    let q0 = q as Limb;
    let mut r = lo.wrapping_sub(q1.wrapping_mul(d));
    if r > q0 {
        q1 = q1.wrapping_sub(1);
        r = r.wrapping_add(d);
    }
    if r >= d {
        q1 += 1;
        r -= d;
    }
    (q1, r)
}

/// Möller–Granlund 3-by-2 reciprocal of the normalized two-limb divisor
/// `(d1, d0)` (`d1`'s top bit set): `v = ⌊(β³ − 1)/(d1·β + d0)⌋ − β` with
/// `β = 2^64` (their Algorithm 6, refining the 2-by-1 reciprocal of `d1`).
#[inline]
fn recip_3by2(d1: Limb, d0: Limb) -> Limb {
    let mut v = recip_2by1(d1);
    let mut p = d1.wrapping_mul(v).wrapping_add(d0);
    if p < d0 {
        v = v.wrapping_sub(1);
        if p >= d1 {
            v = v.wrapping_sub(1);
            p = p.wrapping_sub(d1);
        }
        p = p.wrapping_sub(d1);
    }
    let t = d0 as u128 * v as u128;
    let (t1, t0) = ((t >> LIMB_BITS) as Limb, t as Limb);
    p = p.wrapping_add(t1);
    if p < t1 {
        v = v.wrapping_sub(1);
        if p > d1 || (p == d1 && t0 >= d0) {
            v = v.wrapping_sub(1);
        }
    }
    v
}

/// Divides the three-limb `(u2, u1, u0)` by the normalized `(d1, d0)` given
/// its [`recip_3by2`] reciprocal `v`, returning the one-limb quotient and the
/// two-limb remainder `(r1, r0)` packed in a `u128`. Requires
/// `(u2, u1) < (d1, d0)` (Möller–Granlund Algorithm 5): two multiplications
/// and no hardware division, with a final correction that almost never runs.
#[inline]
fn div_3by2(u2: Limb, u1: Limb, u0: Limb, d1: Limb, d0: Limb, v: Limb) -> (Limb, u128) {
    let d = ((d1 as u128) << LIMB_BITS) | d0 as u128;
    let q = (v as u128) * (u2 as u128) + (((u2 as u128) << LIMB_BITS) | u1 as u128);
    let mut q1 = (q >> LIMB_BITS) as Limb;
    let q0 = q as Limb;
    let r1 = u1.wrapping_sub(q1.wrapping_mul(d1));
    let mut r = (((r1 as u128) << LIMB_BITS) | u0 as u128)
        .wrapping_sub(d0 as u128 * q1 as u128)
        .wrapping_sub(d);
    q1 = q1.wrapping_add(1);
    if (r >> LIMB_BITS) as Limb >= q0 {
        q1 = q1.wrapping_sub(1);
        r = r.wrapping_add(d);
    }
    if r >= d {
        q1 = q1.wrapping_add(1);
        r -= d;
    }
    (q1, r)
}

/// `r -= q·a` over `r.len() == a.len()` limbs, returning the borrow limb out
/// of the top (the high part of the product plus the subtraction borrow).
#[inline]
fn submul_1(r: &mut [Limb], a: &[Limb], q: Limb) -> Limb {
    debug_assert_eq!(r.len(), a.len());
    use crate::limb::DLimb;
    // r − x = !(!r + x): with w = aᵢ·q + !rᵢ (< 2^128, independent of the
    // carry), adding the running carry last gives !(new rᵢ) in the low limb
    // and the borrow-out in the high part — the same short loop-carried chain
    // as the addmul kernels, instead of multiply → add → subtract → add.
    let mut c: Limb = 0;
    let mut step = |ri: &mut Limb, ai: Limb| {
        let w = ai as DLimb * q as DLimb + (!*ri) as DLimb;
        let (s, k) = (w as Limb).overflowing_add(c);
        c = (w >> LIMB_BITS) as Limb + Limb::from(k);
        *ri = !s;
    };
    let (r2, r_tail) = r.as_chunks_mut::<2>();
    let (a2, a_tail) = a.as_chunks::<2>();
    for (x, y) in r2.iter_mut().zip(a2) {
        let [x0, x1] = x;
        step(x0, y[0]);
        step(x1, y[1]);
    }
    if let (Some(x), Some(&y)) = (r_tail.first_mut(), a_tail.first()) {
        step(x, y);
    }
    c
}

/// `r += a` over `r.len() == a.len()` limbs, returning the carry out.
#[inline]
fn add_n_in_place(r: &mut [Limb], a: &[Limb]) -> Limb {
    debug_assert_eq!(r.len(), a.len());
    let mut carry = 0;
    for (ri, &ai) in r.iter_mut().zip(a) {
        let (s, c) = adc(*ri, ai, carry);
        *ri = s;
        carry = c;
    }
    carry
}

/// Writes `src << shift` (`shift < 64`) into the zeroed `dst`; a limb shifted
/// out of the top lands in `dst[src.len()]` when `dst` is longer than `src`.
fn shl_limbs_into(src: &[Limb], shift: u32, dst: &mut [Limb]) {
    debug_assert!(shift < LIMB_BITS && dst.len() >= src.len());
    if shift == 0 {
        dst[..src.len()].copy_from_slice(src);
        return;
    }
    let mut carry: Limb = 0;
    for (o, &x) in dst.iter_mut().zip(src) {
        *o = (x << shift) | carry;
        carry = x >> (LIMB_BITS - shift);
    }
    if dst.len() > src.len() {
        dst[src.len()] = carry;
    } else {
        debug_assert_eq!(carry, 0, "shl_limbs_into dropped bits");
    }
}

/// Shifts the limb slice `x` right by `shift < 64` bits in place.
fn shr_limbs_in_place(x: &mut [Limb], shift: u32) {
    debug_assert!(shift < LIMB_BITS);
    if shift == 0 || x.is_empty() {
        return;
    }
    for i in 0..x.len() - 1 {
        x[i] = (x[i] >> shift) | (x[i + 1] << (LIMB_BITS - shift));
    }
    let last = x.len() - 1;
    x[last] >>= shift;
}

/// Schoolbook division core (Knuth Algorithm D with the 3-by-2 quotient
/// estimate of Möller–Granlund, "Improved division by invariant integers",
/// §5–6): divides `u` in place by the normalized (top bit set) divisor `d`
/// of `n ≥ 2` limbs, writing the `u.len() − n` quotient limbs to `q` and
/// leaving the remainder in `u[..n]` (the limbs above are zeroed).
///
/// Precondition: the top `n` limbs of `u` are `< d`, so every quotient limb
/// fits. Each limb of the quotient comes from one 3-by-2 reciprocal division
/// of the window's top three limbs by `d`'s top two — exact except for a
/// rare one-too-large case, caught by the borrow out of the fused
/// multiply-subtract over the remaining `n − 2` limbs and fixed by adding `d`
/// back once.
fn div_schoolbook_normalized(u: &mut [Limb], d: &[Limb], q: &mut [Limb]) {
    let n = d.len();
    debug_assert!(n >= 2 && d[n - 1] >> (LIMB_BITS - 1) == 1);
    debug_assert_eq!(q.len() + n, u.len());
    let (d1, d0) = (d[n - 1], d[n - 2]);
    let v = recip_3by2(d1, d0);
    let dlo = &d[..n - 2];
    for j in (0..q.len()).rev() {
        let w = &mut u[j..j + n + 1];
        let (n2, n1, n0) = (w[n], w[n - 1], w[n - 2]);
        let qj = if n2 == d1 && n1 == d0 {
            // The 3-by-2 quotient would be β; since the window is < β·d the
            // true quotient limb is exactly β − 1, and the subtraction leaves
            // the top limb zero.
            let borrow = submul_1(&mut w[..n], d, Limb::MAX);
            w[n] = n2.wrapping_sub(borrow);
            debug_assert_eq!(w[n], 0);
            Limb::MAX
        } else {
            let (mut qh, r) = div_3by2(n2, n1, n0, d1, d0, v);
            let cy = submul_1(&mut w[..n - 2], dlo, qh);
            // Fold the borrow into the 3-by-2 remainder's two limbs.
            let (r0, b0) = (r as Limb).overflowing_sub(cy);
            let (r1, b1) = ((r >> LIMB_BITS) as Limb).overflowing_sub(b0 as Limb);
            w[n - 2] = r0;
            w[n - 1] = r1;
            w[n] = 0;
            if b1 {
                // qh was one too large: add the divisor back (the carry out
                // cancels the borrow).
                qh = qh.wrapping_sub(1);
                add_n_in_place(&mut w[..n], d);
            }
            qh
        };
        q[j] = qj;
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

    /// Builds a natural from an owned vector of little-endian limbs, taking it
    /// over without copying (any trailing zeros are stripped).
    pub fn from_limb_vec(limbs: Vec<Limb>) -> Nat {
        let mut n = Nat { limbs };
        n.normalize();
        n
    }

    /// Consumes the natural and returns its little-endian limbs without
    /// copying (normalized: empty for zero, non-zero top limb otherwise).
    pub fn into_limbs(self) -> Vec<Limb> {
        self.limbs
    }

    /// Returns `self + v`.
    pub fn add_u64(&self, v: u64) -> Nat {
        let mut out = Vec::with_capacity(self.limbs.len() + 1);
        out.extend_from_slice(&self.limbs);
        let mut n = Nat { limbs: out };
        n.add_assign_u64(v);
        n
    }

    /// Returns `self − v`, or `None` if `v > self`.
    pub fn checked_sub_u64(&self, v: u64) -> Option<Nat> {
        if self.limbs.len() <= 1 {
            let x = self.limbs.first().copied().unwrap_or(0);
            return x.checked_sub(v).map(Nat::from_u64);
        }
        let mut n = self.clone();
        n.sub_assign_u64_unchecked(v);
        Some(n)
    }

    /// Returns `self · v`.
    pub fn mul_u64(&self, v: u64) -> Nat {
        if v == 0 || self.is_zero() {
            return Nat::zero();
        }
        let mut out = Vec::with_capacity(self.limbs.len() + 1);
        out.extend_from_slice(&self.limbs);
        let mut n = Nat { limbs: out };
        n.mul_assign_u64(v);
        n
    }

    /// `self += v` in place.
    pub(crate) fn add_assign_u64(&mut self, v: u64) {
        let mut carry = v;
        for l in self.limbs.iter_mut() {
            if carry == 0 {
                return;
            }
            let (s, c) = l.overflowing_add(carry);
            *l = s;
            carry = c as Limb;
        }
        if carry != 0 {
            self.limbs.push(carry);
        }
    }

    /// `self −= v` in place; the caller guarantees `v ≤ self`.
    pub(crate) fn sub_assign_u64_unchecked(&mut self, v: u64) {
        let mut borrow = v;
        for l in self.limbs.iter_mut() {
            if borrow == 0 {
                break;
            }
            let (d, b) = l.overflowing_sub(borrow);
            *l = d;
            borrow = b as Limb;
        }
        debug_assert_eq!(borrow, 0, "sub_assign_u64_unchecked underflow");
        self.normalize();
    }

    /// `self ·= v` in place.
    pub(crate) fn mul_assign_u64(&mut self, v: u64) {
        if v == 0 {
            self.limbs.clear();
            return;
        }
        let mut carry = 0;
        for l in self.limbs.iter_mut() {
            let (lo, hi) = mac(0, *l, v, carry);
            *l = lo;
            carry = hi;
        }
        if carry != 0 {
            self.limbs.push(carry);
        }
    }

    /// `self += rhs` in place, reusing `self`'s buffer.
    pub(crate) fn add_assign_limbs(&mut self, rhs: &[Limb]) {
        if rhs.len() > self.limbs.len() {
            self.limbs.resize(rhs.len(), 0);
        }
        let mut carry = 0;
        for (a, &b) in self.limbs.iter_mut().zip(rhs) {
            let (s, c) = adc(*a, b, carry);
            *a = s;
            carry = c;
        }
        let mut i = rhs.len();
        while carry != 0 {
            if i == self.limbs.len() {
                self.limbs.push(carry);
                break;
            }
            let (s, c) = adc(self.limbs[i], 0, carry);
            self.limbs[i] = s;
            carry = c;
            i += 1;
        }
    }

    /// `self −= rhs` in place; the caller guarantees `rhs ≤ self`.
    pub(crate) fn sub_assign_limbs(&mut self, rhs: &[Limb]) {
        let mut borrow = 0;
        for (a, &b) in self.limbs.iter_mut().zip(rhs) {
            let (d, bb) = sbb(*a, b, borrow);
            *a = d;
            borrow = bb;
        }
        let mut i = rhs.len();
        while borrow != 0 {
            let (d, bb) = sbb(self.limbs[i], 0, borrow);
            self.limbs[i] = d;
            borrow = bb;
            i += 1;
        }
        self.normalize();
    }

    /// `self = rhs − self` in place; the caller guarantees `self ≤ rhs`.
    pub(crate) fn rsub_assign_limbs(&mut self, rhs: &[Limb]) {
        debug_assert!(self.limbs.len() <= rhs.len());
        self.limbs.resize(rhs.len(), 0);
        let mut borrow = 0;
        for (a, &b) in self.limbs.iter_mut().zip(rhs) {
            let (d, bb) = sbb(b, *a, borrow);
            *a = d;
            borrow = bb;
        }
        debug_assert_eq!(borrow, 0, "rsub_assign_limbs underflow");
        self.normalize();
    }

    /// Returns the nearest `f64` (ties to even), or `+∞` when the value rounds
    /// past `f64::MAX`.
    pub fn to_f64(&self) -> f64 {
        limbs_to_f64(&self.limbs, true)
    }

    /// Returns the `f64` obtained by truncating toward zero (GMP's
    /// `mpz_get_d`), or `+∞` when the value is `≥ 2^1024`.
    pub fn to_f64_trunc(&self) -> f64 {
        limbs_to_f64(&self.limbs, false)
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
    ///
    /// Bases `0`, `1` and `2^k` are answered directly, so their exponent may be
    /// any `u64` (as long as the result fits in memory).
    pub fn pow(&self, exp: u64) -> Nat {
        if exp == 0 {
            return Nat::one();
        }
        if self.is_zero() || self.is_one() {
            return self.clone();
        }
        let tz = self.trailing_zeros();
        if tz + 1 == self.bit_len() {
            let shift = tz.checked_mul(exp).expect("pow: result too large");
            return Nat::one().shl(shift);
        }
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

    /// Returns `10^k`, computed as `5^k · 2^k`.
    pub fn pow10(k: u64) -> Nat {
        Nat::from_u64(5).pow(k).shl(k)
    }

    /// Returns `self · 10^k`.
    pub fn mul_pow10(&self, k: u64) -> Nat {
        if self.is_zero() || k == 0 {
            return self.clone();
        }
        self.mul(&Nat::from_u64(5).pow(k)).shl(k)
    }

    /// Returns the floor of the square root, `⌊√self⌋`.
    ///
    /// Zimmermann's "Karatsuba square root" (the internal `sqrt_rem`): the divisions it
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
        // Newton's method for the integer kth root, converging quadratically from
        // an overestimate: x ← ⌊((k−1)·x + ⌊N/x^(k−1)⌋) / k⌋, which decreases
        // monotonically toward ⌊N^(1/k)⌋. Start above the root at 2^⌈bits/k⌉, then
        // fix any ±1 slop from the integer divisions with an exact adjustment.
        let km1 = Nat::from_u64((k - 1) as u64);
        let kk = Nat::from_u64(k as u64);
        let mut x = Nat::one().shl(self.bit_len().div_ceil(k as u64));
        loop {
            let div = self.div_rem(&x.pow(u64::from(k - 1))).expect("x >= 1").0;
            let next = x.mul(&km1).add(&div).div_rem(&kk).expect("k >= 1").0;
            if next.cmp_ref(&x) != Ordering::Less {
                break;
            }
            x = next;
        }
        // Exact adjustment: land on the largest x with x^k ≤ N.
        while x.pow(u64::from(k)).cmp_ref(self) == Ordering::Greater {
            x = x.checked_sub(&Nat::one()).expect("root is positive");
        }
        loop {
            let up = x.add(&Nat::one());
            if up.pow(u64::from(k)).cmp_ref(self) == Ordering::Greater {
                break;
            }
            x = up;
        }
        x
    }

    /// Writes the magnitude in the given `radix` (2–62) to `out`.
    ///
    /// Digits follow GMP's convention: `0-9a-z` for radixes up to 36, and
    /// `0-9A-Za-z` (upper case first) for radixes 37–62.
    pub fn write_radix(&self, out: &mut impl fmt::Write, radix: u32) -> fmt::Result {
        assert!((2..=62).contains(&radix), "radix must be in 2..=62");
        if self.is_zero() {
            return out.write_str("0");
        }
        out.write_str(&self.to_radix_string(radix))
    }

    /// Appends the base-`radix` (2–62) ASCII digits to `out` (`"0"` for zero),
    /// without going through `String`/`fmt`. Same alphabet as
    /// [`Nat::write_radix`].
    pub fn write_ascii_radix(&self, out: &mut Vec<u8>, radix: u32) {
        assert!((2..=62).contains(&radix), "radix must be in 2..=62");
        if self.is_zero() {
            out.push(b'0');
        } else {
            self.append_radix_digits(radix, out);
        }
    }

    /// Returns the base-`radix` (2–62) ASCII digits (`"0"` for zero). Same
    /// alphabet as [`Nat::write_radix`].
    pub fn to_ascii_radix(&self, radix: u32) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_ascii_radix(&mut out, radix);
        out
    }

    /// Parses ASCII digits in the given `radix` (2–62) from a byte slice,
    /// without going through `&str`. Same alphabet as [`Nat::write_radix`]
    /// (letters case-insensitive up to radix 36); no sign, whitespace or
    /// separators, and an empty slice is an error.
    pub fn from_ascii_radix(digits: &[u8], radix: u32) -> Result<Nat> {
        parse_radix_bytes(digits, radix)
    }

    /// Returns the minimal (no leading zeros) base-`radix` digit string; see
    /// [`Nat::append_radix_digits`].
    fn to_radix_string(&self, radix: u32) -> String {
        let mut out = Vec::new();
        self.append_radix_digits(radix, &mut out);
        String::from_utf8(out).unwrap_or_default()
    }

    /// Appends the minimal (no leading zeros) base-`radix` digits, using
    /// divide-and-conquer: split off a divisor `radix^len ≈ √self`, recurse on
    /// the quotient and remainder, and zero-pad the remainder to `len` digits.
    /// With sub-quadratic multiplication/division this is `O(M(n)·log n)`.
    /// Every level appends straight into one output buffer.
    fn append_radix_digits(&self, radix: u32, out: &mut Vec<u8>) {
        // ~1 digit per log2(radix) bits; a cheap estimate is enough for the
        // initial capacity (19.3 decimal digits per limb).
        out.reserve(self.limbs.len() * 20 + 1);
        // Base case: a few limbs go straight through single-limb division.
        if self.limbs.len() <= RADIX_RECURSION_LIMBS {
            simple_radix_string(&self.limbs, radix, 0, out);
        } else {
            // Build the ladder `[radix^1, radix^2, radix^4, …]` once (largest
            // entry ≤ self), then share it across the whole recursion instead
            // of re-squaring it at every node.
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
            to_radix_recursive(self, &powers, &[], radix, 0, out);
        }
    }
}

/// Recursive base-`radix` conversion sharing a precomputed power ladder
/// (`powers[k] == radix^(2^k)`, ascending): appends the digits of `v` to `out`,
/// left-padded with zeros to at least `width` digits (`0` = minimal).
pub(crate) fn to_radix_recursive(
    v: &Nat,
    powers: &[Nat],
    inverses: &[Option<PreInv>],
    radix: u32,
    width: usize,
    out: &mut Vec<u8>,
) {
    if v.limbs.len() <= RADIX_RECURSION_LIMBS {
        simple_radix_string(&v.limbs, radix, width, out);
        return;
    }
    // Split by the largest ladder entry `p = radix^(2^k) ≤ v`.
    let k = powers
        .iter()
        .rposition(|p| p.cmp_ref(v) != Ordering::Greater)
        .expect("v is large, so radix <= v");
    let len = 1usize << k;
    let (q, r) = match inverses.get(k) {
        Some(Some(inv)) => inv.div_rem(v),
        _ => v.div_rem(&powers[k]).expect("p is non-zero"),
    };
    // The high part fills whatever the padding leaves above the low `len`
    // digits; the low part is zero-padded to exactly `len`.
    to_radix_recursive(&q, powers, inverses, radix, width.saturating_sub(len), out);
    to_radix_recursive(&r, powers, inverses, radix, len, out);
}

/// Number of limbs at or below which radix conversion uses the simple
/// single-limb-division loop rather than recursing.
pub(crate) const RADIX_RECURSION_LIMBS: usize = 16;

/// Appends the base-`radix` digits of the little-endian `limbs` (at most
/// [`RADIX_RECURSION_LIMBS`]) to `out`, left-padded with zeros to at least
/// `width` digits (a zero value with `width == 0` appends nothing). For small
/// values / the recursion base case.
///
/// Each pass divides the whole number, in place on the stack, by
/// `radix^chunk` (the largest power fitting a `u64`) with a precomputed
/// reciprocal, peeling a whole limb's worth of digits that are then split out
/// with machine-word arithmetic.
pub(crate) fn simple_radix_string(limbs: &[Limb], radix: u32, width: usize, out: &mut Vec<u8>) {
    debug_assert!(limbs.len() <= RADIX_RECURSION_LIMBS);
    // Largest `chunk` with `radix^chunk` fitting a u64, and that power itself.
    let (chunk, base) = {
        let (mut d, mut p) = (0usize, 1u64);
        while let Some(next) = p.checked_mul(radix as u64) {
            p = next;
            d += 1;
        }
        (d, p)
    };
    // The base exceeds 2^32 for every radix, so each limb yields < 2 chunks.
    let mut x = [0 as Limb; RADIX_RECURSION_LIMBS];
    let mut len = limbs.len();
    x[..len].copy_from_slice(limbs);
    while len > 0 && x[len - 1] == 0 {
        len -= 1;
    }
    let mut chunks = [0 as Limb; 2 * RADIX_RECURSION_LIMBS + 1];
    let mut nchunks = 0;
    if len == 1 && x[0] < base {
        chunks[0] = x[0];
        nchunks = 1;
    } else if len > 0 {
        let s = base.leading_zeros();
        let dn = base << s;
        let v = recip_2by1(dn);
        while len > 0 {
            chunks[nchunks] = divrem_1_normalized_in_place(&mut x[..len], dn, s, v);
            nchunks += 1;
            if x[len - 1] == 0 {
                len -= 1;
            }
        }
    }
    // Digits of the leading chunk, then `chunk` per remaining chunk.
    let top = if nchunks == 0 { 0 } else { chunks[nchunks - 1] };
    let top_digits = {
        let (mut t, mut c) = (top, 0usize);
        while t != 0 {
            t /= radix as u64;
            c += 1;
        }
        c
    };
    let total = top_digits + nchunks.saturating_sub(1) * chunk;
    out.resize(out.len() + width.saturating_sub(total), b'0');
    let at = out.len();
    out.resize(at + total, 0);
    let dst = &mut out[at..];
    // Fill each chunk's digits right to left; the specialized arms let the
    // compiler turn the per-digit division into a multiplication.
    let fill = |dst: &mut [u8], r: Limb| match radix {
        10 => fill_digits::<10>(dst, r),
        16 => fill_digits::<16>(dst, r),
        _ => fill_digits_dyn(dst, r, radix as u64),
    };
    fill(&mut dst[..top_digits], top);
    for (i, slot) in dst[top_digits..].chunks_exact_mut(chunk).enumerate() {
        fill(slot, chunks[nchunks - 2 - i]);
    }
}

/// Writes the low `dst.len()` base-`R` digits of `r` into `dst`, most
/// significant first.
#[inline]
fn fill_digits<const R: u64>(dst: &mut [u8], mut r: Limb) {
    for b in dst.iter_mut().rev() {
        *b = digit_char((r % R) as u32, R as u32);
        r /= R;
    }
}

/// [`fill_digits`] for a radix only known at run time.
fn fill_digits_dyn(dst: &mut [u8], mut r: Limb, radix: u64) {
    for b in dst.iter_mut().rev() {
        *b = digit_char((r % radix) as u32, radix as u32);
        r /= radix;
    }
}

/// Divides the limbs `x` in place by `d = dn >> s` (`dn` normalized, `v` its
/// [`recip_2by1`] reciprocal), returning the remainder. The dividend is scaled
/// by `2^s` on the fly, which scales the remainder and leaves the quotient
/// unchanged.
fn divrem_1_normalized_in_place(x: &mut [Limb], dn: Limb, s: u32, v: Limb) -> Limb {
    let n = x.len();
    let mut rem: Limb = if s == 0 {
        0
    } else {
        x[n - 1] >> (LIMB_BITS - s)
    };
    for i in (0..n).rev() {
        let lo = if s == 0 || i == 0 {
            0
        } else {
            x[i - 1] >> (LIMB_BITS - s)
        };
        let cur = (x[i] << s) | lo;
        let (qi, r) = div_2by1(rem, cur, dn, v);
        x[i] = qi;
        rem = r;
    }
    rem >> s
}

/// Converts normalized little-endian limbs to `f64`, rounding to nearest-even
/// (`round`) or truncating toward zero; `+∞` past the `f64` range. Only the
/// top 64 bits plus a sticky bit are inspected, so this is `O(1)` past the
/// sticky scan.
pub(crate) fn limbs_to_f64(limbs: &[Limb], round: bool) -> f64 {
    let Some(&hi) = limbs.last() else {
        return 0.0;
    };
    let n = limbs.len() as u64 * 64 - hi.leading_zeros() as u64;
    if n <= 64 {
        // `as f64` already rounds to nearest-even; truncation drops the bits
        // below the 53-bit significand first.
        return if round || n <= 53 {
            hi as f64
        } else {
            ((hi >> (n - 53)) << (n - 53)) as f64
        };
    }
    // The top 64 bits, and whether anything below them is non-zero.
    let shift = n - 64;
    let (i, off) = ((shift / 64) as usize, (shift % 64) as u32);
    let mut top = limbs[i] >> off;
    if off > 0 {
        top |= limbs[i + 1] << (64 - off);
    }
    let sticky = (off > 0 && limbs[i] << (64 - off) != 0) || limbs[..i].iter().any(|&l| l != 0);
    let mut mant = top >> 11;
    let rest = top & 0x7ff;
    if round && (rest > 0x400 || (rest == 0x400 && (sticky || mant & 1 == 1))) {
        mant += 1; // may carry to 2^53, still exact
    }
    let exp = n - 53; // value ≈ mant · 2^exp, with mant ∈ [2^52, 2^53]
    if exp > 1023 - 52 {
        return f64::INFINITY;
    }
    mant as f64 * f64::from_bits((exp + 1023) << 52)
}

/// Maps a digit value `0..radix` to its ASCII character: `0-9a-z` for
/// `radix ≤ 36`, `0-9A-Za-z` above.
#[inline]
fn digit_char(d: u32, radix: u32) -> u8 {
    match d {
        0..10 => b'0' + d as u8,
        _ if radix <= 36 => b'a' + (d - 10) as u8,
        10..36 => b'A' + (d - 10) as u8,
        _ => b'a' + (d - 36) as u8,
    }
}

/// Maps an ASCII digit to its value in `radix`, the inverse of [`digit_char`]
/// except that letters are case-insensitive for `radix ≤ 36`.
#[inline]
fn digit_value(c: char, radix: u32) -> Option<u32> {
    let v = match c {
        '0'..='9' => c as u32 - '0' as u32,
        'a'..='z' if radix <= 36 => c as u32 - 'a' as u32 + 10,
        'a'..='z' => c as u32 - 'a' as u32 + 36,
        'A'..='Z' => c as u32 - 'A' as u32 + 10,
        _ => return None,
    };
    (v < radix).then_some(v)
}

/// Parses an unsigned integer in the given `radix` (2–62); see
/// [`Nat::write_radix`] for the digit alphabet.
pub(crate) fn parse_radix(s: &str, radix: u32) -> Result<Nat> {
    parse_radix_bytes(s.as_bytes(), radix)
}

/// [`parse_radix`] on ASCII bytes.
pub(crate) fn parse_radix_bytes(bytes: &[u8], radix: u32) -> Result<Nat> {
    let (level, base) = pack_radix_chunks(bytes, radix)?;
    Ok(match level.as_slice() {
        [single] => Nat::from_u64(*single),
        _ => nat_from_base_chunks(&level, base),
    })
}

/// `(chunk, B)`: the largest `chunk` with `B = radix^chunk` fitting a `u64`.
pub(crate) fn radix_chunk(radix: u32) -> (usize, u64) {
    let (mut d, mut p) = (0usize, 1u64);
    while let Some(next) = p.checked_mul(radix as u64) {
        p = next;
        d += 1;
    }
    (d, p)
}

/// Validates ASCII digits in `radix` (2–62) and packs them into base-`B`
/// chunks, least significant first; returns the chunks and `B`.
pub(crate) fn pack_radix_chunks(bytes: &[u8], radix: u32) -> Result<(Vec<Limb>, u64)> {
    if !(2..=62).contains(&radix) || bytes.is_empty() {
        return Err(Error::Parse);
    }
    let (chunk, base) = radix_chunk(radix);
    // Validate and pack `chunk` digits at a time in one pass (`rchunks` walks
    // from the end). Digits are ASCII, so any byte of a multi-byte character
    // is rejected.
    let pack = |digits: &[u8]| -> Result<u64> {
        digits.iter().try_fold(0u64, |val, &b| {
            let dg = digit_value(b as char, radix).ok_or(Error::Parse)?;
            Ok(val * radix as u64 + dg as u64)
        })
    };
    let level: Vec<Limb> = bytes.rchunks(chunk).map(pack).collect::<Result<_>>()?;
    Ok((level, base))
}

/// Number of base-`B` chunks at or below which [`nat_from_base_chunks`] uses
/// in-place Horner evaluation rather than recursing.
pub(crate) const PARSE_RECURSION_CHUNKS: usize = 512;

/// Evaluates `Σ chunks[i]·B^i` (`chunks` least-significant first, each
/// `< B`). Short inputs use Horner's rule in place on one limb buffer (one
/// fused multiply-add pass per chunk); long ones split into a low half of
/// `2^k` chunks and the rest, `high·B^(2^k) + low`, sharing a ladder of
/// `B^(2^k)` powers, which is `O(M(n)·log n)` with sub-quadratic
/// multiplication.
pub(crate) fn nat_from_base_chunks(chunks: &[Limb], base: Limb) -> Nat {
    if chunks.len() <= PARSE_RECURSION_CHUNKS {
        return horner_base_chunks(chunks, base);
    }
    // powers[k] = B^(2^k) for every 2^k < chunks.len().
    let mut powers = alloc::vec![Nat::from_u64(base)];
    while 2usize << (powers.len() - 1) < chunks.len() {
        let sq = powers.last().unwrap().square();
        powers.push(sq);
    }
    chunks_recursive(chunks, &powers, base)
}

/// Recursive step of [`nat_from_base_chunks`] (`powers[k] == B^(2^k)`).
pub(crate) fn chunks_recursive(chunks: &[Limb], powers: &[Nat], base: Limb) -> Nat {
    if chunks.len() <= PARSE_RECURSION_CHUNKS {
        return horner_base_chunks(chunks, base);
    }
    // Largest 2^k strictly below the length, so both halves are non-empty.
    let k = (usize::BITS - 1 - (chunks.len() - 1).leading_zeros()) as usize;
    let (low, high) = chunks.split_at(1 << k);
    let hi = chunks_recursive(high, powers, base);
    let lo = chunks_recursive(low, powers, base);
    hi.mul(&powers[k]).add(&lo)
}

/// Horner evaluation of `Σ chunks[i]·B^i` in one growing limb buffer.
pub(crate) fn horner_base_chunks(chunks: &[Limb], base: Limb) -> Nat {
    let mut acc: Vec<Limb> = Vec::with_capacity(chunks.len() + 1);
    for &c in chunks.iter().rev() {
        // acc = acc·B + c.
        let mut carry = c;
        for l in acc.iter_mut() {
            let (lo, hi) = mac(0, *l, base, carry);
            *l = lo;
            carry = hi;
        }
        if carry != 0 {
            acc.push(carry);
        }
    }
    let mut n = Nat { limbs: acc };
    n.normalize();
    n
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

    /// Montgomery-reduction modpow for an odd `modulus > 1`: left-to-right
    /// sliding-window exponentiation (HAC Algorithm 14.85) over fixed-length
    /// limb buffers, with a precomputed table of odd powers and no allocation
    /// per step.
    fn modpow_montgomery(&self, exp: &Nat, modulus: &Nat) -> Nat {
        let bits = exp.bit_len() as usize;
        if bits == 0 {
            return Nat::one(); // modulus > 1
        }
        let n = modulus.limbs.len();
        let m = modulus.limbs.as_slice(); // exactly n words, top non-zero, odd
        let mut ws = crate::mont::Mont::new(m);

        // R = 2^(64n); R² mod m converts a residue into Montgomery form via one
        // Montgomery multiply.
        let r = Nat::one().shl(n as u64 * LIMB_BITS as u64);
        let r2 = padded(&r.square().div_rem(modulus).expect("non-zero").1, n);
        let base = if self.cmp_ref(modulus) == Ordering::Less {
            padded(self, n)
        } else {
            padded(&self.div_rem(modulus).expect("non-zero").1, n)
        };
        let mut g = alloc::vec![0 as Limb; n];
        ws.mul(&mut g, &base, &r2);

        // Window width minimizing table cost 2^(w−1) plus ≈ bits/(w+1)
        // window multiplies.
        let w = (1..=8usize)
            .min_by_key(|&w| (1usize << (w - 1)) + bits / (w + 1))
            .expect("non-empty range");
        // table[i] = g^(2i+1) (Montgomery form).
        let half = 1usize << (w - 1);
        let mut table = alloc::vec![0 as Limb; half * n];
        table[..n].copy_from_slice(&g);
        if half > 1 {
            let mut g2 = alloc::vec![0 as Limb; n];
            ws.sqr(&mut g2, &g);
            for i in 1..half {
                let (done, rest) = table.split_at_mut(i * n);
                ws.mul(&mut rest[..n], &done[(i - 1) * n..], &g2);
            }
        }

        let el = &exp.limbs;
        let bit = |i: usize| (el[i / 64] >> (i % 64)) & 1 == 1;
        let mut acc = alloc::vec![0 as Limb; n];
        let mut tmp = alloc::vec![0 as Limb; n];
        let mut first = true;
        let mut i = bits; // bits i.. are consumed
        while i > 0 {
            let hi = i - 1;
            if !bit(hi) {
                ws.sqr(&mut tmp, &acc);
                core::mem::swap(&mut acc, &mut tmp);
                i = hi;
                continue;
            }
            // Longest window [lo, hi] of at most w bits ending in a one.
            let mut lo = hi.saturating_sub(w - 1);
            while !bit(lo) {
                lo += 1;
            }
            let mut val = 0usize;
            for k in (lo..=hi).rev() {
                val = (val << 1) | bit(k) as usize;
            }
            let entry = &table[(val >> 1) * n..((val >> 1) + 1) * n];
            if first {
                acc.copy_from_slice(entry);
                first = false;
            } else {
                for _ in lo..=hi {
                    ws.sqr(&mut tmp, &acc);
                    core::mem::swap(&mut acc, &mut tmp);
                }
                ws.mul(&mut tmp, &acc, entry);
                core::mem::swap(&mut acc, &mut tmp);
            }
            i = lo;
        }
        ws.leave(&mut tmp, &acc);
        let mut out = Nat { limbs: tmp };
        out.normalize();
        out
    }

    /// Returns the smallest prime strictly greater than `self`, found by
    /// scanning odd candidates with the deterministic Baillie–PSW test. No RNG
    /// needed.
    pub fn next_prime(&self) -> Nat {
        let two = Nat::from_u64(2);
        if self.cmp_ref(&two) == Ordering::Less {
            return two; // next prime after 0 or 1
        }
        let mut c = self.add(&Nat::one());
        if c.is_even() {
            c = c.add(&Nat::one()); // start at an odd candidate ≥ 3
        }
        // Incremental small-prime residue sieve: residues[i] = c mod p_i, seeded
        // once from the big candidate and then advanced by +2 per step with cheap
        // single-limb arithmetic. A candidate with a small factor (residue 0) is
        // skipped without a full test; survivors still go through `is_prime_bpsw`,
        // which owns the verdict.
        let mut residues: Vec<Limb> = SMALL_PRIMES
            .iter()
            .map(|&p| c.rem_small(p as Limb))
            .collect();
        loop {
            let mut small_factor = false;
            for (i, &p) in SMALL_PRIMES.iter().enumerate() {
                if residues[i] == 0 {
                    // c is divisible by p: prime only if c == p.
                    if c.as_u64() == Some(p as u64) {
                        return c;
                    }
                    small_factor = true;
                    break;
                }
            }
            if !small_factor && c.is_prime_bpsw() {
                return c;
            }
            c = c.add(&two);
            for (i, &p) in SMALL_PRIMES.iter().enumerate() {
                let pl = p as Limb;
                let mut r = residues[i] + 2;
                if r >= pl {
                    r -= pl;
                }
                residues[i] = r;
            }
        }
    }

    /// Returns the largest prime strictly less than `self`, or `None` if there
    /// is none (`self <= 2`). Uses the deterministic Baillie–PSW test; no RNG.
    pub fn prev_prime(&self) -> Option<Nat> {
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
        // Incremental small-prime residue sieve, advanced by −2 per step (see
        // `next_prime`). Survivors are gated on the unchanged `is_prime_bpsw`.
        let mut residues: Vec<Limb> = SMALL_PRIMES
            .iter()
            .map(|&p| c.rem_small(p as Limb))
            .collect();
        loop {
            if c.cmp_ref(&two) == Ordering::Less {
                return Some(two);
            }
            let mut small_factor = false;
            for (i, &p) in SMALL_PRIMES.iter().enumerate() {
                if residues[i] == 0 {
                    if c.as_u64() == Some(p as u64) {
                        return Some(c);
                    }
                    small_factor = true;
                    break;
                }
            }
            if !small_factor && c.is_prime_bpsw() {
                return Some(c);
            }
            c = c.checked_sub(&two).unwrap_or_else(Nat::zero);
            for (i, &p) in SMALL_PRIMES.iter().enumerate() {
                let pl = p as Limb;
                // Advance residue by −2 modulo p: add p−2 then reduce.
                let mut r = residues[i] + pl - 2;
                if r >= pl {
                    r -= pl;
                }
                residues[i] = r;
            }
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
        // Small-prime trial division rejects the vast majority of composites
        // with a few single-limb remainders, before the costly isqrt guard and
        // Miller–Rabin + strong-Lucas test. It only ever *rejects* (or confirms
        // `n` is itself a small prime); survivors fall through to the full test.
        if let Some(verdict) = small_prime_verdict(self) {
            return verdict;
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
    /// The pipeline escalates by difficulty: trial division clears tiny
    /// factors, Pollard's rho splits small ones (up to ~15 digits), the
    /// quadratic sieve handles balanced semiprimes (its cost scaling with
    /// `self`) — the single-polynomial variant below ~40 digits and the
    /// self-initializing multiple-polynomial sieve (SIQS) from there into the
    /// ~50-digit range — and Lenstra's elliptic-curve method reaches medium
    /// factors of larger numbers (its cost scaling with the factor). Each split
    /// factor is confirmed prime with Baillie–PSW. Genuinely hard inputs — two
    /// large balanced factors beyond the sieve's range — remain expensive, as
    /// always.
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
        // Split whatever remains, escalating rho → ECM per composite.
        let mut stack = Vec::new();
        if !n.is_one() {
            stack.push(n);
        }
        while let Some(m) = stack.pop() {
            if m.is_prime_bpsw() {
                factors.push(m);
                continue;
            }
            let factor = split_composite(&m);
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

/// Splits the odd composite `n > 1` (already stripped of factors ≤ 4096) into a
/// non-trivial factor, escalating by expected factor size: bounded Pollard rho
/// for small factors, then Lenstra's ECM for medium ones, then an unbounded rho
/// as a guaranteed-terminating last resort for anything left.
fn split_composite(n: &Nat) -> Nat {
    // Rho is cheapest for small factors; cap its work so a hard composite
    // escalates rather than grinding.
    if let Some(f) = pollard_rho(n, Some(1 << 20)) {
        return f;
    }
    // For a balanced semiprime the quadratic sieve is faster and more reliable
    // than ECM, whose cost would depend on the (large) factor rather than on
    // `n`. The self-initializing MPQS (SIQS) is the method of choice from ~30
    // digits (~100 bits) up to the ~60-digit range: the double-large-prime
    // variation (cycle counting) and the block-Lanczos GF(2) solver together let
    // the factor base grow past what dense Gaussian elimination could handle,
    // reaching balanced ~60-digit semiprimes (measured: ~5–7 s at 55 digits,
    // ~1–2 min at 60). Below ~30 digits, and if SIQS declines to run, the
    // single-polynomial sieve handles it. Each falls through to the next strategy
    // if it fails to split `n`.
    if n.bit_len() >= 100
        && n.bit_len() <= 205
        && let Some(f) = crate::qsieve::siqs_factor(n)
    {
        return f;
    }
    if n.bit_len() <= 152
        && let Some(f) = crate::qsieve::qs_factor(n)
    {
        return f;
    }
    // ECM reaches medium factors of larger `n` (its cost scales with the
    // factor, so a big `n` with a moderate factor is cheap).
    if let Some(f) = crate::ecm::ecm_factor(n) {
        return f;
    }
    // Last resort (guaranteed to terminate, though impractically slow for a
    // large balanced semiprime — genuinely hard, as documented).
    pollard_rho(n, None).expect("unbounded rho returns a factor")
}

/// Pollard's rho: returns a non-trivial factor of the odd composite `n > 1`
/// (Floyd cycle detection over `f(x) = x² + c mod n`, retrying with larger `c`).
/// With `budget = Some(k)`, gives up after about `k` iterations across all
/// polynomials and returns `None`; `budget = None` searches until it succeeds.
fn pollard_rho(n: &Nat, budget: Option<u64>) -> Option<Nat> {
    if n.is_even() {
        return Some(Nat::from_u64(2));
    }
    let one = Nat::one();
    // Work in the Montgomery domain over the odd modulus `n`: `f(x) = x² + c`
    // becomes a Montgomery squaring plus a mod-`n` add, and the per-step
    // `gcd(x − y, n)` is invariant under the Montgomery `R` factor
    // (`gcd(R, n) = 1`), so the difference can be gcd'd directly.
    let mont = MontCtx::new(n);
    // Brent's improvement: accumulate the product of the differences `x − y`
    // over a batch of steps and take a single gcd per batch, backing off to a
    // per-step gcd only when a batch gcd lands on `n`.
    const BATCH: u32 = 128;
    let two_mont = mont.to_mont(&Nat::from_u64(2));
    let mut c = 1u64;
    let mut spent = 0u64;
    loop {
        let c_mont = mont.to_mont(&Nat::from_u64(c));
        let f = |x: &Nat| mont.add(&mont.sqr(x), &c_mont);
        let (mut x, mut y) = (two_mont.clone(), two_mont.clone());
        loop {
            // Run one batch, accumulating `∏ (x − y)` in Montgomery form. The
            // running product starts at the Montgomery form of `1`.
            let (xs, ys) = (x.clone(), y.clone());
            let mut prod = mont.one().clone();
            let mut steps = 0u32;
            let mut budget_hit = false;
            while steps < BATCH {
                x = f(&x);
                y = f(&f(&y));
                steps += 1;
                spent += 1;
                let diff = mont.sub(&x, &y);
                if diff.is_zero() {
                    // `x` and `y` collided: this polynomial has cycled without a
                    // factor. Force the batch gcd to `n` so the replay below
                    // detects the closed cycle and moves to a new polynomial.
                    prod = Nat::zero();
                    break;
                }
                prod = mont.mul(&prod, &diff);
                if budget.is_some_and(|limit| spent >= limit) {
                    budget_hit = true;
                    break;
                }
            }
            let g = if prod.is_zero() {
                n.clone()
            } else {
                prod.gcd(n)
            };
            if g != one && g != *n {
                return Some(g);
            }
            if g == *n {
                // A factor (or a closed cycle) appeared inside the batch: replay
                // it step by step, taking one gcd per step, to isolate it.
                let (mut xr, mut yr) = (xs, ys);
                for _ in 0..steps {
                    xr = f(&xr);
                    yr = f(&f(&yr));
                    let diff = mont.sub(&xr, &yr);
                    let d = if diff.is_zero() {
                        n.clone()
                    } else {
                        diff.gcd(n)
                    };
                    if d != one && d != *n {
                        return Some(d);
                    }
                    if d == *n {
                        break; // closed cycle: fall through to a new polynomial
                    }
                }
                if budget_hit {
                    return None;
                }
                break; // try a different polynomial
            }
            if budget_hit {
                return None;
            }
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
        self.add_assign_limbs(&rhs.limbs);
    }
}

impl core::ops::MulAssign for Nat {
    #[inline]
    fn mul_assign(&mut self, rhs: Nat) {
        *self = Nat::mul(self, &rhs);
    }
}

impl core::ops::AddAssign<&Nat> for Nat {
    /// In-place addition, reusing `self`'s buffer.
    #[inline]
    fn add_assign(&mut self, rhs: &Nat) {
        self.add_assign_limbs(&rhs.limbs);
    }
}

impl core::ops::AddAssign<u64> for Nat {
    #[inline]
    fn add_assign(&mut self, rhs: u64) {
        self.add_assign_u64(rhs);
    }
}

impl core::ops::MulAssign<u64> for Nat {
    #[inline]
    fn mul_assign(&mut self, rhs: u64) {
        self.mul_assign_u64(rhs);
    }
}

impl core::ops::MulAssign<&Nat> for Nat {
    #[inline]
    fn mul_assign(&mut self, rhs: &Nat) {
        *self = Nat::mul(self, rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    #[test]
    fn nth_root_floor_newton_is_exact() {
        // Definition check: x = ⌊N^(1/k)⌋ iff x^k ≤ N < (x+1)^k.
        let check = |n: &Nat, k: u32| {
            let x = n.nth_root_floor(k);
            assert!(
                x.pow(u64::from(k)).cmp_ref(n) != Ordering::Greater,
                "x^k > N (n bits={}, k={})",
                n.bit_len(),
                k
            );
            assert!(
                x.add(&Nat::one()).pow(u64::from(k)).cmp_ref(n) == Ordering::Greater,
                "(x+1)^k <= N (n bits={}, k={})",
                n.bit_len(),
                k
            );
        };
        // Exhaustive small values across many k.
        for n in 0u64..2000 {
            let nat = Nat::from_u64(n);
            for k in 2u32..=12 {
                check(&nat, k);
            }
        }
        // Perfect powers and one below: m^k → m, m^k − 1 → m − 1.
        for m in 2u64..200 {
            for k in 3u32..=8 {
                let mk = Nat::from_u64(m).pow(u64::from(k));
                assert_eq!(mk.nth_root_floor(k), Nat::from_u64(m));
                assert_eq!(
                    mk.checked_sub(&Nat::one()).unwrap().nth_root_floor(k),
                    Nat::from_u64(m - 1)
                );
            }
        }
        // Large pseudo-random operands.
        let mut seed = 0xB007_5EEDu64;
        for _ in 0..200 {
            let mut bytes = alloc::vec::Vec::new();
            let limbs = 1 + (seed as usize % 40);
            for _ in 0..limbs * 8 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                bytes.push((seed >> 40) as u8);
            }
            let n = Nat::from_bytes_le(&bytes);
            for k in [3u32, 4, 5, 7, 11, 13] {
                check(&n, k);
            }
        }
    }

    #[test]
    fn newton_division_matches_knuth() {
        let mut st = 0xdead_beef_1234_5678u64;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let mut rand = |len: usize, mode: u64| -> Nat {
            Nat::from_limb_vec(
                (0..len)
                    .map(|_| match mode {
                        0 => next(),
                        1 => u64::MAX,
                        _ => {
                            if next() % 4 == 0 {
                                next()
                            } else {
                                0
                            }
                        }
                    })
                    .collect(),
            )
        };
        for &(la, lb) in &[
            (5usize, 3usize),
            (400, 200),
            (401, 250),
            (900, 450),
            (1000, 300),
            (1200, 1100),
            (2000, 390),
            (398, 200),
            (399, 200),
            (600, 200),
            (601, 200),
            (396, 200),
            (397, 200),
            (300, 200),
        ] {
            for mode in 0..3u64 {
                let a = rand(la, mode).add_u64(1);
                let mut b = rand(lb, (mode + 1) % 3);
                if b.limbs.len() < 2 {
                    b = b.add(&limbs_shl(&Nat::one(), lb - 1));
                }
                if a.cmp_ref(&b) != Ordering::Greater {
                    continue;
                }
                assert_eq!(
                    newton_div_rem(&a, &b),
                    a.div_rem_knuth(&b),
                    "{la}/{lb} mode {mode}"
                );
            }
            // β^(lb−1) and β^lb − 1 divisors, all-ones dividend.
            let a = rand(la, 1);
            let b = limbs_shl(&Nat::one(), lb - 1);
            assert_eq!(newton_div_rem(&a, &b), a.div_rem_knuth(&b));
            let b = rand(lb, 1);
            assert_eq!(newton_div_rem(&a, &b), a.div_rem_knuth(&b));
            let b = rand(lb, 1).shr(1);
            assert_eq!(newton_div_rem(&a, &b), a.div_rem_knuth(&b));
        }
        // The production dispatch (Nat::div_rem) on a size that selects Newton.
        let a = rand(3900, 0);
        let b = rand(770, 0);
        assert!(newton_div_worthwhile(770, 3900 - 770));
        assert_eq!(a.div_rem(&b).unwrap(), a.div_rem_knuth(&b));
        // newton_recip stays within its error bound
        for n in [3usize, 193, 200, 400, 777] {
            let d = rand(n, 0).add(&limbs_shl(&Nat::one(), n).shr(1));
            let d = Nat::from_limbs(&d.limbs[..n]);
            let d = if d.bit_len() < (n as u64) * 64 {
                d.add(&limbs_shl(&Nat::one(), n).shr(1))
            } else {
                d
            };
            let v = newton_recip(&d);
            let exact = limbs_shl(&Nat::one(), 2 * n).div_rem(&d).unwrap().0;
            let diff = if v > exact {
                v.checked_sub(&exact)
            } else {
                exact.checked_sub(&v)
            }
            .unwrap();
            assert!(diff <= Nat::from_u64(4), "recip error {diff} at n={n}");
        }
    }

    #[test]
    fn sqrt_rem_is_exact() {
        let mut st = 0x0123_4567_89ab_cdefu64;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let check = |a: &Nat| {
            let (s, r) = sqrt_rem(a);
            assert_eq!(s.square().add(&r), *a, "s² + r = a");
            assert!(r.cmp_ref(&s.shl(1)) != Ordering::Greater, "r ≤ 2s");
        };
        for len in (3..=120).chain([150, 193, 200, 301]) {
            for k in 0..6 {
                let mut v: Vec<Limb> = (0..len)
                    .map(|_| match next() % 5 {
                        0 => 0,
                        1 => Limb::MAX,
                        _ => next(),
                    })
                    .collect();
                v[len - 1] |= 1 << 62;
                if k == 1 {
                    v[len - 1] = Limb::MAX;
                }
                let a = Nat::from_limb_vec(v);
                check(&a);
                // Perfect squares and their neighbours stress both corrections.
                let s = a.isqrt();
                let sq = s.square();
                for x in [
                    sq.clone(),
                    sq.add_u64(0).checked_sub_u64(1).unwrap(),
                    s.add_u64(1).square().checked_sub_u64(1).unwrap(),
                ] {
                    if x.limbs.len() == len && x.limbs[len - 1] >> 62 != 0 {
                        check(&x);
                    }
                }
            }
        }
    }

    #[test]
    fn isqrt_u128_is_exact() {
        let check = |v: u128| {
            let x = isqrt_u128(v);
            assert!(x * x <= v, "isqrt({v}) = {x} too big");
            assert!(
                (x + 1).checked_mul(x + 1).is_none_or(|sq| sq > v),
                "isqrt({v}) = {x} too small"
            );
        };
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for sh in 0..128u32 {
            for _ in 0..200 {
                let v = (((next() as u128) << 64) | next() as u128) >> sh;
                check(v);
                let r = isqrt_u128(v);
                check(r * r);
                check((r * r).saturating_sub(1));
            }
        }
        for v in [
            0u128,
            1,
            2,
            3,
            4,
            u64::MAX as u128,
            1 << 64,
            (1 << 64) + 1,
            u128::MAX,
            u128::MAX - 1,
        ] {
            check(v);
        }
        let m = u64::MAX as u128;
        check(m * m);
        check(m * m - 1);
        check(m * m + 2 * m);
        for v in [u64::MAX, u64::MAX - 1, (1 << 52) + 1, 1 << 63] {
            check(v as u128);
            let r = isqrt_u64(v) as u128;
            assert!(r * r <= v as u128 && (r + 1) * (r + 1) > v as u128);
        }
    }

    #[test]
    fn neg_inv_mod_2_64_is_correct() {
        use crate::mont::neg_inv;
        let mut x = 1u64;
        for _ in 0..100_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1) | 1; // odd
            assert_eq!(x.wrapping_mul(neg_inv(x)), u64::MAX, "inverse of {x}");
        }
        assert_eq!(neg_inv(1), u64::MAX);
        assert_eq!(3u64.wrapping_mul(neg_inv(3)), u64::MAX);
    }

    #[test]
    fn ntt_primes_and_montgomery_product() {
        // Each prime is c·2^42 + 1 below 2^62 (the lazy butterflies need 4p <
        // 2^64), its constants are right, and `g` generates the whole group:
        // g^((p−1)/q) ≠ 1 for every prime q | p − 1.
        let factors: [&[u64]; 3] = [&[2, 3, 5, 863], &[2, 3, 181, 1931], &[2, 3, 349_483]];
        for (pr, qs) in MP_PRIMES.iter().zip(factors) {
            let p = pr.p;
            assert!(p < 1 << 62 && (p - 1) % (1 << MP_MAX_LOG2) == 0);
            assert_eq!(p.wrapping_mul(pr.pinv), 1);
            assert_eq!(
                pr.r2 as u128,
                (1u128 << 64) % p as u128 * ((1u128 << 64) % p as u128) % p as u128
            );
            let mut rest = (p - 1) >> MP_MAX_LOG2;
            for &q in qs {
                assert_ne!(
                    mp_pow_slow(pr.g, (p - 1) / q, p),
                    1,
                    "g not primitive mod {p}"
                );
                while rest.is_multiple_of(q) {
                    rest /= q;
                }
            }
            assert_eq!(rest, 1, "incomplete factorization of p − 1");
            // The Montgomery product lands in [1, 2p) and is ≡ a·b·2^-64.
            let r_inv = mp_pow_slow(((1u128 << 64) % p as u128) as u64, p - 2, p);
            let mut s: u64 = 0x9E37_79B9_7F4A_7C15 ^ p;
            let mut next = || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                s
            };
            let edges = [0, 1, p - 1, p, 2 * p - 1, 3 * p, 4 * p - 1];
            let check = |a: u64, b: u64| {
                let r = mp_mul(a, b, p, pr.pinv);
                assert!((1..2 * p).contains(&r), "range {a}·{b} mod {p}");
                let want = (a as u128 % p as u128) * (b as u128 % p as u128) % p as u128;
                let want = (want * r_inv as u128 % p as u128) as u64;
                assert_eq!(r % p, want, "{a}·{b} mod {p}");
            };
            for &a in &edges {
                for &b in &[0, 1, p - 1] {
                    check(a, b); // a < 4p, b < p
                }
                if a < 2 * p {
                    for &b in &edges[..5] {
                        check(a, b); // a, b < 2p
                    }
                }
            }
            for _ in 0..20_000 {
                check(next() % (4 * p), next() % p);
                check(next() % (2 * p), next() % (2 * p));
            }
            // Twiddles: each stage's root ω_2h (stored at tw[h + 1] in Montgomery
            // form) has ω_2h^h = −1.
            let tw = mp_twiddles(1 << 10, pr);
            let mut h = 2;
            while h < 1 << 10 {
                let w = mp_csub(mp_mul(tw[h + 1], 1, p, pr.pinv), p); // out of Montgomery form
                assert_eq!(mp_pow_slow(w, h as u64, p), p - 1, "stage h={h}");
                h *= 2;
            }
        }
        // Transform-length limit: lengths past 2^42 are refused (Toom fallback).
        assert_eq!(mp_shape(1), Some(2));
        assert_eq!(mp_shape(1024), Some(1024));
        assert_eq!(mp_shape(1025), Some(1536)); // 3·2^9 beats 2^11
        assert_eq!(mp_shape(1537), Some(2048));
        assert_eq!(mp_shape(5), Some(6));
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(mp_shape(1 << 42), Some(1 << 42));
            assert_eq!(mp_shape((1 << 42) + 1), Some(3 << 41));
            assert_eq!(mp_shape(3 << 42), Some(3 << 42));
            assert_eq!(mp_shape((3 << 42) + 1), None);
            assert!(!ntt_worthwhile(1 << 42, 1 << 42, false));
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
    fn ntt_matches_reference() {
        // The NTT must agree with the (verified) Karatsuba/Toom
        // paths across sizes straddling power-of-two transform lengths (the
        // fused first stage only applies when an operand fits in the low half),
        // squares, unbalanced shapes, and all-ones limbs (the largest possible
        // convolution coefficients, exercising the full CRT range).
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let rand = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let limbs: Vec<Limb> = (0..cnt).map(|_| f()).collect();
            Nat::from_limbs(&limbs)
        };
        let ones = |cnt: usize| Nat::from_limbs(&alloc::vec![Limb::MAX; cnt]);
        let reference = |a: &Nat, b: &Nat| {
            if a.limbs.len().min(b.limbs.len()) >= TOOM3_THRESHOLD {
                a.mul_toom3(b)
            } else {
                a.mul_karatsuba(b)
            }
        };
        // Tiny shapes against schoolbook (including n = 2 and odd lengths).
        for la in 1..=9 {
            for lb in 1..=9 {
                let (a, b) = (rand(la, &mut next), rand(lb, &mut next));
                assert_eq!(mul_ntt(&a, &b), a.mul_schoolbook(&b), "{la}x{lb}");
                let (a, b) = (ones(la), ones(lb));
                assert_eq!(mul_ntt(&a, &b), a.mul_schoolbook(&b), "ones {la}x{lb}");
            }
        }
        // Around the 2^k boundaries of la + lb − 1 and the block size.
        for &(la, lb) in &[
            (512usize, 513usize),
            (512, 512),
            (513, 512),
            (1000, 24),
            (2048, 2049),
            (2048, 2048),
            (2047, 2050),
            (4096, 4097),
            (3000, 5193),
            (7000, 129),
            (130, 9000),
        ] {
            let (a, b) = (rand(la, &mut next), rand(lb, &mut next));
            assert_eq!(mul_ntt(&a, &b), reference(&a, &b), "{la}x{lb}");
            let (a, b) = (ones(la), ones(lb));
            assert_eq!(mul_ntt(&a, &b), reference(&a, &b), "ones {la}x{lb}");
        }
        // Squares (one forward transform) against Karatsuba squaring.
        for &l in &[1usize, 2, 3, 300, 1024, 1025, 2049, 4097] {
            let a = rand(l, &mut next);
            assert_eq!(mul_ntt(&a, &a), a.square_karatsuba(), "square {l}");
            let a = ones(l);
            assert_eq!(mul_ntt(&a, &a), a.square_karatsuba(), "ones square {l}");
        }
        // Random shapes, with sparse/zero limbs mixed in.
        for _ in 0..12 {
            let la = 1 + (next() % 3000) as usize;
            let lb = 1 + (next() % 3000) as usize;
            let mut a = rand(la, &mut next);
            let b = rand(lb, &mut next);
            if la > 4 {
                let mut l = a.limbs.clone();
                l[1] = 0;
                l[la / 2] = 0;
                a = Nat::from_limbs(&l);
            }
            assert_eq!(mul_ntt(&a, &b), reference(&a, &b), "{la}x{lb}");
        }
    }

    /// Independent reference product: one plain multiply-accumulate row per
    /// limb of `a` (no `addmul_2`, no recursion).
    fn naive_product(a: &[Limb], b: &[Limb]) -> Vec<Limb> {
        let mut out = alloc::vec![0; a.len() + b.len()];
        for (i, &x) in a.iter().enumerate() {
            let mut carry = 0;
            for (j, &y) in b.iter().enumerate() {
                let (lo, hi) = mac(out[i + j], x, y, carry);
                out[i + j] = lo;
                carry = hi;
            }
            out[i + b.len()] = carry;
        }
        out
    }

    #[test]
    fn basecase_kernels_match_naive() {
        // The addmul_2 row kernel (via the basecase product, the accumulating
        // schoolbook and the symmetric square) on every small shape, with
        // random, all-ones and sparse limbs: all-ones maximizes the pending
        // value and deferred carries of both interleaved rows.
        let mut state = 0xb45e_ca5e_0000_0001u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for la in 1..=20usize {
            for lb in 1..=20usize {
                for mode in 0..3 {
                    let mut mk = |n: usize| -> Vec<Limb> {
                        (0..n)
                            .map(|_| match mode {
                                0 => next(),
                                1 => Limb::MAX,
                                _ => [0, Limb::MAX, 1, next()][(next() >> 62) as usize],
                            })
                            .collect()
                    };
                    let (a, b) = (mk(la), mk(lb));
                    let want = naive_product(&a, &b);
                    let mut out = alloc::vec![Limb::MAX; la + lb];
                    mul_basecase(&a, &b, &mut out);
                    assert_eq!(out, want, "basecase {la}x{lb} mode {mode}");
                    let mut out = alloc::vec![0; la + lb];
                    mul_into_schoolbook(&a, &b, &mut out);
                    assert_eq!(out, want, "schoolbook {la}x{lb} mode {mode}");
                    let mut sq = alloc::vec![0; 2 * la];
                    sqr_into(&a, &mut sq);
                    assert_eq!(sq, naive_product(&a, &a), "square {la} mode {mode}");
                }
            }
        }
    }

    #[test]
    fn slice_toom_matches_schoolbook() {
        // Every slice Toom kernel called directly on random and edge-heavy
        // shapes (all-ones / zero limbs drive the evaluation signs and the
        // interpolation carries to their extremes), plus the dispatcher on
        // shapes that select each path, against the schoolbook reference.
        let mut state = 0x7007_c00c_5eed_1234u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut limbs = |n: usize, mode: u64| -> Vec<Limb> {
            (0..n)
                .map(|_| {
                    let r = next();
                    match mode {
                        0 => r,
                        1 => Limb::MAX,
                        2 => [0, Limb::MAX, 1, r][(r >> 62) as usize],
                        _ => r & 0xFF,
                    }
                })
                .collect()
        };
        let reference = naive_product;
        type Kernel = fn(&[Limb], &[Limb], &mut [Limb], &mut [Limb]);
        type Shape = fn(usize, usize) -> bool;
        let kernels: [(&str, Kernel, Shape); 4] = [
            ("toom3", toom3_into, |la, lb| lb > 2 * la.div_ceil(3)),
            ("toom32", toom32_into, |la, lb| {
                lb > la.div_ceil(3) && lb <= 2 * la.div_ceil(3)
            }),
            ("toom4", toom4_into, |la, lb| lb > 3 * la.div_ceil(4)),
            ("slices", mul_slices, |_, _| true),
        ];
        for round in 0..600usize {
            let la = 4 + round * 7 % 300;
            let lb = 1 + (round * 13 + round / 3) % la;
            let mode = (round % 4) as u64;
            let (a, b) = (limbs(la, mode), limbs(lb, (mode + round as u64 / 4) % 4));
            let want = reference(&a, &b);
            for (name, kernel, ok) in &kernels {
                if !ok(la, lb) {
                    continue;
                }
                let mut out = alloc::vec![Limb::MAX; la + lb];
                // Small shapes call the kernels below their dispatch sizes,
                // where `toom_scratch_len` does not budget for them.
                let mut scratch = alloc::vec![Limb::MAX; toom_scratch_len(la) + 8 * la + 64];
                kernel(&a, &b, &mut out, &mut scratch);
                assert_eq!(out, want, "{name} {la}x{lb} mode {mode}");
            }
            for (name, kernel, ok) in [
                (
                    "toom3 sqr",
                    toom3_sqr_into as fn(&[Limb], &mut [Limb], &mut [Limb]),
                    la > 2 * la.div_ceil(3),
                ),
                ("toom4 sqr", toom4_sqr_into, la > 3 * la.div_ceil(4)),
                ("sqr slices", sqr_slices, true),
            ] {
                if !ok {
                    continue;
                }
                let want = reference(&a, &a);
                let mut out = alloc::vec![Limb::MAX; 2 * la];
                let mut scratch = alloc::vec![Limb::MAX; toom_scratch_len(la) + 8 * la + 64];
                kernel(&a, &mut out, &mut scratch);
                assert_eq!(out, want, "{name} {la} mode {mode}");
            }
        }
        // Through `Nat::mul`/`Nat::square` on shapes picking each path:
        // Karatsuba, Toom-3, Toom-3.2, Toom-4, block split, and the squaring
        // ladder.
        for &(la, lb) in &[
            (60usize, 50usize),
            (TOOM3_THRESHOLD, TOOM3_THRESHOLD),
            (200, 190),
            (300, 170),
            (TOOM4_THRESHOLD + 5, TOOM4_THRESHOLD),
            (450, 449),
            (480, 300),
            (900, 100),
            (1000, 333),
            (2000, 450),
        ] {
            for mode in 0..3 {
                let (a, b) = (limbs(la, mode), limbs(lb, mode));
                let (na, nb) = (Nat::from_limbs(&a), Nat::from_limbs(&b));
                let want = Nat::from_limbs(&reference(&a, &b));
                assert_eq!(na.mul(&nb), want, "mul {la}x{lb} mode {mode}");
                assert_eq!(nb.mul(&na), want, "mul {lb}x{la} mode {mode}");
                let sq = Nat::from_limbs(&reference(&a, &a));
                assert_eq!(na.square(), sq, "square {la} mode {mode}");
            }
        }
    }

    #[test]
    fn ntt_wrapped_lengths_match_schoolbook() {
        // Shapes whose coefficient count lands just past a 2^k / 3·2^k step
        // run a shorter cyclic transform and correct the wrapped top
        // coefficients exactly; check those against schoolbook, including
        // all-ones limbs (largest wrapped coefficients) and unbalanced shapes.
        let mut state = 0x5151_7e57_0dd5_eed5u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let rand = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let limbs: Vec<Limb> = (0..cnt).map(|_| f()).collect();
            Nat::from_limbs(&limbs)
        };
        let ones = |cnt: usize| Nat::from_limbs(&alloc::vec![Limb::MAX; cnt]);
        for &(la, lb) in &[
            (5usize, 5usize),
            (40, 40),
            (782, 782),
            (800, 760),
            (1563, 1563),
            (1600, 1500),
            (2000, 1100),
            (1500, 100),
        ] {
            let (n, wrap) = mp_plan(la, lb, false).unwrap();
            assert!(wrap > 0 && n + wrap == la + lb - 1, "{la}x{lb}: no wrap");
            let (a, b) = (rand(la, &mut next), rand(lb, &mut next));
            assert_eq!(mul_ntt(&a, &b), a.mul_schoolbook(&b), "{la}x{lb}");
            let (a, b) = (ones(la), ones(lb));
            assert_eq!(mul_ntt(&a, &b), a.mul_schoolbook(&b), "ones {la}x{lb}");
            let a = rand(la, &mut next);
            assert_eq!(mul_ntt(&a, &a), a.mul_schoolbook(&a), "square {la}");
        }
    }

    #[test]
    fn ntt_dispatch_matches_reference() {
        // `Nat::mul`/`Nat::square` on shapes `ntt_worthwhile` hands to the NTT
        // agree with the Karatsuba/Toom paths.
        let mut state = 0x0123_4567_89ab_cdefu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let rand = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let limbs: Vec<Limb> = (0..cnt).map(|_| f()).collect();
            Nat::from_limbs(&limbs)
        };
        for &(la, lb) in &[
            (2048usize, 2049usize),
            (2100, 2100),
            (4500, 1500),
            (8000, 600),
        ] {
            assert!(ntt_worthwhile(la, lb, false), "{la}x{lb}");
            let (a, b) = (rand(la, &mut next), rand(lb, &mut next));
            assert_eq!(a.mul(&b), a.mul_karatsuba(&b), "{la}x{lb}");
        }
        for &l in &[1800usize, 2048, 2800] {
            assert!(ntt_worthwhile(l, l, true), "square {l}");
            let a = rand(l, &mut next);
            assert_eq!(a.square(), a.square_karatsuba(), "square {l}");
            assert_eq!(a.mul(&a.clone()), a.square_karatsuba(), "self-mul {l}");
        }
        assert!(!ntt_worthwhile(1500, 1500, false) && !ntt_worthwhile(400, 9000, false));
        assert!(!ntt_worthwhile(4000, 500, false) && !ntt_worthwhile(4000, 1000, false));
        assert!(!ntt_worthwhile(1500, 1500, true) && !ntt_worthwhile(2400, 2400, true));
    }

    #[test]
    #[ignore = "slow in debug: cargo test --release -- --ignored ntt_matches_reference_large"]
    fn ntt_matches_reference_large() {
        // Large shapes, including la + lb − 1 landing exactly on and just past
        // a power of two (2^16), against Toom-3 / Karatsuba squaring.
        let mut state = 0xfeed_face_cafe_beefu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let rand = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let limbs: Vec<Limb> = (0..cnt).map(|_| f()).collect();
            Nat::from_limbs(&limbs)
        };
        let ones = |cnt: usize| Nat::from_limbs(&alloc::vec![Limb::MAX; cnt]);
        for &(la, lb) in &[
            (20_000usize, 20_000usize),
            (32_768, 32_769),
            (32_768, 32_770),
            (100_000, 37_000),
            (250_000, 3_000),
        ] {
            let (a, b) = (rand(la, &mut next), rand(lb, &mut next));
            assert_eq!(mul_ntt(&a, &b), a.mul_toom3(&b), "{la}x{lb}");
        }
        let (a, b) = (ones(50_000), ones(50_001));
        assert_eq!(mul_ntt(&a, &b), a.mul_toom3(&b), "ones 50000x50001");
        for &l in &[16_384usize, 40_000] {
            let a = rand(l, &mut next);
            assert_eq!(mul_ntt(&a, &a), a.square_karatsuba(), "square {l}");
            let a = ones(l);
            assert_eq!(mul_ntt(&a, &a), a.square_karatsuba(), "ones square {l}");
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

    /// Best-of timing helper for the measurement tests below.
    fn bench_best(f: &dyn Fn() -> Nat) -> core::time::Duration {
        use std::time::Instant;
        let mut best = core::time::Duration::MAX;
        let _ = f();
        let start = Instant::now();
        let mut runs = 0;
        while runs < 5 || (start.elapsed().as_millis() < 200 && runs < 200) {
            let t = Instant::now();
            let r = f();
            best = best.min(t.elapsed());
            drop(r);
            runs += 1;
        }
        best
    }

    /// Deterministic `limbs`-limb operand with the top bit set.
    fn bench_operand(limbs: usize, seed: usize) -> Nat {
        let v: Vec<Limb> = (0..limbs)
            .map(|i| ((i + seed) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1 << 63)
            .collect();
        Nat::from_limbs(&v)
    }

    /// Sizes from the comma-separated environment variable `var`, or `default`.
    fn bench_sizes(var: &str, default: &[usize]) -> Vec<usize> {
        std::env::var(var)
            .ok()
            .map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect())
            .unwrap_or_else(|| default.to_vec())
    }

    #[test]
    #[ignore = "measurement only: cargo test --release -- --ignored --nocapture measure_mul"]
    fn measure_mul_crossovers() {
        // Each slice kernel forced at the top level (recursing through the
        // normal dispatcher), for retuning the Karatsuba/Toom thresholds.
        // `MUL_SIZES=a,b,c` overrides the size list.
        type Kernel = fn(&[Limb], &[Limb], &mut [Limb], &mut [Limb]);
        type SqrKernel = fn(&[Limb], &mut [Limb], &mut [Limb]);
        let kara: Kernel = |a, b, out, s| {
            out.fill(0);
            kara_into(a, b, out, s);
        };
        let kara_sqr: SqrKernel = |a, out, s| {
            out.fill(0);
            kara_sqr_into(a, out, s);
        };
        let sizes = bench_sizes(
            "MUL_SIZES",
            &[
                16, 24, 32, 48, 64, 96, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1000,
            ],
        );
        for &sz in &sizes {
            let (a, b) = (bench_operand(sz, 1), bench_operand(sz, 7));
            // Kernels forced below their dispatch sizes need more scratch
            // than `toom_scratch_len` budgets.
            let scratch_len = toom_scratch_len(sz) + 8 * sz + 64;
            let run = |k: Kernel| {
                bench_best(&|| {
                    let mut out = alloc::vec![0; 2 * sz];
                    let mut s = alloc::vec![0; scratch_len];
                    k(&a.limbs, &b.limbs, &mut out, &mut s);
                    Nat::from_limb_vec(out)
                })
            };
            let run_sqr = |k: SqrKernel| {
                bench_best(&|| {
                    let mut out = alloc::vec![0; 2 * sz];
                    let mut s = alloc::vec![0; scratch_len];
                    k(&a.limbs, &mut out, &mut s);
                    Nat::from_limb_vec(out)
                })
            };
            let school = run(|a, b, o, _| mul_basecase(a, b, o));
            let k2 = run(kara);
            let t3 = run(toom3_into);
            let t4 = run(toom4_into);
            let sq = run_sqr(|a, o, _| {
                o.fill(0);
                sqr_into(a, o);
            });
            let ksq = run_sqr(kara_sqr);
            let t3s = run_sqr(toom3_sqr_into);
            let t4s = run_sqr(toom4_sqr_into);
            std::println!(
                "sz={sz:<5} mul: school={school:>10.3?} kara={k2:>10.3?} toom3={t3:>10.3?} toom4={t4:>10.3?} | sqr: school={sq:>10.3?} kara={ksq:>10.3?} toom3={t3s:>10.3?} toom4={t4s:>10.3?}"
            );
        }
    }

    #[test]
    #[ignore = "measurement only: cargo test --release -- --ignored --nocapture measure_ntt_paths"]
    fn measure_ntt_paths() {
        // NTT vs the Karatsuba/Toom ladder per balanced shape (mul and
        // square), for retuning `ntt_worthwhile`. `NTT_SIZES=a,b,c`
        // overrides the size list.
        let sizes = bench_sizes(
            "NTT_SIZES",
            &[
                400, 500, 600, 700, 800, 900, 1000, 1100, 1300, 1500, 1700, 2000, 2400, 2800, 3500,
                4100, 5000, 6000, 8000, 12000,
            ],
        );
        for &sz in &sizes {
            let (a, b) = (bench_operand(sz, 1), bench_operand(sz + 1, 7));
            let (la, lb) = (a.limbs.as_slice(), b.limbs.as_slice());
            if std::env::var("NTT_WRAP").is_ok() {
                let t = bench_best(&|| {
                    let w = mp_wrapped_coeffs(la, lb, sz / 2);
                    Nat::from_u64(w.len() as u64)
                });
                let prods = mp_wrap_products(sz / 2, sz, sz + 1);
                std::println!(
                    "wrap sz={sz} {t:?} = {:.2} ns/product",
                    t.as_nanos() as f64 / prods as f64
                );
                continue;
            }
            let toom = bench_best(&|| product_with(la, lb, |o, s| mul_slices(la, lb, o, s)));
            let ntt = bench_best(&|| mul_ntt(&a, &b));
            let toom_sq = bench_best(&|| product_with(la, la, |o, s| sqr_slices(la, o, s)));
            let ntt_sq = bench_best(&|| mul_ntt(&a, &a));
            let plan = mp_plan(sz, sz + 1, false);
            std::println!(
                "sz={sz:<6} plan={plan:?} mul: toom={toom:>10.3?} ntt={ntt:>10.3?} {} | sqr: toom={toom_sq:>10.3?} ntt={ntt_sq:>10.3?} {}",
                if ntt < toom { "NTT" } else { "toom" },
                if ntt_sq < toom_sq { "NTT" } else { "toom" },
            );
        }
    }

    #[test]
    #[ignore = "measurement only: cargo test --release -- --ignored --nocapture measure_ntt_unbalanced"]
    fn measure_ntt_unbalanced() {
        // NTT vs the Toom ladder on unbalanced shapes. `NTT_SHORT=a,b,c`
        // and `NTT_RATIOS=a,b,c` override the shorter sizes and ratios.
        let shorts = bench_sizes(
            "NTT_SHORT",
            &[200, 300, 400, 500, 700, 1000, 1500, 2000, 3000],
        );
        let ratios = bench_sizes("NTT_RATIOS", &[2, 3, 4, 8, 16, 32]);
        for &lb in &shorts {
            for &r in &ratios {
                let la = lb * r;
                let (a, b) = (bench_operand(la, 1), bench_operand(lb, 7));
                let (sa, sb) = (a.limbs.as_slice(), b.limbs.as_slice());
                let toom = bench_best(&|| product_with(sa, sb, |o, s| mul_slices(sa, sb, o, s)));
                let ntt = bench_best(&|| mul_ntt(&a, &b));
                std::println!(
                    "{la:>7}x{lb:<6} toom={toom:>10.3?} ntt={ntt:>10.3?} {} plan={:?}",
                    if ntt < toom { "NTT" } else { "toom" },
                    mp_plan(la, lb, false)
                );
            }
        }
    }
    #[test]
    #[ignore = "measurement only: cargo test --release -- --ignored --nocapture measure_rho"]
    fn measure_rho() {
        use std::time::Instant;
        // Smallest prime ≥ 10^d, via the crate's exact (< 2^64) BPSW test.
        let prime_near = |start: u64| -> Nat {
            let mut c = start | 1;
            loop {
                let v = Nat::from_u64(c);
                if v.is_prime_bpsw() {
                    return v;
                }
                c += 2;
            }
        };
        // Balanced semiprimes squarely in Pollard rho's fast range.
        for &digits in &[6u32, 7, 8, 9, 10] {
            let base = 10u64.pow(digits);
            let p = prime_near(base + 12345);
            let q = prime_near(base + 67891);
            let n = p.mul(&q);
            // Warm up, then take the fastest of several full-factorization runs.
            let mut best = core::time::Duration::MAX;
            let _ = pollard_rho(&n, Some(1 << 22));
            for _ in 0..5 {
                let t = Instant::now();
                let f = pollard_rho(&n, Some(1 << 22)).expect("rho splits it");
                best = best.min(t.elapsed());
                assert!(n.div_rem(&f).expect("f != 0").1.is_zero());
            }
            std::println!("rho  {digits:>2}+{digits:>2}-digit semiprime  {best:>12?}");
        }
    }

    #[test]
    #[ignore = "measurement only: cargo test --release --all-features -- --ignored --nocapture measure_prime_prefilter"]
    fn measure_prime_prefilter() {
        use std::time::Instant;

        // OLD is_prime_bpsw (no small-prime prefilter), to A/B against.
        fn bpsw_no_prefilter(x: &Nat) -> bool {
            let two = Nat::from_u64(2);
            let three = Nat::from_u64(3);
            if x.cmp_ref(&two) == Ordering::Less {
                return false;
            }
            if x.cmp_ref(&three) != Ordering::Greater {
                return true;
            }
            if x.is_even() {
                return false;
            }
            let r = x.isqrt();
            if r.square() == *x {
                return false;
            }
            miller_rabin_witness(&two, x) && lucas_strong(x)
        }

        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mkodd = |bits: u64, f: &mut dyn FnMut() -> u64| -> Nat {
            let limbs = bits.div_ceil(64) as usize;
            let mut v: Vec<Limb> = (0..limbs).map(|_| f()).collect();
            let top = ((bits - 1) % 64) as u32;
            let last = v.len() - 1;
            v[last] &= (1u64 << top) | ((1u64 << top) - 1); // keep within `bits`
            v[last] |= 1u64 << top; // set the high bit
            v[0] |= 1; // odd
            let mut n = Nat { limbs: v };
            n.normalize();
            n
        };

        std::println!("--- is_prime_bpsw on random odd composites (500 samples) ---");
        for &bits in &[64u64, 256, 1024, 4096] {
            let samples: Vec<Nat> = (0..500).map(|_| mkodd(bits, &mut next)).collect();
            let bench = |f: &dyn Fn(&Nat) -> bool| {
                let mut best = core::time::Duration::MAX;
                for _ in 0..3 {
                    let t = Instant::now();
                    let mut acc = 0usize;
                    for s in &samples {
                        acc += f(s) as usize;
                    }
                    std::hint::black_box(acc);
                    best = best.min(t.elapsed());
                }
                best
            };
            let old = bench(&bpsw_no_prefilter);
            let new = bench(&|s| s.is_prime_bpsw());
            let caught = samples
                .iter()
                .filter(|s| small_prime_verdict(s) == Some(false))
                .count();
            std::println!(
                "bits={bits:<5} old={old:>12?} new={new:>12?} speedup={:.2}x  prefilter-caught={}/500",
                old.as_secs_f64() / new.as_secs_f64(),
                caught
            );
        }

        std::println!("--- next_prime (old scan vs new sieve), 20 seeds each ---");
        // OLD next_prime: odd scan gated on old bpsw (no prefilter, no sieve).
        fn old_next_prime(x: &Nat) -> Nat {
            let two = Nat::from_u64(2);
            if x.cmp_ref(&two) == Ordering::Less {
                return two;
            }
            let mut c = x.add(&Nat::one());
            if c.is_even() {
                c = c.add(&Nat::one());
            }
            loop {
                // inline old bpsw
                let three = Nat::from_u64(3);
                let is_p = if c.cmp_ref(&two) == Ordering::Less {
                    false
                } else if c.cmp_ref(&three) != Ordering::Greater {
                    true
                } else if c.is_even() {
                    false
                } else {
                    let r = c.isqrt();
                    r.square() != c && miller_rabin_witness(&two, &c) && lucas_strong(&c)
                };
                if is_p {
                    return c;
                }
                c = c.add(&two);
            }
        }
        for &bits in &[64u64, 256, 1024] {
            let seeds: Vec<Nat> = (0..20).map(|_| mkodd(bits, &mut next)).collect();
            let bench = |f: &dyn Fn(&Nat) -> Nat| {
                let mut best = core::time::Duration::MAX;
                for _ in 0..3 {
                    let t = Instant::now();
                    for s in &seeds {
                        std::hint::black_box(f(s));
                    }
                    best = best.min(t.elapsed());
                }
                best
            };
            let old = bench(&old_next_prime);
            let new = bench(&|s| s.next_prime());
            std::println!(
                "bits={bits:<5} old={old:>12?} new={new:>12?} speedup={:.2}x",
                old.as_secs_f64() / new.as_secs_f64()
            );
        }

        std::println!("--- prefilter rejection rate vs prime-count bound ---");
        {
            let samples: Vec<Nat> = (0..20_000).map(|_| mkodd(256, &mut next)).collect();
            for &k in &[10usize, 25, 54, 100, 167] {
                // Rejection rate using only the first k small primes.
                let mut caught = 0usize;
                for s in &samples {
                    let mut hit = false;
                    let mut i = 0;
                    while i < k {
                        let start = i;
                        let mut m: Limb = 1;
                        while i < k {
                            match m.checked_mul(SMALL_PRIMES[i] as Limb) {
                                Some(v) => {
                                    m = v;
                                    i += 1;
                                }
                                None => break,
                            }
                        }
                        let r = s.rem_small(m);
                        for &p in &SMALL_PRIMES[start..i] {
                            if r.is_multiple_of(p as Limb) {
                                hit = true;
                            }
                        }
                        if hit {
                            break;
                        }
                    }
                    caught += hit as usize;
                }
                std::println!(
                    "primes<={:<4} caught={:.1}%",
                    SMALL_PRIMES[k - 1],
                    100.0 * caught as f64 / samples.len() as f64
                );
            }
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
    fn prefilter_differential_exhaustive() {
        // The small-prime trial-division prefilter must not change any verdict:
        // over a large exhaustive range, `is_prime_bpsw` still agrees with pure
        // trial division for every n (covers all small-prime residues, prime
        // powers, and numbers equal to a small prime — the `n == p` edge).
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
        for m in 0u64..200_000 {
            assert_eq!(
                Nat::from_u64(m).is_prime_bpsw(),
                trial(m),
                "bpsw prefilter {m}"
            );
        }
    }

    #[test]
    fn prefilter_large_constructed() {
        // Known verdicts on large numbers of each shape the prefilter must route
        // correctly: primes (no small factor → full test), semiprimes and prime
        // powers of large primes (no small factor → isqrt/MR/Lucas rejects),
        // composites with a small factor (prefilter rejects), and Carmichaels.
        let large_primes = [
            "1000000007",
            "170141183460469231731687303715884105727", // 2^127 − 1
            "2305843009213693951",                     // 2^61 − 1
            "618970019642690137449562111",             // 2^89 − 1
        ];
        for p in large_primes {
            assert!(n(p).is_prime_bpsw(), "prime {p}");
        }
        // Semiprimes of two large primes (no small factor).
        let a = n("1000000007");
        let b = n("2305843009213693951");
        assert!(!a.mul(&b).is_prime_bpsw(), "large semiprime");
        // Prime power of a large prime (no small factor, not a perfect square
        // for odd exponent; the cube here still fails MR/Lucas).
        assert!(!a.mul(&a).mul(&a).is_prime_bpsw(), "large prime cube");
        // Perfect square of a large prime (exercises the isqrt guard).
        assert!(!a.mul(&a).is_prime_bpsw(), "large prime square");
        // Large composite with a small factor 3·(2^127−1).
        let m127 = n("170141183460469231731687303715884105727");
        assert!(!Nat::from_u64(3).mul(&m127).is_prime_bpsw(), "3·(2^127−1)");
        // Big Carmichael numbers.
        for c in ["162401", "410041", "1024651", "9746347772161"] {
            assert!(!n(c).is_prime_bpsw(), "carmichael {c}");
        }
    }

    #[test]
    fn next_prev_prime_match_scan() {
        // The incremental residue sieve in next_prime/prev_prime must return
        // exactly what a naive candidate scan gated on the (authoritative,
        // unchanged) is_prime_bpsw would — for a dense small range and for a few
        // large seeds where a skipped prime would be visible.
        fn scan_next(x: &Nat) -> Nat {
            let two = Nat::from_u64(2);
            if x.cmp_ref(&two) == Ordering::Less {
                return two;
            }
            let mut c = x.add(&Nat::one());
            if c.is_even() {
                c = c.add(&Nat::one());
            }
            while !c.is_prime_bpsw() {
                c = c.add(&two);
            }
            c
        }
        fn scan_prev(x: &Nat) -> Option<Nat> {
            let two = Nat::from_u64(2);
            if x.cmp_ref(&two) != Ordering::Greater {
                return None;
            }
            let mut c = x.checked_sub(&Nat::one()).unwrap();
            loop {
                if c.cmp_ref(&two) == Ordering::Less {
                    return Some(two);
                }
                if c.is_prime_bpsw() {
                    return Some(c);
                }
                c = c.checked_sub(&Nat::one()).unwrap_or_else(Nat::zero);
            }
        }
        for m in 0u64..5_000 {
            let x = Nat::from_u64(m);
            assert_eq!(x.next_prime(), scan_next(&x), "next_prime {m}");
            assert_eq!(x.prev_prime(), scan_prev(&x), "prev_prime {m}");
        }
        // Large seeds across bit sizes.
        for s in [
            "1000000000000",
            "18446744073709551616",                    // 2^64
            "340282366920938463463374607431768211456", // 2^128
            "123456789012345678901234567890",
        ] {
            let x = n(s);
            assert_eq!(x.next_prime(), scan_next(&x), "next_prime large {s}");
            assert_eq!(x.prev_prime(), scan_prev(&x), "prev_prime large {s}");
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

    /// Structured operand pairs that stress the double-word Lehmer pass:
    /// huge quotients (uncertifiable windows), identical leading words,
    /// Fibonacci neighbours (all quotients 1), powers of two, near-equal
    /// operands and planted common factors, across 1–70 limbs.
    fn lehmer_structured_pairs(next: &mut dyn FnMut() -> u64) -> Vec<(Nat, Nat)> {
        let build = |cnt: usize, f: &mut dyn FnMut() -> u64| {
            let limbs: Vec<Limb> = (0..cnt).map(|_| f()).collect();
            Nat::from_limbs(&limbs)
        };
        let mut out = Vec::new();
        for la in [1usize, 2, 3, 4, 5, 8, 17, 33, 70] {
            for lb in [1usize, 2, 3, 4, 9, 33, 70] {
                let a = build(la, next);
                let b = build(lb, next);
                out.push((a.clone(), b.clone()));
                // Huge first quotient: a·2^k + small.
                out.push((b.shl(64 * la as u64 + 5).add(&a), b.clone()));
                // Shared leading words, different tails.
                let top = build(3, next).shl(64 * la as u64);
                out.push((top.add(&a), top.add(&build(la, next))));
                // Near-equal.
                out.push((a.add(&Nat::one()), a.clone()));
                // Planted common factor (gcd > 1).
                let f = build(1 + lb / 3, next);
                out.push((a.mul(&f), b.mul(&f)));
                // Power-of-two content on both.
                out.push((a.shl(70), b.shl(130)));
            }
        }
        // Consecutive Fibonacci numbers of several sizes.
        let (mut f0, mut f1) = (Nat::zero(), Nat::one());
        for i in 0..3000 {
            let f2 = f0.add(&f1);
            f0 = core::mem::replace(&mut f1, f2);
            if i % 250 == 7 {
                out.push((f1.clone(), f0.clone()));
            }
        }
        out.retain(|(a, b)| !a.is_zero() && !b.is_zero());
        out
    }

    #[test]
    fn lehmer_structured_matches_binary_gcd() {
        let mut next = hgcd_rng(0x0bad_cafe_1234_5678);
        for (a, b) in lehmer_structured_pairs(&mut next) {
            let g = a.gcd_binary(&b);
            assert_eq!(a.gcd_lehmer(&b), g, "gcd_lehmer {a:?} {b:?}");
            assert_eq!(b.gcd_lehmer(&a), g, "gcd_lehmer swapped");
            assert_eq!(a.gcd(&b), g, "dispatched gcd");
        }
    }

    #[test]
    fn extgcd_lehmer_matches_euclid_cofactors() {
        // The extended Lehmer loop must reproduce plain extended Euclid's
        // (minimal) cofactors exactly, for both orders and gcd > 1.
        let mut next = hgcd_rng(0x5eed_e47e_0dd5_0001);
        let mut pairs = lehmer_structured_pairs(&mut next);
        for _ in 0..300 {
            let la = 1 + (next() % 40) as usize;
            let lb = 1 + (next() % 40) as usize;
            let a = hgcd_build(la, &mut next);
            let b = hgcd_build(lb, &mut next);
            if !a.is_zero() && !b.is_zero() {
                pairs.push((a, b));
            }
        }
        for (a, b) in pairs {
            let (ai, bi) = (Int::from(a.clone()), Int::from(b.clone()));
            let (eg, ex, ey) = ai.extended_gcd_euclid(&bi);
            let (g, x, y) = a.extgcd_lehmer_full(&b);
            assert_eq!(Int::from(g.clone()), eg, "g");
            assert_eq!((&x, &y), (&ex, &ey), "cofactors {a:?} {b:?}");
            let (g1, x1) = a.extgcd_lehmer(&b);
            assert_eq!((g1, x1), (g, x), "single-cofactor variant");
        }
    }

    #[test]
    fn int_extended_gcd_structured_matches_euclid() {
        // Exercises the tracked-cofactor fast path and its normalization
        // fallback (gcd > 1, |b| = 2g, ...) for every sign combination.
        let mut next = hgcd_rng(0x1e47_0000_5eed_0002);
        let mut pairs = lehmer_structured_pairs(&mut next);
        pairs.push((Nat::from_u64(6), Nat::from_u64(4)));
        pairs.push((Nat::from_u64(2), Nat::from_u64(7)));
        pairs.push((Nat::from_u64(10), Nat::from_u64(4)));
        for (a, b) in pairs {
            for (sa, sb) in [(1i64, 1i64), (-1, 1), (1, -1), (-1, -1)] {
                let ai = Int::from(a.clone()).mul(&Int::from_i64(sa));
                let bi = Int::from(b.clone()).mul(&Int::from_i64(sb));
                assert_eq!(
                    ai.extended_gcd(&bi),
                    ai.extended_gcd_euclid(&bi),
                    "{ai:?} {bi:?}"
                );
                assert_eq!(
                    bi.extended_gcd(&ai),
                    bi.extended_gcd_euclid(&ai),
                    "{bi:?} {ai:?}"
                );
            }
        }
    }

    #[test]
    fn lehmer_divrem_matches_native() {
        let mut next = hgcd_rng(0x0123_4567_89ab_cdef);
        for i in 0..200_000u32 {
            let x = ((next() as u128) << 64 | next() as u128) >> (next() % 128);
            let y = ((next() as u128) << 64 | next() as u128) >> (next() % 128);
            let (x, y) = if x >= y { (x, y) } else { (y, x) };
            if y == 0 {
                continue;
            }
            // Bias some cases towards small quotients and near-full words.
            let y = if i % 3 == 0 {
                (x >> (1 + next() % 6)).max(1)
            } else {
                y
            };
            assert_eq!(lehmer_divrem(x, y), (x / y, x % y), "{x} / {y}");
        }
    }

    #[test]
    fn hgcd_fast_guard_rejects_corruption() {
        let mut next = hgcd_rng(0x9a4d_0000_0000_0001);
        for _ in 0..50 {
            let m = hgcd_build(12, &mut next);
            let n = hgcd_build(11, &mut next);
            let (m, n) = if m > n { (m, n) } else { (n, m) };
            let (r, va, vb) = hgcd(&m, &n);
            assert!(hgcd_reduction_ok_fast(&r, &m, &n, &va, &vb));
            assert!(!hgcd_reduction_ok_fast(
                &r,
                &m,
                &n,
                &va.add(&Nat::one()),
                &vb
            ));
            assert!(!hgcd_reduction_ok_fast(
                &r,
                &m,
                &n,
                &va,
                &vb.add(&Nat::one().shl(300))
            ));
            let mut bad = r.clone();
            bad.a = bad.a.add(&Int::ONE);
            assert!(!hgcd_reduction_ok_fast(&bad, &m, &n, &va, &vb));
        }
    }

    // Deterministic xorshift helper for the Half-GCD tests.
    fn hgcd_rng(mut state: u64) -> impl FnMut() -> u64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    fn hgcd_build(cnt: usize, f: &mut dyn FnMut() -> u64) -> Nat {
        let bytes: Vec<u8> = (0..cnt * 8).map(|_| f() as u8).collect();
        Nat::from_bytes_le(&bytes)
    }

    // Forces the Half-GCD driver (with full recursion) regardless of the
    // production size threshold, so the tests exercise it on medium inputs where
    // an independent oracle is cheap. `floor` is one above the recursion base so
    // even modest sizes recurse.
    fn gcd_via_hgcd(a: &Nat, b: &Nat) -> Nat {
        if a.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return a.clone();
        }
        a.gcd_hgcd_floor(b, HGCD_RECURSION_BASE_LIMBS + 1)
            .expect("hgcd reduction guard tripped — gcd invariant violated")
    }

    // Forces the Half-GCD driver with an explicit small `floor` so the driver
    // loop actually engages on small operands (paired with a shrunk recursion
    // base). A tripped guard (`None`) panics — that is a bug reproduction.
    fn gcd_via_hgcd_floor(a: &Nat, b: &Nat, floor: usize) -> Nat {
        if a.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return a.clone();
        }
        a.gcd_hgcd_floor(b, floor)
            .expect("hgcd reduction guard tripped — gcd invariant violated")
    }

    // One deterministic differential pass over the danger zones: sizes straddling
    // the (shrunk) recursion base and the split points, near-equal operands, one
    // much larger, and planted common factors that force make-valid back-ups.
    // Every pair is checked against the independent binary-GCD oracle through the
    // guarded Half-GCD driver (both operand orders). `base` is the shrunk
    // recursion base; `floor` (3) keeps the driver looping down to Lehmer.
    fn hgcd_diff_pass(seed: u64, count: usize, base: usize) {
        hgcd_with_test_base(base, || {
            let mut next = hgcd_rng(seed);
            for i in 0..count {
                let span = 4 * base + 4;
                let (la, lb) = match next() % 6 {
                    // Just above the base (base/recurse boundary).
                    0 => (
                        base + 1 + (next() as usize % 3),
                        base + 1 + (next() as usize % 3),
                    ),
                    // Near-equal sizes around two levels.
                    1 => {
                        let s = 1 + next() as usize % span;
                        (s, s.saturating_sub(next() as usize % 2))
                    }
                    // One much larger than the other (unbalanced).
                    2 => (
                        3 * base + (next() as usize % (2 * base)),
                        1 + next() as usize % 3,
                    ),
                    // Small, around one level.
                    3 => (
                        1 + next() as usize % (base + 2),
                        1 + next() as usize % (base + 2),
                    ),
                    // Wide random.
                    _ => (1 + next() as usize % span, 1 + next() as usize % span),
                };
                let mut m = hgcd_build(la, &mut next);
                let mut n = hgcd_build(lb, &mut next);
                if next().is_multiple_of(2) {
                    let g = hgcd_build(1 + next() as usize % (base + 1), &mut next);
                    if !g.is_zero() {
                        m = m.mul(&g);
                        n = n.mul(&g);
                    }
                }
                let oracle = if m.is_zero() {
                    n.clone()
                } else if n.is_zero() {
                    m.clone()
                } else {
                    m.gcd_binary(&n)
                };
                assert_eq!(
                    gcd_via_hgcd_floor(&m, &n, 3),
                    oracle,
                    "seed {seed} i {i} la {la} lb {lb} base {base}"
                );
                assert_eq!(
                    gcd_via_hgcd_floor(&n, &m, 3),
                    oracle,
                    "seed {seed} i {i} sym la {la} lb {lb} base {base}"
                );
            }
        });
    }

    #[test]
    fn hgcd_forced_recursion_deep() {
        // Always-run subset: a few thousand pairs per shrunk base, spanning one
        // to several recursion levels, all checked against the binary-GCD oracle.
        for (seed, base) in [
            (0x1111_2222_3333_4444u64, 3usize),
            (0x5555_6666_7777_8888, 4),
            (0x9999_aaaa_bbbb_cccc, 6),
            (0xdddd_eeee_ffff_0001, 9),
            (0x0f0f_0f0f_1e1e_1e1e, 13),
        ] {
            hgcd_diff_pass(seed, 1200, base);
        }
    }

    #[test]
    fn hgcd_base_boundary_real() {
        // The real 128-limb base: sizes straddling it so both the base case
        // (with the in-place double-word Lehmer pass) and one recursion level
        // are exercised, checked bit-for-bit against Lehmer.
        let mut next = hgcd_rng(0xa5a5_5a5a_c3c3_3c3c);
        for _ in 0..24 {
            let la = HGCD_RECURSION_BASE_LIMBS - 2 + (next() as usize % 8);
            let lb = HGCD_RECURSION_BASE_LIMBS - 2 + (next() as usize % 8);
            let m = hgcd_build(la, &mut next);
            let n = hgcd_build(lb, &mut next);
            if m.is_zero() || n.is_zero() {
                continue;
            }
            let oracle = m.gcd_lehmer(&n);
            assert_eq!(
                gcd_via_hgcd_floor(&m, &n, 3),
                oracle,
                "boundary la {la} lb {lb}"
            );
            // Planted factor so the gcd is nontrivial across the boundary.
            let g = hgcd_build(1 + next() as usize % 40, &mut next);
            if g.is_zero() {
                continue;
            }
            let (gm, gn) = (m.mul(&g), n.mul(&g));
            assert_eq!(
                gcd_via_hgcd_floor(&gm, &gn, 3),
                gm.gcd_lehmer(&gn),
                "boundary planted la {la} lb {lb}"
            );
        }
    }

    #[test]
    #[ignore = "exhaustive stress; run with `cargo test --release -- --ignored hgcd_massive`"]
    fn hgcd_massive_deterministic() {
        // Millions of deterministic pairs across every recursion regime and
        // shrunk base. Any mismatch against the binary-GCD oracle, or any tripped
        // correctness guard, fails loudly with the exact seed/index/sizes.
        let bases = [2usize, 3, 4, 5, 6, 8, 11, 16, 24, 37];
        let mut seed = 0xdead_0000_beef_0000u64;
        for (bi, &base) in bases.iter().enumerate() {
            for pass in 0..12u64 {
                seed = seed
                    .wrapping_mul(0x2545_f491_4f6c_dd1d)
                    .wrapping_add(0x9e37_79b9_7f4a_7c15)
                    ^ ((bi as u64) << 32)
                    ^ pass;
                hgcd_diff_pass(seed | 1, 20_000, base);
            }
        }
        // A smaller tier at the real 128-limb base to exercise the production
        // Lehmer-window base case at volume.
        let mut next = hgcd_rng(0x1357_9bdf_2468_ace0);
        for _ in 0..400 {
            let la = HGCD_RECURSION_BASE_LIMBS - 4 + (next() as usize % 260);
            let lb = HGCD_RECURSION_BASE_LIMBS - 4 + (next() as usize % 260);
            let m = hgcd_build(la, &mut next);
            let n = hgcd_build(lb, &mut next);
            if m.is_zero() || n.is_zero() {
                continue;
            }
            assert_eq!(
                gcd_via_hgcd_floor(&m, &n, 3),
                m.gcd_lehmer(&n),
                "real-base la {la} lb {lb}"
            );
        }
    }

    #[test]
    fn hgcd_matrix_invariant() {
        // `hgcd(m, n)` must return an ordered non-negative pair `(va, vb) =
        // R·(m, n)` straddling the half-length point, with `R` unimodular and
        // gcd-preserving. Sizes span one to several recursion levels (base is
        // 128 limbs).
        let mut next = hgcd_rng(0x9e37_79b9_7f4a_7c15);
        for i in 0..400 {
            // Most in the 1–3 recursion-level range; a few much larger to drive
            // deep recursion. `gcd_lehmer` is the (fast, independently tested)
            // oracle.
            let span = if i % 40 == 0 { 1500 } else { 320 };
            let la = HGCD_RECURSION_BASE_LIMBS - 40 + (next() as usize % span);
            let lb = HGCD_RECURSION_BASE_LIMBS - 40 + (next() as usize % span);
            let mut m = hgcd_build(la, &mut next);
            let mut n = hgcd_build(lb, &mut next);
            if m.is_zero() || n.is_zero() {
                continue;
            }
            if m.cmp_ref(&n) != Ordering::Greater {
                core::mem::swap(&mut m, &mut n);
            }
            if m.cmp_ref(&n) != Ordering::Greater {
                continue; // equal
            }
            let (r, va, vb) = hgcd(&m, &n);
            // R · (m, n) == (va, vb).
            let (cva, cvb) = r.mul_vec(&Int::from(m.clone()), &Int::from(n.clone()));
            assert_eq!(cva, Int::from(va.clone()), "R·(m,n) row0");
            assert_eq!(cvb, Int::from(vb.clone()), "R·(m,n) row1");
            // det R = ±1.
            let det = r.a.mul(&r.d).sub(&r.b.mul(&r.c));
            assert!(det.abs().is_one(), "det must be ±1");
            // Ordered pair, straddling h.
            let h = m.bit_len().div_ceil(2);
            assert!(va.cmp_ref(&vb) == Ordering::Greater, "va > vb");
            assert!(va.bit_len() > h, "va above the straddle point");
            assert!(vb.bit_len() <= h, "vb at/below the straddle point");
            // gcd preserved (against the independent binary/Lehmer oracles).
            assert_eq!(m.gcd_lehmer(&n), va.gcd_lehmer(&vb), "hgcd preserves gcd");
        }
    }

    #[test]
    fn hgcd_gcd_matches_reference() {
        // Thousands of random pairs across a wide size range. Small pairs go
        // through the dispatched `gcd`; every pair is also reduced with the
        // Half-GCD driver forced on (full recursion). All three — dispatched
        // `gcd`, forced Half-GCD, and the binary-GCD oracle — must agree
        // bit-for-bit.
        let mut next = hgcd_rng(0x2545_f491_4f6c_dd1d);
        for iter in 0..4000u32 {
            let scale = match next() % 4 {
                0 => 4,
                1 => 40,
                2 => 130,
                _ => 300,
            };
            let la = 1 + (next() as usize % scale);
            let lb = 1 + (next() as usize % scale);
            let a = hgcd_build(la, &mut next);
            let b = hgcd_build(lb, &mut next);
            let g_ref = if a.is_zero() {
                b.clone()
            } else if b.is_zero() {
                a.clone()
            } else {
                a.gcd_binary(&b)
            };
            assert_eq!(
                a.gcd(&b),
                g_ref,
                "iter {iter}: dispatched gcd la={la} lb={lb}"
            );
            assert_eq!(b.gcd(&a), g_ref, "iter {iter}: symmetry");
            assert_eq!(gcd_via_hgcd(&a, &b), g_ref, "iter {iter}: forced hgcd");
            assert_eq!(gcd_via_hgcd(&b, &a), g_ref, "iter {iter}: forced hgcd sym");
        }
    }

    #[test]
    fn hgcd_deep_recursion_matches_lehmer() {
        // Larger inputs (up to ~3000 limbs → several recursion levels), forced
        // through the Half-GCD driver, must match Lehmer. Includes pairs with a
        // planted common factor so the gcd is nontrivial.
        let mut next = hgcd_rng(0x0123_4567_89ab_cdef);
        for _ in 0..50 {
            let la = 200 + (next() as usize % 1900);
            let lb = 200 + (next() as usize % 1900);
            let a = hgcd_build(la, &mut next);
            let b = hgcd_build(lb, &mut next);
            if a.is_zero() || b.is_zero() {
                continue;
            }
            assert_eq!(gcd_via_hgcd(&a, &b), a.gcd_lehmer(&b), "coprime-ish");

            // Planted common factor g.
            let g = hgcd_build(1 + (next() as usize % 300), &mut next);
            if g.is_zero() {
                continue;
            }
            let ga = g.mul(&a);
            let gb = g.mul(&b);
            assert_eq!(gcd_via_hgcd(&ga, &gb), ga.gcd_lehmer(&gb), "planted factor");
        }
    }

    #[test]
    fn hgcd_dispatch_engages_above_threshold() {
        // Exercise the *public* `gcd` at the production threshold so the
        // dispatch actually runs the Half-GCD driver, and confirm it is
        // bit-identical to Lehmer.
        let mut next = hgcd_rng(0xfeed_face_dead_c0de);
        let n = HGCD_GCD_THRESHOLD + 40;
        for _ in 0..2 {
            let a = hgcd_build(n + (next() as usize % 64), &mut next);
            let b = hgcd_build(n + (next() as usize % 64), &mut next);
            assert!(a.limbs.len().min(b.limbs.len()) >= HGCD_GCD_THRESHOLD);
            assert_eq!(a.gcd(&b), a.gcd_lehmer(&b), "dispatched hgcd vs lehmer");
        }
    }

    #[test]
    fn hgcd_structured_cases() {
        let mut next = hgcd_rng(0xdead_beef_cafe_f00d);
        let big = |c: usize, f: &mut dyn FnMut() -> u64| hgcd_build(c, f);
        // Route through the forced Half-GCD driver so these exercise `hgcd`.
        let g = gcd_via_hgcd;

        // Zero / one edges.
        let x = big(400, &mut next);
        assert_eq!(g(&x, &Nat::zero()), x);
        assert_eq!(g(&Nat::zero(), &x), x);
        assert_eq!(g(&x, &Nat::one()), Nat::one());
        assert_eq!(g(&Nat::one(), &x), Nat::one());
        assert_eq!(g(&Nat::zero(), &Nat::zero()), Nat::zero());

        // Equal operands: gcd(x, x) == x.
        assert_eq!(g(&x, &x), x);

        // One divides the other: gcd(x, k·x) == x.
        let k = big(200, &mut next);
        let mult = x.mul(&k);
        assert_eq!(g(&x, &mult), x);
        assert_eq!(g(&mult, &x), x);

        // Consecutive Fibonacci numbers are coprime (worst case for Euclid).
        let (mut f0, mut f1) = (Nat::zero(), Nat::one());
        for _ in 0..40000 {
            let f2 = f0.add(&f1);
            f0 = core::mem::replace(&mut f1, f2);
        }
        let fib_next = f0.add(&f1);
        assert!(f1.limbs.len() > HGCD_RECURSION_BASE_LIMBS, "fibs recurse");
        assert_eq!(
            g(&f1, &fib_next),
            Nat::one(),
            "consecutive Fibonacci coprime"
        );

        // Powers of two: gcd(2^i, 2^j) == 2^min(i,j).
        let p_big = Nat::one().shl(90_000);
        let p_small = Nat::one().shl(30_000);
        assert_eq!(g(&p_big, &p_small), p_small.clone());
        assert_eq!(g(&p_small, &p_big), p_small);

        // Shared known factor: gcd(g·u, g·v) == g·gcd(u, v).
        for _ in 0..40 {
            let gg = big(1 + (next() as usize % 400), &mut next);
            let u = big(1 + (next() as usize % 500), &mut next);
            let v = big(1 + (next() as usize % 500), &mut next);
            if gg.is_zero() || u.is_zero() || v.is_zero() {
                continue;
            }
            let a = gg.mul(&u);
            let b = gg.mul(&v);
            let expect = gg.mul(&u.gcd_lehmer(&v));
            assert_eq!(g(&a, &b), expect, "gcd(g·u, g·v) == g·gcd(u,v)");
        }

        // Both even, both odd, mixed parity.
        for _ in 0..40 {
            let mut a = big(1 + (next() as usize % 400), &mut next);
            let mut b = big(1 + (next() as usize % 400), &mut next);
            if a.is_zero() || b.is_zero() {
                continue;
            }
            for (pa, pb) in [(false, false), (true, true), (false, true)] {
                let aa = if pa {
                    a.shl(1)
                } else {
                    a.shl(1).add(&Nat::one())
                };
                let bb = if pb {
                    b.shl(1)
                } else {
                    b.shl(1).add(&Nat::one())
                };
                assert_eq!(g(&aa, &bb), aa.gcd_lehmer(&bb), "parity case");
                a = aa;
                b = bb;
            }
        }
    }

    #[test]
    #[ignore = "timing benchmark; run with --release --ignored"]
    fn hgcd_crossover_bench() {
        use std::hint::black_box;
        use std::println;
        use std::time::Instant;
        let mk = |limbs: usize, seed: u64| {
            let mut s = seed;
            hgcd_build(limbs, &mut || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            })
        };
        println!(
            "{:>8} {:>13} {:>13} {:>9}",
            "limbs", "lehmer(ms)", "hgcd(ms)", "speedup"
        );
        for &limbs in &[
            64usize, 128, 256, 512, 1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384, 24576, 32768,
        ] {
            let a = mk(limbs, 0xA1 ^ limbs as u64);
            let b = mk(limbs, 0xB2 ^ limbs as u64);
            let reps = if limbs <= 64 {
                40
            } else if limbs <= 256 {
                10
            } else if limbs <= 1024 {
                3
            } else {
                1
            };
            let gl = a.gcd_lehmer(&b);
            let gh = a.gcd_hgcd(&b);
            assert_eq!(gl, gh, "mismatch at {limbs} limbs");
            let t0 = Instant::now();
            for _ in 0..reps {
                black_box(a.gcd_lehmer(&b));
            }
            let tl = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let t1 = Instant::now();
            for _ in 0..reps {
                black_box(a.gcd_hgcd(&b));
            }
            let th = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;
            println!("{limbs:>8} {tl:>13.3} {th:>13.3} {:>9.2}", tl / th);
        }
    }

    #[test]
    #[ignore = "timing benchmark; run with --release --ignored"]
    fn montgomery_vs_barrett_modint_bench() {
        use std::hint::black_box;
        use std::println;
        use std::time::Instant;
        // Deterministic random odd Nat of `bits` bits (top bit set, low bit set).
        let mk = |bits: usize, seed: u64| {
            let k = bits.div_ceil(LIMB_BITS as usize);
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let mut limbs = alloc::vec::Vec::with_capacity(k);
            for _ in 0..k {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                limbs.push(s as Limb);
            }
            let top = (bits - 1) % LIMB_BITS as usize;
            limbs[k - 1] |= 1 << top; // ensure exactly `bits` bits
            limbs[0] |= 1; // odd
            Nat::from_limbs(&limbs)
        };
        println!(
            "{:>6} {:>8} {:>12} {:>12} {:>9}   {:>12} {:>12} {:>9}",
            "bits",
            "limbs",
            "mul-bar(ns)",
            "mul-mont(ns)",
            "speedup",
            "pow-bar(ms)",
            "pow-mont(ms)",
            "speedup"
        );
        for &bits in &[64usize, 256, 1024, 4096] {
            let m = mk(bits, 0x11 ^ bits as u64);
            let a0 = mk(bits, 0x22 ^ bits as u64).div_rem(&m).unwrap().1;
            let b0 = mk(bits, 0x33 ^ bits as u64).div_rem(&m).unwrap().1;

            // --- mul: Barrett reduce vs resident Montgomery mul ---
            let recip = Reciprocal::new(&m);
            let ctx = MontCtx::new(&m);
            let am = ctx.to_mont(&a0);
            let bm = ctx.to_mont(&b0);
            let reps: usize = if bits <= 256 {
                200_000
            } else if bits <= 1024 {
                20_000
            } else {
                2_000
            };
            // Chain each product back in to defeat the optimizer.
            let t0 = Instant::now();
            let mut a = a0.clone();
            for _ in 0..reps {
                a = recip.reduce(&a.mul(&b0));
            }
            black_box(&a);
            let tb = t0.elapsed().as_secs_f64() * 1e9 / reps as f64;
            let t1 = Instant::now();
            let mut a = am.clone();
            for _ in 0..reps {
                a = ctx.mul(&a, &bm);
            }
            black_box(&a);
            let tm = t1.elapsed().as_secs_f64() * 1e9 / reps as f64;

            // --- pow: Barrett modpow vs Montgomery modpow (same odd modulus) ---
            let e = mk(bits, 0x44 ^ bits as u64);
            let preps: usize = if bits <= 256 {
                2000
            } else if bits <= 1024 {
                200
            } else {
                20
            };
            let rb = a0.modpow_barrett(&e, &m);
            let rm = a0.modpow_montgomery(&e, &m);
            assert_eq!(rb, rm, "pow mismatch at {bits} bits");
            let t2 = Instant::now();
            for _ in 0..preps {
                black_box(a0.modpow_barrett(&e, &m));
            }
            let pb = t2.elapsed().as_secs_f64() * 1e3 / preps as f64;
            let t3 = Instant::now();
            for _ in 0..preps {
                black_box(a0.modpow_montgomery(&e, &m));
            }
            let pm = t3.elapsed().as_secs_f64() * 1e3 / preps as f64;

            println!(
                "{:>6} {:>8} {:>12.2} {:>12.2} {:>9.2}   {:>12.4} {:>12.4} {:>9.2}",
                bits,
                m.limbs.len(),
                tb,
                tm,
                tb / tm,
                pb,
                pm,
                pb / pm
            );
        }
    }

    // --- Half-GCD extended-GCD cofactor driver (backs `Int::modinv`) --------

    /// `Int::extended_gcd` (Lehmer / Half-GCD plus normalization) must return
    /// exactly the plain extended-Euclid triple, for every sign combination,
    /// shared factors, unbalanced sizes, and sizes on both sides of the
    /// Half-GCD threshold.
    #[test]
    fn extended_gcd_matches_euclid() {
        let mut st = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let check = |a: &Int, b: &Int| {
            for (x, y) in [
                (a.clone(), b.clone()),
                (a.neg(), b.clone()),
                (a.clone(), b.neg()),
                (a.neg(), b.neg()),
                (b.clone(), a.clone()),
            ] {
                assert_eq!(
                    x.extended_gcd(&y),
                    x.extended_gcd_euclid(&y),
                    "extended_gcd({x}, {y})"
                );
            }
        };
        let small = [0i64, 1, 2, 3, 4, 5, 6, 7, 10, 12, 15, 46, 240, 1 << 40];
        for &a in &small {
            for &b in &small {
                check(&Int::from(a), &Int::from(b));
            }
        }
        let sizes = [1usize, 2, 3, 5, 9, 17, 33, 40, 70];
        for &la in &sizes {
            for &lb in &sizes {
                for _ in 0..3 {
                    let a = Int::from(hgcd_build(la, &mut next));
                    let b = Int::from(hgcd_build(lb, &mut next));
                    check(&a, &b);
                    // Shared factor (gcd > 1), including |b| = 2g / |a| = 2g.
                    let f = Int::from(hgcd_build(1 + la / 3, &mut next));
                    check(&a.mul(&f), &b.mul(&f));
                    check(&a.mul(&f), &f.mul_2k(1));
                    check(&a.mul(&f), &f);
                }
            }
        }
        // Past the Half-GCD threshold.
        let n = HGCD_MODINV_THRESHOLD + 4;
        let a = Int::from(hgcd_build(n + 9, &mut next));
        let b = Int::from(hgcd_build(n, &mut next));
        let f = Int::from(hgcd_build(3, &mut next));
        assert_eq!(a.extended_gcd(&b), a.extended_gcd_euclid(&b));
        let (af, bf) = (a.mul(&f), b.neg().mul(&f));
        assert_eq!(af.extended_gcd(&bf), af.extended_gcd_euclid(&bf));
    }

    /// Reference modular inverse of `a` mod `m` (`m > 1`) via plain extended
    /// Euclid on `Int`, or `None` when not coprime. Independent of the Half-GCD
    /// path — the oracle the fast `modinv` must match exactly.
    fn modinv_ref(a: &Nat, m: &Nat) -> Option<Nat> {
        let (g, x, _) = Int::from(a.clone()).extended_gcd_euclid(&Int::from(m.clone()));
        if !g.is_one() {
            return None;
        }
        Some(x.rem_euclid(&Int::from(m.clone())).magnitude())
    }

    /// `x·a ≡ g (mod b)`, or `x·a = g` exactly when `b = 0`.
    fn bezout_cofactor_ok(a: &Nat, b: &Nat, g: &Nat, x: &Int) -> bool {
        let d = x.mul(&Int::from(a.clone())).sub(&Int::from(g.clone()));
        if b.is_zero() {
            d.is_zero()
        } else {
            d.rem_euclid(&Int::from(b.clone())).is_zero()
        }
    }

    /// Half-GCD extended-GCD driver invariants, on non-negative operands, with
    /// the recursion base shrunk so tiny inputs still drive the full recursion.
    /// The Bézout cofactor need not be Euclid's minimal one (the Half-GCD matrix
    /// can differ by a kernel term when `gcd > 1`), so this checks what the
    /// driver *guarantees*: `g = gcd`, the exact identity `g = a·x + b·y`, and —
    /// the property `modinv` relies on — `x mod b` reproduces Euclid's inverse.
    fn extgcd_diff_pass(seed: u64, count: usize, base: usize) {
        hgcd_with_test_base(base, || {
            let mut next = hgcd_rng(seed);
            for iter in 0..count {
                // Wide size range: several recursion levels above the shrunk base.
                let la = 1 + (next() as usize % (10 * base + 5));
                let lb = 1 + (next() as usize % (10 * base + 5));
                let mut a = hgcd_build(la, &mut next);
                let mut b = hgcd_build(lb, &mut next);
                // Occasionally plant a shared factor so gcd > 1 is exercised
                // (exactly the case where the cofactor is non-minimal).
                if next() & 3 == 0 {
                    let g = hgcd_build(1 + next() as usize % (base + 1), &mut next);
                    a = a.mul(&g);
                    b = b.mul(&g);
                }
                let (g, x) = a
                    .extgcd_hgcd_floor(&b, 3)
                    .expect("extgcd guard tripped — cofactor invariant violated");
                // Bézout congruence x·a ≡ g (mod b) (equality when b = 0).
                assert!(bezout_cofactor_ok(&a, &b, &g, &x), "iter {iter}: Bézout");
                // g is the true gcd (independent Lehmer oracle).
                let expect_g = if b.is_zero() {
                    a.clone()
                } else {
                    a.gcd_lehmer(&b)
                };
                assert_eq!(g, expect_g, "iter {iter}: g == gcd");
                // Inverse property: x mod b matches plain-Euclid's inverse.
                if g.is_one() && !b.is_zero() {
                    let inv = x.rem_euclid(&Int::from(b.clone())).magnitude();
                    assert_eq!(
                        Some(inv),
                        modinv_ref(&a, &b),
                        "iter {iter}: x mod b == Euclid inverse"
                    );
                }
            }
        });
    }

    #[test]
    fn extgcd_hgcd_driver_forced() {
        // A few shrunk bases so the driver's straddle / make-valid / apply-split
        // and the residual finishing loop are all exercised at volume, spanning
        // several recursion levels (up to ~85 limbs at base 8).
        for (seed, base) in [
            (0x51ed_5aab_1234_0001u64, 2usize),
            (0x51ed_5aab_1234_0002, 3),
            (0x51ed_5aab_1234_0003, 5),
            (0x51ed_5aab_1234_0004, 8),
        ] {
            extgcd_diff_pass(seed, 1500, base);
        }
    }

    #[test]
    fn extgcd_structured_cases_forced() {
        hgcd_with_test_base(3, || {
            let mut next = hgcd_rng(0x00c0_ffee_5eed_1010);
            let big = |c: usize, f: &mut dyn FnMut() -> u64| hgcd_build(c, f);
            // Every structured pair must satisfy the driver's guarantees.
            let check = |a: &Nat, b: &Nat| {
                let (g, x) = a.extgcd_hgcd_floor(b, 3).expect("guard");
                assert!(bezout_cofactor_ok(a, b, &g, &x), "Bézout");
                let expect_g = if b.is_zero() {
                    a.clone()
                } else if a.is_zero() {
                    b.clone()
                } else {
                    a.gcd_lehmer(b)
                };
                assert_eq!(g, expect_g, "g == gcd");
                if g.is_one() && !b.is_zero() {
                    let inv = x.rem_euclid(&Int::from(b.clone())).magnitude();
                    assert_eq!(Some(inv), modinv_ref(a, b), "x mod b == inverse");
                }
            };
            let x = big(200, &mut next);

            // Zero / one edges.
            check(&x, &Nat::zero());
            check(&Nat::zero(), &x);
            check(&Nat::zero(), &Nat::zero());
            check(&x, &Nat::one());
            check(&Nat::one(), &x);
            // Equal operands.
            check(&x, &x);
            // One divides the other.
            let mult = x.mul(&big(120, &mut next));
            check(&x, &mult);
            check(&mult, &x);
            // Consecutive Fibonacci (Euclid worst case, many quotients = 1).
            let (mut f0, mut f1) = (Nat::zero(), Nat::one());
            for _ in 0..600 {
                let f2 = f0.add(&f1);
                f0 = core::mem::replace(&mut f1, f2);
            }
            check(&f1, &f0.add(&f1));
            check(&f0, &f1);
            // Powers of two.
            check(&Nat::one().shl(4000), &Nat::one().shl(1500));
            check(&Nat::one().shl(1500), &Nat::one().shl(4000));
            // Coprime constructed pair (odd·1 and a shifted neighbor).
            let c = big(180, &mut next).shl(1).add(&Nat::one());
            check(&c, &c.add(&Nat::one()));
        });
    }

    #[test]
    fn modinv_dispatch_matches_euclid() {
        // Exercise the *public* `Int::modinv` at the production threshold so real
        // dispatch runs the Half-GCD driver; must be bit-identical to Euclid, for
        // every sign combination of `a` and `modulus`.
        let mut next = hgcd_rng(0x0dd1_1235_8b00_1123);
        let n = HGCD_MODINV_THRESHOLD + 8;
        for _ in 0..4 {
            // Odd modulus keeps a good chance of coprimality; also test a
            // deliberately non-coprime pair (share a factor of 2).
            let m = hgcd_build(n + (next() as usize % 16), &mut next)
                .shl(1)
                .add(&Nat::one());
            let a = hgcd_build(n + (next() as usize % 16), &mut next);
            assert!(a.limb_count().min(m.limb_count()) >= HGCD_MODINV_THRESHOLD);
            for (sa, sm) in [
                (Sign::Positive, Sign::Positive),
                (Sign::Negative, Sign::Positive),
                (Sign::Positive, Sign::Negative),
                (Sign::Negative, Sign::Negative),
            ] {
                let ai = Int::from_sign_magnitude(sa, a.clone());
                let mi = Int::from_sign_magnitude(sm, m.clone());
                // Euclid reference inverse (bypasses the fast path).
                let (g, x, _) = ai.extended_gcd_euclid(&mi);
                let expect = g.is_one().then(|| x.rem_euclid(&mi));
                assert_eq!(ai.modinv(&mi), expect, "dispatched modinv vs Euclid");
                if let Some(inv) = ai.modinv(&mi) {
                    // a·inv ≡ 1 (mod m).
                    assert!(ai.mul(&inv).sub(&Int::ONE).rem_euclid(&mi).is_zero());
                }
            }
        }
    }

    #[test]
    #[ignore = "timing benchmark; run with --release --ignored"]
    fn modinv_crossover_bench() {
        use std::hint::black_box;
        use std::println;
        use std::time::Instant;
        let mk = |limbs: usize, seed: u64| {
            let mut s = seed;
            hgcd_build(limbs, &mut || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            })
        };
        // Plain extended Euclid, the pre-existing modinv path, as the baseline.
        let euclid_modinv = |a: &Int, m: &Int| -> Option<Int> {
            let (g, x, _) = a.extended_gcd_euclid(m);
            g.is_one().then(|| x.rem_euclid(m))
        };
        println!(
            "{:>8} {:>13} {:>13} {:>9}   (modinv)",
            "limbs", "euclid(ms)", "hgcd(ms)", "speedup"
        );
        for &limbs in &[16usize, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192] {
            // Odd modulus so a random `a` is (almost surely) invertible.
            let m = Int::from(mk(limbs, 0xB2 ^ limbs as u64).shl(1).add(&Nat::one()));
            let a = Int::from(mk(limbs, 0xA1 ^ limbs as u64));
            let reps = if limbs <= 128 {
                20
            } else if limbs <= 512 {
                6
            } else if limbs <= 2048 {
                2
            } else {
                1
            };
            assert_eq!(euclid_modinv(&a, &m), a.modinv(&m), "mismatch at {limbs}");
            let t0 = Instant::now();
            for _ in 0..reps {
                black_box(euclid_modinv(&a, &m));
            }
            let te = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let t1 = Instant::now();
            for _ in 0..reps {
                black_box(a.modinv(&m));
            }
            let th = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;
            println!("{limbs:>8} {te:>13.3} {th:>13.3} {:>9.2}", te / th);
        }
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
        // Subtractive middle term: edge-heavy limbs make |a0 − a1| hit zero,
        // equal halves and both sign combinations, over random shapes from
        // the threshold up (including very unbalanced ones).
        let mut state = 0x5eed_ca7a_0000_0042u64;
        for round in 0..400usize {
            let x = KARATSUBA_THRESHOLD + round * 13 % 170;
            let y = KARATSUBA_THRESHOLD + round * 29 % 420;
            let mut al = edgy_limbs(&mut state, x);
            let mut bl = edgy_limbs(&mut state, y);
            if round % 5 == 0 {
                // Equal halves: a0 == a1 exactly.
                let h = x / 2;
                let (lo, hi) = al.split_at_mut(h);
                hi[..h].copy_from_slice(lo);
            }
            if round % 7 == 0 {
                bl.iter_mut().for_each(|l| *l = Limb::MAX);
            }
            let (a, b) = (Nat::from_limbs(&al), Nat::from_limbs(&bl));
            assert_eq!(a.mul_karatsuba(&b), a.mul_schoolbook(&b), "edgy {x}x{y}");
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

    /// Limb generator skewed towards the edge values (0, 1, 2^63, 2^64 − 1)
    /// that exercise the rare quotient-correction branches.
    fn edgy_limbs(state: &mut u64, cnt: usize) -> Vec<Limb> {
        let mut next = || {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };
        (0..cnt)
            .map(|_| match next() % 8 {
                0 => 0,
                1 => Limb::MAX,
                2 => 1 << 63,
                3 => Limb::MAX - 1,
                4 => 1,
                _ => next(),
            })
            .collect()
    }

    #[test]
    fn schoolbook_division_differential() {
        // The 3-by-2 reciprocal schoolbook division against the bitwise
        // reference (small sizes) and the exact identity a = q·b + r, r < b
        // (all sizes, including both sides of the BZ dispatch), over random
        // and edge-heavy limbs.
        let mut state = 0x0bad_5eed_1234_5678u64;
        for round in 0..3000 {
            let nb = 2 + (round % 37);
            let na = nb + (round * 7 % 45);
            let mut bl = edgy_limbs(&mut state, nb);
            if bl[nb - 1] == 0 {
                bl[nb - 1] = 1 + (round as Limb);
            }
            let b = Nat::from_limbs(&bl);
            let a = Nat::from_limbs(&edgy_limbs(&mut state, na));
            let (q, r) = a.div_rem(&b).unwrap();
            assert!(r.cmp_ref(&b) == Ordering::Less, "r >= b at {na}/{nb}");
            assert_eq!(q.mul(&b).add(&r), a, "identity at {na}/{nb}");
            if na <= 5 {
                assert_eq!((q, r), div_rem_binary(&a, &b));
            }
        }
        // Divisors of the form (d1, d0) = top limbs of the window, which hit
        // the q = β − 1 shortcut: a = b·(β^k − 1) + small.
        for nb in 2..12 {
            let b = Nat::from_limbs(&edgy_limbs(&mut state, nb))
                .add(&Nat::one().shl(64 * nb as u64 - 1));
            for k in 1..6u64 {
                let a = b
                    .mul(&Nat::one().shl(64 * k).checked_sub(&Nat::one()).unwrap())
                    .add(&Nat::from_u64(k));
                let (q, r) = a.div_rem(&b).unwrap();
                assert_eq!(q.mul(&b).add(&r), a);
                assert!(r.cmp_ref(&b) == Ordering::Less);
            }
        }
        // 3-by-2 primitive against u128 arithmetic on a reduced range.
        for _ in 0..20000 {
            let l = edgy_limbs(&mut state, 5);
            let d1 = l[0] | (1 << 63);
            let d0 = l[1];
            let v = recip_3by2(d1, d0);
            let d = ((d1 as u128) << 64) | d0 as u128;
            // (u2, u1) < (d1, d0).
            let top = (((l[2] as u128) << 64) | l[3] as u128) % d;
            let (u2, u1, u0) = ((top >> 64) as Limb, top as Limb, l[4]);
            let (q, r) = div_3by2(u2, u1, u0, d1, d0, v);
            // Check q·d + r == (u2, u1, u0) as a 192-bit value, r < d.
            assert!(r < d);
            let lo = (q as u128) * (d0 as u128) + (r as u64) as u128;
            let mid = (q as u128) * (d1 as u128) + (lo >> 64) + (r >> 64);
            assert_eq!(lo as Limb, u0);
            assert_eq!(mid as Limb, u1);
            assert_eq!((mid >> 64) as Limb, u2);
        }
    }

    #[test]
    fn radix_string_differential() {
        // The chunked / divide-and-conquer radix output against a naive
        // digit-at-a-time conversion, for every radix and sizes across the
        // recursion threshold (edge-heavy limbs give long zero/max runs).
        let naive = |x: &Nat, radix: u32| -> String {
            let mut x = x.clone();
            let mut d = Vec::new();
            while !x.is_zero() {
                let (q, r) = x.divmod_small(radix as Limb);
                d.push(digit_char(r as u32, radix));
                x = q;
            }
            d.reverse();
            String::from_utf8(d).unwrap()
        };
        let mut state = 0x7ad1_c0de_0000_0001u64;
        for radix in 2..=36u32 {
            for cnt in (0..4 * RADIX_RECURSION_LIMBS + 3).step_by(if radix == 10 { 1 } else { 5 }) {
                let x = Nat::from_limbs(&edgy_limbs(&mut state, cnt));
                let mut s = String::new();
                x.write_radix(&mut s, radix).unwrap();
                let want = naive(&x, radix);
                assert_eq!(
                    s,
                    if want.is_empty() { "0".into() } else { want },
                    "radix {radix}"
                );
            }
        }
        // Parsing round-trips across the Horner/recursive split (512 chunks
        // ≈ 9.7k decimal digits), with and without leading zeros.
        for (radix, cnt) in [
            (10u32, 3usize),
            (10, 600),
            (10, 1100),
            (10, 2300),
            (7, 700),
            (36, 900),
            (62, 900),
        ] {
            let x = Nat::from_limbs(&edgy_limbs(&mut state, cnt));
            let s = x.to_radix_string(radix);
            assert_eq!(
                parse_radix(&s, radix).unwrap(),
                x,
                "radix {radix} cnt {cnt}"
            );
            let padded = alloc::format!("0000000000000000000000{s}");
            assert_eq!(parse_radix(&padded, radix).unwrap(), x);
        }
        // Powers of the radix (and their neighbours) stress the zero padding.
        for k in [19u32, 38, 190, 400, 1000] {
            let p = Nat::from_u64(10).pow(u64::from(k));
            for x in [
                p.checked_sub(&Nat::one()).unwrap(),
                p.clone(),
                p.add(&Nat::one()),
            ] {
                assert_eq!(x.to_radix_string(10), naive(&x, 10));
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

    /// End-to-end correctness of the public `factorize` on balanced semiprimes
    /// across the scaled SIQS range (40–56 digits): every returned list is a
    /// complete, sorted prime factorization whose product is the input. Routes
    /// through `split_composite`'s SIQS gate (double large primes + block
    /// Lanczos). Heavy — run in release:
    /// `cargo test --release factorize_balanced_semiprimes_scaled -- --ignored`.
    #[test]
    #[ignore = "heavy: release-only end-to-end factorization batch"]
    fn factorize_balanced_semiprimes_scaled() {
        // Smallest prime ≥ 10^(d/2) + offset, over big integers.
        fn prime_from(base: &Nat, offset: u64) -> Nat {
            let mut c = base.add(&Nat::from_u64(offset));
            if c.is_even() {
                c = c.add(&Nat::one());
            }
            while !c.is_prime_bpsw() {
                c = c.add(&Nat::from_u64(2));
            }
            c
        }
        for d in [40u32, 44, 48, 52, 56] {
            let base = Nat::from_u64(10).pow(u64::from(d / 2));
            for seed in 0..2u64 {
                let p = prime_from(&base, seed * 2 + 1);
                let q = prime_from(&base, seed * 1000 + 54321);
                if p == q {
                    continue;
                }
                let n = p.mul(&q);
                let factors = n.factorize();
                // Product of factors equals n, every factor is prime, sorted.
                let mut prod = Nat::one();
                for f in &factors {
                    assert!(f.is_prime_bpsw(), "d={d} seed={seed}: composite factor");
                    prod = prod.mul(f);
                }
                assert_eq!(prod, n, "d={d} seed={seed}: product mismatch");
                for w in factors.windows(2) {
                    assert!(w[0].cmp_ref(&w[1]) != Ordering::Greater, "factors unsorted");
                }
                // A balanced semiprime factors into exactly its two primes.
                assert_eq!(factors.len(), 2, "d={d} seed={seed}: not a 2-prime split");
            }
        }
    }
}
