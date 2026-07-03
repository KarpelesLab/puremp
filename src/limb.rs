//! Fixed-width limb primitives: the machine-word building blocks every
//! arbitrary-precision algorithm is assembled from.
//!
//! A [`Limb`] is a single 64-bit word; products and carries are computed in a
//! 128-bit [`DLimb`] so the whole crate stays in safe, portable Rust with no
//! inline assembly or intrinsics. On a 32-bit host the `u128` operations lower
//! to compiler-provided helpers — correct, if slower.

/// A single machine word. Bignums are little-endian sequences of these.
pub(crate) type Limb = u64;

/// Double-width accumulator for limb products and carries.
pub(crate) type DLimb = u128;

/// Number of value bits in a [`Limb`].
pub(crate) const LIMB_BITS: u32 = Limb::BITS;

/// `a + b + carry_in`, returning `(sum, carry_out)` with `carry_out ∈ {0, 1}`.
#[inline]
pub(crate) const fn adc(a: Limb, b: Limb, carry_in: Limb) -> (Limb, Limb) {
    let t = a as DLimb + b as DLimb + carry_in as DLimb;
    (t as Limb, (t >> LIMB_BITS) as Limb)
}

/// `a - b - borrow_in`, returning `(diff, borrow_out)` with `borrow_out ∈ {0, 1}`.
#[inline]
pub(crate) const fn sbb(a: Limb, b: Limb, borrow_in: Limb) -> (Limb, Limb) {
    // On underflow the 128-bit wrapping difference sets every bit above the low
    // limb, so bit `LIMB_BITS` is exactly the borrow-out.
    let t = (a as DLimb)
        .wrapping_sub(b as DLimb)
        .wrapping_sub(borrow_in as DLimb);
    (t as Limb, ((t >> LIMB_BITS) as Limb) & 1)
}

/// Fused multiply-add-carry: `acc + a·b + carry_in`, returning `(lo, hi)` where
/// `lo` is the low limb and `hi` the carry into the next limb.
///
/// The maximum value is `(2^64−1) + (2^64−1)² + (2^64−1) = 2^128−1`, so this can
/// never overflow a [`DLimb`].
#[inline]
pub(crate) const fn mac(acc: Limb, a: Limb, b: Limb, carry_in: Limb) -> (Limb, Limb) {
    let t = acc as DLimb + a as DLimb * b as DLimb + carry_in as DLimb;
    (t as Limb, (t >> LIMB_BITS) as Limb)
}
