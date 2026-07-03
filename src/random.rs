//! Random-number generation for [`Nat`] and [`Int`].
//!
//! The core exposes an in-house [`RandomSource`] trait so callers can plug in
//! any byte source with no dependency. Enabling the `rand` feature additionally
//! provides a blanket impl of [`RandomSource`] for every [`rand_core::RngCore`],
//! bridging to the `rand` ecosystem.

use crate::int::Int;
use crate::nat::Nat;

/// A source of random bytes.
///
/// Implement this for any RNG to drive [`Nat`]/[`Int`] generation. With the
/// `rand` feature, every `rand_core::RngCore` implements it automatically.
pub trait RandomSource {
    /// Fills `dest` entirely with random bytes.
    fn fill_bytes(&mut self, dest: &mut [u8]);
}

#[cfg(feature = "rand")]
impl<R: rand_core::RngCore> RandomSource for R {
    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        rand_core::RngCore::fill_bytes(self, dest);
    }
}

impl Nat {
    /// Generates a uniform random natural with at most `bits` bits (i.e. in
    /// `[0, 2^bits)`).
    pub fn random_bits(bits: u64, src: &mut impl RandomSource) -> Nat {
        if bits == 0 {
            return Nat::zero();
        }
        let nbytes = bits.div_ceil(8) as usize;
        let mut buf = alloc::vec![0u8; nbytes];
        src.fill_bytes(&mut buf);
        let rem = (bits % 8) as u32;
        if rem != 0 {
            buf[nbytes - 1] &= (1u8 << rem) - 1;
        }
        Nat::from_bytes_le(&buf)
    }

    /// Generates a uniform random natural in `[0, bound)` by rejection sampling,
    /// or `None` if `bound` is zero.
    pub fn random_below(bound: &Nat, src: &mut impl RandomSource) -> Option<Nat> {
        if bound.is_zero() {
            return None;
        }
        let bits = bound.bit_len();
        loop {
            let candidate = Nat::random_bits(bits, src);
            if &candidate < bound {
                return Some(candidate);
            }
        }
    }
}

impl Int {
    /// Generates a uniform random non-negative integer in `[0, bound)`, or
    /// `None` if `bound <= 0`.
    pub fn random_below(bound: &Int, src: &mut impl RandomSource) -> Option<Int> {
        if !bound.is_positive() {
            return None;
        }
        let magnitude = bound.magnitude();
        Nat::random_below(&magnitude, src).map(Int::from)
    }
}
