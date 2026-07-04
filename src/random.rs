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

/// A small deterministic, seedable PRNG ([SplitMix64]) implementing
/// [`RandomSource`]. It needs no host entropy, so it works on bare `no_std` and
/// makes generation reproducible from a seed — e.g. `Int::random_prime(bits,
/// &mut SeedRng::new(seed))`. **Not** cryptographically secure; use a real CSPRNG
/// (via the `rand` feature) for security-sensitive work.
///
/// [SplitMix64]: https://prng.di.unimi.it/splitmix64.c
#[derive(Clone, Debug)]
pub struct SeedRng(u64);

impl SeedRng {
    /// Creates a PRNG from a 64-bit seed. Every seed yields a fixed stream.
    #[inline]
    pub fn new(seed: u64) -> SeedRng {
        SeedRng(seed)
    }

    /// Returns the next 64-bit output and advances the state.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl RandomSource for SeedRng {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
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
