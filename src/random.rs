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

/// The 32-bit Mersenne Twister MT19937 (Matsumoto & Nishimura, 1998),
/// implementing [`RandomSource`]: a seedable, reproducible generator with
/// period `2^19937 − 1`. **Not** cryptographically secure.
///
/// Seeding follows the reference implementation (`init_genrand` for a 32-bit
/// seed, `init_by_array` for a key), so streams match other faithful MT19937
/// implementations such as C++'s `std::mt19937` or Python's `random` module.
/// For GMP's `gmp_randinit_mt` streams use [`MersenneTwister::gmp_seeded`]
/// instead. Combine it with [`Nat::random_bits`]/[`Int::random_bits`]
/// (`mpz_urandomb`-like) and [`Nat::random_below`]/[`Int::random_below`]
/// (`mpz_urandomm`-like).
#[derive(Clone)]
pub struct MersenneTwister {
    state: [u32; MT_N],
    index: usize,
}

const MT_N: usize = 624;
const MT_M: usize = 397;

impl MersenneTwister {
    /// Creates a generator from a 32-bit seed (the reference `init_genrand`).
    pub fn new(seed: u32) -> MersenneTwister {
        let mut state = [0u32; MT_N];
        state[0] = seed;
        for i in 1..MT_N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        MersenneTwister { state, index: MT_N }
    }

    /// Creates a generator from a key of 32-bit words (the reference
    /// `init_by_array`). An empty key is treated as `[0]`.
    pub fn from_key(key: &[u32]) -> MersenneTwister {
        let key = if key.is_empty() { &[0][..] } else { key };
        let mut mt = MersenneTwister::new(19_650_218);
        let s = &mut mt.state;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..MT_N.max(key.len()) {
            let prev = s[i - 1];
            s[i] = (s[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                s[0] = s[MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..MT_N - 1 {
            let prev = s[i - 1];
            s[i] =
                (s[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                s[0] = s[MT_N - 1];
                i = 1;
            }
        }
        s[0] = 0x8000_0000; // guarantees a non-zero state
        mt
    }

    /// Creates a generator keyed by an arbitrary-size seed: the 32-bit words
    /// of `|seed|`, least significant first, through [`MersenneTwister::from_key`]
    /// (zero becomes the key `[0]`).
    pub fn from_seed(seed: &Int) -> MersenneTwister {
        let key: alloc::vec::Vec<u32> = seed
            .limbs()
            .iter()
            .flat_map(|&l| [l as u32, (l >> 32) as u32])
            .collect();
        let used = key.iter().rposition(|&w| w != 0).map_or(0, |p| p + 1);
        MersenneTwister::from_key(&key[..used])
    }

    /// Creates a generator seeded the way GMP's `gmp_randseed` seeds
    /// `gmp_randinit_mt`, so its outputs, and [`Nat::random_bits`] /
    /// [`Int::random_bits`] drawn from it, reproduce `mpz_urandomb` for the
    /// same sequence of request sizes. (Bounded draws such as `mpz_urandomm`
    /// use their own selection algorithm and are not reproduced.)
    ///
    /// The seed is expanded to `(seed mod (2^19937 − 20027) + 2)^0x40118124`
    /// with a folding reduction modulo `2^19937 − 20023`; bit 19936 goes to the
    /// top of state word 0 and bits 0–19935 fill words 1–623. Three twists and
    /// a start index of 128 then stand in for GMP's 2000-output warm-up.
    ///
    /// An unseeded `gmp_randinit_mt` starts in the state of seed `5489`.
    ///
    /// This follows GMP's observed behaviour, not a documented GMP guarantee.
    pub fn gmp_seeded(seed: &Int) -> MersenneTwister {
        const BITS: u64 = 19937;
        let modulus = Int::ONE.mul_2k(BITS).sub(&Int::from_u64(20027));
        let base = seed
            .rem_euclid(&modulus)
            .add(&Int::from_u64(2))
            .into_magnitude();
        // Fold `r` below 2^19937 via 2^19937 ≡ 20023, with no final
        // subtraction of the modulus (the representative matters).
        let fold = |mut r: Nat| {
            while r.bit_len() > BITS {
                r = r
                    .low_bits(BITS)
                    .add(&r.shr(BITS).mul(&Nat::from_u64(20023)));
            }
            r
        };
        // Left-to-right square-and-multiply, skipping the leading one bit.
        let exp: u32 = 0x4011_8124;
        let mut r = base.clone();
        for i in (0..exp.ilog2()).rev() {
            r = fold(r.square());
            if (exp >> i) & 1 == 1 {
                r = fold(r.mul(&base));
            }
        }
        let mut state = [0u32; MT_N];
        if r.bit(BITS - 1) {
            state[0] = 0x8000_0000;
        }
        let low = r.low_bits(BITS - 1);
        for (i, &limb) in low.as_limbs().iter().enumerate() {
            state[1 + 2 * i] = limb as u32;
            if 2 + 2 * i < MT_N {
                state[2 + 2 * i] = (limb >> 32) as u32;
            }
        }
        let mut mt = MersenneTwister { state, index: 0 };
        for _ in 0..3 {
            mt.twist();
        }
        mt.index = 128;
        mt
    }

    /// Regenerates the whole state block (the "twist").
    fn twist(&mut self) {
        let s = &mut self.state;
        for i in 0..MT_N {
            let y = (s[i] & 0x8000_0000) | (s[(i + 1) % MT_N] & 0x7fff_ffff);
            let mag = if y & 1 == 1 { 0x9908_b0df } else { 0 };
            s[i] = s[(i + MT_M) % MT_N] ^ (y >> 1) ^ mag;
        }
        self.index = 0;
    }

    /// Returns the next 32-bit output (the reference `genrand_int32`).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

impl core::fmt::Debug for MersenneTwister {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MersenneTwister")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl RandomSource for MersenneTwister {
    /// Fills `dest` with successive 32-bit outputs in little-endian byte order.
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(4) {
            let bytes = self.next_u32().to_le_bytes();
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
    /// Generates a uniform random non-negative integer in `[0, 2^bits)` (like
    /// GMP's `mpz_urandomb`).
    pub fn random_bits(bits: u64, src: &mut impl RandomSource) -> Int {
        Int::from(Nat::random_bits(bits, src))
    }

    /// Generates a uniform random non-negative integer in `[0, bound)` (like
    /// GMP's `mpz_urandomm`), or `None` if `bound <= 0`.
    pub fn random_below(bound: &Int, src: &mut impl RandomSource) -> Option<Int> {
        if !bound.is_positive() {
            return None;
        }
        let magnitude = bound.magnitude();
        Nat::random_below(&magnitude, src).map(Int::from)
    }
}
