//! [`RadixConverter`]: radix conversion with cached power ladders, for
//! printing or parsing many numbers of similar size in one radix.

use alloc::vec::Vec;
use core::fmt;

use crate::error::Result;
use crate::nat::{
    Nat, PARSE_RECURSION_CHUNKS, PreInv, RADIX_RECURSION_LIMBS, RadixPow, chunks_recursive,
    horner_base_chunks, nat_from_base_chunks, pack_radix_chunks, radix_chunk, simple_radix_string,
    to_radix_recursive,
};

/// Ladder entries of at least this many limbs also cache a Newton reciprocal.
const PREINV_LIMBS: usize = 32;

/// Converts naturals to and from ASCII digits in a fixed radix (2–62), reusing
/// precomputed powers of the radix across calls — a "fast print / parse mode".
///
/// [`Nat::write_ascii_radix`] and [`Nat::from_ascii_radix`] rebuild the power
/// ladder (`radix^(2^k)`, plus Newton reciprocals of its large entries for
/// printing) on every call; for numbers of a few thousand digits that is a
/// sizeable share of the cost. A converter builds them once for values up to
/// `max_bits` bits; larger values still convert correctly, through the
/// uncached path. Digits use the same alphabet as [`Nat::write_radix`].
///
/// ```
/// use puremp::{Nat, RadixConverter};
///
/// let conv = RadixConverter::new(10, 4096);
/// let x = Nat::pow10(1000).add_u64(7);
/// let digits = conv.to_ascii(&x);
/// assert_eq!(digits.len(), 1001);
/// assert_eq!(conv.parse(&digits).unwrap(), x);
/// ```
pub struct RadixConverter {
    radix: u32,
    max_bits: u64,
    /// `radix^(2^k)` with bit length ≤ `max_bits`, for printing.
    powers: Vec<RadixPow>,
    inverses: Vec<Option<PreInv>>,
    /// `B^(2^k)` for `B = radix^chunk`, covering `max_bits`-bit values, for
    /// parsing.
    base: u64,
    chunk_powers: Vec<RadixPow>,
}

impl fmt::Debug for RadixConverter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RadixConverter")
            .field("radix", &self.radix)
            .field("max_bits", &self.max_bits)
            .finish_non_exhaustive()
    }
}

impl RadixConverter {
    /// Precomputes the ladders for `radix` (2–62) and values of up to
    /// `max_bits` bits. Panics if `radix` is out of range.
    pub fn new(radix: u32, max_bits: u64) -> RadixConverter {
        assert!((2..=62).contains(&radix), "radix must be in 2..=62");
        let mut powers = alloc::vec![RadixPow::of(radix as u64)];
        loop {
            let last = powers.last().expect("non-empty");
            if 2 * last.bit_len() - 1 > max_bits {
                break;
            }
            let sq = last.square();
            if sq.bit_len() > max_bits {
                break;
            }
            powers.push(sq);
        }
        let inverses = powers
            .iter()
            .map(|p| (p.odd.as_limbs().len() >= PREINV_LIMBS).then(|| PreInv::new(&p.odd)))
            .collect();
        let (_, base) = radix_chunk(radix);
        // Each full chunk spans B ≥ 2^(bits(B) − 1), so a `max_bits`-bit value
        // has at most max_bits / (bits(B) − 1) + 1 chunks.
        let chunk_bits = (64 - base.leading_zeros() - 1) as u64;
        let max_chunks = (max_bits / chunk_bits) as usize + 1;
        let mut chunk_powers = alloc::vec![RadixPow::of(base)];
        while 2usize << (chunk_powers.len() - 1) < max_chunks {
            let sq = chunk_powers.last().expect("non-empty").square();
            chunk_powers.push(sq);
        }
        RadixConverter {
            radix,
            max_bits,
            powers,
            inverses,
            base,
            chunk_powers,
        }
    }

    /// The radix.
    #[inline]
    pub fn radix(&self) -> u32 {
        self.radix
    }

    /// Appends the digits of `x` to `out` (`"0"` for zero).
    pub fn write_ascii(&self, x: &Nat, out: &mut Vec<u8>) {
        if x.is_zero() {
            out.push(b'0');
        } else if x.bit_len() > self.max_bits {
            x.write_ascii_radix(out, self.radix);
        } else if x.as_limbs().len() <= RADIX_RECURSION_LIMBS {
            simple_radix_string(x.as_limbs(), self.radix, 0, out);
        } else {
            out.reserve(x.as_limbs().len() * 20 + 1);
            to_radix_recursive(x, &self.powers, &self.inverses, self.radix, 0, out);
        }
    }

    /// Returns the digits of `x` (`"0"` for zero).
    pub fn to_ascii(&self, x: &Nat) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_ascii(x, &mut out);
        out
    }

    /// Parses ASCII digits (no sign, whitespace or separators; letters are
    /// case-insensitive up to radix 36). An empty slice is an error.
    pub fn parse(&self, digits: &[u8]) -> Result<Nat> {
        let (level, base) = pack_radix_chunks(digits, self.radix)?;
        debug_assert_eq!(base, self.base);
        Ok(match level.len() {
            1 => Nat::from_u64(level[0]),
            n if n <= PARSE_RECURSION_CHUNKS => horner_base_chunks(&level, base),
            // The recursion needs B^(2^k) for every 2^k < n.
            n if (1usize << self.chunk_powers.len()) >= n => {
                chunks_recursive(&level, &self.chunk_powers, base)
            }
            _ => nat_from_base_chunks(&level, base),
        })
    }
}
