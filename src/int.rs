//! Arbitrary-precision signed integers with small-value inlining.
//!
//! [`Int`] is a tagged representation: values whose magnitude fits in a single
//! 64-bit limb are stored inline (no heap allocation), and only larger
//! magnitudes spill to a heap-backed [`Nat`]. Every operation takes the inline
//! fast path when it can, and every result is re-canonicalized — demoted back to
//! inline form whenever it again fits a limb — so the representation is unique
//! and the derived [`PartialEq`]/[`Eq`]/[`Hash`] are correct.
//!
//! ```text
//! Repr::Small { neg, mag: u64 }   // value = ±mag, 0 ≤ mag ≤ u64::MAX
//! Repr::Large { sign, mag: Nat }  // |value| > u64::MAX
//! ```
//!
//! This is the type the specification calls `Integer`.

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::nat::Nat;

/// The sign of an [`Int`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Sign {
    /// A value strictly less than zero.
    Negative,
    /// The value zero.
    Zero,
    /// A value strictly greater than zero.
    Positive,
}

impl Default for Sign {
    #[inline]
    fn default() -> Self {
        Sign::Zero
    }
}

impl core::ops::Neg for Sign {
    type Output = Sign;
    #[inline]
    fn neg(self) -> Sign {
        match self {
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
            Sign::Positive => Sign::Negative,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Repr {
    /// Inline: `value = (if neg { -1 } else { 1 }) * mag`. Invariant: `mag == 0`
    /// implies `neg == false` (so zero is unique).
    Small { neg: bool, mag: u64 },
    /// Heap-backed: invariant `|value| > u64::MAX`, so `mag` has ≥ 2 limbs and
    /// `sign` is never [`Sign::Zero`].
    Large { sign: Sign, mag: Nat },
}

/// An arbitrary-precision signed integer (the spec's `Integer`).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Int(Repr);

impl Int {
    /// The value `0`.
    pub const ZERO: Int = Int(Repr::Small { neg: false, mag: 0 });
    /// The value `1`.
    pub const ONE: Int = Int(Repr::Small { neg: false, mag: 1 });
    /// The value `-1`.
    pub const MINUS_ONE: Int = Int(Repr::Small { neg: true, mag: 1 });

    /// Builds an inline value, canonicalizing the sign of zero.
    #[inline]
    const fn small(neg: bool, mag: u64) -> Int {
        Int(Repr::Small {
            neg: neg && mag != 0,
            mag,
        })
    }

    /// Returns the integer zero.
    #[inline]
    pub const fn zero() -> Int {
        Int::ZERO
    }

    /// Returns the integer one.
    #[inline]
    pub const fn one() -> Int {
        Int::ONE
    }

    /// Builds an integer from a sign and an unsigned magnitude, demoting to the
    /// inline representation when the magnitude fits a limb.
    pub fn from_sign_magnitude(sign: Sign, mag: Nat) -> Int {
        if mag.is_zero() {
            return Int::ZERO;
        }
        let neg = sign == Sign::Negative;
        match mag.to_u64() {
            Some(u) => Int::small(neg, u),
            None => Int(Repr::Large {
                sign: if neg { Sign::Negative } else { Sign::Positive },
                mag,
            }),
        }
    }

    /// Builds an integer from a `i64`.
    pub fn from_i64(v: i64) -> Int {
        if v < 0 {
            Int::small(true, v.unsigned_abs())
        } else {
            Int::small(false, v as u64)
        }
    }

    /// Builds an integer from a `i128`.
    pub fn from_i128(v: i128) -> Int {
        let mag = v.unsigned_abs();
        if mag <= u64::MAX as u128 {
            Int::small(v < 0, mag as u64)
        } else {
            Int::from_sign_magnitude(
                if v < 0 {
                    Sign::Negative
                } else {
                    Sign::Positive
                },
                Nat::from_u128(mag),
            )
        }
    }

    /// Builds a non-negative integer from a `u64`.
    #[inline]
    pub fn from_u64(v: u64) -> Int {
        Int::small(false, v)
    }

    /// Builds a non-negative integer from a `u128`.
    pub fn from_u128(v: u128) -> Int {
        if v <= u64::MAX as u128 {
            Int::small(false, v as u64)
        } else {
            Int::from_sign_magnitude(Sign::Positive, Nat::from_u128(v))
        }
    }

    /// Builds an integer from a sign and little-endian magnitude limbs.
    pub fn from_limbs(sign: Sign, limbs: &[u64]) -> Int {
        Int::from_sign_magnitude(sign, Nat::from_limbs(limbs))
    }

    /// Expands to sign-magnitude form for the slow path.
    fn to_sign_mag(&self) -> (Sign, Nat) {
        match &self.0 {
            Repr::Small { neg, mag } => {
                if *mag == 0 {
                    (Sign::Zero, Nat::zero())
                } else {
                    (
                        if *neg { Sign::Negative } else { Sign::Positive },
                        Nat::from_u64(*mag),
                    )
                }
            }
            Repr::Large { sign, mag } => (*sign, mag.clone()),
        }
    }

    // --- sign & predicates ---

    /// Returns the sign of this integer.
    #[inline]
    pub fn sign(&self) -> Sign {
        match &self.0 {
            Repr::Small { neg, mag } => {
                if *mag == 0 {
                    Sign::Zero
                } else if *neg {
                    Sign::Negative
                } else {
                    Sign::Positive
                }
            }
            Repr::Large { sign, .. } => *sign,
        }
    }

    /// Returns `-1`, `0`, or `1` according to the sign.
    #[inline]
    pub fn signum(&self) -> i32 {
        match self.sign() {
            Sign::Negative => -1,
            Sign::Zero => 0,
            Sign::Positive => 1,
        }
    }

    /// Returns the unsigned magnitude `|self|`.
    #[inline]
    pub fn magnitude(&self) -> Nat {
        self.to_sign_mag().1
    }

    /// Returns `true` if this value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self.0, Repr::Small { mag: 0, .. })
    }

    /// Returns `true` if this value is one.
    #[inline]
    pub fn is_one(&self) -> bool {
        matches!(self.0, Repr::Small { neg: false, mag: 1 })
    }

    /// Returns `true` if this value is minus one.
    #[inline]
    pub fn is_minus_one(&self) -> bool {
        matches!(self.0, Repr::Small { neg: true, mag: 1 })
    }

    /// Returns `true` if this value is strictly positive.
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.sign() == Sign::Positive
    }

    /// Returns `true` if this value is strictly negative.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.sign() == Sign::Negative
    }

    /// Returns `true` if this value is even (including zero).
    #[inline]
    pub fn is_even(&self) -> bool {
        match &self.0 {
            Repr::Small { mag, .. } => mag & 1 == 0,
            Repr::Large { mag, .. } => mag.is_even(),
        }
    }

    /// Returns `true` if this value is odd.
    #[inline]
    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }

    // --- basic arithmetic ---

    /// Returns the absolute value `|self|`.
    pub fn abs(&self) -> Int {
        match &self.0 {
            Repr::Small { mag, .. } => Int::small(false, *mag),
            Repr::Large { mag, .. } => Int::from_sign_magnitude(Sign::Positive, mag.clone()),
        }
    }

    /// Returns `-self`.
    pub fn neg(&self) -> Int {
        match &self.0 {
            Repr::Small { neg, mag } => Int::small(!*neg, *mag),
            Repr::Large { sign, mag } => Int::from_sign_magnitude(-*sign, mag.clone()),
        }
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Int) -> Int {
        if let (Repr::Small { neg: na, mag: ma }, Repr::Small { neg: nb, mag: mb }) =
            (&self.0, &rhs.0)
        {
            if na == nb {
                if let Some(s) = ma.checked_add(*mb) {
                    return Int::small(*na, s);
                }
            } else if ma >= mb {
                return Int::small(*na, ma - mb);
            } else {
                return Int::small(*nb, mb - ma);
            }
        }
        let (sa, a) = self.to_sign_mag();
        let (sb, b) = rhs.to_sign_mag();
        add_expanded(sa, &a, sb, &b)
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Int) -> Int {
        self.add(&rhs.neg())
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Int) -> Int {
        if let (Repr::Small { neg: na, mag: ma }, Repr::Small { neg: nb, mag: mb }) =
            (&self.0, &rhs.0)
            && let Some(p) = ma.checked_mul(*mb)
        {
            return Int::small(na ^ nb, p);
        }
        let (sa, a) = self.to_sign_mag();
        let (sb, b) = rhs.to_sign_mag();
        let sign = match (sa, sb) {
            (Sign::Zero, _) | (_, Sign::Zero) => return Int::ZERO,
            (x, y) if x == y => Sign::Positive,
            _ => Sign::Negative,
        };
        Int::from_sign_magnitude(sign, a.mul(&b))
    }

    /// Returns `self²` (always non-negative), via the fast squaring path.
    pub fn square(&self) -> Int {
        Int::from(self.magnitude().square())
    }

    /// Returns `self` raised to `exp` (`self^0 == 1`), by square-and-multiply.
    pub fn pow(&self, exp: u32) -> Int {
        let mut result = Int::ONE;
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

    /// Fused multiply-add: `self += a · b`.
    pub fn addmul(&mut self, a: &Int, b: &Int) {
        if let (
            Repr::Small { neg: ns, mag: ms },
            Repr::Small { neg: na, mag: ma },
            Repr::Small { neg: nb, mag: mb },
        ) = (&self.0, &a.0, &b.0)
        {
            let s = if *ns { -(*ms as i128) } else { *ms as i128 };
            let x = if *na { -(*ma as i128) } else { *ma as i128 };
            let y = if *nb { -(*mb as i128) } else { *mb as i128 };
            let r = s + x * y;
            if let Ok(v) = i64::try_from(r) {
                *self = Int::from_i64(v);
                return;
            }
        }
        *self = self.add(&a.mul(b));
    }

    /// Fused multiply-subtract: `self -= a · b`.
    pub fn submul(&mut self, a: &Int, b: &Int) {
        if let (
            Repr::Small { neg: ns, mag: ms },
            Repr::Small { neg: na, mag: ma },
            Repr::Small { neg: nb, mag: mb },
        ) = (&self.0, &a.0, &b.0)
        {
            let s = if *ns { -(*ms as i128) } else { *ms as i128 };
            let x = if *na { -(*ma as i128) } else { *ma as i128 };
            let y = if *nb { -(*mb as i128) } else { *mb as i128 };
            let r = s - x * y;
            if let Ok(v) = i64::try_from(r) {
                *self = Int::from_i64(v);
                return;
            }
        }
        *self = self.sub(&a.mul(b));
    }

    // --- division (see the module & ROADMAP for the three conventions) ---

    /// Truncated division, returning `(quotient, remainder)` with the quotient
    /// rounded toward zero and the remainder taking the sign of `self`, or
    /// `None` if `d` is zero.
    pub fn div_rem(&self, d: &Int) -> Option<(Int, Int)> {
        let (sa, a) = self.to_sign_mag();
        let (sb, b) = d.to_sign_mag();
        let (q, r) = a.div_rem(&b)?;
        let q_sign = match (sa, sb) {
            (Sign::Zero, _) => Sign::Zero,
            (x, y) if x == y => Sign::Positive,
            _ => Sign::Negative,
        };
        Some((
            Int::from_sign_magnitude(q_sign, q),
            Int::from_sign_magnitude(sa, r),
        ))
    }

    /// Truncated `(quotient, remainder)`; panics if `d` is zero.
    #[inline]
    pub fn div_rem_trunc(&self, d: &Int) -> (Int, Int) {
        self.div_rem(d).expect("division by zero")
    }

    /// Truncated quotient (rounds toward zero); panics if `d` is zero.
    #[inline]
    pub fn div_trunc(&self, d: &Int) -> Int {
        self.div_rem_trunc(d).0
    }

    /// Truncated remainder (sign of `self`); panics if `d` is zero.
    #[inline]
    pub fn rem_trunc(&self, d: &Int) -> Int {
        self.div_rem_trunc(d).1
    }

    /// Euclidean `(quotient, remainder)` with `0 ≤ remainder < |d|`; panics if
    /// `d` is zero.
    pub fn div_rem_euclid(&self, d: &Int) -> (Int, Int) {
        let (q, r) = self.div_rem_trunc(d);
        if r.is_negative() {
            // r += |d|; q moves one step so that a == q·d + r still holds.
            let q2 = if d.is_negative() {
                q.add(&Int::ONE)
            } else {
                q.sub(&Int::ONE)
            };
            (q2, r.add(&d.abs()))
        } else {
            (q, r)
        }
    }

    /// Euclidean quotient; panics if `d` is zero.
    #[inline]
    pub fn div_euclid(&self, d: &Int) -> Int {
        self.div_rem_euclid(d).0
    }

    /// Euclidean remainder, always in `[0, |d|)`; panics if `d` is zero.
    #[inline]
    pub fn rem_euclid(&self, d: &Int) -> Int {
        self.div_rem_euclid(d).1
    }

    /// Floored `(quotient, remainder)`: quotient rounds toward −∞, remainder
    /// takes the sign of `d`; panics if `d` is zero.
    pub fn div_rem_floor(&self, d: &Int) -> (Int, Int) {
        let (q, r) = self.div_rem_trunc(d);
        if !r.is_zero() && (r.is_negative() != d.is_negative()) {
            (q.sub(&Int::ONE), r.add(d))
        } else {
            (q, r)
        }
    }

    /// Floored quotient (rounds toward −∞); panics if `d` is zero.
    #[inline]
    pub fn div_floor(&self, d: &Int) -> Int {
        self.div_rem_floor(d).0
    }

    /// Exact division for the case where `d` divides `self` exactly. In debug
    /// builds this asserts the remainder is zero; panics if `d` is zero.
    pub fn div_exact(&self, d: &Int) -> Int {
        let (q, r) = self.div_rem_trunc(d);
        debug_assert!(r.is_zero(), "div_exact: divisor does not divide dividend");
        q
    }

    /// Returns `true` if `self` divides `other` exactly. `0` divides only `0`.
    pub fn divides(&self, other: &Int) -> bool {
        if self.is_zero() {
            other.is_zero()
        } else {
            other.div_rem_trunc(self).1.is_zero()
        }
    }

    // --- number theory & roots ---

    /// Greatest common divisor (non-negative). `gcd(0, 0) == 0`.
    pub fn gcd(&self, b: &Int) -> Int {
        Int::from(self.magnitude().gcd(&b.magnitude()))
    }

    /// Least common multiple (non-negative). `lcm(x, 0) == 0`.
    pub fn lcm(&self, b: &Int) -> Int {
        if self.is_zero() || b.is_zero() {
            return Int::ZERO;
        }
        let ma = self.magnitude();
        let mb = b.magnitude();
        let g = ma.gcd(&mb);
        let reduced = ma.div_rem(&g).expect("gcd is non-zero").0;
        Int::from(reduced.mul(&mb))
    }

    /// Returns `self^exp mod modulus`, a non-negative value in `[0, |modulus|)`.
    /// Panics if `exp` is negative (use [`Int::modinv`] first) or `modulus` is
    /// zero.
    pub fn modpow(&self, exp: &Int, modulus: &Int) -> Int {
        assert!(!exp.is_negative(), "modpow: negative exponent");
        assert!(!modulus.is_zero(), "modpow: zero modulus");
        let m = modulus.magnitude();
        let base = self.rem_euclid(modulus).magnitude(); // in [0, |m|)
        Int::from(base.modpow(&exp.magnitude(), &m))
    }

    /// Returns the modular inverse of `self` mod `modulus` (in `[0, |modulus|)`),
    /// or `None` if `self` is not invertible (`gcd(self, modulus) != 1`).
    pub fn modinv(&self, modulus: &Int) -> Option<Int> {
        if modulus.is_zero() {
            return None;
        }
        let (g, x, _) = self.extended_gcd(modulus);
        if !g.is_one() {
            return None;
        }
        Some(x.rem_euclid(modulus))
    }

    /// Deterministic Baillie–PSW primality test (`false` for `self < 2`).
    pub fn is_prime_bpsw(&self) -> bool {
        !self.is_negative() && self.magnitude().is_prime_bpsw()
    }

    /// Returns the prime factorization of `|self|` as a sorted list of prime
    /// factors with multiplicity (empty for `0`/`±1`; the sign is ignored).
    pub fn factorize(&self) -> alloc::vec::Vec<Int> {
        self.magnitude()
            .factorize()
            .into_iter()
            .map(Int::from)
            .collect()
    }

    /// Generates a uniformly random prime with exactly `bits` bits (`bits >= 2`),
    /// verified with Baillie–PSW.
    pub fn random_prime(bits: u32, rng: &mut impl crate::random::RandomSource) -> Int {
        assert!(bits >= 2, "random_prime: need at least 2 bits");
        let high = Int::ONE.mul_2k(bits - 1); // top bit set
        let limit = Int::ONE.mul_2k(bits);
        loop {
            // Random odd candidate with the high bit set.
            let base = Int::from(Nat::random_bits(bits as u64, rng))
                .bitor(&high)
                .bitor(&Int::ONE);
            let mut c = base;
            while c < limit {
                if c.is_prime_bpsw() {
                    return c;
                }
                c = c.add(&Int::from_i64(2));
            }
            // Ran past 2^bits without a prime; draw a fresh candidate.
        }
    }

    /// Returns the smallest prime strictly greater than `self` (at least 2).
    pub fn next_prime(&self, rng: &mut impl crate::random::RandomSource) -> Int {
        if self < &Int::from_i64(2) {
            return Int::from_i64(2);
        }
        Int::from(self.magnitude().next_prime(rng))
    }

    /// Returns the largest prime strictly less than `self`, or `None` if there
    /// is none (`self <= 2`).
    pub fn prev_prime(&self, rng: &mut impl crate::random::RandomSource) -> Option<Int> {
        if self <= &Int::from_i64(2) {
            return None;
        }
        self.magnitude().prev_prime(rng).map(Int::from)
    }

    /// Extended GCD: returns `(g, x, y)` with `g == self·x + b·y` and `g ≥ 0`.
    pub fn extended_gcd(&self, b: &Int) -> (Int, Int, Int) {
        let (mut old_r, mut r) = (self.clone(), b.clone());
        let (mut old_s, mut s) = (Int::ONE, Int::ZERO);
        let (mut old_t, mut t) = (Int::ZERO, Int::ONE);
        while !r.is_zero() {
            let q = old_r.div_trunc(&r);
            let nr = old_r.sub(&q.mul(&r));
            old_r = core::mem::replace(&mut r, nr);
            let ns = old_s.sub(&q.mul(&s));
            old_s = core::mem::replace(&mut s, ns);
            let nt = old_t.sub(&q.mul(&t));
            old_t = core::mem::replace(&mut t, nt);
        }
        if old_r.is_negative() {
            (old_r.neg(), old_s.neg(), old_t.neg())
        } else {
            (old_r, old_s, old_t)
        }
    }

    /// Returns `⌊√self⌋²`-checked exact square root, `Some(r)` iff `self == r²`
    /// (`self ≥ 0`); `None` for negatives or non-squares.
    pub fn sqrt_exact(&self) -> Option<Int> {
        if self.is_negative() {
            return None;
        }
        let m = self.magnitude();
        let r = m.isqrt();
        (r.mul(&r) == m).then(|| Int::from(r))
    }

    /// Returns the exact `n`th root, `Some(r)` iff `self == rⁿ`. Even roots of
    /// negatives, and `n == 0`, return `None`.
    pub fn nth_root_exact(&self, n: u32) -> Option<Int> {
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(self.clone());
        }
        let neg = self.is_negative();
        if neg && n.is_multiple_of(2) {
            return None;
        }
        let m = self.magnitude();
        let r = m.nth_root_floor(n);
        (r.pow(n) == m)
            .then(|| Int::from_sign_magnitude(if neg { Sign::Negative } else { Sign::Positive }, r))
    }

    // --- power-of-two & bit access ---

    /// Returns `self << k` (multiply by `2^k`).
    pub fn mul_2k(&self, k: u32) -> Int {
        let (s, m) = self.to_sign_mag();
        Int::from_sign_magnitude(s, m.shl(k as u64))
    }

    /// Returns the truncated `self / 2^k` (rounds toward zero).
    pub fn div_2k_trunc(&self, k: u32) -> Int {
        let (s, m) = self.to_sign_mag();
        Int::from_sign_magnitude(s, m.shr(k as u64))
    }

    /// Returns `self mod 2^k`, the non-negative value in `[0, 2^k)` (the low `k`
    /// bits, Euclidean).
    pub fn mod_2k(&self, k: u32) -> Int {
        let (s, m) = self.to_sign_mag();
        let r = m.low_bits(k as u64);
        if s == Sign::Negative && !r.is_zero() {
            let modulus = Nat::one().shl(k as u64);
            Int::from(modulus.checked_sub(&r).expect("2^k > low_bits"))
        } else {
            Int::from(r)
        }
    }

    /// If `|self|` is a power of two `2^k`, returns `Some(k)`; otherwise `None`.
    pub fn is_power_of_two(&self) -> Option<u32> {
        let m = self.magnitude();
        if m.is_zero() {
            return None;
        }
        let tz = m.trailing_zeros();
        (m.bit_len() == tz + 1).then_some(tz as u32)
    }

    /// Returns `⌈log2 |self|⌉` (the exponent of the smallest power of two that is
    /// `≥ |self|`); `0` for zero.
    pub fn next_power_of_two(&self) -> u32 {
        match self.is_power_of_two() {
            Some(k) => k,
            None => self.bit_len(),
        }
    }

    /// Returns `⌊log2 |self|⌋` (the exponent of the largest power of two that is
    /// `≤ |self|`); `0` for zero.
    #[inline]
    pub fn prev_power_of_two(&self) -> u32 {
        self.bit_len().saturating_sub(1)
    }

    /// Returns the number of trailing zero bits of `|self|`; `0` for zero.
    pub fn trailing_zeros(&self) -> u32 {
        match &self.0 {
            Repr::Small { mag, .. } => {
                if *mag == 0 {
                    0
                } else {
                    mag.trailing_zeros()
                }
            }
            Repr::Large { mag, .. } => mag.trailing_zeros() as u32,
        }
    }

    /// Returns the number of bits in `|self|`; `0` for zero.
    pub fn bit_len(&self) -> u32 {
        match &self.0 {
            Repr::Small { mag, .. } => u64::BITS - mag.leading_zeros(),
            Repr::Large { mag, .. } => mag.bit_len() as u32,
        }
    }

    /// Returns `⌊log2 |self|⌋`; `0` for zero and one.
    #[inline]
    pub fn log2_floor(&self) -> u32 {
        self.bit_len().saturating_sub(1)
    }

    /// Returns bit `i` of `|self|` (little-endian; `false` past the top bit).
    pub fn bit(&self, i: u32) -> bool {
        match &self.0 {
            Repr::Small { mag, .. } => i < u64::BITS && (mag >> i) & 1 == 1,
            Repr::Large { mag, .. } => mag.bit(i as u64),
        }
    }

    /// Returns the little-endian limb slice of `|self|` (empty for zero).
    pub fn limbs(&self) -> &[u64] {
        match &self.0 {
            Repr::Small { mag, .. } => {
                if *mag == 0 {
                    &[]
                } else {
                    core::slice::from_ref(mag)
                }
            }
            Repr::Large { mag, .. } => mag.as_limbs(),
        }
    }

    /// Returns the least-significant limb of `|self|` (`0` for zero).
    pub fn least_significant_limb(&self) -> u64 {
        match &self.0 {
            Repr::Small { mag, .. } => *mag,
            Repr::Large { mag, .. } => mag.as_limbs().first().copied().unwrap_or(0),
        }
    }

    // --- two's-complement bitwise (see ROADMAP §2.4) ---

    /// Bitwise AND on the two's-complement representation.
    pub fn bitand(&self, b: &Int) -> Int {
        self.bitwise(b, |x, y| x & y)
    }

    /// Bitwise OR on the two's-complement representation.
    pub fn bitor(&self, b: &Int) -> Int {
        self.bitwise(b, |x, y| x | y)
    }

    /// Bitwise XOR on the two's-complement representation.
    pub fn bitxor(&self, b: &Int) -> Int {
        self.bitwise(b, |x, y| x ^ y)
    }

    fn bitwise<F: Fn(u64, u64) -> u64>(&self, b: &Int, op: F) -> Int {
        let len = self.limbs().len().max(b.limbs().len()) + 1;
        let x = twos_complement(self, len);
        let y = twos_complement(b, len);
        let r: Vec<u64> = x.iter().zip(&y).map(|(&a, &b)| op(a, b)).collect();
        from_twos_complement(&r)
    }

    /// Bitwise NOT within an explicit `width`-bit two's-complement window. The
    /// result is interpreted as a `width`-bit signed value (so `bitnot(x, w)`
    /// flips the low `w` bits of `x`'s two's-complement form and reads the top
    /// bit as the sign).
    pub fn bitnot(&self, width: u32) -> Int {
        if width == 0 {
            return Int::ZERO;
        }
        let len = (width as usize).div_ceil(64);
        let mut v = twos_complement(self, len);
        for limb in v.iter_mut() {
            *limb = !*limb;
        }
        mask_to_width(&mut v, width);
        // Interpret as width-bit two's complement.
        let sign_bit = ((width - 1) / 64) as usize;
        let negative = sign_bit < v.len() && (v[sign_bit] >> ((width - 1) % 64)) & 1 == 1;
        if negative {
            let modulus = Nat::one().shl(width as u64);
            let unsigned = Nat::from_limbs(&v);
            Int::from_sign_magnitude(
                Sign::Negative,
                modulus.checked_sub(&unsigned).expect("2^width > value"),
            )
        } else {
            Int::from(Nat::from_limbs(&v))
        }
    }

    // --- bounded conversions ---

    /// Returns `true` if the value fits in an `i64`.
    #[inline]
    pub fn fits_i64(&self) -> bool {
        self.to_i64().is_some()
    }

    /// Returns `true` if the value fits in a `u64` (non-negative and small).
    #[inline]
    pub fn fits_u64(&self) -> bool {
        self.to_u64().is_some()
    }

    /// Returns the value as an `i64` if it fits.
    pub fn to_i64(&self) -> Option<i64> {
        match &self.0 {
            Repr::Small { neg, mag } => {
                if *neg {
                    // -mag fits iff mag <= 2^63.
                    (*mag <= (i64::MAX as u64) + 1).then(|| -(*mag as i128) as i64)
                } else {
                    (*mag <= i64::MAX as u64).then_some(*mag as i64)
                }
            }
            Repr::Large { .. } => None,
        }
    }

    /// Returns the value as a `u64` if it is non-negative and fits.
    pub fn to_u64(&self) -> Option<u64> {
        match &self.0 {
            Repr::Small { neg, mag } => (!*neg).then_some(*mag),
            Repr::Large { .. } => None,
        }
    }

    /// Returns the value as the nearest `f64` (best-effort; may be `±inf` on
    /// overflow).
    pub fn to_f64(&self) -> f64 {
        let (sign, mag) = self.to_sign_mag();
        let mut f = 0.0f64;
        for &limb in mag.as_limbs().iter().rev() {
            f = f * 18446744073709551616.0 /* 2^64 */ + limb as f64;
        }
        if sign == Sign::Negative { -f } else { f }
    }
}

/// Sign-magnitude addition on the slow (heap) path.
fn add_expanded(sa: Sign, a: &Nat, sb: Sign, b: &Nat) -> Int {
    match (sa, sb) {
        (Sign::Zero, _) => Int::from_sign_magnitude(sb, b.clone()),
        (_, Sign::Zero) => Int::from_sign_magnitude(sa, a.clone()),
        (x, y) if x == y => Int::from_sign_magnitude(x, a.add(b)),
        _ => match a.cmp(b) {
            Ordering::Equal => Int::ZERO,
            Ordering::Greater => Int::from_sign_magnitude(sa, a.checked_sub(b).expect("a > b")),
            Ordering::Less => Int::from_sign_magnitude(sb, b.checked_sub(a).expect("b > a")),
        },
    }
}

/// Little-endian two's-complement limbs of `x` in exactly `len` limbs.
fn twos_complement(x: &Int, len: usize) -> Vec<u64> {
    let (sign, mag) = x.to_sign_mag();
    let mut v = alloc::vec![0u64; len];
    for (slot, &limb) in v.iter_mut().zip(mag.as_limbs()) {
        *slot = limb;
    }
    if sign == Sign::Negative {
        let mut carry = 1u64;
        for limb in v.iter_mut() {
            let (s, c) = (!*limb).overflowing_add(carry);
            *limb = s;
            carry = c as u64;
        }
    }
    v
}

/// Interprets `len` little-endian two's-complement limbs as an [`Int`].
fn from_twos_complement(v: &[u64]) -> Int {
    let negative = v.last().is_some_and(|&top| top >> 63 == 1);
    if negative {
        let mut m = alloc::vec![0u64; v.len()];
        let mut carry = 1u64;
        for (slot, &limb) in m.iter_mut().zip(v) {
            let (s, c) = (!limb).overflowing_add(carry);
            *slot = s;
            carry = c as u64;
        }
        Int::from_sign_magnitude(Sign::Negative, Nat::from_limbs(&m))
    } else {
        Int::from(Nat::from_limbs(v))
    }
}

/// Zeroes all bits at index `>= width`.
fn mask_to_width(v: &mut [u64], width: u32) {
    let w = width as usize;
    for (i, limb) in v.iter_mut().enumerate() {
        let lo = i * 64;
        if lo >= w {
            *limb = 0;
        } else if lo + 64 > w {
            *limb &= (1u64 << (w - lo)) - 1;
        }
    }
}

/// The Legendre symbol via Euler's criterion `a^((p-1)/2) mod p`, on naturals.
fn legendre_symbol(a: &Nat, p: &Nat) -> i32 {
    let a = a.div_rem(p).expect("odd prime modulus").1;
    if a.is_zero() {
        return 0;
    }
    let exp = p.checked_sub(&Nat::one()).unwrap().shr(1); // (p-1)/2
    if a.modpow(&exp, p).is_one() { 1 } else { -1 }
}

/// Tonelli–Shanks modular square root of `a` mod the odd prime `p`.
fn tonelli_shanks(a: &Nat, p: &Nat) -> Option<Nat> {
    let one = Nat::one();
    let a = a.div_rem(p).expect("non-zero modulus").1;
    if a.is_zero() {
        return Some(Nat::zero());
    }
    if legendre_symbol(&a, p) != 1 {
        return None; // quadratic non-residue
    }
    let modp = |x: Nat| x.div_rem(p).unwrap().1;
    let p1 = p.checked_sub(&one).unwrap();
    let s = p1.trailing_zeros();
    let q = p1.shr(s); // odd

    if s == 1 {
        // r = a^((p+1)/4)
        return Some(a.modpow(&p.add(&one).shr(2), p));
    }
    // A quadratic non-residue z.
    let mut z = Nat::from_u64(2);
    while legendre_symbol(&z, p) != -1 {
        z = z.add(&one);
    }
    let mut m = s;
    let mut c = z.modpow(&q, p);
    let mut t = a.modpow(&q, p);
    let mut r = a.modpow(&q.add(&one).shr(1), p); // a^((q+1)/2)
    loop {
        if t.is_one() {
            return Some(r);
        }
        // Least i in (0, m) with t^(2^i) == 1.
        let mut i = 1u64;
        let mut t2 = modp(t.square());
        while !t2.is_one() {
            t2 = modp(t2.square());
            i += 1;
        }
        let b = c.modpow(&one.shl(m - i - 1), p); // c^(2^(m-i-1))
        m = i;
        c = modp(b.square());
        t = modp(t.mul(&c)); // t·b²
        r = modp(r.mul(&b));
    }
}

/// `(F(n), F(n+1))` by fast doubling.
fn fib_pair(n: u64) -> (Int, Int) {
    if n == 0 {
        return (Int::ZERO, Int::ONE);
    }
    let (a, b) = fib_pair(n >> 1); // (F(k), F(k+1)), k = n >> 1
    let c = a.mul(&b.mul_2k(1).sub(&a)); // F(2k)   = F(k)·(2·F(k+1) − F(k))
    let d = a.square().add(&b.square()); // F(2k+1) = F(k)² + F(k+1)²
    if n & 1 == 0 {
        (c, d)
    } else {
        let next = c.add(&d);
        (d, next)
    }
}

impl Int {
    /// Returns `n!` (`0! == 1! == 1`).
    pub fn factorial(n: u64) -> Int {
        let mut acc = Int::ONE;
        for k in 2..=n {
            acc = acc.mul(&Int::from(k));
        }
        acc
    }

    /// Returns the binomial coefficient `C(n, k)` (`0` if `k > n`).
    pub fn binomial(n: u64, k: u64) -> Int {
        if k > n {
            return Int::ZERO;
        }
        let k = k.min(n - k);
        let mut result = Int::ONE;
        for i in 1..=k {
            // Each intermediate C(n-k+i, i) is an integer, so the division exact.
            result = result.mul(&Int::from(n - k + i)).div_exact(&Int::from(i));
        }
        result
    }

    /// Returns the multinomial coefficient `(Σkᵢ)! / ∏(kᵢ!)`.
    pub fn multinomial(ks: &[u64]) -> Int {
        let mut total = 0u64;
        let mut result = Int::ONE;
        for &k in ks {
            total += k;
            result = result.mul(&Int::binomial(total, k));
        }
        result
    }

    /// Returns the `n`th Fibonacci number `F(n)` (`F(0) = 0`, `F(1) = 1`), by
    /// fast doubling.
    pub fn fibonacci(n: u64) -> Int {
        fib_pair(n).0
    }

    /// Returns the `n`th Lucas number `L(n)` (`L(0) = 2`, `L(1) = 1`).
    pub fn lucas(n: u64) -> Int {
        let (f_n, f_n1) = fib_pair(n);
        f_n1.mul_2k(1).sub(&f_n) // L(n) = 2·F(n+1) − F(n)
    }

    /// Returns the Jacobi symbol `(self / n)` for odd `n > 0` (`-1`, `0`, `1`).
    pub fn jacobi(&self, n: &Int) -> i32 {
        assert!(
            n.is_positive() && n.is_odd(),
            "jacobi: n must be odd and > 0"
        );
        crate::nat::jacobi(self, &n.magnitude())
    }

    /// Returns the Legendre symbol `(self / p)` for an odd prime `p`. (Same
    /// computation as [`Int::jacobi`]; correct as a Legendre symbol only when
    /// `p` is prime.)
    #[inline]
    pub fn legendre(&self, p: &Int) -> i32 {
        self.jacobi(p)
    }

    /// Returns a modular square root of `self` mod the prime `p`, i.e. some `r`
    /// with `r² ≡ self (mod p)`, or `None` if `self` is a quadratic non-residue.
    /// The result lies in `[0, p)`. `p` must be prime.
    pub fn sqrt_mod(&self, p: &Int) -> Option<Int> {
        assert!(p.is_positive(), "sqrt_mod: modulus must be positive");
        let pn = p.magnitude();
        if pn == Nat::from_u64(2) {
            return Some(self.rem_euclid(p)); // 0→0, 1→1
        }
        tonelli_shanks(&self.rem_euclid(p).magnitude(), &pn).map(Int::from)
    }

    /// Solves the system `x ≡ residues[i] (mod moduli[i])` by the Chinese
    /// Remainder Theorem, returning the unique `x` in `[0, ∏ moduli)`, or `None`
    /// if the moduli are not pairwise coprime. Panics if the slices differ in
    /// length.
    pub fn crt(residues: &[Int], moduli: &[Int]) -> Option<Int> {
        assert_eq!(residues.len(), moduli.len(), "crt: mismatched lengths");
        let mut x = Int::ZERO;
        let mut m = Int::ONE;
        for (r, mi) in residues.iter().zip(moduli) {
            let mi = mi.abs();
            let diff = r.sub(&x).rem_euclid(&mi);
            let inv = m.modinv(&mi)?; // requires gcd(m, mi) == 1
            let t = diff.mul(&inv).rem_euclid(&mi);
            x = x.add(&m.mul(&t));
            m = m.mul(&mi);
        }
        Some(x.rem_euclid(&m))
    }
}

impl PartialOrd for Int {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Int {
    fn cmp(&self, other: &Self) -> Ordering {
        if let (Repr::Small { neg: na, mag: ma }, Repr::Small { neg: nb, mag: mb }) =
            (&self.0, &other.0)
        {
            let va = if *ma == 0 {
                0i128
            } else if *na {
                -(*ma as i128)
            } else {
                *ma as i128
            };
            let vb = if *mb == 0 {
                0i128
            } else if *nb {
                -(*mb as i128)
            } else {
                *mb as i128
            };
            return va.cmp(&vb);
        }
        let (sa, a) = self.to_sign_mag();
        let (sb, b) = other.to_sign_mag();
        match (sa, sb) {
            (Sign::Negative, Sign::Zero | Sign::Positive) | (Sign::Zero, Sign::Positive) => {
                Ordering::Less
            }
            (Sign::Zero, Sign::Zero) => Ordering::Equal,
            (Sign::Positive, Sign::Zero | Sign::Negative) | (Sign::Zero, Sign::Negative) => {
                Ordering::Greater
            }
            (Sign::Positive, Sign::Positive) => a.cmp(&b),
            (Sign::Negative, Sign::Negative) => b.cmp(&a),
        }
    }
}

impl Default for Int {
    #[inline]
    fn default() -> Int {
        Int::ZERO
    }
}

macro_rules! from_signed {
    ($($t:ty)*) => {$(
        impl From<$t> for Int {
            #[inline]
            fn from(v: $t) -> Int { Int::from_i64(v as i64) }
        }
    )*};
}
from_signed!(i8 i16 i32 i64);

macro_rules! from_unsigned {
    ($($t:ty)*) => {$(
        impl From<$t> for Int {
            #[inline]
            fn from(v: $t) -> Int { Int::from_u64(v as u64) }
        }
    )*};
}
from_unsigned!(u8 u16 u32 u64 usize);

impl From<i128> for Int {
    #[inline]
    fn from(v: i128) -> Int {
        Int::from_i128(v)
    }
}

impl From<Nat> for Int {
    #[inline]
    fn from(mag: Nat) -> Int {
        Int::from_sign_magnitude(Sign::Positive, mag)
    }
}

impl Int {
    /// Parses a decimal integer in the given `radix` (2–36), with an optional
    /// leading `+`/`-`.
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Int> {
        let (sign, digits) = match s.strip_prefix('-') {
            Some(rest) => (Sign::Negative, rest),
            None => (Sign::Positive, s.strip_prefix('+').unwrap_or(s)),
        };
        let mag = crate::nat::parse_radix(digits, radix)?;
        Ok(Int::from_sign_magnitude(sign, mag))
    }

    /// Writes `self` in the given `radix` (2–36) to `out`.
    pub fn write_radix(&self, out: &mut impl fmt::Write, radix: u32) -> fmt::Result {
        if self.is_negative() {
            out.write_str("-")?;
        }
        self.magnitude().write_radix(out, radix)
    }
}

impl FromStr for Int {
    type Err = Error;

    /// Parses a decimal integer with an optional leading `+` or `-`.
    fn from_str(s: &str) -> Result<Self> {
        Int::from_str_radix(s, 10)
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Repr::Small { neg, mag } => {
                if *neg {
                    f.write_str("-")?;
                }
                write!(f, "{mag}")
            }
            Repr::Large { sign, mag } => {
                if *sign == Sign::Negative {
                    f.write_str("-")?;
                }
                fmt::Display::fmt(mag, f)
            }
        }
    }
}

impl fmt::Debug for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Int({self})")
    }
}

macro_rules! binops {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr<Int> for Int {
            type Output = Int;
            #[inline]
            fn $m(self, rhs: Int) -> Int {
                Int::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&Int> for Int {
            type Output = Int;
            #[inline]
            fn $m(self, rhs: &Int) -> Int {
                Int::$m(&self, rhs)
            }
        }
        impl core::ops::$tr<Int> for &Int {
            type Output = Int;
            #[inline]
            fn $m(self, rhs: Int) -> Int {
                Int::$m(self, &rhs)
            }
        }
        impl core::ops::$tr<&Int> for &Int {
            type Output = Int;
            #[inline]
            fn $m(self, rhs: &Int) -> Int {
                Int::$m(self, rhs)
            }
        }
        impl core::ops::$tr<i64> for Int {
            type Output = Int;
            #[inline]
            fn $m(self, rhs: i64) -> Int {
                Int::$m(&self, &Int::from_i64(rhs))
            }
        }
        impl core::ops::$tr<i64> for &Int {
            type Output = Int;
            #[inline]
            fn $m(self, rhs: i64) -> Int {
                Int::$m(self, &Int::from_i64(rhs))
            }
        }
        impl core::ops::$atr<Int> for Int {
            #[inline]
            fn $am(&mut self, rhs: Int) {
                *self = Int::$m(self, &rhs);
            }
        }
        impl core::ops::$atr<&Int> for Int {
            #[inline]
            fn $am(&mut self, rhs: &Int) {
                *self = Int::$m(self, rhs);
            }
        }
        impl core::ops::$atr<i64> for Int {
            #[inline]
            fn $am(&mut self, rhs: i64) {
                *self = Int::$m(self, &Int::from_i64(rhs));
            }
        }
    };
}

binops!(Add, add, AddAssign, add_assign);
binops!(Sub, sub, SubAssign, sub_assign);
binops!(Mul, mul, MulAssign, mul_assign);

impl core::ops::Neg for Int {
    type Output = Int;
    #[inline]
    fn neg(self) -> Int {
        Int::neg(&self)
    }
}

impl core::ops::Neg for &Int {
    type Output = Int;
    #[inline]
    fn neg(self) -> Int {
        Int::neg(self)
    }
}

impl core::iter::Sum for Int {
    fn sum<I: Iterator<Item = Int>>(iter: I) -> Int {
        iter.fold(Int::ZERO, |acc, x| acc.add(&x))
    }
}

impl<'a> core::iter::Sum<&'a Int> for Int {
    fn sum<I: Iterator<Item = &'a Int>>(iter: I) -> Int {
        iter.fold(Int::ZERO, |acc, x| acc.add(x))
    }
}

impl core::iter::Product for Int {
    fn product<I: Iterator<Item = Int>>(iter: I) -> Int {
        iter.fold(Int::ONE, |acc, x| acc.mul(&x))
    }
}

impl<'a> core::iter::Product<&'a Int> for Int {
    fn product<I: Iterator<Item = &'a Int>>(iter: I) -> Int {
        iter.fold(Int::ONE, |acc, x| acc.mul(x))
    }
}
