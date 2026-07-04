//! Fixed-precision `p`-adic numbers — elements of the field `ℚ_p` (and its ring
//! of integers `ℤ_p`), carried to a bounded number of significant `p`-adic
//! digits.
//!
//! # Representation
//!
//! A nonzero value is written in the canonical form
//!
//! ```text
//! x = p^v · u
//! ```
//!
//! where `v = v_p(x)` is the `p`-adic **valuation** (an integer, possibly
//! negative for elements of `ℚ_p ∖ ℤ_p`) and `u` is a **unit** — an integer
//! coprime to `p`, stored reduced modulo `p^N`. `N` is the *relative* precision:
//! the number of significant `p`-adic digits we keep for the unit. Equivalently
//! the value is known modulo `p^{v+N}`; that exponent `v + N` is the *absolute*
//! precision and is what [`Padic`] stores internally (as `abs_prec`), so that
//! cancellation is tracked honestly.
//!
//! The **zero** value carries no valuation (its valuation is `+∞`); it is stored
//! with `unit = 0` and an absolute precision `a`, meaning only that the value is
//! `≡ 0 (mod p^a)`. [`Padic::is_zero`] reports whether all known digits vanish.
//!
//! # Digit order and [`Display`](core::fmt::Display)
//!
//! [`Display`](core::fmt::Display) prints the expansion **low-order digit
//! first** as a sum `Σ dᵢ·p^i` (with `0 ≤ dᵢ < p`, `i` running from the
//! valuation up), terminated by a big-`O` term recording the absolute precision,
//! e.g. `-1` in `ℤ₂` to five digits prints as
//! `1 + 1*2 + 1*2^2 + 1*2^3 + 1*2^4 + O(2^5)`. The zero value prints as
//! `O(p^a)`.
//!
//! # Clean-room provenance
//!
//! Definitions and algorithms are drawn from the open literature — Gouvêa,
//! *p-adic Numbers: An Introduction*; Koblitz, *p-adic Numbers, p-adic Analysis,
//! and Zeta-Functions*; Knuth, *TAOCP* Vol. 2 (Hensel lifting / Newton
//! iteration). No third-party source was consulted.

use core::cmp::Ordering;
use core::fmt;

use alloc::vec::Vec;

use crate::int::Int;
use crate::rational::Rational;

/// A `p`-adic number held to a fixed number of significant digits.
///
/// See the [module documentation](self) for the representation and the
/// [`Display`](core::fmt::Display) format.
#[derive(Clone)]
pub struct Padic {
    /// The prime `p`. Must be prime (checked with a `debug_assert!`).
    p: Int,
    /// Valuation `v`; meaningful only when `unit != 0`.
    val: i64,
    /// The unit `u`, reduced modulo `p^{abs_prec - val}` and coprime to `p`;
    /// `0` for the zero value.
    unit: Int,
    /// Absolute precision `a`: the value is known modulo `p^a`.
    abs_prec: i64,
}

/// Splits `p` out of `m`, returning `(v, cofactor)` with `m = p^v · cofactor`
/// and `cofactor` coprime to `p` (the cofactor keeps the sign of `m`). For
/// `m = 0` returns `(0, 0)`.
fn split_val(mut m: Int, p: &Int) -> (i64, Int) {
    if m.is_zero() {
        return (0, m);
    }
    let mut v = 0i64;
    while m.rem_euclid(p).is_zero() {
        m = m.div_exact(p);
        v += 1;
    }
    (v, m)
}

impl Padic {
    /// Builds the zero value of `ℚ_p` with the given `p` and relative precision
    /// `N` (its absolute precision is `N`, i.e. it is only known to be `≡ 0
    /// (mod p^N)`).
    ///
    /// Panics if `precision < 1`. `p` must be prime (checked in debug builds).
    pub fn new(p: Int, precision: i64) -> Padic {
        assert!(precision >= 1, "p-adic precision must be >= 1");
        debug_assert!(p.is_prime_bpsw(), "p-adic modulus must be prime");
        Padic {
            p,
            val: 0,
            unit: Int::ZERO,
            abs_prec: precision,
        }
    }

    /// The multiplicative identity `1` in `ℚ_p` at relative precision `N`.
    pub fn one(p: Int, precision: i64) -> Padic {
        Padic::from_int(p, precision, Int::ONE)
    }

    /// Builds a raw zero with a given absolute precision.
    fn zero_with(p: Int, abs_prec: i64) -> Padic {
        Padic {
            p,
            val: 0,
            unit: Int::ZERO,
            abs_prec,
        }
    }

    /// The `p`-adic value of an integer, to relative precision `N`.
    ///
    /// Panics if `precision < 1`.
    pub fn from_int(p: Int, precision: i64, n: Int) -> Padic {
        assert!(precision >= 1, "p-adic precision must be >= 1");
        debug_assert!(p.is_prime_bpsw(), "p-adic modulus must be prime");
        if n.is_zero() {
            return Padic::zero_with(p, precision);
        }
        let (v, cof) = split_val(n, &p);
        let modulus = p.pow(precision as u32);
        let unit = cof.rem_euclid(&modulus);
        Padic {
            p,
            val: v,
            unit,
            abs_prec: v + precision,
        }
    }

    /// The `p`-adic value of a rational `a/b`, to relative precision `N`.
    ///
    /// This is defined for **every** rational: a factor of `p` in the
    /// denominator simply contributes a negative valuation (so `1/p` becomes a
    /// genuine element of `ℚ_p` with valuation `-1`).
    ///
    /// Panics if `precision < 1`.
    pub fn from_rational(p: Int, precision: i64, r: &Rational) -> Padic {
        assert!(precision >= 1, "p-adic precision must be >= 1");
        debug_assert!(p.is_prime_bpsw(), "p-adic modulus must be prime");
        if r.is_zero() {
            return Padic::zero_with(p, precision);
        }
        let (va, ac) = split_val(r.numerator().clone(), &p);
        let (vb, bc) = split_val(r.denominator().clone(), &p);
        let v = va - vb;
        let modulus = p.pow(precision as u32);
        let binv = bc
            .rem_euclid(&modulus)
            .modinv(&modulus)
            .expect("denominator cofactor is coprime to p");
        let unit = ac.mul(&binv).rem_euclid(&modulus);
        Padic {
            p,
            val: v,
            unit,
            abs_prec: v + precision,
        }
    }

    /// The prime `p`.
    #[inline]
    pub fn prime(&self) -> &Int {
        &self.p
    }

    /// Returns `true` if every known digit is zero (the value is zero to the
    /// working precision).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.unit.is_zero()
    }

    /// The `p`-adic valuation `v_p(x)`, or `None` for zero (whose valuation is
    /// `+∞`).
    #[inline]
    pub fn valuation(&self) -> Option<i64> {
        if self.is_zero() { None } else { Some(self.val) }
    }

    /// The relative precision — the number of significant unit digits kept. For
    /// a zero value this is its absolute precision (the exponent to which it is
    /// known to vanish).
    #[inline]
    pub fn precision(&self) -> i64 {
        self.abs_prec - self.val
    }

    /// The absolute precision `a`: the value is known modulo `p^a`.
    #[inline]
    pub fn absolute_precision(&self) -> i64 {
        self.abs_prec
    }

    /// The `p`-adic absolute value `|x|_p = p^{-v}` as an exact
    /// [`Rational`]; `0` for the zero value.
    pub fn abs_value(&self) -> Rational {
        if self.is_zero() {
            return Rational::ZERO;
        }
        if self.val >= 0 {
            Rational::new(Int::ONE, self.p.pow(self.val as u32))
        } else {
            Rational::from_integer(self.p.pow((-self.val) as u32))
        }
    }

    /// The known base-`p` digits `dᵢ ∈ [0, p)`, **low-order first**, starting at
    /// the [`valuation`](Padic::valuation). Empty for the zero value.
    ///
    /// For example `-1` in `ℤ₂` yields all-ones.
    pub fn digits(&self) -> Vec<Int> {
        let mut out = Vec::new();
        if self.is_zero() {
            return out;
        }
        let rel = self.abs_prec - self.val;
        let mut u = self.unit.clone();
        for _ in 0..rel {
            let (q, r) = u.div_rem(&self.p).expect("p != 0");
            out.push(r);
            u = q;
        }
        out
    }

    /// Returns a copy known only to absolute precision `a` (never *raises*
    /// precision: if `a >= self.abs_prec` the value is returned unchanged).
    fn reduce_to_abs(&self, a: i64) -> Padic {
        if a >= self.abs_prec {
            return self.clone();
        }
        if self.is_zero() {
            return Padic::zero_with(self.p.clone(), a);
        }
        let rel = a - self.val;
        if rel <= 0 {
            return Padic::zero_with(self.p.clone(), a);
        }
        let modulus = self.p.pow(rel as u32);
        Padic {
            p: self.p.clone(),
            val: self.val,
            unit: self.unit.rem_euclid(&modulus),
            abs_prec: a,
        }
    }

    /// Rebuilds a canonical value from `x = p^{v0}·c`, known modulo `p^{aprec}`,
    /// factoring any further powers of `p` out of `c`.
    fn normalize(p: Int, v0: i64, c: Int, aprec: i64) -> Padic {
        let rel = aprec - v0;
        if rel <= 0 {
            return Padic::zero_with(p, aprec);
        }
        let modulus = p.pow(rel as u32);
        let cr = c.rem_euclid(&modulus);
        if cr.is_zero() {
            return Padic::zero_with(p, aprec);
        }
        let (t, unit_full) = split_val(cr, &p);
        // unit_full is coprime to p and already in [0, p^rel); reduce to the
        // now-shorter unit window for good measure.
        let unit_mod = p.pow((rel - t) as u32);
        let unit = unit_full.rem_euclid(&unit_mod);
        Padic {
            p,
            val: v0 + t,
            unit,
            abs_prec: aprec,
        }
    }

    /// Asserts the two operands live in the same `ℚ_p`.
    fn check_same(&self, other: &Padic) {
        assert!(self.p == other.p, "p-adic operands have different primes");
    }

    /// Returns `-self`.
    pub fn neg(&self) -> Padic {
        if self.is_zero() {
            return self.clone();
        }
        let rel = self.abs_prec - self.val;
        let modulus = self.p.pow(rel as u32);
        Padic {
            p: self.p.clone(),
            val: self.val,
            unit: modulus.sub(&self.unit),
            abs_prec: self.abs_prec,
        }
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Padic) -> Padic {
        self.check_same(rhs);
        let a = self.abs_prec.min(rhs.abs_prec);
        if self.is_zero() {
            return rhs.reduce_to_abs(a);
        }
        if rhs.is_zero() {
            return self.reduce_to_abs(a);
        }
        let vmin = self.val.min(rhs.val);
        let c1 = self.unit.mul(&self.p.pow((self.val - vmin) as u32));
        let c2 = rhs.unit.mul(&rhs.p.pow((rhs.val - vmin) as u32));
        Padic::normalize(self.p.clone(), vmin, c1.add(&c2), a)
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Padic) -> Padic {
        self.add(&rhs.neg())
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Padic) -> Padic {
        self.check_same(rhs);
        // For precision bookkeeping, a zero of absolute precision `a` behaves
        // like a value of valuation `a`.
        let veff = |x: &Padic| if x.is_zero() { x.abs_prec } else { x.val };
        if self.is_zero() || rhs.is_zero() {
            return Padic::zero_with(self.p.clone(), veff(self) + veff(rhs));
        }
        let v = self.val + rhs.val;
        let rel = (self.abs_prec - self.val).min(rhs.abs_prec - rhs.val);
        let modulus = self.p.pow(rel as u32);
        let unit = self.unit.mul(&rhs.unit).rem_euclid(&modulus);
        Padic {
            p: self.p.clone(),
            val: v,
            unit,
            abs_prec: v + rel,
        }
    }

    /// Returns `self / rhs`. Division is defined for every nonzero `rhs` in
    /// `ℚ_p`.
    ///
    /// Panics if `rhs` is zero.
    pub fn div(&self, rhs: &Padic) -> Padic {
        self.check_same(rhs);
        assert!(!rhs.is_zero(), "p-adic division by zero");
        if self.is_zero() {
            return Padic::zero_with(self.p.clone(), self.abs_prec - rhs.val);
        }
        let v = self.val - rhs.val;
        let rel = (self.abs_prec - self.val).min(rhs.abs_prec - rhs.val);
        let modulus = self.p.pow(rel as u32);
        let inv = rhs
            .unit
            .modinv(&modulus)
            .expect("unit is coprime to p, hence invertible");
        let unit = self.unit.mul(&inv).rem_euclid(&modulus);
        Padic {
            p: self.p.clone(),
            val: v,
            unit,
            abs_prec: v + rel,
        }
    }

    /// Returns `1 / self`. Panics if `self` is zero.
    pub fn inv(&self) -> Padic {
        Padic::one(self.p.clone(), self.abs_prec - self.val).div(self)
    }

    /// A representative rational congruent to this value modulo its absolute
    /// precision: the exact number `u · p^v` (an integer when `v ≥ 0`, otherwise
    /// `u / p^{-v}`). Returns `0` for the zero value.
    ///
    /// Note this is one representative of the residue class, not a canonical
    /// small fraction; use it for exact re-injection, not for pretty-printing.
    pub fn to_rational(&self) -> Rational {
        if self.is_zero() {
            return Rational::ZERO;
        }
        if self.val >= 0 {
            Rational::from_integer(self.unit.mul(&self.p.pow(self.val as u32)))
        } else {
            Rational::new(self.unit.clone(), self.p.pow((-self.val) as u32))
        }
    }

    /// A square root of `self` via Hensel lifting, or `None` if `self` is not a
    /// square in `ℚ_p`.
    ///
    /// A nonzero `x = p^v·u` is a square iff `v` is even **and** the unit `u` is
    /// a square unit — a quadratic residue mod `p` for odd `p`, or `≡ 1 (mod 8)`
    /// for `p = 2`. Exactly one of the two roots is returned. The result has the
    /// same relative precision as `self`.
    pub fn sqrt(&self) -> Option<Padic> {
        if self.is_zero() {
            // x ≡ 0 (mod p^a) ⇒ √x ≡ 0 (mod p^⌈a/2⌉).
            return Some(Padic::zero_with(self.p.clone(), (self.abs_prec + 1) / 2));
        }
        if self.val % 2 != 0 {
            return None;
        }
        let rel = self.abs_prec - self.val;
        let root_unit = if self.p == Int::from_i64(2) {
            sqrt_unit_2adic(&self.unit, rel)?
        } else {
            sqrt_unit_odd(&self.p, &self.unit, rel)?
        };
        let v = self.val / 2;
        Some(Padic {
            p: self.p.clone(),
            val: v,
            unit: root_unit,
            abs_prec: v + rel,
        })
    }
}

/// Hensel/Newton square root of a unit `u` modulo `p^rel` for **odd** `p`, or
/// `None` if `u` is a non-residue.
fn sqrt_unit_odd(p: &Int, u: &Int, rel: i64) -> Option<Int> {
    // Seed: a square root mod p.
    let mut r = u.sqrt_mod(p)?;
    let mut k = 1i64;
    while k < rel {
        let k2 = (2 * k).min(rel);
        let modulus = p.pow(k2 as u32);
        // Newton step: r ← r - (r² - u)·(2r)⁻¹  (mod p^{k2}).
        let two_r = r.add(&r).rem_euclid(&modulus);
        let inv = two_r
            .modinv(&modulus)
            .expect("2r is a unit for odd p and unit r");
        let f = r.mul(&r).sub(u).rem_euclid(&modulus);
        r = r.sub(&f.mul(&inv)).rem_euclid(&modulus);
        k = k2;
    }
    let modulus = p.pow(rel as u32);
    Some(r.rem_euclid(&modulus))
}

/// Square root of a 2-adic unit `u` modulo `2^rel`, lifting one bit at a time,
/// or `None` if `u` is not a 2-adic square.
fn sqrt_unit_2adic(u: &Int, rel: i64) -> Option<Int> {
    let two = Int::from_i64(2);
    // Seed at k = min(rel, 3): every square unit is ≡ 1 (mod 8), so r = 1 works
    // provided u already matches to that depth.
    let m0 = rel.min(3);
    let seed_mod = two.pow(m0 as u32);
    if !u.sub(&Int::ONE).rem_euclid(&seed_mod).is_zero() {
        return None; // u ≢ 1 (mod 2^m0): not a square this far.
    }
    let mut r = Int::ONE;
    let mut k = m0;
    while k < rel {
        let mod_next = two.pow((k + 1) as u32);
        let bit = two.pow((k - 1) as u32);
        // Try r and r + 2^{k-1}; keep whichever squares to u mod 2^{k+1}.
        let cand2 = r.add(&bit);
        if r.mul(&r).sub(u).rem_euclid(&mod_next).is_zero() {
            // r already correct at this depth.
        } else if cand2.mul(&cand2).sub(u).rem_euclid(&mod_next).is_zero() {
            r = cand2;
        } else {
            return None;
        }
        k += 1;
    }
    let modulus = two.pow(rel as u32);
    Some(r.rem_euclid(&modulus))
}

impl PartialEq for Padic {
    /// Values are equal when they share a prime and agree on every digit up to
    /// their common absolute precision.
    fn eq(&self, other: &Padic) -> bool {
        if self.p != other.p {
            return false;
        }
        let a = self.abs_prec.min(other.abs_prec);
        let x = self.reduce_to_abs(a);
        let y = other.reduce_to_abs(a);
        match (x.is_zero(), y.is_zero()) {
            (true, true) => true,
            (false, false) => x.val == y.val && x.unit == y.unit,
            _ => false,
        }
    }
}

impl Eq for Padic {}

impl fmt::Display for Padic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "O({}^{})", self.p, self.abs_prec);
        }
        let digits = self.digits();
        let mut first = true;
        for (i, d) in digits.iter().enumerate() {
            if d.is_zero() {
                continue;
            }
            if !first {
                f.write_str(" + ")?;
            }
            first = false;
            let exp = self.val + i as i64;
            match exp {
                0 => write!(f, "{d}")?,
                1 => write!(f, "{d}*{}", self.p)?,
                _ => write!(f, "{d}*{}^{exp}", self.p)?,
            }
        }
        if !first {
            f.write_str(" + ")?;
        }
        write!(f, "O({}^{})", self.p, self.abs_prec)
    }
}

impl fmt::Debug for Padic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "Padic(0 + O({}^{}))", self.p, self.abs_prec)
        } else {
            write!(
                f,
                "Padic({}^{} · {} + O({}^{}))",
                self.p, self.val, self.unit, self.p, self.abs_prec
            )
        }
    }
}

impl PartialOrd for Padic {
    /// `ℚ_p` has no compatible order; this only distinguishes equal values
    /// (returning `Some(Equal)`) from unequal ones (`None`).
    fn partial_cmp(&self, other: &Padic) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else {
            None
        }
    }
}

macro_rules! padic_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for Padic {
            type Output = Padic;
            #[inline]
            fn $m(self, rhs: Padic) -> Padic {
                Padic::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&Padic> for &Padic {
            type Output = Padic;
            #[inline]
            fn $m(self, rhs: &Padic) -> Padic {
                Padic::$m(self, rhs)
            }
        }
        impl core::ops::$atr for Padic {
            #[inline]
            fn $am(&mut self, rhs: Padic) {
                *self = Padic::$m(self, &rhs);
            }
        }
    };
}

padic_binop!(Add, add, AddAssign, add_assign);
padic_binop!(Sub, sub, SubAssign, sub_assign);
padic_binop!(Mul, mul, MulAssign, mul_assign);
padic_binop!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for Padic {
    type Output = Padic;
    #[inline]
    fn neg(self) -> Padic {
        Padic::neg(&self)
    }
}

impl core::ops::Neg for &Padic {
    type Output = Padic;
    #[inline]
    fn neg(self) -> Padic {
        Padic::neg(self)
    }
}
