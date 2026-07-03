//! Quadratic irrationals — exact arithmetic in a field `ℚ(√d)`.
//!
//! A [`Quadratic`] is `a + b·√d` with rational `a`, `b` and a squarefree integer
//! `d`. This is a genuine field: `+ − × ÷` are all exact and closed, provided
//! the operands share the same `d` (or one is purely rational). Values are kept
//! canonical — `d` is reduced to its squarefree part (with the square factor
//! folded into `b`), and a value with `b = 0` is a plain rational.
//!
//! For a general real algebraic number of any degree, see the `Algebraic` type.

use core::cmp::Ordering;
use core::fmt;

use crate::int::Int;
use crate::nat::Nat;
use crate::rational::Rational;

/// An element `a + b·√d` of a quadratic field `ℚ(√d)`.
#[derive(Clone)]
pub struct Quadratic {
    a: Rational,
    b: Rational,
    d: Int, // squarefree; d == 1 iff the value is purely rational (b == 0)
}

/// Writes `d = ext² · sf` with `sf` squarefree (carrying the sign of `d`), and
/// returns `(sf, ext)` with `ext > 0`.
fn squarefree_part(d: &Int) -> (Int, Int) {
    if d.is_zero() {
        return (Int::ZERO, Int::ONE);
    }
    let neg = d.is_negative();
    let factors = d.magnitude().factorize(); // sorted primes with multiplicity
    let mut sf = Nat::one();
    let mut ext = Nat::one();
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i].clone();
        let mut e = 0u32;
        while i < factors.len() && factors[i] == p {
            e += 1;
            i += 1;
        }
        if e % 2 == 1 {
            sf = sf.mul(&p);
        }
        ext = ext.mul(&p.pow(e / 2));
    }
    let sf = if neg {
        Int::from(sf).neg()
    } else {
        Int::from(sf)
    };
    (sf, Int::from(ext))
}

impl Quadratic {
    /// Builds `a + b·√d`, canonicalizing `d` to its squarefree part.
    pub fn new(a: Rational, b: Rational, d: Int) -> Quadratic {
        if b.is_zero() {
            return Quadratic::rational(a);
        }
        let (sf, ext) = squarefree_part(&d);
        let b = b.mul(&Rational::from_integer(ext));
        if sf.is_one() {
            // √d was rational: fold b·√d = b into the rational part.
            Quadratic::rational(a.add(&b))
        } else {
            Quadratic { a, b, d: sf }
        }
    }

    /// Builds a purely rational value.
    pub fn rational(a: Rational) -> Quadratic {
        Quadratic {
            a,
            b: Rational::ZERO,
            d: Int::ONE,
        }
    }

    /// Builds `√d` (canonicalized). Panics if `d` is a perfect square? No — a
    /// perfect square simply yields the rational `√d`.
    pub fn sqrt(d: Int) -> Quadratic {
        Quadratic::new(Rational::ZERO, Rational::ONE, d)
    }

    /// Returns the rational part `a`.
    #[inline]
    pub fn rational_part(&self) -> &Rational {
        &self.a
    }

    /// Returns the coefficient `b` of `√d`.
    #[inline]
    pub fn surd_coefficient(&self) -> &Rational {
        &self.b
    }

    /// Returns the (squarefree) radicand `d`.
    #[inline]
    pub fn radicand(&self) -> &Int {
        &self.d
    }

    /// Returns `true` if this value is purely rational.
    #[inline]
    pub fn is_rational(&self) -> bool {
        self.b.is_zero()
    }

    /// Checks that a binary operation is well-defined, returning the shared `d`.
    /// Panics if both operands are irrational with different radicands.
    fn common_d(&self, other: &Quadratic) -> Int {
        if self.is_rational() {
            other.d.clone()
        } else if other.is_rational() || self.d == other.d {
            self.d.clone()
        } else {
            panic!(
                "Quadratic: operands lie in different fields (√{} vs √{})",
                self.d, other.d
            )
        }
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Quadratic) -> Quadratic {
        let d = self.common_d(rhs);
        Quadratic::new(self.a.add(&rhs.a), self.b.add(&rhs.b), d)
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Quadratic) -> Quadratic {
        let d = self.common_d(rhs);
        Quadratic::new(self.a.sub(&rhs.a), self.b.sub(&rhs.b), d)
    }

    /// Returns `-self`.
    pub fn neg(&self) -> Quadratic {
        Quadratic {
            a: self.a.neg(),
            b: self.b.neg(),
            d: self.d.clone(),
        }
    }

    /// Returns the conjugate `a − b·√d`.
    pub fn conjugate(&self) -> Quadratic {
        Quadratic {
            a: self.a.clone(),
            b: self.b.neg(),
            d: self.d.clone(),
        }
    }

    /// Returns the field norm `a² − b²·d` (a rational).
    pub fn norm(&self) -> Rational {
        let d = Rational::from_integer(self.d.clone());
        self.a.mul(&self.a).sub(&self.b.mul(&self.b).mul(&d))
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Quadratic) -> Quadratic {
        let d = self.common_d(rhs);
        let dr = Rational::from_integer(d.clone());
        // (a1 + b1√d)(a2 + b2√d) = (a1a2 + b1b2 d) + (a1b2 + a2b1)√d
        let a = self.a.mul(&rhs.a).add(&self.b.mul(&rhs.b).mul(&dr));
        let b = self.a.mul(&rhs.b).add(&rhs.a.mul(&self.b));
        Quadratic::new(a, b, d)
    }

    /// Returns `1/self`. Panics if `self` is zero.
    pub fn recip(&self) -> Quadratic {
        let n = self.norm();
        assert!(!n.is_zero(), "Quadratic::recip: reciprocal of zero");
        // 1/(a+b√d) = (a−b√d)/(a²−b²d)
        Quadratic {
            a: self.a.div(&n),
            b: self.b.neg().div(&n),
            d: self.d.clone(),
        }
    }

    /// Returns `self / rhs`. Panics if `rhs` is zero.
    pub fn div(&self, rhs: &Quadratic) -> Quadratic {
        self.mul(&rhs.recip())
    }

    /// Returns `self` raised to the (non-negative) power `n`.
    pub fn pow(&self, mut n: u32) -> Quadratic {
        let mut base = self.clone();
        let mut acc = Quadratic::rational(Rational::ONE);
        while n > 0 {
            if n & 1 == 1 {
                acc = acc.mul(&base);
            }
            n >>= 1;
            if n > 0 {
                base = base.mul(&base);
            }
        }
        acc
    }

    /// Rounds this value to a [`Float`](crate::float::Float). Requires `d ≥ 0`
    /// (a real value).
    pub fn to_float(
        &self,
        precision: u64,
        mode: crate::float::RoundingMode,
    ) -> crate::float::Float {
        use crate::float::Float;
        assert!(
            !self.d.is_negative(),
            "Quadratic::to_float: value is not real (d < 0)"
        );
        let work = precision + 16;
        let a = Float::from_rational(&self.a, work, mode);
        if self.b.is_zero() {
            return a.round(precision, mode);
        }
        let b = Float::from_rational(&self.b, work, mode);
        let root = Float::from_int(&self.d, work, mode).sqrt(work, mode);
        a.add(&b.mul(&root, work, mode), precision, mode)
    }
}

impl PartialEq for Quadratic {
    fn eq(&self, other: &Self) -> bool {
        if self.is_rational() && other.is_rational() {
            return self.a == other.a;
        }
        self.a == other.a && self.b == other.b && self.d == other.d
    }
}

impl Eq for Quadratic {}

impl PartialOrd for Quadratic {
    /// Exact ordering for real values sharing a radicand (or rationals);
    /// `None` when the two lie in different real fields or `d < 0`.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reduce to sign of (self - other) = A + B√d.
        if !self.is_rational() && !other.is_rational() && self.d != other.d {
            return None;
        }
        let d = if self.is_rational() {
            &other.d
        } else {
            &self.d
        };
        if d.is_negative() {
            return None; // non-real
        }
        let big_a = self.a.sub(&other.a);
        let big_b = self.b.sub(&other.b);
        Some(sign_of_a_plus_b_root(&big_a, &big_b, d))
    }
}

/// Sign of `A + B·√d` for `d ≥ 0`.
fn sign_of_a_plus_b_root(a: &Rational, b: &Rational, d: &Int) -> Ordering {
    if b.is_zero() {
        return a.signum().cmp(&0);
    }
    let dr = Rational::from_integer(d.clone());
    let a2 = a.mul(a);
    let b2d = b.mul(b).mul(&dr);
    match (a.signum(), b.signum()) {
        (sa, sb) if sa >= 0 && sb >= 0 => {
            if a.is_zero() && b.is_zero() {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        }
        (sa, sb) if sa <= 0 && sb <= 0 => Ordering::Less,
        // a > 0, b < 0: A + B√d > 0  ⟺  a² > b²d
        (sa, _) if sa > 0 => a2.cmp(&b2d),
        // a < 0, b > 0: A + B√d > 0  ⟺  b²d > a²
        _ => b2d.cmp(&a2),
    }
}

impl fmt::Display for Quadratic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_rational() {
            return fmt::Display::fmt(&self.a, f);
        }
        if !self.a.is_zero() {
            write!(f, "{} + ", self.a)?;
        }
        write!(f, "{}·√{}", self.b, self.d)
    }
}

impl fmt::Debug for Quadratic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quadratic({} + {}·√{})", self.a, self.b, self.d)
    }
}

impl From<Rational> for Quadratic {
    #[inline]
    fn from(r: Rational) -> Quadratic {
        Quadratic::rational(r)
    }
}

impl From<Int> for Quadratic {
    #[inline]
    fn from(n: Int) -> Quadratic {
        Quadratic::rational(Rational::from_integer(n))
    }
}

macro_rules! quad_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for Quadratic {
            type Output = Quadratic;
            #[inline]
            fn $m(self, rhs: Quadratic) -> Quadratic {
                Quadratic::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&Quadratic> for &Quadratic {
            type Output = Quadratic;
            #[inline]
            fn $m(self, rhs: &Quadratic) -> Quadratic {
                Quadratic::$m(self, rhs)
            }
        }
        impl core::ops::$atr for Quadratic {
            #[inline]
            fn $am(&mut self, rhs: Quadratic) {
                *self = Quadratic::$m(self, &rhs);
            }
        }
    };
}

quad_binop!(Add, add, AddAssign, add_assign);
quad_binop!(Sub, sub, SubAssign, sub_assign);
quad_binop!(Mul, mul, MulAssign, mul_assign);
quad_binop!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for Quadratic {
    type Output = Quadratic;
    #[inline]
    fn neg(self) -> Quadratic {
        Quadratic::neg(&self)
    }
}
