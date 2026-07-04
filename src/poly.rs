//! Generic dense univariate polynomials `Poly<T>`.
//!
//! Coefficients are stored low-to-high (`coeffs[i]` multiplies `xⁱ`) and kept
//! trimmed so the leading coefficient is nonzero (the zero polynomial has no
//! coefficients). Like [`Complex`](crate::complex::Complex), `Poly<T>` composes
//! with any component type exposing the right operators: ring operations
//! (`+ - *`, evaluation, derivative) need only `+ - *`; polynomial division and
//! GCD additionally need component division, so they are available for field
//! components (`Rational`, `Decimal`, `FixedFloat`) but not for `Int`.
//!
//! `T::default()` is taken to be the additive identity (zero), matching all of
//! this crate's numeric types.

use alloc::vec::Vec;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Degree (in the smaller operand's coefficient count) at or above which
/// `Poly::mul` switches from schoolbook to Karatsuba.
const POLY_KARATSUBA_THRESHOLD: usize = 24;

/// A dense univariate polynomial with coefficients of type `T`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Poly<T> {
    coeffs: Vec<T>,
}

/// Highest index holding a nonzero entry, or `None` if all are zero.
fn top_nonzero<T: Default + PartialEq>(v: &[T]) -> Option<usize> {
    v.iter().rposition(|c| *c != T::default())
}

impl<T: Clone + Default + PartialEq> Poly<T> {
    /// Builds a polynomial from low-to-high coefficients, trimming trailing zeros.
    pub fn new(mut coeffs: Vec<T>) -> Poly<T> {
        match top_nonzero(&coeffs) {
            Some(i) => coeffs.truncate(i + 1),
            None => coeffs.clear(),
        }
        Poly { coeffs }
    }

    /// The zero polynomial.
    pub fn zero() -> Poly<T> {
        Poly { coeffs: Vec::new() }
    }

    /// The constant polynomial `c`.
    pub fn constant(c: T) -> Poly<T> {
        Poly::new(alloc::vec![c])
    }

    /// The monomial `c·x^degree`.
    pub fn monomial(c: T, degree: usize) -> Poly<T> {
        let mut v = Vec::with_capacity(degree + 1);
        v.resize(degree, T::default());
        v.push(c);
        Poly::new(v)
    }

    /// Returns the coefficients, low-to-high (empty for the zero polynomial).
    #[inline]
    pub fn coeffs(&self) -> &[T] {
        &self.coeffs
    }

    /// Returns `true` if this is the zero polynomial.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Returns the degree, or `None` for the zero polynomial.
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    /// Returns the coefficient of `xⁱ` (zero if out of range).
    pub fn coeff(&self, i: usize) -> T {
        self.coeffs.get(i).cloned().unwrap_or_default()
    }

    /// Returns the leading coefficient, or `None` for the zero polynomial.
    pub fn leading(&self) -> Option<&T> {
        self.coeffs.last()
    }
}

impl<T> Poly<T>
where
    T: Clone + Default + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Poly<T>) -> Poly<T> {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.coeff(i) + rhs.coeff(i));
        }
        Poly::new(out)
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Poly<T>) -> Poly<T> {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.coeff(i) - rhs.coeff(i));
        }
        Poly::new(out)
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Poly<T>) -> Poly<T> {
        if self.is_zero() || rhs.is_zero() {
            return Poly::zero();
        }
        // Karatsuba above a threshold; schoolbook below. Karatsuba trades one
        // coefficient multiplication per split for a few additions, which is a
        // win whenever a coefficient product costs more than an add (e.g. exact
        // `Rational`/`Int` coefficients).
        if self.coeffs.len().min(rhs.coeffs.len()) < POLY_KARATSUBA_THRESHOLD {
            return self.mul_schoolbook(rhs);
        }
        let m = self.coeffs.len().max(rhs.coeffs.len()) / 2;
        let (a0, a1) = self.split_at(m);
        let (b0, b1) = rhs.split_at(m);
        let z0 = Poly::mul(&a0, &b0);
        let z2 = Poly::mul(&a1, &b1);
        // z1 = (a0 + a1)(b0 + b1) − z0 − z2
        let mid = Poly::mul(&Poly::add(&a0, &a1), &Poly::add(&b0, &b1));
        let z1 = Poly::sub(&Poly::sub(&mid, &z0), &z2);
        // z0 + z1·x^m + z2·x^(2m)
        let r = Poly::add(&z0, &z1.shift_up(m));
        Poly::add(&r, &z2.shift_up(2 * m))
    }

    /// Quadratic schoolbook multiplication.
    fn mul_schoolbook(&self, rhs: &Poly<T>) -> Poly<T> {
        let mut out = alloc::vec![T::default(); self.coeffs.len() + rhs.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in rhs.coeffs.iter().enumerate() {
                let prod = a.clone() * b.clone();
                out[i + j] = out[i + j].clone() + prod;
            }
        }
        Poly::new(out)
    }

    /// Splits into `(low, high)` where `self = low + high·x^k` (low has the
    /// coefficients of degree `< k`).
    fn split_at(&self, k: usize) -> (Poly<T>, Poly<T>) {
        if k >= self.coeffs.len() {
            return (self.clone(), Poly::zero());
        }
        (
            Poly::new(self.coeffs[..k].to_vec()),
            Poly::new(self.coeffs[k..].to_vec()),
        )
    }

    /// Returns `self · x^k` (prepends `k` zero coefficients).
    fn shift_up(&self, k: usize) -> Poly<T> {
        if self.is_zero() || k == 0 {
            return self.clone();
        }
        let mut v = Vec::with_capacity(self.coeffs.len() + k);
        v.resize(k, T::default());
        v.extend_from_slice(&self.coeffs);
        Poly::new(v)
    }

    /// Returns `self · scalar`.
    pub fn scalar_mul(&self, scalar: &T) -> Poly<T> {
        Poly::new(
            self.coeffs
                .iter()
                .map(|c| c.clone() * scalar.clone())
                .collect(),
        )
    }

    /// Evaluates the polynomial at `x` (Horner's method).
    pub fn eval(&self, x: &T) -> T {
        let mut acc = T::default();
        for c in self.coeffs.iter().rev() {
            acc = acc * x.clone() + c.clone();
        }
        acc
    }

    /// Returns the formal derivative `d/dx`.
    pub fn derivative(&self) -> Poly<T> {
        if self.coeffs.len() < 2 {
            return Poly::zero();
        }
        let mut out = Vec::with_capacity(self.coeffs.len() - 1);
        for (i, c) in self.coeffs.iter().enumerate().skip(1) {
            // i·c via repeated addition (no integer-scalar trait required).
            let mut acc = T::default();
            for _ in 0..i {
                acc = acc + c.clone();
            }
            out.push(acc);
        }
        Poly::new(out)
    }
}

impl<T> Poly<T>
where
    T: Clone
        + Default
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>,
{
    /// Returns `-self`.
    pub fn neg(&self) -> Poly<T> {
        Poly {
            coeffs: self.coeffs.iter().map(|c| -c.clone()).collect(),
        }
    }
}

impl<T> Poly<T>
where
    T: Clone
        + Default
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    /// Divides `self` by `divisor`, returning `(quotient, remainder)` with
    /// `deg(remainder) < deg(divisor)`. Requires a field component type. Panics
    /// if `divisor` is the zero polynomial.
    pub fn div_rem(&self, divisor: &Poly<T>) -> (Poly<T>, Poly<T>) {
        let dd = divisor
            .degree()
            .expect("Poly::div_rem: division by zero polynomial");
        let lead = divisor.leading().unwrap().clone();
        let mut rem = self.coeffs.clone();
        let mut quot = alloc::vec![T::default(); self.coeffs.len().saturating_sub(dd)];
        while let Some(rd) = top_nonzero(&rem) {
            if rd < dd {
                break;
            }
            let coef = rem[rd].clone() / lead.clone();
            let shift = rd - dd;
            for (i, dc) in divisor.coeffs.iter().enumerate() {
                rem[shift + i] = rem[shift + i].clone() - coef.clone() * dc.clone();
            }
            quot[shift] = coef;
        }
        (Poly::new(quot), Poly::new(rem))
    }

    /// Returns the remainder of `self / divisor`.
    pub fn rem(&self, divisor: &Poly<T>) -> Poly<T> {
        self.div_rem(divisor).1
    }

    /// Returns the monic form (leading coefficient scaled to one), or the zero
    /// polynomial unchanged.
    pub fn monic(&self) -> Poly<T> {
        match self.leading() {
            None => Poly::zero(),
            Some(lead) => {
                let inv_lead = lead.clone();
                Poly::new(
                    self.coeffs
                        .iter()
                        .map(|c| c.clone() / inv_lead.clone())
                        .collect(),
                )
            }
        }
    }

    /// Returns the monic GCD of `self` and `other` (Euclid's algorithm over the
    /// field of coefficients).
    pub fn gcd(&self, other: &Poly<T>) -> Poly<T> {
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a.monic()
    }
}

impl<T: fmt::Display + Clone + Default + PartialEq> fmt::Display for Poly<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate().rev() {
            if *c == T::default() {
                continue;
            }
            if !first {
                f.write_str(" + ")?;
            }
            first = false;
            match i {
                0 => write!(f, "{c}")?,
                1 => write!(f, "{c}·x")?,
                _ => write!(f, "{c}·x^{i}")?,
            }
        }
        Ok(())
    }
}

macro_rules! poly_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl<T> core::ops::$tr for Poly<T>
        where
            T: Clone + Default + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
        {
            type Output = Poly<T>;
            #[inline]
            fn $m(self, rhs: Poly<T>) -> Poly<T> {
                Poly::$m(&self, &rhs)
            }
        }
        impl<T> core::ops::$tr<&Poly<T>> for &Poly<T>
        where
            T: Clone + Default + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
        {
            type Output = Poly<T>;
            #[inline]
            fn $m(self, rhs: &Poly<T>) -> Poly<T> {
                Poly::$m(self, rhs)
            }
        }
        impl<T> core::ops::$atr for Poly<T>
        where
            T: Clone + Default + PartialEq + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
        {
            #[inline]
            fn $am(&mut self, rhs: Poly<T>) {
                *self = Poly::$m(self, &rhs);
            }
        }
    };
}

poly_binop!(Add, add, AddAssign, add_assign);
poly_binop!(Sub, sub, SubAssign, sub_assign);
poly_binop!(Mul, mul, MulAssign, mul_assign);

// ===========================================================================
// Real-root isolation over ℚ (Sturm sequences).
//
// These operate on `Poly<Rational>` and power both the public root-finding API
// below and the `Algebraic` number type. `T::default()` is the additive
// identity, so `Rational::default() == 0`.
// ===========================================================================

#[cfg(feature = "rational")]
use crate::rational::Rational;

/// Number of sign variations of a Sturm chain evaluated at `x` (zeros skipped).
#[cfg(feature = "rational")]
pub fn sturm_variations(chain: &[Poly<Rational>], x: &Rational) -> usize {
    let mut last = 0i32;
    let mut count = 0;
    for p in chain {
        let s = p.eval(x).signum();
        if s != 0 {
            if last != 0 && s != last {
                count += 1;
            }
            last = s;
        }
    }
    count
}

/// Number of distinct real roots of the chain's (squarefree) polynomial in the
/// half-open interval `(lo, hi]`.
#[cfg(feature = "rational")]
pub fn sturm_count(chain: &[Poly<Rational>], lo: &Rational, hi: &Rational) -> usize {
    sturm_variations(chain, lo).saturating_sub(sturm_variations(chain, hi))
}

#[cfg(feature = "rational")]
impl Poly<Rational> {
    /// Returns the squarefree part `self / gcd(self, self′)` (monic).
    pub fn squarefree_part(&self) -> Poly<Rational> {
        if self.degree().unwrap_or(0) < 1 {
            return self.monic();
        }
        let g = self.gcd(&self.derivative());
        self.div_rem(&g).0.monic()
    }

    /// Returns the Sturm chain `p₀ = self, p₁ = self′, pᵢ = −(pᵢ₋₂ mod pᵢ₋₁)`.
    /// For a correct real-root count the polynomial should be squarefree (see
    /// [`squarefree_part`](Poly::squarefree_part)).
    pub fn sturm_chain(&self) -> alloc::vec::Vec<Poly<Rational>> {
        let mut chain = alloc::vec![self.clone(), self.derivative()];
        while !chain.last().unwrap().is_zero() {
            let n = chain.len();
            let r = chain[n - 2].rem(&chain[n - 1]);
            if r.is_zero() {
                break;
            }
            chain.push(r.neg());
        }
        chain
    }

    /// A Cauchy bound: every real root lies in the open interval `(-b, b)`.
    fn real_root_bound(&self) -> Rational {
        let lead = match self.leading() {
            Some(c) => c.abs(),
            None => return Rational::ONE,
        };
        let mut m = Rational::ZERO;
        let deg = self.degree().unwrap_or(0);
        for i in 0..deg {
            let r = Rational::div(&self.coeff(i).abs(), &lead);
            if r > m {
                m = r;
            }
        }
        Rational::add(&m, &Rational::ONE)
    }

    /// Counts the distinct real roots of `self` in `(lo, hi]`.
    pub fn count_real_roots_in(&self, lo: &Rational, hi: &Rational) -> usize {
        let sf = self.squarefree_part();
        if sf.degree().unwrap_or(0) < 1 {
            return 0;
        }
        sturm_count(&sf.sturm_chain(), lo, hi)
    }

    /// Returns the total number of distinct real roots of `self`.
    pub fn real_root_count(&self) -> usize {
        let sf = self.squarefree_part();
        if sf.degree().unwrap_or(0) < 1 {
            return 0;
        }
        let b = sf.real_root_bound();
        sturm_count(&sf.sturm_chain(), &Rational::neg(&b), &b)
    }

    /// Isolates every distinct real root, returning half-open rational intervals
    /// `(lo, hi]` each containing exactly one root (an exact rational root is
    /// returned as a degenerate `[r, r]`). Intervals come back in increasing
    /// order.
    pub fn isolate_real_roots(&self) -> alloc::vec::Vec<(Rational, Rational)> {
        let mut out = alloc::vec::Vec::new();
        let sf = self.squarefree_part();
        if sf.degree().unwrap_or(0) < 1 {
            return out;
        }
        let chain = sf.sturm_chain();
        let two = Rational::from_integer(crate::int::Int::from_i64(2));
        let b = sf.real_root_bound();
        // Work stack of intervals still to resolve. The counting is half-open
        // `(lo, hi]`, so a root exactly at a bisection midpoint stays in the
        // left half and is never double-counted.
        let neg_b = Rational::neg(&b);
        let mut stack = alloc::vec![(neg_b, b)];
        while let Some((lo, hi)) = stack.pop() {
            let c = sturm_count(&chain, &lo, &hi);
            if c == 0 {
                continue;
            }
            if c == 1 {
                out.push((lo, hi));
                continue;
            }
            let mid = Rational::div(&Rational::add(&lo, &hi), &two);
            stack.push((lo, mid.clone()));
            stack.push((mid, hi));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

#[cfg(all(feature = "rational", feature = "float"))]
impl Poly<Rational> {
    /// Returns every distinct real root as a [`Float`](crate::float::Float),
    /// each correctly isolated then refined to `precision` bits.
    pub fn real_roots(
        &self,
        precision: u64,
        mode: crate::float::RoundingMode,
    ) -> alloc::vec::Vec<crate::float::Float> {
        use crate::float::Float;
        let sf = self.squarefree_part();
        let two = Rational::from_integer(crate::int::Int::from_i64(2));
        self.isolate_real_roots()
            .into_iter()
            .map(|(mut lo, mut hi)| {
                // Bisect until both ends round to the same float.
                for _ in 0..(precision + 64) {
                    if lo == hi {
                        break;
                    }
                    let flo = Float::from_rational(&lo, precision, mode);
                    let fhi = Float::from_rational(&hi, precision, mode);
                    if flo == fhi {
                        return flo;
                    }
                    let mid = Rational::div(&Rational::add(&lo, &hi), &two);
                    let sm = sf.eval(&mid).signum();
                    if sm == 0 {
                        lo = mid.clone();
                        hi = mid;
                    } else if sm == sf.eval(&hi).signum() {
                        // Anchor on hi (never a spurious root of the shared poly).
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                Float::from_rational(
                    &Rational::div(&Rational::add(&lo, &hi), &two),
                    precision,
                    mode,
                )
            })
            .collect()
    }
}

#[cfg(all(feature = "poly", feature = "rational"))]
impl Poly<crate::rational::Rational> {
    /// Factors this polynomial into monic irreducible factors over ℚ, returned as
    /// `(factor, multiplicity)` pairs (constants yield an empty list). The product
    /// of the factors raised to their multiplicities equals this polynomial made
    /// monic. Uses Berlekamp–Zassenhaus with van Hoeij's LLL recombination
    /// (factor mod p, Hensel lift, recombine).
    pub fn factor(&self) -> alloc::vec::Vec<(Poly<crate::rational::Rational>, usize)> {
        crate::poly_factor::factor_rational(self)
    }
}
