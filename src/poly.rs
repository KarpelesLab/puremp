//! Generic dense univariate polynomials `Poly<T>`.
//!
//! Coefficients are stored low-to-high (`coeffs[i]` multiplies `xⁱ`) and kept
//! trimmed so the leading coefficient is nonzero (the zero polynomial has no
//! coefficients). Like [`Complex`](crate::complex::Complex), `Poly<T>` composes
//! with any component type that is a [`Ring`]: ring operations
//! (`+ - *`, evaluation, derivative) need only the [`Ring`]
//! operators; polynomial division and GCD additionally need component division,
//! so they are available for field components (`Rational`, `Decimal`,
//! `FixedFloat`, `ModInt`, `GfElement`) but not for `Int`.
//!
//! The additive identity of the component ring is obtained from
//! [`Ring::zero`] of an available coefficient, so
//! context-carrying rings (`ModInt`, `GfElement`) work too; the zero polynomial
//! is the empty-coefficient one.

use crate::ring::Ring;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Div;

/// Degree (in the smaller operand's coefficient count) at or above which
/// `Poly::mul` switches from schoolbook to Karatsuba.
const POLY_KARATSUBA_THRESHOLD: usize = 24;

/// A dense univariate polynomial with coefficients of type `T`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Poly<T> {
    coeffs: Vec<T>,
}

/// Highest index holding a nonzero entry, or `None` if all are zero.
fn top_nonzero<T: Ring>(v: &[T]) -> Option<usize> {
    v.iter().rposition(|c| !c.is_zero())
}

impl<T: Ring> Poly<T> {
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
        v.resize(degree, c.zero());
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
    ///
    /// The out-of-range zero is derived from an existing coefficient's ring, so
    /// this panics only on the zero polynomial (which has no coefficient to draw
    /// a ring from) — use [`is_zero`](Poly::is_zero) to guard that case.
    pub fn coeff(&self, i: usize) -> T {
        match self.coeffs.get(i) {
            Some(c) => c.clone(),
            None => self
                .coeffs
                .first()
                .expect("Poly::coeff: cannot derive a zero from the zero polynomial")
                .zero(),
        }
    }

    /// Returns the leading coefficient, or `None` for the zero polynomial.
    pub fn leading(&self) -> Option<&T> {
        self.coeffs.last()
    }
}

impl<T: Ring> Poly<T> {
    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Poly<T>) -> Poly<T> {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            match (self.coeffs.get(i), rhs.coeffs.get(i)) {
                (Some(a), Some(b)) => out.push(a.clone() + b.clone()),
                (Some(a), None) => out.push(a.clone()),
                (None, Some(b)) => out.push(b.clone()),
                (None, None) => unreachable!("i < max(len) so one side is in range"),
            }
        }
        Poly::new(out)
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Poly<T>) -> Poly<T> {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            match (self.coeffs.get(i), rhs.coeffs.get(i)) {
                (Some(a), Some(b)) => out.push(a.clone() - b.clone()),
                (Some(a), None) => out.push(a.clone()),
                (None, Some(b)) => out.push(-b.clone()),
                (None, None) => unreachable!("i < max(len) so one side is in range"),
            }
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
        // `mul` guarantees both operands are nonzero, so a coefficient exists to
        // derive the ring's zero from.
        let zero = self.coeffs[0].zero();
        let mut out = alloc::vec![zero; self.coeffs.len() + rhs.coeffs.len() - 1];
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
        v.resize(k, self.coeffs[0].zero());
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
        let mut acc = x.zero();
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
            let mut acc = c.zero();
            for _ in 0..i {
                acc = acc + c.clone();
            }
            out.push(acc);
        }
        Poly::new(out)
    }
}

impl<T: Ring> Poly<T> {
    /// Returns `-self`.
    pub fn neg(&self) -> Poly<T> {
        Poly {
            coeffs: self.coeffs.iter().map(|c| -c.clone()).collect(),
        }
    }
}

impl<T> Poly<T>
where
    T: Ring + Div<Output = T>,
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
        let mut quot = alloc::vec![lead.zero(); self.coeffs.len().saturating_sub(dd)];
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

impl<T: fmt::Display + Ring> fmt::Display for Poly<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate().rev() {
            if c.is_zero() {
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
        impl<T: Ring> core::ops::$tr for Poly<T> {
            type Output = Poly<T>;
            #[inline]
            fn $m(self, rhs: Poly<T>) -> Poly<T> {
                Poly::$m(&self, &rhs)
            }
        }
        impl<T: Ring> core::ops::$tr<&Poly<T>> for &Poly<T> {
            type Output = Poly<T>;
            #[inline]
            fn $m(self, rhs: &Poly<T>) -> Poly<T> {
                Poly::$m(self, rhs)
            }
        }
        impl<T: Ring> core::ops::$atr for Poly<T> {
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
// below and the `Algebraic` number type. `Rational::ZERO` (the `Ring` additive
// identity) is the zero coefficient.
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

// ===========================================================================
// Subresultant polynomial remainder sequence (integer arithmetic).
//
// The naive remainder sequence over ℚ (Euclid for `gcd`, and `pᵢ = −(pᵢ₋₂ mod
// pᵢ₋₁)` for Sturm chains) suffers the classical coefficient blow-up: the
// rational coefficients grow exponentially in numerator and denominator size.
//
// The *subresultant* PRS (Collins 1967; Brown & Traub; Cohen, *A Course in
// Computational Algebraic Number Theory* §3.3; Ducos 2000) instead works on
// primitive integer polynomials, replacing each Euclidean remainder by an exact
// pseudo-remainder scaled down by the subresultant coefficient. Concretely, with
// `δ = deg Rᵢ₋₁ − deg Rᵢ`,
//
//     Rᵢ₊₁ = prem(Rᵢ₋₁, Rᵢ) / (g · h^δ),
//
// where `prem(A, B) = lc(B)^{deg A − deg B + 1}·A mod B` is the integer
// pseudo-remainder and `g`, `h` (the `ψ` sequence) are updated by
// `g ← lc(Rᵢ₋₁)`, `h ← g^δ / h^{δ−1}`. Every division is exact and the
// coefficients stay polynomially bounded (Hadamard, not exponential).
//
// This module keeps integer polynomials as `Vec<Int>`, low-to-high, with
// trailing zeros trimmed (empty = zero polynomial).
// ===========================================================================

/// Degree of an integer polynomial, or `None` for the zero polynomial.
#[cfg(feature = "rational")]
fn ip_deg(p: &[crate::int::Int]) -> Option<usize> {
    p.iter().rposition(|c| !c.is_zero())
}

/// Trims trailing zero coefficients in place-ish (returns the trimmed vector).
#[cfg(feature = "rational")]
fn ip_trim(mut p: alloc::vec::Vec<crate::int::Int>) -> alloc::vec::Vec<crate::int::Int> {
    match ip_deg(&p) {
        Some(d) => p.truncate(d + 1),
        None => p.clear(),
    }
    p
}

/// Leading coefficient of a nonzero integer polynomial.
#[cfg(feature = "rational")]
fn ip_lead(p: &[crate::int::Int]) -> &crate::int::Int {
    &p[ip_deg(p).expect("ip_lead: zero polynomial")]
}

/// Content: the positive gcd of the coefficients (`0` for the zero polynomial).
#[cfg(feature = "rational")]
fn ip_content(p: &[crate::int::Int]) -> crate::int::Int {
    let mut g = crate::int::Int::ZERO;
    for c in p {
        g = g.gcd(c);
    }
    g
}

/// Primitive part: divides out the content, keeping the sign of the leading
/// coefficient (so the result is a *positive* rational multiple of `p`).
#[cfg(feature = "rational")]
fn ip_primitive(p: &[crate::int::Int]) -> alloc::vec::Vec<crate::int::Int> {
    let c = ip_content(p);
    if c.is_zero() || c.is_one() {
        return p.to_vec();
    }
    p.iter().map(|x| x.div_exact(&c)).collect()
}

/// Integer pseudo-remainder `prem(a, b) = lc(b)^{deg a − deg b + 1}·a mod b`.
///
/// Computed by Knuth's Algorithm R (TAOCP 4.6.1): iterating `deg a − deg b + 1`
/// times, each time scaling the running remainder by `lc(b)` and cancelling its
/// leading term against `b`. Requires `deg a ≥ deg b` and `b ≠ 0`.
#[cfg(feature = "rational")]
fn ip_prem(a: &[crate::int::Int], b: &[crate::int::Int]) -> alloc::vec::Vec<crate::int::Int> {
    use crate::int::Int;
    let n = ip_deg(b).expect("ip_prem: division by zero polynomial");
    let m = match ip_deg(a) {
        Some(m) if m >= n => m,
        _ => return a.to_vec(), // deg a < deg b: zero iterations, prem = a
    };
    let l = ip_lead(b).clone();
    let mut r = a.to_vec();
    for k in (0..=(m - n)).rev() {
        // Cancel the leading term of `r` (if present at degree n+k), then scale.
        if ip_deg(&r) == Some(n + k) {
            let coef = r[n + k].clone(); // lc(r) *before* scaling
            for c in r.iter_mut() {
                *c = Int::mul(c, &l);
            }
            for (i, bc) in b.iter().enumerate() {
                r[k + i] = Int::sub(&r[k + i], &Int::mul(&coef, bc));
            }
        } else {
            for c in r.iter_mut() {
                *c = Int::mul(c, &l);
            }
        }
        r = ip_trim(r);
    }
    r
}

/// Clears denominators of a `Poly<Rational>` and returns the *primitive* integer
/// polynomial that is a positive rational multiple of it (empty for zero).
#[cfg(feature = "rational")]
fn rational_to_primitive_int(p: &Poly<Rational>) -> alloc::vec::Vec<crate::int::Int> {
    use crate::int::Int;
    if p.is_zero() {
        return alloc::vec::Vec::new();
    }
    // L = lcm of all denominators (positive).
    let mut l = Int::ONE;
    for c in p.coeffs() {
        let d = c.denominator();
        let g = l.gcd(d);
        l = l.div_exact(&g).mul(d);
    }
    // Coefficient i becomes numᵢ · (L / denᵢ), an integer.
    let ints: alloc::vec::Vec<Int> = p
        .coeffs()
        .iter()
        .map(|c| c.numerator().mul(&l.div_exact(c.denominator())))
        .collect();
    ip_primitive(&ints)
}

/// The subresultant PRS for a Sturm chain, on integer polynomials.
///
/// Given `p0`, `p1` that are *positive* rational multiples of `p` and `p′` (a
/// squarefree `p`), returns integer polynomials `R₀, R₁, …, R_k` where each
/// `Rᵢ` is a positive rational multiple of the true Sturm polynomial `pᵢ`
/// (`p₀ = p`, `p₁ = p′`, `pᵢ = −(pᵢ₋₂ mod pᵢ₋₁)`). Being a positive multiple,
/// `Rᵢ(x)` has the same sign as `pᵢ(x)` at every `x`, so the sequence has an
/// identical sign-variation count — a genuine Sturm chain — while its
/// coefficients stay polynomially bounded.
///
/// Sign bookkeeping: with `Rᵢ₋₁ = a·pᵢ₋₁`, `Rᵢ = b·pᵢ` (`a, b > 0`),
/// `rem(Rᵢ₋₁, Rᵢ) = a·rem(pᵢ₋₁, pᵢ) = −a·pᵢ₊₁`, hence
/// `prem(Rᵢ₋₁, Rᵢ) = −a·lc(Rᵢ)^{δ+1}·pᵢ₊₁`, whose sign is
/// `−sign(lc Rᵢ)^{δ+1}`. Multiplying by that sign yields a positive multiple of
/// `pᵢ₊₁`. Dividing by the (positive) subresultant magnitude `g·h^δ` preserves
/// the sign and keeps coefficients bounded.
#[cfg(feature = "rational")]
fn ip_sturm_chain(
    p0: alloc::vec::Vec<crate::int::Int>,
    p1: alloc::vec::Vec<crate::int::Int>,
) -> alloc::vec::Vec<alloc::vec::Vec<crate::int::Int>> {
    use crate::int::Int;
    let mut chain = alloc::vec![p0];
    if ip_deg(&p1).is_none() {
        return chain; // p constant: derivative is zero, chain is just [p₀].
    }
    chain.push(p1);
    let mut g = Int::ONE;
    let mut h = Int::ONE;
    loop {
        let last = chain.len() - 1;
        let dega = ip_deg(&chain[last - 1]).expect("sturm: zero in chain");
        let degb = ip_deg(&chain[last]).expect("sturm: zero in chain");
        let e = dega - degb; // δ ≥ 1 (remainder degrees strictly decrease)
        let praw = ip_prem(&chain[last - 1], &chain[last]);
        if ip_deg(&praw).is_none() {
            break; // exact division: previous element is (a multiple of) the gcd
        }
        // Positive-multiple sign: −sign(lc Rᵢ)^{δ+1}.
        let lead_sign = ip_lead(&chain[last]).signum();
        let pow_sign = if (e + 1).is_multiple_of(2) {
            1
        } else {
            lead_sign
        };
        let flip = -pow_sign < 0; // true ⇒ negate to get the positive multiple
        // Subresultant magnitude denominator g·h^δ (exact divisor of praw).
        let denom = g.mul(&h.pow(e as u32));
        let mut next: alloc::vec::Vec<Int> = praw.iter().map(|c| c.div_exact(&denom)).collect();
        if flip {
            for c in next.iter_mut() {
                *c = Int::neg(c);
            }
        }
        // ψ update (magnitudes, Cohen 3.3.1): g ← |lc(Rᵢ)| (the divisor, which
        // becomes the next dividend), h ← g^δ / h^{δ−1}.
        let new_g = ip_lead(&chain[last]).abs();
        h = if e == 0 {
            h
        } else {
            new_g.pow(e as u32).div_exact(&h.pow((e - 1) as u32))
        };
        g = new_g;
        chain.push(ip_trim(next));
    }
    chain
}

/// The subresultant GCD of two integer polynomials, returned as a primitive
/// integer polynomial (Cohen, Algorithm 3.3.1). The result is the integer GCD up
/// to sign; callers normalize as needed.
#[cfg(feature = "rational")]
fn ip_subresultant_gcd(
    a: &[crate::int::Int],
    b: &[crate::int::Int],
) -> alloc::vec::Vec<crate::int::Int> {
    use crate::int::Int;
    let mut a = ip_primitive(a);
    let mut b = ip_primitive(b);
    if ip_deg(&b).is_none() {
        return a;
    }
    if ip_deg(&a).is_none() {
        return b;
    }
    if ip_deg(&a) < ip_deg(&b) {
        core::mem::swap(&mut a, &mut b);
    }
    let mut g = Int::ONE;
    let mut h = Int::ONE;
    loop {
        let e = ip_deg(&a).unwrap() - ip_deg(&b).unwrap();
        let r = ip_prem(&a, &b);
        match ip_deg(&r) {
            None => break,                           // b divides a: b is the gcd
            Some(0) => return alloc::vec![Int::ONE], // constant remainder: coprime
            Some(_) => {}
        }
        let denom = g.mul(&h.pow(e as u32));
        let next: alloc::vec::Vec<Int> = r.iter().map(|c| c.div_exact(&denom)).collect();
        // Cohen 3.3.1: after A←B, g ← lc(A) = lc(old divisor b).
        let new_g = ip_lead(&b).clone();
        h = if e == 0 {
            h
        } else {
            new_g.pow(e as u32).div_exact(&h.pow((e - 1) as u32))
        };
        g = new_g;
        a = b;
        b = ip_trim(next);
    }
    ip_primitive(&b)
}

/// Converts an integer polynomial to a `Poly<Rational>` (integer coefficients).
#[cfg(feature = "rational")]
fn int_poly_to_rational(p: &[crate::int::Int]) -> Poly<Rational> {
    Poly::new(
        p.iter()
            .map(|c| Rational::from_integer(c.clone()))
            .collect(),
    )
}

#[cfg(feature = "rational")]
impl Poly<Rational> {
    /// Returns the monic GCD of `self` and `other` via the subresultant PRS.
    ///
    /// Mathematically equal to [`gcd`](Poly::gcd) (the monic greatest common
    /// divisor over ℚ), but computed on primitive integer polynomials with
    /// subresultant scaling, so the intermediate coefficients stay polynomially
    /// bounded instead of suffering the Euclidean rational-coefficient blow-up.
    pub fn subresultant_gcd(&self, other: &Poly<Rational>) -> Poly<Rational> {
        if self.is_zero() {
            return other.monic();
        }
        if other.is_zero() {
            return self.monic();
        }
        let a = rational_to_primitive_int(self);
        let b = rational_to_primitive_int(other);
        let g = ip_subresultant_gcd(&a, &b);
        int_poly_to_rational(&g).monic()
    }

    /// Returns the squarefree part `self / gcd(self, self′)` (monic).
    pub fn squarefree_part(&self) -> Poly<Rational> {
        if self.degree().unwrap_or(0) < 1 {
            return self.monic();
        }
        let g = self.subresultant_gcd(&self.derivative());
        self.div_rem(&g).0.monic()
    }

    /// Returns a Sturm chain of `self`: `p₀ = self`, `p₁ = self′`, and each
    /// subsequent term a positive rational multiple of `−(pᵢ₋₂ mod pᵢ₋₁)`.
    ///
    /// The chain is computed by the **subresultant PRS** on primitive integer
    /// polynomials (see the module notes), so intermediate coefficients stay
    /// polynomially bounded rather than exploding as the naive rational
    /// remainder sequence does. Each returned polynomial is a *positive* rational
    /// multiple of the corresponding classical Sturm polynomial, hence has the
    /// same sign at every point: the sign-variation count — and therefore every
    /// real-root count derived from it — is identical to the classical chain. For
    /// a correct real-root count the polynomial should be squarefree (see
    /// [`squarefree_part`](Poly::squarefree_part)).
    pub fn sturm_chain(&self) -> alloc::vec::Vec<Poly<Rational>> {
        let p0 = rational_to_primitive_int(self);
        if ip_deg(&p0).is_none() {
            // Zero polynomial: mirror the classical [zero, zero-derivative] shape.
            return alloc::vec![Poly::zero()];
        }
        let p1 = rational_to_primitive_int(&self.derivative());
        ip_sturm_chain(p0, p1)
            .iter()
            .map(|r| int_poly_to_rational(r))
            .collect()
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
