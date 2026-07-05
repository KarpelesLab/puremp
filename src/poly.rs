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

/// Smaller-operand coefficient count at or above which `Poly<Int>::mul` /
/// `Poly<Rational>::mul` route through Kronecker substitution (pack → one big
/// integer multiply → unpack). Below it, the generic Karatsuba path wins.
#[cfg(feature = "int")]
const KRONECKER_THRESHOLD: usize = 32;

/// Divisor degree (and quotient degree) at or above which `Poly::div_rem`
/// switches from schoolbook long division to Newton-iteration fast division.
/// Below it the schoolbook base case is both faster and the differential
/// reference.
const NEWTON_DIV_THRESHOLD: usize = 40;

/// Degree at or above which `Poly::gcd` switches from Euclid to the recursive
/// Half-GCD. Below it, plain Euclid is the base case and the reference.
const HGCD_THRESHOLD: usize = 48;

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
        // Coefficient-specialized fast multiply (Kronecker substitution for
        // `Int`/`Rational`), when the coefficient ring offers one and the
        // operands are large enough for it to pay off. The hook returns the exact
        // schoolbook product, so this is bit-identical to the generic path.
        if let Some(prod) = T::poly_mul(&self.coeffs, &rhs.coeffs) {
            return Poly::new(prod);
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

    /// Returns `⌊self / x^k⌋` (drops the lowest `k` coefficients).
    fn shift_down(&self, k: usize) -> Poly<T> {
        if k == 0 {
            return self.clone();
        }
        if k >= self.coeffs.len() {
            return Poly::zero();
        }
        Poly::new(self.coeffs[k..].to_vec())
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
    ///
    /// Above a size threshold (and only for an *exact* coefficient ring) this
    /// routes through Newton fast division — `O(M(d))` versus the schoolbook
    /// `O(d²)` — whose `(q, r)` is bit-identical to the schoolbook base case.
    pub fn div_rem(&self, divisor: &Poly<T>) -> (Poly<T>, Poly<T>) {
        let dd = divisor
            .degree()
            .expect("Poly::div_rem: division by zero polynomial");
        if T::EXACT
            && let Some(nd) = self.degree()
            && nd >= dd
            && dd >= NEWTON_DIV_THRESHOLD
            && (nd - dd) >= NEWTON_DIV_THRESHOLD
        {
            return self.div_rem_newton(divisor, nd, dd);
        }
        self.div_rem_schoolbook(divisor, dd)
    }

    /// Schoolbook long division (the base case and differential reference).
    fn div_rem_schoolbook(&self, divisor: &Poly<T>, dd: usize) -> (Poly<T>, Poly<T>) {
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

    /// Newton fast division: `q = rev(rev(a) · rev(b)⁻¹ mod x^{m+1})`, then
    /// `r = a − b·q`. Here `nd = deg self`, `dd = deg divisor`, `nd ≥ dd`, and
    /// `m = nd − dd` is the quotient degree. `rev(b)⁻¹` is the power-series
    /// inverse to precision `m+1` computed by Newton iteration. The `(q, r)` is
    /// the unique field quotient/remainder, hence equals the schoolbook result.
    fn div_rem_newton(&self, divisor: &Poly<T>, nd: usize, dd: usize) -> (Poly<T>, Poly<T>) {
        let m = nd - dd; // quotient degree
        let brev = divisor.reverse_n(dd); // constant term = lead(divisor) ≠ 0
        let brev_inv = brev.inv_series(m + 1);
        let arev = self.reverse_n(nd);
        let qrev = arev.mul_trunc(&brev_inv, m + 1);
        let q = qrev.reverse_n(m);
        let r = self.sub(&divisor.mul(&q));
        (q, r)
    }

    /// The reversal `xⁿ · self(1/x)` as a length-`n+1` polynomial: coefficient
    /// `j` of the result is coefficient `n − j` of `self` (`0` past the degree).
    fn reverse_n(&self, n: usize) -> Poly<T> {
        if self.is_zero() {
            return Poly::zero();
        }
        let mut out = Vec::with_capacity(n + 1);
        for j in 0..=n {
            out.push(self.coeff(n - j));
        }
        Poly::new(out)
    }

    /// Returns `self · rhs` truncated to degree `< t` (i.e. modulo `xᵗ`).
    fn mul_trunc(&self, rhs: &Poly<T>, t: usize) -> Poly<T> {
        let prod = self.mul(rhs);
        if prod.coeffs.len() <= t {
            prod
        } else {
            Poly::new(prod.coeffs[..t].to_vec())
        }
    }

    /// Power-series inverse `g` with `self · g ≡ 1 (mod xᵗ)`, computed by Newton
    /// doubling `g ← g·(2 − self·g)`. Requires `self.coeff(0)` invertible and
    /// `t ≥ 1`. The literal `2 = 1 + 1` is taken in the coefficient ring, so the
    /// iteration is correct in every characteristic (in characteristic 2 the
    /// subtraction `0 − self·g` still yields `1 + e`).
    fn inv_series(&self, t: usize) -> Poly<T> {
        let c0 = self.coeff(0);
        let inv0 = c0.one() / c0;
        let mut g = Poly::constant(inv0);
        let mut prec = 1;
        while prec < t {
            prec = (2 * prec).min(t);
            let one = self.coeff(0).one();
            let two = Poly::constant(one.clone() + one);
            let fg = self.mul_trunc(&g, prec);
            let corr = two.sub(&fg);
            g = g.mul_trunc(&corr, prec);
        }
        g
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

    /// Returns the monic GCD of `self` and `other` over the field of
    /// coefficients.
    ///
    /// Above a degree threshold (and only for an *exact* coefficient ring) this
    /// routes through the recursive Half-GCD — `O(M(d)·log d)` versus the
    /// `O(d²)` Euclid — whose monic result is bit-identical to Euclid's (the
    /// monic GCD is unique). Below the threshold, and for inexact rings, plain
    /// Euclid runs as both the base case and the differential reference.
    pub fn gcd(&self, other: &Poly<T>) -> Poly<T> {
        let big = self.degree().unwrap_or(0).max(other.degree().unwrap_or(0));
        if T::EXACT && !self.is_zero() && !other.is_zero() && big >= HGCD_THRESHOLD {
            return self.gcd_hgcd(other);
        }
        self.gcd_euclid(other)
    }

    /// Plain Euclidean GCD (repeated remainder), monic-normalized. The base case
    /// for [`gcd`](Poly::gcd) and its differential reference.
    fn gcd_euclid(&self, other: &Poly<T>) -> Poly<T> {
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a.monic()
    }

    /// Half-GCD driven GCD: repeatedly apply the recursive Half-GCD to halve the
    /// working degree, cross the half-barrier with one Euclidean step, then
    /// finish with Euclid once the degree drops below the threshold. Both
    /// operands are nonzero on entry (guaranteed by [`gcd`](Poly::gcd)).
    fn gcd_hgcd(&self, other: &Poly<T>) -> Poly<T> {
        let mut a = self.clone();
        let mut b = other.clone();
        // Half-GCD needs deg a > deg b; keep the larger degree in `a`.
        if a.degree() < b.degree() {
            core::mem::swap(&mut a, &mut b);
        }
        let sample = a.leading().expect("gcd_hgcd: a is nonzero").clone();
        while !b.is_zero() && b.degree().unwrap() >= HGCD_THRESHOLD {
            if a.degree().unwrap() == b.degree().unwrap() {
                // Half-GCD requires a strict degree gap; take one Euclid step.
                let r = a.rem(&b);
                a = b;
                b = r;
                continue;
            }
            let mat = HgcdMat::half_gcd(&a, &b, &sample);
            let (a2, b2) = mat.apply(&a, &b);
            a = a2;
            b = b2;
            if b.is_zero() {
                break;
            }
            // One Euclidean step to reduce past the half-degree barrier so the
            // next Half-GCD sees a genuinely smaller instance.
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a.monic()
    }
}

/// A 2×2 matrix of polynomials — the cofactor matrix accumulated by the
/// recursive Half-GCD. Entry layout `[[m00, m01], [m10, m11]]`; the matrix maps
/// a column `(a, b)ᵀ` of the remainder sequence to a later column, so that
/// Half-GCD progress can be composed cheaply.
struct HgcdMat<T> {
    m: [[Poly<T>; 2]; 2],
}

impl<T> HgcdMat<T>
where
    T: Ring + Div<Output = T>,
{
    /// The identity matrix, drawing its `1` from a sample coefficient's ring.
    fn identity(sample: &T) -> HgcdMat<T> {
        let one = Poly::constant(sample.one());
        let zero = Poly::zero();
        HgcdMat {
            m: [[one.clone(), zero.clone()], [zero, one]],
        }
    }

    /// The single-step quotient matrix `[[0, 1], [1, −q]]`: it sends `(s, t)ᵀ`
    /// to `(t, s − q·t)ᵀ`, one Euclidean step with quotient `q`.
    fn quotient(q: &Poly<T>, sample: &T) -> HgcdMat<T> {
        let one = Poly::constant(sample.one());
        HgcdMat {
            m: [[Poly::zero(), one], [Poly::constant(sample.one()), q.neg()]],
        }
    }

    /// Matrix product `self · rhs`.
    fn mul(&self, rhs: &HgcdMat<T>) -> HgcdMat<T> {
        let e = |i: usize, j: usize| -> Poly<T> {
            self.m[i][0]
                .mul(&rhs.m[0][j])
                .add(&self.m[i][1].mul(&rhs.m[1][j]))
        };
        HgcdMat {
            m: [[e(0, 0), e(0, 1)], [e(1, 0), e(1, 1)]],
        }
    }

    /// Applies the matrix to the column `(a, b)ᵀ`, returning the new pair.
    fn apply(&self, a: &Poly<T>, b: &Poly<T>) -> (Poly<T>, Poly<T>) {
        (
            self.m[0][0].mul(a).add(&self.m[0][1].mul(b)),
            self.m[1][0].mul(a).add(&self.m[1][1].mul(b)),
        )
    }

    /// The recursive **Half-GCD** (von zur Gathen & Gerhard, *Modern Computer
    /// Algebra*, Ch. 11). Given `deg a > deg b ≥ 0` with `n = deg a` and
    /// `m = ⌈n/2⌉`, returns a matrix `R` — a product of consecutive Euclidean
    /// quotient matrices — such that `R·(a, b)ᵀ = (r_k, r_{k+1})ᵀ` is the first
    /// remainder-sequence column with `deg r_k ≥ m > deg r_{k+1}`. Because `R` is
    /// built only from genuine quotient matrices, applying it to the full pair
    /// and continuing the Euclidean scheme yields exactly the true GCD.
    fn half_gcd(a: &Poly<T>, b: &Poly<T>, sample: &T) -> HgcdMat<T> {
        let n = a.degree().expect("half_gcd: a is nonzero");
        let m = n.div_ceil(2);
        // Already reduced below the half-degree: nothing to do.
        if b.is_zero() || b.degree().unwrap() < m {
            return HgcdMat::identity(sample);
        }
        // First recursion on the high halves (quotients of the top parts drive
        // the same leading quotient sequence as the full polynomials).
        let a0 = a.shift_down(m);
        let b0 = b.shift_down(m);
        let r = HgcdMat::half_gcd(&a0, &b0, sample);
        let (s, t) = r.apply(a, b);
        if t.is_zero() || t.degree().unwrap() < m {
            return r;
        }
        // One Euclidean step on the reduced pair.
        let (q, u) = s.div_rem(&t);
        let r = HgcdMat::quotient(&q, sample).mul(&r);
        if u.is_zero() || u.degree().unwrap() < m {
            return r;
        }
        // Second recursion pushes the degree the rest of the way down to `m`.
        let dt = t.degree().unwrap();
        let k = 2 * m - dt; // 0 < k ≤ m since m < dt < 2m
        let t0 = t.shift_down(k);
        let u0 = u.shift_down(k);
        HgcdMat::half_gcd(&t0, &u0, sample).mul(&r)
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

impl<T: Ring> core::ops::Neg for Poly<T> {
    type Output = Poly<T>;
    #[inline]
    fn neg(self) -> Poly<T> {
        Poly::neg(&self)
    }
}
impl<T: Ring> core::ops::Neg for &Poly<T> {
    type Output = Poly<T>;
    #[inline]
    fn neg(self) -> Poly<T> {
        Poly::neg(self)
    }
}

// ===========================================================================
// Kronecker-substitution multiplication for integer / rational polynomials.
//
// To multiply `a·b` in ℤ[x], evaluate both at `X = 2ᵏ` for a `k` wide enough
// that every product coefficient (a signed sum of ≤ min(nₐ, n_b) coefficient
// products) fits in its `k`-bit slot without overlapping its neighbours, form
// the two big integers `a(2ᵏ)`, `b(2ᵏ)`, multiply them with the fast `Int`
// multiply (Toom/NTT), and read the product coefficients back out of the base-`2ᵏ`
// digits. This turns polynomial multiplication into a single `O(M(d·k))` integer
// multiply (Kronecker substitution; folklore).
//
// Signed coefficients are handled by a fixed **bias**: add `B = 2^{k−1}` to every
// slot before unpacking, making each stored digit `c_l + B ∈ (0, 2ᵏ)` — strictly
// positive and non-overlapping — so the product can be split with pure unsigned
// base-`2ᵏ` digit extraction and each coefficient recovered as `digit − B`.
// ===========================================================================

#[cfg(feature = "int")]
use crate::int::Int;

/// `sum_i coeffs[i]·2^{k·i}` as a single (signed) big integer, built by a
/// balanced divide-and-conquer of shifts and adds so packing stays near-linear.
#[cfg(feature = "int")]
fn kronecker_pack(coeffs: &[Int], k: u32) -> Int {
    match coeffs.len() {
        0 => Int::ZERO,
        1 => coeffs[0].clone(),
        n => {
            let mid = n / 2;
            let lo = kronecker_pack(&coeffs[..mid], k);
            let hi = kronecker_pack(&coeffs[mid..], k);
            let shift = (k as u64)
                .checked_mul(mid as u64)
                .and_then(|s| u32::try_from(s).ok())
                .expect("kronecker_pack: shift exceeds 2³² bits");
            lo.add(&hi.mul_2k(shift))
        }
    }
}

/// Inverse of [`kronecker_pack`] for a non-negative integer: extracts `num`
/// base-`2ᵏ` digits (each in `[0, 2ᵏ)`), low-to-high, by divide-and-conquer.
#[cfg(feature = "int")]
fn kronecker_unpack(n: &Int, k: u32, num: usize) -> Vec<Int> {
    if num == 1 {
        return alloc::vec![n.clone()];
    }
    let mid = num / 2;
    let bits = (k as u64)
        .checked_mul(mid as u64)
        .and_then(|s| u32::try_from(s).ok())
        .expect("kronecker_unpack: shift exceeds 2³² bits");
    let lo = n.mod_2k(bits);
    let hi = n.div_2k_trunc(bits);
    let mut out = kronecker_unpack(&lo, k, mid);
    out.extend(kronecker_unpack(&hi, k, num - mid));
    out
}

/// Core Kronecker product of two nonempty integer coefficient slices (low-to-high,
/// trimmed). Returns the product coefficients (length `nₐ + n_b − 1`), exactly the
/// schoolbook convolution.
#[cfg(feature = "int")]
fn kronecker_convolve(a: &[Int], b: &[Int]) -> Vec<Int> {
    // Slot width `k`: room for the largest coefficient product magnitude plus a
    // count factor plus a sign/bias bit. With `|a_i| < 2^{ba}`, `|b_j| < 2^{bb}`
    // and ≤ `min_len` products per output coefficient, `|c_l| < 2^{ba+bb+clog}`,
    // so `k = ba + bb + clog + 2` guarantees `|c_l| < 2^{k-1} = B`.
    let ba = a.iter().map(|c| c.bit_len()).max().unwrap_or(0);
    let bb = b.iter().map(|c| c.bit_len()).max().unwrap_or(0);
    let min_len = a.len().min(b.len());
    let clog = if min_len <= 1 {
        0
    } else {
        (min_len - 1).ilog2() + 1
    };
    let k = ba + bb + clog + 2;
    let num = a.len() + b.len() - 1;

    let prod = kronecker_pack(a, k).mul(&kronecker_pack(b, k));
    // Bias by `B` per slot: `D = B · (1 + 2ᵏ + … + 2^{k(num−1)})`, so `prod + D`
    // has every digit in `(0, 2ᵏ)` and is non-negative.
    let repunit = kronecker_pack(&alloc::vec![Int::ONE; num], k);
    let bias = repunit.mul_2k(k - 1); // ·B
    let shifted = prod.add(&bias);
    let b_off = Int::ONE.mul_2k(k - 1); // = B
    kronecker_unpack(&shifted, k, num)
        .into_iter()
        .map(|d| d.sub(&b_off))
        .collect()
}

/// Kronecker multiply hook for `Poly<Int>` (see [`Ring::poly_mul`]). Returns
/// `None` below the degree threshold so small products stay on the Karatsuba
/// path.
#[cfg(feature = "int")]
pub(crate) fn kronecker_mul_int(a: &[Int], b: &[Int]) -> Option<Vec<Int>> {
    if a.is_empty() || b.is_empty() {
        return Some(Vec::new());
    }
    if a.len().min(b.len()) < KRONECKER_THRESHOLD {
        return None;
    }
    Some(kronecker_convolve(a, b))
}

#[cfg(feature = "int")]
impl Poly<Int> {
    /// Multiplies `self · rhs` by **Kronecker substitution** (pack both operands
    /// into big integers, one fast `Int` multiply, unpack). Bit-identical to
    /// [`Poly::mul`](Poly::mul) for every input, which itself dispatches here
    /// once both operands reach the internal Kronecker degree threshold.
    pub fn mul_kronecker(&self, rhs: &Poly<Int>) -> Poly<Int> {
        if self.is_zero() || rhs.is_zero() {
            return Poly::zero();
        }
        Poly::new(kronecker_convolve(self.coeffs(), rhs.coeffs()))
    }
}

/// Kronecker multiply hook for `Poly<Rational>`: clear denominators to integer
/// polynomials, convolve, then divide the common denominator back in. Returns
/// `None` below the degree threshold.
#[cfg(feature = "rational")]
pub(crate) fn kronecker_mul_rational(a: &[Rational], b: &[Rational]) -> Option<Vec<Rational>> {
    if a.is_empty() || b.is_empty() {
        return Some(Vec::new());
    }
    if a.len().min(b.len()) < KRONECKER_THRESHOLD {
        return None;
    }
    Some(kronecker_convolve_rational(a, b))
}

/// Core of the rational Kronecker product (see [`kronecker_mul_rational`]).
#[cfg(feature = "rational")]
fn kronecker_convolve_rational(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let la = a.iter().fold(Int::ONE, |l, c| l.lcm(c.denominator()));
    let lb = b.iter().fold(Int::ONE, |l, c| l.lcm(c.denominator()));
    let ai: Vec<Int> = a
        .iter()
        .map(|c| c.numerator().mul(&la.div_exact(c.denominator())))
        .collect();
    let bi: Vec<Int> = b
        .iter()
        .map(|c| c.numerator().mul(&lb.div_exact(c.denominator())))
        .collect();
    let denom = la.mul(&lb);
    kronecker_convolve(&ai, &bi)
        .into_iter()
        .map(|c| Rational::new(c, denom.clone()))
        .collect()
}

#[cfg(feature = "rational")]
impl Poly<Rational> {
    /// Multiplies `self · rhs` by **Kronecker substitution** over ℚ: clear
    /// denominators, multiply the integer polynomials by packing into big
    /// integers, then divide the common denominator back in. Bit-identical to
    /// [`Poly::mul`](Poly::mul) for every input.
    pub fn mul_kronecker(&self, rhs: &Poly<Rational>) -> Poly<Rational> {
        if self.is_zero() || rhs.is_zero() {
            return Poly::zero();
        }
        Poly::new(kronecker_convolve_rational(self.coeffs(), rhs.coeffs()))
    }
}

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

// ===========================================================================
// In-crate differential tests: the fast paths (Newton division, Half-GCD,
// Kronecker multiplication) must be bit-identical to the schoolbook/Euclid
// references they replace. These call the private base-case functions directly,
// so they compare the *implementations*, not merely the mathematical contract.
// ===========================================================================
#[cfg(all(test, feature = "rational", feature = "int"))]
mod fast_tests {
    use super::*;
    use crate::int::Int;
    use crate::mod_int::ModInt;
    use crate::rational::Rational;
    use alloc::vec;

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Lcg {
            Lcg(seed ^ 0x9e3779b97f4a7c15)
        }
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn range(&mut self, n: i64) -> i64 {
            (self.next() >> 33) as i64 % n
        }
    }

    fn rand_rat(rng: &mut Lcg, spread: i64) -> Rational {
        let num = rng.range(2 * spread + 1) - spread;
        let den = rng.range(spread) + 1;
        Rational::new(Int::from(num), Int::from(den))
    }

    fn rand_poly_rat(rng: &mut Lcg, deg: usize, spread: i64) -> Poly<Rational> {
        let mut v: Vec<Rational> = (0..deg).map(|_| rand_rat(rng, spread)).collect();
        // Force a nonzero leading coefficient of the requested degree.
        loop {
            let lead = rand_rat(rng, spread);
            if !lead.is_zero() {
                v.push(lead);
                break;
            }
        }
        Poly::new(v)
    }

    fn rand_poly_mod(rng: &mut Lcg, deg: usize, p: &Int) -> Poly<ModInt> {
        let mut v: Vec<ModInt> = (0..deg)
            .map(|_| ModInt::new(Int::from(rng.range(1 << 20)), p.clone()))
            .collect();
        loop {
            let c = ModInt::new(Int::from(rng.range(1 << 20)), p.clone());
            if !c.is_zero() {
                v.push(c);
                break;
            }
        }
        Poly::new(v)
    }

    fn rand_poly_int(rng: &mut Lcg, deg: usize, bits: u32) -> Poly<Int> {
        let mk = |rng: &mut Lcg| {
            let mut m = Int::from(rng.range(i64::MAX));
            for _ in 0..(bits / 60) {
                m = m.mul(&Int::from(rng.range(i64::MAX)));
            }
            if rng.next() & 1 == 0 { m.neg() } else { m }
        };
        let mut v: Vec<Int> = (0..deg).map(|_| mk(rng)).collect();
        loop {
            let lead = mk(rng);
            if !lead.is_zero() {
                v.push(lead);
                break;
            }
        }
        Poly::new(v)
    }

    #[test]
    fn newton_div_matches_schoolbook_rational() {
        let mut rng = Lcg::new(1);
        for _ in 0..60 {
            let na = 20 + rng.range(50) as usize;
            let nb = 1 + rng.range(na as i64) as usize;
            let a = rand_poly_rat(&mut rng, na, 12);
            let b = rand_poly_rat(&mut rng, nb, 12);
            let nd = a.degree().unwrap();
            let dd = b.degree().unwrap();
            let school = a.div_rem_schoolbook(&b, dd);
            let newton = a.div_rem_newton(&b, nd, dd);
            assert_eq!(school, newton, "na={na} nb={nb}");
            // and the defining identity
            assert_eq!(a, b.mul(&newton.0).add(&newton.1));
        }
    }

    #[test]
    fn newton_div_matches_schoolbook_gfp() {
        let p = Int::from(2_000_003);
        let mut rng = Lcg::new(2);
        for _ in 0..150 {
            let na = 20 + rng.range(150) as usize;
            let nb = 1 + rng.range(na as i64) as usize;
            let a = rand_poly_mod(&mut rng, na, &p);
            let b = rand_poly_mod(&mut rng, nb, &p);
            let nd = a.degree().unwrap();
            let dd = b.degree().unwrap();
            assert_eq!(
                a.div_rem_schoolbook(&b, dd),
                a.div_rem_newton(&b, nd, dd),
                "na={na} nb={nb}"
            );
        }
    }

    // Characteristic 2: Newton inverse must still be correct (2 ≡ 0).
    #[test]
    fn newton_div_char2() {
        let p = Int::from(2);
        let mut rng = Lcg::new(7);
        for _ in 0..80 {
            let na = 20 + rng.range(120) as usize;
            let nb = 1 + rng.range(na as i64) as usize;
            let a = rand_poly_mod(&mut rng, na, &p);
            let b = rand_poly_mod(&mut rng, nb, &p);
            let nd = a.degree().unwrap();
            let dd = b.degree().unwrap();
            assert_eq!(a.div_rem_schoolbook(&b, dd), a.div_rem_newton(&b, nd, dd));
        }
    }

    #[test]
    fn hgcd_matches_euclid_gfp() {
        let p = Int::from(2_000_003);
        let mut rng = Lcg::new(3);
        for _ in 0..100 {
            let da = 50 + rng.range(150) as usize;
            let db = 1 + rng.range(da as i64) as usize;
            let a = rand_poly_mod(&mut rng, da, &p);
            let b = rand_poly_mod(&mut rng, db, &p);
            assert_eq!(a.gcd_euclid(&b), a.gcd_hgcd(&b), "da={da} db={db}");
            // structured: shared factor
            let dg = 10 + rng.range(30) as usize;
            let g = rand_poly_mod(&mut rng, dg, &p);
            let af = a.mul(&g);
            let bf = b.mul(&g);
            assert_eq!(af.gcd_euclid(&bf), af.gcd_hgcd(&bf));
        }
    }

    #[test]
    fn hgcd_matches_euclid_rational() {
        let mut rng = Lcg::new(4);
        for _ in 0..15 {
            let da = 50 + rng.range(25) as usize;
            let db = 1 + rng.range(da as i64) as usize;
            let a = rand_poly_rat(&mut rng, da, 6);
            let b = rand_poly_rat(&mut rng, db, 6);
            assert_eq!(a.gcd_euclid(&b), a.gcd_hgcd(&b), "da={da} db={db}");
        }
    }

    #[test]
    fn hgcd_structured_divides_and_coprime() {
        let p = Int::from(1_000_003);
        let mut rng = Lcg::new(5);
        for _ in 0..40 {
            // one divides the other
            let dq = 60 + rng.range(60) as usize;
            let q = rand_poly_mod(&mut rng, dq, &p);
            let dd = 20 + rng.range(40) as usize;
            let d = rand_poly_mod(&mut rng, dd, &p);
            let prod = q.mul(&d);
            assert_eq!(prod.gcd_euclid(&d), prod.gcd_hgcd(&d));
            // gcd(f, f) = monic f
            assert_eq!(prod.gcd_euclid(&prod), prod.gcd_hgcd(&prod));
        }
    }

    #[test]
    fn kronecker_matches_schoolbook_int() {
        let mut rng = Lcg::new(6);
        for _ in 0..80 {
            let na = 1 + rng.range(90) as usize;
            let nb = 1 + rng.range(90) as usize;
            let bits = 1 + rng.range(200) as u32;
            let a = rand_poly_int(&mut rng, na, bits);
            let b = rand_poly_int(&mut rng, nb, bits);
            assert_eq!(
                a.mul_schoolbook(&b),
                a.mul_kronecker(&b),
                "na={na} nb={nb} bits={bits}"
            );
        }
    }

    #[test]
    fn kronecker_matches_schoolbook_rational() {
        let mut rng = Lcg::new(8);
        for _ in 0..40 {
            let na = 1 + rng.range(70) as usize;
            let nb = 1 + rng.range(70) as usize;
            let a = rand_poly_rat(&mut rng, na, 40);
            let b = rand_poly_rat(&mut rng, nb, 40);
            assert_eq!(a.mul_schoolbook(&b), a.mul_kronecker(&b), "na={na} nb={nb}");
        }
    }

    #[test]
    fn kronecker_edge_cases() {
        // constants, single terms, zeros, negatives
        let a: Poly<Int> = Poly::new(vec![Int::from(-5)]);
        let b: Poly<Int> = Poly::new(vec![Int::from(3), Int::from(-7), Int::from(11)]);
        assert_eq!(a.mul_schoolbook(&b), a.mul_kronecker(&b));
        assert!(Poly::<Int>::zero().mul_kronecker(&b).is_zero());
        assert!(b.mul_kronecker(&Poly::<Int>::zero()).is_zero());
    }
}
