//! General real algebraic numbers.
//!
//! An [`Algebraic`] is a real root of an integer/rational polynomial, stored as a
//! squarefree defining polynomial together with a rational *isolating interval*
//! that contains exactly one real root of it — the value. This is exact: two
//! `Algebraic`s compare by their true real value (never by a float
//! approximation), and the field operations `+ − × ÷` produce a new exact
//! algebraic number.
//!
//! The representation and algorithms are textbook real-algebraic-number
//! machinery:
//!
//! - **Sturm sequences** count and isolate real roots in an interval.
//! - **Bisection** refines an isolating interval to arbitrary precision.
//! - Sum and product use the fact that the eigenvalues of the *Kronecker sum*
//!   `A ⊗ I + I ⊗ B` (resp. *Kronecker product* `A ⊗ B`) of the companion
//!   matrices are the pairwise sums (resp. products) of the roots. The
//!   characteristic polynomial is obtained by the **Faddeev–LeVerrier**
//!   algorithm (only rational matrix arithmetic — no polynomial-matrix
//!   determinants), and the correct root is then isolated.
//!
//! For the special degree-≤2 case, [`Quadratic`](crate::quadratic::Quadratic) is
//! far cheaper.
//!
//! # Performance
//!
//! Operations are exact but not cheap: a binary operation on degree-`m` and
//! degree-`n` values builds an `mn × mn` matrix and a degree-`mn` resultant, and
//! the Sturm sequence over ℚ suffers the classical polynomial-remainder-sequence
//! coefficient growth. This is fine for modest degrees (sums/products of a few
//! square roots) but grows quickly beyond that; a subresultant PRS would tame the
//! coefficient blow-up and is a natural future optimization.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;

use crate::int::Int;
use crate::matrix::Matrix;
use crate::poly::Poly;
use crate::rational::Rational;

type Q = Rational;
type P = Poly<Rational>;

fn q_i64(v: i64) -> Q {
    Rational::from_integer(Int::from_i64(v))
}

// ===========================================================================
// Polynomial helpers (over ℚ)
// ===========================================================================

// Real-root isolation primitives are shared with the public `Poly<Rational>`
// API in `crate::poly`.
use crate::poly::sturm_count as count_roots;

/// Returns the squarefree part `p / gcd(p, p′)` (monic).
fn squarefree(p: &P) -> P {
    p.squarefree_part()
}

/// Sign of `p(x)` as `-1`, `0`, or `1`.
fn eval_sign(p: &P, x: &Q) -> i32 {
    p.eval(x).signum()
}

/// The Sturm chain of a squarefree polynomial.
fn sturm_chain(p: &P) -> Vec<P> {
    p.sturm_chain()
}

// ===========================================================================
// Companion matrices, Kronecker operations, Faddeev–LeVerrier
// ===========================================================================

/// Companion matrix of a polynomial (made monic first).
fn companion(p: &P) -> Matrix<Q> {
    let p = p.monic();
    let m = p.degree().expect("companion: non-constant polynomial");
    let mut mat = Matrix::zeros(m, m);
    for i in 0..m {
        mat.set(i, m - 1, p.coeff(i).neg()); // last column = −cᵢ
    }
    for i in 1..m {
        mat.set(i, i - 1, Rational::ONE); // sub-diagonal ones
    }
    mat
}

/// Kronecker product `A ⊗ B`.
fn kron(a: &Matrix<Q>, b: &Matrix<Q>) -> Matrix<Q> {
    let (ar, ac) = (a.rows(), a.cols());
    let (br, bc) = (b.rows(), b.cols());
    let mut out = Matrix::zeros(ar * br, ac * bc);
    for i in 0..ar {
        for j in 0..ac {
            let aij = a.get(i, j).clone();
            for k in 0..br {
                for l in 0..bc {
                    out.set(i * br + k, j * bc + l, aij.mul(b.get(k, l)));
                }
            }
        }
    }
    out
}

/// Trace of a square matrix.
fn trace(m: &Matrix<Q>) -> Q {
    let mut t = Rational::ZERO;
    for i in 0..m.rows() {
        t = t.add(m.get(i, i));
    }
    t
}

/// Characteristic polynomial of `m` (monic) via Faddeev–LeVerrier.
fn charpoly(m: &Matrix<Q>) -> P {
    let n = m.rows();
    let mut coeffs = alloc::vec![Rational::ZERO; n + 1];
    coeffs[n] = Rational::ONE;
    let mut mk = Matrix::<Q>::identity(n); // M₁ = I
    for k in 1..=n {
        let amk = m.mul(&mk);
        let ck = trace(&amk).div(&q_i64(k as i64)).neg(); // c_{n-k} = −tr(A Mₖ)/k
        coeffs[n - k] = ck.clone();
        if k < n {
            let ident = Matrix::<Q>::identity(n).scalar_mul(&ck);
            mk = amk.add(&ident); // M_{k+1} = A Mₖ + c_{n-k} I
        }
    }
    Poly::new(coeffs)
}

// ===========================================================================
// The Algebraic type
// ===========================================================================

/// A real algebraic number: a real root of `poly`, isolated in `(lo, hi]`.
#[derive(Clone)]
pub struct Algebraic {
    poly: P, // squarefree, degree ≥ 1
    lo: Q,
    hi: Q,
}

impl Algebraic {
    /// Builds the algebraic number equal to a rational.
    pub fn from_rational(r: Rational) -> Algebraic {
        // Root of x − r.
        let poly = Poly::new(alloc::vec![r.neg(), Rational::ONE]);
        Algebraic {
            poly,
            lo: r.clone(),
            hi: r,
        }
    }

    /// Builds from an integer.
    #[inline]
    pub fn from_int(n: Int) -> Algebraic {
        Algebraic::from_rational(Rational::from_integer(n))
    }

    /// Returns every distinct real root of `poly` as an exact algebraic number,
    /// in increasing order.
    pub fn real_roots_of(poly: &Poly<Rational>) -> Vec<Algebraic> {
        let sf = squarefree(poly);
        poly.isolate_real_roots()
            .into_iter()
            .map(|(lo, hi)| Algebraic::new(sf.clone(), lo, hi))
            .collect()
    }

    /// Builds the unique real root of `poly` lying in `(lo, hi]`. The caller must
    /// guarantee there is exactly one. `poly` is reduced to its squarefree part.
    pub fn new(poly: P, lo: Q, hi: Q) -> Algebraic {
        let poly = squarefree(&poly);
        Algebraic { poly, lo, hi }.normalized()
    }

    /// Collapses to an exact rational representation when the root is rational.
    ///
    /// The interval is half-open `(lo, hi]`, so only `hi` can *be* the value; a
    /// root of the polynomial sitting at the excluded `lo` is a different root.
    fn normalized(self) -> Algebraic {
        if self.lo == self.hi {
            return self;
        }
        if eval_sign(&self.poly, &self.hi) == 0 {
            let r = self.hi.clone();
            return Algebraic::from_rational(r);
        }
        self
    }

    /// Returns the defining (squarefree) polynomial.
    #[inline]
    pub fn defining_polynomial(&self) -> &Poly<Rational> {
        &self.poly
    }

    /// Returns the current isolating interval `(lo, hi)`.
    #[inline]
    pub fn interval(&self) -> (&Rational, &Rational) {
        (&self.lo, &self.hi)
    }

    /// Returns `true` if the value is rational (its interval has collapsed).
    #[inline]
    pub fn is_rational(&self) -> bool {
        self.lo == self.hi
    }

    /// Halves the isolating interval, keeping the half with the root.
    pub fn refine(&mut self) {
        if self.lo == self.hi {
            return;
        }
        let mid = self.lo.add(&self.hi).div(&q_i64(2));
        let sm = eval_sign(&self.poly, &mid);
        if sm == 0 {
            // Inside a proper isolating interval only the (rational) value itself
            // can be a root, so a rational midpoint hit is exactly the value.
            self.lo = mid.clone();
            self.hi = mid;
            return;
        }
        // Anchor on `hi`: hi is never a spurious root (unlike lo, which may be an
        // excluded root of the shared polynomial).
        if sm == eval_sign(&self.poly, &self.hi) {
            self.hi = mid;
        } else {
            self.lo = mid;
        }
    }

    /// Refines until the interval width is below `width`.
    pub fn refine_below(&mut self, width: &Rational) {
        while self.lo != self.hi && self.hi.sub(&self.lo) > *width {
            self.refine();
        }
    }

    /// Returns `-1`, `0`, or `1` according to the sign of the value.
    pub fn signum(&self) -> i32 {
        // Zero only occurs as the exact rational 0 (an irrational value is never
        // zero); otherwise refine until the interval lies on one side of 0.
        if self.is_rational() {
            return self.lo.signum();
        }
        let mut a = self.clone();
        loop {
            if a.lo.is_positive() {
                return 1;
            }
            if a.hi.is_negative() || a.hi.is_zero() {
                return -1;
            }
            a.refine();
        }
    }

    /// Rounds the value to a [`Float`](crate::float::Float).
    pub fn to_float(
        &self,
        precision: u64,
        mode: crate::float::RoundingMode,
    ) -> crate::float::Float {
        use crate::float::Float;
        let mut a = self.clone();
        for _ in 0..(precision + 64) {
            let flo = Float::from_rational(&a.lo, precision, mode);
            let fhi = Float::from_rational(&a.hi, precision, mode);
            if flo == fhi {
                return flo;
            }
            a.refine();
        }
        Float::from_rational(&a.lo.add(&a.hi).div(&q_i64(2)), precision, mode)
    }

    /// Returns `-self`.
    pub fn neg(&self) -> Algebraic {
        // Root of p(−x); interval negates and flips.
        let neg_poly = compose_neg(&self.poly);
        Algebraic {
            poly: squarefree(&neg_poly),
            lo: self.hi.neg(),
            hi: self.lo.neg(),
        }
        .normalized()
    }

    /// Returns `self + rhs`.
    pub fn add(&self, rhs: &Algebraic) -> Algebraic {
        if self.is_rational() {
            return rhs.shift(&self.lo);
        }
        if rhs.is_rational() {
            return self.shift(&rhs.lo);
        }
        let r = charpoly(&kron_sum(&companion(&self.poly), &companion(&rhs.poly)));
        self.combine(rhs, &r, |a, b| (a.lo.add(&b.lo), a.hi.add(&b.hi)))
    }

    /// Returns `self - rhs`.
    pub fn sub(&self, rhs: &Algebraic) -> Algebraic {
        self.add(&rhs.neg())
    }

    /// Returns `self · rhs`.
    pub fn mul(&self, rhs: &Algebraic) -> Algebraic {
        if self.is_rational() {
            return rhs.scale(&self.lo);
        }
        if rhs.is_rational() {
            return self.scale(&rhs.lo);
        }
        let r = charpoly(&kron(&companion(&self.poly), &companion(&rhs.poly)));
        self.combine(rhs, &r, |a, b| interval_mul(&a.lo, &a.hi, &b.lo, &b.hi))
    }

    /// Returns `1/self`. Panics if `self` is zero.
    pub fn recip(&self) -> Algebraic {
        assert!(self.signum() != 0, "Algebraic::recip: reciprocal of zero");
        if self.is_rational() {
            return Algebraic::from_rational(self.lo.recip());
        }
        // Root of the reversed polynomial xⁿ p(1/x); interval is [1/hi, 1/lo]
        // (for a fixed-sign interval, which signum() has guaranteed).
        let mut a = self.clone();
        while a.lo.signum() != a.hi.signum() || a.lo.is_zero() || a.hi.is_zero() {
            a.refine();
        }
        let rev = reverse_poly(&a.poly);
        let (nlo, nhi) = (a.hi.recip(), a.lo.recip());
        Algebraic::new(rev, nlo, nhi)
    }

    /// Returns `self / rhs`. Panics if `rhs` is zero.
    pub fn div(&self, rhs: &Algebraic) -> Algebraic {
        self.mul(&rhs.recip())
    }

    /// Adds a rational (exact interval shift).
    fn shift(&self, c: &Q) -> Algebraic {
        if self.is_rational() {
            return Algebraic::from_rational(self.lo.add(c));
        }
        Algebraic {
            poly: squarefree(&compose_shift(&self.poly, c)),
            lo: self.lo.add(c),
            hi: self.hi.add(c),
        }
        .normalized()
    }

    /// Multiplies by a rational (exact interval scale).
    fn scale(&self, c: &Q) -> Algebraic {
        if c.is_zero() {
            return Algebraic::from_rational(Rational::ZERO);
        }
        if self.is_rational() {
            return Algebraic::from_rational(self.lo.mul(c));
        }
        let (lo, hi) = if c.is_positive() {
            (self.lo.mul(c), self.hi.mul(c))
        } else {
            (self.hi.mul(c), self.lo.mul(c))
        };
        Algebraic {
            poly: squarefree(&compose_scale(&self.poly, c)),
            lo,
            hi,
        }
        .normalized()
    }

    /// Isolates the unique root of `poly` in the interval given by `interval_fn`
    /// applied to (refined) operand intervals.
    fn combine(
        &self,
        rhs: &Algebraic,
        poly: &P,
        interval_fn: impl Fn(&Algebraic, &Algebraic) -> (Q, Q),
    ) -> Algebraic {
        let sf = squarefree(poly);
        let chain = sturm_chain(&sf);
        let mut a = self.clone();
        let mut b = rhs.clone();
        for _ in 0..4096 {
            let (lo, hi) = interval_fn(&a, &b);
            if lo < hi && count_roots(&chain, &lo, &hi) == 1 {
                return Algebraic::new(sf, lo, hi);
            }
            a.refine();
            b.refine();
        }
        panic!("Algebraic::combine: failed to isolate the result root");
    }
}

/// Interval product `[a,b]·[c,d]` = `[min, max]` of the four endpoint products.
fn interval_mul(a: &Q, b: &Q, c: &Q, d: &Q) -> (Q, Q) {
    let ps = [a.mul(c), a.mul(d), b.mul(c), b.mul(d)];
    let mut lo = ps[0].clone();
    let mut hi = ps[0].clone();
    for p in &ps[1..] {
        if *p < lo {
            lo = p.clone();
        }
        if *p > hi {
            hi = p.clone();
        }
    }
    (lo, hi)
}

/// `p(−x)`.
fn compose_neg(p: &P) -> P {
    let mut c = p.coeffs().to_vec();
    for (i, coeff) in c.iter_mut().enumerate() {
        if i % 2 == 1 {
            *coeff = coeff.neg();
        }
    }
    Poly::new(c)
}

/// `p(x − c)` via Horner in the shifted variable.
fn compose_shift(p: &P, c: &Q) -> P {
    // Evaluate p at (x - c): fold from the top, acc = acc*(x-c) + coeff.
    let xmc = Poly::new(alloc::vec![c.neg(), Rational::ONE]); // x - c
    let mut acc = Poly::zero();
    for coeff in p.coeffs().iter().rev() {
        acc = acc.mul(&xmc).add(&Poly::constant(coeff.clone()));
    }
    acc
}

/// `p(x / c)` cleared of denominators (root scales by `c`).
fn compose_scale(p: &P, c: &Q) -> P {
    // Root α of p ↦ cα is a root of p(x/c). Substitute and keep it polynomial:
    // coefficient i becomes cᵢ / cⁱ; multiply through by c^deg to stay integral
    // in spirit (rationals are fine either way).
    let n = p.degree().unwrap_or(0);
    let mut out = Vec::with_capacity(n + 1);
    let mut cpow = Rational::ONE; // c^0
    let cn = pow_q(c, n);
    for i in 0..=n {
        // term i: p_i * c^(n-i)
        let factor = cn.div(&cpow); // c^(n-i)
        out.push(p.coeff(i).mul(&factor));
        cpow = cpow.mul(c);
    }
    Poly::new(out)
}

/// Reverses coefficients: `xⁿ·p(1/x)` (root α ↦ 1/α).
fn reverse_poly(p: &P) -> P {
    let mut c = p.coeffs().to_vec();
    c.reverse();
    Poly::new(c)
}

fn pow_q(c: &Q, n: usize) -> Q {
    let mut acc = Rational::ONE;
    for _ in 0..n {
        acc = acc.mul(c);
    }
    acc
}

/// Kronecker sum `A ⊗ Iₙ + Iₘ ⊗ B`.
fn kron_sum(a: &Matrix<Q>, b: &Matrix<Q>) -> Matrix<Q> {
    let m = a.rows();
    let n = b.rows();
    let left = kron(a, &Matrix::<Q>::identity(n));
    let right = kron(&Matrix::<Q>::identity(m), b);
    left.add(&right)
}

impl PartialEq for Algebraic {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Algebraic {}

impl PartialOrd for Algebraic {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Algebraic {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut a = self.clone();
        let mut b = other.clone();
        // Shared squarefree gcd chain, for detecting equality of the two roots.
        let g = squarefree(&a.poly.gcd(&b.poly));
        let has_common = g.degree().unwrap_or(0) >= 1;
        let g_chain = has_common.then(|| sturm_chain(&g));
        for _ in 0..4096 {
            if a.hi < b.lo {
                return Ordering::Less;
            }
            if b.hi < a.lo {
                return Ordering::Greater;
            }
            if let Some(chain) = &g_chain {
                let ov_lo = if a.lo > b.lo {
                    a.lo.clone()
                } else {
                    b.lo.clone()
                };
                let ov_hi = if a.hi < b.hi {
                    a.hi.clone()
                } else {
                    b.hi.clone()
                };
                if ov_lo < ov_hi && count_roots(chain, &ov_lo, &ov_hi) >= 1 {
                    return Ordering::Equal;
                }
            }
            a.refine();
            b.refine();
        }
        Ordering::Equal
    }
}

impl From<Rational> for Algebraic {
    #[inline]
    fn from(r: Rational) -> Algebraic {
        Algebraic::from_rational(r)
    }
}

impl From<Int> for Algebraic {
    #[inline]
    fn from(n: Int) -> Algebraic {
        Algebraic::from_int(n)
    }
}

impl fmt::Display for Algebraic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_rational() {
            return fmt::Display::fmt(&self.lo, f);
        }
        // "root of <poly> in (lo, hi)"
        write!(f, "root of {} in ({}, {})", self.poly, self.lo, self.hi)
    }
}

impl fmt::Debug for Algebraic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Algebraic({self})")
    }
}

macro_rules! alg_binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl core::ops::$tr for Algebraic {
            type Output = Algebraic;
            #[inline]
            fn $m(self, rhs: Algebraic) -> Algebraic {
                Algebraic::$m(&self, &rhs)
            }
        }
        impl core::ops::$tr<&Algebraic> for &Algebraic {
            type Output = Algebraic;
            #[inline]
            fn $m(self, rhs: &Algebraic) -> Algebraic {
                Algebraic::$m(self, rhs)
            }
        }
        impl core::ops::$atr for Algebraic {
            #[inline]
            fn $am(&mut self, rhs: Algebraic) {
                *self = Algebraic::$m(self, &rhs);
            }
        }
    };
}

alg_binop!(Add, add, AddAssign, add_assign);
alg_binop!(Sub, sub, SubAssign, sub_assign);
alg_binop!(Mul, mul, MulAssign, mul_assign);
alg_binop!(Div, div, DivAssign, div_assign);

impl core::ops::Neg for Algebraic {
    type Output = Algebraic;
    #[inline]
    fn neg(self) -> Algebraic {
        Algebraic::neg(&self)
    }
}

impl Algebraic {
    /// Returns `√self` (the non-negative root). Panics if `self` is negative.
    pub fn sqrt(&self) -> Algebraic {
        assert!(self.signum() >= 0, "Algebraic::sqrt: negative radicand");
        if self.signum() == 0 {
            return Algebraic::from_rational(Rational::ZERO);
        }
        // √α is a root of p(x²); isolate it near the float square root.
        let sub = compose_square(&self.poly);
        let sf = squarefree(&sub);
        let chain = sturm_chain(&sf);
        let mut a = self.clone();
        for _ in 0..4096 {
            // Rational bracket of √[lo, hi] with lo, hi ≥ 0.
            let lo = rational_sqrt_floor(&a.lo);
            let hi = rational_sqrt_ceil(&a.hi);
            if lo < hi && count_roots(&chain, &lo, &hi) == 1 {
                return Algebraic::new(sf, lo, hi);
            }
            a.refine();
        }
        panic!("Algebraic::sqrt: failed to isolate the root");
    }
}

/// `p(x²)`.
fn compose_square(p: &P) -> P {
    let n = p.degree().unwrap_or(0);
    let mut out = alloc::vec![Rational::ZERO; 2 * n + 1];
    for i in 0..=n {
        out[2 * i] = p.coeff(i);
    }
    Poly::new(out)
}

/// A rational `≤ √q` (for `q ≥ 0`), within `2⁻³²`.
fn rational_sqrt_floor(q: &Q) -> Q {
    let (lo, _) = rational_sqrt_bracket(q);
    lo
}
/// A rational `≥ √q` (for `q ≥ 0`), within `2⁻³²`.
fn rational_sqrt_ceil(q: &Q) -> Q {
    let (_, hi) = rational_sqrt_bracket(q);
    hi
}

/// Brackets `√q` between two nearby rationals by bisection.
fn rational_sqrt_bracket(q: &Q) -> (Q, Q) {
    if q.signum() <= 0 {
        return (Rational::ZERO, Rational::ZERO);
    }
    let mut lo = Rational::ZERO;
    // Upper bound: max(1, q).
    let mut hi = if *q > Rational::ONE {
        q.clone()
    } else {
        Rational::ONE
    };
    let eps = Rational::new(Int::ONE, Int::ONE.mul_2k(32));
    while hi.sub(&lo) > eps {
        let mid = lo.add(&hi).div(&q_i64(2));
        if mid.mul(&mid) <= *q {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo, hi)
}
