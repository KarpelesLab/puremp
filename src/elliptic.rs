//! Elliptic curves in short Weierstrass form `y² = x³ + a·x + b`.
//!
//! An [`EllipticCurve`] is defined by its two coefficients `a`, `b` drawn from a
//! [`Field`] `F`. The two primary base fields are:
//!
//! - a prime field `GF(p)` with coordinates as [`ModInt`]
//!   — the main target, carrying the cryptographic/number-theoretic content
//!   (point counting, point orders, `x`-coordinate recovery); and
//! - the rationals `ℚ` with coordinates as [`Rational`](crate::rational::Rational)
//!   — enough to add and double genuine rational points.
//!
//! The set of points — the affine solutions `(x, y)` together with a single
//! *point at infinity* `O` — forms an abelian group under the classical
//! chord-and-tangent law, with `O` as identity and `−(x, y) = (x, −y)`. A
//! [`Point`] carries a clone of its curve, so the group operators can be written
//! `P + Q`, `−P`, `k·P` without threading the curve through every call; adding
//! points from two different curves panics.
//!
//! # Non-singularity
//!
//! [`EllipticCurve::new`] rejects singular coefficients by checking the
//! discriminant
//!
//! ```text
//! Δ = −16 · (4a³ + 27b²) ≠ 0,
//! ```
//!
//! which for a field of characteristic `≠ 2, 3` is exactly the condition that
//! the cubic `x³ + a·x + b` has no repeated root. The [`j-invariant`] is
//! `j = 1728 · 4a³ / (4a³ + 27b²)`.
//!
//! [`j-invariant`]: EllipticCurve::j_invariant
//!
//! # Point counting
//!
//! [`EllipticCurve::curve_order`] and [`EllipticCurve::order_of_point`] over
//! `GF(p)` are implemented by a **naive `O(p)` scan** of the base field (summing
//! Legendre symbols), which is only practical for modest primes — a few million
//! at most. The asymptotically fast Schoof / Schoof–Elkies–Atkin algorithms are
//! left as future work.
//!
//! # Clean-room provenance
//!
//! The group law, discriminant, `j`-invariant and point-order material are drawn
//! from the open literature: Silverman, *The Arithmetic of Elliptic Curves*
//! (§III.1–III.2); Washington, *Elliptic Curves: Number Theory and Cryptography*
//! (§2–§4); the *Handbook of Applied Cryptography* §6; and Cohen, *A Course in
//! Computational Algebraic Number Theory* §7. No third-party source was
//! consulted.

use core::fmt;

use crate::int::Int;
use crate::mod_int::ModInt;
use crate::ring::Field;

/// Returns `n · x` for a small non-negative integer `n`, built from repeated
/// doubling within the field of `x` (so it works for any [`Field`], including
/// the context-carrying `ModInt` whose identities depend on the modulus).
fn field_mul_small<F: Field>(x: &F, mut n: u64) -> F {
    let mut acc = x.zero();
    let mut base = x.clone();
    while n > 0 {
        if n & 1 == 1 {
            acc = acc + base.clone();
        }
        n >>= 1;
        if n > 0 {
            base = base.clone() + base.clone();
        }
    }
    acc
}

/// An elliptic curve `y² = x³ + a·x + b` over a field `F`.
///
/// Construct one with [`EllipticCurve::new`], which validates non-singularity.
/// Over `GF(p)` use `F = ModInt`; over `ℚ` use `F = Rational`.
#[derive(Clone)]
pub struct EllipticCurve<F: Field> {
    a: F,
    b: F,
}

impl<F: Field> EllipticCurve<F> {
    /// Builds the curve `y² = x³ + a·x + b`, returning `None` if it is singular
    /// (discriminant `Δ = −16(4a³ + 27b²) = 0`). The coefficients must live in
    /// the same field (e.g. share a modulus for `ModInt`).
    pub fn new(a: F, b: F) -> Option<EllipticCurve<F>> {
        let curve = EllipticCurve { a, b };
        if curve.discriminant().is_zero() {
            None
        } else {
            Some(curve)
        }
    }

    /// Returns the coefficient `a`.
    #[inline]
    pub fn a(&self) -> &F {
        &self.a
    }

    /// Returns the coefficient `b`.
    #[inline]
    pub fn b(&self) -> &F {
        &self.b
    }

    /// Returns the discriminant `Δ = −16 · (4a³ + 27b²)`.
    ///
    /// A curve is non-singular (a genuine elliptic curve) exactly when `Δ ≠ 0`.
    pub fn discriminant(&self) -> F {
        let a3 = self.a.clone() * self.a.clone() * self.a.clone();
        let b2 = self.b.clone() * self.b.clone();
        let inner = field_mul_small(&a3, 4) + field_mul_small(&b2, 27);
        -field_mul_small(&inner, 16)
    }

    /// Returns the `j`-invariant `j = 1728 · 4a³ / (4a³ + 27b²)`.
    ///
    /// Two curves over the same field are isomorphic (over the algebraic
    /// closure) iff they share a `j`-invariant. In particular `j = 0` when
    /// `a = 0` and `j = 1728` when `b = 0`.
    pub fn j_invariant(&self) -> F {
        let a3 = self.a.clone() * self.a.clone() * self.a.clone();
        let b2 = self.b.clone() * self.b.clone();
        let denom = field_mul_small(&a3, 4) + field_mul_small(&b2, 27);
        // denom = -Δ/16 ≠ 0 for a valid curve, so the division is defined.
        field_mul_small(&a3, 6912) / denom
    }

    /// Returns the identity element, the point at infinity `O`.
    pub fn identity(&self) -> Point<F> {
        Point {
            curve: self.clone(),
            coords: None,
        }
    }

    /// Evaluates the curve's right-hand side `x³ + a·x + b`.
    fn rhs(&self, x: &F) -> F {
        x.clone() * x.clone() * x.clone() + self.a.clone() * x.clone() + self.b.clone()
    }

    /// Builds the affine point `(x, y)` if it lies on the curve, else `None`.
    pub fn point(&self, x: F, y: F) -> Option<Point<F>> {
        let p = Point {
            curve: self.clone(),
            coords: Some((x, y)),
        };
        if p.is_on_curve() { Some(p) } else { None }
    }
}

impl<F: Field + fmt::Display> fmt::Display for EllipticCurve<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y² = x³ + {}·x + {}", self.a, self.b)
    }
}

impl<F: Field + fmt::Debug> fmt::Debug for EllipticCurve<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EllipticCurve {{ a: {:?}, b: {:?} }}", self.a, self.b)
    }
}

impl<F: Field> PartialEq for EllipticCurve<F> {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b
    }
}

/// A point on an [`EllipticCurve`]: either an affine `(x, y)` or the identity
/// (the point at infinity `O`).
///
/// The point carries a clone of its curve, so [`Add`](core::ops::Add),
/// [`Neg`](core::ops::Neg) and [`scalar_mul`](Point::scalar_mul) need no extra
/// context. Adding points from different curves panics.
#[derive(Clone)]
pub struct Point<F: Field> {
    curve: EllipticCurve<F>,
    coords: Option<(F, F)>,
}

impl<F: Field> Point<F> {
    /// Returns the curve this point lives on.
    #[inline]
    pub fn curve(&self) -> &EllipticCurve<F> {
        &self.curve
    }

    /// Returns `true` if this is the point at infinity `O`.
    #[inline]
    pub fn is_infinity(&self) -> bool {
        self.coords.is_none()
    }

    /// Returns the affine coordinates `(x, y)`, or `None` for the point at
    /// infinity.
    #[inline]
    pub fn coordinates(&self) -> Option<(&F, &F)> {
        self.coords.as_ref().map(|(x, y)| (x, y))
    }

    /// Returns the affine `x`-coordinate, or `None` at infinity.
    #[inline]
    pub fn x(&self) -> Option<&F> {
        self.coords.as_ref().map(|(x, _)| x)
    }

    /// Returns the affine `y`-coordinate, or `None` at infinity.
    #[inline]
    pub fn y(&self) -> Option<&F> {
        self.coords.as_ref().map(|(_, y)| y)
    }

    /// Returns `true` if the point satisfies the curve equation (the point at
    /// infinity always does).
    pub fn is_on_curve(&self) -> bool {
        match &self.coords {
            None => true,
            Some((x, y)) => y.clone() * y.clone() == self.curve.rhs(x),
        }
    }

    /// Returns the inverse `−P`. For an affine point `−(x, y) = (x, −y)`; the
    /// identity is its own inverse.
    pub fn neg(&self) -> Point<F> {
        match &self.coords {
            None => self.clone(),
            Some((x, y)) => Point {
                curve: self.curve.clone(),
                coords: Some((x.clone(), -y.clone())),
            },
        }
    }

    /// Returns the doubling `2·P` (the tangent-line case of the group law).
    ///
    /// `O` doubles to `O`; a point of order two (`y = 0`) doubles to `O` because
    /// its tangent is vertical. Otherwise the slope is `λ = (3x² + a) / (2y)`.
    pub fn double(&self) -> Point<F> {
        let (x, y) = match &self.coords {
            None => return self.clone(),
            Some(p) => p,
        };
        if y.is_zero() {
            // Vertical tangent: 2·P = O for a 2-torsion point.
            return self.curve.identity();
        }
        let three_x2 = field_mul_small(&(x.clone() * x.clone()), 3);
        let num = three_x2 + self.curve.a.clone();
        let den = y.clone() + y.clone();
        let lambda = num / den;
        self.line_result(&lambda, x, x, y)
    }

    /// Returns the group sum `self + rhs` using the chord-and-tangent law.
    ///
    /// Identity: `O + Q = Q` and `P + O = P`. Inverse: if `P = −Q` (equal `x`,
    /// opposite `y`) the chord is vertical and the sum is `O`. Equal points fall
    /// through to [`double`](Point::double) (the tangent case); otherwise the
    /// slope is the secant `λ = (y₂ − y₁) / (x₂ − x₁)`.
    ///
    /// # Panics
    /// If `self` and `rhs` lie on different curves.
    pub fn add(&self, rhs: &Point<F>) -> Point<F> {
        assert!(
            self.curve == rhs.curve,
            "Point::add: points lie on different curves"
        );
        let (x1, y1) = match &self.coords {
            None => return rhs.clone(),
            Some(p) => p,
        };
        let (x2, y2) = match &rhs.coords {
            None => return self.clone(),
            Some(p) => p,
        };
        if x1 == x2 {
            // Same x: either P == Q (double) or P == −Q (vertical chord → O).
            if y1 == y2 {
                return self.double();
            }
            return self.curve.identity();
        }
        let lambda = (y2.clone() - y1.clone()) / (x2.clone() - x1.clone());
        self.line_result(&lambda, x1, x2, y1)
    }

    /// Completes the addition/doubling formulas given the slope `λ` and the two
    /// source `x`-coordinates (`x1`, `x2`) plus `y1`:
    /// `x₃ = λ² − x₁ − x₂`, `y₃ = λ(x₁ − x₃) − y₁`.
    fn line_result(&self, lambda: &F, x1: &F, x2: &F, y1: &F) -> Point<F> {
        let x3 = lambda.clone() * lambda.clone() - x1.clone() - x2.clone();
        let y3 = lambda.clone() * (x1.clone() - x3.clone()) - y1.clone();
        Point {
            curve: self.curve.clone(),
            coords: Some((x3, y3)),
        }
    }

    /// Returns the scalar multiple `k·P` by double-and-add. Negative `k` uses
    /// `(−k)·P = −(k·P)`; `k = 0` gives `O`.
    pub fn scalar_mul(&self, k: &Int) -> Point<F> {
        if k.is_zero() || self.is_infinity() {
            return self.curve.identity();
        }
        let mag = k.abs();
        let mut result = self.curve.identity();
        let base = self.clone();
        // Left-to-right binary ladder over the bits of |k|.
        let mut i = mag.bit_len();
        while i > 0 {
            i -= 1;
            result = result.double();
            if mag.bit(i) {
                result = result.add(&base);
            }
        }
        if k.is_negative() {
            result.neg()
        } else {
            result
        }
    }
}

impl<F: Field> PartialEq for Point<F> {
    fn eq(&self, other: &Self) -> bool {
        self.curve == other.curve && self.coords == other.coords
    }
}

impl<F: Field + fmt::Display> fmt::Display for Point<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.coords {
            None => write!(f, "O"),
            Some((x, y)) => write!(f, "({}, {})", x, y),
        }
    }
}

impl<F: Field + fmt::Debug> fmt::Debug for Point<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.coords {
            None => write!(f, "Point(O)"),
            Some((x, y)) => write!(f, "Point({:?}, {:?})", x, y),
        }
    }
}

impl<F: Field> core::ops::Add for Point<F> {
    type Output = Point<F>;
    #[inline]
    fn add(self, rhs: Point<F>) -> Point<F> {
        Point::add(&self, &rhs)
    }
}

impl<F: Field> core::ops::Add<&Point<F>> for &Point<F> {
    type Output = Point<F>;
    #[inline]
    fn add(self, rhs: &Point<F>) -> Point<F> {
        Point::add(self, rhs)
    }
}

impl<F: Field> core::ops::Neg for Point<F> {
    type Output = Point<F>;
    #[inline]
    fn neg(self) -> Point<F> {
        Point::neg(&self)
    }
}

impl<F: Field> core::ops::Neg for &Point<F> {
    type Output = Point<F>;
    #[inline]
    fn neg(self) -> Point<F> {
        Point::neg(self)
    }
}

// --- GF(p)-specific utilities (point counting and orders) ---

impl EllipticCurve<ModInt> {
    /// Returns the base-field prime `p` (the modulus of the coefficients).
    #[inline]
    pub fn field_prime(&self) -> Int {
        self.a.modulus()
    }

    /// Recovers a point from its `x`-coordinate by solving `y² = x³ + a·x + b`
    /// with a modular square root, or returns `None` if the right-hand side is a
    /// quadratic non-residue. When two roots exist the one returned by
    /// [`sqrt_mod`](crate::int::Int::sqrt_mod) (in `[0, p)`) is used; negate the
    /// point for the other.
    pub fn point_from_x(&self, x: &ModInt) -> Option<Point<ModInt>> {
        let p = self.field_prime();
        let rhs = self.rhs(x);
        let y = rhs.to_int().sqrt_mod(&p)?;
        Some(Point {
            curve: self.clone(),
            coords: Some((x.clone(), x.of(y))),
        })
    }

    /// Returns the curve order `#E(GF(p))` — the number of affine points plus
    /// one for the point at infinity — by a naive `O(p)` scan summing Legendre
    /// symbols. Only practical for modest `p` (see the [module docs](self)); the
    /// result satisfies the Hasse bound `|#E − (p + 1)| ≤ 2√p`.
    pub fn curve_order(&self) -> Int {
        let p = self.field_prime();
        // Start at 1 for the point at infinity.
        let mut count = Int::ONE;
        let mut xi = self.a.of(Int::ZERO);
        let one = self.a.of(Int::ONE);
        let mut x = Int::ZERO;
        while x < p {
            let rhs = self.rhs(&xi);
            if rhs.is_zero() {
                count += Int::ONE; // single root y = 0
            } else {
                // 1 + Legendre(rhs, p): two points if a QR, none otherwise.
                let leg = rhs.to_int().legendre(&p);
                count += Int::from(1 + leg);
            }
            xi += one.clone();
            x += Int::ONE;
        }
        count
    }

    /// Returns the order of `point`: the least `n > 0` with `n·P = O` (`1` for
    /// the identity). Computed from the group order `N = #E` (via
    /// [`curve_order`](Self::curve_order)): starting from `N`, each prime factor
    /// is stripped while the point still vanishes, leaving the exact order (a
    /// divisor of `N`, by Lagrange).
    pub fn order_of_point(&self, point: &Point<ModInt>) -> Int {
        assert!(
            *point.curve() == *self,
            "order_of_point: point lies on a different curve"
        );
        if point.is_infinity() {
            return Int::ONE;
        }
        let mut order = self.curve_order();
        for q in order.clone().factorize() {
            // Strip the prime `q` from `order` as long as (order/q)·P = O.
            loop {
                let (quot, rem) = order.div_rem_trunc(&q);
                if !rem.is_zero() {
                    break;
                }
                if !point.scalar_mul(&quot).is_infinity() {
                    break;
                }
                order = quot;
            }
        }
        order
    }
}
