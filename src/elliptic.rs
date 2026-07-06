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

/// Internal Jacobian projective point `(X : Y : Z)` whose affine image is
/// `x = X / Z²`, `y = Y / Z³`, with `Z = 0` the point at infinity.
///
/// Jacobian coordinates let [`Point::scalar_mul`] run its double-and-add ladder
/// with **no per-step field inversion** — the group law becomes a fixed handful
/// of field multiplications, and a single inversion at the very end recovers the
/// affine `(x, y)`. This is the dominant cost saving over `GF(p)` (where an
/// inversion is a modular inverse) and over `ℚ` (a gcd). See Cohen, *A Course in
/// Computational Algebraic Number Theory* §7.2 and Hankerson–Menezes–Vanstone,
/// *Guide to Elliptic Curve Cryptography* §3.2 for the standard formulas.
#[derive(Clone)]
struct Jac<F: Field> {
    x: F,
    y: F,
    z: F,
}

impl<F: Field> Jac<F> {
    /// The point at infinity `(1 : 1 : 0)`, its identities drawn from `sample`'s
    /// ring (so `ModInt` gets the right modulus).
    #[inline]
    fn infinity(sample: &F) -> Jac<F> {
        Jac {
            x: sample.one(),
            y: sample.one(),
            z: sample.zero(),
        }
    }

    /// Whether this is the point at infinity (`Z = 0`).
    #[inline]
    fn is_infinity(&self) -> bool {
        self.z.is_zero()
    }
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

    // --- Jacobian group law (inversion-free, used by the scalar ladder) ---

    /// Jacobian point doubling `2·P` for `y² = x³ + a·x + b`:
    ///
    /// ```text
    /// S  = 4·X·Y²
    /// M  = 3·X² + a·Z⁴
    /// X₃ = M² − 2·S
    /// Y₃ = M·(S − X₃) − 8·Y⁴
    /// Z₃ = 2·Y·Z
    /// ```
    ///
    /// The point at infinity (`Z = 0`) and any 2-torsion point (`Y = 0`, whose
    /// tangent is vertical) both double to infinity — returned explicitly, which
    /// also matches `Z₃ = 2·Y·Z = 0` in those cases.
    fn jac_double(&self, p: &Jac<F>) -> Jac<F> {
        if p.z.is_zero() || p.y.is_zero() {
            return Jac::infinity(&self.a);
        }
        let xx = p.x.clone() * p.x.clone();
        let yy = p.y.clone() * p.y.clone();
        let yyyy = yy.clone() * yy.clone();
        let zz = p.z.clone() * p.z.clone();
        let z4 = zz.clone() * zz;
        // S = 4·X·Y², M = 3·X² + a·Z⁴.
        let s = field_mul_small(&(p.x.clone() * yy), 4);
        let m = field_mul_small(&xx, 3) + self.a.clone() * z4;
        let two_s = s.clone() + s.clone();
        let x3 = m.clone() * m.clone() - two_s;
        let y3 = m * (s - x3.clone()) - field_mul_small(&yyyy, 8);
        let z3 = field_mul_small(&(p.y.clone() * p.z.clone()), 2);
        Jac {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// General Jacobian point addition `P₁ + P₂`:
    ///
    /// ```text
    /// U₁ = X₁·Z₂²   U₂ = X₂·Z₁²
    /// S₁ = Y₁·Z₂³   S₂ = Y₂·Z₁³
    /// H  = U₂ − U₁  r  = S₂ − S₁
    /// X₃ = r² − H³ − 2·U₁·H²
    /// Y₃ = r·(U₁·H² − X₃) − S₁·H³
    /// Z₃ = Z₁·Z₂·H
    /// ```
    ///
    /// Edge cases: if either input is infinity the other is returned. When the
    /// affine `x`-coordinates coincide (`H = 0`) the chord degenerates: `r = 0`
    /// means `P₁ = P₂`, deferred to [`jac_double`](Self::jac_double); `r ≠ 0`
    /// means `P₁ = −P₂`, giving infinity.
    fn jac_add(&self, p1: &Jac<F>, p2: &Jac<F>) -> Jac<F> {
        if p1.is_infinity() {
            return p2.clone();
        }
        if p2.is_infinity() {
            return p1.clone();
        }
        let z1z1 = p1.z.clone() * p1.z.clone();
        let z2z2 = p2.z.clone() * p2.z.clone();
        let u1 = p1.x.clone() * z2z2.clone();
        let u2 = p2.x.clone() * z1z1.clone();
        let s1 = p1.y.clone() * z2z2 * p2.z.clone();
        let s2 = p2.y.clone() * z1z1 * p1.z.clone();
        let h = u2 - u1.clone();
        let r = s2 - s1.clone();
        if h.is_zero() {
            if r.is_zero() {
                return self.jac_double(p1);
            }
            return Jac::infinity(&self.a);
        }
        let h2 = h.clone() * h.clone();
        let h3 = h2.clone() * h.clone();
        let u1h2 = u1 * h2;
        let two_u1h2 = u1h2.clone() + u1h2.clone();
        let x3 = r.clone() * r.clone() - h3.clone() - two_u1h2;
        let y3 = r * (u1h2 - x3.clone()) - s1 * h3;
        let z3 = p1.z.clone() * p2.z.clone() * h;
        Jac {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// Converts a Jacobian point back to an affine [`Point`] with the ladder's
    /// **single** field inversion: `x = X·Z⁻²`, `y = Y·Z⁻³` (infinity if
    /// `Z = 0`).
    fn jac_to_affine(&self, p: Jac<F>) -> Point<F> {
        if p.is_infinity() {
            return self.identity();
        }
        let z_inv = self.a.one() / p.z;
        let z_inv2 = z_inv.clone() * z_inv.clone();
        let z_inv3 = z_inv2.clone() * z_inv;
        Point {
            curve: self.clone(),
            coords: Some((p.x * z_inv2, p.y * z_inv3)),
        }
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
    ///
    /// The ladder runs in inversion-free **Jacobian coordinates** (`x = X/Z²`,
    /// `y = Y/Z³`): the affine base point enters as `(x : y : 1)`, each
    /// double/add is a fixed set of field multiplications with no division, and
    /// exactly one field inversion at the end recovers the affine result. Because
    /// the base fields are exact with a canonical representation, this is
    /// bit-identical to the affine double-and-add it replaces — the affine
    /// [`double`](Self::double)/[`add`](Self::add) remain the public single-step
    /// operators and the differential reference.
    pub fn scalar_mul(&self, k: &Int) -> Point<F> {
        let (x, y) = match &self.coords {
            _ if k.is_zero() => return self.curve.identity(),
            None => return self.curve.identity(),
            Some(p) => p,
        };
        let mag = k.abs();

        // Jacobian coordinates trade the per-step field inversion for extra
        // multiplies on larger coordinates — a huge win when inversion is a full
        // algorithm (GF(p) modular inverse), but a *loss* when it is nearly free
        // (a `Rational` reciprocal is a num/den swap). Dispatch on the field.
        let result = if F::CHEAP_INV {
            // Affine ladder: cheap inversion, and affine coordinates stay small.
            let base = Point {
                curve: self.curve.clone(),
                coords: Some((x.clone(), y.clone())),
            };
            let mut acc = self.curve.identity();
            let mut i = mag.bit_len();
            while i > 0 {
                i -= 1;
                acc = acc.double();
                if mag.bit(i) {
                    acc = acc.add(&base);
                }
            }
            acc
        } else {
            // Jacobian ladder: one inversion at the very end.
            let base = Jac {
                x: x.clone(),
                y: y.clone(),
                z: x.one(),
            };
            let mut acc = Jac::infinity(&self.curve.a);
            let mut i = mag.bit_len();
            while i > 0 {
                i -= 1;
                acc = self.curve.jac_double(&acc);
                if mag.bit(i) {
                    acc = self.curve.jac_add(&acc, &base);
                }
            }
            self.curve.jac_to_affine(acc)
        };
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
