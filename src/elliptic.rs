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
//! [`EllipticCurve::point_count`] (and the equivalent
//! [`EllipticCurve::curve_order`], used by [`EllipticCurve::order_of_point`])
//! over `GF(p)` compute `#E(GF(p)) = p + 1 − t` by two routes, dispatched on the
//! size of `p`:
//!
//! - a **naive `O(p)` scan** of the base field (summing Legendre symbols), used
//!   for small `p` and as the differential cross-check; and
//! - **Schoof's algorithm** ([`schoof_point_count`](EllipticCurve::schoof_point_count)),
//!   polynomial-time in `log p`, used above a size threshold where the scan is
//!   impractical.
//!
//! Schoof recovers the Frobenius trace `t` one small prime `ℓ` at a time. For a
//! set of primes with `∏ ℓ > 4√p`, `t mod ℓ` is found by working in the ring
//! `GF(p)[x] / (ψ_ℓ(x))` (with the curve relation `y² = x³ + a·x + b`), where
//! `ψ_ℓ` is the `ℓ`-th division polynomial: the trace is the unique `t_ℓ` making
//! the Frobenius `φ : (x, y) ↦ (x^p, y^p)` satisfy
//! `φ² − [t_ℓ]φ + [p] ≡ 0` on the `ℓ`-torsion `E[ℓ]`. The residues are combined
//! by the Chinese Remainder Theorem and `t` taken in `[−2√p, 2√p]` (Hasse). The
//! prime `ℓ = 2` is special-cased via `gcd(x^p − x, x³ + a·x + b)` (a nontrivial
//! gcd means a rational 2-torsion point, so `#E` is even, i.e. `t ≡ 0 mod 2`).
//!
//! Only the classical Schoof algorithm is implemented; the Elkies/Atkin
//! improvements (SEA) are left as future work.
//!
//! # Clean-room provenance
//!
//! The group law, discriminant, `j`-invariant and point-order material are drawn
//! from the open literature: Silverman, *The Arithmetic of Elliptic Curves*
//! (§III.1–III.2); Washington, *Elliptic Curves: Number Theory and Cryptography*
//! (§2–§4); the *Handbook of Applied Cryptography* §6; and Cohen, *A Course in
//! Computational Algebraic Number Theory* §7. Schoof's point-counting algorithm
//! follows R. Schoof, *Elliptic curves over finite fields and the computation of
//! square roots mod p*, Math. Comp. **44** (1985); Cohen §7.4–7.5; and
//! Blake–Seroussi–Smart, *Elliptic Curves in Cryptography*, ch. VII. No
//! third-party source was consulted.

use core::fmt;

use alloc::vec::Vec;

use crate::int::Int;
use crate::mod_int::ModInt;
use crate::poly::Poly;
use crate::ring::{Field, Ring};

/// Bit-length of `p` at or above which [`EllipticCurve::point_count`] switches
/// from the naive `O(p)` scan to [Schoof's algorithm](EllipticCurve::schoof_point_count).
/// Below it (and for `p ∈ {2, 3}`) the exact scan is both faster and simpler.
///
/// The measured crossover sits near `p ≈ 10^5`; `2^20 ≈ 10^6` is a small step
/// above it, where Schoof is already several times faster (e.g. ~3.5× at
/// `p ≈ 10^6`, ~70× at `p ≈ 10^8`) and the scan starts to hurt.
const SCHOOF_BITS: u32 = 20;

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
    /// one for the point at infinity.
    ///
    /// Dispatches on the size of `p`: a naive `O(p)` scan for small `p`, and
    /// [Schoof's algorithm](Self::schoof_point_count) (polynomial in `log p`)
    /// above a threshold where the scan is impractical (see the
    /// [module docs](self)). Identical in value to
    /// [`point_count`](Self::point_count); the result satisfies the Hasse bound
    /// `|#E − (p + 1)| ≤ 2√p`.
    pub fn curve_order(&self) -> Int {
        self.point_count()
    }

    /// Returns the number of points `#E(GF(p)) = p + 1 − t` on the curve.
    ///
    /// Uses the naive `O(p)` scan for small `p` (below `2^SCHOOF_BITS`) and
    /// [Schoof's algorithm](Self::schoof_point_count) for larger `p`. Both
    /// return the same value; Schoof is polynomial-time in `log p`, so it is the
    /// only feasible route for cryptographic-size primes.
    pub fn point_count(&self) -> Int {
        let p = self.field_prime();
        // Schoof needs characteristic ≠ 2, 3 and only pays off well above the
        // scan's reach; small `p` (including p ∈ {2, 3}) take the exact scan.
        if p.bit_len() >= SCHOOF_BITS && p > Int::from(3) {
            self.schoof_point_count()
        } else {
            self.naive_curve_order()
        }
    }

    /// Returns `#E(GF(p))` by a naive `O(p)` scan of the base field, summing
    /// `1 + Legendre(x³ + a·x + b, p)` over all `x` (plus one for infinity). Only
    /// practical for modest `p`; it is the differential cross-check for
    /// [`schoof_point_count`](Self::schoof_point_count).
    #[doc(hidden)]
    pub fn naive_curve_order(&self) -> Int {
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

// ===========================================================================
// Schoof's point-counting algorithm over GF(p).
//
// `#E(GF(p)) = p + 1 − t`, where `t` is the trace of the Frobenius
// endomorphism `φ : (x, y) ↦ (x^p, y^p)`. Schoof recovers `t mod ℓ` for enough
// small primes `ℓ` (with `∏ ℓ > 4√p`) and CRTs them, exploiting the identity
// `φ² − [t]φ + [p] ≡ 0` on the `ℓ`-torsion `E[ℓ]`. All work happens in the ring
// `R = GF(p)[x]/(ψ_ℓ(x))` together with the curve relation `y² = x³ + a·x + b`,
// where `ψ_ℓ` is the `ℓ`-th division polynomial (of degree `(ℓ²−1)/2` for odd
// `ℓ`), whose roots are exactly the `x`-coordinates of the nonzero `ℓ`-torsion.
//
// Every point that arises (Frobenius images, and integer multiples of the
// generic torsion point `(x, y)`) has `y`-coordinate of the form `y·b(x)`, so a
// point is stored as `(a, b)` meaning `(a(x), y·b(x))` with `a, b ∈ R`. The
// group law stays inside this shape (the `y²` in `λ²` collapses to
// `f = x³+ax+b`). See Schoof, Math. Comp. 44 (1985); Cohen §7.4–7.5;
// Blake–Seroussi–Smart ch. VII.
// ===========================================================================

/// Outcome of inverting a polynomial modulo `h` in `GF(p)[x]/(h)`.
enum PolyInv {
    /// The inverse — `gcd(a, h)` was a unit, so `a` is invertible mod `h`.
    Unit(Poly<ModInt>),
    /// `a ≡ 0 (mod h)`: genuinely zero, no inverse.
    Zero,
    /// `gcd(a, h)` is a nontrivial proper (monic) factor of `h`, i.e. `h` is
    /// reducible. The caller restarts the `ℓ`-computation modulo this factor.
    Factor(Poly<ModInt>),
}

/// Inverts `a` modulo the monic polynomial `h` over `GF(p)` via the extended
/// Euclidean algorithm, tracking a cofactor `s` with `s·a ≡ r (mod h)`. See
/// [`PolyInv`].
fn poly_inv_mod(a: &Poly<ModInt>, h: &Poly<ModInt>) -> PolyInv {
    let a = a.rem(h);
    if a.is_zero() {
        return PolyInv::Zero;
    }
    let one = a.leading().expect("a is nonzero").one();
    let mut r0 = h.clone();
    let mut r1 = a.clone();
    let mut s0 = Poly::<ModInt>::zero();
    let mut s1 = Poly::constant(one);
    while !r1.is_zero() {
        let (q, r) = r0.div_rem(&r1);
        r0 = r1;
        r1 = r;
        let s = s0.sub(&q.mul(&s1));
        s0 = s1;
        s1 = s;
    }
    // r0 = gcd(a, h) (up to a scalar); s0·a ≡ r0 (mod h). Since `a` is a nonzero
    // remainder its gcd with `h` has degree < deg h, so a nonconstant gcd is a
    // proper factor.
    match r0.degree() {
        Some(0) => {
            let c_inv = r0.coeff(0).inv().expect("a nonzero constant is invertible");
            PolyInv::Unit(s0.scalar_mul(&c_inv).rem(h))
        }
        Some(_) => PolyInv::Factor(r0.monic()),
        None => unreachable!("gcd of nonzero operands is nonzero"),
    }
}

/// `a·b mod h`.
fn poly_mulmod(a: &Poly<ModInt>, b: &Poly<ModInt>, h: &Poly<ModInt>) -> Poly<ModInt> {
    a.mul(b).rem(h)
}

/// `base^exp mod h` by square-and-multiply (`exp ≥ 0`). `one` supplies the
/// coefficient ring's multiplicative identity for the initial accumulator.
fn poly_powmod(base: &Poly<ModInt>, exp: &Int, h: &Poly<ModInt>, one: &ModInt) -> Poly<ModInt> {
    let mut result = Poly::constant(one.clone());
    let mut b = base.rem(h);
    let bits = exp.bit_len();
    for i in 0..bits {
        if exp.bit(i) {
            result = poly_mulmod(&result, &b, h);
        }
        if i + 1 < bits {
            b = poly_mulmod(&b, &b, h);
        }
    }
    result
}

/// A point on `E` over `R = GF(p)[x]/(h)`, with the `y`-coordinate written as
/// `y·b`. `Aff { a, b }` denotes `(a(x), y·b(x))`.
#[derive(Clone)]
enum RingPoint {
    /// The point at infinity.
    Inf,
    /// The affine point `(a, y·b)`.
    Aff { a: Poly<ModInt>, b: Poly<ModInt> },
}

/// Shared data for the ring group law modulo one factor `h` of `ψ_ℓ`.
struct SchoofCtx {
    /// Current modulus (a monic factor of `ψ_ℓ`).
    h: Poly<ModInt>,
    /// `f = x³ + a·x + b` reduced mod `h`.
    f: Poly<ModInt>,
    /// `f⁻¹ mod h` (always exists: `gcd(f, ψ_ℓ) = 1` for odd `ℓ`).
    f_inv: Poly<ModInt>,
    /// The curve coefficient `a` as a constant polynomial.
    a_poly: Poly<ModInt>,
    /// The coefficient ring's `1` (for building small scalar constants).
    one: ModInt,
}

/// Group law `P ⊕ Q` in `R`. Returns `Err(g)` when a required inverse exposes a
/// proper factor `g` of `h` (`h` reducible) — the signal to restart modulo `g`.
fn ring_add(p: &RingPoint, q: &RingPoint, ctx: &SchoofCtx) -> Result<RingPoint, Poly<ModInt>> {
    let (a1, b1) = match p {
        RingPoint::Inf => return Ok(q.clone()),
        RingPoint::Aff { a, b } => (a, b),
    };
    let (a2, b2) = match q {
        RingPoint::Inf => return Ok(p.clone()),
        RingPoint::Aff { a, b } => (a, b),
    };
    let d = a1.sub(a2).rem(&ctx.h);
    if d.is_zero() {
        // Equal x-coordinates: either a doubling or opposite points.
        let bs = b1.sub(b2).rem(&ctx.h);
        if bs.is_zero() {
            return ring_double(p, ctx);
        }
        return Ok(RingPoint::Inf);
    }
    let dinv = match poly_inv_mod(&d, &ctx.h) {
        PolyInv::Unit(v) => v,
        PolyInv::Factor(g) => return Err(g),
        PolyInv::Zero => unreachable!("d is nonzero mod h"),
    };
    // λ' = (b1 − b2)/(a1 − a2); the true slope is y·λ'.
    let lam = poly_mulmod(&b1.sub(b2), &dinv, &ctx.h);
    let lam2 = poly_mulmod(&lam, &lam, &ctx.h);
    // x3 = y²·λ'² − a1 − a2 = f·λ'² − a1 − a2.
    let x3 = poly_mulmod(&ctx.f, &lam2, &ctx.h)
        .sub(a1)
        .sub(a2)
        .rem(&ctx.h);
    // y3 = y·(λ'·(a1 − x3) − b1).
    let b3 = poly_mulmod(&lam, &a1.sub(&x3), &ctx.h).sub(b1).rem(&ctx.h);
    Ok(RingPoint::Aff { a: x3, b: b3 })
}

/// Doubling `[2]P` in `R`. Returns `Err(g)` on a factor of `h`, as
/// [`ring_add`].
fn ring_double(p: &RingPoint, ctx: &SchoofCtx) -> Result<RingPoint, Poly<ModInt>> {
    let (a1, b1) = match p {
        RingPoint::Inf => return Ok(RingPoint::Inf),
        RingPoint::Aff { a, b } => (a, b),
    };
    let b1 = b1.rem(&ctx.h);
    if b1.is_zero() {
        // y-coordinate y·b1 ≡ 0: a 2-torsion point doubles to infinity.
        return Ok(RingPoint::Inf);
    }
    let two = ctx.one.of(Int::from(2));
    let three = ctx.one.of(Int::from(3));
    // μ = (3·a1² + A)/(2·b1); the true slope is μ/y.
    let a1sq = poly_mulmod(a1, a1, &ctx.h);
    let num = a1sq.scalar_mul(&three).add(&ctx.a_poly).rem(&ctx.h);
    let den = b1.scalar_mul(&two).rem(&ctx.h);
    let dinv = match poly_inv_mod(&den, &ctx.h) {
        PolyInv::Unit(v) => v,
        PolyInv::Factor(g) => return Err(g),
        PolyInv::Zero => unreachable!("den is nonzero mod h"),
    };
    let mu = poly_mulmod(&num, &dinv, &ctx.h);
    let mu2 = poly_mulmod(&mu, &mu, &ctx.h);
    // x3 = (μ/y)² − 2·a1 = μ²/f − 2·a1.
    let two_a1 = a1.scalar_mul(&two);
    let x3 = poly_mulmod(&mu2, &ctx.f_inv, &ctx.h)
        .sub(&two_a1)
        .rem(&ctx.h);
    // y3 = y·(μ·(a1 − x3)/f − b1).
    let t = poly_mulmod(&mu, &a1.sub(&x3), &ctx.h);
    let b3 = poly_mulmod(&t, &ctx.f_inv, &ctx.h).sub(&b1).rem(&ctx.h);
    Ok(RingPoint::Aff { a: x3, b: b3 })
}

/// `[k]·base` in `R` by double-and-add (`k` a small non-negative scalar).
fn ring_mul(mut k: u64, base: &RingPoint, ctx: &SchoofCtx) -> Result<RingPoint, Poly<ModInt>> {
    let mut result = RingPoint::Inf;
    let mut b = base.clone();
    while k > 0 {
        if k & 1 == 1 {
            result = ring_add(&result, &b, ctx)?;
        }
        k >>= 1;
        if k > 0 {
            b = ring_double(&b, ctx)?;
        }
    }
    Ok(result)
}

/// Trial-division primality for the small `ℓ`-sieve.
fn is_small_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

impl EllipticCurve<ModInt> {
    /// The curve's right-hand side `f(x) = x³ + a·x + b` as a polynomial over
    /// `GF(p)`.
    fn rhs_poly(&self) -> Poly<ModInt> {
        // Coefficients low-to-high: b + a·x + 0·x² + 1·x³.
        Poly::new(alloc::vec![
            self.b.clone(),
            self.a.clone(),
            self.a.zero(),
            self.a.one(),
        ])
    }

    /// Returns `#E(GF(p)) = p + 1 − t` via **Schoof's algorithm**, polynomial in
    /// `log p`.
    ///
    /// Determines `t mod ℓ` for successive primes `ℓ` (the prime `2` by a gcd
    /// parity test, odd `ℓ` by locating the Frobenius eigenvalue on `E[ℓ]`) until
    /// `∏ ℓ > 4√p`, then recovers `t` by the Chinese Remainder Theorem in the
    /// Hasse interval `[−2√p, 2√p]`. Requires `p > 3` (characteristic `≠ 2, 3`).
    ///
    /// This is the asymptotically fast counterpart to
    /// [`naive_curve_order`](Self::naive_curve_order); both return the same
    /// value. [`point_count`](Self::point_count) selects between them by size.
    pub fn schoof_point_count(&self) -> Int {
        let p = self.field_prime();
        assert!(
            p > Int::from(3),
            "schoof_point_count: requires characteristic ≠ 2, 3"
        );
        let mut moduli: Vec<Int> = Vec::new();
        let mut residues: Vec<Int> = Vec::new();
        let mut prod = Int::ONE;
        // Need ∏ ℓ > 4√p, tested squared as (∏ ℓ)² > 16·p to stay in integers.
        let bound = Int::from(16) * p.clone();
        let mut l: usize = 2;
        loop {
            if is_small_prime(l) && Int::from(l as i64) != p {
                let t_l: u64 = if l == 2 {
                    self.schoof_t_mod_2()
                } else {
                    let t = self.schoof_t_mod_odd_l(l);
                    let lm = l as i64;
                    (((t % lm) + lm) % lm) as u64
                };
                residues.push(Int::from(t_l));
                moduli.push(Int::from(l as i64));
                prod *= Int::from(l as i64);
                if prod.clone() * prod.clone() > bound {
                    break;
                }
            }
            l += 1;
        }
        let m = prod;
        let t0 = Int::crt(&residues, &moduli).expect("moduli are distinct primes");
        // Symmetric representative in (−M/2, M/2]: shift down when 2·t0 > M.
        let t = if t0.clone() * Int::from(2) > m {
            t0 - m
        } else {
            t0
        };
        p + Int::ONE - t
    }

    /// `t mod 2`: `#E` is even iff `E` has a rational 2-torsion point iff
    /// `x³ + a·x + b` has a root in `GF(p)` iff `gcd(x^p − x, f) ≠ 1`. Returns
    /// `0` when even, `1` when odd.
    fn schoof_t_mod_2(&self) -> u64 {
        let p = self.field_prime();
        let one = self.a.one();
        let f = self.rhs_poly();
        let x = Poly::monomial(one.clone(), 1);
        let xp = poly_powmod(&x, &p, &f, &one);
        let g = xp.sub(&x).gcd(&f);
        match g.degree() {
            Some(d) if d >= 1 => 0,
            _ => 1,
        }
    }

    /// `t mod ℓ` for an odd prime `ℓ`, as a symmetric representative in
    /// `(−ℓ/2, ℓ/2]`. Works modulo `ψ_ℓ`, restarting modulo a factor whenever
    /// `ψ_ℓ` is found reducible (via a failed ring inverse) — the trace is the
    /// same on any nonzero factor, and the modulus degree strictly decreases, so
    /// this terminates.
    fn schoof_t_mod_odd_l(&self, l: usize) -> i64 {
        let psis = self.division_polys(l);
        let mut h = psis[l].monic();
        // p mod ℓ, in [1, ℓ−1] (ℓ ≠ p since ℓ < p and both prime).
        let k = self
            .field_prime()
            .rem_euclid(&Int::from(l as i64))
            .to_u64()
            .expect("ℓ fits in u64");
        loop {
            match self.try_schoof_l(l, k, &h) {
                Ok(t) => return t,
                Err(factor) => h = factor,
            }
        }
    }

    /// One attempt at `t mod ℓ` working modulo the factor `h` of `ψ_ℓ`. Returns
    /// `Err(g)` to request a restart modulo a smaller factor `g`.
    fn try_schoof_l(&self, l: usize, k: u64, h: &Poly<ModInt>) -> Result<i64, Poly<ModInt>> {
        let p = self.field_prime();
        let one = self.a.one();
        let f_full = self.rhs_poly();
        let f = f_full.rem(h);
        let f_inv = match poly_inv_mod(&f, h) {
            PolyInv::Unit(v) => v,
            PolyInv::Factor(g) => return Err(g),
            PolyInv::Zero => return Err(h.clone()),
        };
        let ctx = SchoofCtx {
            h: h.clone(),
            f,
            f_inv,
            a_poly: Poly::constant(self.a.clone()),
            one: one.clone(),
        };
        let x = Poly::monomial(one.clone(), 1);
        // Frobenius: x^p, x^{p²}, and the y-factors y^p = y·f^{(p−1)/2},
        // y^{p²} = y·f^{(p²−1)/2}.
        let xp = poly_powmod(&x, &p, h, &one);
        let xpp = poly_powmod(&xp, &p, h, &one);
        let e1 = (p.clone() - Int::ONE).div_floor(&Int::from(2));
        let yp = poly_powmod(&f_full, &e1, h, &one);
        let p2 = p.clone() * p.clone();
        let e2 = (p2 - Int::ONE).div_floor(&Int::from(2));
        let ypp = poly_powmod(&f_full, &e2, h, &one);
        let phi = RingPoint::Aff { a: xp, b: yp };
        let phi2 = RingPoint::Aff { a: xpp, b: ypp };
        // [k]P for the generic torsion point P = (x, y) = (x, y·1).
        let base = RingPoint::Aff {
            a: x.rem(h),
            b: Poly::constant(one.clone()),
        };
        let kp = ring_mul(k, &base, &ctx)?;
        // S = φ²(P) ⊕ [k]P. By φ² − [t]φ + [k] ≡ 0 on E[ℓ], S = [t]·φ(P).
        let s = ring_add(&phi2, &kp, &ctx)?;
        let (sa, sb) = match &s {
            RingPoint::Inf => return Ok(0), // [t]φ(P) = O ⟹ ℓ | t
            RingPoint::Aff { a, b } => (a.clone(), b.clone()),
        };
        // Find τ ∈ [1, (ℓ−1)/2] with S = ±[τ]φ(P), reading the sign off the
        // y-factor: S = +[τ]φ ⟹ t ≡ τ, S = −[τ]φ ⟹ t ≡ −τ.
        let mut t_pt = phi.clone();
        let half = (l - 1) / 2;
        for tau in 1..=half {
            if tau > 1 {
                t_pt = ring_add(&t_pt, &phi, &ctx)?;
            }
            if let RingPoint::Aff { a: ta, b: tb } = &t_pt
                && sa == *ta
            {
                if sb == *tb {
                    return Ok(tau as i64);
                }
                if sb == tb.neg() {
                    return Ok(-(tau as i64));
                }
                // Same x but y-factor neither equal nor negated: a zero divisor,
                // so h is reducible — restart modulo the exposed factor.
                let g = sb.sub(tb).gcd(h);
                if let Some(d) = g.degree()
                    && d >= 1
                    && d < h.degree().expect("h is nonzero")
                {
                    return Err(g);
                }
                return Ok(tau as i64);
            }
        }
        unreachable!("Schoof: no Frobenius eigenvalue found for ℓ = {l}")
    }

    /// The reduced division polynomials `ψ̄_0 … ψ̄_l` as polynomials in `x` over
    /// `GF(p)`, where `ψ_n = ψ̄_n` for odd `n` and `ψ_n = y·ψ̄_n` for even `n`
    /// (so `ψ̄_ℓ` for odd `ℓ` is the honest `ℓ`-division polynomial). With
    /// `F = x³+ax+b`, the standard recurrences reduce (via `y² = F`) to:
    ///
    /// ```text
    /// ψ̄_{2m+1} = ψ̄_{m+2}·ψ̄_m³ − F²·ψ̄_{m−1}·ψ̄_{m+1}³   (m odd)
    /// ψ̄_{2m+1} = F²·ψ̄_{m+2}·ψ̄_m³ − ψ̄_{m−1}·ψ̄_{m+1}³   (m even)
    /// ψ̄_{2m}   = (ψ̄_m/2)·(ψ̄_{m+2}·ψ̄_{m−1}² − ψ̄_{m−2}·ψ̄_{m+1}²)
    /// ```
    fn division_polys(&self, l: usize) -> Vec<Poly<ModInt>> {
        let a = &self.a;
        let b = &self.b;
        let of = |k: i64| a.of(Int::from(k));
        let f = self.rhs_poly();
        let f2 = f.mul(&f);
        let mut psi: Vec<Poly<ModInt>> = Vec::with_capacity(l + 1);
        // ψ̄_0 = 0
        psi.push(Poly::zero());
        // ψ̄_1 = 1
        if l >= 1 {
            psi.push(Poly::constant(a.one()));
        }
        // ψ̄_2 = 2
        if l >= 2 {
            psi.push(Poly::constant(of(2)));
        }
        // ψ̄_3 = 3x⁴ + 6a·x² + 12b·x − a²
        if l >= 3 {
            let c0 = -(a.clone() * a.clone());
            let c1 = of(12) * b.clone();
            let c2 = of(6) * a.clone();
            let c3 = a.zero();
            let c4 = of(3);
            psi.push(Poly::new(alloc::vec![c0, c1, c2, c3, c4]));
        }
        // ψ̄_4 = 4(x⁶ + 5a·x⁴ + 20b·x³ − 5a²·x² − 4ab·x − a³ − 8b²)
        if l >= 4 {
            let a2 = a.clone() * a.clone();
            let a3 = a2.clone() * a.clone();
            let b2 = b.clone() * b.clone();
            let c0 = of(-4) * a3 + of(-32) * b2;
            let c1 = of(-16) * (a.clone() * b.clone());
            let c2 = of(-20) * a2;
            let c3 = of(80) * b.clone();
            let c4 = of(20) * a.clone();
            let c5 = a.zero();
            let c6 = of(4);
            psi.push(Poly::new(alloc::vec![c0, c1, c2, c3, c4, c5, c6]));
        }
        let two_inv = of(2).inv().expect("2 is invertible for p > 2");
        for n in 5..=l {
            let poly_n = if !n.is_multiple_of(2) {
                let m = (n - 1) / 2;
                let psi_m3 = psi[m].mul(&psi[m]).mul(&psi[m]);
                let psi_mp1_3 = psi[m + 1].mul(&psi[m + 1]).mul(&psi[m + 1]);
                let t1 = psi[m + 2].mul(&psi_m3);
                let t2 = psi[m - 1].mul(&psi_mp1_3);
                if m.is_multiple_of(2) {
                    f2.mul(&t1).sub(&t2)
                } else {
                    t1.sub(&f2.mul(&t2))
                }
            } else {
                let m = n / 2;
                let inner = psi[m + 2]
                    .mul(&psi[m - 1].mul(&psi[m - 1]))
                    .sub(&psi[m - 2].mul(&psi[m + 1].mul(&psi[m + 1])));
                psi[m].scalar_mul(&two_inv).mul(&inner)
            };
            psi.push(poly_n);
        }
        psi
    }
}
