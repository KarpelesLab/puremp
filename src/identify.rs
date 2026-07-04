//! Experimental-mathematics helpers: an inverse symbolic calculator and
//! Machin-like formula discovery, both built on the PSLQ integer-relation
//! algorithm ([`crate::lattice::pslq`]).
//!
//! The central idea is that many numeric constants that arise in a computation
//! are *closed forms* — small rational combinations of well-known constants such
//! as `π`, `ln 2`, `√2` or `ζ(3)`. Given such a number to high precision,
//! [`identify`] searches a basis of named constants for an integer relation
//!
//! ```text
//! a₀·x + a₁·c₁ + … + aₖ·cₖ ≈ 0     (a₀ ≠ 0)
//! ```
//!
//! and, when one is found with suitably small coefficients, reports the implied
//! closed form `x = −(Σ aᵢ cᵢ)/a₀` as a legible [`Identification`].
//!
//! [`machin_like`] is the same machinery specialised to arctangent identities:
//! given denominators `[n₁, …, nₘ]` it looks for the integer relation tying
//! `π/4` to `atan(1/n₁), …, atan(1/nₘ)`, recovering Machin-style formulas such
//! as `π/4 = 4·atan(1/5) − atan(1/239)`.
//!
//! # Provenance
//!
//! This is a clean-room implementation drawn from the open experimental-math
//! literature (Ferguson, Bailey & Arno, *Analysis of the PSLQ Integer Relation
//! Algorithm*, Math. Comp. 68 (1999); Bailey & Borwein's inverse-symbolic work),
//! never from the source of any other library.
//!
//! # Caveats
//!
//! A recovered relation is only *evidence* of a closed form, never a proof: PSLQ
//! detects relations to the working precision, so a genuine identity and a very
//! close coincidence are indistinguishable below the noise floor. Use a generous
//! `precision` (a few hundred bits), keep the basis small, and confirm any hit
//! independently. Conversely, `None` only means no small relation was certifiable
//! at the requested precision — a relation may still exist with larger
//! coefficients or in a richer basis.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::float::{Float, RoundingMode};
use crate::int::Int;
use crate::lattice::pslq;

/// A closed form recovered by [`identify`]: the number `x` is expressed as
/// `x = (Σ coeffᵢ·cᵢ) / x_coeff`, where each `cᵢ` is a named basis constant.
///
/// The [`Display`](fmt::Display) implementation renders the closed form legibly,
/// e.g. `π²/6`, `2·ln2 − 1`, `3/4` or `√2`. The raw data is available through
/// [`Identification::x_coeff`] (the always-positive coefficient `a₀` of `x`,
/// which becomes the denominator) and [`Identification::terms`] (the non-zero
/// numerator terms `(coeffᵢ, name)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identification {
    // a₀: the coefficient of x in the relation a₀·x + Σ aᵢ·cᵢ = 0. PSLQ
    // sign-normalizes its first non-zero entry positive and x is that first
    // entry, so a₀ ≥ 1 whenever it is non-zero. It is the denominator of the
    // closed form.
    x_coeff: Int,
    // The numerator terms (coeffᵢ = −aᵢ, name), excluding zero coefficients,
    // ordered with the named constants first and the pure rational term ("1")
    // last for readability.
    terms: Vec<(Int, String)>,
}

impl Identification {
    /// The coefficient `a₀` of `x` in the detected relation — equivalently the
    /// denominator of the closed form `x = (Σ coeffᵢ·cᵢ)/a₀`. Always ≥ 1.
    pub fn x_coeff(&self) -> &Int {
        &self.x_coeff
    }

    /// The non-zero numerator terms `(coeffᵢ, name)` of the closed form
    /// `x = (Σ coeffᵢ·cᵢ)/x_coeff`.
    pub fn terms(&self) -> &[(Int, String)] {
        &self.terms
    }
}

/// Renders one numerator term `coeff·name` into `out`.
///
/// `leading` selects whether this is the first term (a bare `−` prefix for a
/// negative coefficient) or a following one (a ` + `/` − ` joiner). The pure
/// rational constant `"1"` renders as just its integer coefficient.
fn write_term(out: &mut String, coeff: &Int, name: &str, leading: bool) {
    use core::fmt::Write as _;
    let neg = coeff.is_negative();
    if leading {
        if neg {
            out.push('\u{2212}'); // − (minus sign)
        }
    } else if neg {
        out.push_str(" \u{2212} ");
    } else {
        out.push_str(" + ");
    }
    let mag = coeff.abs();
    if name == "1" {
        let _ = write!(out, "{mag}");
    } else if mag.is_one() {
        out.push_str(name);
    } else {
        let _ = write!(out, "{mag}\u{b7}{name}"); // · (middle dot)
    }
}

impl fmt::Display for Identification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return f.write_str("0");
        }
        let mut num = String::new();
        for (i, (coeff, name)) in self.terms.iter().enumerate() {
            write_term(&mut num, coeff, name, i == 0);
        }
        if self.x_coeff.is_one() {
            f.write_str(&num)
        } else if self.terms.len() > 1 {
            write!(f, "({num})/{}", self.x_coeff)
        } else {
            write!(f, "{num}/{}", self.x_coeff)
        }
    }
}

/// The default basis of named constants used by [`identify`], each computed to
/// `precision` bits: `1, π, π², e, ln 2, γ, G, ζ(3), √2, √3, √5`
/// (`γ` = Euler–Mascheroni, `G` = Catalan).
fn default_basis(precision: u64) -> Vec<(&'static str, Float)> {
    let m = RoundingMode::Nearest;
    let one = Float::from_int(&Int::ONE, precision, m);
    let pi = Float::pi(precision, m);
    let pi2 = pi.mul(&pi, precision, m);
    let e = Float::e(precision, m);
    let ln2 = Float::ln2(precision, m);
    let gamma = Float::euler_gamma(precision, m);
    let catalan = Float::catalan(precision, m);
    let zeta3 = Float::from_int(&Int::from_i64(3), precision, m).zeta(precision, m);
    let sqrt = |k: i64| Float::from_int(&Int::from_i64(k), precision, m).sqrt(precision, m);
    alloc::vec![
        ("1", one),
        ("π", pi),
        ("π²", pi2),
        ("e", e),
        ("ln2", ln2),
        ("γ", gamma),
        ("G", catalan),
        ("ζ(3)", zeta3),
        ("√2", sqrt(2)),
        ("√3", sqrt(3)),
        ("√5", sqrt(5)),
    ]
}

/// Attempts to recognize `x` (a real number computed to `precision` bits) as a
/// small rational combination of the default basis of constants
/// `1, π, π², e, ln 2, γ, G, ζ(3), √2, √3, √5`, returning the closed form as an
/// [`Identification`] or `None` if no small relation is certifiable.
///
/// See [`identify_with`] for a custom basis and the module documentation for the
/// caveats around false positives and precision.
///
/// # Examples
///
/// ```
/// use puremp::{Float, RoundingMode};
/// use puremp::identify::identify;
///
/// // ζ(2) = π²/6.
/// let m = RoundingMode::Nearest;
/// let prec = 400;
/// let zeta2 = Float::from_int(&2i64.into(), prec, m).zeta(prec, m);
/// let id = identify(&zeta2, prec).unwrap();
/// assert_eq!(id.to_string(), "π²/6");
/// ```
pub fn identify(x: &Float, precision: u64) -> Option<Identification> {
    identify_with(x, precision, &default_basis(precision))
}

/// Like [`identify`], but searches a caller-supplied `basis` of named constants
/// (each a `(name, Float)` computed to `precision` bits) instead of the default.
///
/// The basis constants should be linearly independent over the rationals for the
/// result to be meaningful; a relation not involving `x` (i.e. `a₀ = 0`) is
/// rejected as `None`.
pub fn identify_with(x: &Float, precision: u64, basis: &[(&str, Float)]) -> Option<Identification> {
    let mut xs = Vec::with_capacity(basis.len() + 1);
    xs.push(x.clone());
    for (_, c) in basis {
        xs.push(c.clone());
    }
    let rel = pslq(&xs, precision)?;

    // rel[0] is the coefficient of x. Without it we cannot solve for x.
    let x_coeff = rel[0].clone();
    if x_coeff.is_zero() {
        return None;
    }

    // x = −(Σ aᵢ·cᵢ)/a₀ = (Σ (−aᵢ)·cᵢ)/a₀. Collect the non-zero numerator terms,
    // named constants first and the pure "1" term last for readability.
    let mut named = Vec::new();
    let mut ones = Vec::new();
    for (i, (name, _)) in basis.iter().enumerate() {
        let ai = &rel[i + 1];
        if ai.is_zero() {
            continue;
        }
        let entry = (ai.neg(), String::from(*name));
        if *name == "1" {
            ones.push(entry);
        } else {
            named.push(entry);
        }
    }
    named.extend(ones);

    Some(Identification {
        x_coeff,
        terms: named,
    })
}

/// Searches for a Machin-like arctangent identity relating `π/4` to the
/// arctangents `atan(1/nᵢ)` of the reciprocals of the given `denominators`.
///
/// Computes `[π/4, atan(1/n₁), …, atan(1/nₘ)]` to `precision` bits and runs PSLQ,
/// returning the integer relation `[a₀, a₁, …, aₘ]` with
/// `a₀·(π/4) + Σ aᵢ·atan(1/nᵢ) = 0` (sign-normalized so the first non-zero entry
/// is positive), or `None` if no small relation is found. Every denominator must
/// be ≥ 2; otherwise `None` is returned.
///
/// # Examples
///
/// ```
/// use puremp::identify::machin_like;
///
/// // Machin's formula: π/4 = 4·atan(1/5) − atan(1/239).
/// let rel = machin_like(&[5, 239], 400).unwrap();
/// let want = [1i64, -4, 1].map(Into::into);
/// assert_eq!(rel, want);
/// ```
pub fn machin_like(denominators: &[i64], precision: u64) -> Option<Vec<Int>> {
    let m = RoundingMode::Nearest;
    let one = Float::from_int(&Int::ONE, precision, m);

    let mut xs = Vec::with_capacity(denominators.len() + 1);
    // π/4.
    let four = Float::from_int(&Int::from_i64(4), precision, m);
    xs.push(Float::pi(precision, m).div(&four, precision, m));

    for &n in denominators {
        if n < 2 {
            return None;
        }
        let nf = Float::from_int(&Int::from_i64(n), precision, m);
        let recip = one.div(&nf, precision, m);
        xs.push(recip.atan(precision, m));
    }

    pslq(&xs, precision)
}
