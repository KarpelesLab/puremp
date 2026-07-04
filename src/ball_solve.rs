//! Rigorous root solving on top of [`Ball`] arithmetic.
//!
//! Given a continuous function `f` implemented in ball arithmetic (so that
//! `f(x)` returns a [`Ball`] *guaranteed* to enclose the true value of the
//! function over the input ball), [`bisect_root`] returns a `Ball` that provably
//! contains a real root of `f` inside a starting bracket `[a, b]`.
//!
//! The rigor comes from two facts:
//!
//! * **Certified sign change.** A root is only reported after `f(a)` and `f(b)`
//!   are shown to have *certain* opposite signs — one ball lies strictly below
//!   zero (its upper endpoint is `< 0`) and the other strictly above (its lower
//!   endpoint is `> 0`). Because these enclosures are rigorous, the true `f`
//!   really does change sign across `[a, b]`, so by the intermediate value
//!   theorem a true root exists in the bracket.
//! * **Bracket-preserving bisection.** At each step `f` is evaluated at the
//!   midpoint as a point ball. If that ball is *certainly* positive or
//!   *certainly* negative we keep the half whose endpoints still straddle the
//!   sign change — the surviving bracket still encloses a true root. If the
//!   midpoint ball straddles zero we cannot decide the sign at this precision, so
//!   we stop and return the current (still rigorous) bracket.
//!
//! Reference: R. E. Moore, *Interval Analysis* (1966); W. Tucker, *Validated
//! Numerics* (2011), §5 (bisection and the interval Newton method).

use crate::ball::Ball;
use crate::float::{Float, RoundingMode};
use crate::int::Sign;
use crate::interval::Interval;

const NEAR: RoundingMode = RoundingMode::Nearest;

/// Whether a ball is *certainly* strictly negative (its whole enclosure is `< 0`).
fn certainly_negative(b: &Ball) -> bool {
    b.upper().sign() == Sign::Negative
}

/// Whether a ball is *certainly* strictly positive (its whole enclosure is `> 0`).
fn certainly_positive(b: &Ball) -> bool {
    b.lower().sign() == Sign::Positive
}

/// Rigorously isolates a root of a continuous `f` in the bracket `[a, b]`.
///
/// `f` must be evaluable in ball arithmetic: for any input ball `x`, `f(x)` must
/// return a `Ball` that encloses `{ f(t) : t ∈ x }`. `precision` is the working
/// precision (bits) for the bisection midpoints; `max_iters` caps the number of
/// bisection steps.
///
/// Returns `Some(ball)` where `ball` provably contains a true root of `f`, or
/// `None` if the endpoints do **not** exhibit a certified sign change (so no root
/// can be guaranteed by this method). The returned ball is refined by bisection
/// until `max_iters` steps are taken or the midpoint sign can no longer be
/// decided at `precision` (a "straddle", at which point the current bracket is
/// returned unchanged).
pub fn bisect_root<F: Fn(&Ball) -> Ball>(
    f: F,
    a: &Float,
    b: &Float,
    precision: u64,
    max_iters: usize,
) -> Option<Ball> {
    let fa = f(&Ball::point(a.clone()));
    let fb = f(&Ball::point(b.clone()));

    // Certify a sign change: one endpoint strictly negative, the other strictly
    // positive. Anything else (including a ball that straddles zero) is refused.
    let lo_negative = if certainly_negative(&fa) && certainly_positive(&fb) {
        true
    } else if certainly_positive(&fa) && certainly_negative(&fb) {
        false
    } else {
        return None;
    };

    let mut lo = a.clone();
    let mut hi = b.clone();
    let half = Float::from_f64(0.5, precision, NEAR);

    for _ in 0..max_iters {
        // Any evaluation point strictly inside the bracket works for rigor; the
        // arithmetic midpoint keeps the bracket shrinking geometrically.
        let mid = lo.add(&hi, precision, NEAR).mul(&half, precision, NEAR);
        let fm = f(&Ball::point(mid.clone()));

        if certainly_negative(&fm) {
            // Midpoint has the sign of the negative endpoint; move that endpoint in.
            if lo_negative {
                lo = mid;
            } else {
                hi = mid;
            }
        } else if certainly_positive(&fm) {
            // Midpoint has the sign of the positive endpoint; move it in.
            if lo_negative {
                hi = mid;
            } else {
                lo = mid;
            }
        } else {
            // Sign undecidable at this precision: the current bracket still
            // encloses a root, so stop and return it.
            break;
        }
    }

    // Rebuild the bracket [lo, hi] as the tightest enclosing ball; it provably
    // contains a true root.
    Some(Ball::from_interval(
        &Interval::new(lo, hi, precision),
        precision,
    ))
}
