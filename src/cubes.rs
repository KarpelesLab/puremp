//! Rational point search, and sums of two rational cubes.
//!
//! # Point search on `y² = f(x)`
//!
//! [`search_quartic_points`] finds the coprime integer solutions of
//! `w² = g(u, v)` for a binary quartic form `g` (the homogenised `y² = f(x)`
//! with `x = u/v`, `y = w/v²`; a cubic `f` is covered by `g(u, v) = v·F(u, v)`
//! with `F` the homogenised cubic) in a box `|u|, v ≤ B`.
//! [`EllipticCurve::search_points`] does the same for a Weierstrass curve
//! `y² = x³ + a·x + b`: on an integral model every rational point has the shape
//! `(u/w², s/w³)` with `gcd(u, w) = 1`, so only square denominators are tried,
//! and every `x` with naive height `max(|u|, w²) ≤ H` is examined in `O(H^{3/2})`
//! steps.
//!
//! Both searches sieve before the exact square-root test: for each denominator
//! the value of the form modulo `64` and modulo the primes `5 … 61` is tabulated
//! as a function of `u` (one bit per residue), and only the `u` whose residues
//! are all squares survive — roughly one in `2¹⁶`. Survivors are tested exactly
//! (in `i128` when the box makes overflow impossible, otherwise with [`Int`]).
//!
//! # Sums of two rational cubes
//!
//! For a cube-free integer `n > 2` the projective cubic
//! `C_n : x³ + y³ = n·z³` is a genus-one curve with the rational point
//! `(1 : −1 : 0)`, and
//!
//! ```text
//! C_n → E_n : Y² = X³ − 432·n²,     X = 12n / (x + y),    Y = 36n·(x − y) / (x + y),
//! E_n → C_n :                       x = (36n + Y) / (6X), y = (36n − Y) / (6X)
//! ```
//!
//! is an isomorphism sending `(1 : −1 : 0)` to the point at infinity `O`
//! (Silverman–Tate, *Rational Points on Elliptic Curves*, §I.3; Cassels,
//! *Lectures on Elliptic Curves*, §18). To check: with `s = x + y`, `d = x − y`,
//! `n = x³ + y³ = s·(s² + 3d²)/4`, so `3d² = 4n/s − s²` and
//! `Y² = 1296·n²·d²/s² = 432·n²·(4n/s − s²)/s² = 1728·n³/s³ − 432·n² = X³ − 432·n²`;
//! conversely `x + y = 12n/X` and `x − y = Y/(3X)`. `X = 0` is impossible
//! (`Y² = −432·n²`), and `x + y = 0` forces `n = 0`, so for `n ≠ 0` affine
//! points correspond to affine points. The torsion of `Y² = X³ + k` is classical
//! (Fueter; it also follows from Nagell–Lutz): for `k = −432·n²` with `n`
//! cube-free it is `ℤ/3` (`(12, ±36)`, i.e. `1 = 1³ + 0³ = 0³ + 1³`) for
//! `n = 1`, `ℤ/2` (`(12, 0)`, i.e. `2 = 1³ + 1³`) for `n = 2`, and trivial for
//! `n > 2`. So for cube-free `n > 2`, `n` is a sum of two rational cubes iff
//! `rank E_n(ℚ) > 0`, and every non-trivial solution has infinite order.
//!
//! ## The 3-isogeny and diagonal cubic coverings
//!
//! `E'_n : Y² = X³ + 16·n²` carries the rational 3-torsion points `(0, ±4n)`,
//! and with `K = 16n²` the map
//!
//! ```text
//! φ : E'_n → E_n,   (X, Y) ↦ ( (X³ + 4K)/X²,  Y·(X³ − 8K)/X³ )
//! ```
//!
//! is a 3-isogeny with kernel `{O, (0, ±4n)}` onto `Y² = X³ − 27K = X³ − 432n²`.
//! (Expanding with `t = X³`: `(t + K)(t − 8K)² = t³ − 15Kt² + 48K²t + 64K³ =
//! (t + 4K)³ − 27K·t²`, which is exactly the identity `Y'² = X'³ − 27K`.)
//!
//! For positive integers `a, b, c` with `a·b·c = n·k³`, the diagonal cubic
//! `D_{a,b,c} : a·x³ + b·y³ = c·z³` maps to `E'_n` by
//!
//! ```text
//! X = −4ab·x·y / (z²·k²),     Y = −4ab·(a·x³ − b·y³) / (z³·k³).
//! ```
//!
//! (Derivation: `U = a·x³`, `V = b·y³`, `T = x·y·z` satisfy
//! `U·V·(U + V) = abc·T³`; with `p = U + V`, `q = U·V` the discriminant
//! `(U − V)² = p² − 4q = p² − 4·abc·T³/p` is a square, and multiplying through by
//! `16·(abc)²/p²` turns it into `Y² = X³ + 16·(abc)²` with
//! `X = −4abc·T/p`, `Y = −4abc·(U − V)/p`; using `p = c·z³` and scaling by `k`
//! lands on `E'_n`.) These are the classical coverings of the 3-isogeny descent
//! (Selmer, *The diophantine equation ax³ + by³ + cz³ = 0*, Acta Math. 85 (1951);
//! Cassels, *Lectures*, §18): every rational point of `E'_n` comes from some
//! `D_{a,b,c}` where `a, b, c` are cube-free products of primes dividing `3n`,
//! taken up to permutation and the scaling `(a, b, c) ↦ (e²a, e²b, e²c)` modulo
//! cubes. A point of `E_n(ℚ) = C_n(ℚ)` lying in `φ(E'_n(ℚ))` is three times
//! "smaller" (in canonical height) on `E'_n`, and nine times smaller on the
//! covering where it is found, than on `C_n` itself — so searching all
//! coverings finds many representations whose coordinates on `C_n` are far
//! beyond a direct search. `D_{1,1,n}` is `C_n` itself, and its points are used
//! directly.
//!
//! For a prime `ℓ ≠ 3` the only coverings are `C_ℓ` itself and `D_{1,3,9ℓ}`,
//! and the latter has no 3-adic point (`x³ + 3y³ ≡ 0 (mod 9)` forces
//! `3 | x, y`, then `3 | z`), so for primes these coverings add nothing; they
//! do help for composite `n` (e.g. `6`: `1 + 2·1³ = 3·1³` on `D_{1,2,3}`).
//!
//! ## The other half: descent over the Eisenstein integers
//!
//! The coverings of `E_n` for `φ` itself become explicit over `ℤ[ω]`
//! (`ω² + ω + 1 = 0`), which is a principal ideal domain with units `±ωⁱ`. Let
//! `x³ + y³ = n·z³` with `gcd(x, y) = 1`, `z ≠ 0`, and put `p = x + y`,
//! `N = x² − x·y + y² = N(x + y·ω)`. Since `N = p² − 3xy`,
//! `gcd(p, N) | 3` and `v₃(N) ≤ 1`. In `ℤ[ω]`, `α = x + y·ω` is not divisible
//! by any rational integer, so an inert prime never divides it and of a split
//! prime `q = π·π̄` at most one of `π, π̄` does; every prime `q ∤ 3n` occurs in
//! `N` to a power divisible by 3 (it divides `N` but not `p`). Hence
//!
//! ```text
//! x + y·ω = ε · λ^e · ρ · (u + v·ω)³,
//! ```
//!
//! with `ε ∈ {1, ω, ω²}` (`−1` is a cube), `λ = 1 − ω` (norm 3), `e ∈ {0, 1}`,
//! and `ρ` a product over the split primes `q | n` (exponent `a_q`) of `1`,
//! `π_q^{a_q}` or `π̄_q^{a_q}`. Writing `γ = ε·λ^e·ρ = g₀ + g₁·ω` and
//! `(u + v·ω)³ = A + B·ω` with `A = u³ − 3uv² + v³`, `B = 3u²v − 3uv²`,
//!
//! ```text
//! x = g₀·A − g₁·B,      y = g₀·B + g₁·(A − B),
//! ```
//!
//! and since `N = N(γ)·N(u + v·ω)³` the remaining condition is that
//! `p·N(γ)/n` is a cube, i.e. `x + y = d·s³` for the cube-free `d` with
//! `d·N(γ)/n ∈ ℚ*³` — a cubic curve `F_γ(u, v) = d·s³`, one per twist `γ`
//! (this is Selmer's and Cassels' treatment of `x³ + y³ = n·z³` by "first
//! descent" over `ℚ(√−3)`; Cassels, *Lectures*, §18; Selmer (1951)). The point
//! is the size: `x, y ≈ |γ|·max(|u|, |v|)³`, so a box `|u|, |v| ≤ B` reaches
//! solutions with numerators near `B³` in `O(B²)` steps. Complex conjugation
//! followed by multiplication by `ω` swaps `x` and `y` and maps `γ` to `ω·γ̄`,
//! so only the `ρ` whose first non-trivial factor is `π` (not `π̄`) are used. A prime
//! `q = s² + 3t²` (`q ≡ 1 mod 3`) splits as `π = (s + t) + 2t·ω`, with
//! `(s, t)` from Cornacchia's algorithm (Cohen, *A Course in Computational
//! Algebraic Number Theory*, Alg. 1.5.2).
//!
//! ## The search
//!
//! [`sum_of_two_cubes`] deepens a box bound `H = 4, 8, 16, …` up to `effort`.
//! At each level it searches every covering `D_{a,b,c}` (sorted `a ≤ b ≤ c`) over
//! coprime `(x, y)` with `|x| ≤ H/a^{1/3}`, `0 < y ≤ H/b^{1/3}` in the residue
//! classes with `a·x³ + b·y³ ≡ 0 (mod c)` (a cube-residue table modulo `c`, or
//! its largest factor below `2²⁰`), then every Eisenstein twist over
//! `|u|, |v| ≤ H`. Cube candidates are filtered by cubic residuosity modulo
//! `7·9·13·19·37` before an exact integer cube root. Every candidate is mapped to
//! `x³ + y³ = n` and checked exactly before it is returned.
//!
//! This is a **search, not a decision procedure**: `None` means nothing was found
//! within the effort bound, not that `n` is not a sum of two cubes. For primes
//! `ℓ ≡ 4, 7, 8 (mod 9)` a solution always exists: OpenAI, *The Selmer converse
//! for elliptic curves at every prime* (preprint, September 24, 2026; **not
//! refereed**), §10, proves that `x³ + y³ = ℓ·z³` has rank one (Sylvester's
//! conjecture for these classes). With `effort = 2000` the search finds a
//! solution for every such prime below 557 (the largest, for `ℓ = 859`, has
//! 12-digit numerators); the first misses are `557, 571, 647, 661, 773, 809,
//! 881, 967` below 1000, of which `557, 773, 881` fall at `effort = 8000`.
//!
//! # Clean-room provenance
//!
//! Silverman–Tate §I.3, §III; Cassels, *Lectures on Elliptic Curves* §14, §18;
//! Selmer (1951); Cohen §1.5 (Cornacchia); Mazur's torsion theorem (Silverman, *The Arithmetic of
//! Elliptic Curves*, Thm. VIII.7.5) and Nagell–Lutz (*ibid.* Cor. VIII.7.2) for
//! the torsion test; the sieve is the textbook quadratic/cubic residue filter. No
//! third-party source code was consulted.

use alloc::vec;
use alloc::vec::Vec;

use crate::elliptic::{EllipticCurve, Point};
use crate::int::Int;
use crate::rational::Rational;

// ---------------------------------------------------------------------------
// Residue masks.
// ---------------------------------------------------------------------------

/// Moduli of the square sieve (each `≤ 64`, so a residue set fits a `u64`).
const SQ_MODULI: [u64; 17] = [
    64, 9, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 61,
];

const fn power_mask(m: u64, e: u32) -> u64 {
    let mut mask = 0u64;
    let mut r = 0u64;
    while r < m {
        let mut v = 1u64;
        let mut i = 0;
        while i < e {
            v = v * r % m;
            i += 1;
        }
        mask |= 1u64 << v;
        r += 1;
    }
    mask
}

const fn square_masks() -> [u64; 17] {
    let mut out = [0u64; 17];
    let mut i = 0;
    while i < 17 {
        out[i] = power_mask(SQ_MODULI[i], 2);
        i += 1;
    }
    out
}

const SQ_MASKS: [u64; 17] = square_masks();

/// Cube-residue filter modulus `7·9·13·19·37` and its factors' residue masks.
const CUBE_MOD: i128 = 7 * 9 * 13 * 19 * 37;
const CUBE_MODULI: [u32; 5] = [7, 9, 13, 19, 37];
const CUBE_MASKS: [u64; 5] = [
    power_mask(7, 3),
    power_mask(9, 3),
    power_mask(13, 3),
    power_mask(19, 3),
    power_mask(37, 3),
];

#[inline]
fn passes_cube_filter(q: i128) -> bool {
    let r = q.rem_euclid(CUBE_MOD) as u32;
    let mut i = 0;
    while i < 5 {
        if (CUBE_MASKS[i] >> (r % CUBE_MODULI[i])) & 1 == 0 {
            return false;
        }
        i += 1;
    }
    true
}

/// Exact integer cube root of `q` (`|q| < 2¹²⁰`).
fn icbrt_exact(q: i128) -> Option<i128> {
    let m = q.unsigned_abs();
    let mut r = libm_cbrt(m as f64) as u128;
    while r > 0 && r * r * r > m {
        r -= 1;
    }
    while (r + 1) * (r + 1) * (r + 1) <= m {
        r += 1;
    }
    if r * r * r != m {
        return None;
    }
    let r = r as i128;
    Some(if q < 0 { -r } else { r })
}

/// A `no_std` cube-root estimate (Newton on `f64`, a few ulps is plenty: the
/// caller corrects to the exact integer root).
fn libm_cbrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Initial guess from the exponent: x = m·2^e  ⇒  ∛x ≈ 2^{e/3}.
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut y = f64::from_bits(((1023 + e / 3) as u64) << 52);
    for _ in 0..100 {
        let ny = (2.0 * y + x / (y * y)) / 3.0;
        if (ny - y).abs() <= 1e-15 * ny {
            return ny;
        }
        y = ny;
    }
    y
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn is_square_i128(v: i128) -> Option<i128> {
    if v < 0 {
        return None;
    }
    let m = v as u128;
    if (0x0202_0212_0203_0213u64 >> (m & 63)) & 1 == 0 {
        return None;
    }
    let r = m.isqrt();
    (r * r == m).then_some(r as i128)
}

// ---------------------------------------------------------------------------
// Square values of binary forms.
// ---------------------------------------------------------------------------

/// Finds `(u, v, w)` with `w² = F(u, v) = Σ cᵢ·uⁱ·v^{d−i}`, `|u| ≤ u_max`, `v`
/// from `vs`, `gcd(u, v) = 1`, `w ≥ 0`.
fn form_square_search(coeffs: &[Int], u_max: u64, vs: &[u64]) -> Vec<(i64, u64, Int)> {
    assert!((2..=5).contains(&coeffs.len()), "forms of degree 1 to 4");
    let d = coeffs.len() - 1;
    let mut out = Vec::new();
    let v_max = vs.iter().copied().max().unwrap_or(0);
    let big = u_max.max(v_max).max(1);
    // i128 path if Σ|cᵢ|·big^d < 2^125.
    let small: Option<Vec<i128>> = coeffs.iter().map(|c| c.to_i128()).collect();
    let fits = small.as_ref().is_some_and(|c| {
        let mut bd: u128 = 1;
        for _ in 0..d {
            match bd.checked_mul(u128::from(big)) {
                Some(x) => bd = x,
                None => return false,
            }
        }
        let s: Option<u128> = c.iter().try_fold(0u128, |acc, ci| {
            acc.checked_add(ci.unsigned_abs().checked_mul(bd)?)
        });
        s.is_some_and(|s| s < (1u128 << 125))
    });
    let coeff_res: Vec<[u64; 17]> = coeffs
        .iter()
        .map(|c| core::array::from_fn(|j| int_mod_u64(c, SQ_MODULI[j])))
        .collect();
    let u_max_i = u_max as i64;
    for &v in vs {
        if v == 0 {
            // Only (u, v) = (1, 0): F(1, 0) = c_d.
            if let Some(w) = coeffs[d].sqrt_exact() {
                out.push((1, 0, w));
            }
            continue;
        }
        // Residue tables: bit r of good[j] set iff F(r, v) mod m_j is a square.
        let good: [u64; 17] = core::array::from_fn(|j| {
            let m = SQ_MODULI[j];
            let vm = v % m;
            let mut vpow = [1u64; 5];
            for k in 1..=d {
                vpow[k] = vpow[k - 1] * vm % m;
            }
            let mut mask = 0u64;
            for r in 0..m {
                let mut acc = 0u64;
                let mut rp = 1u64;
                for (i, cr) in coeff_res.iter().enumerate() {
                    acc = (acc + cr[j] * rp % m * vpow[d - i]) % m;
                    rp = rp * r % m;
                }
                if (SQ_MASKS[j] >> acc) & 1 == 1 {
                    mask |= 1 << r;
                }
            }
            mask
        });
        let mut res: [u64; 17] =
            core::array::from_fn(|j| (-(u_max_i as i128)).rem_euclid(SQ_MODULI[j] as i128) as u64);
        let vi = v as i128;
        let v_int = Int::from(v);
        // terms[i] = cᵢ·v^{d−i} (i128 path only).
        let mut terms = [0i128; 5];
        if let Some(c) = small.as_ref().filter(|_| fits) {
            let mut vp: i128 = 1;
            for i in (0..=d).rev() {
                terms[i] = c[i] * vp;
                vp = vp.saturating_mul(vi);
            }
        }
        for u in -u_max_i..=u_max_i {
            let mut ok = true;
            for j in 0..17 {
                if (good[j] >> res[j]) & 1 == 0 {
                    ok = false;
                    break;
                }
            }
            if ok && gcd_u64(u.unsigned_abs(), v) == 1 {
                let root = if fits {
                    // Horner in u.
                    let ui = u as i128;
                    let mut acc: i128 = 0;
                    for t in terms[..=d].iter().rev() {
                        acc = acc * ui + t;
                    }
                    is_square_i128(acc).map(Int::from)
                } else {
                    let ui = Int::from(u);
                    let mut acc = Int::ZERO;
                    for i in (0..=d).rev() {
                        acc = acc.mul(&ui).add(&coeffs[i].mul(&v_int.pow((d - i) as u32)));
                    }
                    acc.sqrt_exact()
                };
                if let Some(w) = root {
                    out.push((u, v, w));
                }
            }
            for j in 0..17 {
                res[j] += 1;
                if res[j] == SQ_MODULI[j] {
                    res[j] = 0;
                }
            }
        }
    }
    out
}

fn int_mod_u64(c: &Int, m: u64) -> u64 {
    c.rem_euclid(&Int::from(m)).to_u64().expect("residue fits")
}

/// Searches the quartic `w² = g(u, v) = g₄u⁴ + g₃u³v + g₂u²v² + g₁uv³ + g₀v⁴`
/// (`g[i]` is the coefficient of `uⁱ·v^{4−i}`) for coprime integer solutions with
/// `|u| ≤ bound`, `0 ≤ v ≤ bound` (and `(u, v) = (1, 0)` when `g₄` is a square).
///
/// Each solution `(u, v, w)` has `w ≥ 0`; for `v ≠ 0` it is the rational point
/// `(x, y) = (u/v, w/v²)` of `y² = g(x, 1)`. A cubic `y² = f(x)` with integer
/// coefficients is covered by passing `g(u, v) = v·F(u, v)` (`g₄ = 0`), whose
/// solutions give `(x, y) = (u/v, w/v²)` as well. Runs in `O(bound²)` with a
/// quadratic-residue sieve in front of the exact square test.
pub fn search_quartic_points(g: &[Int; 5], bound: u64) -> Vec<(Int, Int, Int)> {
    let bound = bound.min(1 << 24);
    let vs: Vec<u64> = (0..=bound).collect();
    form_square_search(g, bound, &vs)
        .into_iter()
        .map(|(u, v, w)| (Int::from(u), Int::from(v), w))
        .collect()
}

// ---------------------------------------------------------------------------
// Weierstrass point search and torsion test.
// ---------------------------------------------------------------------------

/// `e` with `e⁴·a, e⁶·b ∈ ℤ`.
fn integral_scale(curve: &EllipticCurve<Rational>) -> Int {
    curve.a().denominator().lcm(curve.b().denominator())
}

impl EllipticCurve<Rational> {
    /// Searches for rational points of naive height at most `height_bound`.
    ///
    /// The curve is first scaled to an integral model `y² = x³ + A·x + B`
    /// (`x ↦ e²x`, `y ↦ e³y`, the identity when `a, b ∈ ℤ`); the height of
    /// `x = u/w²` on that model is `max(|u|, w²)`. Every such `x` is examined
    /// (with a quadratic-residue sieve) and each point found is returned once
    /// (with `y > 0`; its negative is implied). Torsion points are omitted, so
    /// the result is empty for a curve of rank 0 and, for positive rank, consists
    /// of points of infinite order sorted by increasing height. Cost
    /// `O(height_bound^{3/2})`; the bound is clamped to `2⁶²`.
    pub fn search_points(&self, height_bound: u64) -> Vec<Point<Rational>> {
        let h = height_bound.min(1 << 62);
        let e = integral_scale(self);
        let e2 = e.square();
        let e3 = e2.mul(&e);
        let a_int = mul_to_int(self.a(), &e2.square());
        let b_int = mul_to_int(self.b(), &e3.square());
        // F(u, v) = u³ + A·u·v² + B·v³, evaluated at v = w².
        let coeffs = [b_int, a_int, Int::ZERO, Int::ONE];
        let w_max = h.isqrt();
        let vs: Vec<u64> = (1..=w_max).map(|w| w * w).collect();
        let mut found: Vec<(u64, Point<Rational>)> = Vec::new();
        for (u, v, s) in form_square_search(&coeffs, h, &vs) {
            if s.is_zero() {
                continue; // 2-torsion
            }
            let w = v.isqrt();
            let wi = Int::from(w);
            let x = Rational::new(Int::from(u), Int::from(v).mul(&e2));
            let y = Rational::new(s, wi.pow(3).mul(&e3));
            let p = self
                .point(x, y)
                .expect("search produced a point on the curve");
            if w == 1 && p.is_torsion() {
                continue; // Nagell–Lutz: w > 1 is never torsion
            }
            found.push((u.unsigned_abs().max(v), p));
        }
        found.sort_by_key(|(ht, _)| *ht);
        found.into_iter().map(|(_, p)| p).collect()
    }
}

/// `r·k`, which must be an integer.
fn mul_to_int(r: &Rational, k: &Int) -> Int {
    r.numerator().mul(k).div_exact(r.denominator())
}

impl Point<Rational> {
    /// Returns `true` if the point has finite order.
    ///
    /// By Mazur's theorem a rational torsion point has order at most 12, and by
    /// Nagell–Lutz the multiples of a torsion point have integer coordinates on
    /// an integral model; the multiples `P, 2P, …, 12P` are walked until one is
    /// `O` (torsion) or non-integral (infinite order).
    pub fn is_torsion(&self) -> bool {
        let e = integral_scale(self.curve());
        let e2 = e.square();
        let e3 = e2.mul(&e);
        let mut q = self.clone();
        for _ in 0..12 {
            let Some((x, y)) = q.coordinates() else {
                return true;
            };
            let xi = x.clone() * Rational::from_integer(e2.clone());
            let yi = y.clone() * Rational::from_integer(e3.clone());
            if !xi.is_integer() || !yi.is_integer() {
                return false;
            }
            q = q.add(self);
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Cubic ↔ Mordell curve maps.
// ---------------------------------------------------------------------------

/// The Mordell curve `E_n : Y² = X³ − 432·n²` birational to `x³ + y³ = n`
/// (`None` for `n = 0`).
pub fn cube_sum_curve(n: &Int) -> Option<EllipticCurve<Rational>> {
    let k = n.square().mul(&Int::from(-432));
    EllipticCurve::new(Rational::from(0i64), Rational::from_integer(k))
}

/// Maps a solution of `x³ + y³ = n` to the point
/// `(12n/(x + y), 36n·(x − y)/(x + y))` of [`cube_sum_curve`]`(n)`.
///
/// Returns `None` if `n = 0` or `(x, y)` is not a solution. (For `n ≠ 0`,
/// `x + y ≠ 0` automatically.)
pub fn cube_sum_to_curve(n: &Int, x: &Rational, y: &Rational) -> Option<Point<Rational>> {
    let curve = cube_sum_curve(n)?;
    if x.pow(3) + y.pow(3) != Rational::from_integer(n.clone()) {
        return None;
    }
    let s = x.clone() + y.clone();
    let nn = Rational::from_integer(n.clone());
    let big_x = Rational::from(12i64) * nn.clone() / s.clone();
    let big_y = Rational::from(36i64) * nn * (x.clone() - y.clone()) / s;
    curve.point(big_x, big_y)
}

/// Maps a point of [`cube_sum_curve`]`(n)` back to `(x, y)` with
/// `x³ + y³ = n`: `x = (36n + Y)/(6X)`, `y = (36n − Y)/(6X)`.
///
/// Returns `None` for the point at infinity (which corresponds to the point
/// `(1 : −1 : 0)` at infinity of the cubic) or a point of a different curve.
pub fn curve_to_cube_sum(n: &Int, p: &Point<Rational>) -> Option<(Rational, Rational)> {
    let curve = cube_sum_curve(n)?;
    if p.curve() != &curve {
        return None;
    }
    let (big_x, big_y) = p.coordinates()?;
    let t = Rational::from_integer(n.mul(&Int::from(36)));
    let six_x = Rational::from(6i64) * big_x.clone();
    let x = (t.clone() + big_y.clone()) / six_x.clone();
    let y = (t - big_y.clone()) / six_x;
    Some((x, y))
}

// ---------------------------------------------------------------------------
// Diagonal cubics a·x³ + b·y³ = c·z³.
// ---------------------------------------------------------------------------

/// Largest cube-residue table modulus.
const TABLE_LIMIT: u64 = 1 << 20;

/// Searches `a·x³ + b·y³ = c·z³` for primitive solutions with `z ≠ 0`,
/// `x·y ≠ 0`, `|x| ≤ bx`, `0 ≤ y ≤ by` (`a, b, c > 0`); `modulus` divides `c`
/// and is used for the residue sieve. Calls `found` on each solution until it
/// returns `true`.
fn diagonal_search(
    a: i128,
    b: i128,
    c: i128,
    modulus: u64,
    bx: u64,
    by: u64,
    found: &mut dyn FnMut(i128, i128, i128) -> bool,
) -> bool {
    let m = modulus.max(1);
    // CSR table: x mod m grouped by a·x³ mod m.
    let am = (a.rem_euclid(m as i128)) as u64;
    let bm = (b.rem_euclid(m as i128)) as u64;
    let cube = |x: u64| -> u64 {
        let x = u128::from(x);
        let m = u128::from(m);
        (x * x % m * x % m) as u64
    };
    let mut start = vec![0u32; m as usize + 1];
    let keys: Vec<u32> = (0..m)
        .map(|x| ((u128::from(am) * u128::from(cube(x))) % u128::from(m)) as u32)
        .collect();
    for &k in &keys {
        start[k as usize + 1] += 1;
    }
    for i in 0..m as usize {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut entries = vec![0u32; m as usize];
    for (x, &k) in keys.iter().enumerate() {
        entries[fill[k as usize] as usize] = x as u32;
        fill[k as usize] += 1;
    }
    drop(keys);
    drop(fill);
    let bx_i = bx as i128;
    for y in 1..=by {
        // a·x³ ≡ −b·y³ (mod m)
        let target = ((u128::from(m) - (u128::from(bm) * u128::from(cube(y % m))) % u128::from(m))
            % u128::from(m)) as usize;
        let yi = y as i128;
        let by3 = b * yi * yi * yi;
        for &x0 in &entries[start[target] as usize..start[target + 1] as usize] {
            // first x ≥ −bx with x ≡ x0 (mod m)
            let off = (i128::from(x0) + bx_i).rem_euclid(m as i128);
            let mut x = -bx_i + off;
            while x <= bx_i {
                if x != 0 {
                    let v = a * x * x * x + by3;
                    if v % c == 0 {
                        let q = v / c;
                        if q != 0
                            && passes_cube_filter(q)
                            && gcd_u64(x.unsigned_abs() as u64, y) == 1
                            && let Some(z) = icbrt_exact(q)
                            && found(x, yi, z)
                        {
                            return true;
                        }
                    }
                }
                x += m as i128;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Sums of two cubes.
// ---------------------------------------------------------------------------

/// Splits `n > 0` into `(m, n')` with `n = m³·n'`, `n'` cube-free, and returns
/// the distinct primes of `n'` with their exponents (1 or 2).
fn cube_free_part(n: &Int) -> (Int, Int, Vec<(Int, u32)>) {
    let mut m = Int::ONE;
    let mut core = Int::ONE;
    let mut primes: Vec<(Int, u32)> = Vec::new();
    let factors = n.factorize();
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i].clone();
        let mut e = 0u32;
        while i < factors.len() && factors[i] == p {
            e += 1;
            i += 1;
        }
        m = m.mul(&p.pow(e / 3));
        if !e.is_multiple_of(3) {
            core = core.mul(&p.pow(e % 3));
            primes.push((p, e % 3));
        }
    }
    (m, core, primes)
}

/// A covering `a·x³ + b·y³ = c·z³` with `abc = n·k³`, stored as exponent
/// vectors (mod 3) over the primes of `3n`.
struct Covering {
    a: Int,
    b: Int,
    c: Int,
    k: Int,
    direct: bool,
}

fn coverings(n: &Int, primes: &[(Int, u32)], max_primes: usize) -> Vec<Covering> {
    // Primes of 3n with n's exponents.
    let mut ps: Vec<(Int, u32)> = primes.to_vec();
    if !ps.iter().any(|(p, _)| p == &Int::from(3)) {
        ps.push((Int::from(3), 0));
    }
    ps.sort();
    let r = ps.len().min(max_primes);
    let n_exp: Vec<u32> = ps.iter().map(|(_, e)| *e).collect();
    let mut seen: Vec<[Vec<u32>; 3]> = Vec::new();
    let mut out = Vec::new();
    let total = 3usize.pow(r as u32);
    for idx in 0..total {
        // β exponents (primes beyond r keep exponent 0).
        let mut beta = vec![0u32; ps.len()];
        let mut t = idx;
        for slot in beta.iter_mut().take(r) {
            *slot = (t % 3) as u32;
            t /= 3;
        }
        let one = vec![0u32; ps.len()];
        let c: Vec<u32> = (0..ps.len())
            .map(|i| (n_exp[i] + 2 * beta[i]) % 3)
            .collect();
        let trip = [one, beta, c];
        // Canonical form: rescale by the square of each member, sort, take min.
        let mut best: Option<[Vec<u32>; 3]> = None;
        for pivot in 0..3 {
            let mut t: [Vec<u32>; 3] = core::array::from_fn(|j| {
                (0..ps.len())
                    .map(|i| (trip[j][i] + 2 * trip[pivot][i]) % 3)
                    .collect()
            });
            t.sort();
            if best.as_ref().is_none_or(|b| &t < b) {
                best = Some(t);
            }
        }
        let best = best.expect("three pivots");
        if seen.contains(&best) {
            continue;
        }
        let val = |ev: &[u32]| -> Int {
            ps.iter()
                .zip(ev)
                .fold(Int::ONE, |acc, ((p, _), &e)| acc.mul(&p.pow(e)))
        };
        let mut coeffs: [Int; 3] = core::array::from_fn(|j| val(&best[j]));
        coeffs.sort();
        // k³ = abc / n.
        let k = (0..ps.len()).fold(Int::ONE, |acc, i| {
            let s = best[0][i] + best[1][i] + best[2][i];
            acc.mul(&ps[i].0.pow((s - n_exp[i]) / 3))
        });
        let [a, b, c] = coeffs;
        let direct = a.is_one() && b.is_one() && &c == n;
        seen.push(best);
        out.push(Covering { a, b, c, k, direct });
    }
    // Search the direct curve first, then by increasing abc.
    out.sort_by(|x, y| {
        y.direct
            .cmp(&x.direct)
            .then_with(|| x.a.mul(&x.b).mul(&x.c).cmp(&y.a.mul(&y.b).mul(&y.c)))
    });
    out
}

/// Maps a point of `a·x³ + b·y³ = c·z³` (`abc = n·k³`) to `x³ + y³ = n` through
/// `E'_n` and the 3-isogeny `φ` (see the module docs).
fn covering_to_cube_sum(
    n: &Int,
    cov: &Covering,
    x: &Int,
    y: &Int,
    z: &Int,
) -> Option<(Rational, Rational)> {
    if cov.direct {
        return Some((
            Rational::new(x.clone(), z.clone()),
            Rational::new(y.clone(), z.clone()),
        ));
    }
    let ab = cov.a.mul(&cov.b);
    let k = &cov.k;
    let x1 = Rational::new(
        ab.mul(x).mul(y).mul(&Int::from(-4)),
        z.square().mul(&k.square()),
    );
    let ax3 = cov.a.mul(&x.pow(3));
    let by3 = cov.b.mul(&y.pow(3));
    let y1 = Rational::new(
        ab.mul(&ax3.sub(&by3)).mul(&Int::from(-4)),
        z.pow(3).mul(&k.pow(3)),
    );
    if x1.is_zero() {
        return None; // 3-torsion of E'_n, killed by φ
    }
    let kk = Rational::from_integer(n.square().mul(&Int::from(16)));
    let x1_3 = x1.pow(3);
    let x2 = (x1_3.clone() + kk.clone() * Rational::from(4i64)) / x1.pow(2);
    let y2 = y1 * (x1_3.clone() - kk * Rational::from(8i64)) / x1_3;
    let t = Rational::from_integer(n.mul(&Int::from(36)));
    let six_x = Rational::from(6i64) * x2;
    Some(((t.clone() + y2.clone()) / six_x.clone(), (t - y2) / six_x))
}

/// Finds rational `x, y` with `x³ + y³ = n`, by bounded search.
///
/// - `n = 0` gives `(0, 0)`; negative `n` is reduced to `−n` by negating both
///   cubes; `n = m³·n'` with `n'` cube-free is solved for `n'` and scaled by `m`.
/// - Cube-free `n' = 1` and `n' = 2` only have the trivial solutions
///   (`1 = 1³ + 0³`, `2 = 1³ + 1³`; the Mordell curve has rank 0), which are
///   returned directly.
/// - Otherwise both halves of the 3-isogeny descent are searched with iterative
///   deepening up to the box bound `effort`: the diagonal cubic coverings
///   `a·x³ + b·y³ = c·z³` (starting with `x³ + y³ = n'·z³` itself, where
///   `x = X/Z`, `y = Y/Z` with `|X|, |Y| ≤ effort`), and the Eisenstein twists
///   `x + y·ω = γ·(u + v·ω)³` with `|u|, |v| ≤ effort`, which reach numerators
///   of size about `effort³`; see the module documentation. The last level costs
///   about `2·effort²` cheap steps per twist (`6·(3^s + 1)/2` twists, `s` the
///   number of primes `≡ 1 mod 3` dividing `n'`); `effort = 2000`
///   takes about a second for a prime in an optimised build. `effort` is clamped
///   to `2²⁰`.
///
/// The returned pair always satisfies `x³ + y³ = n` exactly (it is checked).
/// `None` only means that no solution was found within `effort`: it is **not** a
/// proof that `n` is not a sum of two rational cubes (that needs a descent; e.g.
/// `3, 4, 5` are not, by Euler/Sylvester/Selmer, while some `n` have only very
/// large solutions).
pub fn sum_of_two_cubes(n: &Int, effort: u64) -> Option<(Rational, Rational)> {
    if n.is_zero() {
        return Some((Rational::from(0i64), Rational::from(0i64)));
    }
    let neg = n.is_negative();
    let (m, core, primes) = cube_free_part(&n.abs());
    let (x, y) = if core.is_one() {
        (Rational::from(1i64), Rational::from(0i64))
    } else if core == Int::from(2) {
        (Rational::from(1i64), Rational::from(1i64))
    } else {
        search_cube_free(&core, &primes, effort.min(1 << 20))?
    };
    let s = Rational::from_integer(if neg { m.neg() } else { m });
    let (x, y) = (x * s.clone(), y * s);
    debug_assert!(x.pow(3) + y.pow(3) == Rational::from_integer(n.clone()));
    Some((x, y))
}

// ---------------------------------------------------------------------------
// Descent over the Eisenstein integers.
// ---------------------------------------------------------------------------

/// `(a + b·ω)(c + d·ω)` in `ℤ[ω]`, `ω² = −1 − ω`.
fn eis_mul(p: (Int, Int), q: (&Int, &Int)) -> (Int, Int) {
    let (a, b) = p;
    let (c, d) = q;
    let bd = b.mul(d);
    (a.mul(c).sub(&bd), a.mul(d).add(&b.mul(c)).sub(&bd))
}

/// `(a, b)` with `a² − a·b + b² = q`, i.e. a prime `π = a + b·ω` of norm `q`,
/// for a prime `q ≡ 1 (mod 3)` (Cornacchia on `q = s² + 3t²`, then
/// `π = (s + t) + 2t·ω`).
fn eisenstein_prime(q: &Int) -> Option<(Int, Int)> {
    let r = Int::from(-3).rem_euclid(q).sqrt_mod(q)?;
    for r0 in [r.clone(), q.sub(&r)] {
        let (mut a, mut b) = (q.clone(), r0);
        while b.square() > *q {
            let t = a.rem_euclid(&b);
            a = b;
            b = t;
        }
        let rest = q.sub(&b.square());
        if rest.is_negative() {
            continue;
        }
        let Some((t2, rem)) = rest.div_rem(&Int::from(3)) else {
            continue;
        };
        if !rem.is_zero() {
            continue;
        }
        if let Some(t) = t2.sqrt_exact() {
            return Some((b.add(&t), t.mul(&Int::from(2))));
        }
    }
    None
}

/// One twist `γ = ε·λ^e·ρ` of the Eisenstein descent with the cube class
/// `d` that `x + y` must lie in (`x + y = d·s³`).
struct EisTwist {
    g0: i128,
    g1: i128,
    d: i128,
}

/// The twists `γ` for cube-free `n` (see [`eisenstein_search`]).
fn eisenstein_twists(primes: &[(Int, u32)], max_primes: usize) -> Vec<EisTwist> {
    // ρ: for each split prime q | n (q ≡ 1 mod 3) with exponent a, one of
    // {1, π^a, π̄^a}; tracked with its norm exponent.
    let mut rhos: Vec<((Int, Int), Vec<u32>)> =
        vec![((Int::ONE, Int::ZERO), vec![0; primes.len()])];
    let three = Int::from(3);
    for (i, (q, a)) in primes.iter().enumerate().take(max_primes) {
        if q.rem_euclid(&three) != Int::ONE {
            continue;
        }
        let Some((pa, pb)) = eisenstein_prime(q) else {
            continue;
        };
        // π̄ = (a − b) − b·ω.
        let conj = (pa.sub(&pb), pb.neg());
        let mut next = Vec::new();
        for (rho, ex) in &rhos {
            next.push((rho.clone(), ex.clone()));
            // Complex conjugation composed with ω swaps x and y
            // (ω·(x + y·ω̄) = y + x·ω) and maps the twist γ to ω·γ̄; since every
            // unit is tried, the first non-trivial ρ factor may be taken to be
            // π rather than π̄.
            let first = ex.iter().all(|&e| e == 0);
            let bases = if first {
                vec![(pa.clone(), pb.clone())]
            } else {
                vec![(pa.clone(), pb.clone()), conj.clone()]
            };
            for base in bases {
                let mut r = rho.clone();
                for _ in 0..*a {
                    r = eis_mul(r, (&base.0, &base.1));
                }
                let mut ex2 = ex.clone();
                ex2[i] = *a;
                next.push((r, ex2));
            }
        }
        rhos = next;
    }
    let units = [
        (Int::ONE, Int::ZERO),
        (Int::ZERO, Int::ONE),
        (Int::from(-1), Int::from(-1)),
    ];
    let lambda = (Int::ONE, Int::from(-1));
    let mut out = Vec::new();
    for (rho, ex) in &rhos {
        for e in 0..2u32 {
            // d = Π p^{(v_p(n) − v_p(N γ)) mod 3}, over the primes of 3n.
            let mut d = Int::ONE;
            let mut has_three = false;
            for (i, (p, a)) in primes.iter().enumerate() {
                let mut nexp = ex[i];
                if p == &three {
                    nexp += e;
                    has_three = true;
                }
                d = d.mul(&p.pow((a + 3 - nexp % 3) % 3));
            }
            if !has_three && e == 1 {
                d = d.mul(&three.square());
            }
            for u in &units {
                let mut g = eis_mul(u.clone(), (&rho.0, &rho.1));
                if e == 1 {
                    g = eis_mul(g, (&lambda.0, &lambda.1));
                }
                if let (Some(g0), Some(g1), Some(dv)) = (g.0.to_i128(), g.1.to_i128(), d.to_i128())
                {
                    out.push(EisTwist { g0, g1, d: dv });
                }
            }
        }
    }
    out
}

/// Searches `x + y·ω = γ·(u + v·ω)³` with `|u|, v ≤ bound` for a solution of
/// `x³ + y³ = n·z³` (see the module docs), calling `found(x, y)` on candidates.
fn eisenstein_search(
    twist: &EisTwist,
    bound: u64,
    found: &mut dyn FnMut(i128, i128) -> bool,
) -> bool {
    let (g0, g1, d) = (twist.g0, twist.g1, twist.d);
    let b = bound as i128;
    // |A| ≤ 5b³, |B| ≤ 6b³, |x + y| ≤ (|g0 + g1| + |g0 − 2g1|)·6b³.
    let gmax = (g0 + g1).abs() + (g0 - 2 * g1).abs() + g0.abs() + 2 * g1.abs();
    let fits = b
        .checked_mul(b)
        .and_then(|t| t.checked_mul(b))
        .and_then(|t| t.checked_mul(8))
        .and_then(|t| t.checked_mul(gmax))
        .is_some_and(|t| t < (1i128 << 120));
    if !fits {
        return false;
    }
    let (pa, pb) = (g0 + g1, g0 - 2 * g1);
    for v in 0..=b {
        let (lo, hi) = if v == 0 { (1, 1) } else { (-b, b) };
        let v2 = v * v;
        let v3 = v2 * v;
        for u in lo..=hi {
            // (u + vω)³ = A + Bω, A = u³ − 3uv² + v³, B = 3u²v − 3uv².
            let uv = u * v;
            let big_a = u * u * u - 3 * uv * v + v3;
            let big_b = 3 * uv * (u - v);
            let p = pa * big_a + pb * big_b;
            if p == 0 || p % d != 0 {
                continue;
            }
            let s3 = p / d;
            if !passes_cube_filter(s3) || icbrt_exact(s3).is_none() {
                continue;
            }
            if gcd_u64(u.unsigned_abs() as u64, v as u64) != 1 {
                continue;
            }
            let x = g0 * big_a - g1 * big_b;
            let y = g0 * big_b + g1 * (big_a - big_b);
            if found(x, y) {
                return true;
            }
        }
    }
    false
}

/// `(x/z, y/z)` with `z³ = (x³ + y³)/n`, if that is a rational cube.
fn scale_to_cube_sum(n: &Int, x: i128, y: i128) -> Option<(Rational, Rational)> {
    let (xi, yi) = (Int::from(x), Int::from(y));
    let z3 = Rational::new(xi.pow(3).add(&yi.pow(3)), n.clone());
    if z3.is_zero() || xi.is_zero() || yi.is_zero() {
        return None;
    }
    let zn = z3.numerator().nth_root_exact(3)?;
    let zd = z3.denominator().nth_root_exact(3)?;
    let z = Rational::new(zn, zd);
    Some((
        Rational::from_integer(xi) / z.clone(),
        Rational::from_integer(yi) / z,
    ))
}

/// Maximum number of primes of `3n` whose classes are enumerated (`3^r`
/// coverings before deduplication); beyond that only the leading primes vary.
const MAX_DESCENT_PRIMES: usize = 6;

fn search_cube_free(n: &Int, primes: &[(Int, u32)], effort: u64) -> Option<(Rational, Rational)> {
    let covs = coverings(n, primes, MAX_DESCENT_PRIMES);
    let twists = eisenstein_twists(primes, MAX_DESCENT_PRIMES);
    let target = Rational::from_integer(n.clone());
    let mut h: u64 = 4.min(effort.max(1));
    loop {
        if let Some(r) = search_coverings(n, primes, &covs, h, &target) {
            return Some(r);
        }
        for tw in &twists {
            let mut result = None;
            eisenstein_search(tw, h, &mut |x, y| match scale_to_cube_sum(n, x, y) {
                Some((u, v)) if u.pow(3) + v.pow(3) == target => {
                    result = Some((u, v));
                    true
                }
                _ => false,
            });
            if result.is_some() {
                return result;
            }
        }
        if h >= effort {
            return None;
        }
        h = (h * 2).min(effort);
    }
}

fn search_coverings(
    n: &Int,
    primes: &[(Int, u32)],
    covs: &[Covering],
    h: u64,
    target: &Rational,
) -> Option<(Rational, Rational)> {
    for cov in covs {
        let (Some(a), Some(b), Some(c)) = (cov.a.to_i128(), cov.b.to_i128(), cov.c.to_i128())
        else {
            continue;
        };
        // Box |x| ≤ H/a^{1/3}, y ≤ H/b^{1/3}; keep a·x³, b·y³ well inside i128.
        let bx = (h as f64 / libm_cbrt(a as f64)) as u64 + 1;
        let by = (h as f64 / libm_cbrt(b as f64)) as u64 + 1;
        let lim = 1i128 << 100;
        let fits = |coef: i128, bound: u64| {
            let t = bound as i128;
            t.checked_mul(t)
                .and_then(|v| v.checked_mul(t))
                .and_then(|v| v.checked_mul(coef))
                .is_some_and(|v| v < lim)
        };
        if !fits(a, bx) || !fits(b, by) {
            continue;
        }
        let modulus = sieve_modulus(&cov.c, primes);
        let mut result = None;
        diagonal_search(a, b, c, modulus, bx, by, &mut |x, y, z| {
            let (xi, yi, zi) = (Int::from(x), Int::from(y), Int::from(z));
            match covering_to_cube_sum(n, cov, &xi, &yi, &zi) {
                Some((u, v)) if u.pow(3) + v.pow(3) == *target => {
                    result = Some((u, v));
                    true
                }
                _ => false,
            }
        });
        if result.is_some() {
            return result;
        }
    }
    None
}

/// The largest divisor of `c` built from whole prime powers that is at most
/// [`TABLE_LIMIT`] (`c` is a cube-free product of primes of `3n`).
fn sieve_modulus(c: &Int, primes: &[(Int, u32)]) -> u64 {
    if let Some(cv) = c.to_u64()
        && cv <= TABLE_LIMIT
    {
        return cv;
    }
    let mut cands: Vec<u64> = Vec::new();
    let mut rest = c.clone();
    let mut ps: Vec<Int> = primes.iter().map(|(p, _)| p.clone()).collect();
    ps.push(Int::from(3));
    for p in ps {
        let mut pe = Int::ONE;
        while let Some((q, r)) = rest.div_rem(&p)
            && r.is_zero()
        {
            rest = q;
            pe = pe.mul(&p);
        }
        if let Some(v) = pe.to_u64()
            && v > 1
            && v <= TABLE_LIMIT
        {
            cands.push(v);
        }
    }
    cands.sort_unstable_by(|x, y| y.cmp(x));
    let mut m = 1u64;
    for v in cands {
        if m * v <= TABLE_LIMIT {
            m *= v;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_root_and_filters() {
        for q in [
            -1000i128,
            -27,
            -1,
            0,
            1,
            8,
            1_000_000_000_000,
            123_456_789i128.pow(3),
        ] {
            let r = icbrt_exact(q).unwrap();
            assert_eq!(r * r * r, q);
            assert!(passes_cube_filter(q));
        }
        assert_eq!(icbrt_exact(2), None);
        assert_eq!(icbrt_exact(-9), None);
        assert!(!passes_cube_filter(2));
    }

    #[test]
    fn covering_classes_of_a_prime() {
        let n = Int::from(31);
        let (_, core, primes) = cube_free_part(&n);
        let covs = coverings(&core, &primes, MAX_DESCENT_PRIMES);
        let trips: Vec<(Int, Int, Int)> = covs
            .iter()
            .map(|c| (c.a.clone(), c.b.clone(), c.c.clone()))
            .collect();
        assert_eq!(
            trips,
            vec![
                (Int::ONE, Int::ONE, Int::from(31)),
                (Int::ONE, Int::from(3), Int::from(279))
            ]
        );
        assert!(covs[0].direct);
    }

    #[test]
    fn eisenstein_primes_and_twists() {
        for q in [7i64, 13, 19, 31, 37, 43, 157, 1_000_003, 1_000_000_009] {
            let q = Int::from(q);
            let (a, b) = eisenstein_prime(&q).unwrap();
            assert_eq!(a.square().sub(&a.mul(&b)).add(&b.square()), q);
        }
        // ℓ ≡ 2 (mod 3): ρ = 1 only; ℓ ≡ 1 (mod 3): ρ ∈ {1, π}; in general
        // (3^s + 1)/2 choices of ρ for s split primes.
        let tw = |n: i64| {
            let (_, _, primes) = cube_free_part(&Int::from(n));
            eisenstein_twists(&primes, MAX_DESCENT_PRIMES).len()
        };
        assert_eq!(tw(17), 6);
        assert_eq!(tw(157), 12);
        assert_eq!(tw(7 * 13 * 19), 6 * 14);
        // 157: the solution with numerators ≈ 2·10⁷ comes from (u, v) = (−89, 43).
        let (_, _, primes) = cube_free_part(&Int::from(157));
        let n = Int::from(157);
        let found = eisenstein_twists(&primes, MAX_DESCENT_PRIMES)
            .iter()
            .any(|t| {
                let mut hit = false;
                eisenstein_search(t, 90, &mut |x, y| {
                    hit = scale_to_cube_sum(&n, x, y)
                        .is_some_and(|(u, v)| u.pow(3) + v.pow(3) == Rational::from(157i64));
                    hit
                });
                hit
            });
        assert!(found);
    }

    #[test]
    fn covering_map_for_six() {
        // 1·1³ + 2·1³ = 3·1³ on the covering (1, 2, 3) of x³ + y³ = 6.
        let n = Int::from(6);
        let (_, core, primes) = cube_free_part(&n);
        let covs = coverings(&core, &primes, MAX_DESCENT_PRIMES);
        let cov = covs
            .iter()
            .find(|c| c.a.is_one() && c.b == Int::from(2) && c.c == Int::from(3))
            .unwrap();
        let (x, y) = covering_to_cube_sum(&n, cov, &Int::ONE, &Int::ONE, &Int::ONE).unwrap();
        assert_eq!(x.pow(3) + y.pow(3), Rational::from(6i64));
    }
}
