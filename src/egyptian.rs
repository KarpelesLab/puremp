//! Egyptian fractions: writing a rational as a sum of **distinct** unit
//! fractions `1/d₁ + 1/d₂ + … + 1/dₖ` (Mathematica's `EgyptianFraction`-style
//! expansions).
//!
//! Every method splits `x` into `⌊x⌋ + {x}` and expands the fractional part
//! `{x} ∈ [0, 1)`; the result is an [`EgyptianExpansion`] holding the integer
//! part and the strictly increasing list of denominators. (Expanding a value
//! `≥ 1` purely into unit fractions is possible — the harmonic series diverges —
//! but needs `e^{x}`-many terms, so it is not what callers want.)
//!
//! - [`Rational::egyptian_greedy`] — the Fibonacci–Sylvester greedy algorithm:
//!   repeatedly take the largest unit fraction `≤` the remainder. Always
//!   terminates (the numerator strictly drops), but the denominators can grow
//!   doubly exponentially (`31/311` needs a 537-digit last denominator).
//! - [`Rational::egyptian_short`] — the divisor ("practical number") method of
//!   Bleicher–Erdős / Vose / Tenenbaum–Yokota: few terms and denominators of
//!   size about `b²`, even for huge `b`.
//! - [`Rational::egyptian_shortest`] — an exact minimum-length expansion by
//!   iterative-deepening depth-first search, with a work budget.
//!
//! Every returned expansion has pairwise-distinct positive denominators and sums
//! exactly to the input; debug builds re-check this before returning.

use alloc::vec::Vec;

use crate::int::Int;
use crate::rational::Rational;

/// An Egyptian-fraction expansion `integer_part + 1/d₁ + 1/d₂ + … + 1/dₖ`.
///
/// Produced by the `egyptian_*` methods on [`Rational`]. The denominators are
/// strictly increasing (hence distinct) and all `≥ 2`, so their unit fractions
/// sum to the fractional part `x − ⌊x⌋ ∈ [0, 1)`; `integer_part` is `⌊x⌋`
/// (negative for negative inputs, e.g. `−1/3 = −1 + 1/2 + 1/6`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EgyptianExpansion {
    /// `⌊x⌋`, the integer part of the expanded value.
    pub integer_part: Int,
    /// The unit-fraction denominators, strictly increasing.
    pub denominators: Vec<Int>,
}

impl EgyptianExpansion {
    /// Number of unit fractions (the integer part is not counted).
    #[inline]
    pub fn len(&self) -> usize {
        self.denominators.len()
    }

    /// Returns `true` if there are no unit fractions (the value is an integer).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.denominators.is_empty()
    }

    /// The largest denominator, or `None` for an integer value.
    #[inline]
    pub fn max_denominator(&self) -> Option<&Int> {
        self.denominators.last()
    }

    /// The exact value `integer_part + Σ 1/dᵢ`. Panics if a denominator is zero.
    pub fn value(&self) -> Rational {
        let (num, den) = unit_sum(&self.denominators);
        Rational::new(num.add(&self.integer_part.mul(&den)), den)
    }

    /// Returns `true` if the denominators are strictly increasing, all `≥ 2`,
    /// and the expansion sums exactly to `x` with `integer_part == ⌊x⌋`.
    pub fn is_valid_for(&self, x: &Rational) -> bool {
        let increasing = self.denominators.windows(2).all(|w| w[0] < w[1]);
        let positive = self.denominators.first().is_none_or(|d| *d >= Int::from(2));
        increasing && positive && self.integer_part == x.floor() && self.value() == *x
    }
}

/// Outcome of [`Rational::egyptian_shortest_with_budget`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShortestEgyptian {
    /// A minimum-length expansion (ties broken as documented on
    /// [`Rational::egyptian_shortest_with_budget`]).
    Found(EgyptianExpansion),
    /// Proven: no expansion with at most `max_terms` unit fractions exists.
    NoneWithin,
    /// The search budget ran out before the answer was settled.
    BudgetExhausted,
}

/// Default node budget of [`Rational::egyptian_shortest`].
pub const DEFAULT_SHORTEST_BUDGET: u64 = 2_000_000;

impl Rational {
    /// Splits `self` into `(⌊self⌋, p, q)` with `{self} = p/q`, `0 ≤ p < q`,
    /// `gcd(p, q) = 1`.
    fn egyptian_parts(&self) -> (Int, Int, Int) {
        let (fl, p) = self.numerator().div_rem_floor(self.denominator());
        (fl, p, self.denominator().clone())
    }

    /// Wraps denominators into an expansion, checking it in debug builds.
    fn egyptian_finish(&self, integer_part: Int, mut denominators: Vec<Int>) -> EgyptianExpansion {
        denominators.sort();
        let e = EgyptianExpansion {
            integer_part,
            denominators,
        };
        debug_assert!(e.is_valid_for(self), "invalid Egyptian expansion");
        e
    }

    /// Greedy (Fibonacci–Sylvester) Egyptian-fraction expansion: `⌊self⌋` plus,
    /// for the fractional part `r`, repeatedly the largest unit fraction
    /// `1/⌈1/r⌉ ≤ r`.
    ///
    /// For `r = p/q` the next remainder has numerator `−q mod p < p`, so at most
    /// `p` terms are produced and the denominators come out strictly increasing.
    /// **Beware:** each denominator is roughly the square of the previous one, so
    /// the last can have `~2ᵏ·log q` digits; e.g. `5/121` gives a 5-term
    /// expansion ending in a 25-digit denominator and `31/311` a 10-term one
    /// ending in a 537-digit denominator. With a large numerator (`p ≳ 10⁶`) the
    /// output can be astronomically large — prefer [`Rational::egyptian_short`].
    ///
    /// ```
    /// use puremp::{Int, Rational};
    /// let e = Rational::new(Int::from(4), Int::from(13)).egyptian_greedy();
    /// let d: Vec<i64> = e.denominators.iter().map(|d| d.to_i64().unwrap()).collect();
    /// assert_eq!(d, [4, 18, 468]);
    /// ```
    pub fn egyptian_greedy(&self) -> EgyptianExpansion {
        let (fl, p, q) = self.egyptian_parts();
        let dens = greedy(p, q, None).expect("uncapped greedy always finishes");
        self.egyptian_finish(fl, dens)
    }

    /// Short Egyptian-fraction expansion by the **divisor method** (Bleicher–
    /// Erdős, Vose, Tenenbaum–Yokota): few terms and small denominators.
    ///
    /// For the fractional part `a/b`, pick a *practical* number `M` (one whose
    /// distinct divisors sum to every integer up to `σ(M)`) and write
    ///
    /// `a/b = q/M + s/(bM)`, `q = ⌊aM/b⌋`, `s = aM mod b < b`.
    ///
    /// Both `q` and `s` (when `s ≤ σ(M)`) are sums of distinct divisors of `M`,
    /// and each divisor `d` contributes the unit fraction `d/M = 1/(M/d)`
    /// resp. `d/(bM) = 1/(b·M/d)`. The two families cannot collide (that would
    /// need `b | M`, which forces `s = 0`), so all denominators are distinct by
    /// construction — no duplicate-splitting pass is needed — and bounded by
    /// `bM`.
    ///
    /// Divisor sums use the greedy rule "largest unused divisor `≤` what is
    /// left", which always succeeds for practical `M` (their sorted divisors
    /// satisfy `dᵢ₊₁ ≤ 1 + d₁ + … + dᵢ`). For `b ≤ 2²⁰` every practical
    /// 13-smooth `M ≤ 64b` is tried, so denominators stay below `64b²`. For
    /// larger `b` the method runs in mixed radix: `M = Nᵏ` is the least power
    /// of a highly composite `N ≤ 10¹²` with `M ≥ b`, and each base-`N` digit
    /// of `q` and `s` is split over the divisors of `N` separately (a divisor
    /// `d` of digit `j` gives the divisor `d·Nʲ` of `M`; these never repeat).
    /// That gives `O(log b / log N)` digits of a few terms each — e.g. about 90
    /// terms below `b²·N` for a random 100-digit `a/b`, where greedy would not
    /// finish. The cheapest result wins — fewest terms, then smallest largest
    /// denominator — and the greedy expansion also competes when its
    /// denominators stay below `b²`.
    ///
    /// ```
    /// use puremp::{Int, Rational};
    /// let x = Rational::new(Int::from(31), Int::from(311));
    /// let e = x.egyptian_short();
    /// assert_eq!(e.value(), x);
    /// assert!(e.len() < x.egyptian_greedy().len());
    /// ```
    pub fn egyptian_short(&self) -> EgyptianExpansion {
        let (fl, a, b) = self.egyptian_parts();
        let dens = short(&a, &b);
        self.egyptian_finish(fl, dens)
    }

    /// Minimum-length Egyptian-fraction expansion with at most `max_terms` unit
    /// fractions, searched within [`DEFAULT_SHORTEST_BUDGET`] nodes.
    ///
    /// Returns `None` both when no expansion of length `≤ max_terms` exists and
    /// when the budget is exhausted first; use
    /// [`Rational::egyptian_shortest_with_budget`] to tell the two apart or to
    /// change the budget.
    ///
    /// ```
    /// use puremp::{Int, Rational};
    /// let e = Rational::new(Int::from(5), Int::from(121)).egyptian_shortest(4).unwrap();
    /// let d: Vec<i64> = e.denominators.iter().map(|d| d.to_i64().unwrap()).collect();
    /// assert_eq!(d, [33, 121, 363]);
    /// ```
    pub fn egyptian_shortest(&self, max_terms: usize) -> Option<EgyptianExpansion> {
        match self.egyptian_shortest_with_budget(max_terms, DEFAULT_SHORTEST_BUDGET) {
            ShortestEgyptian::Found(e) => Some(e),
            _ => None,
        }
    }

    /// Minimum-length Egyptian-fraction expansion of the fractional part, by
    /// iterative-deepening depth-first search over increasing denominators.
    ///
    /// For each length `k = 0, 1, …, max_terms` it enumerates `d₁ < d₂ < …`
    /// with the standard bounds: with remainder `r` and `j` terms still to place,
    /// `1/r < dᵢ < j/r` (and `dᵢ = 1/r` exactly on the last term); two-term
    /// tails `r = 1/x + 1/y` are solved directly from the divisors of `q²`
    /// (`(px − q)(py − q) = q²`) when `q` is small enough to factor.
    ///
    /// **Tie-break:** among minimum-length expansions it returns the one with
    /// the smallest largest denominator, and among those the lexicographically
    /// smallest denominator list.
    ///
    /// **Work cap:** every search node (candidate denominator tried, divisor
    /// examined) costs one unit of `budget`; when it runs out the search stops
    /// and returns [`ShortestEgyptian::BudgetExhausted`] — the answer is only
    /// returned once both its length and the tie-break are proven optimal.
    pub fn egyptian_shortest_with_budget(&self, max_terms: usize, budget: u64) -> ShortestEgyptian {
        let (fl, p, q) = self.egyptian_parts();
        if p.is_zero() {
            return ShortestEgyptian::Found(self.egyptian_finish(fl, Vec::new()));
        }
        let mut s = Search {
            budget,
            used: 0,
            exhausted: false,
            best: None,
            cur: Vec::new(),
        };
        for k in 1..=max_terms {
            s.dfs(&p, &q, k, &Int::from(2));
            if s.exhausted {
                return ShortestEgyptian::BudgetExhausted;
            }
            if let Some(best) = s.best.take() {
                return ShortestEgyptian::Found(self.egyptian_finish(fl, best));
            }
        }
        ShortestEgyptian::NoneWithin
    }
}

// --- helpers ---

/// `Σ 1/dᵢ` as an unreduced fraction `(num, den)`, by binary splitting.
fn unit_sum(ds: &[Int]) -> (Int, Int) {
    match ds.len() {
        0 => (Int::ZERO, Int::ONE),
        1 => (Int::ONE, ds[0].clone()),
        n => {
            let (an, ad) = unit_sum(&ds[..n / 2]);
            let (bn, bd) = unit_sum(&ds[n / 2..]);
            (an.mul(&bd).add(&bn.mul(&ad)), ad.mul(&bd))
        }
    }
}

/// Greedy expansion of `p/q` (`0 ≤ p < q`, coprime). With `cap_bits`, gives
/// up (`None`) once a denominator exceeds that many bits.
fn greedy(mut p: Int, mut q: Int, cap_bits: Option<u32>) -> Option<Vec<Int>> {
    let mut out = Vec::new();
    while !p.is_zero() {
        if p.is_one() {
            out.push(q);
            break;
        }
        // p ∤ q (coprime, p > 1), so ⌈q/p⌉ = ⌊q/p⌋ + 1.
        let d = q.div_floor(&p).add(&Int::ONE);
        if cap_bits.is_some_and(|c| d.bit_len() > c) {
            return None;
        }
        let np = p.mul(&d).sub(&q);
        let nq = q.mul(&d);
        let g = np.gcd(&nq);
        p = np.div_exact(&g);
        q = nq.div_exact(&g);
        out.push(d);
    }
    if cap_bits.is_some_and(|c| out.last().is_some_and(|d| d.bit_len() > c)) {
        return None;
    }
    Some(out)
}

/// Candidate cost: fewer terms first, then smaller largest denominator.
fn better(a: &[Int], b: &[Int]) -> bool {
    match a.len().cmp(&b.len()) {
        core::cmp::Ordering::Less => true,
        core::cmp::Ordering::Greater => false,
        core::cmp::Ordering::Equal => a.iter().max() < b.iter().max(),
    }
}

/// A factorization `[(p, e), …]` with ascending primes.
type Factored = Vec<(u64, u32)>;

const SMOOTH_PRIMES: [u64; 6] = [2, 3, 5, 7, 11, 13];

/// Whether `∏ pᵢ^{eᵢ}` (ascending primes, positive exponents) is practical,
/// by Stewart–Sierpiński: `p₁ = 2` and `pᵢ₊₁ ≤ 1 + σ(p₁^{e₁}⋯pᵢ^{eᵢ})`.
fn is_practical(fac: &[(u64, u32)]) -> bool {
    let mut sigma: u128 = 1;
    for (i, &(p, e)) in fac.iter().enumerate() {
        if i == 0 && p != 2 {
            return false;
        }
        if u128::from(p) > sigma + 1 {
            return false;
        }
        let mut s: u128 = 1;
        let mut pk: u128 = 1;
        for _ in 0..e {
            pk *= u128::from(p);
            s += pk;
        }
        sigma = sigma.saturating_mul(s);
    }
    true
}

/// All practical 13-smooth `M ≤ limit` (`M ≥ 2`), with their factorizations.
fn practical_smooth_upto(limit: u64) -> Vec<(u64, Factored)> {
    fn rec(
        idx: usize,
        m: u64,
        limit: u64,
        fac: &mut Vec<(u64, u32)>,
        out: &mut Vec<(u64, Factored)>,
    ) {
        if idx == SMOOTH_PRIMES.len() {
            if m >= 2 && is_practical(fac) {
                out.push((m, fac.clone()));
            }
            return;
        }
        rec(idx + 1, m, limit, fac, out);
        let p = SMOOTH_PRIMES[idx];
        let mut v = m;
        let mut e = 0;
        while let Some(nv) = v.checked_mul(p).filter(|&nv| nv <= limit) {
            v = nv;
            e += 1;
            fac.push((p, e));
            rec(idx + 1, v, limit, fac, out);
            fac.pop();
        }
    }
    let mut out = Vec::new();
    rec(0, 1, limit, &mut Vec::new(), &mut out);
    out
}

/// Sorted divisors of `∏ pᵢ^{eᵢ}`.
fn divisors_u64(fac: &[(u64, u32)]) -> Vec<u64> {
    let mut divs = alloc::vec![1u64];
    for &(p, e) in fac {
        let base = divs.clone();
        let mut pk = 1u64;
        for _ in 0..e {
            pk *= p;
            divs.extend(base.iter().map(|d| d * pk));
        }
    }
    divs.sort_unstable();
    divs
}

/// Writes `x` as a sum of distinct elements of the sorted divisor list of a
/// practical number (greedy, never reusing a divisor). `None` if it fails
/// (only possible when `x > σ(M)`).
fn split_sorted(mut x: u128, divs: &[u64]) -> Option<Vec<u64>> {
    let mut out = Vec::new();
    let mut end = divs.len();
    while x > 0 {
        // Largest divisor among divs[..end] that is ≤ x.
        let k = divs[..end].partition_point(|&d| u128::from(d) <= x);
        if k == 0 {
            return None;
        }
        let d = divs[k - 1];
        out.push(d);
        x -= u128::from(d);
        end = k - 1;
    }
    Some(out)
}

/// Divisor-method expansion over small (`b ≤ 2²⁰`) denominators.
fn short_small(a: u64, b: u64) -> Option<Vec<Int>> {
    let limit = (64 * b).max(720);
    let mut best: Option<Vec<Int>> = None;
    for (m, fac) in practical_smooth_upto(limit) {
        let am = u128::from(a) * u128::from(m);
        let (q, s) = (am / u128::from(b), am % u128::from(b));
        let divs = divisors_u64(&fac);
        let Some(qs) = split_sorted(q, &divs) else {
            continue;
        };
        let Some(ss) = split_sorted(s, &divs) else {
            continue;
        };
        if best
            .as_ref()
            .is_some_and(|bst| qs.len() + ss.len() > bst.len())
        {
            continue;
        }
        let mut dens: Vec<Int> = qs.iter().map(|d| Int::from(m / d)).collect();
        dens.extend(
            ss.iter()
                .map(|d| Int::from_u128(u128::from(b) * u128::from(m / d))),
        );
        if best.as_ref().is_none_or(|bst| better(&dens, bst)) {
            best = Some(dens);
        }
    }
    best
}

/// Highly composite numbers `N ≤ limit` (records of the divisor count among
/// numbers `2^{e₁}·3^{e₂}⋯` with non-increasing exponents), with their
/// factorizations. All of them are practical.
fn highly_composite_upto(limit: u64) -> Vec<(u64, Factored)> {
    const PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    fn rec(
        idx: usize,
        m: u64,
        max_e: u32,
        tau: u64,
        limit: u64,
        fac: &mut Vec<(u64, u32)>,
        out: &mut Vec<(u64, u64, Factored)>,
    ) {
        out.push((m, tau, fac.clone()));
        if idx == PRIMES.len() {
            return;
        }
        let p = PRIMES[idx];
        let mut v = m;
        for e in 1..=max_e {
            match v.checked_mul(p) {
                Some(nv) if nv <= limit => v = nv,
                _ => break,
            }
            fac.push((p, e));
            rec(idx + 1, v, e, tau * u64::from(e + 1), limit, fac, out);
            fac.pop();
        }
    }
    let mut all = Vec::new();
    rec(0, 1, u32::MAX, 1, limit, &mut Vec::new(), &mut all);
    all.sort_unstable_by_key(|&(m, _, _)| m);
    let mut record = 1;
    let mut out = Vec::new();
    for (m, tau, fac) in all {
        if tau > record {
            record = tau;
            out.push((m, fac));
        }
    }
    out
}

/// Divisor-method expansion of `a/b` over `M = Nᵏ`, the least power of `N`
/// that is `≥ b`, using base-`N` digits.
///
/// Writing `x = Σ cⱼ·Nʲ` with digits `cⱼ < N`, each digit is a sum of distinct
/// divisors `d | N` (`N` is practical), giving distinct divisors `d·Nʲ` of `M`
/// (`d·Nʲ = d'·N^{j'}` with `j < j'` would force `d ≥ N > cⱼ`). This is the
/// mixed-radix form of the method: `O(log b / log N)` digits of a few terms
/// each, instead of one huge divisor-sum problem. `None` once more than
/// `max_len` terms are needed.
fn short_radix(a: &Int, b: &Int, n: u64, divs: &[u64], max_len: usize) -> Option<Vec<Int>> {
    let big_n = Int::from(n);
    let mut pows = alloc::vec![Int::ONE];
    while pows.last().expect("nonempty") < b {
        let next = pows.last().expect("nonempty").mul(&big_n);
        pows.push(next);
    }
    let k = pows.len() - 1;
    let (q, s) = a.mul(&pows[k]).div_rem_floor(b);
    let mut dens = Vec::new();
    for (mut x, scale) in [(q, Int::ONE), (s, b.clone())] {
        let mut j = 0;
        while !x.is_zero() {
            let (nx, c) = x.div_rem_floor(&big_n);
            let c = c.to_u64().expect("digit < N");
            for d in split_sorted(u128::from(c), divs)? {
                // 1/(M/(d·Nʲ)) resp. 1/(b·M/(d·Nʲ)), with M/(d·Nʲ) = N^{k−j}/d.
                dens.push(scale.mul(&pows[k - j].div_exact(&Int::from(d))));
            }
            if dens.len() > max_len {
                return None;
            }
            x = nx;
            j += 1;
        }
    }
    Some(dens)
}

/// Divisor-method expansion over large denominators: base-`N` digits for each
/// highly composite `N ≤ 10¹²`, keeping the best.
fn short_large(a: &Int, b: &Int) -> Vec<Int> {
    let mut best: Option<Vec<Int>> = None;
    for (n, fac) in highly_composite_upto(1_000_000_000_000) {
        let divs = divisors_u64(&fac);
        let max_len = best.as_ref().map_or(usize::MAX, Vec::len);
        if let Some(dens) = short_radix(a, b, n, &divs, max_len)
            && best.as_ref().is_none_or(|bst| better(&dens, bst))
        {
            best = Some(dens);
        }
    }
    best.expect("N = 2 always succeeds")
}

/// Short expansion of `a/b` (`0 ≤ a < b`, coprime) — see
/// [`Rational::egyptian_short`].
fn short(a: &Int, b: &Int) -> Vec<Int> {
    if a.is_zero() {
        return Vec::new();
    }
    if a.is_one() {
        return alloc::vec![b.clone()];
    }
    let mut best = match (a.to_u64(), b.to_u64()) {
        (Some(a), Some(b)) if b <= 1 << 20 => short_small(a, b),
        _ => None,
    }
    .unwrap_or_else(|| short_large(a, b));
    if let Some(g) = greedy(a.clone(), b.clone(), Some(2 * b.bit_len()))
        && better(&g, &best)
    {
        best = g;
    }
    best
}

/// Iterative-deepening state for [`Rational::egyptian_shortest_with_budget`].
struct Search {
    budget: u64,
    used: u64,
    exhausted: bool,
    /// Best complete expansion of the current length (largest denominator last).
    best: Option<Vec<Int>>,
    cur: Vec<Int>,
}

impl Search {
    /// Charges `n` units of work; returns `false` once the budget is gone.
    fn charge(&mut self, n: u64) -> bool {
        self.used = self.used.saturating_add(n);
        if self.used > self.budget {
            self.exhausted = true;
        }
        !self.exhausted
    }

    fn record(&mut self, tail: &[Int]) {
        let mut v = self.cur.clone();
        v.extend_from_slice(tail);
        self.best = Some(v);
    }

    /// Bound on denominators imposed by the best solution so far: everything
    /// must stay strictly below its largest denominator.
    fn bound(&self) -> Option<&Int> {
        self.best.as_ref().and_then(|b| b.last())
    }

    /// Expands `p/q` (coprime, `p > 0`) into exactly `j` unit fractions with
    /// denominators `≥ min_d`, strictly increasing.
    fn dfs(&mut self, p: &Int, q: &Int, j: usize, min_d: &Int) {
        if !self.charge(1) {
            return;
        }
        if j == 1 {
            if p.is_one() && q >= min_d && self.bound().is_none_or(|b| q < b) {
                self.record(core::slice::from_ref(q));
            }
            return;
        }
        // 1/r < d < j/r, i.e. ⌊q/p⌋ + 1 ≤ d ≤ ⌊(jq − 1)/p⌋.
        let mut lo = q.div_floor(p).add(&Int::ONE);
        if lo < *min_d {
            lo = min_d.clone();
        }
        let jq = Int::from(j as u64).mul(q);
        let mut hi = jq.sub(&Int::ONE).div_floor(p);
        if let Some(b) = self.bound() {
            let cap = b.sub(&Int::from(j as u64));
            if cap < hi {
                hi = cap;
            }
        }
        if lo > hi {
            return;
        }
        if j == 2 && hi.sub(&lo) > Int::from(64u64) && q.bit_len() <= 96 {
            self.two_terms(p, q, min_d);
            return;
        }
        let mut d = lo;
        while d <= hi {
            if self.exhausted {
                return;
            }
            let np = p.mul(&d).sub(q);
            let nq = q.mul(&d);
            let g = np.gcd(&nq);
            let (np, nq) = (np.div_exact(&g), nq.div_exact(&g));
            let next = d.add(&Int::ONE);
            self.cur.push(d);
            self.dfs(&np, &nq, j - 1, &next);
            d = self.cur.pop().expect("pushed above");
            d = d.add(&Int::ONE);
            // The bound may have tightened.
            if let Some(b) = self.bound() {
                let cap = b.sub(&Int::from(j as u64));
                if cap < hi {
                    hi = cap;
                }
            }
        }
    }

    /// Solves `p/q = 1/x + 1/y` with `min_d ≤ x < y` directly:
    /// `(px − q)(py − q) = q²`, so `x = (q + u)/p` for a divisor `u < q` of
    /// `q²` with `u ≡ −q (mod p)`; the largest such `u` minimizes `y`.
    fn two_terms(&mut self, p: &Int, q: &Int, min_d: &Int) {
        let fac = q.factor_exponents();
        let mut divs = alloc::vec![Int::ONE];
        for (pr, e) in &fac {
            let base = divs.clone();
            let mut pk = Int::ONE;
            for _ in 0..2 * e {
                pk = pk.mul(pr);
                divs.extend(base.iter().map(|d| d.mul(&pk)).filter(|d| d < q));
            }
            if !self.charge(divs.len() as u64) {
                return;
            }
        }
        let q2 = q.square();
        let mut best: Option<(Int, Int)> = None;
        for u in divs {
            let (x, r) = q.add(&u).div_rem_floor(p);
            if !r.is_zero() || x < *min_d {
                continue;
            }
            if best.as_ref().is_none_or(|(bu, _)| u > *bu) {
                best = Some((u, x));
            }
        }
        if let Some((u, x)) = best {
            let y = q.add(&q2.div_exact(&u)).div_exact(p);
            if self.bound().is_none_or(|b| y < *b) {
                self.record(&[x, y]);
            }
        }
    }
}
