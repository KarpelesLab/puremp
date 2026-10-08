//! Jacobsthal's function on [`Int`].
//!
//! `j(n)` is the least `m` such that every run of `m` consecutive integers
//! contains one coprime to `n`; equivalently `j(n) − 1` is the length of the
//! longest run of consecutive integers each sharing a prime factor with `n`.
//! It depends only on the set of distinct primes dividing `n`, so `j(1) = 1`,
//! `j(p) = 2` and `j(n)` for `n = p₁ᵉ¹⋯p_kᵉᵏ` equals `j(p₁⋯p_k)`.
//!
//! - [`Int::jacobsthal`] — `j(|n|)`.
//! - [`Int::jacobsthal_primorial`] — `h(k) = j(p₁p₂⋯p_k)`, the Jacobsthal
//!   function of the `k`-th primorial (OEIS A048670).
//! - [`Int::jacobsthal_witness`] — the start of a run of `j(n) − 1`
//!   consecutive integers none of which is coprime to `n`.
//!
//! # Algorithm
//!
//! By the Chinese Remainder Theorem, a run `s, s+1, …, s+L−1` of integers each
//! divisible by some `p | n` exists iff the positions `{0, …, L−1}` can be
//! covered by choosing *one* residue class `aₚ (mod p)` per prime `p` (take
//! `s ≡ −aₚ (mod p)`). So `j(n) − 1` is the largest coverable `L`, and
//! coverability is monotone in `L`; we test `L = 1, 2, …` until it fails,
//! jumping ahead whenever a found covering already extends past `L`.
//!
//! Each test is an exact branch-and-bound search over the uncovered positions
//! (kept as a bitset), in the spirit of the covering searches of Hagedorn
//! (*Math. Comp.* 78, 2009) and Hajdu–Saradha (*Math. Comp.* 81, 2012):
//!
//! - **Position branching.** Every uncovered position `x` must be hit by some
//!   still-unassigned prime `q` (forcing the class `x mod q`), or by a prime
//!   `q ≥ L`, which can never cover more than one position (such primes are
//!   interchangeable and kept as a counter). We branch on the position whose
//!   candidate classes cover the most (sum of cubes of the newly covered
//!   counts) — this "most-constraining-coverage" choice shrinks the trees by
//!   orders of magnitude versus first-uncovered or fewest-options choices.
//! - **Duplicate elimination.** Branch `i` at `x` forbids the classes through
//!   `x` of the primes tried in branches `< i`, so every covering is reached
//!   once (by the first prime, in branch order, that covers `x`).
//! - **Capacity bound.** Prune when the uncovered count exceeds the number of
//!   free large primes plus, for each unassigned prime, the most uncovered
//!   positions any of its allowed classes could still hit.
//! - **Mirror symmetry.** `x ↦ L−1−x` maps coverings to coverings, so for the
//!   smallest prime whose classes are not all mirror-fixed we forbid one class
//!   of each swapped pair.
//!
//! Cost grows quickly with the number of primes: the search is capped at
//! [`Int::JACOBSTHAL_MAX_PRIMES`] distinct primes, beyond which the functions
//! return `None`.

use alloc::vec;
use alloc::vec::Vec;

use crate::int::Int;

impl Int {
    /// The largest number of distinct prime factors for which
    /// [`Int::jacobsthal`], [`Int::jacobsthal_primorial`] and
    /// [`Int::jacobsthal_witness`] attempt the exact search (they return
    /// `None` above it). At the cap, primorial `h(24)` takes a few minutes in
    /// a release build; `h(20)` takes about a second.
    pub const JACOBSTHAL_MAX_PRIMES: usize = 24;

    /// Jacobsthal's function `j(|self|)`: the least `m` such that every `m`
    /// consecutive integers include one coprime to `self`.
    ///
    /// `j(±1) = 1`, `j(pᵉ) = 2`, and `j` depends only on the distinct primes of
    /// `self`. Returns `None` for `self = 0` (only `±1` are coprime to `0`, so
    /// no finite `m` works) and when `self` has more than
    /// [`Int::JACOBSTHAL_MAX_PRIMES`] distinct prime factors.
    ///
    /// Exact: an exhaustive branch-and-bound covering search (see the module
    /// documentation), not a heuristic or an asymptotic bound.
    ///
    /// ```
    /// use puremp::Int;
    /// assert_eq!(Int::from(30).jacobsthal(), Some(Int::from(6)));
    /// assert_eq!(Int::from(1).jacobsthal(), Some(Int::from(1)));
    /// assert_eq!(Int::from(0).jacobsthal(), None);
    /// ```
    pub fn jacobsthal(&self) -> Option<Int> {
        let primes = self.jacobsthal_primes()?;
        let (len, _) = longest_run(&primes, false)?;
        Some(Int::from_u64(len + 1))
    }

    /// `h(k)`: Jacobsthal's function of the `k`-th primorial `p₁p₂⋯p_k`
    /// (OEIS A048670; `h(0) = j(1) = 1`). Returns `None` for
    /// `k > JACOBSTHAL_MAX_PRIMES`.
    ///
    /// ```
    /// use puremp::Int;
    /// assert_eq!(Int::jacobsthal_primorial(4), Some(Int::from(10)));
    /// ```
    pub fn jacobsthal_primorial(k: u32) -> Option<Int> {
        let k = k as usize;
        if k > Self::JACOBSTHAL_MAX_PRIMES {
            return None;
        }
        let primes = first_primes(k);
        let (len, _) = longest_run(&primes, false)?;
        Some(Int::from_u64(len + 1))
    }

    /// A witness for [`Int::jacobsthal`]: an `s ≥ 1` such that each of the
    /// `j(n) − 1` integers `s, s+1, …, s + j(n) − 2` shares a prime factor with
    /// `n = |self|` — a longest such run, so `s − 1` and `s + j(n) − 1` are
    /// coprime to `n`.
    ///
    /// `s` is the run found by the search, reduced into `[1, rad(n)]`; it is
    /// not necessarily the smallest such start. Returns `None` for `|self| ≤ 1`
    /// (no non-empty run) and in the cases where [`Int::jacobsthal`] does.
    ///
    /// ```
    /// use puremp::Int;
    /// let n = Int::from(30);
    /// let s = n.jacobsthal_witness().unwrap();
    /// for i in 0..5 {
    ///     assert!(!s.add(&Int::from(i)).gcd(&n).is_one());
    /// }
    /// ```
    pub fn jacobsthal_witness(&self) -> Option<Int> {
        let primes = self.jacobsthal_primes()?;
        let (len, cover) = longest_run(&primes, true)?;
        if len == 0 {
            return None;
        }
        cover.map(|c| c.start(&primes))
    }

    /// The distinct primes of `|self|`, or `None` for `0` / beyond the cap.
    fn jacobsthal_primes(&self) -> Option<Vec<Int>> {
        if self.is_zero() {
            return None;
        }
        let primes: Vec<Int> = self
            .factor_exponents()
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        if primes.len() > Self::JACOBSTHAL_MAX_PRIMES {
            return None;
        }
        Some(primes)
    }
}

/// The first `k` primes.
fn first_primes(k: usize) -> Vec<Int> {
    let mut out = Vec::with_capacity(k);
    let mut p = Int::ONE;
    for _ in 0..k {
        p = p.next_prime();
        out.push(p.clone());
    }
    out
}

/// A covering of `{0, …, len−1}`: `classes[i] = Some(a)` means prime `i`
/// (an index into the prime list) takes the class `a`; `singles` are positions
/// each covered by a distinct prime with no class.
struct Cover {
    classes: Vec<Option<u64>>,
    singles: Vec<u64>,
}

impl Cover {
    /// Extends this covering past `len` as far as it goes: a position is
    /// covered by an assigned class, or else by a still-free prime. Returns the
    /// new length.
    fn extend(&mut self, primes: &[u64], mut len: u64) -> u64 {
        let mut free = self.classes.iter().filter(|c| c.is_none()).count() - self.singles.len();
        loop {
            let hit = self
                .classes
                .iter()
                .zip(primes)
                .any(|(c, &p)| matches!(c, Some(a) if p != 0 && len % p == *a));
            if hit {
            } else if free > 0 {
                free -= 1;
                self.singles.push(len);
            } else {
                return len;
            }
            len += 1;
        }
    }

    /// The run start `s ∈ [1, rad]` realizing this covering, by CRT on
    /// `s ≡ −a (mod p)`.
    fn start(&self, primes: &[Int]) -> Int {
        let mut residues = Vec::new();
        let mut moduli = Vec::new();
        let mut singles = self.singles.iter();
        for (p, c) in primes.iter().zip(&self.classes) {
            let pos = match c {
                Some(a) => Some(*a),
                None => singles.next().copied(),
            };
            if let Some(a) = pos {
                residues.push(Int::from_u64(a).neg().rem_euclid(p));
                moduli.push(p.clone());
            }
        }
        debug_assert!(singles.next().is_none());
        let modulus = moduli.iter().fold(Int::ONE, |m, p| m.mul(p));
        let s = Int::crt(&residues, &moduli).expect("distinct primes are coprime");
        if s.is_zero() { modulus } else { s }
    }
}

/// The longest coverable run length for `primes` (distinct primes), with a
/// covering realizing it when `want_cover` (or when cheaply available).
/// `None` only if the run would exceed the bitset capacity (not reachable
/// within the prime-count cap).
fn longest_run(primes: &[Int], want_cover: bool) -> Option<(u64, Option<Cover>)> {
    // Primes too large for u64 are stored as 0: they can only ever cover a
    // single position of any window we can represent.
    let small: Vec<u64> = primes.iter().map(|p| p.to_u64().unwrap_or(0)).collect();
    let mut len = 0u64;
    let mut best: Option<Cover> = None;
    loop {
        let target = len + 1;
        let found = match target.div_ceil(64) {
            0..=1 => cover::<1>(&small, target),
            2 => cover::<2>(&small, target),
            3..=4 => cover::<4>(&small, target),
            5..=8 => cover::<8>(&small, target),
            9..=16 => cover::<16>(&small, target),
            17..=32 => cover::<32>(&small, target),
            33..=64 => cover::<64>(&small, target),
            _ => return None,
        };
        match found {
            Some(mut c) => {
                len = c.extend(&small, target);
                best = Some(c);
            }
            None => break,
        }
    }
    Some((len, if want_cover { best } else { None }))
}

type Bits<const W: usize> = [u64; W];

#[inline]
fn popcount<const W: usize>(b: &Bits<W>) -> u32 {
    b.iter().map(|w| w.count_ones()).sum()
}

#[inline]
fn and<const W: usize>(a: &Bits<W>, b: &Bits<W>) -> Bits<W> {
    core::array::from_fn(|i| a[i] & b[i])
}

#[inline]
fn and_not<const W: usize>(a: &Bits<W>, b: &Bits<W>) -> Bits<W> {
    core::array::from_fn(|i| a[i] & !b[i])
}

#[inline]
fn bit<const W: usize>(b: &Bits<W>, i: usize) -> bool {
    b[i / 64] >> (i % 64) & 1 == 1
}

#[inline]
fn set_bit<const W: usize>(b: &mut Bits<W>, i: usize) {
    b[i / 64] |= 1 << (i % 64);
}

#[inline]
fn clear_bit<const W: usize>(b: &mut Bits<W>, i: usize) {
    b[i / 64] &= !(1 << (i % 64));
}

/// Iterates the set bits of a bitset in increasing order.
fn ones<const W: usize>(b: Bits<W>) -> impl Iterator<Item = usize> {
    (0..W).flat_map(move |w| {
        let mut word = b[w];
        core::iter::from_fn(move || {
            if word == 0 {
                return None;
            }
            let t = word.trailing_zeros() as usize;
            word &= word - 1;
            Some(w * 64 + t)
        })
    })
}

/// Branch-and-bound state for covering `{0, …, len−1}`.
struct Search<const W: usize> {
    /// The primes `< len` ("small": they may cover several positions).
    primes: Vec<usize>,
    /// Index of each small prime in the caller's prime list.
    index: Vec<usize>,
    /// `masks[i][a]`: positions `≡ a (mod primes[i])`.
    masks: Vec<Vec<Bits<W>>>,
    /// Classes each small prime may no longer take.
    forbidden: Vec<Bits<W>>,
    /// Residue taken by each assigned small prime (valid on the success path).
    chosen: Vec<usize>,
    /// Bitmask of the small primes assigned on the success path.
    used: u64,
    /// Positions handed to large primes on the success path.
    singles: Vec<usize>,
}

/// Searches for a covering of `{0, …, len−1}` by one class per prime of
/// `primes` (`0` marks a prime beyond `u64`). `len ≤ 64·W`.
fn cover<const W: usize>(primes: &[u64], len: u64) -> Option<Cover> {
    let l = len as usize;
    let mut s = Search::<W> {
        primes: Vec::new(),
        index: Vec::new(),
        masks: Vec::new(),
        forbidden: Vec::new(),
        chosen: Vec::new(),
        used: 0,
        singles: Vec::new(),
    };
    for (i, &p) in primes.iter().enumerate() {
        if p != 0 && p < len {
            let p = p as usize;
            s.primes.push(p);
            s.index.push(i);
            s.masks.push(
                (0..p)
                    .map(|a| {
                        let mut m = [0u64; W];
                        for x in (a..l).step_by(p) {
                            set_bit(&mut m, x);
                        }
                        m
                    })
                    .collect(),
            );
        }
    }
    let n_small = s.primes.len();
    // The cap keeps the assigned-set bitmask within a u64.
    debug_assert!(n_small <= 64);
    let big = (primes.len() - n_small) as u32;
    s.forbidden = vec![[0u64; W]; n_small];
    s.chosen = vec![0; n_small];
    // Mirror symmetry x ↦ l−1−x sends class a (mod p) to (l−1−a) mod p.
    for (i, &p) in s.primes.iter().enumerate() {
        let mut broke = false;
        for a in 0..p {
            let b = ((l - 1) % p + p - a) % p;
            if a < b {
                set_bit(&mut s.forbidden[i], b);
                broke = true;
            }
        }
        if broke {
            break;
        }
    }
    let mut all = [0u64; W];
    for x in 0..l {
        set_bit(&mut all, x);
    }
    if !s.dfs(all, 0, big) {
        return None;
    }
    let mut classes = vec![None; primes.len()];
    for i in 0..n_small {
        if s.used >> i & 1 == 1 {
            classes[s.index[i]] = Some(s.chosen[i] as u64);
        }
    }
    Some(Cover {
        classes,
        singles: s.singles.iter().map(|&x| x as u64).collect(),
    })
}

impl<const W: usize> Search<W> {
    /// Can the positions in `u` be covered using the small primes not in
    /// `used` plus `big` single-position primes?
    fn dfs(&mut self, u: Bits<W>, used: u64, big: u32) -> bool {
        let c = popcount(&u);
        if c <= big {
            self.used = used;
            self.singles.extend(ones(u));
            return true;
        }
        let n = self.primes.len();
        let free = |i: usize| used >> i & 1 == 0;

        // Capacity bound: each unassigned prime covers at most its best class.
        let mut bound = big;
        for i in (0..n).filter(|&i| free(i)) {
            let forb = &self.forbidden[i];
            let best = self.masks[i]
                .iter()
                .enumerate()
                .filter(|&(a, _)| !bit(forb, a))
                .map(|(_, m)| popcount(&and(&u, m)))
                .max()
                .unwrap_or(0);
            bound += best;
        }
        if bound < c {
            return false;
        }

        // Branch on the position whose candidate classes cover the most.
        let mut pos = usize::MAX;
        let mut pos_key = 0u64;
        for x in ones(u) {
            let mut options = big > 0;
            let mut key = 0u64;
            for i in (0..n).filter(|&i| free(i)) {
                let a = x % self.primes[i];
                if !bit(&self.forbidden[i], a) {
                    options = true;
                    let k = popcount(&and(&u, &self.masks[i][a])) as u64;
                    key += k * k * k;
                }
            }
            if !options {
                return false;
            }
            if pos == usize::MAX || key > pos_key {
                pos = x;
                pos_key = key;
            }
        }
        let x = pos;

        let mut branches: Vec<(u32, usize)> = (0..n)
            .filter(|&i| free(i) && !bit(&self.forbidden[i], x % self.primes[i]))
            .map(|i| (popcount(&and(&u, &self.masks[i][x % self.primes[i]])), i))
            .collect();
        branches.sort_unstable_by_key(|b| core::cmp::Reverse(b.0));

        let mut found = false;
        let mut tried = 0;
        for &(_, i) in &branches {
            let a = x % self.primes[i];
            self.chosen[i] = a;
            if self.dfs(and_not(&u, &self.masks[i][a]), used | 1 << i, big) {
                found = true;
                break;
            }
            // Later branches: `x` is not covered by prime i.
            set_bit(&mut self.forbidden[i], a);
            tried += 1;
        }
        if !found && big > 0 {
            let mut v = u;
            clear_bit(&mut v, x);
            self.singles.push(x);
            if self.dfs(v, used, big - 1) {
                found = true;
            } else {
                self.singles.pop();
            }
        }
        for &(_, i) in &branches[..tried] {
            clear_bit(&mut self.forbidden[i], x % self.primes[i]);
        }
        found
    }
}
