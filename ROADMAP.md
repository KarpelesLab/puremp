# puremp roadmap

`puremp` is a pure-Rust, MIT-licensed, arbitrary-precision arithmetic library.
Its **contract core** is two exact numeric types — arbitrary-precision **signed
integers** and **exact rationals** — designed to be the numeric foundation for
symbolic-computation, computer-algebra, and constraint-solving software, where
they sit on the hot path and where sign/rounding conventions must be exact and
predictable.

Floating point is **not** part of that core contract (see §1 and §9): it is an
optional layer that downstream code can build on top of the integer type. This
crate does ship an optional `Float` type as a convenience, but it is separable
and never a prerequisite for the integer/rational guarantees.

Type-name mapping: this crate calls the signed integer **`Int`** and its
unsigned magnitude backing **`Nat`**; the specification below refers to the
signed type as `Integer`. They are the same thing — names differ, features do
not. `Rational` matches.

This document is the design record, the milestone plan, and — in §7 — an
explicit checklist mapping every required feature to a milestone or to shipped
code. It is meant to stay accurate: when a milestone lands, its items move to
the "Status" section and the checklist row flips to ✅.

---

## 1. Vision & scope

**In scope (the contract):**

- **`Int`** (spec `Integer`) — a signed integer of unbounded magnitude.
- **`Rational`** — an exact rational, always kept in canonical form.

Both are exact: every operation is exact, with rounding **only** where a method
explicitly converts to a bounded type (`to_f64`, `to_i64`, decimal display with
a precision). The types target use as a hot-path numeric core, so the
performance-critical shapes in §2 are part of the contract, not afterthoughts.

**Out of scope of the core contract** (may be layered on top by downstream
crates, or via this crate's optional `float` feature):

- Arbitrary-precision or fixed floating point.
- Specialized numeral forms (e.g. dyadic `n·2^-k`).

The crate remains usable as a Rust library, a C library (the `ffi` feature; see
`include/puremp.h`), and a CLI calculator (the `cli` feature).

## 2. Ground rules & the hard parts

### 2.0 Ground rules

- **Zero runtime dependencies.** Standard library (in fact `core` + `alloc`)
  only. No native code, no build scripts linking C, no third-party runtime
  crates. A **dev-only** reference bignum is permitted *in the test harness* for
  differential/fuzz cross-checks (§8); it is never a runtime or published
  dependency.
- **Exactness.** No hidden rounding; see §1.
- **Canonical `Rational`.** Every value satisfies `den > 0`,
  `gcd(num, den) == 1`, and integers have `den == 1`. This invariant is
  established by every constructor and preserved by every operation.
- **Deterministic.** No global mutable state; results depend only on inputs.
  `Hash` is stable within a build and consistent with `Eq`.
- **Portability & safety.** `#![no_std]` + `alloc`; pure safe Rust
  (`unsafe_code = "deny"`), the sole `unsafe` island being the opt-in `ffi`
  module. 64-bit limbs with `u128` intermediates; validated on 32-bit
  `thumbv7em-none-eabi` in CI.

### 2.1 Small-value inlining (performance — mandatory)

Most values fit in a machine word. `Int` **must** be a tagged representation
that stores those inline and only heap-allocates limbs on overflow:

```rust
enum Repr { Small(i64), Large { sign: Sign, mag: Box<[u64]> } }
```

Every operation takes the fast path when operands are `Small`, and **demotes**
back to `Small` whenever a result again fits the inline word. `i64::MIN`/`MAX`
are explicit boundary cases (note `-i64::MIN` overflows the inline word and must
promote). This is a contractual performance property, cross-checked against the
all-`Large` path in tests (§8).

> Current status: the shipped `0.1.0` `Int` is `Sign + Nat` (always heap). The
> tagged `Small/Large` representation is milestone **M1** — an internal change
> behind the same public API.

### 2.2 Three division/remainder conventions (correctness)

For `a / b`, kept distinct and each satisfying `a == q·b + r` exactly:

| Convention  | Quotient rounds toward | Remainder range/sign        | Std analogue |
|-------------|------------------------|-----------------------------|--------------|
| **Truncated** | zero                 | sign of dividend `a`        | `/`, `%`     |
| **Euclidean** | so that `r ≥ 0`      | `0 ≤ r < |b|`               | `div_euclid`, `rem_euclid` |
| **Floored**   | −∞                   | sign of divisor `b`         | `div_floor`  |

Truncated and Euclidean are **required**; floored is **recommended** (cheap once
the others exist). Each provides a combined `div_rem_*` returning both without
recomputation. Plus **exact division** `div_exact` (precondition `d | self`,
skips remainder handling) and `divides`.

### 2.3 Power-of-two fast paths (performance)

Must bypass the general multiply/divide routines: `mul_2k(k)` (`<< k`),
`div_2k_trunc(k)` (truncated `/2^k`), `mod_2k(k)` (low `k` bits, non-negative),
`is_power_of_two() -> Option<u32>`, `next_power_of_two()`, `prev_power_of_two()`,
`trailing_zeros()`.

### 2.4 Width-aware two's-complement bitwise ops (correctness)

`bitand`/`bitor`/`bitxor`/`bitnot` operate on the **two's-complement**
representation (distinct from sign-magnitude bit twiddling). Because negatives
have infinitely many leading sign bits, **`bitnot` takes an explicit bit-width**
(`bitnot(width)`), and the semantics of all four are documented precisely and
verified against a width-`w` truth table (§8).

### 2.5 Public limb & bit access (interop)

Cheap, public: `bit(i) -> bool`, `limbs() -> &[u64]` (little-endian magnitude),
`least_significant_limb() -> u64`, and `from_limbs(sign, &[u64])`.

> Current status: `Nat` stores limbs but keeps them private. Exposing the slice
> and the accessors above is part of **M1**.

### 2.6 Fused multiply-accumulate (performance)

`addmul(&mut self, a, b)` (`self += a·b`) and `submul` (`self -= a·b`) that avoid
the temporary; the `Small` path uses widening 128-bit intermediates.

## 3. License & provenance (clean-room)

`puremp` is MIT-licensed and **clean-room**. GMP and MPFR are LGPL; **their
source is never consulted**. Algorithms come from the open literature:

- D. E. Knuth, *TAOCP* Vol. 2 §4.3 — schoolbook add/sub/mul on little-endian
  limbs, and **Algorithm D** for multiprecision division.
- R. Brent & P. Zimmermann, *Modern Computer Arithmetic* — sub-quadratic
  multiply/divide, subquadratic GCD, base conversion (freely available).
- A. Menezes, P. van Oorschot & S. Vanstone, *Handbook of Applied Cryptography*.
- Primary papers: Karatsuba; Toom–Cook; Burnikel–Ziegler (recursive division);
  Möller–Granlund (division by invariant integers).

Public numeric answers (`n!`, `2^k`, published GCDs) and a **dev-only** trusted
bignum in the test harness (§8) are the correctness oracles. No LGPL code — and
no code derived from reading LGPL code — enters the tree.

## 4. Architecture

Bottom-up layers; each builds only on the ones below it.

```
                        ┌──────────────────────────────────────┐
   ffi (C ABI)  ◀───────┤   Int (spec Integer) · Rational       ├──────▶  cli (puremp)
                        └──────────────────────────────────────┘
        optional:  float  ──────────────────▲ (layered on Int; not in the core contract)
                                            │
   Int = Repr::Small(i64) | Large{ sign, mag }   ──▶  Nat (Large magnitude)  ──▶  limb primitives
```

- **`limb`** — `adc`/`sbb`/`mac` machine-word carry algebra (pure `const fn`).
- **`Nat`** — unsigned magnitude in normalized little-endian limbs; home of the
  hard algorithms (mul, div, GCD, roots, base conversion). Backs `Large`.
- **`Int`** — the tagged `Small/Large` signed integer; spec `Integer`.
- **`Rational`** — reduced `Int`/`Int` pair (canonical).
- **`float`** *(optional)* — separable convenience layer on `Int`.
- **`ffi`** — opaque-handle C ABI; the only `unsafe`.
- **`bin/puremp`** — REPL calculator.

## 5. Current status

The **integer/rational core contract (§1–§2) is fully implemented and tested**,
the sub-quadratic algorithm ladder is in place, and the optional float layer is
complete through the transcendentals. The only open items are further
performance tuning and a pre-`1.0` review (§6).

- `limb`: `adc`, `sbb`, `mac`.
- `Nat`: normalize/compare, `add`, `checked_sub`, `mul` (schoolbook → Karatsuba
  → Toom-3 → NTT) and `square`, `div_rem` (single-limb, Knuth Algorithm D,
  Burnikel–Ziegler), `shl`/`shr`, `bit`/`bit_len`/`trailing_zeros`/`low_bits`,
  GCD (binary → Lehmer), `pow`, `isqrt`, `nth_root_floor`, byte/limb interop,
  random generation, sub-quadratic radix + decimal I/O, `Hash`.
- `Int`: tagged `Small{neg,mag:u64} | Large{sign,mag}` inline representation with
  demotion; all primitive-int `From`; `ZERO`/`ONE`/`MINUS_ONE`; predicates,
  `signum`, `abs`; `add`/`sub`/`mul`/`pow`; fused `addmul`/`submul`; **all three
  division conventions** + `div_exact`/`divides`; `gcd`/`lcm`/`extended_gcd`;
  power-of-two ops; two's-complement `bitand`/`bitor`/`bitxor`/`bitnot(width)`;
  `bit`/`limbs`/`least_significant_limb`/`from_limbs`; bounded conversions;
  radix + decimal I/O; `Hash`; value/ref/`i64` operator + `*Assign` overloads.
- `Rational`: `const ZERO`/`ONE`/`MINUS_ONE`; `new`/`checked_new`/`from_integer`/
  `power_of_two`; `From`/`FromStr` (`"3"`, `"-3/4"`, `"1.5"`); predicates +
  `signum`; `neg`/`abs`/`recip`/`pow`; arithmetic + fused `addmul`/`submul`;
  `floor`/`ceil`/`trunc`/`to_integer`; `div_floor`/`div_trunc`/`rem_euclid`;
  bounded conversions; `write_decimal`; `Hash`; operators.
- `Float` *(optional)*: IEEE special values; correctly-rounded
  `add`/`sub`/`mul`/`div`/`sqrt` (with ternary flag) in all five `RoundingMode`s;
  `f64`/`f32`/rational/decimal conversions; transcendentals (`pi`/`e`/`ln2`,
  `exp`/`ln`/`sin`/`cos`/`tan`/`atan`) via Ziv's strategy.
- Free `u_gcd`/`u64_gcd`; in-house `RandomSource` (+ optional `rand`/`serde`);
  C ABI over `Int`/`Rational`/`Float`; a `puremp` REPL with rationals, functions,
  and radices.

## 6. Milestones

**All milestones M1–M9 are functionally complete.** The only remaining ▫ items
are two deliberately-deferred engineering tasks — half-GCD and an
allocation-reducing rewrite — each documented below with its rationale (both are
constant-factor/marginal optimizations of already-working, well-tested code, and
carry real regression risk relative to their benefit). ✅ = done, ▫ = deferred.

### M1 — Representation, inlining & core `Int` surface ✅
Tagged `Small/Large` inline representation with demotion; all primitive-int
`From`; consts; predicates/`signum`/`abs`; fused `addmul`/`submul`; public
`bit`/`limbs`/`least_significant_limb`/`from_limbs`; `Hash`; value/ref/`i64`
operator + `*Assign` overloads; `bit_len`/`log2_floor`.

### M2 — Division & remainder (all conventions) ✅
Truncated, Euclidean, and floored `div_*`/`rem_*`/`div_rem_*`; `div_exact`;
`divides`. Plain methods panic on a zero divisor; the `Nat`/checked layer returns
`Option`.

### M3 — Power-of-two & two's-complement bitwise ✅
`mul_2k`/`div_2k_trunc`/`mod_2k`/`is_power_of_two`/`next_power_of_two`/
`prev_power_of_two`/`trailing_zeros`; width-aware two's-complement
`bitand`/`bitor`/`bitxor`/`bitnot(width)`, truth-table tested.

### M4 — Number theory & roots ✅
`gcd`/`lcm`/`extended_gcd`; free `u_gcd`/`u64_gcd`; `sqrt_exact`/`nth_root_exact`;
`modpow` (Montgomery for odd moduli, Barrett for even), `modinv`, Miller–Rabin
`is_probable_prime`, deterministic Baillie–PSW `is_prime_bpsw`, and
`next_prime`/`prev_prime`.

### M5 — Radix & string I/O, bounded conversions ✅
`from_str_radix`/`write_radix`; decimal `FromStr`/`Display`; `fits_i64`/
`fits_u64`/`to_i64`/`to_u64`/`to_f64`; sub-quadratic (divide-and-conquer) radix
conversion.

### M6 — `Rational` full surface ✅
Constructors/consts/`power_of_two`; `From`/`FromStr` incl. decimals; predicates/
`signum`; `recip`/`abs`/`pow`; arithmetic + fused ops; `floor`/`ceil`/`trunc`/
`to_integer`; integer division of rationals; bounded conversions; `write_decimal`;
`Hash`; operators.

### M7 — Fast algorithms (behind the same API) ✅
- ✅ Multiplication ladder: schoolbook → Karatsuba → Toom-3 → Toom-4 → NTT over
  the Goldilocks field, with an adaptive digit width so a single prime covers
  any practical size (no multi-prime CRT needed), plus a dedicated squaring
  path — all differentially tested.
- ✅ Division: Knuth Algorithm D, then Burnikel–Ziegler recursive division above
  64 limbs; `Reciprocal` (Möller–Granlund / Barrett) for division by an
  invariant modulus.
- ✅ Sub-quadratic GCD: Lehmer's algorithm above 16 limbs.
- ✅ Measured crossover thresholds (`measure_mul_crossovers`), retuned from the
  results (Karatsuba 160, Toom-3 256, Toom-4 448, NTT 6000); benchmark harness.
- ▫ Half-GCD (HGCD): **deliberately deferred.** A correct recursive HGCD is one
  of the most error-prone algorithms in the field, and GCD is on the `Rational`
  hot path, so shipping a hastily-written version would risk a correctness bug
  for only a marginal gain over the (already sub-quadratic, well-tested) Lehmer
  GCD. To be revisited against a rigorous reference.

### M8 — Optional floating-point layer (separable) ✅
Outside the core contract (§1); behind the `float` feature.
- ✅ Tagged representation with the IEEE special values (signed zeros, ±∞, NaN).
- ✅ Correctly-rounded `add`/`sub`/`mul`/`div`/`sqrt` in all five rounding modes,
  with the MPFR ternary flag (`*_ternary`).
- ✅ `from_f64`/`from_f32`/`to_f64`/`to_f32`, `from_rational`/`to_rational`,
  `FromStr`, `to_decimal_string`, and an exact `to`/`from_exact_string` codec.
- ✅ Elementary functions via Ziv's strategy: `pi`/`e`/`ln2`, `exp`/`ln`/`pow`,
  `sin`/`cos`/`tan`, `asin`/`acos`/`atan`/`atan2`, `sinh`/`cosh`/`tanh`, and
  `asinh`/`acosh`/`atanh`.
- ✅ Shortest round-tripping decimal (`to_shortest_string`) and an exact codec.
- ✅ Correct-rounding tests: `div`/`sqrt` checked against the exact rational
  rounded reference and higher-precision recomputation, plus directed-rounding
  bracketing.
- ▫ Further: exhaustive hard-to-round vectors and formal proofs (research-grade).

### M9 — Polish, interop & release
- ✅ `core::ops` coverage (value/ref/`i64`/assign) for `Int`; value/ref/assign for
  `Rational`; `Sum`/`Product` for both.
- ✅ Optional in-house `serde` (no `serde_derive`); optional `rand` glue
  (in-house `RandomSource` + `rand_core` bridge) with random `Nat`/`Int`.
- ✅ C ABI over `Int`, `Rational`, and `Float`; a `puremp` REPL with exact
  rationals, functions, and radices.
- ✅ Benchmark harness; crate-level doctest; consistent `From` conversions.
- ✅ Pre-`1.0` API review (naming/consistency audit; the surface is stable and
  documented). Semver commitment follows the first `1.0` tag.
- ▫ Allocation-reducing scratch buffers: **deferred.** A meaningful reduction
  needs in-place limb operations threaded through the recursive multiply/divide
  code — a broad refactor with real regression risk for a constant-factor gain,
  best done with a benchmark-guarded rewrite rather than piecemeal.

## 7. Specification coverage checklist

Every required feature from the spec, mapped to shipped code or a milestone.
(✅ shipped · ▫ planned)

### `Int` (spec `Integer`)

| Feature | Where |
|---|---|
| Small-value inlining `Small/Large`, demotion, fast paths (§2.1) | ✅ M1 |
| `From<i8..i128,u8..u64,usize>`; `FromStr` decimal; `from_str_radix`; `from_limbs` | ✅ M1/M5 |
| `ZERO`/`ONE`/`MINUS_ONE`; `Default=0` | ✅ M1 |
| Predicates `is_zero/one/minus_one/positive/negative/even/odd`; `signum` | ✅ M1 |
| `abs`, `pow(u32)`; `Add/Sub/Mul/Neg` + `*Assign` (value/ref/`i64`) | ✅ M1 |
| Fused `addmul`/`submul` (§2.6) | ✅ M1 |
| `div/rem/div_rem_trunc` | ✅ M2 |
| `div/rem/div_rem_euclid`; `div_floor`; `div_exact`; `divides` (§2.2) | ✅ M2 |
| `gcd` | ✅ M4 |
| `lcm`, `extended_gcd` | ✅ M4 |
| `mul_2k`, `div_2k_trunc`, `mod_2k`, `is_power_of_two`, `next/prev_power_of_two`, `trailing_zeros` (§2.3) | ✅ M3 |
| `sqrt_exact`, `nth_root_exact`; `bit_len`, `log2_floor` | ✅ M4/M1 |
| Two's-complement `bitand/or/xor/not(width)` (§2.4) | ✅ M3 |
| `bit`, `limbs() -> &[u64]`, `least_significant_limb` (§2.5) | ✅ M1 |
| `fits_i64/u64`, `to_i64/u64`, `to_f64` | ✅ M5 |
| `Display` (decimal), `Hash`, `write_radix`; `Clone/Eq/Ord/Debug` | ✅ M1/M5 |

### `Rational`

| Feature | Where |
|---|---|
| `new` (panic on `den==0`), `checked_new`, `from_integer`, `numerator`/`denominator` | ✅ M6 |
| `ZERO`/`ONE`/`MINUS_ONE`, `power_of_two(i32)` | ✅ M6 |
| `From<i64>`, `From<Int>`, `FromStr` (`"3"`, `"-3/4"`, `"1.5"`) | ✅ M6 |
| Predicates + `is_integer` + `signum` | ✅ M6 |
| `Add/Sub/Mul/Div/Neg` + `*Assign`; `recip`, `abs`, `pow(i32)`, `addmul`/`submul` | ✅ M6 |
| `floor`, `ceil`, `trunc`, `to_integer` | ✅ M6 |
| `div_floor`/`div_trunc` (→ `Int`), `rem_euclid` (→ `Rational`) | ✅ M6 |
| `fits_i64`, `to_i64`, `to_f64` | ✅ M6 |
| `Display`, `Hash`, `write_decimal(precision, truncate)`; `Clone/Eq/Ord/Debug/Default` | ✅ M6 |

### Free helpers & canonical invariant

| Feature | Where |
|---|---|
| `u_gcd(u32,u32)`, `u64_gcd(u64,u64)` | ✅ M4 |
| Canonical `Rational` maintained by every op (§2.0) | ✅ (invariant tested) |

## 8. Correctness bar & testing

Aligned to the spec's test list; property tests run randomized, many iterations:

- **Division identity** for all three conventions: `a == q·b + r` with `r` in the
  documented range/sign, for all `a`, all `b ≠ 0`; Euclidean `0 ≤ r < |b|`.
- `gcd(a,b)·lcm(a,b) == |a·b|`; `extended_gcd` returns `g == a·x + b·y`.
- **Round-trips:** `from_str(x.to_string()) == x`;
  `from_limbs(x.sign, x.limbs()) == x`; `from_str_radix(write_radix(x,r),r) == x`.
- **Canonical form:** after every `Rational` op, `gcd(num,den)==1 && den>0`.
- **Small/Large agreement:** ops crossing the inline-word boundary match the
  all-`Large` computation (guards the §2.1 fast paths).
- **Bit tricks vs. general path:** `mul_2k(k) == self·2^k`,
  `mod_2k(k) == rem_euclid(2^k)`, `bitnot(w)` matches the width-`w` truth table.
- **Edge cases:** `0`, `±1`, `i64::MIN`/`MAX` (inline boundary), one-limb-wide vs.
  exactly-at-a-limb-boundary values, deep rational cancellation, remainder sign
  around negative operands.
- **Fuzzing** (`fuzz/`): random operation sequences cross-checked against a
  **dev-only** trusted arbitrary-precision reference (test harness only, never a
  runtime dependency) in addition to the invariant checks; plus the parser
  round-trips already present.
- **Differential** between algorithm tiers (schoolbook ↔ Karatsuba ↔ FFT;
  bit-at-a-time ↔ Knuth-D ↔ Burnikel–Ziegler must agree).
- **C ABI smoke tests** (`tests/ffi_smoke.c`) compiled and run in CI.

## 8a. Extended numeric types ✅

Beyond the core `Int`/`Rational` contract and the optional `Float`, the crate
ships a small family of exact/convenience types layered on the same foundation:

- **`Dyadic`** (`dyadic` feature) — exact dyadic rationals `n·2^-k` (denominator
  a power of two): exact `+`/`-`/`*`/`pow`/`mul_2k`, a terminating-decimal
  `Display`, and conversions to/from `Rational`/`Float`.
- **`InfRational`** (`rational` feature) — `Rational` extended with `±∞`/`NaN`
  and IEEE-style arithmetic, exact on the finite part.
- **`FixedFloat`** (`float` feature) — a fixed-precision wrapper over `Float`
  that bakes in the precision and rounding mode so it supports the ordinary
  `+ - * /` operators and method-style transcendentals.

## 9. Non-goals

- Floating point as part of the core contract — it is a separable optional layer
  (the `float` feature), never a prerequisite for the integer/rational surface.
- Constant-time / side-channel resistance across the general API (for
  constant-time crypto, see the sibling `purecrypto` crate).
- Interval arithmetic, complex numbers, matrices/polynomials — possible future
  crates, not this one.
- Drop-in GMP/MPFR C header compatibility (we ship our own cleaner C ABI).
```
