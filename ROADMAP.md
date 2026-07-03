# puremp roadmap

`puremp` is a pure-Rust, MIT-licensed, arbitrary-precision arithmetic library —
integers, rationals, and MPFR-class floating point — with no foreign-code
dependencies, usable as a Rust crate, a C library, and a command-line tool.

This document is the design record and the milestone plan. It is meant to stay
accurate: when a milestone lands, its algorithms move from "planned" to the
"Status" section, and the milestone entry is checked off.

---

## 1. Vision & scope

Deliver everything a user expects from a GMP + MPFR stack, in safe, portable
Rust:

- **`Nat`** — arbitrary-precision naturals (unsigned).
- **`Int`** — arbitrary-precision signed integers (GMP `mpz` surface).
- **`Rational`** — exact `p/q` fractions in lowest terms (GMP `mpq` surface).
- **`Float`** — binary floating point with caller-chosen precision and directed
  rounding, targeting **MPFR-class correct rounding**, including the
  transcendental functions.

Plus the surrounding surfaces: number theory (GCD, modular arithmetic, modular
exponentiation, primality), string/radix I/O, a C ABI, and a CLI calculator.

## 2. Design decisions

These were settled up front and constrain every milestone:

| Decision | Choice | Consequence |
|---|---|---|
| Portability | `#![no_std]` + `alloc` | Heap-backed types; runs on bare metal with an allocator. No OS assumptions in the core. |
| Safety | Pure safe Rust; `unsafe_code = "deny"` | The **only** `unsafe` is the `ffi` module (opts back in locally). No inline asm, no intrinsics. |
| Limb | 64-bit limbs, `u128` for products/carries | Portable and fast on 64-bit; correct (if slower) on 32-bit via compiler `u128` lowering. Validated on `thumbv7em-none-eabi` in CI. |
| Constant-time | **Not** a goal | Free to use data-dependent branches, early-outs, and fast paths. A constant-time modular layer may be added later but is out of scope for the general API. (For constant-time crypto today, see the sibling `purecrypto` crate.) |
| Floats | MPFR-class correct rounding | Every op takes an explicit output precision and `RoundingMode` and returns the once-rounded result. Larger than GMP `mpf`, which does not guarantee correct rounding. |
| Packaging | One crate + C ABI + CLI | Feature-gated layers (`int` → `rational`/`float`), `ffi` for the C library, `cli` for the `puremp` binary. |

## 3. License & provenance (clean-room)

`puremp` is MIT-licensed and is a **clean-room** implementation. GMP and MPFR
are LGPL; **their source is never consulted**. Algorithms come from the open
literature, which describes them independently of any implementation:

- D. E. Knuth, *The Art of Computer Programming*, Vol. 2 (esp. Algorithm D,
  long division).
- R. Brent & P. Zimmermann, *Modern Computer Arithmetic* (Cambridge, 2010) —
  the primary reference for sub-quadratic multiplication/division, GCD, base
  conversion, and floating-point algorithms. Freely available from the authors.
- A. Menezes, P. van Oorschot & S. Vanstone, *Handbook of Applied Cryptography*
  (modular arithmetic, Montgomery/Barrett, primality).
- J.-M. Muller, *Elementary Functions: Algorithms and Implementation*
  (argument reduction and evaluation of the transcendental functions).
- Primary papers: Karatsuba; Toom–Cook; Schönhage–Strassen; Burnikel–Ziegler
  (recursive division); Möller–Granlund (division by invariant integers);
  Stehlé–Zimmermann / the MPFR papers (for *algorithms*, not source).

Public numeric answers (e.g. `n!`, `2^k`, known GCDs) are used as test oracles.
No LGPL code, and no code derived from reading LGPL code, enters the tree.

## 4. Architecture

Bottom-up layers; each builds only on the ones below it.

```
                        ┌─────────────────────────────────────┐
   ffi (C ABI)  ◀───────┤  Int · Rational · Float public API   ├──────▶  cli (puremp)
                        └─────────────────────────────────────┘
                          float ──▶ ┐
                          rational ─┤──▶  int  ──▶  nat  ──▶  limb primitives
                                    ┘
```

- **`limb`** — `adc`/`sbb`/`mac`, the machine-word carry algebra. Pure `const fn`.
- **`nat`** — unsigned magnitude in normalized little-endian limbs; the home of
  the hard algorithms (mul, div, GCD, modular, roots, base conversion).
- **`int`** — sign + `Nat`, canonicalized.
- **`rational`** — reduced `Int`/`Nat` pair.
- **`float`** — sign + significand + exponent + precision, correctly rounded.
- **`ffi`** — opaque-handle C ABI (`include/puremp.h`); the sole `unsafe` island.
- **`bin/puremp`** — REPL calculator over `Int`.

## 5. Current status (scaffold)

Implemented and tested today (the foundation; correctness-first, not yet tuned):

- `limb`: `adc`, `sbb`, `mac`.
- `Nat`: construct/normalize, compare, `add`, `checked_sub`, schoolbook `mul`,
  `shl`/`shr`, `bit`/`bit_len`/`trailing_zeros`, **binary (Stein) GCD**,
  **bit-at-a-time `div_rem`**, decimal `FromStr`/`Display`, `LowerHex`.
- `Int`: full sign handling, `add`/`sub`/`mul`, `pow` (square-and-multiply),
  truncated `div_rem`, ordering, decimal I/O, operator overloads.
- `Rational`: construction with GCD reduction, `add`/`sub`/`mul`/`div`/`recip`,
  canonical sign, ordering, `Display`.
- `Float`: representation, `RoundingMode`, exact integer conversion, accessors,
  exact `Display` (`±significand·2^exp`). **Arithmetic is not yet implemented.**
- C ABI over `Int`; `puremp` REPL (`+ - * / % **`, variables).

Everything above is deliberately the **simple, obviously-correct** version. The
milestones below replace the quadratic/bit-at-a-time cores with the fast
algorithms and fill in the float layer.

## 6. Milestones

Each milestone is a shippable increment. Ordering favors unblocking the widest
set of downstream features first.

### M1 — Integer core hardening
- Radix I/O for all bases 2–36 (parse + format), sub-quadratic base conversion
  (divide-and-conquer via `10^(2^k)` splitting) for large decimal.
- Bitwise ops on `Int` with two's-complement semantics (`and`/`or`/`xor`/`not`),
  bit set/clear/test, population count.
- `from`/`to` all primitive integer types; `TryFrom` with range checks.
- Fast paths for single-limb operands throughout.
- Exhaustive property tests (see §7) and edge-case coverage.

### M2 — Fast multiplication
- Karatsuba (2-way), then Toom-3 (and Toom-4) with tuned crossover thresholds.
- Schönhage–Strassen / an NTT-based FFT multiply for very large operands.
- Squaring fast path.
- A `benches/` threshold-tuning harness; thresholds captured as consts with a
  documented measurement method.

### M3 — Fast division & roots
- Knuth Algorithm D (schoolbook long division with 3/2 limb quotient estimate),
  replacing the bit-at-a-time core.
- Division by invariant/precomputed reciprocal (Möller–Granlund).
- Burnikel–Ziegler recursive division for large quotients.
- Exact division; `divexact`; Euclidean and floored `div_rem` variants on `Int`.
- Integer square root (`isqrt`) and `k`th root, `is_perfect_power`.

### M4 — GCD & modular arithmetic
- Subquadratic GCD: Lehmer's GCD, then half-GCD (HGCD).
- Extended GCD; modular inverse.
- `mod`, Barrett and Montgomery reduction.
- Modular exponentiation (sliding-window; Montgomery ladder available).
- Jacobi/Legendre symbols, modular square root (Tonelli–Shanks).

### M5 — Number theory
- Miller–Rabin and Baillie–PSW primality; `next_prime`/`prev_prime`.
- `factorial`, `binomial`, `multinomial`, primorial, `fibonacci`, `lucas`.
- `RandomState` trait (caller-supplied entropy — no foreign RNG dep) and random
  `n`-bit / bounded generators, feeding primality and testing.

### M6 — Rationals to full `mpq` surface
- Operators and assign-ops, comparison against integers/floats.
- `from_float` / `to_float` with rounding; continued-fraction best rational
  approximation within a denominator bound.
- Canonicalize-on-demand API for hot loops that batch then reduce.

### M7 — Floating point core (MPFR-class)
- Significand normalization; a `RoundResult` carrying the ternary
  (inexact/toward) flag like MPFR.
- Correctly-rounded `add`/`sub`/`mul`/`div`/`sqrt` in all five rounding modes,
  with correct handling of the round/sticky bits (Ziv's rounding loop where
  needed).
- Special values: signed zeros, ±∞, NaN, and their IEEE interactions.
- `from`/`to` `f32`/`f64` (correctly rounded both ways); decimal string I/O with
  correct rounding (Steele–White / Ryū-style shortest, plus fixed/scientific).
- Exponent-range handling and overflow/underflow semantics.

### M8 — Floating point transcendentals
- Constants: π, log 2, e, Euler–Mascheroni (cached per precision).
- `exp`, `log`, `expm1`, `log1p`, `pow`.
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` (with proper argument
  reduction), and the hyperbolic set.
- Ziv's strategy for correct rounding: evaluate at guard precision, detect the
  hard-to-round cases, and retry with more precision.

### M9 — API polish & interop
- `core::ops` coverage (value/ref/assign) across all types; `Sum`/`Product`.
- `From`/`TryFrom`, `FromStr`, `Display`/`LowerHex`/`UpperHex`/`Binary`/`Octal`.
- Optional `serde` support (implemented in-house, no derive dependency) behind a
  feature; optional `rand`-trait glue behind a feature.
- Full C ABI over `Rational` and `Float`; header kept in lockstep, generated
  header option (cbindgen) as a dev convenience.
- Expand the CLI: rationals & floats, precision/rounding settings, `!`
  factorial, number-theory commands, non-decimal radices.

### M10 — Performance & release
- Scratch-buffer/arena reuse to cut allocations on hot paths.
- Optional small-integer inline storage (avoid heap for values that fit a limb
  or two).
- Benchmark suite vs. reference tools; publish results.
- Documentation pass, `1.0` API review, semver commitment.

## 7. Testing strategy

- **Known-answer tests** against independently computed values (factorials,
  powers, published GCDs, digits of π/e for floats).
- **Property tests**: algebraic laws (commutativity, associativity,
  distributivity), round-trip laws (`parse∘format`, `shl∘shr`,
  `(a/b)*b + a%b == a`, `x == q·d + r ∧ r < d`), and cross-checks between a fast
  algorithm and its simple reference implementation kept in the tree (e.g. new
  fast `mul` vs. schoolbook `mul`). The self-check strategy keeps the "no
  foreign code" rule while still catching regressions.
- **Differential testing** between algorithm tiers (schoolbook ↔ Karatsuba ↔
  FFT must agree; bit-at-a-time div ↔ Knuth-D ↔ Burnikel–Ziegler must agree).
- **Fuzzing** (`fuzz/`, libFuzzer via `cargo-fuzz`): parser round-trips and the
  arithmetic invariants above on random operands. Runs out-of-band in CI.
- **Float correctness**: hard-to-round test vectors; check the ternary flag and
  all five rounding modes; compare `f64` conversions bit-for-bit with the
  hardware where the value is representable.
- **C ABI smoke tests** (`tests/ffi_smoke.c`) compiled and run in CI against the
  static library.

## 8. Non-goals (for now)

- Constant-time / side-channel resistance across the general API (see §2).
- Interval arithmetic, complex numbers, matrices/polynomials — possible future
  crates, not this one.
- Drop-in GMP/MPFR C header compatibility (we ship our own, cleaner C ABI).
- Multi-threaded internals (single-threaded algorithms; the types are `Send`).
```
