# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from `1.0.0`
onward (pre-`1.0`, minor versions may contain breaking changes).

## [Unreleased]

## [0.1.6](https://github.com/KarpelesLab/puremp/compare/v0.1.5...v0.1.6) - 2026-07-04

### Other

- Nat-native Lehmer cofactor combination — gcd ~28% faster
- direct add-into-place recombination for Toom-3/Toom-4 (~10-24%)
- low-half multiply in Montgomery reduction (~12% faster modpow)
- direct add-into-place recombination in Karatsuba multiply
- retune multiplication crossovers — division ~24% faster
- division-free Goldilocks reduction + fix NTT crossover threshold
- update known-optimizations (isqrt SqrtRem and BZ padding now done)
- k-ary windowed modular exponentiation (~14% faster modpow)
- split-loop add/sub with a bulk-copy tail
- power-of-two block padding for Burnikel–Ziegler division; tune threshold
- recursive (Karatsuba) integer square root — isqrt ~8.8× faster
- add CI / crates.io / docs.rs / license badges to the README
- note isqrt SqrtRem and BZ block-padding as future optimizations
- drop redundant full-width clones in the division inner loops
- bounds-check-free schoolbook multiply/square inner loops
- route Nat::from_str through the fast parser; share to_string power ladder
- sub-quadratic base-N parsing (from_string ~29× faster)
- Add fuzz targets for Decimal and Poly
- Property-based hardening for the extended types; fix Algebraic::signum on zero
- serde + num-traits coverage for the extended numeric types
- drop the pre-1.0 status blockquote from the README
- Remove ROADMAP.md; fold provenance/design into the README
- Expose real-root finding on Poly<Rational>; share Sturm code with Algebraic

## [0.1.5](https://github.com/KarpelesLab/puremp/compare/v0.1.4...v0.1.5) - 2026-07-03

### Other

- document Quadratic and Algebraic numbers
- Algebraic numbers (2/4): general real Algebraic via Sturm + Faddeev–LeVerrier
- Algebraic numbers (1/4): Quadratic — exact field ℚ(√d)
- cover the full expanded type family and number-theory toolkit
- decimal Display, precision, and scientific {:e}/{:E}
- TryFrom for primitive integers; to_i128/to_u128; Error::Overflow
- Add Matrix<T>: generic matrices with exact linear algebra
- Add Interval: interval arithmetic with outward rounding
- Add Poly<T>: generic univariate polynomials
- continued fractions and best rational approximation
- Add num-traits bridge; Int Div/Rem, Nat/Rational operators
- Add Complex<T>: generic complex numbers
- Add Decimal: arbitrary-precision base-10 floating point
- Add ModInt: modular integers (ℤ/mℤ) with operators
- Number theory: integer factorization and random_prime
- Number theory: Jacobi/Legendre, sqrt_mod (Tonelli–Shanks), CRT
- Number theory: combinatorics (factorial/binomial/multinomial/fibonacci/lucas)

## [0.1.4](https://github.com/KarpelesLab/puremp/compare/v0.1.3...v0.1.4) - 2026-07-03

### Other

- use the n·2^-k convention (fix inverted exponent sign)

## [0.1.3](https://github.com/KarpelesLab/puremp/compare/v0.1.2...v0.1.3) - 2026-07-03

### Other

- document the extended numeric types (Dyadic/FixedFloat/InfRational)
- Add InfRational: extended rationals with ±∞ and NaN
- Add FixedFloat: fixed-precision float wrapper (mpfx-style)
- Add Dyadic: exact dyadic rationals (n·2^-k)
- pre-1.0 API review; document deferred HGCD and scratch-buffer work
- measure and tune multiplication crossover thresholds
- public Reciprocal (Möller–Granlund division by an invariant)
- adaptive-width NTT (lift the single-prime size cap)
- Toom-4 multiplication
- deterministic Baillie–PSW primality test
- Barrett reduction for even-modulus modpow

## [0.1.2](https://github.com/KarpelesLab/puremp/compare/v0.1.1...v0.1.2) - 2026-07-03

### Other

- correct-rounding verification tests for div and sqrt
- prev_prime
- shortest round-tripping decimal + fix Float::div guard sizing
- Montgomery-reduction modpow and next_prime
- modular arithmetic and primality (modpow, modinv, Miller–Rabin)
- add inverse hyperbolics (asinh/acosh/atanh)

## [0.1.1](https://github.com/KarpelesLab/puremp/compare/v0.1.0...v0.1.1) - 2026-07-03

### Other

- Update ROADMAP/README/CHANGELOG: M1–M8 complete, M9 nearly so
- NTT (FFT) multiplication for very large operands
- Burnikel–Ziegler recursive division
- Lehmer's subquadratic GCD
- Toom-3 multiplication
- dedicated squaring fast path
- sub-quadratic (divide-and-conquer) radix conversion
- expand the benchmark harness
- expand the CLI — exact rationals, functions, and radices
- C ABI over Float
- serde support (hand-written, no serde_derive)
- randomness — in-house RandomSource trait + rand_core bridge
- Float transcendentals via Ziv's strategy
- Fix clippy approx_constant in float test (use core::f64::consts::PI)
- Float special values, ternary flag, and f64/f32/rational/decimal I/O
- expand the C ABI over Rational
- M9 (partial) + docs: Sum/Product, ROADMAP/README/CHANGELOG status
- Implement M8 (core): correctly-rounded Float arithmetic
- Implement M7: Karatsuba multiplication and Knuth Algorithm D division
- Implement M6: full Rational surface
- Implement M1–M4 for Int: inline representation + full integer surface
- Re-enable float in the default feature set
- Expand ROADMAP to cover the Integer/Rational spec; make float opt-in

### Added (fast algorithms, float, and interop)

- **Fast multiplication (M7):** a schoolbook → Karatsuba → Toom-3 → NTT
  (Goldilocks-field) ladder plus a dedicated `square`, all differentially tested.
- **Fast division & GCD (M7):** Burnikel–Ziegler recursive division above 64
  limbs (over Knuth Algorithm D) and Lehmer's subquadratic GCD above 16 limbs.
- **Sub-quadratic radix conversion (M5):** divide-and-conquer base-B formatting
  (`to_string` of huge numbers is ~46× faster).
- **Float layer complete (M8):** IEEE special values (±0/±∞/NaN), the MPFR
  ternary flag (`*_ternary`), `f64`/`f32`/rational/decimal conversions, an exact
  string codec, and correctly-rounded transcendentals via Ziv's strategy
  (`pi`/`e`/`ln2`, `exp`/`ln`/`sin`/`cos`/`tan`/`atan`).
- **Interop (M9):** in-house `RandomSource` with random `Nat`/`Int` generation
  (plus an optional `rand_core` bridge), optional hand-written `serde` support,
  a C ABI over `Rational` and `Float`, `Sum`/`Product`, byte conversions
  (`from_bytes_le`/`to_bytes_le`), and a REPL that evaluates exact rationals with
  functions (`gcd`/`lcm`/`isqrt`/`fact`/…) and non-decimal literals/radices.

### Added (core surface)

- **`Int` full surface (M1–M5):** tagged `Small/Large` inline representation with
  demotion; `From` for all primitive integers; `ZERO`/`ONE`/`MINUS_ONE`;
  predicates, `signum`, `abs`; fused `addmul`/`submul`; truncated/Euclidean/
  floored division (`div_*`/`rem_*`/`div_rem_*`), `div_exact`, `divides`;
  `gcd`/`lcm`/`extended_gcd`; power-of-two ops (`mul_2k`/`div_2k_trunc`/`mod_2k`/
  `is_power_of_two`/`next`/`prev_power_of_two`/`trailing_zeros`); width-aware
  two's-complement `bitand`/`bitor`/`bitxor`/`bitnot`; `sqrt_exact`/
  `nth_root_exact`; `bit`/`limbs`/`least_significant_limb`/`from_limbs`; bounded
  conversions (`fits_*`/`to_i64`/`to_u64`/`to_f64`); `from_str_radix`/`write_radix`;
  `Hash`; value/ref/`i64` operator + `*Assign` overloads.
- **`Rational` full surface (M6):** `const ZERO`/`ONE`/`MINUS_ONE`;
  `new`/`checked_new`/`from_integer`/`power_of_two`; `From`/`FromStr` (including
  decimals like `"1.5"`); predicates + `signum`; `recip`/`abs`/`pow`; fused
  `addmul`/`submul`; `floor`/`ceil`/`trunc`/`to_integer`; integer division of
  rationals; bounded conversions; `write_decimal`; `Hash`; operators.
- **Fast algorithms (M7):** Karatsuba multiplication and Knuth Algorithm D
  division, replacing the schoolbook/bit-at-a-time cores (same public API),
  differentially tested.
- **Float core (M8):** normalized representation and correctly-rounded
  `add`/`sub`/`mul`/`div`/`sqrt` in all five rounding modes, plus `from_int`/
  `round`/`to_f64` and value-based ordering.
- Free `u_gcd`/`u64_gcd`; extensive integer/rational/float integration tests and
  a Knuth-vs-reference differential unit test.

### Changed

- `Rational::new` now panics on a zero denominator (use `checked_new` for the
  fallible form); the denominator accessor returns `&Int`.
- `Int::pow` takes a `u32` exponent (was `u64`).

## [0.1.0] - 2026-07-03

Initial release: the project scaffold and a working integer/rational core.

### Added

- Single `no_std` + `alloc` crate exposing arbitrary-precision `Nat`, `Int`,
  `Rational`, and a `Float` skeleton, plus an optional C ABI (`ffi` feature,
  `include/puremp.h`) and a `puremp` CLI calculator (`cli` feature).
- `Nat`: normalized limb representation, comparison, addition, checked
  subtraction, schoolbook multiplication, bit shifts, `bit`/`bit_len`/
  `trailing_zeros`, binary (Stein) GCD, bit-at-a-time `div_rem`, and decimal /
  hex I/O.
- `Int`: sign handling, `add`/`sub`/`mul`, `pow`, truncated `div_rem`, ordering,
  decimal I/O, and operator overloads.
- `Rational`: construction with GCD reduction, arithmetic, canonical sign, and
  ordering.
- `Float`: representation, `RoundingMode`, exact integer conversion, and exact
  `Display`. Arithmetic is not yet implemented — see `ROADMAP.md`.
- C ABI over `Int` with panic-safe opaque-handle entry points, plus a C smoke
  test (`tests/ffi_smoke.c`).
- `ROADMAP.md` documenting the design decisions, clean-room provenance, and the
  M1–M10 milestone plan.
- CI: format, clippy (`-D warnings`), tests, `no_std` builds (incl. 32-bit
  `thumbv7em-none-eabi`), MSRV 1.88, C ABI smoke test, and docs.

> **Note:** this is an early scaffold. The arithmetic is correctness-first and
> not yet tuned; sub-quadratic multiplication/division, subquadratic GCD, and
> the floating-point arithmetic layer are on the roadmap.

[Unreleased]: https://github.com/KarpelesLab/puremp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/KarpelesLab/puremp/releases/tag/v0.1.0
