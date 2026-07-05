# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from `1.0.0`
onward (pre-`1.0`, minor versions may contain breaking changes).

## [Unreleased]

## [0.2.3](https://github.com/KarpelesLab/puremp/compare/v0.2.2...v0.2.3) - 2026-07-05

### Added

- *(float)* digamma/polygamma, beta, and second-kind Bessel Yₙ/Kₙ
- elliptic curves y² = x³ + a·x + b over GF(p) and ℚ
- *(complex)* add remaining elementary transcendentals to Complex<Float>
- *(matrix)* exact eigenvalues of a rational matrix (algebraic feature)
- *(matrix)* division-free determinant & charpoly over any ring (Berkowitz)

### Other

- *(float)* AGM (Brent–Salamin) ln above a 4096-bit threshold
- roadmap/feature-table updates for the new capabilities; minor cleanups

## [0.2.2](https://github.com/KarpelesLab/puremp/compare/v0.2.1...v0.2.2) - 2026-07-04

### Added

- *(poly)* Cantor–Zassenhaus factorization over finite fields GF(q)
- *(matrix)* generic linear algebra over any Field via FieldMatrix trait
- *(ring)* add Field trait (Ring + Div, with inv)
- *(ring)* generic Ring layer — Poly/Matrix over ModInt, GF(pᵏ), any ring
- add GF(pᵏ) finite field extensions (galois feature)
- *(float)* add gamma, ln_gamma and first-kind Bessel Jₙ/Iₙ
- *(factor)* self-initializing MPQS (SIQS) for balanced semiprimes
- *(primality)* certificate-based primality proving (Pocklington + BLS n^{1/3})
- *(identify)* inverse symbolic calculator + Machin-formula discovery
- *(ball)* rigorous root solving + monotone exp/ln on Ball
- *(dlog)* add Pohlig–Hellman discrete log; dispatch composite orders to it
- *(ball)* add Ball — midpoint–radius rigorous arithmetic (Arb-style)

### Other

- note FieldMatrix and FactorOverField in the generic-ring bullet
- add finite fields (GaloisField) to the intro blurb and feature table
- drop Half-GCD from roadmap (shipped, guarded)
- harden Half-GCD: always-on gcd-preservation guard + exhaustive determinism
- subquadratic Half-GCD for Nat::gcd (O(M(n)·log n))
- drop Subresultant PRS (shipped) and Sharper Newton building blocks from roadmap
- *(algebraic)* subresultant PRS for Sturm chains and polynomial GCD
- prune SIQS/gamma-Bessel/primality from the roadmap; add primality feature
- prune shipped items from the roadmap
- add Ball/p-adic to the intro blurb and dlog/padic to the feature table
- note identify/machin, Pohlig–Hellman, and Ball root-finding as shipped

## [0.2.1](https://github.com/KarpelesLab/puremp/compare/v0.2.0...v0.2.1) - 2026-07-04

### Added

- *(complex)* complete the operator set — all owned/borrowed combinations
- *(padic)* fixed-precision p-adic numbers (ℤ_p / ℚ_p)
- *(dlog)* add discrete-logarithm solving (BSGS + Pollard's rho)
- *(lattice)* add PSLQ integer-relation detection
- *(float)* add erf/erfc and Riemann zeta special functions

### Other

- promote dlog, p-adic, PSLQ, and ζ/erf from roadmap candidates to shipped
- remove accidentally-committed agent memory dir
- remove accidentally-committed agent memory dir

## [0.2.0](https://github.com/KarpelesLab/puremp/compare/v0.1.9...v0.2.0) - 2026-07-04

### Added

- *(float)* Euler–Mascheroni γ and Catalan's constant
- *(float,rational,algebraic)* rounding conveniences + exact-rational detection
- *(int,random)* number-theory helpers, RNG-free prime successors, SeedRng
- *(complex,float)* Complex<Float> support — operators + transcendentals
- multi-prime argument reduction for exp (+ restore const ln2/pi wiring)

### Other

- embed ln2/pi as precomputed constants; remove FloatContext

## [0.1.9](https://github.com/KarpelesLab/puremp/compare/v0.1.8...v0.1.9) - 2026-07-04

### Added

- quadratic sieve for balanced-semiprime factorization
- FloatContext — context-cached exp (caller-held, no global state)
- Lenstra ECM for medium-factor integer factorization
- Algebraic::from_float — recover an exact algebraic number from a float
- van Hoeij LLL recombination for polynomial factorization
- polynomial factorization over ℚ (Poly::factor, Berlekamp–Zassenhaus)
- integer-relation detection and minimal-polynomial recovery (LLL)
- LLL lattice basis reduction (lattice feature)

### Fixed

- use integer sqrt in ECM stage 2 for no_std builds
- collapse if-let in normalize_sign (clippy -D warnings)

### Other

- surface the exact-algebra capabilities in the README
- cite rectangular splitting (Γ) and Brent–McMillan B3 (γ) in the roadmap
- refine roadmap from the literature survey (verified references)
- double-word Lehmer window and O(1) window reads — gcd ~20% faster at 14k bits, ~2x at 200k
- expand future work into a roadmap (faster algorithms + new capabilities)

## [0.1.8](https://github.com/KarpelesLab/puremp/compare/v0.1.7...v0.1.8) - 2026-07-04

### Fixed

- clippy needless_range_loop in the matrix zero-pivot test

### Other

- scaled-integer atanh/atan/sin/cos series — ln ~4x, sin ~3x, atan ~3.5x faster at 1k bits
- don't intra-doc-link private ntt_worthwhile from public square docs
- retire the nth_root_floor future-work item
- Newton's method for nth_root_floor (k>2) — 10-29x faster
- fill-aware NTT dispatch, 24-bit digits and one-transform squaring
- Karatsuba multiplication for Poly<T>
- paired REDC steps and addmul_2 product in Montgomery arithmetic — modpow ~10% faster
- fraction-free Matrix<Rational> solve/inverse — solve up to ~8x
- binary splitting for ln2/atan series and integer-series exp — pi ~2.5x, exp ~2.6x faster
- fraction-free (Bareiss) Matrix<Rational> determinant
- Möller–Granlund reciprocal in divmod_small — pi ~26%, exp ~19% faster

## [0.1.7](https://github.com/KarpelesLab/puremp/compare/v0.1.6...v0.1.7) - 2026-07-04

### Fixed

- clippy clean-up for the CIOS Montgomery commit

### Other

- do not intra-doc-link the private sqrt_rem from public isqrt docs
- changelog entries for the performance work
- unit fast paths in Nat::mul, gcd and Rational::normalize — integer-valued rationals ~4x faster
- slice-recursion Karatsuba into shared out/scratch buffers
- retune Burnikel–Ziegler base case to 96 half-block limbs
- skip the discarded top-of-ladder squarings in radix I/O — ~16% faster
- fused single-pass Lehmer cofactor application — gcd ~3x faster
- second range-reduction stage in exp — ~40% faster
- Zimmermann square root with remainder — isqrt ~2.9x faster
- raise the radix-conversion base case to 10 limbs
- machine-word fast path for small gcd — small Rational ops ~4x faster
- evaluate pi and ln2 by scaled integer series — pi ~4.7x, exp ~2x faster
- fold significand trailing zeros into the exponent in Float mul/div
- retune multiplication crossovers for the addmul_2 basecase
- dedicated Montgomery squaring + bounds-check-free CIOS — modpow ~18% faster
- paired triangle rows (addmul_2) in schoolbook squaring
- addmul_2 inner loop for schoolbook multiplication — ~2.5x faster basecase
- emit 19 digits per division in the radix base case — to_string ~10% faster
- precompute NTT twiddle factors per stage — NTT ~40% faster
- CIOS Montgomery multiplication — modpow ~32% faster

### Other

- unit fast paths in Nat::mul, gcd and Rational::normalize — integer-valued rationals ~4x faster
- slice-recursion Karatsuba into shared out/scratch buffers (2 allocations per multiply)
- retune Burnikel–Ziegler base case for the addmul_2 basecase
- skip the discarded top-of-ladder squarings in radix I/O (~16% faster)
- fused single-pass Lehmer cofactor application — gcd ~3x faster
- second range-reduction stage in exp — ~40% faster
- Zimmermann square root with remainder — isqrt ~2.9x faster
- machine-word fast path for small gcd — small Rational ops ~4x faster
- evaluate pi and ln2 by scaled integer series — pi ~4.7x, exp ~2x faster
- fold significand trailing zeros into the exponent in Float mul/div (pi ~39%, exp ~35% faster)
- dedicated Montgomery squaring + bounds-check-free CIOS — modpow ~18% faster
- retune multiplication crossovers for the addmul_2 basecase
- addmul_2 inner loops for schoolbook multiplication and squaring (~2.5x faster basecase)
- emit 19 digits per division in the radix base case — to_string ~10% faster

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
