# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from `1.0.0`
onward (pre-`1.0`, minor versions may contain breaking changes).

## [Unreleased]

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
