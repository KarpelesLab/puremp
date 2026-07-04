# puremp

[![CI](https://github.com/KarpelesLab/puremp/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/puremp/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/puremp.svg)](https://crates.io/crates/puremp)
[![docs.rs](https://img.shields.io/docsrs/puremp)](https://docs.rs/puremp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust, MIT-licensed, arbitrary-precision arithmetic — **integers,
rationals, MPFR-class floating point, and base-10 decimals**, plus derived
**modular integers, complex numbers, polynomials, matrices, intervals, and exact
real algebraic numbers** — with no foreign-code dependencies. Usable as a Rust
crate, a C library, and a command-line calculator.

## Why

A GMP + MPFR-class toolkit that is:

- **Pure, safe Rust** — no C, no inline assembly, no intrinsics. The only
  `unsafe` in the crate is the opt-in C ABI module.
- **Clean-room & MIT-licensed** — algorithms come from the open literature
  (Knuth; Brent & Zimmermann's *Modern Computer Arithmetic*; the HAC), never
  from GMP/MPFR source. Use it anywhere, including closed-source projects.
- **`no_std` + `alloc`** — runs on bare metal with an allocator; no OS
  assumptions in the core. Verified on 32-bit `thumbv7em-none-eabi` in CI.

## Quick start (Rust)

```toml
[dependencies]
puremp = "0"
```

```rust
use puremp::{Int, Rational};

let big = Int::from_i64(2).pow(100);
assert_eq!(big.to_string(), "1267650600228229401496703205376");

let sum = Rational::new(Int::from_i64(1), Int::from_i64(2))?   // 1/2
    .add(&Rational::new(Int::from_i64(1), Int::from_i64(3))?); // + 1/3
assert_eq!(sum.to_string(), "5/6");
# Ok::<(), puremp::Error>(())
```

## Quick start (CLI)

```console
$ cargo run --bin puremp
puremp> 2 ** 100
1267650600228229401496703205376
puremp> x = 1000
puremp> x * x - 1
999999
puremp> (2**64) * (2**64)
340282366920938463463374607431768211456
puremp> :quit
```

Supports `+ - * / % **`, parentheses, unary minus, decimal literals, and
`name = expr` variables (`/` and `%` are truncated integer division).

## Quick start (C)

Build the static and/or shared library and link against the header in
[`include/puremp.h`](include/puremp.h):

```console
$ cargo rustc --lib --release --features ffi --crate-type staticlib
$ cargo rustc --lib --release --features ffi --crate-type cdylib
$ cc myprog.c -I include target/release/libpuremp.a -lpthread -ldl -lm -o myprog
```

```c
#include "puremp.h"
#include <stdio.h>

int main(void) {
    PurempInt *two = puremp_int_from_i64(2);
    PurempInt *big = puremp_int_pow(two, 100);
    char *s = puremp_int_to_string(big);
    printf("2^100 = %s\n", s);
    puremp_string_free(s);
    puremp_int_free(big);
    puremp_int_free(two);
    return 0;
}
```

## Feature flags

| Feature | Default | Enables |
|---|:---:|---|
| `std` | ✔ | `std::error::Error`, the CLI, system I/O (implies `alloc`) |
| `alloc` | ✔ | Heap-backed arbitrary-precision types (required by every layer) |
| `int` | ✔ | `Nat` and `Int` |
| `rational` | ✔ | `Rational` and `InfRational` (implies `int`) |
| `dyadic` | ✔ | `Dyadic` — exact `n·2⁻ᵏ` binary fractions (implies `int`) |
| `decimal` | ✔ | `Decimal` — exact base-10 floating point (implies `int`) |
| `complex` | ✔ | `Complex<T>` — generic complex / Gaussian integers |
| `poly` | ✔ | `Poly<T>` — generic univariate polynomials |
| `matrix` | ✔ | `Matrix<T>` — dense matrices with exact linear algebra |
| `lattice` | ✔ | `lll_reduce` — exact LLL lattice basis reduction (implies `rational`) |
| `interval` | ✔ | `Interval` — outward-rounded interval arithmetic (implies `float`) |
| `algebraic` | ✔ | `Quadratic` (ℚ(√d)) and general real `Algebraic` numbers |
| `float` | ✔ | Separable `Float` + `FixedFloat` layer (implies `int`); not part of the core contract, disable via `--no-default-features` |
| `num-traits` | | Implements `num-traits` interfaces for `Int`/`Rational`/`Nat`/`Decimal`/`Complex` |
| `ffi` | | The C ABI module (`include/puremp.h`) |
| `cli` | ✔ | The `puremp` binary |

Beyond the base types, `Int`/`Rational` provide a number-theory toolkit —
`factorize` (trial division → Pollard rho → Lenstra ECM → quadratic sieve),
`sqrt_mod` (Tonelli–Shanks), `jacobi`/`legendre`, `crt`, `random_prime`,
`factorial`/`binomial`/`fibonacci`, and continued-fraction `approximate` — plus
`ModInt` for modular arithmetic.

The exact-algebra layers stack on top of these:

- `Poly::factor` factors a rational polynomial into irreducible factors over ℚ
  (Berlekamp–Zassenhaus with **van Hoeij** LLL recombination).
- `Algebraic` is a real root of a rational polynomial, with exact `+ − × ÷`,
  comparison, and `sqrt` — and `Algebraic::from_float` recovers one *exactly* from
  a floating-point approximation.
- `lattice::lll_reduce` is an exact LLL lattice reduction; on top of it,
  `find_integer_relation` recognizes constants (e.g. is a number `a·√2 + b`?) and
  `minimal_polynomial` recovers a real algebraic number's defining polynomial.

For a bare `no_std` build: `--no-default-features` (add `--features int` for the
integer types).

## Design & provenance

Bottom-up layers, each building only on the ones below: machine-word carry
primitives (`adc`/`sbb`/`mac`) → unsigned magnitudes (`Nat`, home of the hard
algorithms) → tagged signed `Int` → `Rational`, with the optional `Float` and
the derived types layered on top. Signed integers inline single-limb magnitudes
(no heap allocation until a value exceeds 64 bits).

The implementation is **clean-room**: GMP and MPFR are LGPL and their source is
never consulted. Algorithms come from the open literature —

- Knuth, *TAOCP* Vol. 2 §4.3 (schoolbook arithmetic; Algorithm D for division);
- Brent & Zimmermann, *Modern Computer Arithmetic* (sub-quadratic multiply/
  divide, GCD, base conversion);
- Menezes, van Oorschot & Vanstone, *Handbook of Applied Cryptography*;
- primary papers: Karatsuba; Toom–Cook; Burnikel–Ziegler; Möller–Granlund;
  Faddeev–LeVerrier (algebraic numbers); Sturm sequences (real-root isolation);
  Lenstra–Lenstra–Lovász (LLL); Cantor–Zassenhaus and van Hoeij (polynomial
  factorization).

Correctness is checked against published values and, in the dev-only test
harness, a trusted reference — never a runtime dependency.

**Non-goals:** constant-time / side-channel resistance across the general API
(for constant-time crypto see the sibling `purecrypto` crate); drop-in GMP/MPFR
C-header compatibility (puremp ships its own cleaner C ABI).

Run `cargo run --release --example bench` for a throughput harness across the
core operations and the derived types.

## Roadmap

Candidate directions, all specifiable from open literature (so they preserve the
clean-room provenance); items marked *(shipped)* are already implemented. Ordering
is rough interest, not commitment. Brent & Zimmermann's *Modern Computer
Arithmetic* (MCA; freely available drafts) is the umbrella reference for much of
this list.

**Faster algorithms** (existing operations, correct today, just not maximally fast):

- **Embedded transcendental constants** *(shipped)* — `ln 2` and `π` are stored as
  precomputed 65536-bit significands (clean-room: generated by this crate's own
  series) and simply *rounded* to the working precision, instead of re-running a
  series on every call. This speeds up `exp`, `sin`/`cos`, and `ln` across the
  whole sub-65k-bit range with no global state.
- **Multi-prime argument reduction** *(shipped, `exp`)* — Johansson's method
  (arXiv:2207.02501): reduce `x` by a combination of prime logarithms `Σ eᵢ·ln pᵢ`
  found with an f64 Babai nearest-plane on a precomputed LLL-reduced lattice, so
  the leftover argument is tiny and the Taylor series needs *no* halving squarings.
  ~1.3–1.65× on `exp` in the ~1k–8k-bit band, correctly rounded (a bounded-error
  interval check falls back to the `√`-precision path when the rounding is
  ambiguous). Extending it to higher precision (deeper Babai) and to `sin`/`cos`
  is future work.
- **Sharper Newton building blocks** — power-series square root in `(4/3)·M(n)`
  (from ~1.83) and reciprocal in `(13/9)·M(n)` (from 1.5) via a third-order
  iteration whose extra term is nearly free; feeds `Float::sqrt`, reciprocal, and
  Newton nth-root. Harvey, *Faster algorithms for the square root and reciprocal
  of power series* (arXiv:0910.1926).
- **AGM-based transcendentals** — π and `log`/`exp` via the arithmetic–geometric
  mean (Brent–Salamin / Gauss–Legendre), `O(M(n)·log n)` in ~2·lg n quadratically
  converging steps; a large implicit constant means it *complements* binary
  splitting, winning only at very high precision. Brent (1976); Borwein & Borwein,
  *Pi and the AGM*; MCA §4.8.
- **Half-GCD** for subquadratic `Rational` reduction — a recursive 2×2 cofactor
  matrix HGCD, `O(M(n)·log n)` vs. the current `~O(n²)` Lehmer. The canonical
  clean-room reference is Möller's left-to-right variant (its stop condition
  removes the back-up steps, "much simpler to implement"); only wins at very large
  operand sizes. Möller, *Math. Comp.* 77 (2008); MCA §1.6.
- **Subresultant PRS** to tame Sturm-sequence coefficient growth for high-degree
  `Algebraic` operations (Collins–Brown; MCA §2.4).

**Candidate new capabilities** (new operations / types):

- **Integer factorization beyond trial division + Pollard rho** — both shipped.
  Lenstra's **ECM** (Montgomery-curve arithmetic in projective `(X : Z)`
  coordinates, Suyama parameterization, two-stage with a baby-step/giant-step
  continuation) covers medium factors, its cost scaling with the *factor* size;
  the single-polynomial **quadratic sieve** (factor base of quadratic residues,
  a log-sieve with trial-division confirmation, `GF(2)` linear algebra by
  Gaussian elimination) covers *balanced* semiprimes into the mid-40-digit
  range, its cost scaling with `n`. `factorize` escalates trial division → rho
  → ECM → QS automatically. The next step — pushing the sieve toward 100
  digits — is the self-initializing multiple-polynomial variant (**SIQS**),
  whose small per-polynomial intervals lift the single-polynomial memory limit.
  Zimmermann's ECM survey; Crandall & Pomerance, *Prime Numbers: A
  Computational Perspective*; Pomerance, *A Tale of Two Sieves*; the HAC.
- **Primality *proving*** — upgrade probabilistic Miller–Rabin to a certificate
  via **ECPP** (Goldwasser–Kilian → Atkin → Morain; heuristic `Õ((log N)⁵)`, fast
  variant `Õ((log N)⁴)`) or the deterministic **APR-CL**.
- **Building on the `lattice` LLL** (shipped, with `find_integer_relation`,
  `minimal_polynomial`, `Algebraic::from_float` — recover an exact algebraic
  number from a float approximation — and **PSLQ** integer-relation detection
  (`lattice::pslq`, one-level PSLQ with γ = 2/√3; Ferguson, Bailey & Arno) — all
  shipped). Further Diophantine-approximation refinements remain future work.
- **`Poly::factor`** (shipped) factors rational polynomials over ℚ by
  Berlekamp–Zassenhaus — square-free decomposition (Yun), Cantor–Zassenhaus mod
  `p`, Hensel lifting, and **van Hoeij**'s LLL-knapsack recombination (built on the
  `lattice` LLL): the true factors are short vectors of a lattice of the modular
  factors' power sums, recovered in polynomial time rather than by an exponential
  subset search — so Swinnerton–Dyer-style inputs (many modular factors, few
  rational ones) factor quickly. Trial recombination remains the verified
  fallback.
- **Special functions** for `Float` — *shipped*: Euler's constant γ
  (`euler_gamma`, Brent–McMillan), Catalan's constant (`catalan`), the Riemann ζ
  (`zeta`, Borwein's acceleration of the alternating η, real `s > 0, s ≠ 1`), and
  `erf` / `erfc` (all-positive Kummer series with a continued-fraction tail for
  large arguments) — all correctly rounded via the Ziv strategy. Still candidate:
  Γ / `lgamma` (Stirling by **rectangular splitting**, ~2√n full multiplications;
  Johansson, arXiv:2109.08392) and Bessel functions (MCA §4.7.1).
- **Discrete logarithm** *(shipped, `dlog` feature)* — baby-step/giant-step and
  Pollard's rho for logs (`dlog::discrete_log`, `ModInt::discrete_log`; HAC §3.6),
  `factorize`-style automatic dispatch by group-order size.
- **`p`-adic numbers** *(shipped, `padic` feature)* — `Padic`, fixed-precision
  ℤ_p / ℚ_p as `p^v·u` (unit `u`), with valuation-aware `+ − × ÷`, digit
  expansion, and Hensel-lifted `sqrt`.

## License

MIT — see [`LICENSE`](LICENSE).
