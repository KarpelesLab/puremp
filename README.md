# puremp

[![CI](https://github.com/KarpelesLab/puremp/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/puremp/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/puremp.svg)](https://crates.io/crates/puremp)
[![docs.rs](https://img.shields.io/docsrs/puremp)](https://docs.rs/puremp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust, MIT-licensed, arbitrary-precision arithmetic — **integers,
rationals, MPFR-class floating point, and base-10 decimals**, plus derived
**modular integers, complex numbers, polynomials, matrices, and intervals** —
with no foreign-code dependencies. Usable as a Rust crate, a C library, and a
command-line calculator.

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
`factorize`, `sqrt_mod` (Tonelli–Shanks), `jacobi`/`legendre`, `crt`,
`random_prime`, `factorial`/`binomial`/`fibonacci`, and continued-fraction
`approximate` — plus `ModInt` for modular arithmetic.

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
  Faddeev–LeVerrier (algebraic numbers); Sturm sequences (real-root isolation).

Correctness is checked against published values and, in the dev-only test
harness, a trusted reference — never a runtime dependency.

**Non-goals:** constant-time / side-channel resistance across the general API
(for constant-time crypto see the sibling `purecrypto` crate); drop-in GMP/MPFR
C-header compatibility (puremp ships its own cleaner C ABI).

Run `cargo run --release --example bench` for a throughput harness across the
core operations and the derived types.

## Roadmap

Candidate directions, all specifiable from open literature (so they preserve the
clean-room provenance). Nothing here is implemented yet; ordering is rough
interest, not commitment. Brent & Zimmermann's *Modern Computer Arithmetic* (MCA;
freely available drafts) is the umbrella reference for most of this list.

**Faster algorithms** (existing operations, correct today, just not maximally fast):

- **Multi-prime argument reduction** for elementary functions (`exp`/`log`/`sin`/
  `cos`) — reduces the argument using Diophantine combinations of logarithms of
  many primes, solved with a fast integer-relation search. Reported ~2× speedup
  from a few thousand bits to millions; applies directly to the series we already
  ship. Johansson, *Arbitrary-precision computation of the gamma function* /
  "faster elementary functions" (arXiv:2207.02501, 2022). **Most actionable.**
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

- **Integer factorization beyond trial division + Pollard rho** — Lenstra's
  **ECM** (elliptic-curve method, the best method whose cost scales with the
  *factor* size, using Montgomery-curve arithmetic), then the **quadratic sieve**
  for the 40–100-digit range. Zimmermann's ECM survey; Crandall & Pomerance,
  *Prime Numbers: A Computational Perspective*; the HAC.
- **Primality *proving*** — upgrade probabilistic Miller–Rabin to a certificate
  via **ECPP** (Goldwasser–Kilian → Atkin → Morain; heuristic `Õ((log N)⁵)`, fast
  variant `Õ((log N)⁴)`) or the deterministic **APR-CL**.
- **Building on the `lattice` LLL** (shipped, with `find_integer_relation` and
  `minimal_polynomial` on top) — **PSLQ** integer relations and Diophantine
  approximation as further refinements.
- **`Poly::factor`** (shipped) factors rational polynomials over ℚ by
  Berlekamp–Zassenhaus — square-free decomposition (Yun), Cantor–Zassenhaus mod
  `p`, Hensel lifting, and trial recombination. Still open: **van Hoeij**'s
  LLL-knapsack recombination, which replaces the worst-case-exponential subset
  search with a polynomial-time lattice step (matters for many-modular-factor
  inputs like Swinnerton–Dyer polynomials).
- **Special functions** for `Float` — Γ / `lgamma` (Stirling series evaluated by
  **rectangular splitting**, ~2√n full multiplications; Johansson, arXiv:2109.08392,
  2021), the Riemann ζ (Euler–Maclaurin / Borwein), `erf`/`erfc` (incomplete-Γ
  series), Bessel functions (MCA §4.7.1), and Euler's constant γ (Brent–McMillan
  B3 via Bessel I₀/K₀, binary-split; the fastest known method), all correctly
  rounded.
- **Discrete logarithm** (baby-step/giant-step, Pollard rho for DLP) and
  **`p`-adic numbers** (ℤ_p/ℚ_p) as new exact-arithmetic types.

## License

MIT — see [`LICENSE`](LICENSE).
