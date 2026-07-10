# puremp

[![CI](https://github.com/KarpelesLab/puremp/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/puremp/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/puremp.svg)](https://crates.io/crates/puremp)
[![docs.rs](https://img.shields.io/docsrs/puremp)](https://docs.rs/puremp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust, MIT-licensed, arbitrary-precision arithmetic — **integers, rationals,
MPFR-class floating point, and base-10 decimals**, plus derived **modular
integers, complex numbers, polynomials, matrices, interval & ball arithmetic,
p-adic numbers, finite fields, elliptic curves, and exact real algebraic numbers** — with no foreign-code
dependencies. Usable as a Rust crate, a C library, and a command-line calculator.

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
| `padic` | ✔ | `Padic` — fixed-precision `p`-adic numbers ℤ_p/ℚ_p (implies `rational`) |
| `complex` | ✔ | `Complex<T>` — generic complex / Gaussian integers |
| `poly` | ✔ | `Poly<T>` — generic univariate polynomials |
| `matrix` | ✔ | `Matrix<T>` — dense matrices with exact linear algebra |
| `lattice` | ✔ | `lll_reduce` — exact LLL lattice basis reduction (implies `rational`) |
| `interval` | ✔ | `Interval` — outward-rounded interval arithmetic (implies `float`) |
| `ball` | ✔ | `Ball` — midpoint–radius (mid-rad) rigorous arithmetic, Arb-style (implies `interval`) |
| `algebraic` | ✔ | `Quadratic` (ℚ(√d)) and general real `Algebraic` numbers |
| `galois` | ✔ | `GaloisField` / `GfElement` — finite field extensions `GF(pᵏ)` (implies `int`) |
| `elliptic` | ✔ | `EllipticCurve` / `Point` over GF(p) and ℚ — group law, Jacobian scalar mult, Schoof point counting (implies `rational`) |
| `numberfield` | ✔ | Algebraic number fields `ℚ(θ)` — element arithmetic, ring of integers, ideals + prime factorization, unit group / regulator, class group (implies `rational` + `poly` + `matrix` + `lattice` + `algebraic`) |
| `identify` | ✔ | Inverse symbolic calculator (`identify`, `machin_like`) via PSLQ (implies `lattice` + `float`) |
| `primality` | ✔ | Primality proving with auditable certificates — Pocklington + BLS `n∓1`, and Atkin–Morain **ECPP** for the general case (implies `int` + `poly`) |
| `float` | ✔ | Separable `Float` + `FixedFloat` layer (implies `int`); not part of the core contract, disable via `--no-default-features` |
| `dlog` | | Discrete logarithm — BSGS, Pollard rho, Pohlig–Hellman (implies `int`) |
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
- `Poly<T>` and `Matrix<T>` are generic over any `Ring` — including the
  context-carrying `ModInt` (ℤ/nℤ) and `GfElement` (GF(pᵏ)), so polynomials and
  matrices over finite fields and modular integers work, not just over `Int`/`Rational`.
  Over a `Field`, `FieldMatrix` gives `determinant`/`inverse`/`solve`/`rank` by
  Gaussian elimination, `FactorOverField` factors polynomials over a finite
  field (Cantor–Zassenhaus), and `RingMatrix` gives a division-free determinant
  and characteristic polynomial (Samuelson–Berkowitz) over *any* commutative ring
  — including non-fields like ℤ/nℤ (composite `n`) or `Matrix<Poly<Int>>`.

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

Candidate directions not yet implemented, all specifiable from open literature
(so they preserve the clean-room provenance). Ordering is rough interest, not
commitment. Brent & Zimmermann's *Modern Computer Arithmetic* (MCA; freely
available drafts) is the umbrella reference for much of this list.

**Faster algorithms** (existing operations, correct today, just not maximally fast):

- **Deeper multi-prime argument reduction** — the current `exp` multi-prime fast
  path finds the reducing prime-log combination with an f64 Babai nearest-plane,
  which caps its winning range; a higher-precision Babai over more primes would
  widen it. Johansson, arXiv:2207.02501.

**Candidate new capabilities** (new operations / types):

- **Factorization past ~60 digits (GNFS)** — SIQS, with the single/double
  large-prime variations and a block-Lanczos GF(2) solver, factors balanced
  semiprimes reliably to ~55–58 digits; the general number-field sieve is the path
  to ~100+ digits. It needs number-field arithmetic, polynomial selection, lattice
  sieving, and a number-field square root — a substantial subsystem of its own.
- **SEA point counting** — the Elkies/Atkin improvements to Schoof (using modular
  polynomials and isogenies) drop the per-prime cost from working modulo the
  degree-`(ℓ²−1)/2` division polynomial to a small factor, extending exact point
  counting from the current tens-of-bits `p` to cryptographic sizes. Schoof (1985)
  → Elkies → Atkin; Blake–Seroussi–Smart ch. VII.
- **Wider ECPP** — the Atkin–Morain prover uses a fixed small table of
  class-number ≤ 2 discriminants, leaving a small fraction of primes `Unproven`;
  computing Hilbert class polynomials at runtime (or APR-CL as a deterministic
  alternative) would close the gap.

## License

MIT — see [`LICENSE`](LICENSE).
