# puremp

Pure-Rust, MIT-licensed, arbitrary-precision arithmetic — **integers,
rationals, MPFR-class floating point, and base-10 decimals**, plus derived
**modular integers, complex numbers, polynomials, matrices, and intervals** —
with no foreign-code dependencies. Usable as a Rust crate, a C library, and a
command-line calculator.

> **Status: feature-complete, pre-1.0.** The full signed-integer and exact
> rational surface is implemented and tested (small-value inlining, all three
> division conventions, power-of-two ops, two's-complement bitwise, gcd/lcm/
> extended-gcd, roots, radix I/O). Multiplication runs a schoolbook → Karatsuba
> → Toom-3 → NTT ladder; division uses Knuth Algorithm D and Burnikel–Ziegler;
> GCD uses Lehmer's algorithm. The optional `Float` layer is correctly-rounded
> through the transcendentals (`exp`/`ln`/`sin`/`cos`/`atan`, `pi`/`e`) with the
> IEEE special values and the MPFR ternary flag. Remaining work is performance
> tuning and a pre-`1.0` API review — see [`ROADMAP.md`](ROADMAP.md).

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
| `interval` | ✔ | `Interval` — outward-rounded interval arithmetic (implies `float`) |
| `algebraic` | ✔ | `Quadratic` (ℚ(√d)) and general real `Algebraic` numbers |
| `float` | ✔ | Separable `Float` + `FixedFloat` layer (implies `int`); not part of the core contract, disable via `--no-default-features` |
| `num-traits` | | Implements `num-traits` interfaces for `Int`/`Rational`/`Nat` |
| `ffi` | | The C ABI module (`include/puremp.h`) |
| `cli` | ✔ | The `puremp` binary |

Beyond the base types, `Int`/`Rational` provide a number-theory toolkit —
`factorize`, `sqrt_mod` (Tonelli–Shanks), `jacobi`/`legendre`, `crt`,
`random_prime`, `factorial`/`binomial`/`fibonacci`, and continued-fraction
`approximate` — plus `ModInt` for modular arithmetic.

For a bare `no_std` build: `--no-default-features` (add `--features int` for the
integer types).

## License

MIT — see [`LICENSE`](LICENSE).
