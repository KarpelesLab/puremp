//! `puremp` — arbitrary-precision arithmetic written entirely in Rust, depending
//! on no foreign code.
//!
//! It provides a family of numeric types, built bottom-up:
//!
//! 1. **Integers** — unsigned [`Nat`] and signed [`Int`], the workhorse layer
//!    that carries the hard limb-level algorithms (multiplication, division,
//!    GCD, modular arithmetic, …). Enabled by the `int` feature.
//! 2. **Rationals** — [`Rational`], exact `p/q` fractions kept in lowest terms;
//!    plus [`InfRational`], the same extended with `±∞`/`NaN`. `rational` feature.
//! 3. **Dyadics** — [`Dyadic`], exact `n·2^-k` binary fractions. `dyadic` feature.
//! 4. **Floats** — [`Float`], binary floating-point with a caller-chosen
//!    precision and directed [`RoundingMode`], aiming at MPFR-class correct
//!    rounding, plus [`FixedFloat`], a fixed-precision wrapper with operators.
//!    `float` feature.
//! 5. **Decimals** — [`Decimal`], exact base-10 floating point (Python
//!    `Decimal`-style), with directed rounding. `decimal` feature.
//!
//! Built on top of these are several *derived* structures, each generic or
//! specialised as noted:
//!
//! - [`ModInt`] — modular integers `ℤ/mℤ` with automatic reduction (`int`).
//! - [`Complex`] — generic complex numbers / Gaussian integers (`complex`).
//! - [`Poly`] — generic univariate polynomials (`poly`).
//! - [`Matrix`] — dense matrices with exact determinant/inverse/solve
//!   (`matrix`).
//! - [`Interval`] — outward-rounded interval arithmetic (`interval`).
//! - [`Ball`] — midpoint–radius (mid-rad) rigorous arithmetic, Arb-style (`ball`).
//! - [`Padic`] — fixed-precision `p`-adic numbers in `ℚ_p` (`padic`).
//! - [`Quadratic`] / [`Algebraic`] — exact quadratic irrationals `ℚ(√d)` and
//!   general real algebraic numbers (`algebraic`).
//!
//! `Int`/`Rational` also carry a number-theory toolkit (factorization,
//! `sqrt_mod`, Jacobi/Legendre, CRT, `random_prime`, combinatorics,
//! continued-fraction approximation), and an optional `num-traits` bridge slots
//! the types into generic numeric code.
//!
//! `puremp` is usable as a Rust library, a C library (the `ffi` feature; see
//! `include/puremp.h`), and a standalone command-line calculator (the `cli`
//! feature; the `puremp` binary).
//!
//! This is a clean-room implementation: it is MIT-licensed and its algorithms
//! are drawn from the open literature (Knuth; Brent & Zimmermann's *Modern
//! Computer Arithmetic*; the HAC), never from GMP/MPFR source. See the README's
//! "Design & provenance" section for the algorithm references.
//!
//! # Example
//!
//! ```
//! use puremp::{Int, Rational};
//!
//! // Arbitrary-precision integers.
//! let big = Int::from(2).pow(128);
//! assert_eq!(big.to_string(), "340282366920938463463374607431768211456");
//! assert_eq!(Int::from(1071).gcd(&Int::from(462)).to_string(), "21");
//! assert_eq!(Int::from(2).modpow(&Int::from(10), &Int::from(1000)).to_string(), "24");
//!
//! // Exact rationals, always in lowest terms.
//! let third = Rational::new(Int::from(1), Int::from(3));
//! let sum = &(&third + &third) + &third;
//! assert_eq!(sum.to_string(), "1");
//! ```
//!
//! # `no_std`
//!
//! The crate is `#![no_std]` at its core. Arbitrary-precision types are
//! heap-backed, so they need the `alloc` crate; the `alloc` feature (implied by
//! every type layer) pulls it in. The `std` feature (enabled by default) adds
//! the pieces that genuinely need the operating system — the CLI, `std::error`
//! integration, and system I/O. Build with `--no-default-features` for a bare
//! `no_std` target.

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod error;

#[cfg(feature = "int")]
mod limb;

#[cfg(feature = "int")]
pub mod int;
#[cfg(feature = "int")]
pub mod nat;

#[cfg(feature = "int")]
pub mod random;

#[cfg(feature = "int")]
mod ecm;

#[cfg(feature = "int")]
mod qsieve;

#[cfg(feature = "num-traits")]
mod num_traits_impls;

#[cfg(feature = "rational")]
pub mod rational;

#[cfg(feature = "rational")]
pub mod inf_rational;

#[cfg(feature = "int")]
pub mod mod_int;

#[cfg(feature = "dyadic")]
pub mod dyadic;

#[cfg(feature = "padic")]
pub mod padic;

#[cfg(feature = "decimal")]
pub mod decimal;

#[cfg(feature = "complex")]
pub mod complex;

#[cfg(feature = "poly")]
pub mod poly;

#[cfg(feature = "matrix")]
pub mod matrix;

#[cfg(all(feature = "poly", feature = "rational"))]
mod poly_factor;

#[cfg(feature = "lattice")]
pub mod lattice;

#[cfg(feature = "dlog")]
pub mod dlog;

#[cfg(feature = "algebraic")]
pub mod quadratic;

#[cfg(feature = "algebraic")]
pub mod algebraic;

#[cfg(feature = "float")]
pub mod float;
#[cfg(feature = "float")]
mod float_consts;
#[cfg(feature = "float")]
mod float_mp;
#[cfg(feature = "float")]
mod float_mp_consts;

#[cfg(feature = "float")]
pub mod fixed_float;

#[cfg(feature = "interval")]
pub mod interval;

#[cfg(feature = "ball")]
pub mod ball;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "serde")]
mod serde_impls;

pub use error::{Error, Result};

#[cfg(feature = "int")]
pub use int::{Int, Sign};
#[cfg(feature = "int")]
pub use nat::{Nat, Reciprocal, u_gcd, u64_gcd};
#[cfg(feature = "int")]
pub use random::{RandomSource, SeedRng};

#[cfg(feature = "rational")]
pub use inf_rational::InfRational;
#[cfg(feature = "rational")]
pub use rational::Rational;

#[cfg(feature = "int")]
pub use mod_int::ModInt;

#[cfg(feature = "dyadic")]
pub use dyadic::Dyadic;

#[cfg(feature = "padic")]
pub use padic::Padic;

#[cfg(feature = "decimal")]
pub use decimal::{Decimal, Rounding};

#[cfg(feature = "complex")]
pub use complex::Complex;

#[cfg(feature = "poly")]
pub use poly::Poly;

#[cfg(feature = "matrix")]
pub use matrix::Matrix;

#[cfg(feature = "lattice")]
pub use lattice::{lll_reduce, lll_reduce_delta};

#[cfg(feature = "dlog")]
pub use dlog::{bsgs, discrete_log, pollard_rho};

#[cfg(feature = "algebraic")]
pub use algebraic::Algebraic;
#[cfg(feature = "algebraic")]
pub use quadratic::Quadratic;

#[cfg(feature = "float")]
pub use fixed_float::FixedFloat;
#[cfg(feature = "float")]
pub use float::{Float, RoundingMode};

#[cfg(feature = "ball")]
pub use ball::Ball;
#[cfg(feature = "interval")]
pub use interval::Interval;

/// The crate version string (`CARGO_PKG_VERSION`), exposed for the C ABI and CLI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
