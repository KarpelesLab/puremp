//! `puremp` — arbitrary-precision arithmetic written entirely in Rust, depending
//! on no foreign code.
//!
//! It provides three families of numbers, built bottom-up:
//!
//! 1. **Integers** — unsigned [`Nat`] and signed [`Int`], the workhorse layer
//!    that carries the hard limb-level algorithms (multiplication, division,
//!    GCD, modular arithmetic, …). Enabled by the `int` feature.
//! 2. **Rationals** — [`Rational`], exact `p/q` fractions kept in lowest terms.
//!    Enabled by the `rational` feature.
//! 3. **Floats** — [`Float`], binary floating-point with a caller-chosen
//!    precision and directed [`RoundingMode`], aiming at MPFR-class correct
//!    rounding. Enabled by the `float` feature.
//!
//! `puremp` is usable as a Rust library, a C library (the `ffi` feature; see
//! `include/puremp.h`), and a standalone command-line calculator (the `cli`
//! feature; the `puremp` binary).
//!
//! This is a clean-room implementation: it is MIT-licensed and its algorithms
//! are drawn from the open literature, never from GMP/MPFR source. See
//! `ROADMAP.md` for the design, the algorithm references, and the milestone
//! plan.
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

#[cfg(feature = "rational")]
pub mod rational;

#[cfg(feature = "float")]
pub mod float;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use error::{Error, Result};

#[cfg(feature = "int")]
pub use int::{Int, Sign};
#[cfg(feature = "int")]
pub use nat::{Nat, u_gcd, u64_gcd};
#[cfg(feature = "int")]
pub use random::RandomSource;

#[cfg(feature = "rational")]
pub use rational::Rational;

#[cfg(feature = "float")]
pub use float::{Float, RoundingMode};

/// The crate version string (`CARGO_PKG_VERSION`), exposed for the C ABI and CLI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
