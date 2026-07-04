---
name: mathesis-consumer
description: Mathesis is a Wolfram-style frontend that delegates ALL math to puremp; drives puremp's public API requirements
metadata:
  type: project
---

**Mathesis** is a Mathematica-style calculator frontend that implements *no* mathematics itself — it delegates everything to puremp. So puremp's public API is Mathesis's math engine, and gaps there directly block Mathematica builtins.

**Why:** puremp should expose derivations (EulerPhi, Divisors, complex transcendentals, …) rather than force frontends to reimplement number theory / complex analysis on principle.

**How to apply:** when adding public API, consider the "Wolfram builtin" shape — exact-first, ergonomic operators, Display/tex-friendly output. Requested (2026-07, by priority): P0 Complex<Float> usable (Float operators + complex exp/ln/sqrt/pow/sin/cos, abs/arg→Float); P1 Int number-theory helpers on top of factorize (euler_phi, divisors, divisor_sigma/count, moebius_mu, radical) + RNG-free next_prime/prev_prime + seedable no_std RandomSource; P2 Float floor/ceil/round/trunc→Int, Rational::round, Algebraic/Quadratic ergonomics (sqrt(&Rational), operators, to_f64/decimal, Display); P3 Float γ & Catalan constants, Poly symbolic surface. Already-clean: modpow/modinv/extended_gcd/jacobi/crt/sqrt_mod/continued_fraction, LLL, Matrix<Rational>, transcendental suite.
