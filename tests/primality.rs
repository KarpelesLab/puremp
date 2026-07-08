//! Tests for certificate-based primality **proving** (`primality` feature):
//! Pocklington–Lehmer, the BLS `n^{1/3}` refinement, and Atkin–Morain ECPP for
//! `n∓1`-hard inputs.
//!
//! These check three things: that [`prove_prime`] agrees with the reference
//! Baillie–PSW test on known primes and composites (including Carmichael numbers
//! and products of large primes), that every returned certificate re-verifies
//! independently via [`PrimalityCertificate::verify`], and that a certificate
//! bound to the wrong number is rejected. (Tampering with a certificate's
//! *internal* fields is exercised by the unit tests inside `src/primality.rs`,
//! which have access to the private representation.)
#![cfg(feature = "primality")]

use puremp::Int;
use puremp::primality::{Bound, Primality, prove_prime};

/// Parse a decimal string into an [`Int`].
fn n(s: &str) -> Int {
    s.parse().expect("valid integer literal")
}

/// `2^k − 1`.
fn mersenne(k: u32) -> Int {
    Int::from(2).pow(k).sub(&Int::ONE)
}

/// Assert `x` is proved prime and its certificate re-verifies.
#[track_caller]
fn assert_proved_prime(x: &Int) -> Primality {
    let outcome = prove_prime(x);
    match &outcome {
        Primality::Prime(cert) => {
            assert!(cert.verify(x), "certificate failed to verify for {x}");
            assert_eq!(cert.n(), x, "certificate is about the wrong number");
        }
        other => panic!("expected Prime for {x}, got {other:?}"),
    }
    outcome
}

#[test]
fn small_primes_are_proved() {
    for p in [2u64, 3, 5, 7, 11, 13, 97, 101, 7919, 104729] {
        assert_proved_prime(&Int::from(p));
    }
}

#[test]
fn small_composites_are_rejected() {
    // Includes 0 and 1 (neither prime) and Carmichael numbers that fool Fermat.
    for c in [
        0u64, 1, 4, 6, 8, 9, 15, 100, 561, 1105, 1729, 41041, 63973, 104728,
    ] {
        assert!(
            matches!(prove_prime(&Int::from(c)), Primality::Composite),
            "{c} should be Composite"
        );
    }
}

#[test]
fn negatives_are_not_prime() {
    assert!(matches!(prove_prime(&Int::from(-7)), Primality::Composite));
    assert!(matches!(prove_prime(&Int::from(-1)), Primality::Composite));
}

#[test]
fn mersenne_prime_2p89_pocklington() {
    // 2^89 − 1 is prime; n − 1 = 2·(2^88 − 1) factors into small primes, so the
    // full √n Pocklington bound is reached.
    let m = mersenne(89);
    assert!(m.is_prime_bpsw());
    let outcome = assert_proved_prime(&m);
    if let Primality::Prime(cert) = outcome {
        assert_eq!(cert.bound(), Some(Bound::Sqrt));
    }
}

#[test]
fn mersenne_prime_2p127_bls() {
    // 2^127 − 1 (39 digits): here the fully-factored part of n − 1 only clears
    // the n^{1/3} bound, exercising the BLS discriminant test.
    let m = mersenne(127);
    assert!(m.is_prime_bpsw());
    let outcome = assert_proved_prime(&m);
    if let Primality::Prime(cert) = outcome {
        assert_eq!(cert.bound(), Some(Bound::Cbrt));
    }
}

#[test]
fn large_proth_primes_are_proved() {
    // Proth primes k·2^m + 1 have a trivially smooth n − 1 = k·2^m.
    // 102·2^256 + 1 (80 digits) and 267·2^300 + 1 (93 digits).
    let p80 = Int::from(102).mul(&Int::from(2).pow(256)).add(&Int::ONE);
    assert_eq!(p80.to_string().len(), 80);
    assert!(p80.is_prime_bpsw());
    assert_proved_prime(&p80);

    let p93 = Int::from(267).mul(&Int::from(2).pow(300)).add(&Int::ONE);
    assert_eq!(p93.to_string().len(), 93);
    assert!(p93.is_prime_bpsw());
    assert_proved_prime(&p93);
}

#[test]
fn hundred_digit_prime_is_proved() {
    // 135·2^330 + 1 is a 102-digit Proth prime.
    let p = Int::from(135).mul(&Int::from(2).pow(330)).add(&Int::ONE);
    assert_eq!(p.to_string().len(), 102);
    assert!(p.is_prime_bpsw());
    assert_proved_prime(&p);
}

#[test]
fn product_of_two_large_primes_is_composite() {
    // (2^61 − 1)·(2^89 − 1): a product of two large Mersenne primes.
    let semiprime = mersenne(61).mul(&mersenne(89));
    assert!(matches!(prove_prime(&semiprime), Primality::Composite));

    // A balanced product of two distinct ~40-digit primes.
    let p = Int::from(10).pow(40).next_prime();
    let q = Int::from(7).mul(&Int::from(10).pow(39)).next_prime();
    assert!(p != q && p.is_prime_bpsw() && q.is_prime_bpsw());
    assert!(matches!(prove_prime(&p.mul(&q)), Primality::Composite));
}

#[test]
fn certificate_bound_to_wrong_number_fails() {
    let m = mersenne(89);
    let cert = match prove_prime(&m) {
        Primality::Prime(c) => c,
        other => panic!("expected Prime, got {other:?}"),
    };
    assert!(cert.verify(&m));
    // Same certificate, different (also prime) target: must be rejected.
    assert!(!cert.verify(&mersenne(127)));
    // A composite target: also rejected.
    assert!(!cert.verify(&m.add(&Int::from(2))));
}

#[test]
fn ecpp_proves_when_n_minus_1_is_hard() {
    // Build a probable prime n whose n − 1 = 112·p·q with p, q two ~40-digit
    // primes: the smooth part (112) is far below n^{1/3}, and p·q is a large hard
    // composite the bounded factoring refuses to split — so no n − 1 proof exists.
    // Atkin–Morain ECPP takes over and produces a verifiable certificate whose
    // recorded bound is `None` (it is not an n − 1 proof).
    let p = Int::from(10).pow(40).next_prime();
    let q = Int::from(3).mul(&Int::from(10).pow(40)).next_prime();
    let n = Int::from(112).mul(&p).mul(&q).add(&Int::ONE);
    assert!(
        n.is_prime_bpsw(),
        "constructed n must be a (probable) prime"
    );
    match prove_prime(&n) {
        Primality::Prime(cert) => {
            assert_eq!(
                cert.bound(),
                None,
                "n − 1 is hard, so this must be an ECPP proof"
            );
            assert!(cert.verify(&n), "the ECPP certificate must verify");
            // Wrong target must be rejected.
            assert!(!cert.verify(&n.add(&Int::from(2))));
        }
        other => panic!("ECPP should prove this n − 1-hard prime, got {other:?}"),
    }
}

#[test]
fn agrees_with_bpsw_over_a_range() {
    // The proof result must agree with Baillie–PSW: Prime ⟺ bpsw-prime, and a
    // non-Prime outcome for a composite is always the certain Composite (never
    // Unproven) in this small range.
    for k in 0u64..1500 {
        let x = Int::from(k);
        let bpsw = x.is_prime_bpsw();
        match prove_prime(&x) {
            Primality::Prime(cert) => {
                assert!(bpsw, "{k}: proved prime but bpsw says composite");
                assert!(cert.verify(&x));
            }
            Primality::Composite => assert!(!bpsw, "{k}: said composite but bpsw says prime"),
            Primality::Unproven => panic!("{k}: small values should never be Unproven"),
        }
    }
}

#[test]
fn agrees_with_bpsw_on_larger_samples() {
    // A scattering of larger values straddling 2^64 and beyond.
    let samples = [
        n("18446744073709551557"),                    // largest prime below 2^64
        n("18446744073709551615"),                    // 2^64 − 1 (composite)
        n("18446744073709551629"),                    // smallest prime above 2^64
        n("340282366920938463463374607431768211297"), // a prime near 2^128
        n("340282366920938463463374607431768211455"), // 2^128 − 1 (composite)
    ];
    for x in samples {
        let bpsw = x.is_prime_bpsw();
        match prove_prime(&x) {
            Primality::Prime(cert) => {
                assert!(bpsw, "{x}: proved prime but bpsw disagrees");
                assert!(cert.verify(&x));
            }
            Primality::Composite => assert!(!bpsw, "{x}: composite but bpsw says prime"),
            Primality::Unproven => assert!(bpsw, "{x}: only probable primes may be Unproven"),
        }
    }
}
