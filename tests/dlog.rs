//! Tests for discrete-logarithm solving (`dlog` feature): baby-step giant-step
//! and Pollard's rho, checked against brute force and known values.
#![cfg(feature = "dlog")]

use puremp::dlog::{bsgs, discrete_log, pohlig_hellman, pollard_rho};
use puremp::{Int, ModInt};

/// Brute-force reference: least non-negative `x < order` with `g^x ≡ h (mod n)`.
fn brute(g: u64, h: u64, n: u64, order: u64) -> Option<u64> {
    let mut acc = 1u64 % n;
    for x in 0..order {
        if acc == h % n {
            return Some(x);
        }
        acc = (acc * g) % n;
    }
    None
}

fn i(v: u64) -> Int {
    Int::from(v)
}

#[test]
fn bsgs_matches_brute_force() {
    // (ℤ/101ℤ)* is cyclic of order 100; 2 is a generator.
    let (modulus, order) = (101u64, 100u64);
    for target in 1..modulus {
        let expected = brute(2, target, modulus, order);
        let got = bsgs(&i(2), &i(target), &i(modulus), &i(order));
        assert_eq!(
            got.as_ref().and_then(Int::to_u64),
            expected,
            "target={target}"
        );
    }
}

#[test]
fn bsgs_known_values() {
    // 2^37 mod 101 = 55; recover the exponent.
    let n = i(101);
    let g = i(2);
    let h = g.modpow(&i(37), &n);
    assert_eq!(h, i(55));
    assert_eq!(bsgs(&g, &h, &n, &i(100)), Some(i(37)));

    // Identity target -> exponent 0.
    assert_eq!(bsgs(&g, &Int::ONE, &n, &i(100)), Some(Int::ZERO));
}

#[test]
fn bsgs_least_solution() {
    // 5 has order 4 modulo 13: 5,12,8,1. The class of x=1 also contains x=5,
    // 9, ...; the least (1) must be returned even with a loose order bound.
    let n = i(13);
    let g = i(5);
    let h = g.modpow(&i(1), &n); // = 5
    assert_eq!(bsgs(&g, &h, &n, &i(12)), Some(i(1)));
}

#[test]
fn no_solution_returns_none() {
    // 3 generates the subgroup {1,3,9,5,4} of order 5 in (ℤ/11ℤ)*; 2 is not in
    // it, so 3^x ≡ 2 has no solution.
    let n = i(11);
    let g = i(3);
    assert_eq!(brute(3, 2, 11, 5), None);
    assert_eq!(bsgs(&g, &i(2), &n, &i(5)), None);
    assert_eq!(discrete_log(&g, &i(2), &n, &i(5)), None);

    // No power of 2 is ≡ 0 mod a prime.
    assert_eq!(discrete_log(&i(2), &Int::ZERO, &i(101), &i(100)), None);
}

#[test]
fn pollard_rho_matches_brute_force() {
    // Prime 1019, base 3 (order 1018). Check several targets; retry over seeds.
    let (g, n, order) = (i(3), i(1019), i(1018));
    for &e in &[1u64, 2, 17, 222, 500, 1017] {
        let h = g.modpow(&i(e), &n);
        let x = (0..16)
            .find_map(|s| pollard_rho(&g, &h, &n, &order, s))
            .expect("rho should converge within 16 seeds");
        assert_eq!(g.modpow(&x, &n), h, "e={e}");
    }
}

#[test]
fn discrete_log_random_roundtrip() {
    // For many (base, exponent) pairs, h = g^x, then recover and verify.
    // A small deterministic LCG stands in for randomness (no_std-friendly test).
    let n = i(1_000_003); // prime
    let order = i(1_000_002);
    let mut state = 0x1234_5678u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    for _ in 0..40 {
        let g_val = 2 + next() % 1000;
        let x = next() % 1_000_002;
        let g = i(g_val);
        let h = g.modpow(&i(x), &n);
        let found = discrete_log(&g, &h, &n, &order).expect("solution exists (h = g^x)");
        assert_eq!(g.modpow(&found, &n), h, "g={g_val} x={x}");
    }
}

#[test]
fn discrete_log_dispatches_to_rho_for_large_order() {
    // A prime just above 2^40 forces the Pollard-rho branch (order bit_len > 40).
    let p = Int::from(1u64 << 40).next_prime();
    let order = p.sub(&Int::ONE);
    assert!(order.magnitude().bit_len() > 40);
    let g = i(2);
    let x = i(123_456_789);
    let h = g.modpow(&x, &p);
    let found = discrete_log(&g, &h, &p, &order).expect("rho should find a solution");
    assert_eq!(g.modpow(&found, &p), h);
}

#[test]
fn modint_method() {
    let g = ModInt::new(i(2), i(101));
    let h = g.pow(&i(73));
    assert_eq!(g.discrete_log(&h, &i(100)), Some(i(73)));

    // No-solution case through the method.
    let g3 = ModInt::new(i(3), i(11));
    let two = ModInt::new(i(2), i(11));
    assert_eq!(g3.discrete_log(&two, &i(5)), None);
}

#[test]
fn degenerate_inputs() {
    // Identity target is always exponent 0.
    assert_eq!(
        discrete_log(&i(7), &Int::ONE, &i(97), &i(96)),
        Some(Int::ZERO)
    );
    // Modulus 1: everything collapses to 0.
    assert_eq!(
        discrete_log(&i(7), &i(5), &Int::ONE, &i(10)),
        Some(Int::ZERO)
    );
}

#[test]
fn pohlig_hellman_matches_brute_force() {
    // (ℤ/1009ℤ)* is cyclic of order 1008 = 2^4·3^2·7 (smooth); 11 is a generator.
    // Cross-check every recovered exponent against the brute-force reference.
    let (modulus, order) = (1009u64, 1008u64);
    for e in 0..order {
        let h = i(11).modpow(&i(e), &i(modulus));
        let got = pohlig_hellman(&i(11), &h, &i(modulus), &i(order));
        let x = got.expect("solution exists (h = g^e)");
        // Generator ⇒ order is exact ⇒ the unique exponent equals e.
        assert_eq!(x.to_u64(), Some(e), "e={e}");
        assert_eq!(i(11).modpow(&x, &i(modulus)), h);
    }
}

#[test]
fn pohlig_hellman_known_and_identity() {
    // Known value with a smooth prime.
    let (g, n, order) = (i(11), i(1009), i(1008));
    let h = g.modpow(&i(555), &n);
    assert_eq!(h, i(149));
    let x = pohlig_hellman(&g, &h, &n, &order).unwrap();
    assert_eq!(x, i(555));
    assert_eq!(g.modpow(&x, &n), h);

    // Identity target -> exponent 0.
    assert_eq!(pohlig_hellman(&g, &Int::ONE, &n, &order), Some(Int::ZERO));
}

#[test]
fn pohlig_hellman_no_solution() {
    // 3 generates the order-5 subgroup {1,3,9,5,4} of (ℤ/11ℤ)*; 2 is outside it.
    // Order 5 is prime, but PH must still report no solution (subgroup dlog None).
    assert_eq!(pohlig_hellman(&i(3), &i(2), &i(11), &i(5)), None);

    // No power of a unit is ≡ 0 modulo a prime.
    assert_eq!(pohlig_hellman(&i(11), &Int::ZERO, &i(1009), &i(1008)), None);
}

#[test]
fn pohlig_hellman_smooth_advantage_large_order() {
    // p - 1 = 2^34·3·5^2 is smooth, and the order is > 2^40, so a naive
    // baby-step table (~2^20.5 entries) would be costly while Pohlig–Hellman,
    // whose per-subgroup cost is Σ eᵢ·√pᵢ (all primes ≤ 5), is fast.
    let p = Int::from(1_288_490_188_801u64);
    let order = p.sub(&Int::ONE); // 1_288_490_188_800
    assert!(order.magnitude().bit_len() > 40);
    let g = i(11); // a generator of (ℤ/pℤ)*
    let x = i(987_654_321);
    let h = g.modpow(&x, &p);
    let found = pohlig_hellman(&g, &h, &p, &order).expect("PH must find the log");
    assert_eq!(found, x);
    assert_eq!(g.modpow(&found, &p), h);

    // The dispatcher routes composite orders through PH, so it is fast here too.
    let via_dispatch = discrete_log(&g, &h, &p, &order).expect("dispatch to PH");
    assert_eq!(g.modpow(&via_dispatch, &p), h);
}

/// `g^e mod n` for small `u64`, for computing multiplicative orders in tests.
fn pow_mod(mut g: u64, mut e: u64, n: u64) -> u64 {
    let (mut r, m) = (1u64 % n, n as u128);
    g %= n;
    while e > 0 {
        if e & 1 == 1 {
            r = ((r as u128 * g as u128) % m) as u64;
        }
        g = ((g as u128 * g as u128) % m) as u64;
        e >>= 1;
    }
    r
}

/// Multiplicative order of `g` modulo 1009 (a divisor of 1008 = 2^4·3^2·7).
fn order_mod_1009(g: u64) -> u64 {
    let mut t = 1008u64;
    for &p in &[2u64, 3, 7] {
        while t.is_multiple_of(p) && pow_mod(g, t / p, 1009) == 1 {
            t /= p;
        }
    }
    t
}

#[test]
fn pohlig_hellman_random_roundtrips() {
    // Several (base, exponent) round-trips over the smooth-order group (ℤ/1009ℤ)*.
    // Pohlig–Hellman needs the *exact* order of the base, so it is computed per g.
    let n = i(1009);
    let mut state = 0x9e37_79b9u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    for _ in 0..40 {
        // 1009 is prime, so every base in [2, 1008] is a unit.
        let g_val = 2 + next() % 1006;
        let ord = order_mod_1009(g_val);
        let x = next() % ord;
        let g = i(g_val);
        let h = g.modpow(&i(x), &n);
        let found = pohlig_hellman(&g, &h, &n, &i(ord)).expect("solution exists");
        // With the exact order, the representative in [0, ord) is unique.
        assert_eq!(found, i(x), "g={g_val} x={x}");
        assert_eq!(g.modpow(&found, &n), h, "g={g_val} x={x}");
    }
}
