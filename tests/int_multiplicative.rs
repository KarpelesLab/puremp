//! Multiplicative order, Carmichael λ, primitive roots and inverse totient,
//! cross-checked against brute force for small moduli.
#![cfg(feature = "int")]

use puremp::Int;

const N: u64 = 2000;

fn int(v: u64) -> Int {
    Int::from_u64(v)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Naive multiplicative order of `k` mod `n` (`gcd(k, n) = 1`, `n ≥ 2`).
fn naive_order(k: u64, n: u64) -> u64 {
    let mut x = k % n;
    let mut m = 1;
    while x != 1 {
        x = x * k % n;
        m += 1;
    }
    m
}

/// Totient sieve up to `limit` inclusive.
fn phi_table(limit: usize) -> Vec<u64> {
    let mut phi: Vec<u64> = (0..=limit as u64).collect();
    for p in 2..=limit {
        if phi[p] == p as u64 {
            for m in (p..=limit).step_by(p) {
                phi[m] -= phi[m] / p as u64;
            }
        }
    }
    phi
}

#[test]
fn order_lambda_roots_match_brute_force() {
    let phi = phi_table(N as usize);
    for n in 2..=N {
        let units: Vec<u64> = (1..n).filter(|&k| gcd(k, n) == 1).collect();
        let orders: Vec<u64> = units.iter().map(|&k| naive_order(k, n)).collect();
        // multiplicative_order (spot-check a spread of k, plus non-units → None).
        for (i, &k) in units.iter().enumerate() {
            if n <= 300 || i % 37 == 0 {
                assert_eq!(
                    int(k).multiplicative_order(&int(n)),
                    Some(int(orders[i])),
                    "ord({k}, {n})"
                );
            }
        }
        if let Some(k) = (2..n).find(|&k| gcd(k, n) != 1) {
            assert_eq!(int(k).multiplicative_order(&int(n)), None, "ord({k}, {n})");
        }
        // carmichael_lambda = max element order.
        let lambda = *orders.iter().max().unwrap();
        assert_eq!(int(n).carmichael_lambda(), int(lambda), "λ({n})");
        // primitive roots = units of order φ(n).
        let roots: Vec<Int> = units
            .iter()
            .zip(&orders)
            .filter(|&(_, &o)| o == phi[n as usize])
            .map(|(&k, _)| int(k))
            .collect();
        assert_eq!(
            int(n).primitive_root(),
            roots.first().cloned(),
            "PrimitiveRoot[{n}]"
        );
        if n <= 600 {
            assert_eq!(
                int(n).primitive_root_list(),
                roots,
                "PrimitiveRootList[{n}]"
            );
        }
    }
}

#[test]
fn inverse_totient_matches_sieve() {
    // n/φ(n) < 6 for every n < 2·3·5·…·23, so φ(n) ≤ N forces n < 6N.
    let limit = 6 * N as usize;
    let phi = phi_table(limit);
    let mut inv: Vec<Vec<Int>> = vec![Vec::new(); N as usize + 1];
    for n in 1..=limit {
        if phi[n] <= N {
            inv[phi[n] as usize].push(int(n as u64));
        }
    }
    for m in 0..=N {
        assert_eq!(int(m).inverse_totient(), inv[m as usize], "φ⁻¹({m})");
        assert_eq!(
            int(m).is_totient(),
            !inv[m as usize].is_empty(),
            "is_totient({m})"
        );
    }
}

#[test]
fn edge_cases() {
    // Order: n = ±1, n = 0, negative inputs.
    assert_eq!(int(5).multiplicative_order(&Int::ONE), Some(Int::ONE));
    assert_eq!(int(5).multiplicative_order(&Int::from(-1)), Some(Int::ONE));
    assert_eq!(Int::ONE.multiplicative_order(&Int::ZERO), Some(Int::ONE));
    assert_eq!(Int::from(-1).multiplicative_order(&Int::ZERO), Some(int(2)));
    assert_eq!(int(3).multiplicative_order(&Int::ZERO), None);
    assert_eq!(int(3).multiplicative_order(&Int::from(-7)), Some(int(6)));
    assert_eq!(Int::from(-1).multiplicative_order(&int(7)), Some(int(2)));
    assert_eq!(Int::from(-3).multiplicative_order(&int(7)), Some(int(3)));
    // λ.
    assert_eq!(Int::ZERO.carmichael_lambda(), Int::ZERO);
    assert_eq!(Int::ONE.carmichael_lambda(), Int::ONE);
    assert_eq!(Int::from(-15).carmichael_lambda(), int(4));
    assert_eq!(int(561).carmichael_lambda(), int(80));
    // Primitive roots.
    assert_eq!(Int::ZERO.primitive_root(), None);
    assert_eq!(Int::ONE.primitive_root(), Some(Int::ZERO));
    assert_eq!(Int::ONE.primitive_root_list(), vec![Int::ZERO]);
    assert_eq!(int(2).primitive_root_list(), vec![Int::ONE]);
    assert_eq!(int(8).primitive_root(), None);
    assert!(int(8).primitive_root_list().is_empty());
    assert_eq!(Int::from(-7).primitive_root(), Some(int(3)));
    assert_eq!(int(4).primitive_root(), Some(int(3)));
    // Inverse totient.
    assert_eq!(Int::ONE.inverse_totient(), vec![int(1), int(2)]);
    assert!(Int::ZERO.inverse_totient().is_empty());
    assert!(Int::from(-4).inverse_totient().is_empty());
    assert!(int(14).inverse_totient().is_empty());
    assert!(!int(14).is_totient());
    let want: Vec<Int> = [35u64, 39, 45, 52, 56, 70, 72, 78, 84, 90]
        .iter()
        .map(|&v| int(v))
        .collect();
    assert_eq!(int(24).inverse_totient(), want);
}

#[test]
fn large_prime_cases() {
    // p = 2^61 − 1 (Mersenne prime): p − 1 = 2·3²·5²·7·11·13·31·41·61·151·331·1321.
    let p = Int::ONE.mul_2k(61).sub(&Int::ONE);
    let pm1 = p.sub(&Int::ONE);
    assert_eq!(p.carmichael_lambda(), pm1);
    let g = p.primitive_root().unwrap();
    assert_eq!(g, int(37));
    assert_eq!(g.multiplicative_order(&p), Some(pm1.clone()));
    // 2 has order 61 mod 2^61 − 1.
    assert_eq!(int(2).multiplicative_order(&p), Some(int(61)));
    // A primitive root mod q² and 2q² has order q(q−1) (q = 2^31 − 1; a smaller
    // prime keeps the repeated factorization of q² cheap).
    let q = Int::ONE.mul_2k(31).sub(&Int::ONE);
    let lam = q.mul(&q.sub(&Int::ONE));
    for n in [q.square(), q.square().mul(&int(2))] {
        let r = n.primitive_root().unwrap();
        assert_eq!(r.multiplicative_order(&n), Some(lam.clone()));
        assert_eq!(n.carmichael_lambda(), lam);
    }
    assert_eq!(q.square().mul(&int(4)).primitive_root(), None);
    // φ⁻¹(p − 1) contains p and 2p.
    let inv = pm1.inverse_totient();
    assert!(inv.contains(&p) && inv.contains(&p.mul(&int(2))));
    for n in &inv {
        assert_eq!(n.euler_phi(), pm1);
    }
}

#[test]
fn inverse_totient_highly_composite() {
    // 10^12 = 2^12·5^12 (169 divisors) and 963761198400 (6720 divisors).
    for m in [1_000_000_000_000u64, 963_761_198_400] {
        let m = int(m);
        let inv = m.inverse_totient();
        assert!(!inv.is_empty());
        assert!(m.is_totient());
        assert!(inv.windows(2).all(|w| w[0] < w[1]));
        for n in inv.iter().step_by(inv.len() / 50 + 1) {
            assert_eq!(n.euler_phi(), m);
        }
    }
}
