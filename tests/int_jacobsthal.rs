//! Jacobsthal's function: the covering search cross-checked against a direct
//! scan over one period `rad(n)`.
#![cfg(feature = "int")]

use puremp::Int;

fn int(v: u64) -> Int {
    Int::from_u64(v)
}

/// Distinct prime factors of `n ≥ 1`, by trial division.
fn distinct_primes(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut p = 2;
    while p * p <= n {
        if n.is_multiple_of(p) {
            out.push(p);
            while n.is_multiple_of(p) {
                n /= p;
            }
        }
        p += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Direct period scan: `(j, start)` where `j − 1` is the longest run of
/// integers in one period of `rad = ∏ primes` each divisible by some prime,
/// and `start` is the smallest positive start of such a run (`None` if the
/// longest run is empty). Independent of the library's search.
fn scan(primes: &[u64]) -> (u64, Option<u64>) {
    let rad: u64 = primes.iter().product();
    if rad == 1 {
        return (1, None);
    }
    let mut hit = vec![false; rad as usize + 2];
    for &p in primes {
        for m in (0..hit.len()).step_by(p as usize) {
            hit[m] = true;
        }
    }
    // 1 and rad + 1 are coprime to rad, so runs inside [1, rad + 1] are the
    // runs of one full period.
    let (mut best, mut best_start, mut run) = (0u64, None, 0u64);
    for x in 1..=rad + 1 {
        if hit[x as usize] {
            run += 1;
        } else {
            if run > best {
                best = run;
                best_start = Some(x - run);
            }
            run = 0;
        }
    }
    (best + 1, best_start)
}

fn first_primes(k: usize) -> Vec<u64> {
    let mut out = Vec::new();
    let mut p = 2;
    while out.len() < k {
        if distinct_primes(p) == [p] {
            out.push(p);
        }
        p += 1;
    }
    out
}

/// Checks that `s, …, s + len − 1` all share a factor with `n` and that the
/// run is maximal.
fn check_witness(n: &Int, s: &Int, j: &Int) {
    assert!(s.is_positive(), "witness start must be positive");
    let len = j.to_u64().unwrap() - 1;
    for i in 0..len {
        let x = s.add(&int(i));
        assert!(!x.gcd(n).is_one(), "n={n} s={s}: {x} is coprime");
    }
    assert!(s.sub(&Int::ONE).gcd(n).is_one(), "n={n}: run extends left");
    assert!(s.add(&int(len)).gcd(n).is_one(), "n={n}: run extends right");
}

#[test]
fn matches_period_scan_up_to_3000() {
    for n in 1..=3000u64 {
        let primes = distinct_primes(n);
        let (j, _) = scan(&primes);
        let ni = int(n);
        assert_eq!(ni.jacobsthal(), Some(int(j)), "j({n})");
        assert_eq!(ni.neg().jacobsthal(), Some(int(j)), "j(-{n})");
        match ni.jacobsthal_witness() {
            Some(s) => check_witness(&ni, &s, &int(j)),
            None => assert_eq!(n, 1),
        }
    }
}

#[test]
fn primorials_match_period_scan() {
    for k in 0..=8usize {
        let (j, _) = scan(&first_primes(k));
        assert_eq!(Int::jacobsthal_primorial(k as u32), Some(int(j)), "h({k})");
    }
}

#[test]
fn odd_and_sparse_prime_sets_match_scan() {
    let sets: &[&[u64]] = &[
        &[3, 5, 7, 11, 13, 17],
        &[3, 5, 7, 11, 13, 17, 19],
        &[2, 5, 7, 11, 13, 17, 19],
        &[2, 3, 7, 11, 13, 17],
        &[5, 7, 11, 13, 17, 19],
        &[2, 3, 5, 7, 29, 31, 37],
        &[2, 101, 103, 107],
    ];
    for primes in sets {
        let n = primes.iter().fold(Int::ONE, |m, &p| m.mul(&int(p)));
        let (j, _) = scan(primes);
        assert_eq!(n.jacobsthal(), Some(int(j)), "j({primes:?})");
        check_witness(&n, &n.jacobsthal_witness().unwrap(), &int(j));
    }
}

/// OEIS A048670, h(1..=14); the first eight are re-derived by the scan above.
const H: [u64; 14] = [2, 4, 6, 10, 14, 22, 26, 34, 40, 46, 58, 66, 74, 90];

#[test]
fn primorial_values() {
    assert_eq!(Int::jacobsthal_primorial(0), Some(Int::ONE));
    for (k, &h) in H.iter().enumerate() {
        let k = k + 1;
        assert_eq!(Int::jacobsthal_primorial(k as u32), Some(int(h)), "h({k})");
    }
    // The primorial itself, through factorization, with a witness.
    let n = first_primes(12)
        .iter()
        .fold(Int::ONE, |m, &p| m.mul(&int(p)));
    assert_eq!(n.jacobsthal(), Some(int(H[11])));
    check_witness(&n, &n.jacobsthal_witness().unwrap(), &int(H[11]));
}

#[test]
fn conventions_and_cap() {
    assert_eq!(Int::ZERO.jacobsthal(), None);
    assert_eq!(Int::ZERO.jacobsthal_witness(), None);
    assert_eq!(Int::ONE.jacobsthal(), Some(Int::ONE));
    assert_eq!(Int::ONE.neg().jacobsthal(), Some(Int::ONE));
    assert_eq!(Int::ONE.jacobsthal_witness(), None);
    // Exponents do not matter.
    assert_eq!(int(2u64.pow(40)).jacobsthal(), Some(int(2)));
    assert_eq!(int(2 * 2 * 3 * 3 * 3 * 5).jacobsthal(), Some(int(6)));

    let cap = Int::JACOBSTHAL_MAX_PRIMES;
    assert_eq!(Int::jacobsthal_primorial(cap as u32 + 1), None);
    // cap + 1 primes that each cover one position: rejected by the cap.
    let mut n = Int::ONE;
    let mut p = int(1000);
    for _ in 0..=cap {
        p = p.next_prime();
        n = n.mul(&p);
    }
    assert_eq!(n.jacobsthal(), None);
    assert_eq!(n.jacobsthal_witness(), None);
}

#[test]
fn primes_beyond_u64() {
    // Only one large prime per n so that factoring stays trivial.
    let p = Int::ONE.mul_2k(89).sub(&Int::ONE); // Mersenne prime M89
    assert_eq!(p.jacobsthal(), Some(int(2)));
    check_witness(&p, &p.jacobsthal_witness().unwrap(), &int(2));
    // {2, 3, P} behaves like {2, 3, 5}: j = 6.
    let n = p.mul(&int(6));
    assert_eq!(n.jacobsthal(), Some(int(6)));
    check_witness(&n, &n.jacobsthal_witness().unwrap(), &int(6));
    // {2, 3, 5, 7, 11, Q} with Q = M127 behaves like the 6th primorial.
    let q = Int::ONE.mul_2k(127).sub(&Int::ONE);
    let n = q.mul(&int(2 * 3 * 5 * 7 * 11));
    assert_eq!(n.jacobsthal(), Some(int(22)));
    check_witness(&n, &n.jacobsthal_witness().unwrap(), &int(22));
}

/// OEIS A048670, h(15..=24). Slow in debug builds; run with
/// `cargo test --release --test int_jacobsthal -- --ignored --nocapture`.
#[test]
#[ignore]
fn primorial_values_heavy() {
    let h = [100u64, 106, 118, 132, 152, 174, 190, 200, 216, 234];
    for (i, &v) in h.iter().enumerate() {
        let k = 15 + i as u32;
        let t = std::time::Instant::now();
        assert_eq!(Int::jacobsthal_primorial(k), Some(int(v)), "h({k})");
        println!("h({k}) = {v}  [{:?}]", t.elapsed());
    }
}
