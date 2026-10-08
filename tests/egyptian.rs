//! Egyptian-fraction expansions: exactness and distinctness of every method,
//! known minimum lengths, and agreement of the shortest search with an
//! independent exhaustive search for all `a/b` with `b ≤ 30`.
#![cfg(feature = "rational")]

use puremp::{EgyptianExpansion, Int, Rational, SeedRng, ShortestEgyptian};

fn q(a: i64, b: i64) -> Rational {
    Rational::new(Int::from(a), Int::from(b))
}

fn small(e: &EgyptianExpansion) -> Vec<u64> {
    e.denominators.iter().map(|d| d.to_u64().unwrap()).collect()
}

/// Independent validity check: strictly increasing denominators `≥ 2`, the
/// integer part is the floor, and the sum (accumulated term by term) is `x`.
fn check(x: &Rational, e: &EgyptianExpansion) {
    for w in e.denominators.windows(2) {
        assert!(w[0] < w[1], "{x}: denominators not strictly increasing");
    }
    if let Some(d) = e.denominators.first() {
        assert!(*d >= Int::from(2), "{x}: denominator below 2");
    }
    assert_eq!(e.integer_part, x.floor(), "{x}: integer part");
    let mut sum = Rational::from_integer(e.integer_part.clone());
    for d in &e.denominators {
        sum = sum.add(&Rational::new(Int::ONE, d.clone()));
    }
    assert_eq!(sum, *x, "{x}: expansion does not sum to the input");
    assert_eq!(e.value(), *x);
    assert!(e.is_valid_for(x));
}

fn random_int(rng: &mut SeedRng, digits: usize) -> Int {
    let mut s = String::new();
    s.push(char::from(b'1' + (rng.next_u64() % 9) as u8));
    for _ in 1..digits {
        s.push(char::from(b'0' + (rng.next_u64() % 10) as u8));
    }
    s.parse().unwrap()
}

#[test]
fn greedy_known_values() {
    assert_eq!(small(&q(4, 13).egyptian_greedy()), [4, 18, 468]);
    assert_eq!(small(&q(4, 5).egyptian_greedy()), [2, 4, 20]);
    assert_eq!(small(&q(4, 17).egyptian_greedy()), [5, 29, 1233, 3039345]);
    let e = q(5, 121).egyptian_greedy();
    assert_eq!(e.len(), 5);
    assert_eq!(
        e.max_denominator().unwrap().to_string(),
        "1527612795642093418846225"
    );
    // The notorious 31/311: ten terms, the last one 537 digits long.
    let e = q(31, 311).egyptian_greedy();
    check(&q(31, 311), &e);
    assert_eq!(e.len(), 10);
    assert_eq!(e.max_denominator().unwrap().to_string().len(), 537);
}

#[test]
fn integer_parts_and_signs() {
    let e = q(7, 3).egyptian_short();
    assert_eq!(e.integer_part, Int::from(2));
    assert_eq!(small(&e), [3]);
    for x in [q(-1, 3), q(-22, 7), q(17, 4), q(5, 1), q(0, 1), q(-3, 1)] {
        check(&x, &x.egyptian_greedy());
        check(&x, &x.egyptian_short());
        check(&x, &x.egyptian_shortest(5).unwrap());
    }
    assert_eq!(small(&q(-1, 3).egyptian_shortest(3).unwrap()), [2, 6]);
    assert!(q(5, 1).egyptian_greedy().is_empty());
}

#[test]
fn shortest_known_values() {
    // 4/5 has two 3-term expansions, 1/2+1/4+1/20 and 1/2+1/5+1/10; the
    // smaller largest denominator wins.
    assert_eq!(small(&q(4, 5).egyptian_shortest(5).unwrap()), [2, 5, 10]);
    // 4/13 and 5/121 are not sums of two unit fractions.
    assert_eq!(q(4, 13).egyptian_shortest(5).unwrap().len(), 3);
    assert_eq!(
        small(&q(5, 121).egyptian_shortest(5).unwrap()),
        [33, 121, 363]
    );
    assert_eq!(small(&q(4, 17).egyptian_shortest(5).unwrap()), [6, 17, 102]);
    let e = q(31, 311).egyptian_shortest(5).unwrap();
    assert_eq!(small(&e), [12, 63, 2799, 8708]);
    check(&q(31, 311), &e);
}

#[test]
fn shortest_outcomes() {
    assert_eq!(
        q(4, 5).egyptian_shortest_with_budget(2, 1_000_000),
        ShortestEgyptian::NoneWithin
    );
    assert_eq!(q(4, 5).egyptian_shortest(2), None);
    assert_eq!(
        q(31, 311).egyptian_shortest_with_budget(5, 10),
        ShortestEgyptian::BudgetExhausted
    );
    match q(3, 1).egyptian_shortest_with_budget(0, 0) {
        ShortestEgyptian::Found(e) => assert!(e.is_empty() && e.integer_part == Int::from(3)),
        other => panic!("{other:?}"),
    }
    assert_eq!(small(&q(1, 7).egyptian_shortest(1).unwrap()), [7]);
}

/// Exhaustive search for all `k`-term expansions of `p/q` with increasing
/// denominators `≥ min`, using only the necessary bounds `1/r < d < k/r`
/// (no two-term shortcut, no pruning by the best solution so far).
fn all_expansions(
    p: u128,
    q: u128,
    k: usize,
    min: u128,
    cur: &mut Vec<u128>,
    out: &mut Vec<Vec<u128>>,
) {
    if k == 1 {
        if p == 1 && q >= min {
            cur.push(q);
            out.push(cur.clone());
            cur.pop();
        }
        return;
    }
    let lo = (q / p + 1).max(min);
    let hi = (k as u128 * q - 1) / p;
    for d in lo..=hi {
        let (np, nq) = (p * d - q, q * d);
        let g = gcd(np, nq);
        cur.push(d);
        all_expansions(np / g, nq / g, k - 1, d + 1, cur, out);
        cur.pop();
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[test]
fn shortest_matches_brute_force() {
    let mut lengths = [0usize; 6];
    for b in 2..=30u128 {
        for a in 1..b {
            if gcd(a, b) != 1 {
                continue;
            }
            let x = q(a as i64, b as i64);
            let got = x.egyptian_shortest(5).unwrap();
            check(&x, &got);
            let mut want = None;
            for k in 1..=5 {
                let mut sols = Vec::new();
                all_expansions(a, b, k, 2, &mut Vec::new(), &mut sols);
                if let Some(best) = sols
                    .into_iter()
                    .min_by(|u, v| (u.last(), &**u).cmp(&(v.last(), &**v)))
                {
                    want = Some(best);
                    break;
                }
            }
            let want = want.expect("every a/b with b ≤ 30 needs at most 5 terms");
            let got: Vec<u128> = got
                .denominators
                .iter()
                .map(|d| d.to_u128().unwrap())
                .collect();
            assert_eq!(got, want, "{a}/{b}");
            lengths[got.len()] += 1;
        }
    }
    // Sanity: the brute force exercised every length from 1 to 5.
    assert!(lengths[1..=5].iter().all(|&n| n > 0), "{lengths:?}");
}

#[test]
fn greedy_and_short_random_small() {
    let mut rng = SeedRng::new(0x5eed_e9e7);
    for _ in 0..300 {
        let b = 2 + (rng.next_u64() % 5000) as i64;
        let a = 1 + (rng.next_u64() % (b as u64 - 1)) as i64;
        let x = q(a, b);
        check(&x, &x.egyptian_short());
        if a <= 40 {
            check(&x, &x.egyptian_greedy());
        }
        if b <= 200 {
            check(
                &x,
                &x.egyptian_shortest(4).unwrap_or_else(|| x.egyptian_short()),
            );
        }
    }
}

#[test]
fn greedy_and_short_large_denominators() {
    let mut rng = SeedRng::new(0xe9_7957);
    for digits in [7, 20, 50, 100, 120] {
        for _ in 0..3 {
            let b = random_int(&mut rng, digits);
            // Greedy blows up with the numerator, so feed it small ones.
            let a = Int::from(1 + rng.next_u64() % 12);
            let x = Rational::new(a, b.clone());
            check(&x, &x.egyptian_greedy());
            check(&x, &x.egyptian_short());
            // Short handles full-size numerators too.
            let a = random_int(&mut rng, digits - 1);
            let x = Rational::new(a, b.clone());
            let e = x.egyptian_short();
            check(&x, &e);
            let bits = x.denominator().bit_len();
            assert!(e.max_denominator().unwrap().bit_len() <= 2 * bits + 64);
            // Mixed numbers with a huge denominator.
            let x = x.add(&Rational::from_integer(Int::from(-7)));
            check(&x, &x.egyptian_short());
        }
    }
}

#[test]
fn short_beats_greedy_on_notorious_cases() {
    for (a, b) in [(5, 121), (31, 311), (4, 17), (99, 100), (17, 19)] {
        let x = q(a, b);
        let g = x.egyptian_greedy();
        let s = x.egyptian_short();
        check(&x, &s);
        assert!(s.len() <= g.len(), "{a}/{b}");
        assert!(s.max_denominator() < g.max_denominator(), "{a}/{b}");
    }
    assert_eq!(small(&q(5, 121).egyptian_short()), [33, 121, 363]);
    let s = q(31, 311).egyptian_short();
    assert!(s.len() <= 4 && *s.max_denominator().unwrap() < Int::from(100_000));
}
