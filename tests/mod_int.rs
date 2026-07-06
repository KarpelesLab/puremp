//! Tests for the modular-integer type.
#![cfg(feature = "int")]

use puremp::random::SeedRng;
use puremp::{Int, ModInt, Nat};

fn m(v: i64, modulus: i64) -> ModInt {
    ModInt::new(Int::from(v), Int::from(modulus))
}

#[test]
fn modular_operators() {
    // mod 7
    let a = m(5, 7);
    let b = m(4, 7);
    assert_eq!((&a + &b).to_int().to_string(), "2"); // 9 mod 7
    assert_eq!((&a - &b).to_int().to_string(), "1");
    assert_eq!((&b - &a).to_int().to_string(), "6"); // -1 mod 7
    assert_eq!((&a * &b).to_int().to_string(), "6"); // 20 mod 7
    assert_eq!((-a.clone()).to_int().to_string(), "2"); // -5 mod 7

    // inverse and division
    assert_eq!(m(3, 7).inv().unwrap().to_int().to_string(), "5"); // 3*5=15≡1
    assert!(m(2, 6).inv().is_none()); // gcd(2,6)=2
    assert_eq!((&a / &b).to_int(), (&a * &b.inv().unwrap()).to_int());

    // pow, incl. Fermat and negative exponent
    let p = Int::from(1000000007);
    let g = ModInt::new(Int::from(123456), p.clone());
    assert!(g.pow(&p.sub(&Int::ONE)).to_int().is_one()); // a^(p-1) ≡ 1
    assert_eq!(g.pow(&Int::from(-1)), g.inv().unwrap()); // a^-1

    // `of` shares the ring
    let x = m(2, 97);
    let y = x.of(Int::from(50));
    assert_eq!((&x * &y).to_int().to_string(), "3"); // 100 mod 97

    // Assign ops and equality
    let mut z = m(10, 13);
    z *= m(10, 13);
    assert_eq!(z, m(9, 13)); // 100 mod 13
    assert_eq!(m(20, 13), m(7, 13));
}

/// Differential check: every `ModInt` operation must return the exact residue an
/// independent `Int`-based reference produces, for many random operands and
/// moduli of assorted sizes — covering both odd moduli (Montgomery-resident
/// internally) and even moduli (Barrett path). This pins the Montgomery path to
/// be bit-identical to plain modular arithmetic.
#[test]
fn differential_vs_int_reference() {
    let mut rng = SeedRng::new(0x0DDE_AE45_1234_5678);
    // Reference residue in [0, m) as an Int.
    let refmod = |v: &Int, m: &Int| v.rem_euclid(m);
    for &bits in &[8u64, 32, 64, 96, 200, 300, 512, 1200] {
        for trial in 0..40 {
            // Random modulus >= 2, then two variants: forced odd and forced even.
            let base = Int::from(Nat::random_bits(bits.max(2), &mut rng)) + Int::from(2);
            for parity_even in [false, true] {
                // Set the low bit to pick parity while keeping m >= 2.
                let m = if parity_even {
                    &base - (&base % &Int::from(2)) // clear low bit -> even (>= 2)
                } else {
                    let e = &base - (&base % &Int::from(2));
                    &e + Int::from(1) // make odd (>= 3)
                };
                if m < Int::from(2) {
                    continue;
                }
                let a_int = Int::from(Nat::random_bits(bits + 16, &mut rng));
                let b_int = Int::from(Nat::random_bits(bits + 16, &mut rng));
                let a = ModInt::new(a_int.clone(), m.clone());
                let b = ModInt::new(b_int.clone(), m.clone());

                assert_eq!((&a + &b).to_int(), refmod(&(&a_int + &b_int), &m));
                assert_eq!((&a - &b).to_int(), refmod(&(&a_int - &b_int), &m));
                assert_eq!((&b - &a).to_int(), refmod(&(&b_int - &a_int), &m));
                assert_eq!((&a * &b).to_int(), refmod(&(&a_int * &b_int), &m));
                assert_eq!((-a.clone()).to_int(), refmod(&(-&a_int), &m));

                // `residue()` and `Display` agree with `to_int`.
                assert_eq!(Int::from(a.residue()), a.to_int());
                assert_eq!(a.to_string(), a.to_int().to_string());

                // pow against an independent Int::modpow, small-ish exponents.
                let e = Int::from(Nat::random_bits(24, &mut rng));
                assert_eq!(a.pow(&e).to_int(), a_int.modpow(&e, &m));

                // inv: matches Int::modinv, and a·a⁻¹ ≡ 1 when invertible.
                match a.inv() {
                    Some(ai) => {
                        assert_eq!(Some(ai.to_int()), a_int.modinv(&m));
                        assert_eq!((&a * &ai).to_int(), refmod(&Int::ONE, &m));
                        // div = mul by inverse.
                        assert_eq!((&b / &a).to_int(), (&b * &ai).to_int());
                    }
                    None => assert!(a_int.modinv(&m).is_none()),
                }
                let _ = trial;
            }
        }
    }
}
