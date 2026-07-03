//! Tests for the modular-integer type.
#![cfg(feature = "int")]

use puremp::{Int, ModInt};

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
