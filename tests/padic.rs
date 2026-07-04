//! Integration tests for [`puremp::Padic`] — fixed-precision `p`-adic numbers.
//!
//! These check known facts of `ℤ_p` / `ℚ_p`: the all-ones expansion of `-1` in
//! `ℤ₂`, the `1/3` reciprocal, valuation bookkeeping (including negative
//! valuations in `ℚ_p`), the arithmetic identities, and Hensel-lifted square
//! roots for both odd `p` and `p = 2`.

#![cfg(feature = "padic")]

use puremp::padic::Padic;
use puremp::{Int, Rational};

fn p(v: i64) -> Int {
    Int::from_i64(v)
}

fn rat(a: i64, b: i64) -> Rational {
    Rational::new(Int::from_i64(a), Int::from_i64(b))
}

#[test]
fn minus_one_in_z2_is_all_ones() {
    // -1 = …1111 in ℤ₂: every 2-adic digit is 1.
    let n = 8;
    let x = Padic::from_int(p(2), n, p(-1));
    assert_eq!(x.valuation(), Some(0));
    let digits = x.digits();
    assert_eq!(digits.len(), n as usize);
    for d in &digits {
        assert_eq!(*d, Int::ONE);
    }
    // Its representative modulo 2^8 is 2^8 - 1 = 255.
    assert_eq!(x.to_rational(), Rational::from_integer(p(255)));
}

#[test]
fn one_third_reciprocal_in_z2() {
    let n = 20;
    let third = Padic::from_rational(p(2), n, &rat(1, 3));
    let three = Padic::from_int(p(2), n, p(3));
    // 3 · (1/3) == 1.
    assert_eq!(&third * &three, Padic::one(p(2), n));
    // 1/3 is a 2-adic integer (denominator coprime to 2): valuation 0.
    assert_eq!(third.valuation(), Some(0));
    // Known low digits of 1/3 in ℤ₂: 1,1,0,1,0,1,0,1,… (…01011).
    let d = third.digits();
    let expect = [1, 1, 0, 1, 0, 1];
    for (i, e) in expect.iter().enumerate() {
        assert_eq!(d[i], p(*e), "digit {i}");
    }
}

#[test]
fn addition_matches_direct_construction() {
    let n = 12;
    let a = Padic::from_int(p(5), n, p(123456));
    let b = Padic::from_int(p(5), n, p(-987654));
    let sum_direct = Padic::from_int(p(5), n, p(123456 - 987654));
    assert_eq!(&a + &b, sum_direct);
}

#[test]
fn multiplication_matches_direct_construction() {
    let n = 12;
    let a = Padic::from_int(p(7), n, p(1234));
    let b = Padic::from_int(p(7), n, p(5678));
    let prod_direct = Padic::from_int(p(7), n, p(1234 * 5678));
    assert_eq!(&a * &b, prod_direct);
}

#[test]
fn self_minus_self_is_zero() {
    let n = 10;
    let a = Padic::from_rational(p(3), n, &rat(7, 4));
    let z = &a - &a;
    assert!(z.is_zero());
    assert_eq!(z, Padic::new(p(3), n));
}

#[test]
fn unit_times_reciprocal_is_one() {
    let n = 15;
    for val in [2i64, 5, 17, 100] {
        let a = Padic::from_int(p(11), n, p(val));
        let inv = a.inv();
        assert_eq!(&a * &inv, Padic::one(p(11), n));
        assert_eq!(&a / &a, Padic::one(p(11), n));
    }
}

#[test]
fn valuation_of_p_power_times_unit() {
    let n = 10;
    // v_p(p^k · unit) == k, for a unit coprime to p.
    for k in 0..5u32 {
        let value = p(7).pow(k).mul(&p(6)); // 6 is a unit mod 7
        let x = Padic::from_int(p(7), n, value);
        assert_eq!(x.valuation(), Some(k as i64));
    }
}

#[test]
fn negative_valuation_in_qp() {
    let n = 8;
    // 1/p has valuation -1.
    let x = Padic::from_rational(p(5), n, &rat(1, 5));
    assert_eq!(x.valuation(), Some(-1));
    // |1/p|_p = p.
    assert_eq!(x.abs_value(), Rational::from_integer(p(5)));
    // 1/p^3 has valuation -3.
    let y = Padic::from_rational(p(5), n, &rat(1, 125));
    assert_eq!(y.valuation(), Some(-3));
    // p · (1/p) == 1.
    let five = Padic::from_int(p(5), n, p(5));
    assert_eq!(&x * &five, Padic::one(p(5), n));
}

#[test]
fn abs_value_of_zero_and_units() {
    let n = 6;
    assert_eq!(Padic::new(p(3), n).abs_value(), Rational::ZERO);
    // |unit|_p == 1.
    let u = Padic::from_int(p(3), n, p(2));
    assert_eq!(u.abs_value(), Rational::ONE);
    // |p^2|_p == 1/p^2.
    let x = Padic::from_int(p(3), n, p(9));
    assert_eq!(x.abs_value(), Rational::new(Int::ONE, p(9)));
}

#[test]
fn sqrt_odd_prime() {
    let n = 12;
    // 2 is a quadratic residue mod 7 (3² ≡ 2), so √2 exists in ℤ₇.
    let two = Padic::from_int(p(7), n, p(2));
    let r = two.sqrt().expect("2 is a 7-adic square");
    assert_eq!(&r * &r, two);
    // A non-residue (3 mod 7) has no square root.
    let three = Padic::from_int(p(7), n, p(3));
    assert!(three.sqrt().is_none());
}

#[test]
fn sqrt_two_adic() {
    let n = 16;
    // 17 ≡ 1 (mod 8), so √17 exists in ℤ₂.
    let x = Padic::from_int(p(2), n, p(17));
    let r = x.sqrt().expect("17 is a 2-adic square");
    assert_eq!(&r * &r, x);
    // 3 ≢ 1 (mod 8): not a 2-adic square.
    assert!(Padic::from_int(p(2), n, p(3)).sqrt().is_none());
}

#[test]
fn sqrt_with_even_valuation() {
    let n = 10;
    // 7² · 2 has valuation 2 (even) and unit 2, a 7-adic square.
    let value = p(7).pow(2).mul(&p(2));
    let x = Padic::from_int(p(7), n, value);
    let r = x.sqrt().expect("even valuation, residue unit");
    assert_eq!(r.valuation(), Some(1));
    assert_eq!(&r * &r, x);
    // Odd valuation ⇒ no square root.
    let odd = Padic::from_int(p(7), n, p(7).mul(&p(2)));
    assert!(odd.sqrt().is_none());
}

#[test]
fn display_format() {
    let n = 5;
    let x = Padic::from_int(p(2), n, p(-1));
    assert_eq!(x.to_string(), "1 + 1*2 + 1*2^2 + 1*2^3 + 1*2^4 + O(2^5)");
    assert_eq!(Padic::new(p(2), n).to_string(), "O(2^5)");
}

#[test]
fn to_rational_roundtrip() {
    let n = 30;
    // A 5-adic unit reconstructs to a representative congruent mod 5^N.
    let x = Padic::from_int(p(5), n, p(123));
    assert_eq!(x.to_rational(), Rational::from_integer(p(123)));
}
