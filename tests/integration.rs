//! End-to-end tests for the public `puremp` API.
//!
//! These check behaviour against values computed independently (by hand or with
//! another arbitrary-precision tool). No arithmetic oracle crate is used — the
//! crate ships no foreign code, and the same discipline extends to its tests.

use puremp::{Int, Nat, Rational, Sign};

fn nat(s: &str) -> Nat {
    s.parse().expect("valid natural literal")
}

fn int(s: &str) -> Int {
    s.parse().expect("valid integer literal")
}

#[test]
fn nat_parse_display_roundtrip() {
    for s in [
        "0",
        "1",
        "9",
        "10",
        "255",
        "18446744073709551616",
        &"9".repeat(200),
    ] {
        assert_eq!(nat(s).to_string(), s, "roundtrip {s}");
    }
}

#[test]
fn nat_add_and_mul_across_limb_boundary() {
    // 2^64 - 1 + 1 == 2^64
    let max = nat("18446744073709551615");
    assert_eq!(max.add(&Nat::one()).to_string(), "18446744073709551616");
    // 2^64 * 2^64 == 2^128
    let two64 = nat("18446744073709551616");
    assert_eq!(
        two64.mul(&two64).to_string(),
        "340282366920938463463374607431768211456"
    );
}

#[test]
fn factorial_20_and_50() {
    fn fact(n: u64) -> Int {
        (2..=n).fold(Int::one(), |a, k| a.mul(&Int::from_i64(k as i64)))
    }
    assert_eq!(fact(20).to_string(), "2432902008176640000");
    assert_eq!(
        fact(50).to_string(),
        "30414093201713378043612608166064768844377641568960512000000000000"
    );
}

#[test]
fn power_of_two() {
    assert_eq!(
        Int::from_i64(2).pow(100).to_string(),
        "1267650600228229401496703205376"
    );
    assert_eq!(Int::from_i64(2).pow(0).to_string(), "1");
    assert_eq!(Int::from_i64(0).pow(0).to_string(), "1");
}

#[test]
fn div_rem_invariant() {
    let cases = [
        ("1000000000000000000000", "7"),
        ("123456789012345678901234567890", "987654321"),
        ("5", "5"),
        ("4", "5"),
        ("0", "5"),
    ];
    for (a_s, b_s) in cases {
        let a = nat(a_s);
        let b = nat(b_s);
        let (q, r) = a.div_rem(&b).expect("non-zero divisor");
        // a == q*b + r  and  r < b
        assert_eq!(q.mul(&b).add(&r), a, "reconstruct {a_s}/{b_s}");
        assert!(r < b, "remainder < divisor for {a_s}/{b_s}");
    }
    assert!(nat("1").div_rem(&Nat::zero()).is_none());
}

#[test]
fn gcd_matches_known_values() {
    assert_eq!(nat("1071").gcd(&nat("462")).to_string(), "21");
    assert_eq!(nat("0").gcd(&nat("5")).to_string(), "5");
    // Two large Fibonacci numbers are coprime.
    assert_eq!(nat("6765").gcd(&nat("10946")).to_string(), "1");
}

#[test]
fn shifts() {
    let one = Nat::one();
    assert_eq!(
        one.shl(128).to_string(),
        "340282366920938463463374607431768211456"
    );
    assert_eq!(one.shl(128).shr(64).to_string(), "18446744073709551616");
    assert_eq!(nat("12345").shr(1000).to_string(), "0");
}

#[test]
fn signed_arithmetic() {
    assert_eq!(int("-5").add(&int("3")).to_string(), "-2");
    assert_eq!(int("3").sub(&int("5")).to_string(), "-2");
    assert_eq!(int("-4").mul(&int("-6")).to_string(), "24");
    assert_eq!(int("-7").neg().to_string(), "7");
    assert_eq!(int("0").neg().sign(), Sign::Zero);
    assert!(int("-100") < int("-99"));
    assert!(int("-1") < int("0"));
    assert!(int("0") < int("1"));
}

#[test]
fn int_truncated_div_rem() {
    // -13 = (-3)*4 + (-1): quotient truncates toward zero, remainder follows dividend.
    let (q, r) = int("-13").div_rem(&int("4")).unwrap();
    assert_eq!(q.to_string(), "-3");
    assert_eq!(r.to_string(), "-1");
}

#[test]
fn rational_reduces_and_computes() {
    let half = Rational::new(int("2"), int("4")).unwrap();
    assert_eq!(half.to_string(), "1/2");

    // 1/2 + 1/3 == 5/6
    let a = Rational::new(int("1"), int("2")).unwrap();
    let b = Rational::new(int("1"), int("3")).unwrap();
    assert_eq!(a.add(&b).to_string(), "5/6");

    // 2/3 * 3/4 == 1/2
    let c = Rational::new(int("2"), int("3")).unwrap();
    let d = Rational::new(int("3"), int("4")).unwrap();
    assert_eq!(c.mul(&d).to_string(), "1/2");

    // 6/3 is the integer 2
    let e = Rational::new(int("6"), int("3")).unwrap();
    assert!(e.is_integer());
    assert_eq!(e.to_string(), "2");

    // ordering: 1/3 < 1/2
    assert!(a > b);
    assert!(Rational::new(int("0"), int("5")).unwrap().is_zero());
    assert!(Rational::new(int("1"), int("0")).is_err());
}

#[test]
fn rational_sign_is_canonical() {
    // Negative denominator moves the sign to the numerator.
    let r = Rational::new(int("1"), int("-2")).unwrap();
    assert_eq!(r.to_string(), "-1/2");
    assert_eq!(r.numerator().to_string(), "-1");
    assert_eq!(r.denominator().to_string(), "2");
}
