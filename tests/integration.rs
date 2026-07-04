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
    let half = Rational::new(int("2"), int("4"));
    assert_eq!(half.to_string(), "1/2");

    // 1/2 + 1/3 == 5/6
    let a = Rational::new(int("1"), int("2"));
    let b = Rational::new(int("1"), int("3"));
    assert_eq!(a.add(&b).to_string(), "5/6");

    // 2/3 * 3/4 == 1/2
    let c = Rational::new(int("2"), int("3"));
    let d = Rational::new(int("3"), int("4"));
    assert_eq!(c.mul(&d).to_string(), "1/2");

    // 6/3 is the integer 2
    let e = Rational::new(int("6"), int("3"));
    assert!(e.is_integer());
    assert_eq!(e.to_string(), "2");

    // ordering: 1/3 < 1/2
    assert!(a > b);
    assert!(Rational::new(int("0"), int("5")).is_zero());
    assert!(Rational::checked_new(int("1"), int("0")).is_none());
}

#[test]
fn rational_sign_is_canonical() {
    // Negative denominator moves the sign to the numerator.
    let r = Rational::new(int("1"), int("-2"));
    assert_eq!(r.to_string(), "-1/2");
    assert_eq!(r.numerator().to_string(), "-1");
    assert_eq!(r.denominator().to_string(), "2");
}

#[test]
fn rational_full_surface() {
    // Constructors and consts.
    assert_eq!(Rational::ZERO.to_string(), "0");
    assert_eq!(Rational::MINUS_ONE.to_string(), "-1");
    assert_eq!(Rational::power_of_two(-3).to_string(), "1/8");
    assert_eq!(Rational::power_of_two(4).to_string(), "16");
    assert_eq!(Rational::from(3i64).to_string(), "3");

    // FromStr: integer, fraction, decimal.
    assert_eq!("3".parse::<Rational>().unwrap().to_string(), "3");
    assert_eq!("-3/4".parse::<Rational>().unwrap().to_string(), "-3/4");
    assert_eq!("1.5".parse::<Rational>().unwrap().to_string(), "3/2");
    assert_eq!("-0.125".parse::<Rational>().unwrap().to_string(), "-1/8");

    // recip / abs / pow (incl. negative exponent).
    let r = Rational::new(int("2"), int("3"));
    assert_eq!(r.recip().to_string(), "3/2");
    assert_eq!(r.neg().abs().to_string(), "2/3");
    assert_eq!(r.pow(3).to_string(), "8/27");
    assert_eq!(r.pow(-2).to_string(), "9/4");

    // rounding to Int.
    let s = Rational::new(int("-7"), int("2")); // -3.5
    assert_eq!(s.floor().to_string(), "-4");
    assert_eq!(s.ceil().to_string(), "-3");
    assert_eq!(s.trunc().to_string(), "-3");
    assert!(Rational::new(int("6"), int("3")).to_integer().is_some());
    assert!(Rational::new(int("7"), int("3")).to_integer().is_none());

    // integer division of rationals.
    let a = Rational::new(int("7"), int("2")); // 3.5
    let b = Rational::new(int("1"), int("2")); // 0.5
    assert_eq!(a.div_floor(&b).to_string(), "7");
    assert_eq!(a.div_trunc(&b).to_string(), "7");

    // bounded conversions.
    assert_eq!(Rational::from(42i64).to_i64(), Some(42));
    assert_eq!(Rational::new(int("1"), int("2")).to_i64(), None);
    assert!((Rational::new(int("1"), int("4")).to_f64() - 0.25).abs() < 1e-12);

    // Canonical form after ops: gcd(num,den)==1, den>0.
    let x = Rational::new(int("6"), int("-8")); // -3/4
    assert_eq!(x.numerator().to_string(), "-3");
    assert_eq!(x.denominator().to_string(), "4");
}

#[test]
fn rational_write_decimal() {
    let mut out = String::new();
    // 1/3 to 5 digits, rounded then truncated.
    Rational::new(int("1"), int("3"))
        .write_decimal(&mut out, 5, false)
        .unwrap();
    assert_eq!(out, "0.33333");
    out.clear();
    // 2/3 rounds up the last digit.
    Rational::new(int("2"), int("3"))
        .write_decimal(&mut out, 4, false)
        .unwrap();
    assert_eq!(out, "0.6667");
    out.clear();
    Rational::new(int("2"), int("3"))
        .write_decimal(&mut out, 4, true)
        .unwrap();
    assert_eq!(out, "0.6666");
    out.clear();
    // Negative, and a carry that reaches the integer part (0.999… -> 1.00).
    Rational::new(int("-1"), int("8"))
        .write_decimal(&mut out, 3, false)
        .unwrap();
    assert_eq!(out, "-0.125");
    out.clear();
    Rational::new(int("999"), int("1000"))
        .write_decimal(&mut out, 2, false)
        .unwrap();
    assert_eq!(out, "1.00");
}

// ---- Int: small/large inline representation ----

#[test]
fn small_large_boundary_arithmetic() {
    // Values straddling the single-limb (u64) inline boundary.
    let u64_max = int("18446744073709551615"); // 2^64 - 1 (largest inline magnitude)
    let two64 = int("18446744073709551616"); // 2^64 (first Large value)
    assert_eq!(u64_max.add(&Int::ONE), two64);
    assert_eq!(two64.sub(&Int::ONE), u64_max);
    // Demotion: Large - Large that fits back inline.
    assert_eq!(two64.sub(&two64), Int::ZERO);
    assert!(two64.sub(&Int::ONE).fits_u64());

    // i64::MIN / i64::MAX inline edges, and -i64::MIN which overflows i64.
    let imin = Int::from_i64(i64::MIN);
    assert_eq!(imin.to_i64(), Some(i64::MIN));
    assert_eq!(imin.neg().to_string(), "9223372036854775808"); // 2^63, no longer fits i64
    assert!(!imin.neg().fits_i64());
    assert_eq!(Int::from_i64(i64::MAX).to_i64(), Some(i64::MAX));
}

#[test]
fn from_primitives_and_conversions() {
    assert_eq!(Int::from(-5i8).to_string(), "-5");
    assert_eq!(Int::from(u64::MAX).to_string(), "18446744073709551615");
    assert_eq!(
        Int::from(i128::MIN).to_string(),
        "-170141183460469231731687303715884105728"
    );
    assert_eq!(Int::from(255u8).to_u64(), Some(255));
    assert_eq!(int("-1").to_u64(), None);
    assert_eq!(int("99999999999999999999999").to_i64(), None);
    assert_eq!(Int::from(42i64).to_f64(), 42.0);
    assert_eq!(int("-1000000").to_f64(), -1_000_000.0);
    assert_eq!(int("18446744073709551616").to_f64(), 2f64.powi(64));
}

// ---- Int: three division conventions, cross-checked against i64 ----

#[test]
fn division_conventions_match_i64() {
    for a in -25i64..=25 {
        for b in -7i64..=7 {
            if b == 0 {
                continue;
            }
            let (ia, ib) = (Int::from_i64(a), Int::from_i64(b));

            // Truncated: matches Rust's `/` and `%`.
            let (qt, rt) = ia.div_rem_trunc(&ib);
            assert_eq!(qt.to_i64(), Some(a / b), "trunc q {a}/{b}");
            assert_eq!(rt.to_i64(), Some(a % b), "trunc r {a}/{b}");
            assert_eq!(qt.mul(&ib).add(&rt), ia, "trunc identity {a}/{b}");

            // Euclidean: matches div_euclid/rem_euclid; 0 <= r < |b|.
            let (qe, re) = ia.div_rem_euclid(&ib);
            assert_eq!(qe.to_i64(), Some(a.div_euclid(b)), "euclid q {a}/{b}");
            assert_eq!(re.to_i64(), Some(a.rem_euclid(b)), "euclid r {a}/{b}");
            assert!(!re.is_negative() && re < ib.abs(), "euclid range {a}/{b}");
            assert_eq!(qe.mul(&ib).add(&re), ia, "euclid identity {a}/{b}");

            // Floored: quotient toward -inf, remainder sign of divisor.
            let qf_oracle = {
                let mut q = a / b;
                if a % b != 0 && (a % b < 0) != (b < 0) {
                    q -= 1;
                }
                q
            };
            let (qfl, rfl) = ia.div_rem_floor(&ib);
            assert_eq!(qfl.to_i64(), Some(qf_oracle), "floor q {a}/{b}");
            assert_eq!(qfl.mul(&ib).add(&rfl), ia, "floor identity {a}/{b}");
            assert!(
                rfl.is_zero() || (rfl.is_negative() == (b < 0)),
                "floor r sign {a}/{b}"
            );
        }
    }
}

#[test]
fn division_big_operands() {
    let a = int("-123456789012345678901234567890");
    let b = int("987654321987654321");
    let (q, r) = a.div_rem_trunc(&b);
    assert_eq!(q.mul(&b).add(&r), a);
    assert!(r.abs() < b.abs());
    assert!(r.is_negative()); // trunc remainder follows dividend

    let (qe, re) = a.div_rem_euclid(&b);
    assert!(!re.is_negative() && re < b.abs());
    assert_eq!(qe.mul(&b).add(&re), a);

    assert!(int("1").div_rem(&Int::ZERO).is_none());
    assert!(int("100").div_exact(&int("4")) == int("25"));
    assert!(int("4").divides(&int("100")));
    assert!(!int("3").divides(&int("100")));
}

// ---- Int: number theory ----

#[test]
fn gcd_lcm_extended() {
    let a = int("461952");
    let b = int("116298");
    let g = a.gcd(&b);
    assert_eq!(g.to_string(), "18");
    // gcd*lcm == |a*b|
    assert_eq!(g.mul(&a.lcm(&b)), a.mul(&b).abs());
    // extended: g == a*x + b*y
    let (g2, x, y) = a.extended_gcd(&b);
    assert_eq!(g2, g);
    assert_eq!(a.mul(&x).add(&b.mul(&y)), g);
    // negatives
    let (g3, x3, y3) = int("-12").extended_gcd(&int("18"));
    assert_eq!(g3.to_string(), "6");
    assert_eq!(int("-12").mul(&x3).add(&int("18").mul(&y3)), g3);

    use puremp::{u_gcd, u64_gcd};
    assert_eq!(u64_gcd(1071, 462), 21);
    assert_eq!(u_gcd(48, 36), 12);
}

#[test]
fn roots() {
    assert_eq!(int("144").sqrt_exact().unwrap().to_string(), "12");
    assert!(int("145").sqrt_exact().is_none());
    assert!(int("-4").sqrt_exact().is_none());
    // (10^15)^2
    assert_eq!(
        int("1000000000000000000000000000000")
            .sqrt_exact()
            .unwrap()
            .to_string(),
        "1000000000000000"
    );
    assert_eq!(int("27").nth_root_exact(3).unwrap().to_string(), "3");
    assert_eq!(int("-27").nth_root_exact(3).unwrap().to_string(), "-3");
    assert!(int("-16").nth_root_exact(4).is_none());
    assert!(int("28").nth_root_exact(3).is_none());
}

// ---- Int: power-of-two & bit access ----

#[test]
fn power_of_two_ops() {
    let x = int("12345");
    assert_eq!(x.mul_2k(10), x.mul(&Int::from_i64(1024)));
    assert_eq!(x.div_2k_trunc(3).to_string(), "1543"); // 12345 >> 3
    // mod_2k == rem_euclid(2^k), always non-negative
    for k in 0u32..12 {
        let m = Int::from_i64(1i64 << k);
        assert_eq!(x.mod_2k(k), x.rem_euclid(&m), "mod_2k {k}");
        assert_eq!(
            int("-12345").mod_2k(k),
            int("-12345").rem_euclid(&m),
            "neg mod_2k {k}"
        );
    }
    assert_eq!(int("1024").is_power_of_two(), Some(10));
    assert_eq!(int("-1024").is_power_of_two(), Some(10));
    assert_eq!(int("1000").is_power_of_two(), None);
    assert_eq!(int("48").trailing_zeros(), 4); // 48 = 16*3
    assert_eq!(int("0").trailing_zeros(), 0);
    assert_eq!(int("255").bit_len(), 8);
    assert_eq!(int("256").log2_floor(), 8);
}

#[test]
fn twos_complement_bitwise() {
    // Match Rust's i64 two's-complement operators, including negatives.
    for a in [-9i64, -1, 0, 5, 12, 255, -256, 1023] {
        for b in [-7i64, -1, 0, 3, 8, 100, -100] {
            let (ia, ib) = (Int::from_i64(a), Int::from_i64(b));
            assert_eq!(ia.bitand(&ib).to_i64(), Some(a & b), "{a} & {b}");
            assert_eq!(ia.bitor(&ib).to_i64(), Some(a | b), "{a} | {b}");
            assert_eq!(ia.bitxor(&ib).to_i64(), Some(a ^ b), "{a} ^ {b}");
        }
        // bitnot within an 8-bit window: sign-extend (!a & 0xFF).
        let u = (!a as u64) & 0xFF;
        let oracle = if u & 0x80 != 0 {
            u as i64 - 256
        } else {
            u as i64
        };
        assert_eq!(
            Int::from_i64(a).bitnot(8).to_i64(),
            Some(oracle),
            "bitnot8 {a}"
        );
    }
}

// ---- Int: limb access, hashing, radix ----

#[test]
fn limb_roundtrip_and_access() {
    for s in [
        "0",
        "1",
        "-1",
        "18446744073709551616",
        "-340282366920938463463374607431768211457",
    ] {
        let x = int(s);
        let rebuilt = Int::from_limbs(x.sign(), x.limbs());
        assert_eq!(rebuilt, x, "limb roundtrip {s}");
    }
    let big = int("340282366920938463463374607431768211456"); // 2^128
    assert_eq!(big.limbs(), &[0, 0, 1]);
    assert_eq!(big.least_significant_limb(), 0);
    assert_eq!(int("18446744073709551617").least_significant_limb(), 1);
    assert!(int("1024").bit(10));
    assert!(!int("1024").bit(9));
}

#[test]
fn hash_is_consistent_with_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(int("340282366920938463463374607431768211456")); // 2^128 built via parse
    // Same value built a different way must be found (equal => equal hash).
    let via_mul = int("18446744073709551616").mul(&int("18446744073709551616"));
    assert!(set.contains(&via_mul));
    assert!(set.contains(&Int::from(2i64).pow(128)));
    assert!(!set.contains(&int("7")));
}

#[test]
fn radix_roundtrip() {
    assert_eq!(Int::from_str_radix("ff", 16).unwrap().to_string(), "255");
    assert_eq!(Int::from_str_radix("-101", 2).unwrap().to_string(), "-5");
    for s in ["0", "255", "-4096", "123456789012345678901234567890"] {
        let x = int(s);
        for radix in [2u32, 8, 16, 36] {
            let mut buf = String::new();
            x.write_radix(&mut buf, radix).unwrap();
            assert_eq!(
                Int::from_str_radix(&buf, radix).unwrap(),
                x,
                "radix {radix} for {s}"
            );
        }
    }
}

#[test]
fn karatsuba_agrees_and_is_correct() {
    // Large multiplication must cross the Karatsuba threshold and match a
    // value computed a different way: (10^k)^2 == 10^(2k), and factorials.
    let p = Int::from_i64(10).pow(500); // ~1662 bits, well past the threshold
    let sq = p.mul(&p);
    let mut expected = String::from("1");
    expected.push_str(&"0".repeat(1000));
    assert_eq!(sq.to_string(), expected);

    // Associativity/commutativity on large operands (exercises the recursion).
    let a = Int::from_i64(7).pow(400);
    let b = Int::from_i64(3).pow(410);
    let c = Int::from_i64(11).pow(390);
    assert_eq!(a.mul(&b).mul(&c), c.mul(&a).mul(&b));
    assert_eq!(a.mul(&b), b.mul(&a));

    // Distributivity: a*(b+c) == a*b + a*c on large operands.
    assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));

    // 200! computed by product still matches the known trailing-zero count (49).
    let fact200 = (2..=200u64).fold(Int::one(), |acc, k| acc.mul(&Int::from_i64(k as i64)));
    let s = fact200.to_string();
    assert_eq!(s.len() - s.trim_end_matches('0').len(), 49);
}

#[test]
fn fused_addmul_submul() {
    let mut acc = int("1000");
    acc.addmul(&int("3"), &int("7")); // 1000 + 21
    assert_eq!(acc.to_string(), "1021");
    acc.submul(&int("2"), &int("11")); // 1021 - 22
    assert_eq!(acc.to_string(), "999");
    // Large operands
    let mut big = Int::ZERO;
    big.addmul(&int("18446744073709551616"), &int("18446744073709551616"));
    assert_eq!(big.to_string(), "340282366920938463463374607431768211456");
}

#[test]
fn sum_and_product_iterators() {
    let xs = [int("10"), int("20"), int("30")];
    let s: Int = xs.iter().sum();
    assert_eq!(s.to_string(), "60");
    let p: Int = (1..=10i64).map(Int::from_i64).product();
    assert_eq!(p.to_string(), "3628800"); // 10!
}

#[test]
fn random_generation() {
    use puremp::RandomSource;
    // A tiny deterministic xorshift RNG implementing the in-house trait.
    struct Xorshift(u64);
    impl RandomSource for Xorshift {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for b in dest.iter_mut() {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                *b = x as u8;
            }
        }
    }
    let mut rng = Xorshift(0x9e3779b97f4a7c15);

    // random_bits stays within range.
    for _ in 0..200 {
        let n = Nat::random_bits(100, &mut rng);
        assert!(n.bit_len() <= 100);
    }
    // random_below stays below the bound and (statistically) exercises it.
    let bound = int("1000000000000000000000");
    let mut max_seen = Int::ZERO;
    for _ in 0..500 {
        let r = Int::random_below(&bound, &mut rng).unwrap();
        assert!(r >= Int::ZERO && r < bound);
        if r > max_seen {
            max_seen = r;
        }
    }
    assert!(
        max_seen > bound.div_trunc(&int("2")),
        "distribution looks skewed"
    );
    assert!(Int::random_below(&Int::ZERO, &mut rng).is_none());

    // byte round-trip.
    let x = nat("123456789012345678901234567890");
    assert_eq!(Nat::from_bytes_le(&x.to_bytes_le()), x);
}

#[test]
fn toom4_matches_reference() {
    // Operands large enough to cross the Toom-4 threshold (>320 limbs, ~20k
    // bits): 10^6000 * 10^6100 == 10^12100, plus algebraic laws.
    let p = Int::from_i64(10).pow(6000);
    let q = Int::from_i64(10).pow(6100);
    let prod = p.mul(&q);
    let mut expected = String::from("1");
    expected.push_str(&"0".repeat(12100));
    assert_eq!(prod.to_string(), expected);

    let a = Int::from_i64(7).pow(7000);
    let b = Int::from_i64(3).pow(7100);
    let c = Int::from_i64(11).pow(6900);
    assert_eq!(a.mul(&b), b.mul(&a));
    assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
}

#[test]
fn toom3_matches_reference() {
    // Operands large enough to cross the Toom-3 threshold (>128 limbs, ~8200
    // bits). Check against a value computed a different way: (10^m)·(10^m') is
    // 10^(m+m'), plus algebraic laws.
    let p = Int::from_i64(10).pow(3000); // ~9966 bits
    let q = Int::from_i64(10).pow(3100);
    let prod = p.mul(&q);
    let mut expected = String::from("1");
    expected.push_str(&"0".repeat(6100));
    assert_eq!(prod.to_string(), expected);

    // Commutativity and distributivity on Toom-3-sized operands.
    let a = Int::from_i64(7).pow(3500);
    let b = Int::from_i64(3).pow(3600);
    let c = Int::from_i64(11).pow(3400);
    assert_eq!(a.mul(&b), b.mul(&a));
    assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
    // (a+b)^2 == a^2 + 2ab + b^2 cross-checks Toom-3 vs the squaring path.
    let ab = a.add(&b);
    assert_eq!(
        ab.square(),
        a.square().add(&a.mul(&b).mul_2k(1)).add(&b.square())
    );
}

#[test]
fn square_matches_mul() {
    // Squaring must agree with the general multiply across sizes (schoolbook and
    // Karatsuba squaring paths), including a value that crosses the threshold.
    for e in [1u32, 5, 50, 400, 900] {
        let x = Int::from_i64(7).pow(e).add(&Int::from_i64(123456789));
        assert_eq!(x.square(), x.mul(&x), "7^{e}+c squared");
        let nx = x.neg();
        assert_eq!(nx.square(), nx.mul(&nx), "negative squared is positive");
    }
    assert_eq!(Int::ZERO.square(), Int::ZERO);
    // Nat-level too.
    let n = nat("123456789012345678901234567890").pow(20);
    assert_eq!(n.square(), n.mul(&n));
}

#[test]
fn modular_arithmetic() {
    // modpow
    assert_eq!(int("2").modpow(&int("10"), &int("1000")).to_string(), "24"); // 1024 mod 1000
    assert_eq!(int("3").modpow(&int("0"), &int("7")).to_string(), "1");
    // Fermat: a^(p-1) ≡ 1 (mod p) for prime p, gcd(a,p)=1.
    let p = int("1000000007");
    assert_eq!(int("123456").modpow(&p.sub(&int("1")), &p).to_string(), "1");
    // Big modular exponentiation.
    let big = int("2").modpow(&int("1000000"), &int("999999999999999999999"));
    assert!(big >= Int::ZERO && big < int("999999999999999999999"));

    // modular inverse
    assert_eq!(int("3").modinv(&int("11")).unwrap().to_string(), "4"); // 3*4=12≡1
    assert!(int("6").modinv(&int("9")).is_none()); // gcd(6,9)=3
    let inv = int("123456789").modinv(&p).unwrap();
    assert_eq!(int("123456789").mul(&inv).rem_euclid(&p).to_string(), "1");

    // negative base reduces correctly
    assert_eq!(int("-1").modpow(&int("3"), &int("5")).to_string(), "4"); // (-1)^3 = -1 ≡ 4
}

#[test]
fn primality_testing() {
    use puremp::RandomSource;
    struct Lcg(u64);
    impl RandomSource for Lcg {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for b in dest.iter_mut() {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *b = (self.0 >> 33) as u8;
            }
        }
    }
    let mut rng = Lcg(0xdeadbeef);
    let is_prime = |s: &str, rng: &mut Lcg| nat(s).is_probable_prime(40, rng);

    for p in [
        "2",
        "3",
        "5",
        "97",
        "1000000007",
        "170141183460469231731687303715884105727",
    ] {
        assert!(is_prime(p, &mut rng), "{p} should be prime");
    }
    for c in [
        "1",
        "4",
        "100",
        "1000000009000000000",
        "170141183460469231731687303715884105721",
    ] {
        assert!(!is_prime(c, &mut rng), "{c} should be composite");
    }
    // A Carmichael number (561 = 3·11·17) must be caught by Miller–Rabin.
    assert!(!nat("561").is_probable_prime(40, &mut rng));
}

#[test]
fn next_prime_works() {
    use puremp::RandomSource;
    struct Lcg(u64);
    impl RandomSource for Lcg {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for b in dest.iter_mut() {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (self.0 >> 33) as u8;
            }
        }
    }
    let mut rng = Lcg(12345);
    assert_eq!(int("0").next_prime(&mut rng).to_string(), "2");
    assert_eq!(int("2").next_prime(&mut rng).to_string(), "3");
    assert_eq!(int("7").next_prime(&mut rng).to_string(), "11");
    assert_eq!(int("100").next_prime(&mut rng).to_string(), "101");
    // Next prime after a large power of ten (10^30 + 57 is the known answer).
    let p = int("1000000000000000000000000000000").next_prime(&mut rng);
    assert_eq!(p.to_string(), "1000000000000000000000000000057");
    assert!(p.magnitude().is_probable_prime(40, &mut rng));

    // prev_prime
    assert_eq!(int("11").prev_prime(&mut rng).unwrap().to_string(), "7");
    assert_eq!(int("3").prev_prime(&mut rng).unwrap().to_string(), "2");
    assert!(int("2").prev_prime(&mut rng).is_none());
    assert_eq!(int("100").prev_prime(&mut rng).unwrap().to_string(), "97");
}

#[test]
fn reciprocal_reduce() {
    use puremp::Reciprocal;
    // reduce(x) == x mod m for x < m², across sizes.
    for m_s in [
        "1000000007",
        "18446744073709551629",
        "340282366920938463463374607431768211507",
    ] {
        let m = nat(m_s);
        let r = Reciprocal::new(&m);
        assert_eq!(r.modulus(), &m);
        for a_s in [
            "0",
            "1",
            "999999999999999999999",
            "123456789012345678901234567890",
        ] {
            let a = nat(a_s);
            if a >= m {
                continue;
            }
            // (a * b) mod m with a, b < m so the product < m².
            let b = m.checked_sub(&Nat::one()).unwrap();
            let prod = a.mul(&b);
            assert_eq!(
                r.reduce(&prod),
                prod.div_rem(&m).unwrap().1,
                "{a_s} mod {m_s}"
            );
        }
        // m² - 1 is the largest valid input.
        let big = m.square().checked_sub(&Nat::one()).unwrap();
        assert_eq!(r.reduce(&big), big.div_rem(&m).unwrap().1);
    }
}

#[test]
fn combinatorics() {
    assert_eq!(Int::factorial(0).to_string(), "1");
    assert_eq!(Int::factorial(10).to_string(), "3628800");
    assert_eq!(Int::factorial(25).to_string(), "15511210043330985984000000");

    assert_eq!(Int::binomial(10, 3).to_string(), "120");
    assert_eq!(Int::binomial(52, 5).to_string(), "2598960");
    assert_eq!(Int::binomial(5, 8).to_string(), "0");
    assert_eq!(
        Int::binomial(100, 50).to_string(),
        "100891344545564193334812497256"
    );
    // symmetry
    assert_eq!(Int::binomial(30, 7), Int::binomial(30, 23));

    // multinomial(1,2,3) = 6!/(1!2!3!) = 60
    assert_eq!(Int::multinomial(&[1, 2, 3]).to_string(), "60");
    assert_eq!(Int::multinomial(&[2, 2, 2]).to_string(), "90"); // 6!/(2!2!2!)

    // Fibonacci / Lucas
    assert_eq!(Int::fibonacci(0).to_string(), "0");
    assert_eq!(Int::fibonacci(10).to_string(), "55");
    assert_eq!(Int::fibonacci(100).to_string(), "354224848179261915075");
    assert_eq!(Int::lucas(0).to_string(), "2");
    assert_eq!(Int::lucas(10).to_string(), "123");
    // Identity: L(n) = F(n-1) + F(n+1)
    for n in 1..40u64 {
        assert_eq!(
            Int::lucas(n),
            Int::fibonacci(n - 1).add(&Int::fibonacci(n + 1))
        );
    }
}

#[test]
fn jacobi_sqrt_mod_crt() {
    // Jacobi / Legendre
    assert_eq!(int("2").jacobi(&int("15")), 1); // (2/15)
    assert_eq!(int("5").jacobi(&int("21")), 1);
    assert_eq!(int("3").legendre(&int("7")), -1); // 3 is a non-residue mod 7
    assert_eq!(int("2").legendre(&int("7")), 1); // 2 ≡ 3² = 9 ≡ 2 (mod 7)
    assert_eq!(int("7").jacobi(&int("7")), 0);

    // Modular square root r² ≡ a (mod p), with s=1 (7, 13, 1000000007) and
    // s>1 (17: s=4, 41: s=3) valuations of p-1.
    for (a, p) in [
        ("2", "7"),
        ("10", "13"),
        ("2", "17"),
        ("5", "41"),
        ("123456", "1000000007"),
    ] {
        let (a, p) = (int(a), int(p));
        match a.sqrt_mod(&p) {
            Some(r) => assert_eq!(
                r.mul(&r).rem_euclid(&p),
                a.rem_euclid(&p),
                "sqrt {a} mod {p}"
            ),
            None => assert_eq!(a.legendre(&p), -1),
        }
    }
    assert!(int("3").sqrt_mod(&int("7")).is_none()); // non-residue

    // Large modular square root mod the Mersenne prime 2^127 - 1, for a value
    // that is a quadratic residue by construction (x²).
    let p = int("170141183460469231731687303715884105727");
    let x = int("123456789012345678901234567890");
    let a = x.mul(&x).rem_euclid(&p);
    let r = a.sqrt_mod(&p).unwrap();
    assert_eq!(r.mul(&r).rem_euclid(&p), a);

    // CRT: x ≡ 2 (mod 3), 3 (mod 5), 2 (mod 7) → 23
    let x = Int::crt(
        &[int("2"), int("3"), int("2")],
        &[int("3"), int("5"), int("7")],
    )
    .unwrap();
    assert_eq!(x.to_string(), "23");
    // Non-coprime moduli → None.
    assert!(Int::crt(&[int("1"), int("2")], &[int("4"), int("6")]).is_none());
}

#[test]
fn factorization_and_random_prime() {
    use puremp::RandomSource;
    // Helper to render a factorization compactly.
    let facs = |s: &str| -> String {
        int(s)
            .factorize()
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join("*")
    };
    assert_eq!(facs("1"), "");
    assert_eq!(facs("2"), "2");
    assert_eq!(facs("360"), "2*2*2*3*3*5");
    assert_eq!(facs("1000000007"), "1000000007"); // prime
    assert_eq!(facs("600851475143"), "71*839*1471*6857"); // Project Euler #3
    // A product of two ~10-digit primes.
    let semiprime = int("32416190071").mul(&int("32416187567"));
    let f = semiprime.factorize();
    assert_eq!(f.len(), 2);
    assert_eq!(f[0].mul(&f[1]), semiprime);
    // Product of all factors reconstructs the input.
    let n = int("123456789012345678");
    let prod = n.factorize().iter().fold(Int::ONE, |a, p| a.mul(p));
    assert_eq!(prod, n);

    // random_prime
    struct Lcg(u64);
    impl RandomSource for Lcg {
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for b in dest.iter_mut() {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (self.0 >> 33) as u8;
            }
        }
    }
    let mut rng = Lcg(999);
    for bits in [16u32, 32, 64, 128] {
        let p = Int::random_prime(bits, &mut rng);
        assert!(p.is_prime_bpsw(), "{p} not prime");
        assert_eq!(p.bit_len(), bits, "wrong bit length for {p}");
    }
}

#[test]
fn continued_fractions() {
    let r = |s: &str| -> Rational { s.parse().unwrap() };
    let cf = |terms: &[i64]| -> Vec<Int> { terms.iter().map(|&t| Int::from(t)).collect() };

    // 415/93 = [4; 2, 6, 7]
    assert_eq!(r("415/93").continued_fraction(), cf(&[4, 2, 6, 7]));
    // reconstruct
    assert_eq!(
        Rational::from_continued_fraction(&cf(&[4, 2, 6, 7])).to_string(),
        "415/93"
    );
    // negative: -7/2 = [-4; 2]
    assert_eq!(r("-7/2").continued_fraction(), cf(&[-4, 2]));
    assert_eq!(
        Rational::from_continued_fraction(&cf(&[-4, 2])).to_string(),
        "-7/2"
    );

    // Best rational approximation of a good rational π (355/113 has den 113).
    let pi = r("314159265358979/100000000000000");
    assert_eq!(pi.approximate(&Int::from(10)).to_string(), "22/7");
    assert_eq!(pi.approximate(&Int::from(113)).to_string(), "355/113");
    assert_eq!(pi.approximate(&Int::from(150)).to_string(), "355/113");
    // A value that already fits is returned unchanged.
    assert_eq!(r("3/4").approximate(&Int::from(10)).to_string(), "3/4");
    // Every approximation respects the denominator bound.
    for bound in [1i64, 2, 5, 50, 1000] {
        let a = pi.approximate(&Int::from(bound));
        assert!(a.denominator() <= &Int::from(bound));
    }
}

#[test]
fn tryfrom_primitive_conversions() {
    use core::convert::TryFrom;
    assert_eq!(i32::try_from(&int("-42")).unwrap(), -42);
    assert_eq!(u8::try_from(&int("255")).unwrap(), 255u8);
    assert!(u8::try_from(&int("256")).is_err());
    assert!(u32::try_from(&int("-1")).is_err()); // negative into unsigned
    assert_eq!(
        i128::try_from(int("170141183460469231731687303715884105727")).unwrap(),
        i128::MAX
    );
    assert!(i128::try_from(&int("170141183460469231731687303715884105728")).is_err()); // i128::MAX + 1
    assert_eq!(
        u128::try_from(&int("340282366920938463463374607431768211455")).unwrap(),
        u128::MAX
    );
    assert!(i64::try_from(&Int::from(2).pow(200)).is_err());
    // owned form too
    assert_eq!(
        u64::try_from(int("18446744073709551615")).unwrap(),
        u64::MAX
    );
}

#[test]
fn large_parse_roundtrip_multi_radix() {
    // A large, irregular value (not all one digit).
    let n = int("7")
        .pow(4000)
        .mul(&int("11").pow(2000))
        .add(&int("123456789"));
    for radix in [2u32, 8, 10, 16, 36] {
        let mut s = String::new();
        n.write_radix(&mut s, radix).unwrap();
        let back = Int::from_str_radix(&s, radix).unwrap();
        assert_eq!(back, n, "radix {radix} round-trip");
    }
    // Leading-zero and boundary strings.
    assert_eq!(nat("000123").to_string(), "123");
    assert_eq!(nat(&"9".repeat(500)).to_string(), "9".repeat(500));
    // Exactly one base-10^19 chunk boundary (19 and 20 digit values).
    assert_eq!(
        nat("9999999999999999999").to_string(),
        "9999999999999999999"
    );
    assert_eq!(
        nat("10000000000000000000").to_string(),
        "10000000000000000000"
    );
}

#[test]
fn isqrt_exhaustive_and_large() {
    // Small values 0..2000: floor-sqrt property.
    for v in 0u64..2000 {
        let s = nat(&v.to_string()).isqrt().to_u64().unwrap();
        assert!(s * s <= v && (s + 1) * (s + 1) > v, "isqrt({v}) = {s}");
    }
    // Perfect squares and their neighbours across the 128-bit base boundary and
    // into the recursive range.
    for k_str in [
        "1",
        "65535",
        "4294967296",
        "18446744073709551616",
        "340282366920938463463374607431768211457", // > 2^128
        "99999999999999999999999999999999999999999999999999",
    ] {
        let k = int(k_str).magnitude();
        let sq = k.mul(&k);
        assert_eq!(sq.isqrt(), k, "isqrt(k²) for k={k_str}");
        // (k² - 1) has floor-sqrt k-1
        let below = sq.checked_sub(&Nat::one()).unwrap();
        assert_eq!(
            below.isqrt(),
            k.checked_sub(&Nat::one()).unwrap(),
            "isqrt(k²-1)"
        );
        // (k² + 2k) < (k+1)², floor-sqrt k
        let above = sq.add(&k).add(&k);
        assert_eq!(above.isqrt(), k, "isqrt(k²+2k)");
    }
    // Very large irregular value: verify the invariant s² ≤ n < (s+1)².
    let n = int("7")
        .pow(3000)
        .mul(&int("11").pow(1500))
        .add(&int("123456789"))
        .magnitude();
    let s = n.isqrt();
    assert!(s.mul(&s) <= n);
    assert!(s.add(&Nat::one()).square() > n);
}

#[test]
fn division_large_divisors_padded_bz() {
    // Exercise Burnikel–Ziegler above the threshold across many divisor sizes,
    // including odd limb counts (which drive the power-of-two block padding).
    // Verify q·b + r == a and r < b against an independent construction.
    for bits in [17000u32, 20000, 24000, 30000, 40000, 64000] {
        let b = int("7").pow(bits / 3).add(&int("1")); // irregular, ~bits/3·2.8 bits
        let q_ref = int("11").pow(bits / 4).add(&int("999999"));
        let r_ref = int("123456789012345678901234567890"); // < b for these sizes
        let a = q_ref.mul(&b).add(&r_ref);
        let (q, r) = a.div_rem(&b).unwrap();
        assert_eq!(q, q_ref, "quotient at {bits} bits");
        assert_eq!(r, r_ref, "remainder at {bits} bits");
        assert!(r < b);
        // and the fundamental identity
        assert_eq!(q.mul(&b).add(&r), a);
    }
}

#[test]
fn modpow_windowed_vs_reference() {
    // Independent plain right-to-left square-and-multiply reference.
    fn ref_modpow(mut base: Int, exp: &Int, m: &Int) -> Int {
        let mut result = Int::ONE;
        base = base.rem_euclid(m);
        let bits = exp.magnitude().bit_len();
        for i in 0..bits {
            if exp.magnitude().bit(i) {
                result = result.mul(&base).rem_euclid(m);
            }
            base = base.mul(&base).rem_euclid(m);
        }
        result
    }
    // Cover odd and even moduli (Montgomery vs Barrett paths), single- and
    // multi-limb, and exponent sizes straddling window widths (2..6).
    let cases = [
        ("2", "10", "1000"),
        ("7", "255", "13"),                       // exp spans a window boundary
        ("123456789", "987654321", "1000000007"), // odd modulus (Montgomery)
        ("123456789", "987654321", "1000000006"), // even modulus (Barrett)
        ("3", "65537", "340282366920938463463374607431768211297"), // large odd
        ("5", "0", "97"),                         // exp 0 -> 1
        ("5", "1", "97"),                         // exp 1
    ];
    for (b, e, m) in cases {
        let (b, e, m) = (int(b), int(e), int(m));
        assert_eq!(
            b.modpow(&e, &m),
            ref_modpow(b.clone(), &e, &m),
            "modpow {b}^{e} mod {m}"
        );
    }
    // Exponents of every bit length 1..80 (window-boundary stress).
    let (b, m) = (int("6"), int("1000000007"));
    for k in 1..80u32 {
        let e = int("2").pow(k).sub(&int("1")); // k ones
        assert_eq!(
            b.modpow(&e, &m),
            ref_modpow(b.clone(), &e, &m),
            "2^{k}-1 exponent"
        );
    }
}

#[test]
fn square_matches_mul_across_ladder() {
    // square() must equal mul-by-self at every size that crosses a ladder tier
    // (schoolbook / Karatsuba / Toom-3 / Toom-4).
    for limbs in [1usize, 40, 160, 260, 460, 900, 2000] {
        let x = int("7").pow((limbs * 64 / 3) as u32).magnitude();
        let y = x.add(&Nat::zero()); // distinct object, equal value
        // mul(&y) still routes to square via the limbs== check, so cross-check
        // against a genuinely different multiply: x*(x-1)+x == x*x.
        let xm1 = x.checked_sub(&Nat::one()).unwrap();
        assert_eq!(
            x.square(),
            x.mul(&xm1).add(&x),
            "square identity at ~{limbs} limbs"
        );
        let _ = y;
    }
}

#[test]
fn modpow_random_differential() {
    // Independent reference (right-to-left binary), checked against modpow over
    // many random bases/exponents/moduli of varied sizes and parities — this
    // stresses the CIOS Montgomery path (odd moduli) and the Barrett path (even).
    fn ref_modpow(mut base: Int, exp: &Int, m: &Int) -> Int {
        let mut result = Int::ONE;
        base = base.rem_euclid(m);
        let bits = exp.magnitude().bit_len();
        for i in 0..bits {
            if exp.magnitude().bit(i) {
                result = result.mul(&base).rem_euclid(m);
            }
            base = base.mul(&base).rem_euclid(m);
        }
        result
    }
    let mut s: u64 = 0xF00D_CAFE;
    let mut rng_int = |bits: u32, s: &mut u64| -> Int {
        let mut v = Int::ZERO;
        let limbs = (bits / 64 + 1).max(1);
        for _ in 0..limbs {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            v = v.mul(&Int::from_u64(u64::MAX)).add(&Int::from_u64(*s));
        }
        v.abs()
    };
    for _ in 0..400 {
        let bits = 32 + (s % 800) as u32;
        let mut m = rng_int(bits, &mut s).add(&Int::from(2));
        // Test both odd (Montgomery) and even (Barrett) moduli.
        if s & 1 == 0 && m.is_odd() {
            m = m.add(&Int::ONE);
        }
        let base = rng_int(bits, &mut s);
        let exp = rng_int(1 + (s % 400) as u32, &mut s);
        assert_eq!(
            base.modpow(&exp, &m),
            ref_modpow(base.clone(), &exp, &m),
            "modpow mismatch (odd_mod={})",
            m.is_odd()
        );
    }
}
