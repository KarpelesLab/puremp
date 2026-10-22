//! GMP-parity `Int` operations: float conversion, Kronecker symbols,
//! two's-complement bit access, rounding shifts, radixes up to 62, roots and
//! perfect powers, byte import/export. Expected values were cross-checked
//! against GMP 6 (`mpz_*`) as a black box.

use puremp::{Int, Nat};

fn int(s: &str) -> Int {
    s.parse().unwrap()
}

#[test]
fn to_f64_rounds_to_nearest_even() {
    // 2^128 + 2^75 + 1 is just above the midpoint between 2^128 and 2^128 + 2^76.
    let x = Int::ONE
        .mul_2k(128)
        .add(&Int::ONE.mul_2k(75))
        .add(&Int::ONE);
    assert_eq!(x.to_f64(), 2f64.powi(128) + 2f64.powi(76));
    assert_eq!(x.neg().to_f64(), -(2f64.powi(128) + 2f64.powi(76)));
    // Exactly the midpoint: ties to even (down to 2^128).
    let tie = Int::ONE.mul_2k(128).add(&Int::ONE.mul_2k(75));
    assert_eq!(tie.to_f64(), 2f64.powi(128));
    // Midpoint above an odd significand rounds up.
    let odd_tie = Int::ONE
        .mul_2k(128)
        .add(&Int::ONE.mul_2k(76))
        .add(&Int::ONE.mul_2k(75));
    assert_eq!(odd_tie.to_f64(), 2f64.powi(128) + 2f64.powi(77));
    // Overflow: 2^1024 − 2^970 is the first value that rounds to infinity.
    let max_mid = Int::ONE.mul_2k(1024).sub(&Int::ONE.mul_2k(970));
    assert_eq!(max_mid.to_f64(), f64::INFINITY);
    assert_eq!(max_mid.sub(&Int::ONE).to_f64(), f64::MAX);
    assert_eq!(Int::from_u64(u64::MAX).to_f64(), 2f64.powi(64));
}

#[test]
fn to_f64_trunc_matches_mpz_get_d() {
    let x = Int::ONE
        .mul_2k(128)
        .add(&Int::ONE.mul_2k(76))
        .sub(&Int::ONE);
    assert_eq!(x.to_f64_trunc(), 2f64.powi(128));
    assert_eq!(x.neg().to_f64_trunc(), -2f64.powi(128));
    assert_eq!(
        Int::from_u64(u64::MAX).to_f64_trunc(),
        18446744073709549568.0
    );
    assert_eq!(
        Int::ONE.mul_2k(1024).sub(&Int::ONE).to_f64_trunc(),
        f64::MAX
    );
    assert_eq!(Int::ONE.mul_2k(1024).to_f64_trunc(), f64::INFINITY);
    assert_eq!(Nat::from_u64(3).to_f64_trunc(), 3.0);
}

#[test]
fn kronecker_and_jacobi_accept_any_modulus() {
    let k = |a: i64, n: i64| Int::from_i64(a).kronecker(&Int::from_i64(n));
    assert_eq!(k(5, -8), -1);
    assert_eq!(k(-3, 6), 0);
    assert_eq!(k(7, 0), 0);
    assert_eq!(k(-1, 0), 1);
    assert_eq!(k(0, -1), 1);
    assert_eq!(k(-5, -12), -1);
    assert_eq!(k(2, 15), 1);
    // `jacobi` no longer panics outside odd positive n.
    assert_eq!(Int::from_i64(5).jacobi(&Int::from_i64(-8)), -1);
    assert_eq!(Int::from_i64(3).jacobi(&Int::from_i64(4)), 1);
}

#[test]
fn twos_complement_bits() {
    let m6 = Int::from_i64(-6); // …11010
    assert!(!m6.test_bit(0));
    assert!(m6.test_bit(1));
    assert!(!m6.test_bit(2));
    assert!(m6.test_bit(100));
    assert!(m6.bit(2)); // `bit` reads |−6| = 110

    let with = |x: i64, f: fn(&mut Int, u64), i: u64| {
        let mut v = Int::from_i64(x);
        f(&mut v, i);
        v
    };
    assert_eq!(with(-6, Int::set_bit, 0), Int::from_i64(-5));
    assert_eq!(with(-6, Int::set_bit, 2), Int::from_i64(-2));
    assert_eq!(with(-6, Int::clear_bit, 1), Int::from_i64(-8));
    assert_eq!(with(5, Int::clear_bit, 100), Int::from_i64(5));
    assert_eq!(
        with(-6, Int::complement_bit, 100),
        int("-1267650600228229401496703205382")
    );
    assert_eq!(!Int::from_i64(5), Int::from_i64(-6));
}

#[test]
fn popcount_hamming_scan() {
    assert_eq!(Int::from_i64(0b1011).count_ones(), Some(3));
    assert_eq!(Int::from_i64(-6).count_ones(), None);
    assert_eq!(
        Int::from_i64(-6).hamming_distance(&Int::from_i64(-1)),
        Some(2)
    );
    assert_eq!(Int::from_i64(6).hamming_distance(&Int::from_i64(-1)), None);
    assert_eq!(
        Int::from_i64(6).hamming_distance(&Int::from_i64(9)),
        Some(4)
    );
    assert_eq!(Int::from_i64(-1).scan0(5), None);
    assert_eq!(Int::from_i64(-8).scan1(0), Some(3));
    assert_eq!(Int::from_i64(-8).scan0(3), None);
    assert_eq!(Int::from_i64(5).scan1(3), None);
    assert_eq!(Int::from_i64(5).scan0(3), Some(3));
    assert_eq!(Int::ONE.mul_2k(200).scan1(1), Some(200));
    assert_eq!(Nat::from_u64(0b1011).scan0(0), 2);
}

#[test]
fn rounding_power_of_two_division() {
    let m7 = Int::from_i64(-7);
    assert_eq!(m7.div_2k_floor(1), Int::from_i64(-4));
    assert_eq!(&m7 >> 1, Int::from_i64(-4));
    assert_eq!(m7.div_2k_trunc(1), Int::from_i64(-3));
    assert_eq!(m7.div_2k_ceil(1), Int::from_i64(-3));
    assert_eq!(Int::from_i64(7).div_2k_ceil(1), Int::from_i64(4));
    assert_eq!(m7.rem_2k_trunc(2), Int::from_i64(-3));
    assert_eq!(m7.mod_2k(2), Int::from_i64(1));
    assert_eq!(Int::from_i64(3) << 70, Int::from_i64(3).mul_2k(70));
    let mut x = Int::from_i64(-1);
    x >>= 1000;
    assert_eq!(x, Int::from_i64(-1));
}

#[test]
fn radix_up_to_62() {
    let s = |v: i64, r: u32| {
        let mut out = String::new();
        Int::from_i64(v).write_radix(&mut out, r).unwrap();
        out
    };
    assert_eq!(s(3843, 61), "120");
    assert_eq!(s(-3843, 62), "-zz");
    assert_eq!(s(123456789, 36), "21i3v9");
    assert_eq!(s(10, 37), "A");
    assert_eq!(s(36, 37), "a");
    assert_eq!(s(36, 62), "a");
    assert_eq!(
        Int::from_str_radix("-zz", 62).unwrap(),
        Int::from_i64(-3843)
    );
    assert_eq!(
        Int::from_str_radix("Zz", 62).unwrap(),
        Int::from_i64(35 * 62 + 61)
    );
    assert_eq!(
        Int::from_str_radix("21I3V9", 36).unwrap(),
        Int::from_i64(123456789)
    );
    assert!(Int::from_str_radix("a", 37).is_ok());
    assert!(Int::from_str_radix("a", 10).is_err());
    assert!(Int::from_str_radix("1", 63).is_err());
}

#[test]
fn binomial_of_integer_n() {
    assert_eq!(Int::binomial_int(&Int::from_i64(-5), 3), Int::from_i64(-35));
    assert_eq!(Int::binomial_int(&Int::from_i64(-5), 0), Int::ONE);
    assert_eq!(Int::binomial_int(&Int::from_i64(10), 3), Int::from_i64(120));
    assert_eq!(Int::binomial_int(&Int::from_i64(3), 10), Int::ZERO);
    assert_eq!(
        Int::binomial_int(&Int::ONE.mul_2k(128), 3),
        int(
            "6567001032732413202046506683357268967455393833792249680279596741536609674656346870358531073282244939666292022968320"
        )
    );
}

#[test]
fn roots_and_perfect_powers() {
    assert_eq!(
        Int::from_i64(-1000).root_rem(3),
        (Int::from_i64(-10), Int::ZERO)
    );
    assert_eq!(
        Int::from_i64(1_000_000).root_rem(4),
        (Int::from_i64(31), Int::from_i64(76479))
    );
    assert_eq!(Int::from_i64(-1001).root(3), Int::from_i64(-10));
    assert_eq!(Int::from_i64(5).root(u64::MAX), Int::ONE);
    assert_eq!(Int::from_i64(-5).root(u64::MAX), Int::MINUS_ONE);
    assert_eq!(
        Int::from_i64(99).isqrt_rem(),
        (Int::from_i64(9), Int::from_i64(18))
    );

    for (v, want) in [
        (-8, true),
        (-4, false),
        (0, true),
        (1, true),
        (-1, true),
        (2, false),
        (1162261467, true),
        (72, false),
    ] {
        assert_eq!(Int::from_i64(v).is_perfect_power(), want, "{v}");
    }
    let big_prime_power = int("1000000007").pow(13);
    assert!(big_prime_power.is_perfect_power());
    assert!(big_prime_power.neg().is_perfect_power());
    assert!(!big_prime_power.add(&Int::ONE).is_perfect_power());
    assert!(Int::from_i64(1 << 40).is_perfect_square());
    assert!(!Int::from_i64(-4).is_perfect_square());
}

#[test]
fn byte_import_export() {
    let x = int("4759477275222530853130"); // 0x0102030405060708090a
    assert_eq!(x.to_bytes_be(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert_eq!(
        Int::from_bytes_be(&[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        x
    );
    assert_eq!(x.neg().to_bytes_be(), x.to_bytes_be());
    assert_eq!(Int::ZERO.to_bytes_be(), Vec::<u8>::new());
    assert_eq!(Int::from_i64(-1).to_signed_bytes_be(), [0xff]);
    assert_eq!(Int::from_i64(128).to_signed_bytes_be(), [0x00, 0x80]);
    assert_eq!(Int::from_i64(-129).to_signed_bytes_le(), [0x7f, 0xff]);
    assert_eq!(
        Int::from_signed_bytes_be(&[0xff, 0x7f]),
        Int::from_i64(-129)
    );
    assert_eq!(Int::from_signed_bytes_le(&[]), Int::ZERO);
}

#[test]
fn extended_gcd_follows_gmp_normalization() {
    let g = |a: i64, b: i64| {
        let (g, s, t) = Int::from_i64(a).extended_gcd(&Int::from_i64(b));
        (
            g.to_i64().unwrap(),
            s.to_i64().unwrap(),
            t.to_i64().unwrap(),
        )
    };
    assert_eq!(g(0, 0), (0, 0, 0));
    assert_eq!(g(-5, 5), (5, 0, 1));
    assert_eq!(g(5, -5), (5, 0, -1));
    assert_eq!(g(240, 46), (2, -9, 47));
    assert_eq!(g(-240, 46), (2, 9, 47));
    assert_eq!(g(0, -5), (5, 0, -1));
}

#[test]
fn magnitude_access_and_wide_pow() {
    let big = Int::ONE.mul_2k(300).neg();
    assert!(matches!(big.magnitude_ref(), std::borrow::Cow::Borrowed(_)));
    assert_eq!(*big.magnitude_ref(), big.magnitude());
    assert_eq!(big.clone().into_magnitude(), big.magnitude());
    assert_eq!(Int::MINUS_ONE.pow(u64::MAX), Int::MINUS_ONE);
    assert_eq!(Int::ZERO.pow(1 << 40), Int::ZERO);
    assert_eq!(
        Int::from_i64(-2).pow(5_000_001),
        Int::ONE.mul_2k(5_000_001).neg()
    );
}

#[test]
fn mersenne_twister_reference_streams() {
    use puremp::MersenneTwister;
    // C++ requires the 10000th output of a default-seeded std::mt19937 to be
    // 4123659995.
    let mut mt = MersenneTwister::new(5489);
    assert_eq!(mt.next_u32(), 3499211612);
    for _ in 0..9998 {
        mt.next_u32();
    }
    assert_eq!(mt.next_u32(), 4123659995);

    // `from_seed` keys by the seed's 32-bit words, like Python's `random.seed`.
    for (seed, first, nth) in [
        ("0", [3626764237, 1654615998, 3255389356], 2846861434),
        ("12345", [1789368711, 3146859322, 43676229], 556952727),
        (
            "1267650600228229401496703205383",
            [1786885132, 353954449, 1403444918],
            467335321,
        ),
        (
            "18446744073709551616",
            [4198958755, 3158798261, 1593907883],
            3490021708,
        ),
    ] {
        let mut mt = MersenneTwister::from_seed(&int(seed));
        assert_eq!(
            [mt.next_u32(), mt.next_u32(), mt.next_u32()],
            first,
            "{seed}"
        );
        for _ in 0..2000 {
            mt.next_u32();
        }
        assert_eq!(mt.next_u32(), nth, "{seed}");
    }

    // urandomb / urandomm equivalents stay in range.
    let mut mt = MersenneTwister::new(1);
    let bound = int("1000000000000000000000000000");
    for _ in 0..100 {
        assert!(Int::random_bits(77, &mut mt).bit_len() <= 77);
        let r = Int::random_below(&bound, &mut mt).unwrap();
        assert!(!r.is_negative() && r < bound);
    }
}

#[test]
fn mersenne_twister_gmp_seeding() {
    use puremp::MersenneTwister;
    // `mpz_urandomb` request sequences observed from GMP 6: seeded through
    // `gmp_randseed`, and (seed 5489) an unseeded `gmp_randinit_mt`.
    for (seed, want, nth) in [
        (
            "5489",
            [
                "968665204",
                "1490187595",
                "11567",
                "62403",
                "1096904973926824930696315518408",
                "0",
                "99",
            ],
            2011339640,
        ),
        (
            "12345",
            [
                "4239593463",
                "3179460376",
                "5840",
                "32854",
                "328329684707142779870883755138",
                "0",
                "108",
            ],
            2726340516,
        ),
        (
            "-98765432109876543210",
            [
                "3671402544",
                "970948138",
                "58725",
                "57512",
                "842703645622114313430616283195",
                "0",
                "81",
            ],
            1420354290,
        ),
        (
            "0",
            [
                "1823441003",
                "3558641944",
                "26433",
                "35955",
                "249407599312337939642398084807",
                "0",
                "124",
            ],
            1676504353,
        ),
    ] {
        let mut mt = MersenneTwister::gmp_seeded(&int(seed));
        let got: Vec<String> = [32, 32, 16, 16, 100, 0, 7]
            .iter()
            .map(|&n| Int::random_bits(n, &mut mt).to_string())
            .collect();
        assert_eq!(got, want, "seed {seed}");
        for _ in 0..2000 {
            Int::random_bits(32, &mut mt);
        }
        assert_eq!(mt.next_u32(), nth, "seed {seed}");
    }
}
