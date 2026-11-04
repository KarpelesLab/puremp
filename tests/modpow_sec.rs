//! `modpow_sec` (constant-time) must agree with `modpow` everywhere.
#![cfg(feature = "int")]

use puremp::{Int, Nat, Sign};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn nat(&mut self, limbs: usize) -> Nat {
        let v: Vec<u64> = (0..limbs)
            .map(|_| match self.next() % 8 {
                0 => 0,
                1 => u64::MAX,
                _ => self.next(),
            })
            .collect();
        Nat::from_limb_vec(v)
    }
}

#[test]
fn matches_modpow() {
    let mut rng = Rng(0xC0FFEE);
    for ml in [1usize, 2, 3, 5, 8, 17] {
        for _ in 0..12 {
            let m = rng.nat(ml).add_u64(0).shl(1).add_u64(1); // odd, non-zero
            for bl in [0usize, 1, ml, ml + 2] {
                for el in [0usize, 1, 2, 5] {
                    let b = rng.nat(bl);
                    let e = rng.nat(el);
                    assert_eq!(b.modpow_sec(&e, &m), b.modpow(&e, &m), "{b}^{e} mod {m}");
                    let bi = Int::from_sign_magnitude(Sign::Negative, b.clone());
                    let (ei, mi) = (Int::from(e.clone()), Int::from(m.clone()));
                    assert_eq!(bi.modpow_sec(&ei, &mi), bi.modpow(&ei, &mi));
                    assert_eq!(bi.modpow_sec(&ei, &mi.neg()), bi.modpow(&ei, &mi.neg()));
                }
            }
        }
    }
}

#[test]
fn edge_cases() {
    let m = Nat::from_u64(1_000_000_007);
    assert_eq!(Nat::from_u64(5).modpow_sec(&Nat::zero(), &m), Nat::one());
    assert_eq!(Nat::zero().modpow_sec(&Nat::zero(), &m), Nat::one());
    assert_eq!(Nat::zero().modpow_sec(&Nat::from_u64(3), &m), Nat::zero());
    assert_eq!(
        Nat::from_u64(5).modpow_sec(&Nat::zero(), &Nat::one()),
        Nat::zero()
    );
    assert_eq!(
        Nat::from_u64(7).modpow_sec(&Nat::from_u64(9), &Nat::one()),
        Nat::zero()
    );
    // base a multiple of the modulus, and a negative one
    assert_eq!(m.mul_u64(3).modpow_sec(&Nat::from_u64(2), &m), Nat::zero());
    let mi = Int::from(m.clone());
    assert_eq!(mi.neg().modpow_sec(&Int::from(2), &mi), Int::ZERO);
    assert_eq!(
        Int::from(-2).modpow_sec(&Int::from(3), &Int::from(11)),
        Int::from(3)
    );
    // Fermat on a 2-limb prime: 2^127 − 1
    let p = Nat::one().shl(127).checked_sub_u64(1).unwrap();
    let pm1 = p.checked_sub_u64(1).unwrap();
    assert_eq!(Nat::from_u64(3).modpow_sec(&pm1, &p), Nat::one());
}

/// Plain square-and-multiply with division-based reduction: an oracle that
/// shares no code with the Montgomery kernels.
fn reference_modpow(b: &Nat, e: &Nat, m: &Nat) -> Nat {
    let reduce = |x: Nat| x.div_rem(m).unwrap().1;
    let mut acc = reduce(Nat::one());
    let b = reduce(b.clone());
    for i in (0..e.bit_len()).rev() {
        acc = reduce(acc.square());
        if e.bit(i) {
            acc = reduce(acc.mul(&b));
        }
    }
    acc
}

/// Odd moduli across the column/dot-product kernel switch-overs (12 and 32
/// limbs), with extreme limb patterns (all-ones modulus, base m − 1, ...).
#[test]
fn montgomery_kernel_sizes() {
    let mut rng = Rng(0xFACADE);
    for ml in [
        2usize, 4, 7, 8, 11, 12, 13, 16, 23, 24, 31, 32, 33, 40, 48, 64,
    ] {
        let ones = Nat::from_limb_vec(vec![u64::MAX; ml]);
        let mut limbs: Vec<u64> = (0..ml).map(|_| rng.next()).collect();
        limbs[0] |= 1;
        limbs[ml - 1] |= 1 << 63;
        let random = Nat::from_limb_vec(limbs);
        let low_top = Nat::one().shl(64 * (ml as u64 - 1)).add_u64(1);
        for m in [ones, random, low_top] {
            let m1 = m.checked_sub_u64(1).unwrap();
            let b_rand = rng.nat(ml + 1);
            for b in [m1.clone(), b_rand, Nat::from_u64(2)] {
                for e in [rng.nat(2), m1.clone(), Nat::from_u64(65537)] {
                    let want = reference_modpow(&b, &e, &m);
                    assert_eq!(b.modpow(&e, &m), want, "modpow {ml} limbs");
                    assert_eq!(b.modpow_sec(&e, &m), want, "modpow_sec {ml} limbs");
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "modulus must be odd")]
fn even_modulus_panics() {
    Nat::from_u64(3).modpow_sec(&Nat::one(), &Nat::from_u64(10));
}

#[test]
#[should_panic(expected = "modulus must be odd")]
fn zero_modulus_panics() {
    Nat::from_u64(3).modpow_sec(&Nat::one(), &Nat::zero());
}
