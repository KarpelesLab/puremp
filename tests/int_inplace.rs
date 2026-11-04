//! Differential tests for the clone-free / in-place `Int` and `Nat` paths
//! (borrowed-magnitude add/sub/mul/div/cmp, `+=`/`-=`, the `u64` helpers and
//! the owned-limb constructors) against an `i128` oracle and against each
//! other on multi-limb values.
#![cfg(feature = "int")]

use core::cmp::Ordering;
use puremp::{Int, Nat, Sign};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 ^ (self.0 >> 29)
    }
    /// A value with 0–4 limbs, a random sign and, often, edge limbs.
    fn int(&mut self) -> Int {
        let n = (self.next() % 5) as usize;
        let limbs: Vec<u64> = (0..n)
            .map(|_| match self.next() % 4 {
                0 => u64::MAX,
                1 => 0,
                _ => self.next(),
            })
            .collect();
        let sign = if self.next() & 1 == 0 {
            Sign::Positive
        } else {
            Sign::Negative
        };
        Int::from_limbs(sign, &limbs)
    }
}

/// Reference via two's-complement-free sign/magnitude on Nat (independent of
/// the Int fast paths): value = sign · magnitude.
fn sm(x: &Int) -> (bool, Nat) {
    (x.is_negative(), x.magnitude())
}

fn add_ref(a: &Int, b: &Int) -> Int {
    let ((na, ma), (nb, mb)) = (sm(a), sm(b));
    let sign = |neg: bool| if neg { Sign::Negative } else { Sign::Positive };
    if na == nb {
        return Int::from_sign_magnitude(sign(na), ma.add(&mb));
    }
    match ma.cmp(&mb) {
        Ordering::Equal => Int::ZERO,
        Ordering::Greater => Int::from_sign_magnitude(sign(na), ma.checked_sub(&mb).unwrap()),
        Ordering::Less => Int::from_sign_magnitude(sign(nb), mb.checked_sub(&ma).unwrap()),
    }
}

#[test]
fn small_values_match_i128() {
    let vals: Vec<i64> = vec![0, 1, -1, 2, -2, i64::MAX, i64::MIN, 12345, -98765];
    for &a in &vals {
        for &b in &vals {
            let (ia, ib) = (Int::from(a), Int::from(b));
            let (wa, wb) = (a as i128, b as i128);
            assert_eq!(ia.add(&ib), Int::from(wa + wb));
            assert_eq!(ia.sub(&ib), Int::from(wa - wb));
            assert_eq!(ia.mul(&ib), Int::from(wa * wb));
            assert_eq!(ia.cmp(&ib), wa.cmp(&wb));
            let mut x = ia.clone();
            x += &ib;
            assert_eq!(x, Int::from(wa + wb));
            let mut x = ia.clone();
            x -= &ib;
            assert_eq!(x, Int::from(wa - wb));
            if b != 0 {
                let (q, r) = ia.div_rem_trunc(&ib);
                assert_eq!((q, r), (Int::from(wa / wb), Int::from(wa % wb)));
            }
            let ub = b.unsigned_abs();
            assert_eq!(ia.add_u64(ub), Int::from(wa + ub as i128));
            assert_eq!(ia.sub_u64(ub), Int::from(wa - ub as i128));
            assert_eq!(ia.mul_u64(ub), Int::from(wa * ub as i128));
        }
    }
}

#[test]
fn multi_limb_paths_agree() {
    let mut rng = Rng(0x5eed);
    for _ in 0..4000 {
        let a = rng.int();
        let b = rng.int();
        let sum = add_ref(&a, &b);
        let diff = add_ref(&a, &b.neg());
        assert_eq!(a.add(&b), sum);
        assert_eq!(&a + &b, sum);
        assert_eq!(a.clone() + &b, sum);
        assert_eq!(a.clone() + b.clone(), sum);
        assert_eq!(a.sub(&b), diff);
        assert_eq!(a.clone() - &b, diff);
        let mut x = a.clone();
        x += &b;
        assert_eq!(x, sum);
        x -= &b;
        assert_eq!(x, a);
        assert_eq!(-a.clone(), a.neg());
        // cmp is consistent with the sign of the difference
        assert_eq!(a.cmp(&b), diff.signum().cmp(&0));
        // mul: commutative, and agrees with the Nat product
        let p = a.mul(&b);
        assert_eq!(p, b.mul(&a));
        assert_eq!(p.magnitude(), a.magnitude().mul(&b.magnitude()));
        assert_eq!(
            p.is_negative(),
            a.is_negative() != b.is_negative() && !p.is_zero()
        );
        // division identity
        if !b.is_zero() {
            let (q, r) = a.div_rem_trunc(&b);
            assert_eq!(q.mul(&b).add(&r), a);
            assert!(r.magnitude() < b.magnitude());
            assert!(r.is_zero() || r.is_negative() == a.is_negative());
        }
        // fused forms
        let mut f = a.clone();
        f.addmul(&a, &b);
        assert_eq!(f, a.add(&p));
        let mut f = a.clone();
        f.submul(&a, &b);
        assert_eq!(f, a.sub(&p));
        // u64 helpers
        let v = rng.next();
        let iv = Int::from(v);
        assert_eq!(a.add_u64(v), a.add(&iv));
        assert_eq!(a.sub_u64(v), a.sub(&iv));
        assert_eq!(a.mul_u64(v), a.mul(&iv));
        let mut y = a.clone();
        y.add_assign_u64(v);
        assert_eq!(y, a.add(&iv));
        y.sub_assign_u64(v);
        assert_eq!(y, a);
        y.mul_assign_u64(v);
        assert_eq!(y, a.mul(&iv));
    }
}

#[test]
fn nat_owned_limbs_and_u64_ops() {
    let mut rng = Rng(42);
    for _ in 0..2000 {
        let a = rng.int().magnitude();
        let v = rng.next();
        let nv = Nat::from_u64(v);
        assert_eq!(a.add_u64(v), a.add(&nv));
        assert_eq!(a.mul_u64(v), a.mul(&nv));
        assert_eq!(a.checked_sub_u64(v), a.checked_sub(&nv));
        let mut x = a.clone();
        x += v;
        assert_eq!(x, a.add(&nv));
        let mut x = a.clone();
        x *= v;
        assert_eq!(x, a.mul(&nv));
        let b = rng.int().magnitude();
        let mut x = a.clone();
        x += &b;
        assert_eq!(x, a.add(&b));
        let mut x = a.clone();
        x *= &b;
        assert_eq!(x, a.mul(&b));
        // owned limb round trip, with denormalized input
        let mut limbs = a.clone().into_limbs();
        assert_eq!(limbs.as_slice(), a.as_limbs());
        limbs.push(0);
        limbs.push(0);
        assert_eq!(Nat::from_limb_vec(limbs), a);
        let neg = Int::from_limb_vec(Sign::Negative, a.clone().into_limbs());
        assert_eq!(neg, Int::from_limbs(Sign::Negative, a.as_limbs()));
    }
    assert_eq!(Nat::from_limb_vec(vec![0, 0]), Nat::zero());
    assert_eq!(Nat::zero().checked_sub_u64(1), None);
    assert_eq!(Nat::from_u64(5).checked_sub_u64(5), Some(Nat::zero()));
}

#[test]
fn ascii_radix_io_round_trips() {
    let mut rng = Rng(7);
    for _ in 0..500 {
        let x = rng.int();
        for radix in [2u32, 10, 16, 36, 62] {
            let mut s = String::new();
            x.write_radix(&mut s, radix).unwrap();
            let bytes = x.to_ascii_radix(radix);
            assert_eq!(bytes, s.as_bytes());
            assert_eq!(Int::from_ascii_radix(&bytes, radix).unwrap(), x);
            let m = x.magnitude();
            let mb = m.to_ascii_radix(radix);
            assert_eq!(Nat::from_ascii_radix(&mb, radix).unwrap(), m);
            // appends after existing content
            let mut buf = b"x=".to_vec();
            x.write_ascii_radix(&mut buf, radix);
            assert_eq!(&buf[2..], &bytes[..]);
        }
    }
    // long values go through the divide-and-conquer paths
    let big = Int::pow10(5000).sub_u64(1).neg();
    let b = big.to_ascii_radix(10);
    assert_eq!(b.len(), 5001);
    assert_eq!(Int::from_ascii_radix(&b, 10).unwrap(), big);
    assert_eq!(Int::from_ascii_radix(b"+0", 10).unwrap(), Int::ZERO);
    assert_eq!(Nat::zero().to_ascii_radix(10), b"0");
    assert!(Int::from_ascii_radix(b"", 10).is_err());
    assert!(Int::from_ascii_radix(b"-", 10).is_err());
    assert!(Nat::from_ascii_radix(b"-1", 10).is_err());
    assert!(Nat::from_ascii_radix(b"12a", 10).is_err());
    assert!(Nat::from_ascii_radix(b"1\xff", 16).is_err());
}

#[test]
fn pow10_and_wide_pow() {
    let mut p = Nat::one();
    for k in 0..200u64 {
        assert_eq!(Nat::pow10(k), p);
        assert_eq!(Int::pow10(k), Int::from(p.clone()));
        assert_eq!(Nat::from_u64(37).mul_pow10(k), p.mul_u64(37));
        assert_eq!(
            Int::from(-37).mul_pow10(k),
            Int::from(p.clone()).mul_u64(37).neg()
        );
        p = p.mul_u64(10);
    }
    assert_eq!(Nat::zero().mul_pow10(1 << 40), Nat::zero());
    assert_eq!(Nat::pow10(30).to_string(), format!("1{}", "0".repeat(30)));
    // u64 exponents: trivial bases and powers of two skip the loop
    assert_eq!(Nat::one().pow(u64::MAX), Nat::one());
    assert_eq!(Nat::zero().pow(u64::MAX), Nat::zero());
    assert_eq!(Nat::zero().pow(0), Nat::one());
    assert_eq!(Nat::from_u64(4).pow(40), Nat::one().shl(80));
    assert_eq!(Nat::from_u64(3).pow(5), Nat::from_u64(243));
}

#[test]
fn isqrt_is_floor_root() {
    let mut rng = Rng(99);
    for limbs in (0..40usize).chain([63, 64, 100, 150, 191, 192, 193, 250]) {
        for _ in 0..30 {
            let v: Vec<u64> = (0..limbs)
                .map(|_| match rng.next() % 6 {
                    0 => 0,
                    1 => u64::MAX,
                    _ => rng.next(),
                })
                .collect();
            let a = Nat::from_limb_vec(v);
            for x in [
                a.clone(),
                a.square(),
                a.square().add_u64(0).checked_sub_u64(1).unwrap_or_default(),
            ] {
                let r = x.isqrt();
                assert!(r.square() <= x, "isqrt({x}) too big");
                assert!(r.add_u64(1).square() > x, "isqrt({x}) too small");
            }
        }
    }
}

#[test]
fn radix_converter_round_trips() {
    use puremp::RadixConverter;
    let mut rng = Rng(1234);
    for radix in [2u32, 10, 16, 36, 62] {
        let conv = RadixConverter::new(radix, 20_000);
        assert_eq!(conv.radix(), radix);
        assert_eq!(conv.to_ascii(&Nat::zero()), b"0");
        for limbs in [1usize, 5, 16, 17, 40, 100, 312, 313, 400] {
            for _ in 0..3 {
                let v: Vec<u64> = (0..limbs).map(|_| rng.next()).collect();
                let x = Nat::from_limb_vec(v);
                let want = x.to_ascii_radix(radix);
                assert_eq!(conv.to_ascii(&x), want, "radix {radix}, {limbs} limbs");
                assert_eq!(conv.parse(&want).unwrap(), x);
                let mut buf = b"#".to_vec();
                conv.write_ascii(&x, &mut buf);
                assert_eq!(&buf[1..], &want[..]);
            }
        }
        // powers of the radix and their neighbours (digit boundaries)
        let p = Nat::from_u64(radix as u64).pow(3000 / radix as u64 + 900);
        for x in [p.clone(), p.checked_sub_u64(1).unwrap(), p.add_u64(1)] {
            assert_eq!(conv.to_ascii(&x), x.to_ascii_radix(radix));
            assert_eq!(conv.parse(&x.to_ascii_radix(radix)).unwrap(), x);
        }
        assert!(conv.parse(b"").is_err());
        assert!(conv.parse(b"1-2").is_err());
    }
}

#[test]
fn addmul_small_operands_do_not_overflow() {
    // u64::MAX² overflows i128: the inline fast path must fall back.
    let m = Int::from(u64::MAX);
    let want = m.mul(&m);
    let mut x = Int::ZERO;
    x.addmul(&m, &m);
    assert_eq!(x, want);
    let mut x = Int::ZERO;
    x.submul(&m, &m);
    assert_eq!(x, want.neg());
    let mut x = Int::from(i64::MIN);
    x.addmul(&m, &m.neg());
    assert_eq!(x, Int::from(i64::MIN).sub(&want));
}
