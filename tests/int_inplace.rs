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
