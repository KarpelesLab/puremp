//! Integration tests for `GF(pᵏ)` finite field extensions.
//!
//! These exercise the [`GaloisField`]/[`GfElement`] types against known facts and
//! axioms: agreement with [`ModInt`] for prime fields, the documented AES
//! `GF(2⁸)` inverse pair `0x53 ↔ 0xCA`, the field axioms over several fields,
//! Fermat/Lagrange (`a^(pᵏ−1) = 1`, `a^(pᵏ) = a`), the freshman's dream
//! `(a+b)ᵖ = aᵖ + bᵖ`, Rabin irreducibility (including rejection of reducible
//! moduli and non-prime characteristics), and exhaustive checks over the small
//! fields `GF(2³)` and `GF(3²)`.

use puremp::galois::{GaloisField, GfElement};
use puremp::{Int, ModInt};

fn i(v: i64) -> Int {
    Int::from_i64(v)
}

/// All `pᵏ` elements of a field, as coefficient vectors (base-`p` counter).
fn all_elements(field: &GaloisField) -> Vec<GfElement> {
    let p = field.characteristic().to_u64().unwrap() as usize;
    let k = field.degree();
    let mut out = Vec::new();
    let total: usize = p.pow(k as u32);
    for mut n in 0..total {
        let mut coeffs = Vec::with_capacity(k);
        for _ in 0..k {
            coeffs.push(i((n % p) as i64));
            n /= p;
        }
        out.push(field.element(&coeffs));
    }
    out
}

// ---------------------------------------------------------------------------
// GF(p), k = 1: must match ModInt.
// ---------------------------------------------------------------------------

#[test]
fn prime_field_matches_mod_int() {
    for p in [5i64, 7] {
        let field = GaloisField::create(i(p), 1).unwrap();
        for a in 0..p {
            for b in 0..p {
                let ga = field.from_int(&i(a));
                let gb = field.from_int(&i(b));
                let ma = ModInt::new(i(a), i(p));
                let mb = ModInt::new(i(b), i(p));

                assert_eq!(ga.add(&gb).to_coefficients()[0], ma.add(&mb).to_int());
                assert_eq!(ga.mul(&gb).to_coefficients()[0], ma.mul(&mb).to_int());
                assert_eq!(ga.sub(&gb).to_coefficients()[0], ma.sub(&mb).to_int());

                if a != 0 {
                    assert_eq!(
                        ga.inv().unwrap().to_coefficients()[0],
                        ma.inv().unwrap().to_int()
                    );
                    // pow against ModInt::pow
                    let e = i(3);
                    assert_eq!(ga.pow(&e).to_coefficients()[0], ma.pow(&e).to_int());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AES field GF(2⁸), modulus x⁸ + x⁴ + x³ + x + 1 (0x11B).
// ---------------------------------------------------------------------------

/// AES modulus coefficients, low-to-high.
fn aes_modulus() -> Vec<Int> {
    // bits set at 0,1,3,4,8
    let mut m = vec![i(0); 9];
    for &b in &[0usize, 1, 3, 4, 8] {
        m[b] = i(1);
    }
    m
}

/// A byte as a length-8 coefficient bit-vector (low bit first).
fn byte_elem(field: &GaloisField, byte: u8) -> GfElement {
    let coeffs: Vec<Int> = (0..8).map(|b| i(((byte >> b) & 1) as i64)).collect();
    field.element(&coeffs)
}

/// A field element's low 8 coefficients packed back into a byte.
fn elem_byte(e: &GfElement) -> u8 {
    let mut byte = 0u8;
    for (b, c) in e.to_coefficients().iter().enumerate() {
        if c.is_one() {
            byte |= 1 << b;
        }
    }
    byte
}

#[test]
fn aes_field_inverse_pair() {
    let field = GaloisField::new(i(2), &aes_modulus()).expect("AES modulus is irreducible");
    assert_eq!(field.degree(), 8);
    assert_eq!(field.order(), i(256));

    let a = byte_elem(&field, 0x53);
    let b = byte_elem(&field, 0xCA);
    // Documented AES inverse pair: 0x53 · 0xCA = 1.
    assert!(a.mul(&b).is_one(), "0x53 · 0xCA must equal 1 in GF(2^8)");
    assert_eq!(elem_byte(&a.inv().unwrap()), 0xCA);
    assert_eq!(elem_byte(&b.inv().unwrap()), 0x53);
}

#[test]
fn aes_known_multiplications() {
    let field = GaloisField::new(i(2), &aes_modulus()).unwrap();
    // Classic AES worked example: 0x57 · 0x83 = 0xC1.
    let a = byte_elem(&field, 0x57);
    let b = byte_elem(&field, 0x83);
    assert_eq!(elem_byte(&a.mul(&b)), 0xC1);
    // 0x57 · 0x13 = 0xFE (from the AES specification's example set).
    let c = byte_elem(&field, 0x13);
    assert_eq!(elem_byte(&a.mul(&c)), 0xFE);
}

#[test]
fn aes_every_nonzero_has_inverse() {
    let field = GaloisField::new(i(2), &aes_modulus()).unwrap();
    for byte in 1u8..=255 {
        let e = byte_elem(&field, byte);
        let inv = e
            .inv()
            .expect("every nonzero GF(2^8) element is invertible");
        assert!(e.mul(&inv).is_one(), "byte {byte:#x} inverse check");
    }
}

// ---------------------------------------------------------------------------
// Field axioms over several fields.
// ---------------------------------------------------------------------------

fn check_axioms(field: &GaloisField, samples: &[GfElement]) {
    let zero = field.zero();
    let one = field.one();
    for a in samples {
        // a + 0 = a ; a · 1 = a
        assert_eq!(&a.add(&zero), a);
        assert_eq!(&a.mul(&one), a);
        // a · a⁻¹ = 1 for a ≠ 0
        if !a.is_zero() {
            assert!(a.mul(&a.inv().unwrap()).is_one());
        }
        for b in samples {
            // commutativity
            assert_eq!(a.add(b), b.add(a));
            assert_eq!(a.mul(b), b.mul(a));
            for c in samples {
                // associativity
                assert_eq!(a.add(b).add(c), a.add(&b.add(c)));
                assert_eq!(a.mul(b).mul(c), a.mul(&b.mul(c)));
                // distributivity
                assert_eq!(a.mul(&b.add(c)), a.mul(b).add(&a.mul(c)));
            }
        }
    }
}

#[test]
fn axioms_over_several_fields() {
    // GF(2³)
    let f8 = GaloisField::create(i(2), 3).unwrap();
    check_axioms(&f8, &all_elements(&f8));
    // GF(3²)
    let f9 = GaloisField::create(i(3), 2).unwrap();
    check_axioms(&f9, &all_elements(&f9));
    // GF(2⁸): a handful of samples (full enumeration is too big here).
    let f256 = GaloisField::new(i(2), &aes_modulus()).unwrap();
    let samples: Vec<GfElement> = [0u8, 1, 2, 0x53, 0xCA, 0x57, 0x83, 0xFF]
        .iter()
        .map(|&b| byte_elem(&f256, b))
        .collect();
    check_axioms(&f256, &samples);
    // GF(101²): a larger prime.
    let f = GaloisField::create(i(101), 2).unwrap();
    let samples: Vec<GfElement> = [
        f.zero(),
        f.one(),
        f.generator(),
        f.element(&[i(5), i(7)]),
        f.element(&[i(100), i(50)]),
        f.from_int(&i(42)),
    ]
    .to_vec();
    check_axioms(&f, &samples);
}

// ---------------------------------------------------------------------------
// Fermat / Lagrange: a^(pᵏ−1) = 1 (a≠0), a^(pᵏ) = a.
// ---------------------------------------------------------------------------

#[test]
fn fermat_lagrange() {
    for (p, k) in [(2i64, 3usize), (3, 2), (5, 2)] {
        let field = GaloisField::create(i(p), k).unwrap();
        let order = field.order();
        let n = order.sub(&Int::ONE); // pᵏ − 1
        for a in all_elements(&field) {
            // a^(pᵏ) = a  for all a
            assert_eq!(a.pow(&order), a);
            if !a.is_zero() {
                // a^(pᵏ−1) = 1  for a ≠ 0
                assert!(a.pow(&n).is_one());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Freshman's dream: (a+b)ᵖ = aᵖ + bᵖ, (a·b)ᵖ = aᵖ·bᵖ.
// ---------------------------------------------------------------------------

#[test]
fn freshmans_dream() {
    for (p, k) in [(2i64, 3usize), (3, 2), (5, 2)] {
        let field = GaloisField::create(i(p), k).unwrap();
        let pe = i(p);
        let elems = all_elements(&field);
        for a in &elems {
            for b in &elems {
                let lhs = a.add(b).pow(&pe);
                let rhs = a.pow(&pe).add(&b.pow(&pe));
                assert_eq!(lhs, rhs, "(a+b)^p = a^p + b^p");
                assert_eq!(a.mul(b).pow(&pe), a.pow(&pe).mul(&b.pow(&pe)));
                // frobenius agrees with elem^p
                assert_eq!(field.frobenius(a), a.pow(&pe));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Irreducibility: create's modulus passes Rabin; new rejects reducible/bad p.
// ---------------------------------------------------------------------------

#[test]
fn create_produces_irreducible() {
    for (p, k) in [(2i64, 8usize), (3, 4), (5, 3), (7, 2)] {
        let field = GaloisField::create(i(p), k).unwrap();
        assert_eq!(field.degree(), k);
        // Rebuilding from the found modulus via `new` must succeed (re-runs Rabin).
        assert!(GaloisField::new(i(p), field.modulus()).is_some());
    }
}

#[test]
fn new_rejects_reducible_modulus() {
    // x² − 1 = (x−1)(x+1) over GF(3): reducible → None.
    // coeffs low-to-high: [-1, 0, 1] ≡ [2, 0, 1] mod 3.
    assert!(GaloisField::new(i(3), &[i(-1), i(0), i(1)]).is_none());
    // x² + 1 over GF(2) = (x+1)² : reducible → None.
    assert!(GaloisField::new(i(2), &[i(1), i(0), i(1)]).is_none());
    // But x² + x + 1 over GF(2) is irreducible → Some.
    assert!(GaloisField::new(i(2), &[i(1), i(1), i(1)]).is_some());
}

#[test]
fn new_rejects_non_prime() {
    // Characteristic 4 is not prime, even with an otherwise-fine modulus shape.
    assert!(GaloisField::new(i(4), &[i(1), i(0), i(1)]).is_none());
    assert!(GaloisField::create(i(6), 2).is_none());
    // k == 0 is rejected.
    assert!(GaloisField::create(i(5), 0).is_none());
}

#[test]
fn non_monic_modulus_rejected() {
    // Leading coefficient 2 over GF(3): not monic → None.
    assert!(GaloisField::new(i(3), &[i(1), i(0), i(2)]).is_none());
}

// ---------------------------------------------------------------------------
// Exhaustive small fields: GF(2³) and GF(3²).
// ---------------------------------------------------------------------------

fn exhaustive_field(field: &GaloisField) {
    let elems = all_elements(field);
    let order = field.order().to_u64().unwrap() as usize;
    assert_eq!(elems.len(), order);

    // Every nonzero element has a unique inverse; no zero divisors.
    for a in &elems {
        if a.is_zero() {
            assert!(a.inv().is_none());
            continue;
        }
        let inv = a.inv().unwrap();
        assert!(a.mul(&inv).is_one());
        // uniqueness: exactly one x with a·x = 1
        let count = elems.iter().filter(|x| a.mul(x).is_one()).count();
        assert_eq!(count, 1);
        // no zero divisors: a·b = 0 only when b = 0
        for b in &elems {
            if a.mul(b).is_zero() {
                assert!(b.is_zero());
            }
        }
    }

    // A primitive element exists whose powers cover all of GF*.
    let g = field.primitive_element();
    let mut seen = std::collections::BTreeSet::new();
    let mut acc = field.one();
    for _ in 0..(order - 1) {
        let bytes: Vec<i64> = acc
            .to_coefficients()
            .iter()
            .map(|c| c.to_u64().unwrap() as i64)
            .collect();
        assert!(seen.insert(bytes), "primitive powers must be distinct");
        acc = acc.mul(&g);
    }
    // After pᵏ−1 multiplications we return to 1.
    assert!(acc.is_one());
    assert_eq!(seen.len(), order - 1); // all nonzero elements covered
}

#[test]
fn exhaustive_gf8() {
    exhaustive_field(&GaloisField::create(i(2), 3).unwrap());
}

#[test]
fn exhaustive_gf9() {
    exhaustive_field(&GaloisField::create(i(3), 2).unwrap());
}

// ---------------------------------------------------------------------------
// Operators, Display, mismatched-field panic.
// ---------------------------------------------------------------------------

#[test]
fn operators_and_display() {
    let field = GaloisField::create(i(3), 2).unwrap();
    let a = field.element(&[i(1), i(2)]); // 2·a + 1
    let b = field.generator(); // a
    // Operator round-trips (all owned/borrowed combos delegate to the methods).
    assert_eq!(&a + &b, a.add(&b));
    assert_eq!(a.clone() + b.clone(), a.add(&b));
    assert_eq!(&a * &b, a.mul(&b));
    assert_eq!(&a - &b, a.sub(&b));
    assert_eq!(-&a, a.neg());
    let nonzero = field.element(&[i(2), i(1)]);
    assert_eq!(&a / &nonzero, a.div(&nonzero));

    // Display uses the generator symbol `a`.
    assert_eq!(field.element(&[i(1), i(2)]).to_string(), "2·a + 1");
    assert_eq!(field.generator().to_string(), "a");
    assert_eq!(field.zero().to_string(), "0");
    assert_eq!(field.from_int(&i(2)).to_string(), "2");
}

#[test]
#[should_panic(expected = "different fields")]
fn mismatched_fields_panic() {
    let f1 = GaloisField::create(i(2), 3).unwrap();
    let f2 = GaloisField::create(i(3), 2).unwrap();
    let _ = f1.one().add(&f2.one());
}

// ---------------------------------------------------------------------------
// Differential: the optimized `mul` (fused schoolbook + precomputed x^k…x^{2k-2}
// reduction table) must be bit-identical to reducing the raw schoolbook product
// through the untouched `GaloisField::element` path (schoolbook `poly_mul` +
// general `poly_rem`). Also confirms `pow`, `inv`, `div` (built on `mul`) stay
// consistent with the field axioms over many random elements.
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

/// A random coefficient in `[0, p)` for a possibly-large prime `p`.
fn rand_coeff(p: &Int, rng: &mut Lcg) -> Int {
    // Assemble up to three 64-bit words, then reduce mod p.
    let mut v = Int::from_u64(rng.next_u64());
    for _ in 0..2 {
        v = v
            .mul(&Int::from_u64(u64::MAX))
            .add(&Int::from_u64(rng.next_u64()));
    }
    v.rem_euclid(p)
}

fn rand_elem_diff(field: &GaloisField, p: &Int, rng: &mut Lcg) -> GfElement {
    let coeffs: Vec<Int> = (0..field.degree()).map(|_| rand_coeff(p, rng)).collect();
    field.element(&coeffs)
}

/// Reference product: schoolbook multiply of the two coefficient vectors over
/// `ℤ` (no reduction), then reduce through `GaloisField::element`, which runs
/// the original, unmodified `poly_mul`/`poly_rem` reduction.
fn reference_mul(field: &GaloisField, a: &GfElement, b: &GfElement) -> GfElement {
    let ac = a.to_coefficients();
    let bc = b.to_coefficients();
    let mut prod = vec![Int::ZERO; ac.len() + bc.len() - 1];
    for (i, ai) in ac.iter().enumerate() {
        for (j, bj) in bc.iter().enumerate() {
            prod[i + j] = prod[i + j].add(&ai.mul(bj));
        }
    }
    field.element(&prod)
}

#[test]
fn mul_matches_reference_reduction() {
    // Small word-size primes with k = 2…8, plus a couple of larger primes.
    let cases: &[(Int, usize)] = &[
        (i(2), 2),
        (i(2), 5),
        (i(2), 8),
        (i(3), 3),
        (i(3), 6),
        (i(5), 4),
        (i(7), 7),
        (i(101), 4),
        (i(65537), 3),
        (i(2_147_483_647), 2),
    ];
    for (p, k) in cases {
        let field = GaloisField::create(p.clone(), *k).unwrap();
        let mut rng = Lcg::new(p.to_u64().unwrap_or(0) ^ (*k as u64 * 0x1000));
        for _ in 0..400 {
            let a = rand_elem_diff(&field, p, &mut rng);
            let b = rand_elem_diff(&field, p, &mut rng);
            let got = a.mul(&b);
            let want = reference_mul(&field, &a, &b);
            assert_eq!(
                got.to_coefficients(),
                want.to_coefficients(),
                "mul mismatch in GF({p}^{k}) for {a:?} · {b:?}"
            );
            // pow / inv / div consistency (all built on the optimized mul).
            assert_eq!(a.mul(&b), b.mul(&a));
            if !a.is_zero() {
                assert!(a.mul(&a.inv().unwrap()).is_one());
                assert_eq!(b.div(&a).mul(&a), b);
            }
        }
        // A high exponent stresses the square-and-multiply chain.
        let g = field.generator();
        assert_eq!(g.pow(&field.order()), g); // g^(p^k) = g
    }
}

/// A large multi-limb prime characteristic (~130 bits) with k = 2, 3, 4.
#[test]
fn mul_matches_reference_large_prime() {
    let p = Int::from_str_radix("170141183460469231731687303715884105727", 10)
        .unwrap()
        .next_prime();
    for k in [2usize, 3, 4] {
        let field = GaloisField::create(p.clone(), k).unwrap();
        let mut rng = Lcg::new(0xdead_beef ^ (k as u64));
        for _ in 0..60 {
            let a = rand_elem_diff(&field, &p, &mut rng);
            let b = rand_elem_diff(&field, &p, &mut rng);
            assert_eq!(
                a.mul(&b).to_coefficients(),
                reference_mul(&field, &a, &b).to_coefficients(),
                "large-prime mul mismatch in GF(p^{k})"
            );
        }
    }
}
