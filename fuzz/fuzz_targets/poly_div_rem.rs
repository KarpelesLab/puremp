#![no_main]

use libfuzzer_sys::fuzz_target;
use puremp::{Int, Poly, Rational};

fn to_poly(bytes: &[u8]) -> Poly<Rational> {
    Poly::new(
        bytes
            .iter()
            .map(|&b| Rational::from_integer(Int::from_i64((b as i8) as i64)))
            .collect(),
    )
}

// Polynomial division identity over ℚ: a = q·b + r with deg(r) < deg(b).
fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let mid = data.len() / 2;
    let a = to_poly(&data[..mid]);
    let b = to_poly(&data[mid..]);
    if b.is_zero() {
        return;
    }
    let (q, r) = a.div_rem(&b);
    assert_eq!(q.mul(&b).add(&r), a, "a == q·b + r");
    if !r.is_zero() {
        assert!(r.degree().unwrap() < b.degree().unwrap(), "deg r < deg b");
    }
});
