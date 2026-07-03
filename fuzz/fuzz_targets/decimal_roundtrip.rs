#![no_main]

use libfuzzer_sys::fuzz_target;
use puremp::{Decimal, Int};

fn to_decimal(bytes: &[u8]) -> Decimal {
    if bytes.len() < 2 {
        return Decimal::zero();
    }
    let neg = bytes[0] & 1 == 1;
    let exp = (bytes[1] % 13) as i64 - 6; // exponent in [-6, 6]
    let mut s = String::new();
    for b in &bytes[2..] {
        s.push(char::from(b'0' + (b % 10)));
    }
    if s.is_empty() {
        s.push('0');
    }
    let mag: Int = s.parse().expect("ASCII digits");
    Decimal::new(if neg { mag.neg() } else { mag }, exp)
}

// Decimal is exact under + - *, to_rational is a ring homomorphism, and the
// plain-decimal Display round-trips through FromStr.
fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let mid = data.len() / 2;
    let a = to_decimal(&data[..mid]);
    let b = to_decimal(&data[mid..]);
    assert_eq!(a.add(&b).sub(&b), a, "(a + b) - b == a");
    assert_eq!(a.mul(&b), b.mul(&a), "commutative *");
    assert_eq!(
        a.add(&b).to_rational(),
        a.to_rational().add(&b.to_rational()),
        "to_rational homomorphism"
    );
    assert_eq!(a.to_string().parse::<Decimal>().unwrap(), a, "Display round-trip");
});
