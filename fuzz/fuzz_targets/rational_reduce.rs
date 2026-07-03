#![no_main]

use libfuzzer_sys::fuzz_target;
use puremp::{Int, Nat, Rational};

fn to_int(bytes: &[u8], negative: bool) -> Int {
    let mut s = String::new();
    if negative {
        s.push('-');
    }
    for b in bytes {
        s.push(char::from(b'0' + (b % 10)));
    }
    if bytes.is_empty() {
        s.push('0');
    }
    s.parse().expect("optional sign followed by ASCII digits")
}

// Invariant: a constructed rational is in lowest terms — gcd(|num|, den) == 1,
// and zero is exactly 0/1.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let mid = data.len() / 2;
    let num = to_int(&data[..mid], data[0] & 1 == 1);
    let den = to_int(&data[mid..], data[mid] & 1 == 1);

    match Rational::new(num, den.clone()) {
        Ok(r) => {
            if r.numerator().is_zero() {
                assert_eq!(r.denominator(), &Nat::one(), "zero is 0/1");
            } else {
                let g = r.numerator().magnitude().gcd(r.denominator());
                assert_eq!(g, Nat::one(), "fraction is in lowest terms");
            }
        }
        Err(_) => assert!(den.is_zero(), "construction only fails on a zero denominator"),
    }
});
