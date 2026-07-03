#![no_main]

use libfuzzer_sys::fuzz_target;
use puremp::Nat;

// Invariant: parsing then formatting a decimal natural is the identity, once
// leading zeros are canonicalized away.
fuzz_target!(|data: &[u8]| {
    let mut s: String = data.iter().map(|b| char::from(b'0' + (b % 10))).collect();
    if s.is_empty() {
        s.push('0');
    }
    let n: Nat = s.parse().expect("all characters are ASCII digits");

    let canon = s.trim_start_matches('0');
    let canon = if canon.is_empty() { "0" } else { canon };
    assert_eq!(n.to_string(), canon);
});
