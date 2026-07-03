#![no_main]

use libfuzzer_sys::fuzz_target;
use puremp::Int;

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

// Invariant of truncated division: q·d + r == n, and |r| < |d| when d ≠ 0.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let mid = data.len() / 2;
    let n = to_int(&data[..mid], data[0] & 1 == 1);
    let d = to_int(&data[mid..], data[mid] & 1 == 1);

    match n.div_rem(&d) {
        Some((q, r)) => {
            assert_eq!(q.mul(&d).add(&r), n, "q·d + r == n");
            assert!(r.magnitude() < d.magnitude(), "|r| < |d|");
        }
        None => assert!(d.is_zero(), "div_rem only fails on a zero divisor"),
    }
});
