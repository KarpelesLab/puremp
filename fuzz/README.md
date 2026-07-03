# puremp fuzz targets

Coverage-guided fuzzing via [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz)
(libFuzzer). This crate is intentionally **excluded** from the main workspace
(see the root `Cargo.toml`), because libFuzzer needs nightly Rust.

```console
$ cargo install cargo-fuzz
$ cargo +nightly fuzz run nat_roundtrip
$ cargo +nightly fuzz run int_div_rem
$ cargo +nightly fuzz run rational_reduce
```

Targets assert arithmetic **invariants** on arbitrary inputs rather than
comparing against a foreign oracle (the crate ships no foreign code):

- `nat_roundtrip` — `parse(format(n)) == n` for decimal naturals.
- `int_div_rem` — `q·d + r == n` and `|r| < |d|` for truncated division.
- `rational_reduce` — constructed rationals are in lowest terms and preserve
  value under a round trip.

CI runs these weekly and on manual dispatch (see `.github/workflows/fuzz.yml`).
Crashing inputs are uploaded as artifacts; add them to `corpus/<target>/` as
regression fixtures.
