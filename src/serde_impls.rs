//! Hand-written `serde` support (the `serde` feature).
//!
//! No `serde_derive` dependency: each number type serializes as its exact string
//! form and deserializes by parsing it back, so the encoding is human-readable
//! and stable across formats (JSON, TOML, …).

use alloc::string::String;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::int::Int;
use crate::nat::Nat;

macro_rules! serde_via_string {
    ($ty:ty) => {
        impl Serialize for $ty {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.collect_str(self)
            }
        }
        impl<'de> Deserialize<'de> for $ty {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let s = String::deserialize(d)?;
                s.parse().map_err(D::Error::custom)
            }
        }
    };
}

serde_via_string!(Nat);
serde_via_string!(Int);

#[cfg(feature = "rational")]
serde_via_string!(crate::rational::Rational);

// Float uses its exact (lossless) string encoding rather than `Display`.
#[cfg(feature = "float")]
impl Serialize for crate::float::Float {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_exact_string())
    }
}

#[cfg(feature = "float")]
impl<'de> Deserialize<'de> for crate::float::Float {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        crate::float::Float::from_exact_string(&s).map_err(D::Error::custom)
    }
}
