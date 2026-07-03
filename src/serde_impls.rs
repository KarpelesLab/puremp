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

// ===========================================================================
// Extended numeric types.
// ===========================================================================

// --- string round-trip types (Display + FromStr) ---

#[cfg(feature = "dyadic")]
serde_via_string!(crate::dyadic::Dyadic);

#[cfg(feature = "decimal")]
serde_via_string!(crate::decimal::Decimal);

#[cfg(feature = "rational")]
serde_via_string!(crate::inf_rational::InfRational);

// --- composite types serialized as tuples of their components ---

#[cfg(feature = "int")]
impl Serialize for crate::mod_int::ModInt {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.to_int(), self.modulus()).serialize(s)
    }
}
#[cfg(feature = "int")]
impl<'de> Deserialize<'de> for crate::mod_int::ModInt {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (value, modulus) = <(Int, Int)>::deserialize(d)?;
        Ok(crate::mod_int::ModInt::new(value, modulus))
    }
}

#[cfg(feature = "algebraic")]
impl Serialize for crate::quadratic::Quadratic {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (
            self.rational_part(),
            self.surd_coefficient(),
            self.radicand(),
        )
            .serialize(s)
    }
}
#[cfg(feature = "algebraic")]
impl<'de> Deserialize<'de> for crate::quadratic::Quadratic {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use crate::rational::Rational;
        let (a, b, dd) = <(Rational, Rational, Int)>::deserialize(d)?;
        Ok(crate::quadratic::Quadratic::new(a, b, dd))
    }
}

#[cfg(feature = "algebraic")]
impl Serialize for crate::algebraic::Algebraic {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let (lo, hi) = self.interval();
        (self.defining_polynomial(), lo, hi).serialize(s)
    }
}
#[cfg(feature = "algebraic")]
impl<'de> Deserialize<'de> for crate::algebraic::Algebraic {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use crate::poly::Poly;
        use crate::rational::Rational;
        let (poly, lo, hi) = <(Poly<Rational>, Rational, Rational)>::deserialize(d)?;
        Ok(crate::algebraic::Algebraic::new(poly, lo, hi))
    }
}

#[cfg(feature = "interval")]
impl Serialize for crate::interval::Interval {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.lower(), self.upper(), self.precision()).serialize(s)
    }
}
#[cfg(feature = "interval")]
impl<'de> Deserialize<'de> for crate::interval::Interval {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use crate::float::Float;
        let (lo, hi, precision) = <(Float, Float, u64)>::deserialize(d)?;
        Ok(crate::interval::Interval::new(lo, hi, precision))
    }
}

// FixedFloat: the exact float plus a rounding-mode code.
#[cfg(feature = "float")]
fn mode_to_u8(m: crate::float::RoundingMode) -> u8 {
    use crate::float::RoundingMode::*;
    match m {
        Nearest => 0,
        TowardZero => 1,
        TowardPositive => 2,
        TowardNegative => 3,
        AwayFromZero => 4,
    }
}
#[cfg(feature = "float")]
fn mode_from_u8(v: u8) -> Option<crate::float::RoundingMode> {
    use crate::float::RoundingMode::*;
    Some(match v {
        0 => Nearest,
        1 => TowardZero,
        2 => TowardPositive,
        3 => TowardNegative,
        4 => AwayFromZero,
        _ => return None,
    })
}

#[cfg(feature = "float")]
impl Serialize for crate::fixed_float::FixedFloat {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.as_float(), mode_to_u8(self.rounding_mode())).serialize(s)
    }
}
#[cfg(feature = "float")]
impl<'de> Deserialize<'de> for crate::fixed_float::FixedFloat {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use crate::float::Float;
        let (value, code) = <(Float, u8)>::deserialize(d)?;
        let mode = mode_from_u8(code).ok_or_else(|| D::Error::custom("invalid rounding mode"))?;
        Ok(crate::fixed_float::FixedFloat::from_float(value, mode))
    }
}

// --- generic container types ---

#[cfg(feature = "complex")]
impl<T: Serialize> Serialize for crate::complex::Complex<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (&self.re, &self.im).serialize(s)
    }
}
#[cfg(feature = "complex")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for crate::complex::Complex<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (re, im) = <(T, T)>::deserialize(d)?;
        Ok(crate::complex::Complex::new(re, im))
    }
}

#[cfg(feature = "poly")]
impl<T: Serialize + Clone + Default + PartialEq> Serialize for crate::poly::Poly<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.coeffs().serialize(s)
    }
}
#[cfg(feature = "poly")]
impl<'de, T> Deserialize<'de> for crate::poly::Poly<T>
where
    T: Deserialize<'de> + Clone + Default + PartialEq,
{
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let coeffs = alloc::vec::Vec::<T>::deserialize(d)?;
        Ok(crate::poly::Poly::new(coeffs))
    }
}

#[cfg(feature = "matrix")]
impl<T: Serialize + Clone + Default> Serialize for crate::matrix::Matrix<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.rows(), self.cols(), self.as_slice()).serialize(s)
    }
}
#[cfg(feature = "matrix")]
impl<'de, T> Deserialize<'de> for crate::matrix::Matrix<T>
where
    T: Deserialize<'de> + Clone + Default,
{
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (rows, cols, data) = <(usize, usize, alloc::vec::Vec<T>)>::deserialize(d)?;
        Ok(crate::matrix::Matrix::new(rows, cols, data))
    }
}
