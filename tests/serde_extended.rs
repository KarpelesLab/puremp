//! serde round-trip tests for the extended numeric types.
#![cfg(feature = "serde")]

use puremp::{Int, Rational};

fn roundtrip<T>(v: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let json = serde_json::to_string(v).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[cfg(feature = "dyadic")]
#[test]
fn dyadic() {
    use puremp::Dyadic;
    let d: Dyadic = "0.375".parse().unwrap();
    assert_eq!(roundtrip(&d), d);
}

#[cfg(feature = "decimal")]
#[test]
fn decimal() {
    use puremp::Decimal;
    let d: Decimal = "1.50".parse().unwrap();
    let back: Decimal = roundtrip(&d);
    assert_eq!(back, d);
    assert_eq!(back.to_string(), "1.50"); // scale preserved
    assert_eq!(
        roundtrip(&"-2E-8".parse::<Decimal>().unwrap()).to_string(),
        "-0.00000002"
    );
}

#[cfg(feature = "rational")]
#[test]
fn inf_rational() {
    use puremp::InfRational;
    for s in ["inf", "-inf", "3/4", "-5"] {
        let v: InfRational = s.parse().unwrap();
        assert_eq!(roundtrip(&v), v);
    }
    // NaN survives structurally (NaN != NaN, so check display)
    let nan: InfRational = "nan".parse().unwrap();
    assert_eq!(roundtrip(&nan).to_string(), "NaN");
}

#[test]
fn mod_int() {
    use puremp::ModInt;
    let m = ModInt::new(Int::from(123), Int::from(1000));
    assert_eq!(roundtrip(&m), m);
}

#[cfg(feature = "complex")]
#[test]
fn complex() {
    use puremp::Complex;
    let c = Complex::new(Int::from(3), Int::from(-4));
    assert_eq!(roundtrip(&c), c);
    let cr = Complex::new(Rational::new(1.into(), 2.into()), Rational::from(3));
    assert_eq!(roundtrip(&cr), cr);
}

#[cfg(feature = "poly")]
#[test]
fn poly() {
    use puremp::Poly;
    let p: Poly<Rational> = Poly::new(vec![
        Rational::from(1),
        Rational::from(-2),
        Rational::from(1),
    ]);
    assert_eq!(roundtrip(&p), p);
    let pi: Poly<Int> = Poly::new(vec![Int::from(5), Int::from(0), Int::from(7)]);
    assert_eq!(roundtrip(&pi), pi);
}

#[cfg(feature = "matrix")]
#[test]
fn matrix() {
    use puremp::Matrix;
    let m = Matrix::new(
        2,
        2,
        vec![Int::from(1), Int::from(2), Int::from(3), Int::from(4)],
    );
    assert_eq!(roundtrip(&m), m);
}

#[cfg(feature = "algebraic")]
#[test]
fn algebraic_types() {
    use puremp::{Algebraic, Poly, Quadratic};
    let q = Quadratic::sqrt(Int::from(2)).add(&Quadratic::from(Int::ONE)); // 1 + √2
    assert_eq!(roundtrip(&q), q);

    // √2 as an Algebraic
    let a = Algebraic::new(
        Poly::new(vec![
            Rational::from(-2),
            Rational::from(0),
            Rational::from(1),
        ]),
        Rational::from(0),
        Rational::from(2),
    );
    assert_eq!(roundtrip(&a), a);
}

#[cfg(feature = "float")]
#[test]
fn fixed_float_and_interval() {
    use puremp::{FixedFloat, Float, Interval, RoundingMode};
    let n = RoundingMode::Nearest;
    let f = FixedFloat::from_f64(1.5, 64, RoundingMode::TowardPositive);
    let back: FixedFloat = roundtrip(&f);
    assert_eq!(back, f);
    assert_eq!(back.rounding_mode(), RoundingMode::TowardPositive);

    #[cfg(feature = "interval")]
    {
        let iv = Interval::new(Float::from_f64(1.0, 53, n), Float::from_f64(2.0, 53, n), 53);
        let bk: Interval = roundtrip(&iv);
        assert_eq!(bk.lower().to_f64(), 1.0);
        assert_eq!(bk.upper().to_f64(), 2.0);
        assert_eq!(bk.precision(), 53);
    }
}
