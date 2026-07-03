//! Tests for the num-traits bridge.
#![cfg(feature = "num-traits")]

use num_traits::{Num, One, Signed, ToPrimitive, Zero};
use puremp::{Int, Rational};

// A generic function usable only via the num-traits interface.
fn sum_of<T: Zero + Clone + core::ops::Add<Output = T>>(xs: &[T]) -> T {
    xs.iter().fold(T::zero(), |a, x| a + x.clone())
}

#[test]
fn generic_num_traits() {
    // Zero/One (trait-qualified so we exercise the impls, not inherent methods)
    assert!(<Int as Zero>::zero().is_zero());
    assert!(<Int as One>::one().is_one());
    assert_eq!(
        sum_of(&[Int::from(1), Int::from(2), Int::from(3)]),
        Int::from(6)
    );

    // Num::from_str_radix
    assert_eq!(
        <Int as Num>::from_str_radix("ff", 16).unwrap(),
        Int::from(255)
    );
    assert_eq!(
        <Rational as Num>::from_str_radix("-3/4", 10)
            .unwrap()
            .to_string(),
        "-3/4"
    );

    // Signed
    assert_eq!(Signed::abs(&Int::from(-7)), Int::from(7));
    assert_eq!(Signed::signum(&Int::from(-3)), Int::from(-1));
    assert!(Signed::is_negative(&Rational::from(-2i64)));

    // ToPrimitive (trait-qualified)
    assert_eq!(ToPrimitive::to_i64(&Int::from(42)), Some(42));
    assert_eq!(ToPrimitive::to_i64(&Int::from(2).pow(100)), None);

    // Operators required by Num now exist on Int (truncated div/rem).
    assert_eq!((Int::from(17) / Int::from(5)).to_string(), "3");
    assert_eq!((Int::from(-17) % Int::from(5)).to_string(), "-2");

    // Rational as a Num field
    let r = <Rational as Num>::from_str_radix("7/2", 10).unwrap();
    assert!((r % Rational::from(1i64)).to_string() == "1/2");
}

#[cfg(feature = "decimal")]
#[test]
fn decimal_num_traits() {
    use num_traits::{One, Zero};
    use puremp::Decimal;
    assert!(<Decimal as Zero>::zero().is_zero());
    assert!(<Decimal as One>::one().is_one());
    // usable in a generic Zero-based sum
    fn total<T: Zero + Clone + core::ops::Add<Output = T>>(xs: &[T]) -> T {
        xs.iter().fold(T::zero(), |a, x| a + x.clone())
    }
    let d = |s: &str| -> Decimal { s.parse().unwrap() };
    assert_eq!(total(&[d("0.1"), d("0.2"), d("0.3")]).to_string(), "0.6");
}

#[cfg(feature = "complex")]
#[test]
fn complex_num_traits() {
    use num_traits::{One, Zero};
    use puremp::{Complex, Int};
    let z = <Complex<Int> as Zero>::zero();
    assert!(z.is_zero());
    let o = <Complex<Int> as One>::one();
    assert_eq!(o, Complex::new(Int::ONE, Int::ZERO));
    // i * i = -1  built from the One/Zero identities
    let i = Complex::new(Int::ZERO, Int::ONE);
    assert_eq!(&i * &i, Complex::new(-Int::ONE, Int::ZERO));
}
