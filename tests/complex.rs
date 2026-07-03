//! Tests for the generic complex type.
#![cfg(feature = "complex")]

use puremp::{Complex, Int};

fn c(re: i64, im: i64) -> Complex<Int> {
    Complex::new(Int::from(re), Int::from(im))
}

#[test]
fn gaussian_integers() {
    // (1+2i) + (3+4i) = 4+6i
    assert_eq!(&c(1, 2) + &c(3, 4), c(4, 6));
    assert_eq!(&c(1, 2) - &c(3, 4), c(-2, -2));
    // (1+2i)(3+4i) = 3+4i+6i+8i² = -5+10i
    assert_eq!(&c(1, 2) * &c(3, 4), c(-5, 10));
    assert_eq!((-c(1, 2)), c(-1, -2));
    assert_eq!(c(3, -4).conj(), c(3, 4));
    // norm(3+4i) = 25
    assert_eq!(c(3, 4).norm_sqr(), Int::from(25));
    assert!(c(0, 0).is_zero());
    assert!(c(5, 0).is_real());
    // i² = -1
    let i = Complex::imaginary(Int::ONE);
    assert_eq!(&i * &i, c(-1, 0));

    let mut acc = c(2, 3);
    acc *= c(2, 3);
    assert_eq!(acc, c(-5, 12)); // (2+3i)² = 4+12i-9
    assert_eq!(c(2, 3).to_string(), "2 + 3i");
}

#[cfg(feature = "rational")]
#[test]
fn complex_rational_division() {
    use puremp::Rational;
    let cr = |re: i64, im: i64| Complex::new(Rational::from(re), Rational::from(im));
    // (1+i)/(1-i) = i
    let q = cr(1, 1).div(&cr(1, -1));
    assert_eq!(q.re.to_string(), "0");
    assert_eq!(q.im.to_string(), "1");
    // (5+0i)/(1+2i) = (5(1-2i))/5 = 1-2i
    let q2 = &cr(5, 0) / &cr(1, 2);
    assert_eq!(q2, cr(1, -2));
}
