#![cfg(all(feature = "complex", feature = "float"))]
//! Complex<Float> arithmetic + transcendentals against known identities.

use puremp::{Complex, Float, RoundingMode};

const P: u64 = 200;
const M: RoundingMode = RoundingMode::Nearest;

fn f(v: f64) -> Float {
    Float::from_f64(v, P, M)
}
fn cx(re: f64, im: f64) -> Complex<Float> {
    Complex {
        re: f(re),
        im: f(im),
    }
}
fn close(a: &Float, b: f64) -> bool {
    (a.to_f64() - b).abs() < 1e-12 * (1.0 + b.abs())
}
fn cclose(z: &Complex<Float>, re: f64, im: f64) -> bool {
    close(&z.re, re) && close(&z.im, im)
}

#[test]
fn arithmetic_and_operators() {
    // (1+2i)(3+i) = 1 + 7i
    let z = cx(1.0, 2.0).mul(&cx(3.0, 1.0));
    assert!(cclose(&z, 1.0, 7.0), "{:?}", (z.re.to_f64(), z.im.to_f64()));
    // Float operators compile & compute: (1.5 + 2.5) * 2 = 8
    let s = (f(1.5) + f(2.5)) * f(2.0);
    assert!(close(&s, 8.0));
}

#[test]
fn abs_and_arg() {
    assert!(close(&cx(3.0, 4.0).abs(), 5.0));
    assert!(close(&cx(0.0, 1.0).arg(), core::f64::consts::FRAC_PI_2));
    assert!(close(&cx(-1.0, 0.0).arg(), core::f64::consts::PI));
}

#[test]
fn exp_ln_sqrt_pow() {
    // exp(iπ) = -1
    let ipi = Complex {
        re: f(0.0),
        im: Float::pi(P, M),
    };
    assert!(
        cclose(&ipi.exp(), -1.0, 0.0),
        "{:?}",
        (ipi.exp().re.to_f64(), ipi.exp().im.to_f64())
    );
    // sqrt(-2) = i·√2
    assert!(cclose(&cx(-2.0, 0.0).sqrt(), 0.0, 2f64.sqrt()));
    // ln(exp(z)) = z for z=1+0.5i
    let z = cx(1.0, 0.5);
    assert!(cclose(&z.exp().ln(), 1.0, 0.5));
    // (1+i)^2 = 2i
    assert!(cclose(&cx(1.0, 1.0).pow(&cx(2.0, 0.0)), 0.0, 2.0));
}

#[test]
fn sin_cos() {
    // sin²+cos² = 1 for a complex argument (real part 1, imag 0)
    let z = cx(0.7, 0.3);
    let (s, c) = (z.sin(), z.cos());
    let one = s.mul(&s).add(&c.mul(&c));
    assert!(
        cclose(&one, 1.0, 0.0),
        "{:?}",
        (one.re.to_f64(), one.im.to_f64())
    );
}

#[test]
fn all_operator_combinations() {
    let a = cx(1.0, 2.0);
    let b = cx(3.0, 1.0);
    let expect = cx(4.0, 3.0); // a + b
    // all four owned/borrowed combinations of each operator
    assert!(cclose(&(a.clone() + b.clone()), 4.0, 3.0));
    assert!(cclose(&(a.clone() + &b), 4.0, 3.0));
    assert!(cclose(&(&a + b.clone()), 4.0, 3.0));
    assert!(cclose(&(&a + &b), 4.0, 3.0));
    assert!(cclose(&(&a - &b), -2.0, 1.0));
    assert!(cclose(&(a.clone() - &b), -2.0, 1.0));
    assert!(cclose(&(&a * &b), 1.0, 7.0)); // (1+2i)(3+i) = 1+7i
    assert!(cclose(&(&a / &b), 0.5, 0.5)); // (1+2i)/(3+i) = 0.5+0.5i
    assert!(cclose(&(-&a), -1.0, -2.0));
    assert!(cclose(&(-a.clone()), -1.0, -2.0));
    // assign forms, owned and borrowed rhs
    let mut c = a.clone();
    c += &b;
    assert!(cclose(&c, 4.0, 3.0));
    let mut d = a.clone();
    d *= b.clone();
    assert!(cclose(&d, 1.0, 7.0));
    let _ = expect;
}
