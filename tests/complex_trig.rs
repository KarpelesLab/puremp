#![cfg(all(feature = "complex", feature = "float"))]
//! Complex<Float> trigonometric / hyperbolic transcendentals and their inverses,
//! checked against Pythagorean identities, defining formulas, real-axis values,
//! and principal-branch round-trips (tolerance ~1e-12).

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
/// `a` and `b` agree to ~1e-12 in both components.
fn ceq(a: &Complex<Float>, b: &Complex<Float>) -> bool {
    cclose(a, b.re.to_f64(), b.im.to_f64())
}
fn one() -> Complex<Float> {
    cx(1.0, 0.0)
}

// --- tan / cot -------------------------------------------------------------

#[test]
fn tan_is_sin_over_cos() {
    let z = cx(0.7, 0.3);
    // tan = sin/cos
    assert!(ceq(&z.tan(), &z.sin().div(&z.cos())));
    // cot = cos/sin, and tan·cot = 1
    assert!(ceq(&z.cot(), &z.cos().div(&z.sin())));
    let prod = z.tan().mul(&z.cot());
    assert!(cclose(&prod, 1.0, 0.0));
    // tan of a real matches Float::tan-ish via sin/cos of real
    let r = cx(0.4, 0.0).tan();
    assert!(close(&r.re, 0.4f64.tan()));
    assert!(close(&r.im, 0.0));
}

// --- sinh / cosh / tanh ----------------------------------------------------

#[test]
fn hyperbolic_identities() {
    let z = cx(0.6, 0.8);
    let (s, c) = (z.sinh(), z.cosh());
    // cosh² − sinh² = 1
    let d = c.mul(&c).sub(&s.mul(&s));
    assert!(cclose(&d, 1.0, 0.0), "{:?}", (d.re.to_f64(), d.im.to_f64()));
    // tanh = sinh/cosh
    assert!(ceq(&z.tanh(), &s.div(&c)));
    // real axis: tanh(a+0i) = Float tanh(a)
    let t = cx(0.9, 0.0).tanh();
    assert!(close(&t.re, 0.9f64.tanh()));
    assert!(close(&t.im, 0.0));
    // sinh(iy) = i·sin(y), cosh(iy) = cos(y)
    let y = 0.5;
    assert!(cclose(&cx(0.0, y).sinh(), 0.0, y.sin()));
    assert!(cclose(&cx(0.0, y).cosh(), y.cos(), 0.0));
}

// --- inverse round-trips on the principal domain ---------------------------

#[test]
fn inverse_trig_round_trips() {
    let z = cx(0.3, 0.4);
    assert!(ceq(&z.asin().sin(), &z));
    assert!(ceq(&z.acos().cos(), &z));
    assert!(ceq(&z.atan().tan(), &z));
}

#[test]
fn inverse_hyperbolic_round_trips() {
    let z = cx(0.3, 0.4);
    assert!(ceq(&z.asinh().sinh(), &z));
    assert!(ceq(&z.atanh().tanh(), &z));
    // acosh principal range has Re ≥ 0; pick z with positive real part
    let w = cx(1.7, 0.5);
    assert!(ceq(&w.acosh().cosh(), &w));
}

// --- cross-checks against known values -------------------------------------

#[test]
fn known_values() {
    // atan(1+0i) = π/4
    assert!(close(&one().atan().re, core::f64::consts::FRAC_PI_4));
    assert!(close(&one().atan().im, 0.0));

    // asin(2) = π/2 − i·acosh(2) = π/2 − i·ln(2+√3)
    let asin2 = cx(2.0, 0.0).asin();
    let acosh2 = (2.0 + 3f64.sqrt()).ln();
    assert!(cclose(&asin2, core::f64::consts::FRAC_PI_2, -acosh2));

    // acosh(2) is real = ln(2+√3)
    let ac = cx(2.0, 0.0).acosh();
    assert!(cclose(&ac, acosh2, 0.0));

    // atanh of a real matches Float: atanh(0.5) = ½ln 3
    let at = cx(0.5, 0.0).atanh();
    assert!(close(&at.re, 0.5 * 3f64.ln()));
    assert!(close(&at.im, 0.0));

    // atan(0.5i) = 0 + i·atanh(0.5)  (from atan(iy) = i·atanh(y))
    let ai = cx(0.0, 0.5).atan();
    assert!(cclose(&ai, 0.0, 0.5 * 3f64.ln()));
}

// --- consistency identities -------------------------------------------------

#[test]
fn asin_plus_acos_is_half_pi() {
    let z = cx(0.35, -0.2);
    let s = z.asin().add(&z.acos());
    assert!(cclose(&s, core::f64::consts::FRAC_PI_2, 0.0));
}

#[test]
fn atan_log_identity() {
    // atan z = ½i·(ln(i+z) − ln(i−z))  (Re halved by the i factor)
    let z = cx(0.4, 0.25);
    let ipz = Complex {
        re: z.re.clone(),
        im: Float::add(&z.im, &f(1.0), P, M),
    }; // i + z
    let imz = Complex {
        re: Float::neg(&z.re),
        im: Float::sub(&f(1.0), &z.im, P, M),
    }; // i − z
    let diff = ipz.ln().sub(&imz.ln());
    // ½i·diff = (−diff.im/2, diff.re/2)
    let expect = Complex {
        re: Float::div(&Float::neg(&diff.im), &f(2.0), P, M),
        im: Float::div(&diff.re, &f(2.0), P, M),
    };
    assert!(ceq(&z.atan(), &expect));
}
