//! C ABI for `puremp` (the `ffi` feature).
//!
//! This is the only module permitted to use `unsafe` (the crate sets
//! `unsafe_code = "deny"`, not `forbid`, for exactly this purpose). It exposes
//! `extern "C"` entry points over opaque handles, declared in
//! `include/puremp.h`.
//!
//! ## Conventions
//! - Arbitrary-precision integers are held behind the opaque [`PurempInt`]
//!   handle. Every constructor (`puremp_int_from_*`, the arithmetic operators)
//!   returns a freshly allocated handle that the caller must release with
//!   [`puremp_int_free`]; passing `NULL` where a handle is expected yields a
//!   `NULL` (or sentinel) result rather than undefined behaviour.
//! - Fallible calls return `NULL` (for pointer results) or a documented sentinel.
//! - Strings returned by the library are heap-allocated C strings that must be
//!   released with [`puremp_string_free`].
//! - Every entry point catches panics at the boundary, so a Rust panic surfaces
//!   as a `NULL`/sentinel result rather than unwinding into C.
//!
//! Build a C library with, e.g.:
//! `cargo rustc --lib --release --features ffi --crate-type staticlib`
//! (or `--crate-type cdylib`).
#![allow(unsafe_code)]
#![allow(unreachable_pub)]

use core::ffi::{c_char, c_int};
use core::ptr;
use core::str::FromStr;
use std::panic::{AssertUnwindSafe, catch_unwind};

use alloc::boxed::Box;
use alloc::string::ToString;
use std::ffi::{CStr, CString};

use crate::int::{Int, Sign};
use crate::rational::Rational;

/// Opaque handle wrapping an arbitrary-precision signed integer.
pub struct PurempInt(Int);

/// Opaque handle wrapping an exact rational number.
pub struct PurempRat(Rational);

#[inline]
fn to_handle(i: Int) -> *mut PurempInt {
    Box::into_raw(Box::new(PurempInt(i)))
}

/// Returns the library version as a static NUL-terminated C string.
#[unsafe(no_mangle)]
pub extern "C" fn puremp_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Creates an integer from a signed 64-bit value.
#[unsafe(no_mangle)]
pub extern "C" fn puremp_int_from_i64(v: i64) -> *mut PurempInt {
    to_handle(Int::from_i64(v))
}

/// Parses a decimal integer (optional leading `+`/`-`) from a NUL-terminated C
/// string. Returns `NULL` on a null pointer, invalid UTF-8, or a parse error.
///
/// # Safety
/// `s` must be `NULL` or a valid pointer to a NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_from_str(s: *const c_char) -> *mut PurempInt {
    if s.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let cs = unsafe { CStr::from_ptr(s) };
        let text = cs.to_str().ok()?;
        Int::from_str(text).ok()
    }));
    match r {
        Ok(Some(i)) => to_handle(i),
        _ => ptr::null_mut(),
    }
}

/// Frees an integer handle. A `NULL` argument is ignored.
///
/// # Safety
/// `h` must be `NULL` or a handle returned by this library and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_free(h: *mut PurempInt) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// Applies a binary operation to two handles, returning a fresh handle or `NULL`.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
unsafe fn binop<F>(a: *const PurempInt, b: *const PurempInt, f: F) -> *mut PurempInt
where
    F: Fn(&Int, &Int) -> Int,
{
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { &(*a).0 };
        let y = unsafe { &(*b).0 };
        to_handle(f(x, y))
    }));
    r.unwrap_or(ptr::null_mut())
}

/// Returns `a + b` as a fresh handle, or `NULL` on a null argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_add(
    a: *const PurempInt,
    b: *const PurempInt,
) -> *mut PurempInt {
    unsafe { binop(a, b, |x, y| x.add(y)) }
}

/// Returns `a - b` as a fresh handle, or `NULL` on a null argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_sub(
    a: *const PurempInt,
    b: *const PurempInt,
) -> *mut PurempInt {
    unsafe { binop(a, b, |x, y| x.sub(y)) }
}

/// Returns `a · b` as a fresh handle, or `NULL` on a null argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_mul(
    a: *const PurempInt,
    b: *const PurempInt,
) -> *mut PurempInt {
    unsafe { binop(a, b, |x, y| x.mul(y)) }
}

/// Returns `base^exp` as a fresh handle, or `NULL` on a null argument.
///
/// # Safety
/// `base` must be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_pow(base: *const PurempInt, exp: u64) -> *mut PurempInt {
    if base.is_null() {
        return ptr::null_mut();
    }
    let Ok(exp) = u32::try_from(exp) else {
        return ptr::null_mut();
    };
    let r = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { &(*base).0 };
        to_handle(x.pow(exp))
    }));
    r.unwrap_or(ptr::null_mut())
}

/// Returns `-a` as a fresh handle, or `NULL` on a null argument.
///
/// # Safety
/// `a` must be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_neg(a: *const PurempInt) -> *mut PurempInt {
    if a.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { &(*a).0 };
        to_handle(x.neg())
    }));
    r.unwrap_or(ptr::null_mut())
}

/// Compares two integers, returning `-1`, `0`, or `1`. Returns `-2` if either
/// argument is `NULL`.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_cmp(a: *const PurempInt, b: *const PurempInt) -> c_int {
    if a.is_null() || b.is_null() {
        return -2;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { &(*a).0 };
        let y = unsafe { &(*b).0 };
        match x.cmp(y) {
            core::cmp::Ordering::Less => -1,
            core::cmp::Ordering::Equal => 0,
            core::cmp::Ordering::Greater => 1,
        }
    }));
    r.unwrap_or(-2)
}

/// Returns the sign of `a` as `-1`, `0`, or `1`. Returns `-2` if `a` is `NULL`.
///
/// # Safety
/// `a` must be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_sign(a: *const PurempInt) -> c_int {
    if a.is_null() {
        return -2;
    }
    let r = catch_unwind(AssertUnwindSafe(|| match unsafe { &(*a).0 }.sign() {
        Sign::Negative => -1,
        Sign::Zero => 0,
        Sign::Positive => 1,
    }));
    r.unwrap_or(-2)
}

/// Formats `a` as a decimal string. The result is a heap-allocated C string the
/// caller must release with [`puremp_string_free`]. Returns `NULL` on a null
/// argument or allocation failure.
///
/// # Safety
/// `a` must be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_int_to_string(a: *const PurempInt) -> *mut c_char {
    if a.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { &(*a).0 };
        CString::new(x.to_string()).ok()
    }));
    match r {
        Ok(Some(c)) => c.into_raw(),
        _ => ptr::null_mut(),
    }
}

/// Frees a string returned by this library. A `NULL` argument is ignored.
///
/// # Safety
/// `s` must be `NULL` or a string returned by this library and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

// --- Rational C ABI ---

#[inline]
fn rat_handle(r: Rational) -> *mut PurempRat {
    Box::into_raw(Box::new(PurempRat(r)))
}

/// Builds a rational `num/den` from two integer handles, reduced to lowest
/// terms. Returns `NULL` on a null argument or a zero denominator.
///
/// # Safety
/// `num` and `den` must each be `NULL` or a valid live [`PurempInt`] handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_rat_new(
    num: *const PurempInt,
    den: *const PurempInt,
) -> *mut PurempRat {
    if num.is_null() || den.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let n = unsafe { &(*num).0 };
        let d = unsafe { &(*den).0 };
        Rational::checked_new(n.clone(), d.clone()).map(rat_handle)
    }));
    r.ok().flatten().unwrap_or(ptr::null_mut())
}

/// Parses a rational from a NUL-terminated C string (`"3"`, `"-3/4"`, `"1.5"`).
/// Returns `NULL` on a null pointer, invalid UTF-8, or a parse error.
///
/// # Safety
/// `s` must be `NULL` or a valid pointer to a NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_rat_from_str(s: *const c_char) -> *mut PurempRat {
    if s.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let text = unsafe { CStr::from_ptr(s) }.to_str().ok()?;
        text.parse::<Rational>().ok()
    }));
    match r {
        Ok(Some(rat)) => rat_handle(rat),
        _ => ptr::null_mut(),
    }
}

/// Frees a rational handle. A `NULL` argument is ignored.
///
/// # Safety
/// `h` must be `NULL` or a handle returned by this library and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_rat_free(h: *mut PurempRat) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// Applies a binary operation to two rational handles.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live [`PurempRat`] handle.
unsafe fn rat_binop<F>(a: *const PurempRat, b: *const PurempRat, f: F) -> *mut PurempRat
where
    F: Fn(&Rational, &Rational) -> Option<Rational>,
{
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { &(*a).0 };
        let y = unsafe { &(*b).0 };
        f(x, y).map(rat_handle)
    }));
    r.ok().flatten().unwrap_or(ptr::null_mut())
}

/// Returns `a + b`, or `NULL` on a null argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_rat_add(
    a: *const PurempRat,
    b: *const PurempRat,
) -> *mut PurempRat {
    unsafe { rat_binop(a, b, |x, y| Some(x.add(y))) }
}

/// Returns `a - b`, or `NULL` on a null argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_rat_sub(
    a: *const PurempRat,
    b: *const PurempRat,
) -> *mut PurempRat {
    unsafe { rat_binop(a, b, |x, y| Some(x.sub(y))) }
}

/// Returns `a · b`, or `NULL` on a null argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_rat_mul(
    a: *const PurempRat,
    b: *const PurempRat,
) -> *mut PurempRat {
    unsafe { rat_binop(a, b, |x, y| Some(x.mul(y))) }
}

/// Returns `a / b`, or `NULL` on a null argument or division by zero.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_rat_div(
    a: *const PurempRat,
    b: *const PurempRat,
) -> *mut PurempRat {
    unsafe { rat_binop(a, b, |x, y| if y.is_zero() { None } else { Some(x.div(y)) }) }
}

/// Compares two rationals, returning `-1`, `0`, or `1`. Returns `-2` if either
/// argument is `NULL`.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_rat_cmp(a: *const PurempRat, b: *const PurempRat) -> c_int {
    if a.is_null() || b.is_null() {
        return -2;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        match unsafe { &(*a).0 }.cmp(unsafe { &(*b).0 }) {
            core::cmp::Ordering::Less => -1,
            core::cmp::Ordering::Equal => 0,
            core::cmp::Ordering::Greater => 1,
        }
    }));
    r.unwrap_or(-2)
}

/// Formats `r` as `"n"` or `"n/d"`. The result is a heap-allocated C string the
/// caller must release with [`puremp_string_free`]. Returns `NULL` on a null
/// argument or allocation failure.
///
/// # Safety
/// `r` must be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_rat_to_string(r: *const PurempRat) -> *mut c_char {
    if r.is_null() {
        return ptr::null_mut();
    }
    let out = catch_unwind(AssertUnwindSafe(|| {
        CString::new(unsafe { &(*r).0 }.to_string()).ok()
    }));
    match out {
        Ok(Some(c)) => c.into_raw(),
        _ => ptr::null_mut(),
    }
}

// --- Float C ABI ---

use crate::float::{Float, RoundingMode};

/// Opaque handle wrapping an arbitrary-precision float.
pub struct PurempFloat(Float);

#[inline]
fn flt_handle(f: Float) -> *mut PurempFloat {
    Box::into_raw(Box::new(PurempFloat(f)))
}

/// Maps a C rounding code to a [`RoundingMode`] (0=nearest, 1=toward-zero,
/// 2=toward-+∞, 3=toward-−∞, 4=away-from-zero; anything else = nearest).
fn rounding_from_c(mode: c_int) -> RoundingMode {
    match mode {
        1 => RoundingMode::TowardZero,
        2 => RoundingMode::TowardPositive,
        3 => RoundingMode::TowardNegative,
        4 => RoundingMode::AwayFromZero,
        _ => RoundingMode::Nearest,
    }
}

/// Creates a float from a C `double` at `precision` bits.
#[unsafe(no_mangle)]
pub extern "C" fn puremp_float_from_double(x: f64, precision: u64) -> *mut PurempFloat {
    flt_handle(Float::from_f64(x, precision, RoundingMode::Nearest))
}

/// Creates a float from an integer handle, rounded to `precision` bits.
///
/// # Safety
/// `n` must be `NULL` or a valid live [`PurempInt`] handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_float_from_int(
    n: *const PurempInt,
    precision: u64,
    rounding: c_int,
) -> *mut PurempFloat {
    if n.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { &(*n).0 };
        flt_handle(Float::from_int(x, precision, rounding_from_c(rounding)))
    }));
    r.unwrap_or(ptr::null_mut())
}

/// Returns π rounded to `precision` bits.
#[unsafe(no_mangle)]
pub extern "C" fn puremp_float_pi(precision: u64, rounding: c_int) -> *mut PurempFloat {
    flt_handle(Float::pi(precision, rounding_from_c(rounding)))
}

/// Frees a float handle. A `NULL` argument is ignored.
///
/// # Safety
/// `h` must be `NULL` or a handle returned by this library and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_float_free(h: *mut PurempFloat) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// # Safety
/// `a` and `b` must each be `NULL` or a valid live [`PurempFloat`] handle.
unsafe fn flt_binop<F>(
    a: *const PurempFloat,
    b: *const PurempFloat,
    precision: u64,
    rounding: c_int,
    f: F,
) -> *mut PurempFloat
where
    F: Fn(&Float, &Float, u64, RoundingMode) -> Float,
{
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { &(*a).0 };
        let y = unsafe { &(*b).0 };
        flt_handle(f(x, y, precision, rounding_from_c(rounding)))
    }));
    r.unwrap_or(ptr::null_mut())
}

/// Returns `a + b` at `precision` bits. `NULL` on a null argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_float_add(
    a: *const PurempFloat,
    b: *const PurempFloat,
    precision: u64,
    rounding: c_int,
) -> *mut PurempFloat {
    unsafe { flt_binop(a, b, precision, rounding, Float::add) }
}

/// Returns `a - b`. `NULL` on a null argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_float_sub(
    a: *const PurempFloat,
    b: *const PurempFloat,
    precision: u64,
    rounding: c_int,
) -> *mut PurempFloat {
    unsafe { flt_binop(a, b, precision, rounding, Float::sub) }
}

/// Returns `a · b`. `NULL` on a null argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_float_mul(
    a: *const PurempFloat,
    b: *const PurempFloat,
    precision: u64,
    rounding: c_int,
) -> *mut PurempFloat {
    unsafe { flt_binop(a, b, precision, rounding, Float::mul) }
}

/// Returns `a / b` (`x/0` is signed infinity, `0/0` is NaN). `NULL` on a null
/// argument.
///
/// # Safety
/// `a` and `b` must each be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_float_div(
    a: *const PurempFloat,
    b: *const PurempFloat,
    precision: u64,
    rounding: c_int,
) -> *mut PurempFloat {
    unsafe { flt_binop(a, b, precision, rounding, Float::div) }
}

/// Returns `√a`. `NULL` on a null argument.
///
/// # Safety
/// `a` must be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_float_sqrt(
    a: *const PurempFloat,
    precision: u64,
    rounding: c_int,
) -> *mut PurempFloat {
    if a.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let x = unsafe { &(*a).0 };
        flt_handle(x.sqrt(precision, rounding_from_c(rounding)))
    }));
    r.unwrap_or(ptr::null_mut())
}

/// Returns the value as a C `double`. Returns NaN if `a` is `NULL`.
///
/// # Safety
/// `a` must be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_float_to_double(a: *const PurempFloat) -> f64 {
    if a.is_null() {
        return f64::NAN;
    }
    catch_unwind(AssertUnwindSafe(|| unsafe { &(*a).0 }.to_f64())).unwrap_or(f64::NAN)
}

/// Formats `a` as a fixed-point decimal string with `frac_digits` fractional
/// digits (caller frees with [`puremp_string_free`]). `NULL` on a null argument.
///
/// # Safety
/// `a` must be `NULL` or a valid live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn puremp_float_to_string(
    a: *const PurempFloat,
    frac_digits: u32,
) -> *mut c_char {
    if a.is_null() {
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        CString::new(unsafe { &(*a).0 }.to_decimal_string(frac_digits)).ok()
    }));
    match r {
        Ok(Some(c)) => c.into_raw(),
        _ => ptr::null_mut(),
    }
}
