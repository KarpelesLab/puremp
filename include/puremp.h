/*
 * puremp.h — C ABI for the `puremp` arbitrary-precision arithmetic library.
 *
 * These declarations mirror the `#[unsafe(no_mangle)] extern "C"` entry points
 * in `src/ffi.rs`, built when the crate is compiled with the `ffi` feature:
 *
 *     cargo rustc --lib --release --features ffi --crate-type staticlib
 *     cargo rustc --lib --release --features ffi --crate-type cdylib
 *
 * Conventions:
 *   - `PurempInt` is an opaque, heap-allocated arbitrary-precision signed
 *     integer. Every constructor returns a handle the caller must release with
 *     `puremp_int_free`.
 *   - Fallible calls return NULL (pointer results) or a documented sentinel.
 *   - Strings returned by the library must be released with
 *     `puremp_string_free`.
 *   - Passing NULL where a handle is expected yields NULL / a sentinel rather
 *     than crashing; every call catches Rust panics at the boundary.
 *
 * This library is MIT-licensed. See LICENSE.
 */
#ifndef PUREMP_H
#define PUREMP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque arbitrary-precision signed integer. */
typedef struct PurempInt PurempInt;

/* Library version as a static NUL-terminated string (do not free). */
const char *puremp_version(void);

/* --- construction --- */

/* From a signed 64-bit value. */
PurempInt *puremp_int_from_i64(int64_t v);

/* Parse a decimal integer (optional leading '+'/'-'). NULL on error. */
PurempInt *puremp_int_from_str(const char *s);

/* Release an integer handle. NULL is ignored. */
void puremp_int_free(PurempInt *h);

/* --- arithmetic (each returns a fresh handle, NULL on a NULL argument) --- */

PurempInt *puremp_int_add(const PurempInt *a, const PurempInt *b);
PurempInt *puremp_int_sub(const PurempInt *a, const PurempInt *b);
PurempInt *puremp_int_mul(const PurempInt *a, const PurempInt *b);
PurempInt *puremp_int_pow(const PurempInt *base, uint64_t exp);
PurempInt *puremp_int_neg(const PurempInt *a);

/* --- inspection --- */

/* Compare: -1, 0, 1; returns -2 if either argument is NULL. */
int puremp_int_cmp(const PurempInt *a, const PurempInt *b);

/* Sign: -1, 0, 1; returns -2 if the argument is NULL. */
int puremp_int_sign(const PurempInt *a);

/* Decimal string (caller frees with puremp_string_free). NULL on error. */
char *puremp_int_to_string(const PurempInt *a);

/* Release a string returned by the library. NULL is ignored. */
void puremp_string_free(char *s);

/* --- Rational: opaque exact fraction (always in lowest terms) --- */

typedef struct PurempRat PurempRat;

/* Build num/den from two integer handles (reduced). NULL if den is zero. */
PurempRat *puremp_rat_new(const PurempInt *num, const PurempInt *den);

/* Parse "3", "-3/4", or a decimal like "1.5". NULL on error. */
PurempRat *puremp_rat_from_str(const char *s);

/* Release a rational handle. NULL is ignored. */
void puremp_rat_free(PurempRat *h);

/* Arithmetic (each returns a fresh handle, NULL on a NULL argument). */
PurempRat *puremp_rat_add(const PurempRat *a, const PurempRat *b);
PurempRat *puremp_rat_sub(const PurempRat *a, const PurempRat *b);
PurempRat *puremp_rat_mul(const PurempRat *a, const PurempRat *b);
PurempRat *puremp_rat_div(const PurempRat *a, const PurempRat *b); /* NULL if b==0 */

/* Compare: -1, 0, 1; returns -2 if either argument is NULL. */
int puremp_rat_cmp(const PurempRat *a, const PurempRat *b);

/* "n" or "n/d" string (caller frees with puremp_string_free). NULL on error. */
char *puremp_rat_to_string(const PurempRat *r);

/* --- Float: opaque arbitrary-precision binary float --- */

typedef struct PurempFloat PurempFloat;

/*
 * Rounding modes for the float operations:
 *   0 = to nearest (ties to even), 1 = toward zero, 2 = toward +inf,
 *   3 = toward -inf, 4 = away from zero. Other values are treated as nearest.
 */

/* Construct from a C double / from an integer handle (rounded to `precision`). */
PurempFloat *puremp_float_from_double(double x, uint64_t precision);
PurempFloat *puremp_float_from_int(const PurempInt *n, uint64_t precision, int rounding);

/* Pi rounded to `precision` bits. */
PurempFloat *puremp_float_pi(uint64_t precision, int rounding);

/* Release a float handle. NULL is ignored. */
void puremp_float_free(PurempFloat *h);

/* Arithmetic at `precision` bits (each returns a fresh handle, NULL on NULL). */
PurempFloat *puremp_float_add(const PurempFloat *a, const PurempFloat *b, uint64_t precision, int rounding);
PurempFloat *puremp_float_sub(const PurempFloat *a, const PurempFloat *b, uint64_t precision, int rounding);
PurempFloat *puremp_float_mul(const PurempFloat *a, const PurempFloat *b, uint64_t precision, int rounding);
PurempFloat *puremp_float_div(const PurempFloat *a, const PurempFloat *b, uint64_t precision, int rounding);
PurempFloat *puremp_float_sqrt(const PurempFloat *a, uint64_t precision, int rounding);

/* Convert to a C double (NaN if the handle is NULL). */
double puremp_float_to_double(const PurempFloat *a);

/* Fixed-point decimal string (caller frees with puremp_string_free). */
char *puremp_float_to_string(const PurempFloat *a, uint32_t frac_digits);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PUREMP_H */
