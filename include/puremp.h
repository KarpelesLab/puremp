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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PUREMP_H */
