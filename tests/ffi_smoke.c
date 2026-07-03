/*
 * ffi_smoke.c — smoke test for the puremp C ABI.
 *
 * Built and run by the `c_abi` CI job against the static library:
 *
 *     cargo rustc --lib --release --features ffi --crate-type staticlib
 *     cc tests/ffi_smoke.c -I include target/release/libpuremp.a -lpthread -ldl -lm -o ffi_smoke
 *     ./ffi_smoke
 *
 * Exits non-zero on the first failed check.
 */
#include "puremp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check_str(const char *what, char *got, const char *want) {
    if (got == NULL) {
        fprintf(stderr, "FAIL %s: got NULL, want \"%s\"\n", what, want);
        failures++;
        return;
    }
    if (strcmp(got, want) != 0) {
        fprintf(stderr, "FAIL %s: got \"%s\", want \"%s\"\n", what, got, want);
        failures++;
    } else {
        printf("ok   %s = %s\n", what, got);
    }
    puremp_string_free(got);
}

int main(void) {
    printf("puremp version: %s\n", puremp_version());

    /* 2^100 via repeated FFI multiplication of a parsed base. */
    PurempInt *two = puremp_int_from_i64(2);
    PurempInt *acc = puremp_int_pow(two, 100);
    check_str("2^100", puremp_int_to_string(acc),
              "1267650600228229401496703205376");

    /* (2^64) * (2^64) == 2^128 */
    PurempInt *big = puremp_int_from_str("18446744073709551616");
    PurempInt *sq = puremp_int_mul(big, big);
    check_str("2^64 * 2^64", puremp_int_to_string(sq),
              "340282366920938463463374607431768211456");

    /* Signed subtraction: 3 - 5 == -2 */
    PurempInt *three = puremp_int_from_i64(3);
    PurempInt *five = puremp_int_from_i64(5);
    PurempInt *diff = puremp_int_sub(three, five);
    check_str("3 - 5", puremp_int_to_string(diff), "-2");

    /* Comparison and sign sentinels. */
    if (puremp_int_cmp(three, five) != -1) {
        fprintf(stderr, "FAIL cmp(3,5) != -1\n");
        failures++;
    }
    if (puremp_int_sign(diff) != -1) {
        fprintf(stderr, "FAIL sign(-2) != -1\n");
        failures++;
    }
    /* NULL handling must not crash. */
    if (puremp_int_cmp(NULL, five) != -2) {
        fprintf(stderr, "FAIL cmp(NULL,_) != -2\n");
        failures++;
    }
    if (puremp_int_from_str("not a number") != NULL) {
        fprintf(stderr, "FAIL parse of junk returned non-NULL\n");
        failures++;
    }

    puremp_int_free(two);
    puremp_int_free(acc);
    puremp_int_free(big);
    puremp_int_free(sq);
    puremp_int_free(three);
    puremp_int_free(five);
    puremp_int_free(diff);

    if (failures != 0) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all C ABI smoke checks passed\n");
    return 0;
}
