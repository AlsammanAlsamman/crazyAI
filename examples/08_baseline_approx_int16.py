"""08 - the fair baseline for a byte/bit-level invent pilot.

Ports `approx_int16_madd` from the sister `matrixmultiply` repo's C lab
(cmm/src/k_approx.c) into crazyai's own kernel_bench contract, verbatim
except the function name and an inlined IDX macro, and measures it on
*this* machine with the same sizes/budget `Invent.step_measure` uses for
every matmul artifact - so it's a genuine baseline for whatever a
`--assumption 4` invent batch finds, not a number quoted from the
original lab's machine.

Two of that lab's three low-precision "crazy" kernels are deliberately
NOT ported here: `crazy_kronecker_int` assumes integer entries 0-15
(kernel_bench's random doubles in [0,1) would truncate to 0 everywhere)
and `crazy_bitpack_binary` assumes 0/1 matrices (every entry of a random
double is nonzero, so it would compute a constant). Both solve a
different, easier problem than the one this harness actually tests -
comparing them here would be a category error, not a fair baseline.
"""

import json
import os

from crazyai.toolkit.registry import build_toolkit

APPROX_INT16_MADD = r"""
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#define IDX(i, j) ((size_t)(i) * n + (j))
#define FXS 512
void kernel(int n, const double *A, const double *B, double *C) {
    int W = (n + 15) / 16 * 16;
    int16_t *qa = calloc((size_t)n * W, sizeof(int16_t)), *qbt = calloc((size_t)n * W, sizeof(int16_t));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            qa[(size_t)i * W + k]  = (int16_t)lrint(A[IDX(i, k)] * FXS);
            qbt[(size_t)k * W + i] = (int16_t)lrint(B[IDX(i, k)] * FXS);
        }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            __m256i acc = _mm256_setzero_si256();
            const int16_t *a = qa + (size_t)i * W, *b = qbt + (size_t)j * W;
            for (int k = 0; k < W; k += 16)
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(
                    _mm256_loadu_si256((const __m256i *)(a + k)), _mm256_loadu_si256((const __m256i *)(b + k))));
            int32_t lanes[8]; _mm256_storeu_si256((__m256i *)lanes, acc);
            int64_t s = 0; for (int t = 0; t < 8; t++) s += lanes[t];
            C[IDX(i, j)] = (double)s / ((double)FXS * FXS);
        }
    free(qa); free(qbt);
}
"""

if __name__ == "__main__":
    print(f"cpu_count={os.cpu_count()}")
    tk = build_toolkit(0)
    # same sizes/budget Invent.step_measure uses for every matmul artifact - what makes this comparable
    r = tk.call("kernel_bench", {"source": APPROX_INT16_MADD, "sizes": [64, 256, 512], "budget": 0.3})
    print(json.dumps(r, indent=2))
