#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------- the product-village: a board of pegs, set once ---------- */
static uint32_t *QTAB = NULL;
#define QTAB_N 131071u   /* max index = 65535+65535 */

static void build_qtab(void) {
    uint32_t *t = (uint32_t*)malloc((size_t)QTAB_N * sizeof(uint32_t));
    for (uint32_t s = 0; s < QTAB_N; s++) {
        uint64_t ss = (uint64_t)s * (uint64_t)s;
        t[s] = (uint32_t)(ss >> 2);
    }
    QTAB = t;
}

/* one tooth-pair meeting: exact 16x16->32 product by looking, not reckoning.
   a*b = floor((a+b)^2/4) - floor((a-b)^2/4)   (exact quarter-square identity) */
static inline uint32_t qmul16(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    uint32_t d = (a >= b) ? (a - b) : (b - a);
    return QTAB[s] - QTAB[d];
}

/* a comb of two teeth (hi,lo 16-bit limbs), assembled from four look-ups */
static inline uint64_t mul32(uint32_t a, uint32_t b) {
    uint32_t a1 = a >> 16, a0 = a & 0xFFFFu;
    uint32_t b1 = b >> 16, b0 = b & 0xFFFFu;
    uint32_t p11 = qmul16(a1, b1);
    uint32_t p10 = qmul16(a1, b0);
    uint32_t p01 = qmul16(a0, b1);
    uint32_t p00 = qmul16(a0, b0);
    uint64_t mid = (uint64_t)p10 + (uint64_t)p01;
    return ((uint64_t)p11 << 32) + (mid << 16) + (uint64_t)p00;
}

/* a double as a comb of pin-notches: sign + 32-bit mantissa-comb + exponent;
   the product is formed by looking up the combs' meeting, never by native '*' */
static inline double dmul(double x, double y) {
    if (x == 0.0 || y == 0.0) return 0.0;
    int ex, ey;
    double fx = frexp(x, &ex);
    double fy = frexp(y, &ey);
    double sign = ((fx < 0.0) != (fy < 0.0)) ? -1.0 : 1.0;
    uint32_t mx = (uint32_t)(fabs(fx) * 4294967296.0); /* comb-ify: in [2^31,2^32) */
    uint32_t my = (uint32_t)(fabs(fy) * 4294967296.0);
    uint64_t prod = mul32(mx, my);            /* look, don't reckon */
    return sign * ldexp((double)prod, ex + ey - 64);
}

#define BS 64  /* a block small enough to fit under one hand */

void kernel(int n, const double *A, const double *B, double *C) {
    if (!QTAB) {
        #pragma omp critical
        { if (!QTAB) build_qtab(); }
    }

    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < n; ib += BS) {
        int imax = ib + BS < n ? ib + BS : n;
        for (int jb = 0; jb < n; jb += BS) {
            int jmax = jb + BS < n ? jb + BS : n;
            /* row-block of A and column-block of B are wedged in place;
               the shared index k slides tooth by tooth beneath one peg */
            for (int i = ib; i < imax; i++) {
                for (int j = jb; j < jmax; j++) {
                    double peg = 0.0;                       /* the holding-peg */
                    for (int k = 0; k < n; k++) {
                        peg += dmul(A[(size_t)i * n + k], B[(size_t)k * n + j]);
                    }
                    C[(size_t)i * n + j] = peg;              /* onto the third fabric */
                }
            }
        }
    }
}
