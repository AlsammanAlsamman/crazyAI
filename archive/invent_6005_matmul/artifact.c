#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- The Table of Grafts: every digit-pair's product, carved once, looked up forever ---- */
static uint16_t GRAFT[256][256];
static int graft_ready = 0;

static void build_graft_table(void) {
    if (graft_ready) return;
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++)
            GRAFT[a][b] = (uint16_t)(a * b);      /* memorized match, built ahead of time */
    graft_ready = 1;
}

/* unweld one double into its 7 byte-limbs (a 56-bit fixed-point mantissa)
 * plus a base-2 exponent and sign -- "a quantity is a limb-chain, not one mark". */
static inline void unweld(double v, uint8_t limb[7], int *exp2, int *sgn) {
    if (v == 0.0) { memset(limb, 0, 7); *exp2 = 0; *sgn = 0; return; }
    *sgn = (v < 0.0) ? -1 : 1;
    int e;
    double m = frexp(fabs(v), &e);                /* m in [0.5,1) */
    double scaled = m * 72057594037927936.0;       /* m * 2^56 */
    uint64_t M = (uint64_t)(scaled + 0.5);
    if (M >> 56) { M >>= 1; e++; }                 /* guard the rare round-up to 2^56 */
    for (int p = 0; p < 7; p++) { limb[p] = (uint8_t)(M & 0xFF); M >>= 8; }
    *exp2 = e;
}

/* multiply two doubles by unwelding them into limbs, walking every limb-pair
 * to the Table of Grafts, and carrying the looked-up grafts onto a wax
 * accumulator-strip -- no hardware '*' ever touches x or y. */
static inline double graft_multiply(double x, double y) {
    uint8_t bx[7], by[7];
    int ex, ey, sx, sy;
    unweld(x, bx, &ex, &sx);
    unweld(y, by, &ey, &sy);
    if (sx == 0 || sy == 0) return 0.0;

    unsigned __int128 strip = 0;                   /* the wax accumulator-strip */
    for (int p = 0; p < 7; p++) {
        unsigned __int128 row = 0;
        for (int q = 0; q < 7; q++)
            row += (unsigned __int128)GRAFT[bx[p]][by[q]] << (8 * q);
        strip += row << (8 * p);                   /* overflow-limb carried sideways */
    }                                               /* strip is scraped bare on return */

    double mag = ldexp((double)strip, ex + ey - 112);
    return (sx * sy < 0) ? -mag : mag;
}

#ifndef PLOT
#define PLOT 48   /* the slab's reach: a dozen-ish rows married to a dozen-ish columns */
#endif

void kernel(int n, const double *A, const double *B, double *C) {
    build_graft_table();
    memset(C, 0, (size_t)n * n * sizeof(double));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < n; ii += PLOT) {
        for (int jj = 0; jj < n; jj += PLOT) {
            int i_max = ii + PLOT < n ? ii + PLOT : n;
            int j_max = jj + PLOT < n ? jj + PLOT : n;
            for (int kk = 0; kk < n; kk += PLOT) {
                int k_max = kk + PLOT < n ? kk + PLOT : n;
                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        double sum = C[(size_t)i * n + j];
                        for (int k = kk; k < k_max; k++) {
                            double a = A[(size_t)i * n + k];
                            double b = B[(size_t)k * n + j];
                            sum += graft_multiply(a, b);   /* banked into the third ledger's cell */
                        }
                        C[(size_t)i * n + j] = sum;
                    }
                }
            }
        }
    }
}
