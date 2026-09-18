#include <string.h>
#include <math.h>
#include <stdint.h>

/* the strawberry-chart: a memorized rung-product table, shared by every stalk */
static uint16_t BYTE_MUL[256][256];
static int BYTE_MUL_READY = 0;

static void init_strawberry_chart(void) {
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++)
            BYTE_MUL[a][b] = (uint16_t)(a * b);
    BYTE_MUL_READY = 1;
}

/* multiply two nonnegative finite doubles by decomposing each into a
 * row of 8-bit rung-coins, looking up every rung-against-rung product
 * on the strawberry-chart, and letting the partial products cascade
 * through carry-baskets into one settled basin. */
static inline double lut_mul_mag(double a, double b) {
    if (a == 0.0 || b == 0.0) return 0.0;

    int ea, eb;
    double ma = frexp(a, &ea);   /* a = ma * 2^ea, ma in [0.5,1) */
    double mb = frexp(b, &eb);

    /* the coin-row: exact 53-bit integer mantissa, read off (not multiplied) */
    uint64_t Ma = (uint64_t)ldexp(ma, 53);
    uint64_t Mb = (uint64_t)ldexp(mb, 53);

    uint8_t Alimb[7], Blimb[7];
    for (int i = 0; i < 7; i++) {
        Alimb[i] = (uint8_t)(Ma & 0xFF); Ma >>= 8;
        Blimb[i] = (uint8_t)(Mb & 0xFF); Mb >>= 8;
    }

    /* the lattice of stalks: one carry-basket per output rung position */
    uint32_t basket[14];
    memset(basket, 0, sizeof(basket));

    for (int p = 0; p < 7; p++) {
        if (!Alimb[p]) continue;
        for (int q = 0; q < 7; q++) {
            if (!Blimb[q]) continue;
            basket[p + q] += BYTE_MUL[Alimb[p]][Blimb[q]]; /* strawberry-chart lookup */
        }
    }

    /* overflow tumbles downstream, rung by rung */
    uint8_t settled[14];
    uint32_t carry = 0;
    for (int pos = 0; pos < 14; pos++) {
        uint32_t v = basket[pos] + carry;
        settled[pos] = (uint8_t)(v & 0xFF);
        carry = v >> 8;
    }

    /* the basin: fold the settled rungs, most significant first, into one double */
    double D = 0.0;
    for (int pos = 13; pos >= 0; pos--)
        D = ldexp(D, 8) + (double)settled[pos];

    return ldexp(D, ea + eb - 106);
}

static inline double lut_mul(double a, double b) {
    if (!isfinite(a) || !isfinite(b)) return a * b; /* stalks don't cover nan/inf */
    double r = lut_mul_mag(fabs(a), fabs(b));
    int neg = (a < 0.0) != (b < 0.0);
    return neg ? -r : r;
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (!BYTE_MUL_READY) init_strawberry_chart();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double basin = 0.0; /* the basin at the bottom of this cell */
            for (int k = 0; k < n; k++) {
                double coin_a = A[(size_t)i * n + k];
                double coin_b = B[(size_t)k * n + j];
                basin += lut_mul(coin_a, coin_b);
            }
            C[(size_t)i * n + j] = basin;
        }
    }
}
