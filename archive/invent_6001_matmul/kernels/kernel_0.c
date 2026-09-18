#include <string.h>
#include <stdint.h>

/* ---- The altar-board: every pairing of single knot (nibble, 0..15)
   against single knot (0..15) already carved. 16x16 = 256 cells. */
static unsigned char ALTAR[16][16];
static int altar_ready = 0;

static void carve_altar(void) {
    for (int a = 0; a < 16; a++)
        for (int b = 0; b < 16; b++)
            ALTAR[a][b] = (unsigned char)(a * b); /* max 15*15=225, fits a byte */
    altar_ready = 1;
}

/* String a 53-bit mantissa into 14 knots (nibbles), ones-knot first. */
static inline void string_the_cord(uint64_t m53, unsigned char cord[14]) {
    for (int i = 0; i < 14; i++) {
        cord[i] = (unsigned char)(m53 & 0xF);
        m53 >>= 4;
    }
}

/* Multiply two doubles by consulting the altar-board, not by a bald '*' act.
   cordA is already strung (hoisted out of the caller's j-loop, since the
   same cord waits unchanged against many different partners). */
static inline double lut_mul_cordA(const unsigned char cordA[14], double b) {
    if (b == 0.0) return 0.0;

    uint64_t bits_b;
    memcpy(&bits_b, &b, 8);
    int eb = (int)((bits_b >> 52) & 0x7FF);
    uint64_t fb = bits_b & 0xFFFFFFFFFFFFFULL;
    uint64_t mb = fb | (1ULL << 52);

    unsigned char cordB[14];
    string_the_cord(mb, cordB);

    /* Ledger of partial sums, one strip per combined place: 14+14=28 places,
       +1 for a final carry that may run over the top. */
    uint32_t ledger[29];
    memset(ledger, 0, sizeof(ledger));

    for (int i = 0; i < 14; i++) {
        unsigned char da = cordA[i];
        if (!da) continue;
        const unsigned char *row = ALTAR[da];
        for (int j = 0; j < 14; j++)
            ledger[i + j] += row[cordB[j]];
    }

    /* Push the carry-stone one strip higher, place by place. */
    for (int k = 0; k < 28; k++) {
        ledger[k + 1] += ledger[k] >> 4;
        ledger[k] &= 0xF;
    }

    /* Reassemble the cord of answer-knots into a 112-bit integer. */
    unsigned __int128 prod = 0;
    for (int k = 27; k >= 0; k--)
        prod = (prod << 4) | (unsigned __int128)ledger[k];

    int topbit = 105;
    while (topbit >= 0 && !((prod >> topbit) & 1)) topbit--;
    if (topbit < 0) return 0.0;

    int shift = topbit - 52;
    uint64_t mant53 = (uint64_t)(prod >> shift);
    if (shift > 0 && ((prod >> (shift - 1)) & 1)) mant53 += 1; /* round-to-nearest, ties up */
    if (mant53 >> 53) { mant53 >>= 1; shift += 1; }

    return mant53 ? 0.0 /*placeholder, filled below*/ : 0.0;
}

/* Full multiply: strings both cords, carries sign/exponent bookkeeping. */
static inline double lut_mul(double a, double b) {
    if (a == 0.0 || b == 0.0) return 0.0;

    uint64_t bits_a, bits_b;
    memcpy(&bits_a, &a, 8);
    memcpy(&bits_b, &b, 8);

    int sign = (int)((bits_a >> 63) ^ (bits_b >> 63));
    int ea = (int)((bits_a >> 52) & 0x7FF);
    int eb = (int)((bits_b >> 52) & 0x7FF);
    uint64_t ma = (bits_a & 0xFFFFFFFFFFFFFULL) | (1ULL << 52);
    uint64_t mb = (bits_b & 0xFFFFFFFFFFFFFULL) | (1ULL << 52);

    unsigned char cordA[14], cordB[14];
    string_the_cord(ma, cordA);
    string_the_cord(mb, cordB);

    uint32_t ledger[29];
    memset(ledger, 0, sizeof(ledger));

    for (int i = 0; i < 14; i++) {
        unsigned char da = cordA[i];
        if (!da) continue;
        const unsigned char *row = ALTAR[da];
        for (int j = 0; j < 14; j++)
            ledger[i + j] += row[cordB[j]];
    }
    for (int k = 0; k < 28; k++) {
        ledger[k + 1] += ledger[k] >> 4;
        ledger[k] &= 0xF;
    }

    unsigned __int128 prod = 0;
    for (int k = 27; k >= 0; k--)
        prod = (prod << 4) | (unsigned __int128)ledger[k];

    int topbit = 105;
    while (topbit >= 0 && !((prod >> topbit) & 1)) topbit--;
    if (topbit < 0) return 0.0;

    int shift = topbit - 52;
    uint64_t mant53 = (uint64_t)(prod >> shift);
    if (shift > 0 && ((prod >> (shift - 1)) & 1)) mant53 += 1;
    int exp = ea + eb - 1023 + (topbit - 104);
    if (mant53 >> 53) { mant53 >>= 1; exp += 1; }

    uint64_t out_bits = ((uint64_t)sign << 63)
                       | ((uint64_t)(exp & 0x7FF) << 52)
                       | (mant53 & 0xFFFFFFFFFFFFFULL);
    double result;
    memcpy(&result, &out_bits, 8);
    return result;
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (!altar_ready) carve_altar();
    memset(C, 0, (size_t)n * n * sizeof(double));

    const int BS = 64; /* a timber hold sized to fit the counting-deck (cache) */

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += BS) {
        int imax = ii + BS < n ? ii + BS : n;
        for (int kk = 0; kk < n; kk += BS) {
            int kmax = kk + BS < n ? kk + BS : n;
            for (int i = ii; i < imax; i++) {
                double *Crow = C + (size_t)i * n;
                for (int k = kk; k < kmax; k++) {
                    double a = A[(size_t)i * n + k];
                    if (a == 0.0) continue;
                    const double *Brow = B + (size_t)k * n;
                    for (int j = 0; j < n; j++)
                        Crow[j] += lut_mul(a, Brow[j]);
                }
            }
        }
    }
}
