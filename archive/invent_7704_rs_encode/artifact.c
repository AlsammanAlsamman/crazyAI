#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

static unsigned char gf_mul(unsigned char a, unsigned char b) {
    unsigned char p = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) p ^= a;
        int hi = a & 0x80;
        a = (unsigned char)(a << 1);
        if (hi) a ^= 0x1D;
        b >>= 1;
    }
    return p;
}
static unsigned char gf_pow(unsigned char a, int e) { unsigned char r = 1; for (int i = 0; i < e; i++) r = gf_mul(r, a); return r; }
static void gen_poly(int nsym, unsigned char *gen) {
    gen[0] = 1;
    for (int i = 1; i <= nsym; i++) gen[i] = 0;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = gf_pow(2, i);
        unsigned char newgen[64];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) { newgen[j] ^= gen[j]; newgen[j + 1] ^= gf_mul(gen[j], root); }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }
}

/* the sun's jigsaw sky: 256 cell-rows x nsym slots, folded once, reused forever */
static unsigned char g_tbl[256][32];
static int g_tbl_nsym = -1;

static void build_table(int nsym) {
    unsigned char gen[64];
    gen_poly(nsym, gen);
    for (int b = 0; b < 256; b++) {
        unsigned char bb = (unsigned char)b;
        for (int j = 1; j <= nsym; j++) g_tbl[b][j - 1] = gf_mul(gen[j], bb);
    }
    g_tbl_nsym = nsym;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    if (nsym != g_tbl_nsym) build_table(nsym);

    unsigned char *msg = (unsigned char *)calloc((size_t)(k + nsym), 1);
    memcpy(msg, data, (size_t)k);

    if (nsym == 16) {
        for (int i = 0; i < k; i++) {
            unsigned char coef = msg[i];
            if (coef) {
                unsigned char *dst = msg + i + 1;
                __m128i d = _mm_loadu_si128((const __m128i *)dst);
                __m128i t = _mm_loadu_si128((const __m128i *)g_tbl[coef]);
                _mm_storeu_si128((__m128i *)dst, _mm_xor_si128(d, t));
            }
        }
    } else {
        for (int i = 0; i < k; i++) {
            unsigned char coef = msg[i];
            if (coef) {
                unsigned char *dst = msg + i + 1;
                const unsigned char *row = g_tbl[coef];
                for (int j = 0; j < nsym; j++) dst[j] ^= row[j];
            }
        }
    }

    memcpy(parity, msg + k, (size_t)nsym);
    free(msg);
}
