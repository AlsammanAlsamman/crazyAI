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

/* TAP_VEC[c][m] = gf_mul(gen[m+1], c) for m = 0..15 (nsym fixed at 16 by the harness).
 * Each row is the complete, once-and-for-all "pressing" of one possible feedback
 * byte value into all 16 seal lanes at once -- no shift-and-reduce is ever done
 * while folding a real message byte; the answer already exists in the row.
 * Built once (rim-waterfall "dust settling"), then trusted for every later press. */
static unsigned char TAP_VEC[256][16] __attribute__((aligned(16)));
static int g_built = 0;
static int g_built_nsym = -1;

static void build_tap_vectors(int nsym) {
    unsigned char gen[64];
    gen_poly(nsym, gen);
    for (int c = 0; c < 256; c++) {
        unsigned char cc = (unsigned char)c;
        for (int m = 0; m < 16; m++)
            TAP_VEC[c][m] = (m < nsym) ? gf_mul(gen[m + 1], cc) : 0;
    }
    g_built = 1;
    g_built_nsym = nsym;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;

    if (nsym == 16) {
        if (!g_built || g_built_nsym != nsym) build_tap_vectors(nsym);

        /* Load the seal window: the first 16 message bytes (zero-padded if k<16).
           lane m == byte offset m, low byte = lane 0, matching x86 little-endian. */
        unsigned char win[16];
        for (int m = 0; m < 16; m++) win[m] = (m < k) ? data[m] : 0;
        __m128i reg = _mm_loadu_si128((const __m128i *)win);

        int i = 0;
        int main_end = k - 16;   /* while i < main_end, data[i+16] is a real, in-bounds byte */
        for (; i < main_end; i++) {
            unsigned char coef   = (unsigned char)_mm_cvtsi128_si32(reg); /* lane 0 = low byte */
            unsigned char newtop = data[i + 16];
            __m128i shifted = _mm_srli_si128(reg, 1);                    /* lane m <- lane m+1, lane15<-0 */
            __m128i tap     = _mm_load_si128((const __m128i *)TAP_VEC[coef]);
            __m128i inj     = _mm_set_epi8((char)newtop, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
            reg = _mm_xor_si128(_mm_xor_si128(shifted, tap), inj);
        }
        for (; i < k; i++) {   /* tail: i+16 >= k, so the new-top byte is implicitly 0 */
            unsigned char coef = (unsigned char)_mm_cvtsi128_si32(reg);
            __m128i shifted = _mm_srli_si128(reg, 1);
            __m128i tap     = _mm_load_si128((const __m128i *)TAP_VEC[coef]);
            reg = _mm_xor_si128(shifted, tap);
        }
        _mm_storeu_si128((__m128i *)parity, reg);
        return;
    }

    /* generic scalar fallback for nsym != 16 (defensive only; harness fixes nsym=16) */
    {
        unsigned char gen[64];
        gen_poly(nsym, gen);
        unsigned char *win = (unsigned char *)calloc((size_t)nsym, 1);
        for (int m = 0; m < nsym && m < k; m++) win[m] = data[m];
        for (int i = 0; i < k; i++) {
            unsigned char coef   = win[0];
            unsigned char newtop = (i + nsym < k) ? data[i + nsym] : 0;
            for (int m = 0; m < nsym - 1; m++) win[m] = (unsigned char)(win[m + 1] ^ gf_mul(gen[m + 1], coef));
            win[nsym - 1] = (unsigned char)(newtop ^ gf_mul(gen[nsym], coef));
        }
        memcpy(parity, win, (size_t)nsym);
        free(win);
    }
}
