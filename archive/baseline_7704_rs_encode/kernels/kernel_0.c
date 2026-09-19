#include <string.h>
#include <immintrin.h>

#define NSYM 16

static unsigned char gf_mul_c(unsigned char a, unsigned char b) {
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
static unsigned char gf_pow_c(unsigned char a, int e) {
    unsigned char r = 1;
    for (int i = 0; i < e; i++) r = gf_mul_c(r, a);
    return r;
}

static __m128i bigtab[256];
static volatile int bigtab_ready = 0;

static void build_bigtab(void) {
    unsigned char gen[NSYM + 1];
    gen[0] = 1;
    for (int i = 1; i <= NSYM; i++) gen[i] = 0;
    int glen = 1;
    for (int i = 0; i < NSYM; i++) {
        unsigned char root = gf_pow_c(2, i);
        unsigned char newgen[NSYM + 2];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) {
            newgen[j]     ^= gen[j];
            newgen[j + 1] ^= gf_mul_c(gen[j], root);
        }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }
    /* gen[0] is always 1 by construction; taps are gen[1..NSYM] */
    unsigned char taps[NSYM];
    for (int j = 0; j < NSYM; j++) taps[j] = gen[j + 1];

    unsigned char tmp[16];
    for (int fb = 0; fb < 256; fb++) {
        for (int j = 0; j < NSYM; j++) tmp[j] = gf_mul_c(taps[j], (unsigned char)fb);
        bigtab[fb] = _mm_loadu_si128((const __m128i *)tmp);
    }
    bigtab_ready = 1;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    (void)n; /* n-k is fixed at 16 by contract */
    if (!bigtab_ready) build_bigtab();

    __m128i par = _mm_setzero_si128();

    for (int i = 0; i < k; i++) {
        unsigned char p0 = (unsigned char)_mm_cvtsi128_si32(par);
        unsigned char fb = data[i] ^ p0;
        par = _mm_xor_si128(_mm_srli_si128(par, 1), bigtab[fb]);
    }

    _mm_storeu_si128((__m128i *)parity, par);
}
