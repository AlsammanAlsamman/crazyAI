#include <string.h>
#include <immintrin.h>

#define GF_POLY 0x1D
#define MAX_NSYM 255

static unsigned char GF_EXP[512];
static unsigned char GF_LOG[256];
static int gf_ready = 0;

static void gf_init(void) {
    if (gf_ready) return;
    unsigned char x = 1;
    for (int i = 0; i < 255; i++) {
        GF_EXP[i] = x;
        GF_LOG[x] = (unsigned char)i;
        unsigned char hi = (unsigned char)(x & 0x80);
        x = (unsigned char)(x << 1);
        if (hi) x ^= GF_POLY;
    }
    for (int i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
    GF_LOG[0] = 0;
    gf_ready = 1;
}

static inline unsigned char gfmul(unsigned char a, unsigned char b) {
    if (a == 0 || b == 0) return 0;
    return GF_EXP[(int)GF_LOG[a] + (int)GF_LOG[b]];
}

static int cached_nsym = -1;
static unsigned char cached_MTt[256 * (MAX_NSYM + 1)];

static void build_for_nsym(int nsym) {
    gf_init();
    if (cached_nsym == nsym) return;

    unsigned char gen[MAX_NSYM + 1];
    gen[0] = 1;
    for (int i = 1; i <= nsym; i++) gen[i] = 0;
    unsigned char r = 1;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = r;
        for (int j = glen; j >= 1; j--) {
            gen[j] = (unsigned char)(gen[j] ^ gfmul(gen[j - 1], root));
        }
        glen++;
        unsigned char hi = (unsigned char)(r & 0x80);
        r = (unsigned char)(r << 1);
        if (hi) r ^= GF_POLY;
    }

    unsigned char loggen[MAX_NSYM + 1];
    for (int j = 0; j <= nsym; j++) loggen[j] = gen[j] ? GF_LOG[gen[j]] : 0;

    for (int c = 0; c < 256; c++) {
        unsigned char *row = cached_MTt + (size_t)c * (nsym + 1);
        if (c == 0) {
            for (int j = 0; j <= nsym; j++) row[j] = 0;
        } else {
            int lc = GF_LOG[(unsigned char)c];
            for (int j = 0; j <= nsym; j++) {
                row[j] = gen[j] ? GF_EXP[lc + loggen[j]] : 0;
            }
        }
    }
    cached_nsym = nsym;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    if (nsym <= 0) return;
    if (nsym > MAX_NSYM) nsym = MAX_NSYM;

    build_for_nsym(nsym);
    const unsigned char *MTt = cached_MTt;
    const int stride = nsym + 1;

#if defined(__SSE4_1__)
    if (nsym == 16) {
        unsigned char Rbuf[16];
        for (int j = 0; j < 16; j++) Rbuf[j] = (j < k) ? data[j] : 0;
        __m128i vR = _mm_loadu_si128((const __m128i *)Rbuf);
        for (int i = 0; i < k; i++) {
            unsigned char coef = (unsigned char)(_mm_cvtsi128_si32(vR) & 0xFF);
            unsigned char next_in = (i + 16 < k) ? data[i + 16] : 0;
            const unsigned char *row = MTt + (size_t)coef * 17;
            __m128i vrow = _mm_loadu_si128((const __m128i *)(row + 1));
            __m128i shifted = _mm_srli_si128(vR, 1);
            shifted = _mm_insert_epi8(shifted, next_in, 15);
            vR = _mm_xor_si128(shifted, vrow);
        }
        _mm_storeu_si128((__m128i *)parity, vR);
        return;
    }
#endif

    {
        unsigned char R[MAX_NSYM];
        for (int j = 0; j < nsym; j++) R[j] = (j < k) ? data[j] : 0;
        for (int i = 0; i < k; i++) {
            unsigned char coef = R[0];
            unsigned char next_in = (i + nsym < k) ? data[i + nsym] : 0;
            const unsigned char *row = MTt + (size_t)coef * stride;
            for (int j = 0; j < nsym - 1; j++) {
                R[j] = (unsigned char)(R[j + 1] ^ row[j + 1]);
            }
            R[nsym - 1] = (unsigned char)(next_in ^ row[nsym]);
        }
        memcpy(parity, R, (size_t)nsym);
    }
}
