#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define NSYM_FIXED 16

static unsigned char GF_EXP[512];
static unsigned char GF_LOG[256];
static int gf_ready = 0;

static void init_gf(void) {
    if (gf_ready) return;
    unsigned char x = 1;
    for (int i = 0; i < 255; i++) {
        GF_EXP[i] = x;
        GF_LOG[x] = (unsigned char)i;
        int hi = x & 0x80;
        x = (unsigned char)(x << 1);
        if (hi) x ^= 0x1D;
    }
    for (int i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
    GF_LOG[0] = 0;
    gf_ready = 1;
}

static inline unsigned char gf_mul(unsigned char a, unsigned char b) {
    if (a == 0 || b == 0) return 0;
    return GF_EXP[(int)GF_LOG[a] + (int)GF_LOG[b]];
}

static unsigned char gf_pow(unsigned char a, int e) {
    unsigned char r = 1;
    for (int i = 0; i < e; i++) r = gf_mul(r, a);
    return r;
}

/* The "board": one clutch-vector (nsym bytes) per house (message
   position i) = the exact parity produced by a lone 1-valued stone
   placed in house i. Built once per (k,nsym) and reused across every
   later call — the board remembers its houses, not the process. */
static unsigned char *cache_M = NULL;
static int cache_k = -1;
static int cache_nsym = -1;

static void build_board(int k, int nsym) {
    init_gf();

    unsigned char gen[NSYM_FIXED + 1];
    gen[0] = 1;
    for (int i = 1; i <= nsym; i++) gen[i] = 0;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = gf_pow(2, i);
        unsigned char newgen[NSYM_FIXED + 1];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) {
            newgen[j] ^= gen[j];
            newgen[j + 1] ^= gf_mul(gen[j], root);
        }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }

    unsigned char *M = (unsigned char *)malloc((size_t)k * (size_t)nsym);

    /* g(0): clutch of the LAST house (walk starts at the outer edge) */
    unsigned char reg[NSYM_FIXED];
    for (int j = 0; j < nsym; j++) reg[j] = gen[j + 1];
    memcpy(M + (size_t)(k - 1) * nsym, reg, (size_t)nsym);

    /* tightening spiral: one zero-input LFSR clock per house, walking
       inward from the edge; each turn yields a fresh, never-repeated
       clutch for the next house */
    for (int i = k - 2; i >= 0; i--) {
        unsigned char coef = reg[0];
        unsigned char newreg[NSYM_FIXED];
        for (int t = 0; t < nsym - 1; t++)
            newreg[t] = (unsigned char)(reg[t + 1] ^ gf_mul(gen[t + 1], coef));
        newreg[nsym - 1] = gf_mul(gen[nsym], coef);
        memcpy(reg, newreg, (size_t)nsym);
        memcpy(M + (size_t)i * nsym, reg, (size_t)nsym);
    }

    cache_M = M;
    cache_k = k;
    cache_nsym = nsym;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;

    if (k <= 0) {
        for (int j = 0; j < nsym; j++) parity[j] = 0;
        return;
    }

    if (cache_M == NULL || cache_k != k || cache_nsym != nsym) {
        if (cache_M) { free(cache_M); cache_M = NULL; }
        build_board(k, nsym);
    }

    const unsigned char *M = cache_M;
    unsigned char acc[NSYM_FIXED];
    for (int j = 0; j < nsym; j++) acc[j] = 0;

    /* every message-stone is combined with its own house's clutch and
       thrown into the seal via the erasing arithmetic (xor); since the
       contributions are independent, many rivers (threads) can carve
       their own partial seal and be combined at the end without
       confusion */
    #pragma omp parallel for reduction(^:acc[:NSYM_FIXED]) schedule(static) if(k >= 4096)
    for (int i = 0; i < k; i++) {
        unsigned char v = data[i];
        if (v) {
            unsigned char lv = GF_LOG[v];
            const unsigned char *row = M + (size_t)i * nsym;
            for (int j = 0; j < nsym; j++) {
                unsigned char c = row[j];
                if (c) acc[j] = (unsigned char)(acc[j] ^ GF_EXP[(int)lv + (int)GF_LOG[c]]);
            }
        }
    }

    for (int j = 0; j < nsym; j++) parity[j] = acc[j];
}
