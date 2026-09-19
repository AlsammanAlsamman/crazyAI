#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NSYM_MAX 16

/* ---- GF(256) tables (reduction poly 0x11D, primitive element 2) ---- */
static unsigned char GF_EXP[512];
static unsigned char GF_LOG[256];
static int gf_tables_ready = 0;

static void gf_init_tables(void) {
    if (gf_tables_ready) return;
    unsigned char x = 1;
    for (int i = 0; i < 255; i++) {
        GF_EXP[i] = x;
        GF_LOG[x] = (unsigned char)i;
        x = (unsigned char)((x << 1) ^ ((x & 0x80) ? 0x1D : 0));
    }
    for (int i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
    gf_tables_ready = 1;
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

static void gen_poly(int nsym, unsigned char *gen) {
    gen[0] = 1;
    for (int i = 1; i <= nsym; i++) gen[i] = 0;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = gf_pow(2, i);
        unsigned char newgen[NSYM_MAX + 1];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) {
            newgen[j] ^= gen[j];
            newgen[j + 1] ^= gf_mul(gen[j], root);
        }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }
}

/* "pure shift by x mod g(x)" step on the remainder register (no new data injected) */
static inline void x_shift(unsigned char *S, const unsigned char *gen, int nsym) {
    unsigned char f = S[0];
    for (int j = 0; j < nsym - 1; j++)
        S[j] = (unsigned char)(S[j + 1] ^ gf_mul(f, gen[j + 1]));
    S[nsym - 1] = gf_mul(f, gen[nsym]);
}

/* ---- cached basis: R[i][0..nsym-1] = impulse response of data[i]=1, rest 0 ---- */
static unsigned char *cache_R = NULL;   /* k * nsym bytes */
static int cache_k = -1;
static int cache_nsym = -1;
static unsigned char cache_gen[NSYM_MAX + 1];

static void build_basis(int k, int nsym) {
    gf_init_tables();
    gen_poly(nsym, cache_gen);

    unsigned char *R = (unsigned char *)malloc((size_t)k * (size_t)nsym);

    /* base vector B = response to data[k-1]=1 alone: STEP(0,1) */
    unsigned char S[NSYM_MAX];
    for (int j = 0; j < nsym - 1; j++) S[j] = cache_gen[j + 1];
    S[nsym - 1] = cache_gen[nsym];
    memcpy(R + (size_t)(k - 1) * nsym, S, (size_t)nsym);

    for (int i = k - 2; i >= 0; i--) {
        x_shift(S, cache_gen, nsym);
        memcpy(R + (size_t)i * nsym, S, (size_t)nsym);
    }

    free(cache_R);
    cache_R = R;
    cache_k = k;
    cache_nsym = nsym;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    if (nsym <= 0) return;
    if (nsym > NSYM_MAX) nsym = NSYM_MAX;

    if (k <= 0) { memset(parity, 0, (size_t)nsym); return; }

    if (cache_R == NULL || cache_k != k || cache_nsym != nsym) {
        build_basis(k, nsym);
    }

    unsigned char acc_eye[NSYM_MAX];   /* eye-fold: even positions */
    unsigned char acc_ankle[NSYM_MAX]; /* ankle-fold: odd positions */
    memset(acc_eye, 0, (size_t)nsym);
    memset(acc_ankle, 0, (size_t)nsym);

    const unsigned char *R = cache_R;
    int i = 0;
    for (; i + 1 < k; i += 2) {
        unsigned char be = data[i];
        unsigned char bo = data[i + 1];
        const unsigned char *Re = R + (size_t)i * nsym;
        const unsigned char *Ro = R + (size_t)(i + 1) * nsym;
        if (be) {
            int le = GF_LOG[be];
            for (int j = 0; j < nsym; j++)
                if (Re[j]) acc_eye[j] ^= GF_EXP[le + GF_LOG[Re[j]]];
        }
        if (bo) {
            int lo = GF_LOG[bo];
            for (int j = 0; j < nsym; j++)
                if (Ro[j]) acc_ankle[j] ^= GF_EXP[lo + GF_LOG[Ro[j]]];
        }
    }
    for (; i < k; i++) { /* leftover byte on odd k, folded into eye pit */
        unsigned char b = data[i];
        if (b) {
            const unsigned char *Ri = R + (size_t)i * nsym;
            int lb = GF_LOG[b];
            for (int j = 0; j < nsym; j++)
                if (Ri[j]) acc_eye[j] ^= GF_EXP[lb + GF_LOG[Ri[j]]];
        }
    }

    for (int j = 0; j < nsym; j++)
        parity[j] = (unsigned char)(acc_eye[j] ^ acc_ankle[j]); /* threshold fold */
}
