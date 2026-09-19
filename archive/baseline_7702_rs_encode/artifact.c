#include <stdlib.h>
#include <string.h>

/* ---- GF(256) log/antilog tables, primitive element 2, poly 0x11D ---- */
static unsigned char gf_log[256];
static unsigned char gf_exp[512];
static int gf_tables_ready = 0;

static void init_gf_tables(void) {
    unsigned int x = 1;
    for (int i = 0; i < 255; i++) {
        gf_exp[i] = (unsigned char)x;
        gf_log[x] = (unsigned char)i;
        x <<= 1;
        if (x & 0x100) x ^= 0x11D;
    }
    for (int i = 255; i < 512; i++) gf_exp[i] = gf_exp[i - 255];
    gf_log[0] = 0; /* never dereferenced meaningfully */
    gf_tables_ready = 1;
}

/* ---- generator polynomial cache, keyed on nsym ---- */
static unsigned char cached_gen[258];
static unsigned char cached_gennz[258];
static unsigned char cached_genlog[258];
static int cached_nsym = -1;

static void build_generator(int nsym, unsigned char *gen) {
    unsigned char newgen[258];
    gen[0] = 1;
    for (int i = 1; i <= nsym; i++) gen[i] = 0;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = gf_exp[i % 255];
        unsigned char rlog = gf_log[root];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) {
            newgen[j] ^= gen[j];
            if (gen[j]) newgen[j + 1] ^= gf_exp[(unsigned int)gf_log[gen[j]] + rlog];
        }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }
}

static void ensure_generator(int nsym) {
    if (!gf_tables_ready) init_gf_tables();
    if (cached_nsym == nsym) return;
    build_generator(nsym, cached_gen);
    for (int j = 0; j <= nsym; j++) {
        cached_gennz[j] = (unsigned char)(cached_gen[j] != 0);
        cached_genlog[j] = cached_gennz[j] ? gf_log[cached_gen[j]] : 0;
    }
    cached_nsym = nsym;
}

/* ---- reusable scratch buffer (avoid malloc/free every call) ---- */
static unsigned char *scratch = NULL;
static size_t scratch_cap = 0;

static unsigned char *get_scratch(size_t need) {
    if (need > scratch_cap) {
        size_t newcap = need * 2 + 32;
        unsigned char *p = (unsigned char *)realloc(scratch, newcap);
        if (!p) { p = (unsigned char *)realloc(scratch, need); newcap = need; }
        scratch = p;
        scratch_cap = newcap;
    }
    return scratch;
}

/* specialized, fully-unrollable path for the contractual nsym == 16 */
static void encode_nsym16(int k, unsigned char *msg) {
    for (int i = 0; i < k; i++) {
        unsigned char coef = msg[i];
        if (coef) {
            unsigned int lc = gf_log[coef];
            unsigned char *m = msg + i;
            const unsigned char *ge = gf_exp + lc;
            #pragma GCC unroll 17
            for (int j = 0; j <= 16; j++) {
                if (cached_gennz[j]) m[j] ^= ge[cached_genlog[j]];
            }
        }
    }
}

/* generic fallback for any other nsym (defensive; contract fixes nsym=16) */
static void encode_generic(int k, int nsym, unsigned char *msg) {
    for (int i = 0; i < k; i++) {
        unsigned char coef = msg[i];
        if (coef) {
            unsigned int lc = gf_log[coef];
            unsigned char *m = msg + i;
            const unsigned char *ge = gf_exp + lc;
            for (int j = 0; j <= nsym; j++) {
                if (cached_gennz[j]) m[j] ^= ge[cached_genlog[j]];
            }
        }
    }
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    ensure_generator(nsym);

    size_t total = (size_t)k + (size_t)nsym;
    unsigned char *msg = get_scratch(total);
    memcpy(msg, data, (size_t)k);
    memset(msg + k, 0, (size_t)nsym);

    if (nsym == 16) encode_nsym16(k, msg);
    else            encode_generic(k, nsym, msg);

    memcpy(parity, msg + k, (size_t)nsym);
}
