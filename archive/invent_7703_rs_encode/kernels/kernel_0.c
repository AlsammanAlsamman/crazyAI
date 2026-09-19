#include <string.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define KERNEL_HAVE_SSE2 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define KERNEL_NSYM_MAX 16
#define KERNEL_PERIOD   255

/* ---- GF(256) log/antilog tables, poly 0x11D, primitive element 2 ---- */
static unsigned char k_log[256];
static unsigned char k_exp[512];
static int k_field_ready = 0;

static void kernel_build_field(void) {
    if (k_field_ready) return;
    unsigned char x = 1;
    for (int i = 0; i < 255; i++) {
        k_exp[i] = x;
        k_log[x] = (unsigned char)i;
        unsigned char hi = (unsigned char)(x & 0x80);
        x = (unsigned char)(x << 1);
        if (hi) x = (unsigned char)(x ^ 0x1D);
    }
    for (int i = 255; i < 512; i++) k_exp[i] = k_exp[i - 255];
    k_field_ready = 1;
}

static inline unsigned char kernel_gf_mul(unsigned char a, unsigned char b) {
    if (a == 0 || b == 0) return 0;
    int s = (int)k_log[a] + (int)k_log[b];
    if (s >= 255) s -= 255;
    return k_exp[s];
}

/* ---- corridors: R_e = x^e mod g(x) for e = 0..254 (period is exactly
   255 because 2 is primitive), and the full "cup's walk" table
   T[e][v] = v * R_e, a 16-byte row = every corridor's mark for a cup
   of value v standing at position-class e ---- */
static unsigned char k_R[KERNEL_PERIOD][KERNEL_NSYM_MAX];
static unsigned char k_T[KERNEL_PERIOD][256][KERNEL_NSYM_MAX];
static int k_tables_ready = 0;
static int k_cached_nsym = -1;

static void kernel_build_tables(int nsym) {
    if (k_tables_ready && k_cached_nsym == nsym) return;
    kernel_build_field();

    /* generator polynomial: g(x) = prod_{i=0}^{nsym-1} (x - 2^i) */
    unsigned char gen[KERNEL_NSYM_MAX + 1];
    gen[0] = 1;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = k_exp[i];
        unsigned char newgen[KERNEL_NSYM_MAX + 1];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) {
            newgen[j] = (unsigned char)(newgen[j] ^ gen[j]);
            newgen[j + 1] = (unsigned char)(newgen[j + 1] ^ kernel_gf_mul(gen[j], root));
        }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }

    /* R_0 = 1 (x^0 term = 1, stored as the last coefficient) */
    unsigned char r[KERNEL_NSYM_MAX];
    for (int j = 0; j < nsym; j++) r[j] = 0;
    r[nsym - 1] = 1;

    for (int e = 0; e < KERNEL_PERIOD; e++) {
        for (int j = 0; j < nsym; j++) k_R[e][j] = r[j];
        unsigned char overflow = r[0];
        unsigned char newr[KERNEL_NSYM_MAX];
        for (int j = 0; j < nsym - 1; j++) newr[j] = r[j + 1];
        newr[nsym - 1] = 0;
        if (overflow) {
            for (int j = 0; j < nsym; j++) newr[j] = (unsigned char)(newr[j] ^ gen[j + 1]);
        }
        for (int j = 0; j < nsym; j++) r[j] = newr[j];
    }

    for (int e = 0; e < KERNEL_PERIOD; e++) {
        for (int v = 0; v < 256; v++) {
            if (v == 0) {
                for (int j = 0; j < nsym; j++) k_T[e][v][j] = 0;
            } else {
                int lv = k_log[(unsigned char)v];
                for (int j = 0; j < nsym; j++) {
                    unsigned char rj = k_R[e][j];
                    if (rj == 0) k_T[e][v][j] = 0;
                    else {
                        int s = lv + k_log[rj];
                        if (s >= 255) s -= 255;
                        k_T[e][v][j] = k_exp[s];
                    }
                }
            }
        }
    }
    k_cached_nsym = nsym;
    k_tables_ready = 1;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    kernel_build_tables(nsym);

    long e0l = ((long)k + nsym - 1) % KERNEL_PERIOD;
    if (e0l < 0) e0l += KERNEL_PERIOD;
    int e0 = (int)e0l;

#ifdef KERNEL_HAVE_SSE2
    if (nsym == 16) {
        __m128i vacc = _mm_setzero_si128();

#ifdef _OPENMP
        if (k >= 20000) {
            int maxt = omp_get_max_threads();
            if (maxt > 128) maxt = 128;
            __m128i partial[128];
            for (int t = 0; t < maxt; t++) partial[t] = _mm_setzero_si128();

            #pragma omp parallel num_threads(maxt)
            {
                int tid = omp_get_thread_num();
                __m128i local = _mm_setzero_si128();
                #pragma omp for schedule(static)
                for (int i = 0; i < k; i++) {
                    int e = e0 - i;
                    e %= KERNEL_PERIOD;
                    if (e < 0) e += KERNEL_PERIOD;
                    local = _mm_xor_si128(local, _mm_loadu_si128((const __m128i*)k_T[e][data[i]]));
                }
                partial[tid] = local;
            }
            for (int t = 0; t < maxt; t++) vacc = _mm_xor_si128(vacc, partial[t]);
            _mm_storeu_si128((__m128i*)parity, vacc);
            return;
        }
#endif
        int e = e0;
        for (int i = 0; i < k; i++) {
            vacc = _mm_xor_si128(vacc, _mm_loadu_si128((const __m128i*)k_T[e][data[i]]));
            e--;
            if (e < 0) e += KERNEL_PERIOD;
        }
        _mm_storeu_si128((__m128i*)parity, vacc);
        return;
    }
#endif

    /* generic fallback for nsym != 16 or non-x86 targets */
    unsigned char acc[KERNEL_NSYM_MAX];
    memset(acc, 0, (size_t)nsym);
    int e = e0;
    for (int i = 0; i < k; i++) {
        const unsigned char *row = k_T[e][data[i]];
        for (int j = 0; j < nsym; j++) acc[j] ^= row[j];
        e--;
        if (e < 0) e += KERNEL_PERIOD;
    }
    memcpy(parity, acc, (size_t)nsym);
}
