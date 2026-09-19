# MAPPING

**SEED 1** — "corridors chosen by folding each cup's position"

| World object | Problem object |
|---|---|
| cup | one message byte `data[i]` |
| line of cups | the message array `data[0..k-1]` |
| the house / wider aboard of dark | the GF(2^8) encoding computation |
| corridor | one parity-byte slot `j = 0..nsym-1` |
| corridor's number (knocks: once, twice, fours, eights…) | the corridor's root, literally the powers of 2: `2^j`, the evaluation points that define the generator polynomial `g(x)=∏(x-2^j)` |
| "folding a cup's position" to pick corridors | the fixed GF(256) coefficient linking source position `i` to target corridor `j` — depends only on `i` (mod 255, since `x` has multiplicative order 255) |
| "a cup can walk more than one corridor — that's the whole trick" | one data byte contributes, by lookup, to *all 16* parity bytes at once, and different bytes' contributions are independent (order-free, XOR-associative) |

Breaks: *"every message token must be folded into the seal individually, one at a time"* and *"a combination's answer cannot be known without shift-reduce."*

**SEED 2** — basin arithmetic (`a⊕a=0`, odd arrivals mark the wax)

| World object | Problem object |
|---|---|
| basin | per-corridor accumulator |
| pouring | XOR-accumulate |
| self-pour empties it / odd count marks it | definition of XOR as the field's addition |

This just re-states that the accumulator is XOR — every implementation (naive included) already does this. It doesn't suggest anything structurally new about *how* multiplication happens.

**SEED 3** — residues skimmed once basins go still, strung in corridor order

| World object | Problem object |
|---|---|
| basin going still | accumulator reaching its final value after all bytes folded in |
| residues strung in corridor order | `parity[0..nsym-1]` written out in position order |

Breaks *"tokens must be produced strictly in order, most significant first"* — but only about **output packaging**, not about the arithmetic itself; the decoding remainder is out of scope for an encode-only kernel.

# CHOSEN SEED
SEED 1. It is the only one that is both maximally literal (every noun has a distinct, non-overlapping target: cup, corridor, corridor-number, position-fold) and maximally different from the known way: the known way's "headroom" is still a *sequential*, *per-bit-then-per-byte* multiply (even with log/antilog tables, it's still `k×nsym` multiply-lookups inside a running LFSR register). SEED 1 says a cup's whole multi-corridor walk is fixed by its *position alone* — i.e. precomputable once, off the hot path — and that cups don't need to be folded in any particular order into any particular corridor sequentially.

# ASSUMPTION BROKEN
"Each token combination must be computed from scratch by shifting and reducing" **and** "the strange arithmetic has no shortcut table, only the shift-reduce rule." Since RS systematic encoding is GF(256)-linear in the data bytes, and since `x^e mod g(x)` is periodic in `e` with period exactly 255 (because 2 is primitive, order 255, and `g(x)` factors into 16 distinct linear GF(256) factors), the entire per-position "walk down every corridor" can be pretabulated as a `255×256` table of 16-byte rows, built once (cost independent of `k`, amortized to ~0 across calls). Encoding a message degenerates to `k` sixteen-byte table lookups XORed together — no multiplication, no bit-serial shifting, ever, on the hot path.

# ARTIFACT

```c
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
```

PREDICTION: speedup_vs_naive = 40

# MEASUREMENT
No tool execution is available in this session (explicitly stated at the top of the task), so `rs_encode_bench` could not actually be invoked here — I cannot in good conscience report a measured number I never obtained. Reasoned estimate only: naive does `k×(nsym+1)=17k` bit-serial `gf_mul` calls (each an 8-iteration shift/branch loop, ~15-25 cycles), i.e. roughly 300-400 cycles/byte. The table-walk kernel does one 16-byte unaligned load + XOR per byte (~2-4 cycles compute, plus L2-range latency since the 1 MB table won't fully live in L1 — estimate ~8-15 cycles/byte effective once warm, with the build cost amortized to ~0 after the first call since it depends only on `nsym`, fixed at 16). That puts single-thread speedup in the 25-50x range; the OpenMP path should push it higher for large `k` if the harness measures big buffers, bounded by memory bandwidth rather than core count. I am flagging this as unverified and asking that the pipeline's actual `rs_encode_bench` run be treated as the real answer, not this estimate.

# VERDICT
The artifact is a literal, non-hedged translation of SEED 1: corridors are the 16 roots `2^j`, a cup's "folded position" is turned into the periodic-255 position-class `e`, and its full multi-corridor walk is baked into one precomputed 16-byte row so that encoding becomes pure lookup-and-XOR with no shift-reduce anywhere on the hot path — genuinely different from, not a relabeling of, the log/antilog LFSR "known way." Correctness was checked by hand against the reference algebra (`parity(x) = data(x)·x^nsym mod g(x)`, linearity, and the `nsym=1` toy case matched the reference bit-for-bit) but not against the actual harness. Speed is a reasoned prediction (~40x), explicitly unmeasured in this session — that gap should be closed by an actual `rs_encode_bench` run before trusting the number.