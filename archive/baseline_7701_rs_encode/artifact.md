# APPROACH

The reference implementation has two costly properties for large `k`:

1. It allocates and repeatedly writes into a scratch buffer of size `k+nsym`, touching memory proportional to the whole message on every pass (poor cache/memory-bandwidth behavior for large `k`).
2. Every GF(256) multiply is done via an 8-step shift-and-reduce loop with branches — the "known way" the prompt describes.

Both can be removed without changing the observable output, because RS systematic encoding via polynomial division admits an exactly-equivalent **LFSR (shift-register) form**: instead of a `k+nsym`-byte buffer, keep only an `nsym`-byte register that always holds the "in-flight" remainder. I derived the register update rule directly from the reference's semantics (verified index-by-index) so it produces bit-identical output:

```
R[j] = R[j+1] ^ gf_mul(gen[j+1], coef)   for j = 0..nsym-2
R[nsym-1] = next_in ^ gf_mul(gen[nsym], coef)
coef = R[0] ;  next_in = data[i+nsym] (or 0 past the end)
```

This is O(k·nsym) exactly like the reference, but touches only `nsym` bytes of state per step instead of writing across a `k+nsym` buffer.

For the multiply itself, instead of even the "two lookups + mod-255 add" log/antilog trick, I precompute — once per distinct `nsym` (cached across calls) — a **row-major multiplication table** `MTt[coef][j] = gf_mul(gen[j], coef)`, built once from log/antilog tables (4352 bytes for `nsym=16`, easily L1-resident). Each encode step then needs a *single* table lookup per output byte (no branch, no add/mask), and because the table is row-major, the whole row (`row[1..nsym]`) is contiguous — which lets me special-case `nsym==16` (guaranteed by the harness) with SSE4.1: the register fits exactly in one `__m128i`, and one step becomes a 16-byte unaligned load + `srli` + `insert` + `xor`, replacing a 15-wide unrolled scalar XOR loop with 4 vector instructions. A scalar fallback handles any other `nsym` for safety.

The generator polynomial and multiplication table are computed once and cached (guarded by a flag checking the last-seen `nsym`), so repeated benchmark calls pay the O(nsym²) setup cost only once.

# ARTIFACT

```c
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
```

# PREDICTION: 8

(Predicted ~8x throughput improvement over the naive reference for large `k`: elimination of the `k+nsym`-byte scratch buffer plus replacing an 8-step shift-reduce multiply with one table lookup, and for the guaranteed `nsym=16` case a 4-instruction SSE step replacing a 15-wide scalar XOR/branch loop.)

# MEASUREMENT

No tool access was available in this session (explicitly disabled — "No tools are available in this session"), so `rs_encode_bench`/`rs_encode_contract` could not actually be invoked here. I am reporting this plainly rather than fabricating numbers: the prediction above is untested and the implementation has not been run through the harness in this session. It should be compiled and measured by the pipeline as stated in the instructions.

# VERDICT

Unverified. The algorithmic argument (O(nsym)-space LFSR form proven index-for-index equivalent to the reference's O(k+nsym)-space convolution, single-lookup GF multiply via a cached per-`nsym` table, SSE4.1 fast path for the guaranteed `nsym=16` case, scalar fallback for any other value) should make this correct and meaningfully faster than the shift-and-reduce reference, but this needs the actual `rs_encode_bench`/`rs_encode_contract` run to confirm correctness and the ~8x prediction — that verification did not happen in this session.