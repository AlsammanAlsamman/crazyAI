# APPROACH

The reference (and the described "known way") both process each message byte through the generator polynomial one tap at a time, either with 16 raw carry-less multiplications (bit-loop) or with a log/antilog LFSR (2 lookups + branch per tap). Since `n-k` is **fixed at 16** by the harness, the entire 16-byte parity shift register fits in a single SSE `__m128i`. That lets us collapse the whole "shift by one, then XOR in `gen[j]*feedback` for all 16 taps" step into:

1. one 8-bit feedback value `fb = data[i] ^ parity[0]`,
2. one 128-bit right-shift-by-one-byte (`_mm_srli_si128`, shifts old `parity[1..15]` into place, zero-fills the new low tap),
3. one 128-bit XOR with a **precomputed 16-byte vector** `bigtab[fb]`, where `bigtab[fb][j] = gen[j+1] * fb` for `j = 0..15`.

`bigtab` has only 256 possible `fb` values, so it's a 256×16 = 4 KiB table (fits in L1), built once (lazily, cached across calls since `nsym=16` never changes). Per message byte the inner work becomes: 1 scalar XOR, 1 register extract, 1 aligned 16-byte table load, 1 shift, 1 XOR — no branches, no per-tap loop, no log/antilog lookups at all. This removes both the "one bit at a time" cost and the "16 separate table lookups per byte" cost of the two baselines, turning the encoder into an O(k) loop with a tiny, branch-free, shift-register recurrence.

The generator polynomial itself is computed once (16 steps) the first time `kernel` runs and cached in a static table, since it depends only on `nsym = n-k = 16`, never on `k` or the data.

# ARTIFACT

```c
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
```

# PREDICTION: 10

# MEASUREMENT

No benchmarking tools (`rs_encode_bench`, `rs_encode_contract`) were available in this session to actually execute — the tool list in the system context is empty for this run, so I cannot produce real timing numbers. I'm reporting this plainly rather than fabricating a result: this implementation is unverified by direct measurement here and relies on the external pipeline (mentioned in the task) to compile and time it.

Reasoning behind the prediction (~10x over the log/antilog LFSR "known way", likely 50-100x over the naive bit-loop reference): the per-byte cost drops from "16 taps × (2 log/antilog lookups + branch)" to "1 scalar XOR + 1 register extract + 1 16-byte table load + 1 shift + 1 XOR", eliminating all branching and reducing memory traffic to one 16-byte load per input byte from a table that stays resident in L1. The recurrence through `par` is still strictly sequential (true dependency chain), so it cannot be parallelized across bytes without a more complex state-transition-matrix approach, but the per-step latency (extract → shift/xor, ~2-3 cycles) should already make this memory/latency-bound rather than compute-bound.

# VERDICT

Correctness: believed correct — the shift/tap decomposition (`feedback = data[i] ^ parity[0]`; shift; XOR in `gen[1..16]*feedback`) is the standard, well-known equivalence between the schoolbook systematic-encode loop (as in the reference `kernel`) and an LFSR with a 16-byte sliding window, and `gen[0]=1` is invariant by construction of `gen_poly`, matching the reference's coefficient-cancellation behavior. This should produce byte-identical parity to the reference for all `k`, with `n-k=16` as guaranteed by the contract.

Performance: predicted to beat the described "known way" (log/antilog LFSR) by roughly an order of magnitude, since it removes per-tap branching and lookups entirely in favor of one fixed-size SIMD table hit and a shift+XOR per input byte — but this claim is **unverified in this session** since no measurement tool was actually invoked here; the external pipeline result is needed to confirm or refute the prediction.