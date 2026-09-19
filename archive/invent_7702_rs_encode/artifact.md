# MAPPING

| World object | Problem object | Silent assumption it threatens |
|---|---|---|
| **SEED 1** — river-stone, notch | message byte `data[i]`, its GF(256) value | — |
| seal-stone | parity byte `parity[j]` | — |
| "combining a distinct clutch of stones" | `parity[j]` expressed as a fixed GF(256)-linear combination of *all* message bytes, one distinct weight-vector per `j` | breaks *"each token combination must be computed from scratch by shifting and reducing"* and *"the strange arithmetic has no shortcut table, only the shift-reduce rule"* |
| "arithmetic where a thing added to itself is erased" | XOR (char-2 field addition, `a⊕a=0`) — the operation that lets partial sums recombine in any order | breaks *"every message token must be folded into the seal individually, one at a time [in sequence]"* |
| **SEED 2** — cross-marked board, houses | the index array of size `k+nsym` | — |
| tightening snake-spiral, walked from the outer edge inward | the backward sweep `i = k-1 … 0` that clocks the LFSR register with **zero input** to build the per-house coefficient table | breaks *"the seal's tokens must be produced strictly in order, most significant first"* |
| "gather a different clutch… never the same clutch twice… casting falls into many rivers without confusing them" | disjoint ranges of `i` handled by independent accumulators (OpenMP threads/lanes), combined by XOR at the end | same assumption as above, at the *execution* level |
| trailing stones = number of snake-loops, set past the last message-stone | the `nsym=16` parity bytes appended after the `k` message bytes (systematic layout) | (structural detail, not an assumption break) |
| **SEED 3** — lost stone recovered by recombining survivors + trailing seal-stones along the same spiral | erasure decoding (solving the linear system for a missing symbol) | not applicable — the contract is encode-only; no decode/recovery call is exposed |

# CHOSEN SEED
**SEED 1**, with SEED 2 supplying the concrete mechanism ("distinct clutches" = disjoint, order-free ranges) by which SEED 1's precomputed combination is actually carried out.

# ASSUMPTION BROKEN
- *"each token combination must be computed from scratch by shifting and reducing"* — the coefficient of every message house is derived **once** (the "board"), not on every call.
- *"the strange arithmetic has no shortcut table, only the shift-reduce rule"* — we build exactly the shortcut table the story insists exists.
- *"the seal's tokens must be produced strictly in order, most significant first"* — encoding a real message becomes an order-free XOR-reduction over independent per-house contributions, so it can be split across threads.

Because RS systematic encoding is GF(256)-linear in the data bytes, `parity = A·data` for a fixed `k×16` matrix `A` that depends only on `(k, nsym, generator)`. Column `i` of `A` is exactly the parity produced by a lone "1" stone in house `i` — the "distinct clutch" the native describes. That column can be obtained by clocking the same LFSR feedback register backward from the last house to the first (one zero-input step per house — the "tightening spiral"), which costs `O(k·nsym)` **once**, after which every subsequent call is a pure, order-independent XOR-reduction that can be handed to multiple "rivers" (OpenMP threads).

# ARTIFACT

```c
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
```

I verified the backward zero-input recurrence by hand against the reference for `(k,nsym)=(1,1),(2,1),(2,2)`: derived coefficients reproduce the reference's output byte-for-byte in every case I checked, so I'm treating exactness as established by construction (linearity of the reference over GF(256)), not by hope.

PREDICTION: speedup_vs_naive = 15

# MEASUREMENT
Not performed — no tools are available in this session (per the environment note), so `rs_encode_bench` could not actually be invoked here. I am not fabricating a number for this section: the prediction above is a reasoned estimate (≈4–8× from replacing the naive bit-loop `gf_mul` with two log/antilog lookups, times a further multi-core factor from the now order-free XOR-reduction, discounted for the fact that this is a memory-streaming, XOR-bound kernel where bandwidth caps the parallel gain). The actual number should come from the pipeline running `rs_encode_bench` against this artifact.

# VERDICT
Reasoning-only verdict, pending real measurement: the construction is a genuine structural departure from the "known way" — it replaces the sequential shift-reduce LFSR (which has an unavoidable byte-to-byte dependency chain, the same one the story's "most significant first" phrase names) with a one-time precomputed per-house coefficient table plus an order-free, parallelizable XOR-reduction, exactly as SEED 1 and SEED 2 describe. Its win is conditional and I say so plainly: it pays off only when the harness calls the kernel repeatedly at the same `(k, n)` (so the O(k·16) table-build amortizes); on a single cold call it does strictly more work than the naive encoder and would measure *slower*, not faster. That trade-off is the honest, falsifiable claim this design makes.