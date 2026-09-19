## MAPPING (per SEED)

| World object | Problem object | Assumption it breaks |
|---|---|---|
| **SEED 1 — eye-fold**: every second ash-pit gathered into one running pit at the near end | Even-indexed message bytes `data[0,2,4,...]` accumulated into a running 16-byte accumulator `acc_eye`, independent of the odd stream | "every message token must be folded into the seal individually, one at a time" — here two disjoint streams accumulate in parallel with no shared dependency chain |
| **SEED 2 — ankle-fold**: the pits skipped by the first walk gathered into a second running pit at the far end | Odd-indexed bytes `data[1,3,5,...]` accumulated into a second independent accumulator `acc_ankle` | Same as above, plus "tokens must be produced strictly in order, most significant first" — order between the two streams (and within each, relative to each other) doesn't matter, only position→basis-vector correctness does |
| **SEED 3 — a mark upon its own likeness levels to bare earth** | XOR: `parity[j] = acc_eye[j] ^ acc_ankle[j]`; and more deeply, the whole encoder is GF(256)-linear, so contribution of `data[i]` can be precomputed once as a fixed vector `R_i` and just XOR-folded in at call time | "each token combination must be computed from scratch by shifting and reducing" and "the strange arithmetic has no shortcut table" — the shift-reduce work is done **once**, off the hot path, turning every call into pure lookups + XORs |

## CHOSEN SEED
SEED 1 (eye-fold), with SEED 2 as its mirror and SEED 3 as the fold operator that makes the whole thing legal. SEED 1 is the most structurally different from the known way: the known way is a *single* serial LFSR register with one unbroken dependency chain of length `k`. SEED 1 literally forces the message into two interleaved, independently-foldable streams — that's not a tabulation trick on top of the LFSR, it's a different shape of computation.

## ASSUMPTION BROKEN
"every message token must be folded into the seal individually, one at a time" (serial single-accumulator LFSR) — replaced by two independent accumulators fed from precomputed per-position basis vectors, fused with a final XOR fold. As a consequence "each combination computed from scratch" and "no shortcut table" also fall: the generator-polynomial reduction is done once per `(k,n)` and cached (scratch memory), not redone every call.

**Literal mapping of objects:**
- token = one message byte `data[i]`, a GF(256) element ("mark pinched from a palm-print").
- the strange arithmetic = GF(256) multiply/add (mod 0x11D, primitive 2), done here as log/antilog lookups for the per-element scaling, plus XOR for addition.
- seal = the `n-k = 16` parity bytes; the threshold pit = final XOR of `acc_eye` and `acc_ankle`.
- "fold what's already there" = the precomputed basis table `R[i][0..15] = x^{(k-1-i)+16} mod g(x)`, built once via the *same* shift-reduce recurrence as the reference encoder, but amortized across calls instead of paid per call.

## ARTIFACT

```c
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
```

Correctness reasoning: the reference encoder is the remainder of `data(x)·x^16 mod g(x)`, and this remainder operation is GF(256)-linear in `data`. So `parity = Σᵢ data[i]·Rᵢ` where `Rᵢ = x^{(k-1-i)+16} mod g(x)`, computed once via the same shift-reduce recurrence the reference uses internally (`x_shift`), then reused on every call — nothing about the arithmetic changes, only *when* the shift-reduce work happens.

PREDICTION: speedup_vs_naive = 15

## MEASUREMENT
Not measured in this session — `rs_encode_bench` / `rs_encode_contract` are not in this session's available toolset (explicitly confirmed no tools are attached here). The prediction above stands to be verified by the external pipeline when it compiles and runs this kernel against the naive reference. I am reporting this plainly rather than fabricating a number.

## VERDICT
Cannot honestly claim measured success or failure — only the reasoning is verified (linearity derivation double-checked against the reference's polynomial-long-division convention, index ordering, and edge cases k≤0/odd k). The structural claim worth flagging for whoever runs the harness: the expected win is not from a smarter GF(256) multiply (log/antilog is the same "known way" shortcut) but from turning the encoder's single k-long serial dependency chain into two independent chains plus a cached, amortized basis table — if `rs_encode_bench` calls `kernel()` once per message rather than repeatedly at fixed k, the basis-cache amortization assumption fails and the real speedup will be much smaller (first-call cost ≈ same order as the naive encoder). That caching assumption is the single biggest risk to the predicted 15×, and should be the first thing checked against actual measurement.