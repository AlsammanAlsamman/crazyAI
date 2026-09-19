# MAPPING (world-object → problem-object, per SEED)

**SEED 1 — "cancel on the jigsaw sun, keep the leftover red as seal"**

| World object | Problem object |
|---|---|
| token (coin-word) | one message byte `data[i]` |
| tollbooth market / coin alphabet | the 256 possible byte values (GF(2⁸) elements) |
| weighing stall / moon-dial, "needle settles" | a value that is already fully resolved, not something still mid-computation |
| the sun's sky | a fixed lookup structure built once, not rebuilt per token |
| jigsaw cells (black-outlined, red-filled) | the `nsym`=16 byte slots of the LFSR remainder register |
| "press weights into cells two and two, equal cancel" | XOR-accumulation: `reg[j] ^= table_row[j]` (since `a^a=0` in GF(2)) |
| "fold the sky, halving each fold, until a short row remains" | repeatedly XOR-folding the table row into the register across all *k* tokens, converging to just the final `nsym` "stubborn" bytes |

Breaks: *"each token combination must be computed from scratch by shifting and reducing"* and *"the strange arithmetic has no shortcut table, only the shift-reduce rule"* — replace per-coefficient `gf_mul` shift-reduce with one precomputed 256×16 XOR table, hit once per token.

**SEED 2 — "leap under gravity to the island of Conclusions"**

| World object | Problem object |
|---|---|
| short row of leftover cells | the final `nsym`-byte parity register |
| the leap's arc length | the fixed MSB-first byte order the parity bytes are written out in |
| island of Conclusions | the output `parity[]` buffer |

Mostly reaffirms *"seal's tokens produced strictly in order, most significant first"* — it's about deterministic output serialization, not a computational shortcut. Weakest seed for a new kernel idea.

**SEED 3 — "poured through the cupbearer's cup, cancelled weights down the coin-slot"**

| World object | Problem object |
|---|---|
| cupbearer's cup, "only door out" | the single `memcpy(parity, msg+k, nsym)` at the end |
| coin-slot, "gone for good" | the scratch `msg` working buffer, `free()`d after use |

Mostly about memory packaging/cleanup, not arithmetic — touches *"every token folded individually"* only by affirmation.

# CHOSEN SEED
SEED 1 — it is the only one that proposes a genuinely different arithmetic shortcut (a precomputed cancel-table) rather than just describing I/O ordering or buffer hygiene, and it is maximally different from the stated "known way" (log/antilog multiply-and-reduce): it eliminates GF multiplication from the hot loop entirely.

# ASSUMPTION BROKEN
"Each token combination must be computed from scratch by shifting and reducing" and "the strange arithmetic has no shortcut table, only the shift-reduce rule." Instead: build one 256×16 table of *already-reduced* contributions once, then encoding a token is a single 128-bit XOR (the jigsaw fold), never a multiply.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

static unsigned char gf_mul(unsigned char a, unsigned char b) {
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
static unsigned char gf_pow(unsigned char a, int e) { unsigned char r = 1; for (int i = 0; i < e; i++) r = gf_mul(r, a); return r; }
static void gen_poly(int nsym, unsigned char *gen) {
    gen[0] = 1;
    for (int i = 1; i <= nsym; i++) gen[i] = 0;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = gf_pow(2, i);
        unsigned char newgen[64];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) { newgen[j] ^= gen[j]; newgen[j + 1] ^= gf_mul(gen[j], root); }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }
}

/* the sun's jigsaw sky: 256 cell-rows x nsym slots, folded once, reused forever */
static unsigned char g_tbl[256][32];
static int g_tbl_nsym = -1;

static void build_table(int nsym) {
    unsigned char gen[64];
    gen_poly(nsym, gen);
    for (int b = 0; b < 256; b++) {
        unsigned char bb = (unsigned char)b;
        for (int j = 1; j <= nsym; j++) g_tbl[b][j - 1] = gf_mul(gen[j], bb);
    }
    g_tbl_nsym = nsym;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    if (nsym != g_tbl_nsym) build_table(nsym);

    unsigned char *msg = (unsigned char *)calloc((size_t)(k + nsym), 1);
    memcpy(msg, data, (size_t)k);

    if (nsym == 16) {
        for (int i = 0; i < k; i++) {
            unsigned char coef = msg[i];
            if (coef) {
                unsigned char *dst = msg + i + 1;
                __m128i d = _mm_loadu_si128((const __m128i *)dst);
                __m128i t = _mm_loadu_si128((const __m128i *)g_tbl[coef]);
                _mm_storeu_si128((__m128i *)dst, _mm_xor_si128(d, t));
            }
        }
    } else {
        for (int i = 0; i < k; i++) {
            unsigned char coef = msg[i];
            if (coef) {
                unsigned char *dst = msg + i + 1;
                const unsigned char *row = g_tbl[coef];
                for (int j = 0; j < nsym; j++) dst[j] ^= row[j];
            }
        }
    }

    memcpy(parity, msg + k, (size_t)nsym);
    free(msg);
}
```

Correctness check: this is the exact same polynomial long-division the reference performs (`msg[i+j] ^= gf_mul(gen[j],coef)` for `j=0..nsym`); `gen[0]=1` always makes `msg[i]` cancel to zero and is never read again, so it's safe to skip. The table only precomputes `gf_mul(gen[1..nsym], coef)`, i.e. exactly the `j=1..nsym` terms, applied as one 128-bit XOR since `nsym=16` fixed by the harness matches an XMM register exactly (the "jigsaw sky" is literally 16 cells wide).

PREDICTION: speedup_vs_naive = 20

# MEASUREMENT
No tools were available in this session (per the environment note), so `rs_encode_bench` could not actually be invoked here — the prediction above is stated honestly as a prediction only, not a measured result. It should be run by the pipeline against `rs_encode_contract` before the number above is trusted.

# VERDICT
Unverified in this session. The design is a literal, non-trivial translation of SEED 1 (precomputed cancel-table replacing shift-reduce, table built once and folded per token) and is structurally different from the stated "known way" (log/antilog LFSR), since it removes GF multiplication from the hot loop entirely in favor of a single 128-bit XOR per token. Expect a large (double-digit) speedup over the naive per-coefficient `gf_mul` reference, but this claim is not yet backed by an actual `rs_encode_bench` run — that must be done before treating the 20× figure as anything but a hypothesis.