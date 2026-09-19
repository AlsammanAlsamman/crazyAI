## MAPPING

| World object | Problem object | Assumption (from the list) it targets |
|---|---|---|
| **SEED 1** — blank red‑on‑black cell pressed against each mote, matching red erases to black | | |
| dust‑mote, itself already a red‑on‑black jigsaw cell | one message byte `data[i]` | — |
| garden‑path row, one cell per stone | the sequential index `i = 0..k-1` over the message | — |
| blank cell "of the same cloth" carried down the row | a **precomputed 16‑byte answer vector** `TAP_VEC[coef]` — the complete effect that byte‑value `coef` has on all 16 seal lanes at once | breaks **#3** ("a combination's answer cannot be known without shift‑reduce") — the answer for every possible byte value is already sitting in the table |
| "pressing… matching red erases to black… self‑cancelling arithmetic" | XOR — GF(2) addition, used both to fold `TAP_VEC[coef]` into the 16‑byte seal register and to read out the next feedback byte | is literally GF(2^8) addition; the *whole* fold happens as one bulk XOR of 16 lanes, not lane‑by‑lane | 
| one pressing per mote, all 16 "jigsaw facets" moved together | one SIMD step folds all 16 register lanes simultaneously | breaks **#4** at the sub‑byte granularity — no more looping over `j=0..nsym` doing one multiply‑XOR at a time |
| **SEED 2** — three trailing cells from walking all/even/odd stones | | |
| walking the row three times at different strides | splitting the k message bytes into interleaved sub‑streams, encoding each independently, XOR‑merging the partial registers at the end | would break **#2** (strict MSB‑first order) and **#4** (fold one at a time) |
| three trailing cells, later used to recover a scoured mote | redundancy for erasure *repair* (not required by the target problem, which is encode‑only) | n/a for this contract |
| *(evaluated and rejected)* | Because the LFSR recurrence is a genuine linear **recurrence with state carried forward** (`S_i` depends on `S_{i-1}`), splitting even/odd positions and running each as an ordinary step‑by‑step LFSR does **not** save work — each sub‑pass would still have to advance the register once per skipped byte too, doubling total work instead of halving it. A real win would need precomputed `L²` transition matrices, which for `nsym=16` cost 16×16 multiplies per two bytes — worse than 2×16 scalar taps. Honest verdict: literal, but not computationally beneficial here. |
| **SEED 3** — rim‑dust settles to fix the seal before any pressing is trusted | | |
| "until settling happens the seal can be pressed wrong, so I touch nothing" | build `gen[]` and `TAP_VEC[256][16]` **completely** before folding any real byte, guarded by a "built" flag | breaks **#5** ("no shortcut table, only shift‑reduce") |
| "the garden path itself remembers the order… the only ledger I trust" | the loop index `i` implicitly tracks position; no separate order/ledger array is stored | supports #1 |

## CHOSEN SEED
**SEED 1** — it is the most literal (one mote, one pressing, one already‑complete answer) and structurally the most different from the "known way": the known way still does `nsym` separate scalar log/antilog lookups per byte (two lookups + mod‑255 add, per tap, in a loop). SEED 1's reading collapses that entire per‑tap loop into **one single 16‑lane vector lookup + one XOR**, folding an entire mote in one press instead of nsym presses.

## ASSUMPTION BROKEN
Primarily **#3** ("a combination's answer cannot be known without performing the shift‑reduce steps") and, as a consequence, **#4** at sub‑byte granularity (no more looping tap‑by‑tap). `nsym=16` happens to equal one SSE/XMM register exactly, so "the whole jigsaw cell in one press" is literally one `__m128i` load + XOR.

Derivation note (had to be re‑checked by hand, not trusted from memory): the register form of the reference's systematic division is **not** "XOR data[i] into the low lane", it's "new data enters at the *top* of the 16‑byte sliding window while the *bottom* lane leaves as feedback" — verified by hand for `nsym=2, k=1,2` against the reference array algorithm before trusting it.

## ARTIFACT

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

/* TAP_VEC[c][m] = gf_mul(gen[m+1], c) for m = 0..15 (nsym fixed at 16 by the harness).
 * Each row is the complete, once-and-for-all "pressing" of one possible feedback
 * byte value into all 16 seal lanes at once -- no shift-and-reduce is ever done
 * while folding a real message byte; the answer already exists in the row.
 * Built once (rim-waterfall "dust settling"), then trusted for every later press. */
static unsigned char TAP_VEC[256][16] __attribute__((aligned(16)));
static int g_built = 0;
static int g_built_nsym = -1;

static void build_tap_vectors(int nsym) {
    unsigned char gen[64];
    gen_poly(nsym, gen);
    for (int c = 0; c < 256; c++) {
        unsigned char cc = (unsigned char)c;
        for (int m = 0; m < 16; m++)
            TAP_VEC[c][m] = (m < nsym) ? gf_mul(gen[m + 1], cc) : 0;
    }
    g_built = 1;
    g_built_nsym = nsym;
}

void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;

    if (nsym == 16) {
        if (!g_built || g_built_nsym != nsym) build_tap_vectors(nsym);

        /* Load the seal window: the first 16 message bytes (zero-padded if k<16).
           lane m == byte offset m, low byte = lane 0, matching x86 little-endian. */
        unsigned char win[16];
        for (int m = 0; m < 16; m++) win[m] = (m < k) ? data[m] : 0;
        __m128i reg = _mm_loadu_si128((const __m128i *)win);

        int i = 0;
        int main_end = k - 16;   /* while i < main_end, data[i+16] is a real, in-bounds byte */
        for (; i < main_end; i++) {
            unsigned char coef   = (unsigned char)_mm_cvtsi128_si32(reg); /* lane 0 = low byte */
            unsigned char newtop = data[i + 16];
            __m128i shifted = _mm_srli_si128(reg, 1);                    /* lane m <- lane m+1, lane15<-0 */
            __m128i tap     = _mm_load_si128((const __m128i *)TAP_VEC[coef]);
            __m128i inj     = _mm_set_epi8((char)newtop, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
            reg = _mm_xor_si128(_mm_xor_si128(shifted, tap), inj);
        }
        for (; i < k; i++) {   /* tail: i+16 >= k, so the new-top byte is implicitly 0 */
            unsigned char coef = (unsigned char)_mm_cvtsi128_si32(reg);
            __m128i shifted = _mm_srli_si128(reg, 1);
            __m128i tap     = _mm_load_si128((const __m128i *)TAP_VEC[coef]);
            reg = _mm_xor_si128(shifted, tap);
        }
        _mm_storeu_si128((__m128i *)parity, reg);
        return;
    }

    /* generic scalar fallback for nsym != 16 (defensive only; harness fixes nsym=16) */
    {
        unsigned char gen[64];
        gen_poly(nsym, gen);
        unsigned char *win = (unsigned char *)calloc((size_t)nsym, 1);
        for (int m = 0; m < nsym && m < k; m++) win[m] = data[m];
        for (int i = 0; i < k; i++) {
            unsigned char coef   = win[0];
            unsigned char newtop = (i + nsym < k) ? data[i + nsym] : 0;
            for (int m = 0; m < nsym - 1; m++) win[m] = (unsigned char)(win[m + 1] ^ gf_mul(gen[m + 1], coef));
            win[nsym - 1] = (unsigned char)(newtop ^ gf_mul(gen[nsym], coef));
        }
        memcpy(parity, win, (size_t)nsym);
        free(win);
    }
}
```

PREDICTION: speedup_vs_naive = 25

## MEASUREMENT
No tools were available in this session (explicitly stated in the environment: *"No tools are available in this session… the pipeline will compile and measure it"*), so `rs_encode_bench` / `rs_encode_contract` could not actually be invoked here. I have not fabricated a number — the figure above is a stated **prediction only**, based on: naive path does `nsym+1 = 17` branchy 8‑iteration `gf_mul` calls per message byte (~130+ scalar ops with data‑dependent branches) plus scattered writes into a `k+nsym` buffer; the artifact does one `_mm_cvtsi128_si32`, one `_mm_srli_si128`, one aligned load, one or two `_mm_xor_si128` per byte (5–6 cheap, branch‑free vector ops), with the multiply/reduce work fully retired into a one‑time 4 KB table build. I expect the real ratio to land in the 15×–40× band; 25× is my point estimate, not a measured value.

## VERDICT
Correctness: re‑derived the register recurrence directly from the reference's array‑based algorithm (my first instinct — "coef = data[i] XOR reg[0], reg[nsym-1] = coef" — was **wrong**, since `gen[nsym]` is not generally 1 and new data enters at the *top* of the window, not via XOR at the bottom; caught and fixed by hand‑checking against the reference for `nsym=2, k=1` and `k=2` before trusting it). Speed: unmeasured in this session — the number above is a prediction to be checked by the external pipeline, not a claim of an achieved result. SEED 2's literal reading (three strides) was honestly evaluated and rejected: for a genuinely sequential linear recurrence with carried state, splitting into independent even/odd passes doesn't reduce work the way it first sounds like it should — that failure is reported here rather than quietly swapped for the textbook method.