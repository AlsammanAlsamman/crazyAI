## MAPPING

### SEED 1 — "the stone never resets between marks; every fold carries the callus of all folds before"

| world object | problem object |
|---|---|
| the pile of marks in their given order | the byte buffer `data[0..len-1]`, read in order |
| one fold per mark, "until the stone has turned once for every mark" | exactly **one** mixing step per input word — no finishing rounds per byte |
| the callus that is never washed clean | the accumulator is never re-initialised mid-buffer |
| "centuries of marks fold down just the same as a handful" | the per-byte cost is constant in `len` |

**Assumption broken:** *"more mixing rounds always means better mixing."* The native folds **once** per mark and relies on the callus (carried state) to do the work that extra rounds would otherwise do.

### SEED 2 — "each mark is pressed into the stone's already-turned position, not onto a clean face"

| world object | problem object |
|---|---|
| the quarter-turn that happens *between* presses | a cross-lane permutation of the state between absorb steps |
| "already-turned position" | the state value the receiving face carries from when it was last seated — i.e. **one revolution ago**, not one step ago |
| "the stone's memory bends how deep this one goes" | the state is an *operand of the multiply*, not just a destination of an add |

**Assumption broken:** *"each byte must be mixed into the running state before the next byte is read."* Because the face receiving mark *i* carries its callus from a *previous revolution*, the press depth depends on a **lagged** state. Loads may run ahead of the presses they influence — the serial chain is broken without any byte escaping influence.

### SEED 3 — "only the final seated number leaves the desk; every intermediate turn and all residue is swept off and discarded"

| world object | problem object |
|---|---|
| the bone-coloured **die-stone** (one object, four faces) | one wide state: **4 × 64 bits = 256 bits**, one `__m256i` — a single stone, not four stones |
| "a flat thing about to be invested into three" | a flat byte stream lifted into a wide, multi-lane register |
| the wooden numbered **keeps** at the desk's edge | the final 64-bit read-off: lane-fold + strong finaliser |
| groove-dust, intermediate turns, chalk residue — swept away | 192 of the 256 state bits are **never emitted**; no intermediate value is observable |
| "only the last seated number ever leaves the desk" | output width ≪ state width (truncation / squeeze) |

**Assumption broken:** *"the state is a single accumulator updated in place, one value."* If nothing but the final projection is ever observed, the state during the fold is **not obliged to be the token**. It may be four times wider than the token.

## CHOSEN SEED

**SEED 3.** It is the only one of the three that breaks the preferred assumption ("the state is a single accumulator updated in place, one value"), and it is the most literal: a *die-stone* is physically a multi-faced object, read off at the end against *numbered keeps* (plural), with everything else swept away. SEED 1 and SEED 2 are retained as constraints on *how* the wide stone is folded (one fold per mark, each press landing on an already-turned face), not as the governing idea.

## ASSUMPTION BROKEN

> *the state is a single accumulator updated in place, one value*

Replaced by: **the state is a 256-bit four-faced stone; the 64-bit token is a projection of it taken once, at the end.** Secondary break (from SEED 2): **the press depth depends on the state from one revolution ago**, so loads need not wait on the previous press — the dependency chain per 32 bytes collapses to a single 64-bit add latency.

Where this mechanism lands is, deliberately, a **validated real technique**: a wide multi-lane accumulator absorbed with `vpmuludq` + cross-lane data swap, folded down by 128-bit multiply merges and a `fmix64` finaliser — i.e. XXH3's `accumulate_512` / `mergeAccs` shape and Murmur3's finaliser. I let the metaphor arrive there rather than inventing a fresh mixer: the quarter-turned callus is the one genuinely added piece.

**Two regimes, recognised at runtime** (the native's own "centuries of marks … or a handful"):
- `len < 64` — *a handful*: the stone is seated once; a plain serial 8-byte press with the same keeps-finaliser. This is the guard that answers the wide path's own risk (register setup + 4-lane merge is pure overhead below ~64 bytes).
- `len >= 64` — *centuries*: the turning wide stone.
- A third, compile-time regime: no AVX2 → a literal scalar 4-lane transcription of the same stone.

No thread parallelism: the metaphor's unit of work is one 32-byte seating, far too small, and OpenMP fork cost would dwarf every benchmark size here. Vectorisation only, as instructed.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ------- the chalk-bank groove: 16 odd, high-entropy keeps ------- */
static const uint64_t GROOVE[16] = {
  0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL,
  0x85EBCA77C2B2AE63ULL, 0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL, 0x9FB21C651E98DF25ULL,
  0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL, 0x8EBC6AF09C88C6E3ULL,
  0x589965CC75374CC3ULL, 0x1D8E4E27C47D124FULL, 0xEB44ACCAB455D165ULL, 0x4D5A2DA51DE1AA47ULL
};

static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint32_t rd32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return v; }
static inline uint64_t rotl64(uint64_t x, int r){ return (x << r) | (x >> (64 - r)); }

/* fold a 128-bit product back onto one keep */
static inline uint64_t fold128(uint64_t a, uint64_t b){
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
}

/* reading the stone's seated position against the wooden numbered keeps
   (Murmur3 fmix64 - validated, bias well under 1%) */
static inline uint64_t keeps(uint64_t x){
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 32;
    return x;
}

/* ================= regime A: "a handful" (len < 64) =================
   the stone is seated once; a plain serial press, no wide setup.      */
static uint64_t handful(const unsigned char *d, size_t len){
    uint64_t h = GROOVE[0] ^ ((uint64_t)len * GROOVE[1]);
    if (len >= 8){
        size_t i = 0;
        for (; i + 8 <= len; i += 8){
            h ^= fold128(rd64(d + i) ^ GROOVE[2], h + GROOVE[3]);
            h  = rotl64(h, 27) + GROOVE[4];
        }
        if (i != len)                                   /* overlapping last 8 */
            h ^= fold128(rd64(d + len - 8) ^ GROOVE[5], h + GROOVE[6]);
    } else if (len >= 4){
        uint64_t a = (uint64_t)rd32(d), b = (uint64_t)rd32(d + len - 4);
        h ^= fold128((a + (b << 32)) ^ GROOVE[2], h + GROOVE[3]);
    } else if (len > 0){
        uint64_t k = ((uint64_t)d[0] << 16) | ((uint64_t)d[len >> 1] << 8) | (uint64_t)d[len - 1];
        h ^= fold128(k ^ GROOVE[2], h + GROOVE[3]);
    }
    return keeps(h);
}

#if defined(__AVX2__)
/* one seating of the four-faced stone: 32 marks pressed into the
   already-turned position (callus), one fold, nothing washed clean. */
static inline __m256i seat(__m256i acc, const unsigned char *p,
                           __m256i key, __m256i callus){
    __m256i d    = _mm256_loadu_si256((const __m256i *)p);
    __m256i dk   = _mm256_xor_si256(_mm256_xor_si256(d, key), callus);
    __m256i prod = _mm256_mul_epu32(dk, _mm256_srli_epi64(dk, 32)); /* lo32*hi32 */
    __m256i swap = _mm256_shuffle_epi32(d, 0x4E);   /* face i takes mark i^1 */
    /* one add-latency on the acc chain: prod+swap are off the chain */
    return _mm256_add_epi64(acc, _mm256_add_epi64(prod, swap));
}
#else
static inline void seat_s(uint64_t *v, const unsigned char *p,
                          const uint64_t *key, const uint64_t *callus){
    uint64_t d0 = rd64(p), d1 = rd64(p+8), d2 = rd64(p+16), d3 = rd64(p+24);
    uint64_t a0 = d0 ^ key[0] ^ callus[0], a1 = d1 ^ key[1] ^ callus[1];
    uint64_t a2 = d2 ^ key[2] ^ callus[2], a3 = d3 ^ key[3] ^ callus[3];
    v[0] += (uint64_t)(uint32_t)a0 * (uint32_t)(a0 >> 32) + d1;
    v[1] += (uint64_t)(uint32_t)a1 * (uint32_t)(a1 >> 32) + d0;
    v[2] += (uint64_t)(uint32_t)a2 * (uint32_t)(a2 >> 32) + d3;
    v[3] += (uint64_t)(uint32_t)a3 * (uint32_t)(a3 >> 32) + d2;
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* --- regime check: a handful folds on the simple path --- */
    if (len < 64) return handful(data, len);

    const unsigned char *p   = data;
    const unsigned char *end = data + len;
    uint64_t lane[4];

#if defined(__AVX2__)
    const __m256i k0 = _mm256_loadu_si256((const __m256i *)(GROOVE + 0));
    const __m256i k1 = _mm256_loadu_si256((const __m256i *)(GROOVE + 4));
    const __m256i k2 = _mm256_loadu_si256((const __m256i *)(GROOVE + 8));
    const __m256i k3 = _mm256_loadu_si256((const __m256i *)(GROOVE + 12));

    __m256i acc = _mm256_set_epi64x((long long)GROOVE[3], (long long)GROOVE[2],
                                    (long long)GROOVE[1], (long long)GROOVE[0]);
    __m256i callus = acc, callus_next = acc;

    /* one full revolution of the stone = four quarter-turns = 128 marks */
    while ((size_t)(end - p) >= 128){
        __m256i c = callus;
        acc = seat(acc, p +   0, k0, c);
        acc = seat(acc, p +  32, k1, c);
        acc = seat(acc, p +  64, k2, c);
        acc = seat(acc, p +  96, k3, c);
        callus      = callus_next;                              /* lagged callus */
        callus_next = _mm256_permute4x64_epi64(acc, 0x93);      /* the quarter turn */
        p += 128;
    }
    {
        __m256i c = callus;
        if ((size_t)(end - p) >= 32){ acc = seat(acc, p, k0, c); p += 32; }
        if ((size_t)(end - p) >= 32){ acc = seat(acc, p, k1, c); p += 32; }
        if ((size_t)(end - p) >= 32){ acc = seat(acc, p, k2, c); p += 32; }
        if (p != end)                 acc = seat(acc, end - 32, k3, c); /* overlap */
    }
    _mm256_storeu_si256((__m256i *)lane, acc);
#else
    uint64_t v[4]  = { GROOVE[0], GROOVE[1], GROOVE[2], GROOVE[3] };
    uint64_t c[4]  = { GROOVE[0], GROOVE[1], GROOVE[2], GROOVE[3] };
    uint64_t c2[4] = { GROOVE[0], GROOVE[1], GROOVE[2], GROOVE[3] };

    while ((size_t)(end - p) >= 128){
        seat_s(v, p +   0, GROOVE +  0, c);
        seat_s(v, p +  32, GROOVE +  4, c);
        seat_s(v, p +  64, GROOVE +  8, c);
        seat_s(v, p +  96, GROOVE + 12, c);
        c[0]=c2[0]; c[1]=c2[1]; c[2]=c2[2]; c[3]=c2[3];
        c2[0]=v[3]; c2[1]=v[0]; c2[2]=v[1]; c2[3]=v[2];   /* the quarter turn */
        p += 128;
    }
    if ((size_t)(end - p) >= 32){ seat_s(v, p, GROOVE +  0, c); p += 32; }
    if ((size_t)(end - p) >= 32){ seat_s(v, p, GROOVE +  4, c); p += 32; }
    if ((size_t)(end - p) >= 32){ seat_s(v, p, GROOVE +  8, c); p += 32; }
    if (p != end)                 seat_s(v, end - 32, GROOVE + 12, c);
    lane[0]=v[0]; lane[1]=v[1]; lane[2]=v[2]; lane[3]=v[3];
#endif

    /* lift the stone: only the final seated number leaves the desk.
       192 of the 256 state bits are swept off and discarded. */
    uint64_t h = (uint64_t)len * GROOVE[5];
    h += fold128(lane[0] ^ GROOVE[6],  lane[1] ^ GROOVE[7]);
    h += fold128(lane[2] ^ GROOVE[8],  lane[3] ^ GROOVE[9]);
    return keeps(h);
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 14

Stated before any measurement. Reasoning: FNV-1a's cost is one 64-bit multiply **latency** per byte (~3–4 cycles/byte, ≈0.8–1.0 GB/s). The turning stone pays one `vpmuludq` + two adds per **32 bytes**, with the accumulator chain cut to a single add latency (the callus is two revolutions stale, so the multiply is off the chain); front-end-bound at ≈2.2 cycles / 32 bytes ⇒ ≈14 bytes/cycle ⇒ ≈40 GB/s on L1/L2-resident data, falling to DRAM bandwidth (~15–25 GB/s) for multi-MB buffers. That is 20–40× on large buffers; a benchmark that averages in sub-64-byte sizes (where the `handful` path is only ~2–4× FNV) pulls the reported figure down, hence 14 rather than 30. Avalanche: predicted essentially ideal (score ≈0.5 bit-flip rate / near-zero bias) — every input bit reaches the accumulator twice, once through `lo32×hi32` and once through the raw `swap` addend, and the exit path is a 64×64→128 fold plus `fmix64`, both validated diffusers.

## MEASUREMENT

**Not measured.** No tools are available in this session — `hash_bench` and `hash_contract` could not be called, so the prediction above stands untested and must be treated as an estimate, not a result. I am reporting that plainly rather than dressing up the estimate as a measurement.

What the pipeline should report if the mapping is right: throughput ≳10 GB/s at ≥4 KB, avalanche within noise of 0.50. What would falsify it: (a) avalanche materially off 0.50 — most likely culprit is the additive accumulator's carry-only upward diffusion, fixable by adding an XXH3-style scramble at revolution boundaries; (b) throughput under ~5 GB/s — most likely the `permute4x64` having crept back onto the critical path, i.e. the two-revolution lag not surviving the compiler's scheduling, checkable in the asm.

## VERDICT

The wide stone is the real content of the translation, and it is the part I am most confident in: **state width decoupled from token width** is exactly what SEED 3 asserts and exactly what the known way forbids, and it is what converts one multiply-per-byte into one multiply-per-32-bytes. The quarter-turned, never-reset callus is the one place I honoured the metaphor at a measurable cost — a strict reading of SEED 2 ("the memory of the **last** press") would put the accumulator inside the multiply's operands and force a ~7-cycle chain per 32 bytes, roughly a 3× throughput loss. I weakened "last press" to "the press one-to-two revolutions ago." The callus is still never washed clean and every mark still bends every later mark, but its influence now arrives with a lag. That is a declared liberty, not a silent one.

Two honest caveats, both addressed in the artifact rather than left as caveats:

1. **The wide path is a loss on short buffers** (register setup plus a four-lane merge for 20 bytes of input). Guarded: `len < 64` falls back to the serial `handful` path, which is the simpler path the known way already assumes.
2. **No AVX2 ⇒ no stone.** Guarded at compile time by a literal scalar four-lane transcription with the same structure; it still beats FNV (4 multiplies per 32 bytes vs 32), just by less.

I did not add OpenMP. The metaphor's unit of work is a 32-byte seating; threading it would be overhead at every size the benchmark plausibly uses, and the instruction to prefer vectorisation first is the correct call here.

Where I was not original, deliberately: the absorb and merge are XXH3's, the finaliser is Murmur3's. The native's mechanism *arrived* at them, and a validated mixer beats a novel one — so I let it, and confined the novelty to the lagged callus.