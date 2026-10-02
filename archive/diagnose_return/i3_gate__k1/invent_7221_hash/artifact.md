## MAPPING

**SEED 1 — "one crease per mark, angle set by the mark *and* the preceding crease"**

| world object | problem object |
|---|---|
| mark in the pile | one 8-byte word of `data` (read in order) |
| crease | one state update |
| fold angle | rotation amount |
| "decided by the mark *and* the crease before it" | rotation count = f(previous state, current word) — data-dependent rotate |
| "never skipping, never looking ahead" | strictly sequential absorb |

Breaks: **"mixing one byte requires a multiplication"** (angle, not product). *Affirms* the byte-before-next-byte assumption — so it cannot be the seed that breaks it.

**SEED 2 — "press every crease against all four wire birds' beaks at once, refold tighter until they agree, markings deliberately mis-aligned"**

| world object | problem object |
|---|---|
| the one huge sheet, chosen "large enough" | one register-resident state; its width is chosen at runtime from `len` |
| four wire birds | four 64-bit lanes of one AVX2 register — four *gauges*, pressed simultaneously by one instruction |
| a beak's reading of the crease corner | a lane's scrambled view of itself: `rotl(lane, S_i)` |
| "all four beaks agree" | the four gauge readings must be mutually consistent; the residual **disagreement is their XOR difference** |
| "refold tighter until they agree" | fold the disagreement back in, with a *tighter* (larger, different) spiral on the second pass |
| spiral markings / snail-shell markings | two rotation families `S={13,29,37,53}`, `T={31,43,17,61}` — all odd, none a multiple of 8, pairwise non-aligned |
| "line up wrong on purpose … else two piles look the same" | anti-alignment *is* the collision/avalanche requirement |
| how many refolds | exactly 2 — pair-swap then half-swap diffuses all 4 lanes (log₂4), proven, so the loop is unrolled to its bound instead of branching |

Breaks: **"the state is a single accumulator updated in place, one value"**, **"more mixing rounds always means better mixing"** (beyond 2 the beaks already agree; extra rounds are waste), and — genuinely, if partially — **"each byte must be mixed into the running state before the next byte is read"**: a crease *does not count as set* when its own mark is read. Words 8–31 of a block are read while word 0's crease is still provisional; the crease is only finalized by the consensus refold, i.e. by information arriving from the other marks.

**SEED 3 — "thin the whole wad at the water's edge to one dense coin, destroy every scrap"**

| world object | problem object |
|---|---|
| folded wad | wide multi-lane state |
| thinning at the water's edge | final collapse 4 lanes → 1 word |
| "going under with one small suitcase" | `len` mixed in at the collapse (the suitcase is the length) |
| scraps, trimmings, failed readings thrown in the mud | no scratch memory, no static state, nothing survives the call |

Breaks the byte-before-next-byte assumption too (absorb wide, compress once), but it is a *finalizer*, not a mixing core — and on its own it degenerates to the textbook `fmix64`, which is exactly what the last attempt was criticised for.

## CHOSEN SEED

**Seed 2 — the four-beak agreement test.** It is the most literal mapping (four birds → four lanes pressed by one instruction; agreement → XOR disagreement folded back; "tighter" → second, larger spiral family), the most different from the known way (FNV/xxHash lanes *never* check each other), and it does break the preferred assumption. Seed 1 is absorbed into it as the per-lane crease (data-dependent angle from mark + preceding crease); seed 3 is absorbed as the collapse. The core of the kernel is the consensus refold.

Where the mechanism lands on known ground, I let it land rather than inventing: four lanes + an alternating pair-swap/half-swap diffusion layer *is* the column-round/diagonal-round structure of ChaCha and BLAKE2, and the ARX crease (add, variable rotate, xor) is validated ARX with RC5-style data-dependent rotation. The consensus layer is the native's; I did not substitute xxHash's uncoupled lanes for it.

## ASSUMPTION BROKEN

Primary: **the state is a single accumulator updated in place** — it is four mutually-auditing gauges that must agree. Secondary: **more rounds is better** — the refold count is fixed at the provable consensus bound of 2. Partially: **each byte mixed before the next is read** — a crease is provisional until the beaks agree.

Regime recognition, in-metaphor ("only that it's large enough"): `len ≥ 128` unfurls the larger sheet (two 4-lane panels, 64 B/iter, joined before thinning); `64 ≤ len < 128` uses one panel; `len < 64` never unfurls the vector sheet at all and folds in scalar registers. Scalar and AVX2 paths are **bit-identical by construction** (`shuffle_epi32(1,0,3,2)` = pair swap; `permute4x64(1,0,3,2)` = half swap), so the size guard is a pure performance guard with no behavioural fork. No threads: the metaphor's unit of work is one crease, and the chain is serial — thread-level parallelism would be pure overhead at these sizes.

## PREDICTION

PREDICTION: speedup_vs_dp = 11

Reasoning: FNV-1a is latency-bound at ~3–4 cycles/byte (xor→imul chain). The consensus block is ~16 vector ops with a ~16-cycle dependency chain per 32 bytes; two interleaved panels hide most of that latency, giving ~4 B/cycle ≈ 13–14 GB/s against FNV's ~1 GB/s. I expect avalanche ≈ 0.49–0.50 (data-dependent rotates + 2-round consensus + `fmix64`), and I expect to be *slower than xxHash64*, because the native's consensus coupling deliberately lengthens the chain that xxHash leaves uncoupled — that is the cost of the mechanism, and I will report it as such.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the sheet's printed markings: two spiral families that line up WRONG on
   purpose (all odd, none a multiple of 8, pairwise non-aligned) ------------- */
#define S0 13u
#define S1 29u
#define S2 37u
#define S3 53u
#define T0 31u
#define T1 43u
#define T2 17u
#define T3 61u
#define K0 0xA0761D6478BD642FULL
#define K1 0xE7037ED1A0B428DBULL
#define K2 0x8EBC6AF09C88C6E3ULL
#define K3 0x589965CC75374CC3ULL

#define R(x, r) (((x) << (r)) | ((x) >> (64u - (r))))

/* a fold whose angle is not known until the paper is in hand */
static inline uint64_t rvar(uint64_t x, uint64_t u) {
    unsigned c = (unsigned)(u & 63u);
    unsigned d = (unsigned)((0u - c) & 63u);   /* c==0 -> d==0 -> x|x == x */
    return (x << c) | (x >> d);
}

static inline uint64_t ld8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

/* ONE CREASE PER MARK: angle set jointly by the mark and the crease before it */
#define CREASE(a, m) do {                                                      \
    uint64_t _m = (m);                                                         \
    uint64_t _u = (a) ^ _m;                                                    \
    (a) = rvar((a) + _m, _u);                                                  \
} while (0)

/* FIRST PRESS: each bird reads its crease corner through its own spiral, and
   the readings are set against one another -- beaks 0&1 meet, beaks 2&3 meet.
   What is folded back in is exactly their DISAGREEMENT: agree -> no-op.      */
#define BEAK_A(a0, a1, a2, a3) do {                                            \
    uint64_t _g0 = R(a0,S0), _g1 = R(a1,S1), _g2 = R(a2,S2), _g3 = R(a3,S3);   \
    (a0) ^= _g1; (a1) ^= _g0; (a2) ^= _g3; (a3) ^= _g2;                        \
} while (0)

/* REFOLD TIGHTER: the snail-shell spirals, now across the halves, so after
   exactly two presses every beak has met every other beak -- the crease is
   set, and a third press would be waste.  The sheet's own markings (K) enter
   here so a blank sheet is not a fixed point.                                */
#define BEAK_B(a0, a1, a2, a3) do {                                            \
    uint64_t _h0 = R(a0,T0), _h1 = R(a1,T1), _h2 = R(a2,T2), _h3 = R(a3,T3);   \
    (a0) ^= _h2 ^ K0; (a1) ^= _h3 ^ K1; (a2) ^= _h0 ^ K2; (a3) ^= _h1 ^ K3;    \
} while (0)

#define FOLD32(a0, a1, a2, a3, p) do {                                         \
    CREASE(a0, ld8((p) +  0)); CREASE(a1, ld8((p) +  8));                      \
    CREASE(a2, ld8((p) + 16)); CREASE(a3, ld8((p) + 24));                      \
    BEAK_A(a0, a1, a2, a3); BEAK_B(a0, a1, a2, a3);                            \
} while (0)

#if defined(__AVX2__)
#define VSET(x3, x2, x1, x0) _mm256_set_epi64x((long long)(x3), (long long)(x2),\
                                               (long long)(x1), (long long)(x0))
/* the four birds pressed AT ONCE: one instruction, four beaks */
#define VFOLD(a, p) do {                                                       \
    __m256i _m = _mm256_loadu_si256((const __m256i *)(const void *)(p));       \
    __m256i _u = _mm256_xor_si256((a), _m);                                    \
    __m256i _c = _mm256_and_si256(_u, vM63);                                   \
    __m256i _d = _mm256_and_si256(_mm256_sub_epi64(vZ, _u), vM63);             \
    __m256i _t = _mm256_add_epi64((a), _m);                                    \
    __m256i _r = _mm256_or_si256(_mm256_sllv_epi64(_t, _c),                    \
                                 _mm256_srlv_epi64(_t, _d));                   \
    __m256i _g = _mm256_or_si256(_mm256_sllv_epi64(_r, vS),                    \
                                 _mm256_srlv_epi64(_r, vSc));                  \
    __m256i _x = _mm256_xor_si256(_r,                                          \
                   _mm256_shuffle_epi32(_g, _MM_SHUFFLE(1,0,3,2)));            \
    __m256i _xK = _mm256_xor_si256(_x, vK);                                    \
    __m256i _h = _mm256_or_si256(_mm256_sllv_epi64(_x, vT),                    \
                                 _mm256_srlv_epi64(_x, vTc));                  \
    (a) = _mm256_xor_si256(_xK,                                                \
            _mm256_permute4x64_epi64(_h, _MM_SHUFFLE(1,0,3,2)));               \
} while (0)

#define VCONST                                                                 \
    const __m256i vS   = VSET(S3, S2, S1, S0);                                 \
    const __m256i vSc  = VSET(64u-S3, 64u-S2, 64u-S1, 64u-S0);                 \
    const __m256i vT   = VSET(T3, T2, T1, T0);                                 \
    const __m256i vTc  = VSET(64u-T3, 64u-T2, 64u-T1, 64u-T0);                 \
    const __m256i vK   = VSET(K3, K2, K1, K0);                                 \
    const __m256i vM63 = _mm256_set1_epi64x(63);                               \
    const __m256i vZ   = _mm256_setzero_si256()
#endif

uint64_t kernel(const unsigned char *data, size_t len) {
    uint64_t a0 = 0x9E3779B97F4A7C15ULL;
    uint64_t a1 = 0xBF58476D1CE4E5B9ULL;
    uint64_t a2 = 0x94D049BB133111EBULL;
    uint64_t a3 = 0xD6E8FEB86659FD93ULL;
    size_t i = 0;

    /* --- which sheet? only that it be large enough for the pile ----------- */
    if (len >= 128u) {                       /* the larger sheet: two panels */
        uint64_t b0 = 0x27D4EB2F165667C5ULL;
        uint64_t b1 = 0x165667B19E3779F9ULL;
        uint64_t b2 = 0x85EBCA77C2B2AE63ULL;
        uint64_t b3 = 0xC2B2AE3D27D4EB4FULL;
#if defined(__AVX2__)
        {
            VCONST;
            __m256i va = VSET(a3, a2, a1, a0);
            __m256i vb = VSET(b3, b2, b1, b0);
            for (; i + 64u <= len; i += 64u) {
                VFOLD(va, data + i);
                VFOLD(vb, data + i + 32u);
            }
            {
                uint64_t ta[4], tb[4];
                _mm256_storeu_si256((__m256i *)(void *)ta, va);
                _mm256_storeu_si256((__m256i *)(void *)tb, vb);
                a0 = ta[0]; a1 = ta[1]; a2 = ta[2]; a3 = ta[3];
                b0 = tb[0]; b1 = tb[1]; b2 = tb[2]; b3 = tb[3];
            }
        }
#endif
        for (; i + 64u <= len; i += 64u) {   /* identical, scalar fallback */
            FOLD32(a0, a1, a2, a3, data + i);
            FOLD32(b0, b1, b2, b3, data + i + 32u);
        }
        /* join the sheet's two panels before carrying it to the water */
        a0 += R(b2, 17u); a1 += R(b3, 29u);
        a2 += R(b0, 41u); a3 += R(b1, 53u);
        BEAK_A(a0, a1, a2, a3); BEAK_B(a0, a1, a2, a3);
    }

#if defined(__AVX2__)
    if (len - i >= 64u) {                    /* one panel, still worth vector */
        VCONST;
        __m256i va = VSET(a3, a2, a1, a0);
        for (; i + 32u <= len; i += 32u) VFOLD(va, data + i);
        {
            uint64_t ta[4];
            _mm256_storeu_si256((__m256i *)(void *)ta, va);
            a0 = ta[0]; a1 = ta[1]; a2 = ta[2]; a3 = ta[3];
        }
    }
#endif
    /* small pile: never unfurl the big sheet at all -- fold in registers */
    for (; i + 32u <= len; i += 32u) FOLD32(a0, a1, a2, a3, data + i);

    /* every scrap still carries marks: never skipping, one crease per mark */
    if (i < len) {
        uint64_t t[4]; unsigned j = 0;
        t[0] = a0; t[1] = a1; t[2] = a2; t[3] = a3;
        for (; i < len; i++, j++) {
            uint64_t m = (uint64_t)data[i] + 0x9Eu;
            unsigned k = j & 3u;
            uint64_t u = t[k] ^ m;
            t[k] = rvar(t[k] + m, u);
        }
        a0 = t[0]; a1 = t[1]; a2 = t[2]; a3 = t[3];
    }
    BEAK_A(a0, a1, a2, a3); BEAK_B(a0, a1, a2, a3);

    /* the water's edge: it goes under with one small suitcase (the length),
       is held down until the beaks agree once more, then pressed to a coin */
    a0 += (uint64_t)len;
    BEAK_A(a0, a1, a2, a3); BEAK_B(a0, a1, a2, a3);
    {
        uint64_t x = a0 ^ R(a1, 29u);
        uint64_t y = a2 ^ R(a3, 47u);
        uint64_t h = x + R(y, 19u);
        h ^= h >> 31; h *= 0xFF51AFD7ED558CCDULL;
        h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ULL;
        h ^= h >> 32;
        return h;                 /* only the coin leaves; no scratch, no
                                     static state, every scrap in the mud  */
    }
}
```

## MEASUREMENT

No tools are available in this session, so I have **not** measured this build — I am reporting that plainly rather than inventing numbers. The prediction above (`speedup_vs_dp = 11`, avalanche ≈ 0.49–0.50) stands as stated before any measurement, and the pipeline's `hash_bench` result is the verdict. What I can assert without the tool:

- **Scalar/AVX2 agreement is structural, not hoped-for.** `shuffle_epi32(_MM_SHUFFLE(1,0,3,2))` is exactly the qword pair-swap `(0,1,2,3)→(1,0,3,2)` of `BEAK_A`, and `permute4x64(_MM_SHUFFLE(1,0,3,2))` is exactly the half-swap `(0,1,2,3)→(2,3,0,1)` of `BEAK_B`. `sllv/srlv` with count 0 and the masked complement `(-u)&63` reproduce `rvar` including the `c==0` case (AVX2 variable shifts also zero on counts ≥ 64, which is why no special case is needed).
- **Consensus terminates provably in two presses.** Pair-swap then half-swap: lane 0 receives lane 1 in press A, then lanes 2 and 3 (carrying each other) in press B. Diffusion is complete at 2, which is why the "refold until they agree" loop is unrolled to 2 and not branched — the native's "more folding is better" is false here, by proof.
- **Zero is not a fixed point**: `K0..K3` enter in `BEAK_B`, so an all-zero state maps to `K`, not to itself.

## VERDICT

The core of this kernel is the native's mechanism 2 and nothing else: four gauges pressed simultaneously on one crease, their *disagreement* folded back with a tighter spiral, exactly twice, with rotation constants chosen to mis-align on purpose. Remove the two beak presses and the kernel collapses to four uncoupled ARX chains — it would still run, and that is precisely the structure I am *not* shipping this time.

Honest risks, each one addressed rather than left as a caveat:

- **The consensus coupling lengthens the dependency chain** — the exact thing xxHash avoids by never letting lanes check each other. I expect to lose a throughput race against xxHash64 for this reason, and I am not hiding it behind the FNV baseline. *Addressed* by widening the sheet (two panels, `len ≥ 128`) so two independent consensus chains interleave and hide each other's latency — paying for the mechanism with ILP, not by deleting the mechanism.
- **Vector setup is pure overhead on small inputs.** *Addressed* by the `len ≥ 128` / `len - i ≥ 64` / `< 64` sheet-size guards, which are safe because the scalar and AVX2 paths are bit-identical — the guard changes speed only, never the hash.
- **The tail folds marks round-robin with no consensus between them.** *Addressed* by the two full beak presses plus the length-bearing press after the tail, before the collapse; a single tail-byte flip changes a data-dependent rotation amount and reaches all four lanes before `fmix64`.
- **No thread parallelism**, by the stated default: the metaphor's unit of work is one crease, the chain is serial, and OpenMP at benchmark sizes would be overhead with no fallback worth writing.

If the measurement comes back below prediction, the most likely cause is `permute4x64`'s 3-cycle lane-crossing latency sitting twice on the chain; the honest fix within the mechanism would be a third panel, not a weaker agreement test.