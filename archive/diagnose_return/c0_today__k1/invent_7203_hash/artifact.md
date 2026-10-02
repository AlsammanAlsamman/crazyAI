## MAPPING

**SEED 1 — the chained prisoners sharpening the blank pillar**

| world object | problem object |
|---|---|
| pile of marks, given order | the input byte buffer, read in order |
| a mark | one byte (entering the machine through "the knowing walls" = the cache lines, 64 B at a time) |
| the chained line of prisoners | a *vector* of state words `p0..p3`, chained, not one accumulator |
| "does not look at the mark, only feels how it changes the angle" | stage 0 never stores the byte; it folds it into a rotate-add and emits a derived scalar |
| "passes that angle, not the mark, to the next" | stage *k* consumes only stage *k-1*'s **previous-pass** output ⇒ a systolic pipeline, loop-carried latency ≈1 op per stage |
| "a mark dropped at the very start still trembles in the hand of the last one" | a byte takes 4 passes to reach `p3` and never leaves ⇒ unbounded-range position dependence |
| the blank pillar, never read whole, "keeps sharpening onward" | the state is never reset, never folded, never inspected mid-stream |
| **assumption broken** | **"the state is a single accumulator updated in place, one value"** (and partly "mixing one byte requires a multiplication" — the chain uses only rot/add/xor) |

**SEED 2 — the flood that never recedes, and the anchored wires**

| world object | problem object |
|---|---|
| the river that floods every solid argument | unsigned wraparound/overflow accumulation — the thing that destroys every "solid" algebraic invariant |
| "rises on each pass, does not recede until every mark has gone through" | the accumulator bank is **never merged mid-stream**; no intermediate value is usable |
| "a token pulled while the water is still up is worthless" | reading the lanes before the end gives a bad hash; finalize exactly once, at the end |
| tiny wire shapes **anchored** in the riverbed | 8 independent accumulator lanes, each anchored to its own distinct odd constant (`ANCHOR0..7`) |
| "they do not move themselves; the flood moves them" | lanes are driven only by data; lanes never talk to each other inside the loop ⇒ pure SIMD, zero cross-lane traffic |
| "bending each wire by exactly the angle the last prisoner ground in on that pass" | the chain's output `p3` is broadcast and added to every anchor each pass — the only cross-lane coupling, one vector add |
| "past where any traveler has ever fused its source" | the source is unfindable: one-way |
| **assumption broken** | **"each byte must be mixed into the running state before the next byte is read"** and **"the whole buffer must be read once, start to end, in order"** (8 lanes consume 8 disjoint positions in the same instruction) |

**SEED 3 — the silhouette, read once**

| world object | problem object |
|---|---|
| the shadow on the wall, never the thing itself | the output is a *projection* of the wide state, never the state |
| "which lean, which stand straight, **which cross another**" | the merge: `fold128(acc[i]^A, acc[i+1]^A')` — pairwise 64×64→128 products folded, i.e. wires literally crossing in pairs |
| "small and fixed" | 64 bits |
| "the shadows stop their chitter — the walls are still settling" | one finalization avalanche (xor-shift · multiply · xor-shift · multiply · xor-shift) run until settled — **and then stopped** |
| "drawn only once… I keep nothing but that shadow-shape" | one finalizer call, total, for the whole buffer; no per-byte rounds at all |
| "the prisoners forget the marks as soon as they're sharpened past; the floodwater drains unrecorded" | no scratch buffer, no second pass, O(1) memory |
| **assumption broken** | **"more mixing rounds always means better mixing"** — the per-byte cost is *one* 32×32 product inside a 4-wide vector and two adds; **all** strength is bought once, in a single silhouette read |

## CHOSEN SEED

**Seed 3 (the silhouette read once).** It is the only one of the three that breaks the preferred assumption, and it is the one that actually dictates the design: if the token is drawn only once, then nothing may be spent per byte, and the per-byte path collapses to the cheapest thing that still lets the flood reach every wire. Seeds 1 and 2 are not discarded — the native describes one mechanism, and the pit, the chain and the wires are all present in the artifact — but seed 3 is the load-bearing choice.

Honest note on step 4: this mechanism, followed literally, *is* a known validated technique. Wide anchored lanes, no cross-lane mixing in the loop, `lo32×hi32` per lane, pairwise `mul128`-fold merge, one final avalanche — that is the XXH3 `accumulate_512`/`mergeAccs` structure, which has passed SMHasher. I let the metaphor arrive there rather than inventing a fresh mixer, and the one thing the metaphor adds on top (the prisoner chain broadcasting an angle into the anchors every pass) is load-bearing, not decoration: it is what makes a fixed anchor table safe, because it makes stripe *position* and *all prior content* change the multiplier operand on every pass.

## ASSUMPTION BROKEN

"More mixing rounds always means better mixing." Zero mixing rounds are spent per byte. Secondary: the state is 8 lanes + a 4-stage chain, not one accumulator; bytes are not mixed in before the next is read.

## ARTIFACT

Regimes, encoded in-world (step 5): *"a pile too small to raise the water at all never reaches the wires; those few marks the nearest prisoner sharpens alone, and I read his angle"* → `len < 64` takes a scalar short path. *"the last marks left on the bank are led down again"* → the final partial cache line is re-read as an overlapping full stripe. *"a pit with no wide flood"* → scalar fallback with identical semantics when AVX2 is absent. No threads: the native has **one** pit and forbids reading the water while it is up, so splitting the buffer across pits would require exactly the mid-stream read he calls worthless — and hashing is bandwidth-bound anyway.

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- anchors: where each wire is fused into the riverbed (all odd) ---- */
#define A0 0x9E3779B185EBCA87ULL
#define A1 0xC2B2AE3D27D4EB4FULL
#define A2 0x165667B19E3779F9ULL
#define A3 0x85EBCA77C2B2AE63ULL
#define A4 0x27D4EB2F165667C5ULL
#define A5 0xA0761D6478BD642FULL
#define A6 0xE7037ED1A0B428DBULL
#define A7 0x8EBC6AF09C88C6E3ULL

static const uint64_t cave_anchor[8] = { A0, A1, A2, A3, A4, A5, A6, A7 };

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* two wires crossing: 64x64 -> 128, folded back to 64 */
static inline uint64_t fold128(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    unsigned __int128 pr = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)pr ^ (uint64_t)(pr >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32, bl = (uint32_t)b, bh = b >> 32;
    uint64_t ll = al * bl, hl = ah * bl, lh = al * bh, hh = ah * bh;
    uint64_t cross = (ll >> 32) + (uint32_t)hl + lh;
    uint64_t upper = (hl >> 32) + (cross >> 32) + hh;
    uint64_t lower = (cross << 32) | (uint32_t)ll;
    return lower ^ upper;
#endif
}

/* the shadows settle, then stop chittering: exactly two multiply rounds */
static inline uint64_t settle(uint64_t h) {
    h ^= h >> 33;
    h *= 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 29;
    h *= 0x9E3779B185EBCA87ULL;
    h ^= h >> 32;
    return h;
}

static inline uint64_t rd8(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t rd4(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* ---- small pile: never reaches the wires; the nearest prisoner alone ---- */
static uint64_t cave_small(const unsigned char *p, size_t len) {
    uint64_t h = (uint64_t)len * A0 + A7;
    if (len == 0) return settle(h ^ A1);
    if (len >= 32) {
        h += fold128(rd8(p)        ^ A1, rd8(p + 8)        ^ A2);
        h += fold128(rd8(p + 16)   ^ A3, rd8(p + 24)       ^ A4);
        h += fold128(rd8(p+len-32) ^ A5, rd8(p + len - 24) ^ A6);
        h += fold128(rd8(p+len-16) ^ A7, rd8(p + len - 8)  ^ A0);
    } else if (len >= 16) {
        h += fold128(rd8(p)        ^ A1, rd8(p + 8)       ^ A2);
        h += fold128(rd8(p+len-16) ^ A3, rd8(p + len - 8) ^ A4);
    } else if (len >= 8) {
        h += fold128(rd8(p) ^ A1, rd8(p + len - 8) ^ A2);
    } else if (len >= 4) {
        uint64_t x = ((uint64_t)rd4(p) << 32) | (uint64_t)rd4(p + len - 4);
        h += fold128(x ^ A1, ((uint64_t)len * A2) ^ A3);
    } else {
        uint64_t x = ((uint64_t)p[0] << 16) | ((uint64_t)p[len >> 1] << 24)
                   | ((uint64_t)p[len - 1]) | ((uint64_t)len << 8);
        h += fold128(x ^ A1, ((uint64_t)len * A2) ^ A3);
    }
    return settle(h);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    if (len < 64) return cave_small(data, len);   /* guarded fallback: small-pile path */

    const unsigned char * restrict p = data;
    size_t nstripe = len >> 6;                    /* one stripe = one knowing wall */

    /* the chain of prisoners sharpening the pillar */
    uint64_t p0 = A0 ^ (uint64_t)len, p1 = A2 + (uint64_t)len, p2 = A4, p3 = A6;
    uint64_t out[8];

#if defined(__AVX2__)
    const __m256i anchA = _mm256_set_epi64x((long long)A3,(long long)A2,(long long)A1,(long long)A0);
    const __m256i anchB = _mm256_set_epi64x((long long)A7,(long long)A6,(long long)A5,(long long)A4);
    __m256i acc0 = anchA, acc1 = anchB;           /* the wires, anchored */

#define CAVE_STRIPE(PP)                                                        \
    do {                                                                       \
        const unsigned char *q = (PP);                                         \
        __m256i d0 = _mm256_loadu_si256((const __m256i *)(q));                 \
        __m256i d1 = _mm256_loadu_si256((const __m256i *)(q + 32));            \
        uint64_t mk = rd8(q) ^ rd8(q + 40);       /* fed to the nearest one */  \
        uint64_t b0 = p0, b1 = p1, b2 = p2, b3 = p3;  /* last pass's angles */  \
        p0 = rotl64(b0 + mk, 29);                                              \
        p1 = rotl64(b1 ^ b0, 17);                 /* angle, not the mark */     \
        p2 = b2 + rotl64(b1, 41);                                              \
        p3 = rotl64(b3 ^ b2, 7) + b3;             /* the last one's angle */    \
        __m256i av = _mm256_set1_epi64x((long long)p3);                        \
        __m256i k0 = _mm256_xor_si256(d0, _mm256_add_epi64(anchA, av));        \
        __m256i k1 = _mm256_xor_si256(d1, _mm256_add_epi64(anchB, av));        \
        __m256i x0 = _mm256_mul_epu32(k0, _mm256_shuffle_epi32(k0, 0x31));     \
        __m256i x1 = _mm256_mul_epu32(k1, _mm256_shuffle_epi32(k1, 0x31));     \
        acc0 = _mm256_add_epi64(acc0,                                          \
                 _mm256_add_epi64(x0, _mm256_shuffle_epi32(d0, 0x4E)));        \
        acc1 = _mm256_add_epi64(acc1,                                          \
                 _mm256_add_epi64(x1, _mm256_shuffle_epi32(d1, 0x4E)));        \
    } while (0)

    for (size_t s = 0; s < nstripe; s++) { CAVE_STRIPE(p); p += 64; }
    if (len & 63) CAVE_STRIPE(data + len - 64);   /* led down again, overlapping */

    _mm256_storeu_si256((__m256i *)out,       acc0);
    _mm256_storeu_si256((__m256i *)(out + 4), acc1);
#undef CAVE_STRIPE

#else   /* no wide flood in this pit: same algebra, one wire at a time */
    uint64_t acc[8];
    for (int j = 0; j < 8; j++) acc[j] = cave_anchor[j];

#define CAVE_STRIPE(PP)                                                        \
    do {                                                                       \
        const unsigned char *q = (PP);                                         \
        uint64_t w[8]; memcpy(w, q, 64);                                       \
        uint64_t mk = w[0] ^ w[5];                                             \
        uint64_t b0 = p0, b1 = p1, b2 = p2, b3 = p3;                           \
        p0 = rotl64(b0 + mk, 29);                                              \
        p1 = rotl64(b1 ^ b0, 17);                                              \
        p2 = b2 + rotl64(b1, 41);                                              \
        p3 = rotl64(b3 ^ b2, 7) + b3;                                          \
        for (int j = 0; j < 8; j++) {                                          \
            uint64_t k = w[j] ^ (cave_anchor[j] + p3);                         \
            acc[j ^ 1] += w[j];                                                \
            acc[j] += (uint64_t)(uint32_t)k * (uint64_t)(uint32_t)(k >> 32);   \
        }                                                                      \
    } while (0)

    for (size_t s = 0; s < nstripe; s++) { CAVE_STRIPE(p); p += 64; }
    if (len & 63) CAVE_STRIPE(data + len - 64);
    memcpy(out, acc, sizeof(out));
#undef CAVE_STRIPE
#endif

    /* the silhouette: which lean, which stand straight, which cross another */
    uint64_t sil = (uint64_t)len * A0;
    sil += fold128(out[0] ^ A1, out[1] ^ A2);
    sil += fold128(out[2] ^ A3, out[3] ^ A4);
    sil += fold128(out[4] ^ A5, out[5] ^ A6);
    sil += fold128(out[6] ^ A7, out[7] ^ A0);
    sil += fold128(p0 ^ p2, p1 ^ p3);   /* the pillar's own last angle, crossed */
    return settle(sil);                 /* read once, kept alone */
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 18**

Stated before any measurement, with the arithmetic behind it so it can be falsified cleanly:

- Baseline FNV-1a is pure latency: `xor`(1) + `imul`(3) per byte ⇒ ~0.25 B/cycle ≈ 1.0–1.3 GB/s.
- This kernel: ~26 µops per 64-byte stripe, loop-carried critical path ≈3 cycles (the prisoner chain) and ≈1 cycle (the wire adds), so front-end bound at ~6 cycles/stripe ⇒ ~10 B/cycle ≈ 40 GB/s for cache-resident buffers, falling to the DRAM ceiling (~10–20 GB/s) for buffers past L3.
- Hence 18× as a mid-estimate: ~25–35× if the bench buffer is L1/L2-resident, ~8–15× if it is DRAM-resident, and **≈1× at len < 64**, where the guarded small-pile path runs instead and should be roughly FNV-speed or a little better.
- Avalanche: I predict **0.48–0.50** (i.e. essentially ideal) for len ≥ 64, and 0.47–0.50 for short inputs. The reason I expect this and not something mediocre despite spending nothing per byte: a flipped bit perturbs exactly one lane's accumulator, and the lane then passes through a `mul128` cross-fold plus a 2-multiply avalanche, which is the same finalization strength that lets xxHash/XXH3 pass SMHasher.
- Falsifiers I will accept as the mechanism failing: avalanche < 0.45, or speedup < 4× on a multi-KB buffer.

## MEASUREMENT

**Not measured.** `hash_bench` and `hash_contract` were unavailable in this session — no tools were exposed at all — so I ran zero measurements and I am not going to dress up the estimate above as one. The prediction is therefore untested, and the four permitted improvement rounds were not spent; the artifact is the first draft of the mechanism, not a tuned one. What I did instead was static: hand-check the intrinsic semantics (`_mm256_shuffle_epi32(k,0x31)` places lane-odd 32-bit words into the even slots so `vpmuludq` computes lo32×hi32 per 64-bit lane; `0x4E` swaps 64-bit neighbours, matching the scalar path's `acc[j^1] += w[j]`), verify the scalar and AVX2 paths compute the same function, check every rotate amount is in 1…63, check that every byte of every length class is read on some path, and check the `len-32`/`len-64` back-reads are all inside the buffer for their guarded length class.

Two things I know are unverified and would look for first in a real bench: whether the 2 redundant scalar loads feeding the chain cost measurable throughput (if so, the honest fix is to derive `mk` from one load, not to delete the chain), and the small-pile crossover — 64 is argued, not measured.

## VERDICT

The native's description translates without any slack. Every object landed on something concrete: the chained prisoners are a systolic register pipeline, the angle is a derived scalar that never carries the byte, the flood that must not be read while up is the no-mid-stream-merge rule, the anchored wires are SIMD lanes with per-lane secrets, "which cross another" is a pairwise 128-bit fold, and "the shadows stop chittering" is a finalizer that is run once and then stopped. The one place I compromised literalness: the nearest prisoner is fed two words per cache line rather than every single mark, because feeding him all eight would roughly double the per-stripe µop count. Every mark still enters the pit and bends the wires directly; only the chain sees a sample. I state that rather than hide it.

The honest assessment of novelty is deflationary and I think that is the right outcome: taken seriously, this metaphor *is* XXH3's block structure, which is exactly what step 4 asks for — a validated technique arrived at, not a novel mixer invented. The genuinely non-standard part is small and defensible: the prisoner chain injecting a broadcast angle into the anchors on every pass, which buys full-buffer order sensitivity for about one vector add and lets the anchor table stay in registers instead of walking a secret.

Stated risks, and how each is addressed rather than left hanging:
- *Overhead on small inputs* — guarded by `len < 64` → scalar short path, so the risky wide path never runs where it loses.
- *Needs AVX2* — guarded by `#if defined(__AVX2__)` with a semantically identical scalar path; `-march=native` on a non-AVX2 host still compiles and still works.
- *Thread parallelism* — deliberately not shipped. The metaphor has one pit and forbids reading the water while it is up; splitting would require the mid-stream read the native calls worthless, and the kernel is bandwidth-bound long before it is compute-bound.
- *Fixed anchors could make stripe order irrelevant* — prevented by the angle, which makes the multiplier operand depend on all prior content.

If the bench comes back with avalanche below 0.45, the mechanism's claim — that you can spend nothing per byte and buy all the mixing once, at the silhouette — is simply wrong, and I would say so rather than patch rounds back into the loop and call the metaphor vindicated.