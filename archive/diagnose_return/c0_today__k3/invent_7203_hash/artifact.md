## MAPPING

**SEED 1 — the chained line sharpening the blank pillar**

| World object | Problem object |
|---|---|
| pile of marks, given order | `data[0..len-1]`, read in order |
| "knowing walls" we lead marks past | the load path; nothing is branched on |
| chained prisoner | one lane of state (one 64-bit accumulator) |
| "does not look at the mark, only feels how it changes the angle" | the byte is never inspected/compared; it only perturbs a rotation+add — **no multiply** |
| "the angle, not the mark, to the next in the chain" | lane *j* receives lane *j−1*'s previous angle; the byte itself is discarded |
| "a mark dropped at the very start still trembles in the hand of the last" | closed cross-feed ring (lane 0 ← lane 7) so early bytes reach every lane |
| blank pillar, "nothing read whole" | the lane bank is never read as a whole inside the loop |
| "prisoners forget the mark as soon as it's sharpened past" | streaming; no buffer retained, O(1) state |

**Breaks:** *"mixing one byte requires a multiplication."* The prisoner's whole repertoire is *rotate and add an angle*. It also partly breaks *"the state is a single accumulator"* (many hands, many angles).

**SEED 2 — the flooding river and the anchored wire shapes**

| World object | Problem object |
|---|---|
| the flood, rising on each pass | one in-order sweep over the buffer |
| "does not recede until every mark has gone through" | the loop may not be exited or sampled early |
| tiny wire shapes, **anchored**, "do not move themselves" | 8 independent accumulator lanes, fixed seeds, no cross-dependency except the chain's one link |
| "past where any traveler has ever fused its source" | lane seeds are fixed irrational-digit constants, not derived from data |
| "bending each wire by exactly the angle the last prisoner ground" | each lane is updated by the same ARX form… |
| "which lean, which stand straight" | …but with its **own rotation amount and own key**, so one angle bends each wire differently |
| the flood moving all wires at once | 8-way ILP / SIMD-shaped update; 64 bytes per step |

**Breaks:** *"the state is a single accumulator updated in place, one value."* The state is a bank. Also softens *"the whole buffer must be read once, start to end, in order"* — the order survives only through the chain link, not through a single serial accumulator.

**SEED 3 — the silhouette read once against the sun**

| World object | Problem object |
|---|---|
| horizonless pit, water still up | the loop still running; state not yet final |
| "a token pulled while the water is still up is worthless, the shapes beneath it still swimming" | **no avalanche round inside the loop** — intermediate state is deliberately under-mixed and that is fine |
| "the shadows stop their chitter / walls still settling" | all 8 lane writes retired before anything is folded |
| climbing to the rim, reading the silhouette | the finalizer, executed exactly once |
| "which lean, which stand straight, **which cross another**" | per-lane fold + cross-lane fold via 64×64→128 multiply, high⊕low |
| "small and fixed" | 64 bits out, independent of `len` |
| "I keep nothing but that shadow-shape" | discard all 512 bits of lane state |
| "I never ask the river where its source lies" | one-wayness: the finalizer is non-invertible and len-bound |

**Breaks:** *"more mixing rounds always means better mixing."* Mixing rounds are moved **out** of the per-byte loop entirely. There is exactly **one** strong round, at the end. The native's claim is that *one* round after the flood beats *len* rounds during it — both in cost and in avalanche.

---

## CHOSEN SEED

**SEED 3.** It is the one seed that attacks the preferred assumption, and it is the most literal reading of the native's single strongest statement ("a token pulled while the water is still up is worthless… I wait for that recession, never before"). The three seeds are facets of one machine, so the artifact necessarily embodies all three — the chain (cheap angle-only absorb), the wires (multi-lane bank), the silhouette (single terminal avalanche) — but SEED 3 is what *governs the design*: it is the reason the loop is allowed to be weak.

It is also maximally far from the known way: FNV-1a / xxHash put a full mixing round (`xor` + `imul`) on every single byte, on the critical path, in one accumulator. The native puts **zero** mixing rounds on any byte and one on the whole buffer.

## ASSUMPTION BROKEN

> **"more mixing rounds always means better mixing"**

Rejected, in the strong form. The native's loop is a cheap, nearly-linear ARX absorb whose only job is to be *injective enough* — to get every bit of every mark into the wire bank without losing it. Diffusion is not attempted there at all. All diffusion is bought in one terminal multiply-fold + avalanche. Result: FNV-1a pays ~4 cycles of serial `imul` latency *per byte* for diffusion it then partially destroys by the next byte; the native pays ~0.1 cycles per byte for absorption and ~35 cycles **once** for diffusion.

Secondary casualty: *"mixing one byte requires a multiplication"* — there is no multiply anywhere in the hot loop.

**Step 4 check — does this land on a validated technique?** Yes, and I let it. "Weak wide absorb + strong terminal fold" is exactly the architecture of xxHash3/wyhash/HighwayHash (multi-lane accumulators, overlapping last stripe, `mul128_fold64` finalization) and, further back, of UMAC/NH (a cheap linear compressor finalized by a strong primitive). My `crossw()` is xxh3's `mul128_fold64` verbatim; my `silhouette()` is MurmurHash3's `fmix64` verbatim. I invented neither. What the metaphor contributes that xxh3 does *not* have is the strictly multiply-free absorb (xxh3's `accumulate_512` still multiplies) plus the single-link prisoner chain that makes lanes order-dependent on each other.

**Step 5 — two regimes, recognized in-world.** The native's own test is whether the flood rises above the wires' anchors. Literally: if the pile of marks is too small to raise the water, the wire bank is never reached and the pit holds only two of the chained. `len < 64` → two-lane shallow path with a single cross-fold; `len >= 64` → full eight-wire flood. Both paths share the same absorb algebra, so this is one mechanism with a regime test, not two bolted-together hashes.

**Thread parallelism: deliberately refused.** The metaphor says *one* flood, *one* chain, angles carrying forward. Splitting the river into independent floods would sever the chain and change the function. Vectorization-shaped ILP (8 independent lanes, `restrict`, `memcpy` loads, constant rotate immediates, 64-byte stripes) is taken instead, per the instruction's default ordering. No OpenMP.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ================= the world, made literal =========================
   marks              -> input bytes, in their given order
   chained prisoners  -> 8 lanes; each holds one mark of the stripe in flight
   "the angle"        -> rotate-then-add. No prisoner multiplies.
                         No prisoner looks at the mark (no compare, no branch).
   angle passed on    -> lane j adds lane j-1's PREVIOUS angle; lane 0 <- lane 7,
                         a closed chain, so the first mark still trembles in the last hand
   blank pillar       -> the lane bank; never read whole inside the loop
   the flood          -> one in-order sweep; does not recede until every mark is through
   wire shapes        -> the 8 anchored lanes: same angle in, own rotation + own key,
                         so each wire leans differently
   THE SILHOUETTE     -> ONE multiply-fold + ONE avalanche, after the flood, read once.
                         This is the only mixing round in the entire hash.
   shallow water      -> len < 64: the flood never reaches the wires, so the pit
                         holds only two of the chained (same algebra, 2 lanes)
   ================================================================== */

static const uint64_t K[16] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL,
    0x2545F4914F6CDD1DULL, 0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL,
    0x8EBC6AF09C88C6E3ULL, 0x589965CBA07BB1F9ULL, 0x2D358DCCAA6C78A5ULL,
    0x8BB84B93962EACC9ULL, 0x4B33A62ED433D4A3ULL, 0x4D5A2DA51DE1AA47ULL,
    0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
    0x27D4EB2F165667C5ULL
};

static inline uint64_t ld64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t ld32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }
static inline uint64_t rotl64(uint64_t x, unsigned r){ return (x << r) | (x >> (64u - r)); }

/* "which cross another" - two wires casting one shadow.
   64x64->128 multiply, high half folded onto low. (xxh3's mul128_fold64.) */
static inline uint64_t crossw(uint64_t a, uint64_t b)
{
#if defined(__SIZEOF_INT128__)
    __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)prod ^ (uint64_t)(prod >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32;
    uint64_t bl = (uint32_t)b, bh = b >> 32;
    uint64_t m0 = al*bl, m1 = ah*bl, m2 = al*bh, m3 = ah*bh;
    uint64_t mid = (m0 >> 32) + (uint32_t)m1 + (uint32_t)m2;
    uint64_t lo  = (m0 & 0xFFFFFFFFULL) | (mid << 32);
    uint64_t hi  = m3 + (m1 >> 32) + (m2 >> 32) + (mid >> 32);
    return lo ^ hi;
#endif
}

/* the shadow-shape itself, drawn once the flood is gone. (MurmurHash3 fmix64.) */
static inline uint64_t silhouette(uint64_t h)
{
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33; return h;
}

/* one pass of the flood over the eight wires: 8 loads, 8 xors, 8 rotates, 8 adds.
   all eight updates read only the PREVIOUS pass's angles -> 8 independent chains,
   latency-free, issue-bound. No multiply. Nothing is read whole. */
#define FLOOD(k0,k1,k2,k3,k4,k5,k6,k7, r0,r1,r2,r3,r4,r5,r6,r7, q)   \
    do {                                                             \
        uint64_t t0=a0,t1=a1,t2=a2,t3=a3,t4=a4,t5=a5,t6=a6,t7=a7;    \
        a0 = rotl64(t0 ^ (ld64((q) +  0) ^ (k0)), (r0)) + t7;        \
        a1 = rotl64(t1 ^ (ld64((q) +  8) ^ (k1)), (r1)) + t0;        \
        a2 = rotl64(t2 ^ (ld64((q) + 16) ^ (k2)), (r2)) + t1;        \
        a3 = rotl64(t3 ^ (ld64((q) + 24) ^ (k3)), (r3)) + t2;        \
        a4 = rotl64(t4 ^ (ld64((q) + 32) ^ (k4)), (r4)) + t3;        \
        a5 = rotl64(t5 ^ (ld64((q) + 40) ^ (k5)), (r5)) + t4;        \
        a6 = rotl64(t6 ^ (ld64((q) + 48) ^ (k6)), (r6)) + t5;        \
        a7 = rotl64(t7 ^ (ld64((q) + 56) ^ (k7)), (r7)) + t6;        \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;

    /* ---------- regime test: does the flood reach the wires' anchors? ---------- */
    if (len < 64) {
        /* shallow water: only two of the chained. Same angle algebra, no wire bank,
           and a cheaper silhouette (one cross), so short piles are never penalised
           by a finalizer sized for eight wires. */
        uint64_t h0 = K[12] ^ ((uint64_t)len * K[13]);
        uint64_t h1 = K[14] ^ rotl64((uint64_t)len + 1u, 32);
        const unsigned char *q = p;
        size_t n = len;
        while (n >= 16) {
            uint64_t u0 = h0, u1 = h1;
            h0 = rotl64(u0 ^ (ld64(q    ) ^ K[0]), 27) + u1;
            h1 = rotl64(u1 ^ (ld64(q + 8) ^ K[1]), 41) + u0;
            q += 16; n -= 16;
        }
        if (n >= 8) { h0 = rotl64(h0 ^ (ld64(q) ^ K[2]), 23) + h1; q += 8; n -= 8; }
        if (n >= 4) { h1 = rotl64(h1 ^ (ld32(q) ^ K[3]), 31) + h0; q += 4; n -= 4; }
        while (n) { h0 = rotl64(h0 ^ ((uint64_t)(*q++) ^ K[4]), 11) + h1; --n; }
        return silhouette(crossw(h0 ^ K[5], h1 ^ K[6]) ^ rotl64(h0 + h1, 32));
    }

    /* ---------- the flood: eight wires anchored in the riverbed ---------- */
    uint64_t a0 = K[0] ^ ((uint64_t)len * K[8]);
    uint64_t a1 = K[1], a2 = K[2], a3 = K[3];
    uint64_t a4 = K[4], a5 = K[5], a6 = K[6], a7 = K[7];

    size_t nstripe = len >> 6;
    const unsigned char *q = p;
    for (size_t b = 0; b < nstripe; ++b, q += 64)
        FLOOD(K[0],K[1],K[2],K[3],K[4],K[5],K[6],K[7],
              13,29,41,7,53,19,37,59, q);

    /* the water does not recede until EVERY mark has gone through: the final
       stripe is re-read overlapping the tail, with the other anchoring and the
       rotations reversed, so no mark is left dry and nothing cancels. */
    FLOOD(K[8],K[9],K[10],K[11],K[12],K[13],K[14],K[15],
          59,37,19,53,7,41,29,13, p + (len - 64));

    /* ---------- the silhouette: read once, kept alone ---------- */
    uint64_t s = (uint64_t)len * K[14];
    s ^= crossw(a0 ^ K[0], a1 ^ K[1]);             /* which lean       */
    s += crossw(a2 ^ K[2], a3 ^ K[3]);
    s ^= crossw(a4 ^ K[4], a5 ^ K[5]);
    s += crossw(a6 ^ K[6], a7 ^ K[7]);
    s ^= crossw((a0 + a4) ^ K[9],  (a2 + a6) ^ K[10]);  /* which cross another */
    s += crossw((a1 + a5) ^ K[11], (a3 + a7) ^ K[12]);
    return silhouette(s);                          /* the shadow-shape */
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 12**

Stated before any measurement, and no measurement has been taken yet.

Reasoning behind the number, so it is falsifiable rather than decorative:

- **FNV-1a (the reference)**: `h = (h ^ byte) * prime` is a serial dependency of `imul` (3–4 cycle latency) + `xor` per byte ⇒ ≈ **0.25 B/cycle**, ≈ 0.9 GB/s at 3.5 GHz.
- **This kernel, hot loop**: 64 bytes costs 8 loads + 8 xor + 8 `rol`-by-immediate + 8 `add` ≈ 32 µops, all eight chains independent ⇒ issue-bound at ≈ **8–10 B/cycle**, ≈ 30 GB/s — a *compute* ratio of ~35×.
- Therefore the answer depends entirely on the benchmark's size mix: **~35×** if buffers live in L1/L2, **~10–12×** if DRAM bandwidth caps us at ~10 GB/s, **~5×** at 64–128 bytes where the ~35-cycle terminal silhouette dominates, **~5×** in the shallow regime.
- 12 is my estimate of the blend. I will be wrong in a specific direction if the harness uses one large resident buffer (then I am under by ~3×) or a storm of ≤64-byte keys (then I am over by ~2×).

**Avalanche prediction:** ≈ 0.50 flipped output bits per flipped input bit, with per-(input-bit, output-bit) bias within ~±0.01 for all `len`. Mechanism: a single input bit flip reaches at least one lane; every lane enters the silhouette through at least two independent 64×64→128 multiply-folds (a one-bit change in a multiplicand perturbs the whole 128-bit product, and the high half is folded onto bits below the flip), and the result then passes `fmix64`, which is a measured-good 64→64 avalanche. If the measured avalanche comes back near 0.50 *and* the throughput ratio is >5, SEED 3's claim — that one terminal round beats `len` per-byte rounds — is confirmed. If avalanche lands below ~0.47, SEED 3 is *falsified as stated* and the honest fix is to put mixing back in the loop, i.e. to concede the assumption.

## MEASUREMENT

**Not performed. No tools were available in this session** — `hash_bench`, `hash_contract` and the symbolic/unconventional tools were all absent, so I have no measured throughput and no measured avalanche score to report. I am reporting that plainly rather than inventing numbers: everything in the PREDICTION section is a static cost model and a diffusion argument, not data.

What I *was* able to check by hand, and what remains open:

| Checked by hand | Result |
|---|---|
| Every byte of `data[0..len-1]` is absorbed at least once, all `len` | ✔ (`nstripe` stripes + overlapping final stripe; `len >= 64` guaranteed on that path) |
| No out-of-bounds read | ✔ `p + (len-64)` only reached when `len >= 64`; shallow path steps 16/8/4/1 |
| `rotl64` never called with r ∈ {0,64} | ✔ all rotations are literals in 7..59 |
| Double-absorbed tail bytes cannot cancel | ✔ lane *j*'s two rotations differ (13↔59, 29↔37, 41↔19, 7↔53, …) so a flipped data bit lands at two distinct bit positions |
| `len = 0` well-defined | ✔ returns `silhouette(crossw(K[12]^K[5], (K[14]^2^32)^K[6]) ^ …)` |
| Portable when `__int128` absent | ✔ 32×32 schoolbook path |
| **Measured throughput** | ✖ unknown |
| **Measured avalanche** | ✖ unknown |
| **Collision behaviour** (SMHasher-class) | ✖ unknown — and *not* claimed. The absorb is ARX-with-carries, not a proven-good compressor. |

## VERDICT

The native's idea translates with unusual cleanliness and the translation is not a dressed-up FNV: there is genuinely **no multiply in the hot loop**, the state is genuinely a bank of eight independently-anchored wires rather than one accumulator, and the single mixing round genuinely happens after the flood has fully receded. SEED 3's inversion — *fewer rounds, placed later, beat many rounds placed per-byte* — is the real content, and it is the reason the loop can be 30× cheaper than FNV's while plausibly ending up *better* mixed.

I want to be precise about how much credit the metaphor deserves. The destination is a known, validated place: multi-lane accumulators, an overlapping last stripe, `mul128_fold64`, `fmix64` — xxh3 and wyhash live there, and I used their primitives rather than inventing substitutes, exactly as step 4 demands. The metaphor's distinct contribution is narrower but real: the strictly multiply-free absorb (xxh3 still multiplies per stripe) and the closed prisoner chain, which is the one structural element I have *not* seen validated elsewhere and therefore the one I trust least.

Named risks, and what I did about each rather than leaving them as caveats:

1. **Small inputs pay a finalizer sized for eight wires.** This is the one condition where the mechanism could lose to FNV-1a. Guarded: `len < 64` takes the two-prisoner shallow path, same algebra, one `crossw` instead of six. Not a caveat — a branch in the code.
2. **Thread parallelism would be a net loss and would break the chain.** Dropped entirely, not guarded-and-shipped. The metaphor forbids it (one flood) and the sizes don't justify it.
3. **The absorb is near-linear over GF(2) plus carries.** This is the honest soft spot. Avalanche should be fine because the silhouette is non-linear and non-invertible, but *collision* resistance against an adversary choosing inputs is **not** established by anything I did, and I am not claiming it. A differential that cancels across the cross-feed ring would not show up in an avalanche score at all. Before this went anywhere real it would need SMHasher, not `hash_bench`.

So: the mechanism is sound, literal, and guarded — but it is **unmeasured**, and an unmeasured hash is a hypothesis. My prediction of 12× and ~0.50 avalanche is on the record above, written before any measurement, and I would rather it be checked and found wrong than quoted as a result.