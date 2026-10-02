# MAPPING

## Seed 1 — "A growing sailor-chain hung between fire and wall makes every shadow depend on the whole row, not just the newest knot."

| World object | Problem object |
|---|---|
| pile of marks, in their given order | input byte buffer `data[0..len)`, read in order |
| one mark | one input byte |
| sailor-chain, knotted, never untied, only lengthened | a *growing* state vector: lanes are appended/extended, never collapsed |
| "add one more knot per mark" | state width grows with position — monotone, append-only |
| the shadow is of *the whole chain*, not the newest knot | each emitted token depends on all prior state simultaneously (horizontal reduction over all lanes), not on a single accumulator cell |
| "the whole line hangs together or not at all" | the lanes are read as one wide register — a SIMD vector, mixed laterally |
| fire → chain → wall | input → state → output projection |

**Assumption broken:** "the state is a single accumulator updated in place, one value." Here the state is an extended row and every output is a function of the entire row.

## Seed 2 — "A mark's own pebble-count sets the incense-smoke's swing, bending the firelight before it lands."

| World object | Problem object |
|---|---|
| mark's own strokes counted as pebbles into a cup | a cheap, *data-derived* scalar extracted from the byte itself — its popcount / low bits |
| the cup's fill | that scalar's value, small integer in 0..8 |
| how far I swing the brick-incense on its short chain | a **rotation / shift distance**, taken from the data, applied to the state |
| smoke *bends the fire's throw before it lands* | the data value modulates the *mixing operator*, not the mixed value — data-dependent rotate instead of data-added constant |
| the swing is on a *short* chain — bounded arc | rotation amount is bounded (mod 64 / mod 8), one instruction |
| "wait for the smoke to thin to one straight thread" | the modulation must settle — one well-defined rotate amount per step, no accumulation of shift state |

**Assumption broken:** "mixing one byte requires a multiplication." A byte-derived *rotate distance* is a mixing primitive with multiplicative-like diffusion and no multiply. This is the one seed that attacks the multiply.

## Seed 3 — "The wet-clay print taken from the settled shadow, hardened in the stream and read by its ridges, is the only thing kept as the token."

| World object | Problem object |
|---|---|
| six fixed cracks in the wall, same six every time | a fixed-width output frame — six (here: lanes) positions the token will ever occupy |
| press wet clay to the settled shadow | take a snapshot of the state once it is stable (finalization, not per-byte) |
| carry the stub to the stream, let water run until only ridges remain | a *final* avalanche/erosion step — xor-shift folding that erodes soft material and leaves only high-entropy ridges |
| throw away smoke, pebble cup, old print | no retained per-byte intermediate state; only the ridges survive |
| "a hooded reader could work backward from print to pile" | one-wayness: finalization is non-invertible without the discarded scratch |
| ridges in one fist | the 64-bit return value |

**Assumption broken:** "each byte must be mixed into the running state before the next byte is read" and "more mixing rounds always means better mixing" — the heavy mixing happens *once*, at the stream, not per byte.

---

# CHOSEN SEED

**Seed 2** — "A mark's own pebble-count sets the incense-smoke's swing, bending the firelight before it lands."

It is the only one of the three that breaks *"mixing one byte requires a multiplication"*, which the instructions tell me to prefer. It is also maximally distant from FNV-1a/xxHash, whose entire per-byte step *is* a multiply.

I do not discard the other two: the native's single account contains all three, and they compose literally. Seed 1 gives the chain (a wide, laterally-coupled state rather than one accumulator); Seed 2 gives the per-mark mixing primitive (a data-derived swing — rotate, no multiply); Seed 3 gives the stream (one finalization at the end, ridges only). I take Seed 2 as the mechanism under test and let 1 and 3 supply the state shape and the finalizer, exactly as the native describes them.

# ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."**

The native never multiplies. He counts the strokes on a mark (popcount), and that count *swings the censer* — it sets how far the light is bent. In machine terms: the byte's own content selects a **rotation distance** applied to the whole chain. Rotation is the one cheap primitive that moves information *across* bit positions without carry propagation, and when the distance is itself data-dependent, the position of every later bit becomes a function of all earlier bytes. That is precisely the diffusion a multiply buys, obtained with `rol` instead of `imul`.

Two honest consequences I must handle rather than hide:

1. **Rotation alone does not avalanche.** XOR + rotate is linear over GF(2); a single flipped input bit produces a fixed, sparse pattern. The *data-dependence* of the rotate amount is what breaks linearity — and the native insists on it. But it is weak per step. This is exactly why Seed 3's stream matters: the native's token is not the shadow, it is the **ridges left after the water**. The finalization must carry the avalanche. So I let the native's own structure arrive at a validated real-world technique: `xmx`/`splitmix64`-style xor-shift-multiply finalization (step 4 of the instructions — a validated known finalizer beats one I invent). The *per-byte* path stays multiply-free, as the assumption demands; the one-time finalization is allowed to multiply because the native explicitly permits one distinct act at the stream, and the per-byte claim is untouched by a single O(1) epilogue.

2. **The chain must actually be wide and the swing must actually be short.** The censer is on a *short* chain — a bounded arc. I take rotate-by-(popcount-derived small value), realized SIMD-wide so the "whole line hangs together."

## Regime recognition — the native's own test

The known_way section describes two regimes: a buffer "read once, start to end, in order" (sequential, any length) and the implicit short-buffer case where per-byte work dominates. The native encodes the test himself: **"I wait, always, for the smoke to thin to one straight thread before I read."** Smoke that has not thinned is a pile too small for the piazza — too few marks for the chain to hang between fire and wall at all. The chain needs enough knots to span the gap. So:

- **len < 32:** the chain cannot span. One thread of smoke, no swing, no wall — walk the marks straight past the fire on a single short path (scalar, byte-at-a-time, multiply-free rotate), then straight to the stream. This is the guarded fallback to the simpler path.
- **len ≥ 32:** the chain spans; hang the full row, swing the censer, read all six cracks.

I deliberately use **no thread parallelism**. The metaphor's unit of work is one mark past one fire — at benchmark buffer sizes the chain is a few cache lines, far too small for OpenMP to pay, and thread-level splitting would also break the native's absolute insistence that moving a mark changes every shadow after it (order dependence). Vectorization only, per step 4's default.

# ARTIFACT

Design, stated plainly:

- **State = the sailor-chain:** four 64-bit lanes held as one `__m256i` (`v`), plus the scalar `len`-folded tail. Four lanes = the chain hanging across the gap; they are mixed *laterally* every block so no lane is an independent accumulator.
- **Per mark (per 8-byte group, vector-wide):** load the marks, XOR into the chain (the fire falls on it), then **rotate each lane by a distance derived from the data itself** — the pebble count. AVX2 has `_mm256_sllv_epi64`/`_mm256_srlv_epi64`, i.e. *per-lane variable* shifts, which is exactly a per-lane censer swing. The swing amount comes from the loaded bytes, so the operator is data-selected, breaking linearity.
- **Lateral coupling (the whole line hangs together):** a lane permute each block, so lane *i*'s next swing sees lane *i+1*'s content.
- **The stream (finalization):** horizontal fold of the four lanes, then `splitmix64`'s validated finalizer (xor-shift, multiply, xor-shift, multiply, xor-shift) — the water running over the clay until only ridges remain. One constant-time epilogue; everything else discarded.
- **Tail / short path:** multiply-free scalar rotate loop, then the same stream.

```c
#include <stdint.h>
#include <stddef.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the stream: water runs over the clay until only ridges remain ----
   splitmix64's validated finalizer (Steele/Lea/Vigna). One O(1) epilogue;
   the per-mark path above it never multiplies. */
static inline uint64_t ridges(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    r &= 63u;
    return (x << r) | (x >> ((64u - r) & 63u));
}

/* ---- one short path of smoke: the chain cannot span, walk marks straight.
   Multiply-free: XOR the mark in, then swing by a distance the mark itself
   sets (its low bits + a fixed short arc). ---- */
static uint64_t walk_short(const unsigned char *data, size_t len) {
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;
    for (size_t i = 0; i < len; i++) {
        unsigned char m = data[i];
        h ^= (uint64_t)m;
        /* the mark's own count sets the swing: short chain, bounded arc */
        h = rotl64(h, (unsigned)((m & 31u) + 7u));
        h ^= h >> 17;
    }
    return ridges(h ^ (uint64_t)len);
}

uint64_t kernel(const unsigned char *data, size_t len) {
#if defined(__AVX2__)
    /* smoke has not thinned: too few marks for the chain to span the piazza */
    if (len < 32) return walk_short(data, len);

    const unsigned char *restrict p = data;
    size_t n = len;

    /* the sailor-chain: four knots hung between fire and wall, distinct seeds
       so no two lanes are the same row */
    __m256i v = _mm256_set_epi64x(
        (int64_t)0x2545F4914F6CDD1DULL,
        (int64_t)0x9E3779B97F4A7C15ULL,
        (int64_t)0xC2B2AE3D27D4EB4FULL,
        (int64_t)0x165667B19E3779F9ULL);
    v = _mm256_xor_si256(v, _mm256_set1_epi64x((int64_t)len));

    const __m256i arc = _mm256_set1_epi64x(7);          /* short chain */
    const __m256i mask5 = _mm256_set1_epi64x(31);       /* bounded swing */
    const __m256i c64 = _mm256_set1_epi64x(64);

    while (n >= 32) {
        __m256i m = _mm256_loadu_si256((const __m256i *)p);

        /* the fire falls on the chain: every knot takes the mark */
        v = _mm256_xor_si256(v, m);

        /* the mark's own pebble-count sets the censer's swing, per lane.
           count the strokes of this group of marks, bound the arc. */
        __m256i cnt = _mm256_and_si256(_mm256_srli_epi64(m, 3), mask5);
        __m256i sw  = _mm256_add_epi64(cnt, arc);               /* 7..38 */
        __m256i isw = _mm256_sub_epi64(c64, sw);

        /* rotate: no multiplication anywhere on this path */
        v = _mm256_or_si256(_mm256_sllv_epi64(v, sw),
                            _mm256_srlv_epi64(v, isw));

        /* the whole line hangs together: couple the knots laterally so
           knot i's next swing sees knot i+1's content */
        v = _mm256_xor_si256(v, _mm256_permute4x64_epi64(v, 0x39)); /* 0,3,2,1 */

        /* a second swing whose distance comes from the chain itself, not a
           constant -- keeps the operator data-selected (nonlinear) */
        __m256i c2  = _mm256_and_si256(v, mask5);
        __m256i sw2 = _mm256_add_epi64(c2, arc);
        __m256i is2 = _mm256_sub_epi64(c64, sw2);
        v = _mm256_or_si256(_mm256_sllv_epi64(v, sw2),
                            _mm256_srlv_epi64(v, is2));
        v = _mm256_xor_si256(v, m);

        p += 32; n -= 32;
    }

    /* read the six fixed cracks: fold the hanging chain to one shadow */
    uint64_t lane[4];
    _mm256_storeu_si256((__m256i *)lane, v);
    uint64_t h = lane[0];
    h = rotl64(h, 13) ^ lane[1];
    h = rotl64(h, 29) ^ lane[2];
    h = rotl64(h, 41) ^ lane[3];

    /* the last few marks, walked straight */
    while (n > 0) {
        unsigned char mk = *p++;
        h ^= (uint64_t)mk;
        h = rotl64(h, (unsigned)((mk & 31u) + 7u));
        h ^= h >> 17;
        n--;
    }

    /* throw away the smoke and the cup; keep only the ridges */
    return ridges(h ^ (uint64_t)len);
#else
    return walk_short(data, len);
#endif
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 3.2**

Reasoning, before any measurement: FNV-1a is a serial dependency chain of `xor; imul` per byte — on modern x86 that is ~3 cycles of latency per *byte*, since the multiply latency is on the critical path and nothing overlaps. My per-32-byte block is roughly: load, xor, two variable-shift pairs (`vpsllvq`/`vpsrlvq`, ~1 cycle each, 2 per rotate), a permute (~3 cycles), two xors — call it ~12–14 cycles of latency for 32 bytes, i.e. ~0.4 cycles/byte against FNV's ~3. That is a theoretical ~7×, but the chain is genuinely serial block-to-block (the native insists on it), the permute is a latency bottleneck, and short buffers pay the scalar fallback and the fixed `ridges` epilogue. I discount hard for the serial block dependency and small-size overhead and predict **~3.2×**.

On avalanche I am less confident, and I will say so now rather than after: I expect a *good but not excellent* score. The per-byte path is XOR + data-dependent rotate, which is nonlinear but weakly so per step; the heavy lifting falls on the one-time `splitmix64` finalizer. `splitmix64`'s finalizer is a validated, measured near-perfect 64→64 bit mixer, so any difference that reaches the fold at all should avalanche properly. My concrete worry is *difference cancellation inside the loop* — a flipped bit that changes a swing distance in a way that happens to leave the folded `h` unchanged. I predict avalanche in the **0.45–0.50** range (where 0.5 is ideal), and I would call anything below 0.45 a real failure of the mechanism, not a tuning issue.

# MEASUREMENT

**Not measured.** No tools are available in this session — `hash_bench` and `hash_contract` could not be called, so I have no throughput number and no avalanche score. The prediction above is stated blind, which is the correct order, but it is unverified. I am reporting that plainly rather than writing a plausible-looking table: everything in the MEASUREMENT section would be fabricated, and a fabricated measurement is worse than none.

What I would check first, in order, and what each outcome would mean:

1. **Avalanche < 0.45** → the mechanism is failing, not the tuning. The likely cause is that `_mm256_srli_epi64(m, 3)` draws the swing from only a narrow slice of each lane's bits, so many input bits never influence any swing distance. Fix within the metaphor: count the marks' strokes properly — a real popcount (`_mm256_sad_epu8` over a nibble-LUT popcount, or simply folding the lane with `m ^ (m>>32)` before masking) so *every* stroke on the mark contributes to the cup's fill.
2. **Avalanche ~0.5 but speedup < 1.5** → the permute is the bottleneck. Hang two chains side by side (two `__m256i` accumulators over 64-byte blocks) so independent work overlaps the serial latency, keeping one permute each.
3. **Small-size regression** → tighten the `len < 32` threshold upward (to 64 or 128), since `walk_short` is itself multiply-free-cheap and the vector path's fixed setup plus fold may not amortize below a few blocks.

I have already addressed the one risk my own design names: the `len < 32` guard with the `walk_short` fallback, and the `#if defined(__AVX2__)` guard so the kernel is correct and reasonable on a machine without AVX2 rather than failing to compile.

# VERDICT

The native's mechanism translates with unusual directness, and the translation is honest on one point I want to be explicit about: **the per-mark path contains no multiplication at all.** The assumption "mixing one byte requires a multiplication" is genuinely broken, not finessed — every per-byte operation is XOR, variable rotate, or permute. The multiply appears exactly once, in the O(1) epilogue, and the native's own account demands a distinct final act at the stream ("carry that stub to the mouth of the stream"), so this is the metaphor's structure rather than a concession to the textbook method.

The three seeds turned out to be one design, which I take as evidence the native was describing a single coherent machine: the chain is the state shape (Seed 1, breaking the single-accumulator assumption), the pebble-swing is the mixing primitive (Seed 2, breaking the multiply), and the stream is the finalizer (Seed 3, breaking "mix every byte before reading the next" and "more rounds is always better" — the rounds are concentrated, not spread). Per step 4 I let Seed 3's stream *arrive at* `splitmix64`'s validated finalizer rather than inventing a novel one; that is the single place where a known technique is strictly better than anything I would make up, and the metaphor pointed straight at it.

Where I expect this to be weaker than the known way: **xxHash is extremely well-tuned and will likely beat this on pure throughput at large sizes**, because it runs four *independent* accumulator streams with no cross-lane permute, so its multiplies fully pipeline. My design deliberately couples the lanes every block — the native is emphatic that "the whole line hangs together or not at all" and that moving one mark changes every later shadow — and that coupling is a real latency cost I chose to pay for fidelity. Against plain FNV-1a I am confident of a large win; against xxHash I would not bet on throughput, only on the more interesting claim that you can get there without the multiply.

The honest bottom line: a well-specified, guarded, multiply-free-per-byte kernel whose predicted numbers are stated but **unmeasured in this session**. The 3.2× and the 0.45–0.50 avalanche are predictions on record, and the pipeline's measurement should be read as the actual verdict on them — including the possibility that it falsifies both.