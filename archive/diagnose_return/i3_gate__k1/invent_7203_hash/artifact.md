## MAPPING

I read all three seeds against the same world. The world is Plato's cave, re-cut as a hash: nothing is ever *looked at*, only its shadow is ever read, and the reading happens exactly once.

### SEED 1 — the chained line sharpening the blank pillar

| World object | Problem object |
|---|---|
| pile of marks, in their given order | `data[0..len)`, read in index order |
| the knowing walls they're led past | the cache hierarchy the bytes stream through |
| the chained line of prisoners | a fixed-length array of state lanes (`l0..l7`, 8 × `uint64_t`) — **not one accumulator** |
| the blank pillar | registers that never hold a raw byte; "nothing is read whole" |
| "does not look at the mark, only feels how it changes the angle" | the byte/word enters only as a *rotation-and-carry* perturbation: `acc = rotl(acc ^ w, 29) + w` — add/rotate/xor, no multiply |
| "passes that angle, not the mark" | what moves between lanes is derived state, never the input word |
| "a mark dropped at the very start still trembles in the hand of the last one" | every input byte must reach every output bit (avalanche) |
| "the prisoners forget the marks as soon as they're sharpened past" | single streaming pass, no scratch retention |

**Breaks:** *"the state is a single accumulator updated in place, one value"* and *"mixing one byte requires a multiplication"*.

### SEED 2 — the flood that recedes only when all marks are through

| World object | Problem object |
|---|---|
| the river that floods every solid argument | the main streaming loop; "water up" = loop not yet finished |
| the flood rising on each pass | one 64-byte block consumed per iteration |
| "does not recede until every mark has gone through" | no output is formed until `len` bytes are consumed |
| "a token pulled while the water is still up is worthless, the shapes beneath it still swimming" | reading the state mid-stream yields a value with incomplete diffusion |
| tiny wire shapes anchored in the riverbed | the output-bearing lane array — *fixed size, independent of `len`* |
| "the flood moves them, bending each wire by exactly the angle ground on that pass" | lanes are updated in bulk per pass, by the angle, not per individual byte |
| "past where any traveler has ever fused its source" / "I never ask the river where its source lies" | one-wayness; no inversion of the hash |

**Breaks:** *"each byte must be mixed into the running state before the next byte is read"*. The per-byte act is only the cheap angle-passing; the wires are bent by the **flood**, i.e. per pass, which licenses loading 64 bytes at once before any of them touch the output-bearing state (SIMD / wide loads).

### SEED 3 — the silhouette read once

| World object | Problem object |
|---|---|
| the sun above the pit | the single projection from multi-lane state down to one 64-bit value |
| the silhouette the wires throw | the finalizer's output: the returned `uint64_t` |
| "which lean, which stand straight, which cross another" | the fold: each lane rotated by its own angle, xored, multiplied — lanes *crossing* = cross-lane coupling happens **here**, not per byte |
| "drawn only once the flood is gone" | finalize strictly after the streaming loop |
| "the shadows stop their chitter — the walls are still settling" | the finalizer must run to *completion* (both multiply stages of a full avalanche) before the value is read |
| "read once and kept alone"; "I keep nothing but that shadow-shape" | **one** finalization, then return; no repeated rounds, no retained state |
| "small and fixed" | constant-cost finalization, O(1) in `len` |

**Breaks:** *"more mixing rounds always means better mixing"*. The native's whole economy is: almost no mixing per mark, and **one** correctly-timed strong read. A second reading of the same silhouette adds nothing; what matters is that the first one is taken after the water is down.

---

## CHOSEN SEED

**SEED 3 — the silhouette, read once.**

It is the one seed of the three that breaks the preferred assumption, so by the stated rule it wins. It is also the most literal: "the silhouette the wires throw against the sun" is exactly a projection of an 8-word state array onto one 64-bit word, and "read once and kept alone" is exactly a single finalization. Seeds 1 and 2 remain in force as the *same world's* furniture — they supply the prisoners (lanes), the angle (rotate-xor-add), and the flood (the per-pass block loop) that SEED 3's silhouette is cast *from*.

Where it differs from FNV-1a/xxHash-as-stated: the known way puts the mixing strength **per byte** (a multiply every byte, into one accumulator, 100 000 rounds for 100 000 bytes). The native puts essentially *zero* strength per byte and spends it all in one fixed, constant-cost read at the end.

## ASSUMPTION BROKEN

> "more mixing rounds always means better mixing"

The native's token is drawn **once**. The per-mark work is a single rotate-xor-add (no multiply, no round function); all diffusion is concentrated in one constant-cost silhouette at the end. Mixing rounds scale as O(1), not O(len) — and the avalanche does not get worse, because the finalizer is a bijection fed a *provably nonzero* difference.

Secondarily broken, because the same world demands it: the state is eight lanes, not one accumulator (SEED 1), and the wires are bent per *flood pass*, not per byte (SEED 2) — which is what makes the 64-byte wide load legal.

**Arriving at a validated technique rather than inventing (step 4).** I did not invent anything here, and I checked before writing: the mechanism the metaphor forces is exactly the union of three independently validated real-world constructions.
- "no multiply, only an angle" → **ARX** (add-rotate-xor), the primitive of ChaCha, BLAKE2 and SipHash.
- "a chained *line* of prisoners, bent per flood pass" → **striped multi-accumulator streaming**, xxHash64's four-lane loop.
- "the silhouette, read once, after the chittering stops" → **Murmur3's `fmix64`** finalizer (the same construction as splitmix64's output stage), the standard validated strong 64-bit avalanche function.

The novel part is only the *allocation of effort* the seed dictates — all strength at the single read, none per mark.

**Non-cancellation (the one thing I can prove rather than measure).** The native's claim that "a mark dropped at the very start still trembles in the hand of the last one sharpening" is checkable. With `stroke(acc,w) = rotl(acc ^ w, A) + w` and `A = 29`: flipping bit `b` of `w` changes the result by exactly `±2^((b+A) mod 64) ± 2^b`. That is zero only if `(b+A) ≡ b (mod 64)`, i.e. `A ≡ 0`, which it is not. So **every single-bit input flip changes its lane**. Each subsequent stroke, each fold step `(v ^ rotl(l,k)) * odd`, and `fmix64` itself are all bijections in the running value, so that nonzero difference can never be destroyed downstream. A single input bit therefore always reaches the finalizer, and the finalizer is the validated avalanche function. This is why the metaphor's "almost no rounds" is not reckless.

**Regimes (step 5).** The assumption list describes two regimes (a whole buffer streamed in order; and the "overhead if small" case). The metaphor encodes the discriminator itself: *the flood has to rise high enough to reach the wires.* A thin pile of marks never raises the water, the wires never bend, and the native reads the shadow of a **single** wire instead. That is the runtime check `len < 64`: one accumulator, no eight-lane setup, no eight-multiply fold, same single silhouette. The eight-lane/64-byte path is also guarded by `__AVX2__` with a scalar eight-lane fallback that computes the *identical* hash.

**No thread parallelism.** The metaphor has one chained line and one flood — there is no second line in the world, and at hash-benchmark sizes a 64-byte unit of work is far below any OpenMP threshold. Vectorization hints only (`restrict`, `memcpy` loads, AVX2 wide loads).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- eight odd anchors: where the wires are fixed in the riverbed ---- */
#define W1 0x9E3779B185EBCA87ULL
#define W2 0xC2B2AE3D27D4EB4FULL
#define W3 0x165667B19E3779F9ULL
#define W4 0x85EBCA77C2B2AE63ULL
#define W5 0x27D4EB2F165667C5ULL
#define W6 0xD6E8FEB86659FD93ULL
#define W7 0xA0761D6478BD642FULL
#define W8 0xE7037ED1A0B428DBULL

/* the angle the chained prisoner grinds into the pillar.  MUST be 0 < A < 64:
   that is exactly the condition under which no mark can cancel itself out. */
#define ANGLE 29

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }

/* ONE STROKE.  The prisoner never looks at the mark; he only feels how it
   tilts what he was already sharpening, and hands on the angle, not the mark.
   Add-rotate-xor only: no multiplication anywhere in the streaming path.
   Flipping bit b of w moves this by +/-2^((b+ANGLE)%64) +/- 2^b, never 0. */
static inline uint64_t stroke(uint64_t acc, uint64_t w) {
    return rotl64(acc ^ w, ANGLE) + w;
}

/* THE SILHOUETTE.  Drawn once, only after the flood has drained and the
   shadows have stopped chittering.  Two multiply stages = "the walls have
   settled"; a third would add nothing, which is the whole point.  Bijective,
   so no upstream difference is ever destroyed here. */
static inline uint64_t silhouette(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 32;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    const unsigned char * const e = data + len;

    /* ---- the water level: regime check, in the metaphor's own terms ----
       A thin pile of marks never raises the flood as far as the wires, so
       there is nothing to bend and nothing to cross.  The native then reads
       the shadow of a single wire.  Guards the small-input overhead of the
       eight-lane setup and the eight-step fold. */
    if (len < 64) {
        uint64_t h = W5 ^ ((uint64_t)len * W1);
        while ((size_t)(e - p) >= 8) { h = stroke(h, ld64(p));        p += 8; }
        if    ((size_t)(e - p) >= 4) { h = stroke(h, ld32(p) ^ W6);   p += 4; }
        while (p < e)                { h = stroke(h, (uint64_t)(*p++) ^ W7); }
        return silhouette(h);                 /* read once, kept alone */
    }

    /* ---- eight wires, anchored, fixed in number regardless of len ---- */
    uint64_t l0=W1, l1=W2, l2=W3, l3=W4, l4=W5, l5=W6, l6=W7, l7=W8;

#if defined(__AVX2__)
    /* The flood moves the wires per PASS, not per mark: 64 bytes enter the
       riverbed at once and bend all eight wires by the same ground angle. */
    {
        __m256i a = _mm256_set_epi64x((long long)W4,(long long)W3,(long long)W2,(long long)W1);
        __m256i b = _mm256_set_epi64x((long long)W8,(long long)W7,(long long)W6,(long long)W5);
        while ((size_t)(e - p) >= 64) {
            __m256i x0 = _mm256_loadu_si256((const __m256i *)(const void *)(p));
            __m256i x1 = _mm256_loadu_si256((const __m256i *)(const void *)(p + 32));
            __m256i t0 = _mm256_xor_si256(a, x0);
            __m256i t1 = _mm256_xor_si256(b, x1);
            a = _mm256_add_epi64(_mm256_or_si256(_mm256_slli_epi64(t0, ANGLE),
                                                 _mm256_srli_epi64(t0, 64 - ANGLE)), x0);
            b = _mm256_add_epi64(_mm256_or_si256(_mm256_slli_epi64(t1, ANGLE),
                                                 _mm256_srli_epi64(t1, 64 - ANGLE)), x1);
            p += 64;
        }
        uint64_t sa[4], sb[4];
        _mm256_storeu_si256((__m256i *)(void *)sa, a);
        _mm256_storeu_si256((__m256i *)(void *)sb, b);
        l0=sa[0]; l1=sa[1]; l2=sa[2]; l3=sa[3];
        l4=sb[0]; l5=sb[1]; l6=sb[2]; l7=sb[3];
    }
#else
    /* identical hash, eight independent chains, no SIMD required */
    while ((size_t)(e - p) >= 64) {
        l0 = stroke(l0, ld64(p     )); l1 = stroke(l1, ld64(p +  8));
        l2 = stroke(l2, ld64(p + 16)); l3 = stroke(l3, ld64(p + 24));
        l4 = stroke(l4, ld64(p + 32)); l5 = stroke(l5, ld64(p + 40));
        l6 = stroke(l6, ld64(p + 48)); l7 = stroke(l7, ld64(p + 56));
        p += 64;
    }
#endif

    /* the last marks of the pile, fed singly to the nearest prisoner */
    if ((size_t)(e - p) >= 8) { l0 = stroke(l0, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l1 = stroke(l1, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l2 = stroke(l2, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l3 = stroke(l3, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l4 = stroke(l4, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l5 = stroke(l5, ld64(p)); p += 8; }
    if ((size_t)(e - p) >= 8) { l6 = stroke(l6, ld64(p)); p += 8; }
    {
        uint64_t tw = 0; unsigned sh = 0;
        while (p < e) { tw |= (uint64_t)(*p++) << sh; sh += 8; }   /* never over-reads */
        l7 = stroke(l7, tw ^ ((uint64_t)sh * W6));
    }

    /* ---- THE SILHOUETTE: which lean, which stand straight, which cross
       another.  This is the only place the wires touch each other, and the
       only place a multiplication appears.  Every step is a bijection of v,
       so the nonzero difference guaranteed by stroke() survives to the end. */
    {
        uint64_t v = (uint64_t)len * W1;
        v = (v ^ rotl64(l0,  3)) * W2;
        v = (v ^ rotl64(l1, 11)) * W3;
        v = (v ^ rotl64(l2, 19)) * W4;
        v = (v ^ rotl64(l3, 27)) * W5;
        v = (v ^ rotl64(l4, 35)) * W6;
        v = (v ^ rotl64(l5, 43)) * W7;
        v = (v ^ rotl64(l6, 51)) * W8;
        v = (v ^ rotl64(l7, 59)) * W2;
        return silhouette(v);     /* read once; I keep nothing but the shadow */
    }
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 8.0

Stated before any measurement, with the reasoning so it can be falsified cleanly:

- **Throughput.** The reference does `xor; imul` per byte on one dependency chain — `imul` latency 3, so ~3 cycles/byte. Mine does 12 µops per 64 bytes on the AVX2 path (2 loads, 2 xor, 4 shifts, 2 or, 2 add) with two independent 4-cycle chains → L1/L2-resident should run at well under 0.1 cycles/byte, i.e. >30× on large buffers, falling to memory bandwidth for out-of-cache sizes and to ~3× on sub-64-byte inputs where the short path's serial strokes dominate. **8.0 is my estimate of the geometric mean across a mixed size sweep**, not the large-buffer peak. If the harness reports only a single large size, I expect 15–30× and my prediction is badly low; if it is dominated by ≤32-byte keys I expect 2–4× and my prediction is high.
- **Avalanche.** I predict **0.49–0.50** (≈ half the output bits flip). This is not a hope: single-bit flips are proved never to cancel in `stroke`, every downstream step is a bijection, and the final stage is Murmur3's `fmix64`, whose single-bit avalanche bias is known to be small. The specific failure I would be watching for is an avalanche *floor* near 0.5 but with a visible bias on the lowest-indexed bytes of very short inputs; the `len`-into-`v` seeding and the `sh * W6` tail marker are there to blunt that.

## MEASUREMENT

**Not measured. No tools were available in this session** — `hash_bench`, `hash_contract` and the rest were listed but the session carries none, so I could not compile, could not run the benchmark, and could not take any of the four allowed improvement iterations. I used **zero of four**.

I am reporting that plainly rather than filling this section with plausible numbers. What exists above is a prediction and a static argument, nothing more. The pipeline's compile-and-measure pass is the first real test this kernel will face. The honest status of each claim:

| Claim | Status |
|---|---|
| every single-bit input flip changes the output | **proved** statically (non-cancellation + bijective chain); not measured |
| avalanche ≈ 0.50 | **predicted**, rests on `fmix64`'s published bias; **not measured** |
| ~8× mean throughput vs. the FNV-1a reference | **predicted** from µop counts and latency; **not measured** |
| AVX2 and scalar paths agree bit-for-bit | **argued** (same 8-lane stripe map, same `ANGLE`); **not measured** |
| no over-read past `data + len` | **argued** by construction (byte loop for the `<8` tail); **not measured** |

## VERDICT

The metaphor's central bet — *spend almost nothing per mark, spend everything on one silhouette read at the end* — is the correct bet for this problem, and the mechanism landed on validated ground rather than novel ground: ARX strokes (ChaCha/BLAKE/SipHash family), xxHash-style striped lanes, Murmur3's `fmix64` finalizer. The seed's own "read once and kept alone" is what rules out the tempting error of adding a third finalizer stage or a per-byte multiply; the non-cancellation proof is what makes that safety claim rather than optimism. I consider the design sound and the *reasoning* for it stronger than the reasoning behind the stated known way.

Two named risks, both already addressed in the shipped artifact rather than left as caveats, per the rule that I may not ship a risk I name:

1. **"Overhead if small."** The eight-lane setup plus eight-multiply fold is dead weight below ~64 bytes. Guarded by the `len < 64` water-level check with a single-accumulator fallback that still ends in the same single silhouette. Not a comment — a branch.
2. **AVX2 may be absent.** Guarded by `#if defined(__AVX2__)` with a scalar eight-lane path computing the identical hash.

What I deliberately did **not** do: no OpenMP. The world contains one chained line and one flood; there is no second line to map a second thread onto, and 64-byte units are orders of magnitude below any threading threshold. Adding threads would have been me substituting my own idea for the native's.

Where this could still lose, stated so it can be checked rather than defended: if the harness measures only sub-32-byte keys, the short path's serial `stroke` chain gives maybe 2–4×, not 8×, and a dedicated 1–16-byte branch (two overlapping 8-byte loads, one fold, one `fmix64` — the wyhash short-input trick) would beat it. I left that out because the metaphor gives it no object: the native has one way of reading a thin pile, not two. If the measurement shows small-key throughput is what dominates the score, that is the honest next iteration — and it would be the first place the world's furniture runs out before the problem does.