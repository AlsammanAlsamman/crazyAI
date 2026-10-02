## MAPPING

| World object (native's words) | Problem object | Why this and nothing else |
|---|---|---|
| pile of marks, in their given order | the byte buffer `data[0..len)`, read in order | "given order" = he never reorders or skips |
| laying marks on the trail one by one | sequential absorption of the buffer | |
| **the sphere** at the trail's high mouth | **one** state register, initialized to fixed constants | "my only memory" = no side tables, no second accumulator |
| the sphere's **face** (a face, not a point) | the register's lanes — a 256-bit body with four 64-bit facets | a sphere is a body; a face takes several impressions at once |
| pressing a mark's shape into the face | `state ^= loaded_input` | "press/imprint" = XOR, the imprint that can be undone |
| letting it go; it **tumbles a fixed count of turns** | a fixed number of rotations of the state (spin about its axis = per-lane bit-rotate; progress along the coil = the whole body rolling, carrying far facets round to the face) | "turn" = rotation. Nothing here multiplies. |
| **coiled** limestone trail (a coil, not a straight run) | the rotation is cyclic and wraps across the whole 256-bit body, not a one-way shift | |
| striking a stalk → **sobs once, cracking a hairline into itself** | the rolled copy is struck back into the body: `s += rot(s)` / `s ^= rot(s)` (add = the carry, the "sob" that is not reversible lane-locally) | the crack is *in itself*, i.e. self-mixing, not mixing with an outside number |
| "the angle of that crack is the next weight to carry forward" | the post-round state is the only thing carried | |
| footprint at a stalk, never looked back at | the pre-round state is dead: no history buffer, no second pass, no backtracking | |
| one wrong mark → **whole sphere reshapes**, every stalk downstream sobs differently | avalanche: a one-bit input change must change essentially all output bits | |
| the **last, smallest crack in the final stalk** is handed over | a single strong finalization applied once at the end; the 64-bit return value | |
| stone dust, discarded footprints, eggshells swept into the plumbing | intermediate states are *not* required to be strong, and are discarded | |
| what "a priest would call true" | the reference/validated construction (the thing that is allowed to be the authority) | |

Which silent assumption each seed breaks:

- **SEED 1** (sphere = sole carried memory): breaks nothing new about multiplication; it *keeps* "one accumulator" but widens what one value means — one body, four facets. It is the least different from the known way (FNV also has one accumulator).
- **SEED 2** (a *fixed count of tumbles* mixes each mark): breaks **"mixing one byte requires a multiplication"** — a tumble is a rotation, a crack is a self-XOR/self-add; there is no multiply anywhere in the native's account. Secondarily it breaks **"more mixing rounds always means better"** — the count is fixed, "no more, no fewer", and never scales with the pile.
- **SEED 3** (only the final crack is kept, intermediates swept away): breaks **"more mixing rounds always means better mixing"** and **"each byte must be mixed into the running state before the next byte is read"** — strength is deferred to one place, so per-mark work can be cheap and wide.

## CHOSEN SEED

**SEED 2** — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken." It is the seed the instructions prefer (it is the one that breaks the multiplication assumption), and its mapping is the most mechanical: tumble = rotate, strike = self-XOR/self-add, fixed count = fixed round number. SEED 3 is used only as the licence it literally grants (intermediate faces may be weak), not as the chosen mechanism.

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."** The kernel contains **no multiply at all** — absorption and finalization are pure add–rotate–xor.

And per step 4, I let the mechanism *land on* the validated real-world technique rather than inventing: multiplication-free ARX mixing with a one-shot strong finalization is exactly **SipHash** (a four-word state, `SipRound` = add/rotate/xor only, 4 finalization rounds). So the native's "read the angle of the crack" round **is** `SipRound`, his sphere's four facets **are** `v0..v3`, and his "last smallest crack in the final stalk" **is** SipHash's finalization. The only thing I add on top is his wide 32-byte-per-press coil trail for long piles, which SEED 3 licenses.

Two regimes are named in the known-way section (FNV-1a one-byte-at-a-time vs. xxHash wide striping), so the native must recognize the regime in-world: **can the sphere complete a full coil?** If the pile is shorter than one coil (`len < 32`) he uses the *hand trail* — one word per press — which is literally validated SipHash-1-4. Otherwise he uses the coil, then finishes the remainder on the hand trail. This also guards the only risk my own verdict names (vector setup/over-read overhead on short piles): below 32 bytes the vector path is never entered. No OpenMP: the metaphor has exactly one sphere, and at benchmark sizes a thread fork costs more than the whole absorb.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROTL(x, b) (((uint64_t)(x) << (b)) | ((uint64_t)(x) >> (64 - (b))))

/* reading the angle of the crack: add-rotate-xor only, no multiplication.
   This is SipHash's round -- the validated multiply-free mixer. */
#define SIPROUND(v0, v1, v2, v3) do {                       \
    v0 += v1; v1 = ROTL(v1, 13); v1 ^= v0; v0 = ROTL(v0, 32); \
    v2 += v3; v3 = ROTL(v3, 16); v3 ^= v2;                  \
    v0 += v3; v3 = ROTL(v3, 21); v3 ^= v0;                  \
    v2 += v1; v1 = ROTL(v1, 17); v1 ^= v2; v2 = ROTL(v2, 32); \
} while (0)

static inline uint64_t load64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    size_t n = len;

    /* the sphere set at the trail's high mouth: one body, four facets */
    uint64_t v0 = 0x736f6d6570736575ULL;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL;

#if defined(__AVX2__)
    /* REGIME TEST: can the sphere complete a full coil?  (>= 32 marks) */
    if (n >= 32) {
        __m256i s = _mm256_set_epi64x((long long)v3, (long long)v2,
                                      (long long)v1, (long long)v0);
        do {
            /* press 32 marks into the sphere's face, then let it go */
            s = _mm256_xor_si256(s, _mm256_loadu_si256((const __m256i *)p));

            /* THREE TURNS -- no more, no fewer.  Each turn ends on a stalk,
               and the sphere cracks itself there.  No multiplication. */

            /* turn 1: spin about the axis (23), struck in with a carry */
            s = _mm256_add_epi64(s, _mm256_or_si256(_mm256_slli_epi64(s, 23),
                                                    _mm256_srli_epi64(s, 41)));
            /* turn 2: spin the other arc (41), struck in as a hairline */
            s = _mm256_xor_si256(s, _mm256_or_si256(_mm256_slli_epi64(s, 41),
                                                    _mm256_srli_epi64(s, 23)));
            /* turn 3: roll along the coil -- the far side of the body comes
               round to the face, so facets do not stay separate tracks */
            {
                __m256i a = _mm256_permute4x64_epi64(s, 0x93); /* lane i <- lane i-1 */
                s = _mm256_add_epi64(s, _mm256_or_si256(_mm256_slli_epi64(s, 13),
                                                        _mm256_srli_epi64(a, 51)));
            }
            p += 32; n -= 32;
        } while (n >= 32);

        {   /* the sphere, unchanged in identity, handed to the hand trail */
            uint64_t t[4];
            _mm256_storeu_si256((__m256i *)t, s);
            v0 = t[0]; v1 = t[1]; v2 = t[2]; v3 = t[3];
        }
    }
#endif

    /* the hand trail: small pile, or the tail a coil cannot hold.
       One word per press, one tumble each -- validated SipHash-1-x. */
    while (n >= 8) {
        uint64_t m = load64(p);
        v3 ^= m; SIPROUND(v0, v1, v2, v3); v0 ^= m;
        p += 8; n -= 8;
    }
    {
        uint64_t b = ((uint64_t)len) << 56;
        switch (n) {
            case 7: b |= (uint64_t)p[6] << 48; /* fall through */
            case 6: b |= (uint64_t)p[5] << 40; /* fall through */
            case 5: b |= (uint64_t)p[4] << 32; /* fall through */
            case 4: b |= (uint64_t)p[3] << 24; /* fall through */
            case 3: b |= (uint64_t)p[2] << 16; /* fall through */
            case 2: b |= (uint64_t)p[1] <<  8; /* fall through */
            case 1: b |= (uint64_t)p[0];       /* fall through */
            case 0: break;
        }
        v3 ^= b; SIPROUND(v0, v1, v2, v3); v0 ^= b;
    }

    /* the last, smallest crack in the final stalk -- the only token kept.
       Everything else (every intermediate face) is swept away on purpose. */
    v2 ^= 0xffULL;
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;
}
```

## PREDICTION

Stated before any measurement (and no measurement was possible in this session, so it is stated before all of them):

**PREDICTION: speedup_vs_dp = 8.0**

Derivation, so it is falsifiable rather than vibes: FNV-1a's critical path is `xor`(1) + `imul`(3) = **4 cycles per byte**. My coil path's critical path per 32-byte press is `xor`(1) + turn1(shift 1, or 1, add 1 = 3) + turn2(3) + turn3(`permute4x64` 3 → `srli` 1 → `or` 1 → `add` 1 = 6) ≈ **13 cycles per 32 bytes = 0.41 cycles/byte** → 4 / 0.41 ≈ 9.8×, discounted to ~8× for loop overhead, the fixed ~45-cycle finalization, and possible DRAM-bandwidth clipping at ~10 GB/s for buffers above L2.

Secondary prediction: **avalanche ≈ 0.50 flip rate (near-ideal, ≥0.99 on a 0–1 quality scale)** — a one-bit flip anywhere reaches the finalizer as a nonzero word difference, and 1 compression + 4 `SipRound`s is SipHash-2-4's own validated full-avalanche finalization.

Predicted failure mode if I am wrong: `permute4x64_epi64` is 3-cycle latency *and* 1/cycle throughput on the serial chain, so if the measured machine's port pressure is worse than assumed, turn 3 alone could push the chain to ~19 cycles/32 B and the speedup down to ~5–6×. If speedup comes back below 3×, the likely cause is memory bandwidth, not the mixer, and the honest fix would be a wider sphere (AVX-512) — **not** more tumbles.

## MEASUREMENT

**Not performed.** `hash_bench` / `hash_contract` were unavailable in this session (no tools), so I have no numbers and will not manufacture any. The artifact above is submitted unmeasured and unrevised — zero of my allowed four improvement rounds were used, because an improvement round without a measurement is just redecorating.

What the pipeline should report if my model of the machine is right: throughput ≈ 7–10× the FNV-1a reference for buffers of a few KB and up, ≈1–2× at 64–256 bytes, and ≈0.6–1.0× below 16 bytes (where the fixed finalization dominates); avalanche ≈ 0.5.

## VERDICT

The native's account is, read literally, **not a weaker cousin of the textbook method — it is SipHash**, and that is the finding. Every object landed on something real with no slack: sphere → four-word state; tumble → rotation; crack → self-add/self-xor; "no more, no fewer" → a fixed round count that never grows with the pile; "the last smallest crack is what I hand over" → one-shot finalization with all intermediate faces discarded; "I never look back at the footprint" → no history, no second pass. The multiplication that the standard solution treats as mandatory never appears anywhere, and does not need to: the carry in `+` supplies the nonlinearity that `*` was being used for, at 1-cycle latency instead of 3. That the metaphor converged on a published, cryptanalyzed construction rather than a novelty is the strongest evidence that the native was describing something true.

The one place I extended him is the coil trail: pressing 32 marks per release instead of 8. That is licensed by his own sweeping-away of intermediate cracks (SEED 3), and it is where the speed comes from.

Honest weaknesses, named rather than hidden:

1. **Below one coil the coil is a liability.** Named and *guarded*: `if (n >= 32)` gates the entire vector path, and the fallback is the strictly simpler hand trail. Under 32 bytes the kernel is byte-for-byte the validated SipHash-1-4 and nothing of mine runs at all.
2. **The fixed ~45-cycle final crack cannot be amortized on tiny inputs.** For `len < 16`, this kernel may be *slower per byte* than FNV-1a. I did not "guard" that by falling back to FNV, because FNV-1a has no finalization and therefore essentially no avalanche on short inputs — falling back would trade away the exact property being measured. This is a real cost, accepted deliberately, not an oversight: the single final crack is the native's central claim, and dropping it on short inputs would be quietly replacing his idea with the textbook one.
3. **During absorption, the four facets are not yet one body in the strong sense.** Turn 3 does carry facets around, but a flip in the *last* chunk relies on the finalizer for full diffusion. That is exactly what the native says happens — his claim "the whole sphere reshapes" is a claim about the token handed over, not about every intermediate face, which he explicitly sweeps into the plumbing. If the measured avalanche comes back low (< 0.45), this is the first thing I would be wrong about, and the correct repair is a fifth `SipRound` in the finalizer — still no multiplication.