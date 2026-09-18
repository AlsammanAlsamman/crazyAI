## MAPPING

**SEED1** — "A chained line of prisoners sharpens a blank pillar mark by mark, each stroke's angle carrying forward from every stroke before it."

| world object | problem object |
|---|---|
| pile of marks, in given order | `data[0..len-1]`, read in original order |
| chained line of prisoners | the sequence of loop iterations of one running accumulator |
| blank pillar | the accumulator register `A` |
| "feels how it changes the angle... passes the angle, not the mark" | update rule `A = f(A, byte)`; only `A` survives to the next step, never the raw byte |
| "a mark dropped at the start still trembles in the last hand" | full dependency chain of multiplicative folding — every byte influences the final `A` |

Assumption broken: **none decisively.** This seed just re-narrates "single accumulator, sequential, in order, multiply-based" — it *is* the known FNV/xxHash core loop in different words. Rejected as the chosen seed for that reason.

**SEED2** — "A flooding river that never recedes until all marks have passed bends anchored wire shapes by the angle ground into the pillar on each pass."

| world object | problem object |
|---|---|
| flood rises, does not recede until every mark has gone through | a barrier: the full-buffer fold must run to completion before its result is used at all |
| "a token pulled while the water is still up is worthless" | partial/intermediate accumulator values are never read out as output |
| tiny wire shapes, anchored in the riverbed, past where any traveler has fused its source | a fixed array of N=64 constants, independent of the input, untouched during the per-byte loop |
| "they do not move themselves; the flood moves them, bending each wire by exactly the angle" | one broadcast update: after the fold finishes, the *same* final angle `A` transforms **all** 64 anchors in one batch step |

Assumption broken: **"the state is a single accumulator updated in place, one value."** Output is carried by a plural, fixed set of registers, updated by one broadcast, not by an incrementally-mutated scalar.

**SEED3** — "The token is the fixed silhouette the wires throw against the sun once the flood drains and the shadows stop chittering, read once and kept alone."

| world object | problem object |
|---|---|
| "which lean, which stand straight, which cross another" | each output bit comes from *observing* (comparing/sign-testing) a finalized wire, not from returning a raw value |
| "read once and kept alone" | the readout happens exactly once, after finalization |
| "marks forgotten, pillar sharpens onward, floodwater drains unrecorded" | no per-byte history is kept; only the final wire states matter |
| "insect-noise... walls still settling" | don't trust the result until it has actually finished mixing — motivates a *strong* one-shot finalizer rather than a weak one |

Assumption broken: **"more mixing rounds always means better mixing."** Quality here comes from one strong, wide finalization step over a fixed-size set of registers, not from repeating passes over the data.

## CHOSEN SEED

SEED2 — most literal (barrier + fixed anchor array + one broadcast update) and structurally furthest from FNV/xxHash's single mutable scalar.

## ASSUMPTION BROKEN

"The state is a single accumulator updated in place, one value" is replaced by: one scalar fold over the bytes (unavoidable — the chain still has to read the pile in order) produces a single **angle**, which is never itself the output; instead it is broadcast, once, into 64 fixed anchor registers ("wires"), and the output is read as a *silhouette* (bit pattern) over those 64 registers, not as the raw accumulator.

## ARTIFACT

Object mapping, kept literal:
- **byte** = one element of `data[]`, fed "singly" to the fold, in order.
- **the pillar / its angle** = the single scalar `A`, updated by xor+multiply per byte (this part deliberately matches the known method — SEED1 doesn't ask us to change it).
- **the flood recedes / token pulled too early is worthless** = `A` is only used *after* the loop over the whole buffer has completed — never mid-loop.
- **the 64 anchored wires** = fixed, input-independent 64-bit constants `GOLDEN*(i+1)`, never touched during the per-byte loop.
- **"bending every wire by exactly the angle"** = every wire is XORed with the *same* `A` (one broadcast, same value for all 64).
- **"they do not move themselves"** = wires never depend on anything but `A` and their own fixed anchor/rotation — no cross-wire interaction.
- **the silhouette (lean / straight / cross)** = one observed bit per wire (its sign/top bit) after a strong settle-mix, assembled into the 64-bit hash.
- **"read once and kept alone"** = the settle-mix (`bend`) runs exactly once per wire, at the end, nothing is looped back into `A`.

```c
#include <stdint.h>

#define NWIRE 64

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    /* r is constructed to always lie in [1,63] -- no shift-by-0/64 UB */
    return (x << r) | (x >> (64 - r));
}

/* the "settling": a strong two-round avalanche mixer, applied once,
   per wire, at readout time -- never repeated, never applied mid-buffer */
static inline uint64_t bend(uint64_t v) {
    v ^= v >> 30;
    v *= 0xBF58476D1CE4E5B9ULL;
    v ^= v >> 27;
    v *= 0x94D049BB133111EBULL;
    v ^= v >> 31;
    return v;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the chained line of prisoners: ONE accumulator, sharpened mark by
       mark, in the given order -- this is the pillar and its angle */
    uint64_t A = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        A ^= (uint64_t)data[i];
        A *= 1099511628211ULL;
    }
    /* A is the angle ground into the pillar on this pass. It is never
       returned directly -- "a token pulled while the water is still up
       is worthless." We only use A after the loop above has fully
       finished, i.e. after every mark has gone through. */

    /* the flood: rises once, after every mark has gone through, and
       bends every one of the 64 anchored wires by exactly that same
       angle A. The wires never move themselves and are never touched
       during the loop above -- each is anchored at its own fixed,
       input-independent position. */
    uint64_t hash = 0;
    const uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;
    for (unsigned i = 0; i < NWIRE; i++) {
        uint64_t anchor = GOLDEN * (uint64_t)(i + 1);   /* wire i's own
                                                            fixed anchor */
        unsigned r = (i % 63) + 1;                       /* 1..63 */
        uint64_t bent = rotl64(anchor ^ A, r);           /* the SAME
                                                            angle A bends
                                                            every wire */
        bent = bend(bent);                               /* let the
                                                            shivering
                                                            settle */
        /* the silhouette: read once whether this wire leans one way */
        hash |= (bent >> 63) << i;
    }
    return hash;                    /* the shadow-shape, kept alone */
}
```

## PREDICTION

The per-byte loop is byte-for-byte identical in cost to `dp` (one xor, one multiply per byte). The only added cost is a fixed ~64-iteration finalization (rotate + one multiply-based anchor + a 2-round splitmix-style mixer), independent of `len`. For buffers of any realistic benchmark size this overhead is a small, non-scaling fraction of total work, so I expect throughput to be close to but slightly below `dp`.

Avalanche: `A` (however weakly FNV's own diffusion mixes a single flipped input bit) is piped through 64 independent applications of a strong 2-round avalanche mixer (splitmix64-style), each keyed by a different anchor/rotation. Even a partial, few-bit difference in `A` should be amplified by `bend()` into ~50% output-bit difference per wire, largely independently across wires — I expect avalanche score noticeably closer to 0.5 than plain FNV-1a achieves on its own.

PREDICTION: speedup_vs_dp = 0.9

## MEASUREMENT

Not executed: no tools (`hash_bench`, `hash_contract`) are available in this session, so I did not run the harness myself — the prediction above is analytical, not measured. Per the environment note, the external pipeline is expected to compile and measure this kernel; I have not fabricated a throughput or avalanche number in its place, and I did not use the "at most four improvement" iterations since there is no measurement loop to react to here.

## VERDICT

The design is a literal, checkable translation of SEED2 (barrier → single broadcast angle → fixed anchor array → one-shot readout), breaking the "single mutable accumulator is the output" assumption while deliberately leaving the per-byte fold (SEED1's territory) untouched to avoid adding serial cost. Expected outcome: throughput ≈ dp (small fixed overhead, not scaling with `len`), avalanche meaningfully better than raw FNV-1a because the weak per-byte diffusion is passed through a strong, independently-keyed finalizer 64 times before being read. This is a prediction, not a result — actual verdict is pending a real `hash_bench` run.