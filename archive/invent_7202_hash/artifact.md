# MAPPING

**SEED 1 — "A growing sailor-chain hung between fire and wall makes every shadow depend on the whole row, not just the newest knot."**

| World object | Problem object |
|---|---|
| pile of marks, walked one at a time | input bytes, processed in order |
| fire | the fixed transform ("light") that projects state into an output |
| chained row of sailors | a small **array** of state lanes (the "chain"), not one scalar |
| tying one more knot, never breaking an old one | updating exactly one lane per byte, other lanes left untouched that step |
| wall | the register that receives the combined result |
| shadow on the wall | the *combined* value read off the whole lane-array, every step |
| "shadow is never just the new knot's shadow… whole chain hangs together or not at all" | the read-out must be a function of **all** lanes every round, not of the just-updated lane alone |

Silent assumption broken: **"the state is a single accumulator updated in place, one value."**

**SEED 2 — "A mark's own pebble-count sets the incense-smoke's swing, bending the firelight."**

| World object | Problem object |
|---|---|
| mark's strokes counted as pebbles | popcount (or raw value) of the byte |
| cup fill | a small data-dependent parameter |
| incense swing distance | a rotate/shift amount derived from the byte itself |
| smoke bending the fire's throw | a data-dependent rotation replacing a fixed multiply |

Assumption broken: **"mixing one byte requires a multiplication."**

**SEED 3 — "The wet-clay print taken from the settled shadow… is the only thing kept as the token."**

| World object | Problem object |
|---|---|
| settled (non-trembling) shadow | the raw accumulator once fully formed |
| clay print pressed once | a single finalization/avalanche pass |
| hardening in the stream | xor-shift-multiply diffusion of the print |
| throwing away smoke, pebbles, old print | discarding scratch state immediately after finalizing |

Assumption broken (weakly): "more mixing rounds always means better mixing" — replaced by "one settled read, finalized once, everything else discarded."

# CHOSEN SEED

**Seed 1** — most literal (an explicit growing chain, an explicit "whole-row" shadow) and most structurally different from FNV-1a/xxHash, which is exactly one scalar folded byte-by-byte. Seed 2 is just "swap multiply for rotate" inside the same single-accumulator skeleton; Seed 3 is basically the textbook finalizer already used everywhere. Seed 1 forces a genuinely different state shape.

# ASSUMPTION BROKEN

"The state is a single accumulator updated in place, one value." Here the state is an 8-lane **chain** array (`chain[8]`), and the value read out each round (`comb`) is always the full recombination of every lane — literally the shadow of the whole rope, not of the newest knot.

Concretely: each byte ties a knot onto exactly one lane (`idx = i & 7`), leaving the other 7 lanes untouched that step (never breaking an old knot), and `comb` is maintained as an always-exact XOR-fold of all 8 rotated lanes, updated incrementally (swap out the old lane's rotated contribution, swap in the new one) so it's cheap (O(1)/byte) yet mathematically always "the whole chain's shadow," never a partial one.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the rope: 8 sailors standing between fire and wall */
    uint64_t chain[8];
    static const uint64_t INIT[8] = {
        0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
        0x94D049BB133111EBULL, 0x2545F4914F6CDD1DULL,
        0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
        0x27D4EB2F165667C5ULL, 0x85EBCA6B85EBCA6BULL
    };
    static const int ROT[8] = {7,13,19,23,29,31,37,41};

    for (int j = 0; j < 8; j++)
        chain[j] = INIT[j] + (uint64_t)len * 0x9E3779B1ULL;

    /* comb is always the shadow of the WHOLE chain, kept exact and
       updated incrementally: never just the newest knot's shadow */
    uint64_t comb = 0;
    for (int j = 0; j < 8; j++) comb ^= rotl64(chain[j], ROT[j]);

    for (size_t i = 0; i < len; i++) {
        int idx = (int)(i & 7);
        uint64_t old = chain[idx];

        /* tie one more knot onto this sailor's spot in the rope;
           every other sailor's knot is left exactly as it was */
        uint64_t nw = old + (((uint64_t)data[i] + 0x9E3779B97F4A7C15ULL)
                              ^ rotl64(old, 11));
        chain[idx] = nw;

        /* the fire recasts the shadow of the whole row: remove the old
           lane's contribution, add the new one, so comb is always the
           full recombination of all 8 knots, not just this one */
        comb ^= rotl64(old, ROT[idx]) ^ rotl64(nw, ROT[idx]);
    }

    /* wait for the shadow to settle, press the clay, harden it in the
       stream -- one finalizing avalanche pass, nothing else kept */
    uint64_t h = comb ^ (uint64_t)len;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return h;
}
```

# PREDICTION

Per byte the FNV-1a baseline does ~2 cheap ops (xor + multiply). This kernel does ~8-9 cheap ops per byte (an add, an xor, two rotates for the tie, two more rotates + two xors for the incremental `comb` update), all single-cycle-class instructions with a short loop-carried dependency only through `comb` (a 64-bit XOR chain, low latency) and through `chain[idx]` (only re-touched every 8th byte, so there's real instruction-level parallelism across lanes). I expect the extra op count to dominate more than the ILP recovers.

PREDICTION: speedup_vs_dp = 0.6

(Avalanche expectation: the incremental whole-chain fold plus a murmur-style fmix64 finalizer should give close-to-ideal avalanche, roughly 0.48–0.50 fraction of output bits flipped per single input-bit flip.)

# MEASUREMENT

No tools were available in this session (the tool list shown is not actually invocable here — there is no `hash_bench` call I could make). I did not fabricate a number: I am reporting this plainly rather than inventing a measurement. The prediction above is written and frozen *before* any measurement, as required; actual throughput and avalanche numbers need to come from the external pipeline that compiles and runs `hash_bench` against this artifact. Consequently the "improve it up to four times" loop could not be executed in this session — there was no measured feedback to improve against, so I stopped at one artifact rather than pretend to iterate on numbers I don't have.

# VERDICT

Unverified pending external measurement. What I can state honestly:

- The mapping is literal: the single scalar accumulator of FNV-1a/xxHash was genuinely replaced by an 8-lane chain whose read-out is an always-exact function of every lane (the assumption "state is one accumulator" is actually broken, not just renamed).
- The design keeps the recombination O(1)/byte via incremental lane-swap rather than literally re-walking the whole chain every round (which would have been O(n·K) and dishonestly slow just to look literal) — so it should stay in the same complexity class as the known way, just with a larger constant factor.
- My honest expectation is a real slowdown (~0.6x baseline throughput) in exchange for state that is structurally richer than a single word, with avalanche quality that should be good but not verified.
- I am not claiming this beats FNV-1a/xxHash on speed — the seed doesn't point toward speed, it points toward *whole-chain dependence*, and that's what got built and predicted against, not silently swapped for the textbook method.