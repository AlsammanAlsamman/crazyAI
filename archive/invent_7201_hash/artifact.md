# MAPPING

**Seed 1 — the wax trough**

| World object | Problem object |
|---|---|
| Weighing Hall | the kernel / hashing process as a whole |
| pile of marks | input buffer `data[0..len)` |
| a mark | one byte `data[i]` |
| trough of grey wax at blood heat | a running pre-mix accumulator `W` that is never "finalized" between bytes |
| "each mark presses before the last shape has cooled" | `W` is updated by folding the new byte into the still-uncommitted (un-multiplied) impression of all prior bytes — no multiply happens at this stage |
| "never a clean picture of one mark, only the sum of every collision" | `W` after byte *i* is an inseparable overlap of bytes `0..i`, not a discardable per-byte value |

Broken assumption: **"mixing one byte requires a multiplication"** — the wax step mixes purely by rotate+xor overlap; the multiply is deferred to a later stage.

**Seed 2 — seven swarms and the forward-only wheel**

| World object | Problem object |
|---|---|
| seven bell-jars, each tuned to a different hunger | seven independent accumulator lanes `h[0..6]`, each with its own multiplicative constant `MUL[i]` |
| "each jar's swarm bites in its own count and rhythm" | lane's own rotate amount `ROT[i]` ("rhythm") and multiplier `MUL[i]` ("count/hunger") |
| dip the wire into the wax after every mark | feed the current wax state `W` into one lane each byte |
| slate wheel that only ever turns forward | a monotonically advancing lane-selector (`lane = i mod 7`), never decremented or reset |
| "no bite can be walked off once it's landed" | no update in the loop is ever undone; all lane state is strictly cumulative |

Broken assumption: **"the state is a single accumulator updated in place, one value"** — there are literally seven, not one.

**Seed 3 — the brine basin**

| World object | Problem object |
|---|---|
| basin of chilling brine | a one-time finalization/avalanche stage after the loop |
| "must not be rushed... wait until frost-lace stops spreading" | apply enough xor-shift/multiply avalanche rounds once, at the end, until diffusion saturates |
| "fixed lattice of six frost-flowers" | six pairwise-fold steps combining the seven lanes down to one 64-bit value |
| "sketched onto tin, never the wax... scraped and remelted" | the returned `uint64_t` is a fresh value; `h[]`/`W` are stack scratch, reinitialized every call |

Broken assumption: **"more mixing rounds always means better mixing"** (inverted: quality is concentrated in one deliberate final step, not spread evenly through the loop).

# CHOSEN SEED

Seed 2 — the seven tuned swarms and the forward-only wheel. It is the most literal (jar↦lane, hunger↦constant, rhythm↦rotate-amount, wheel↦round-robin index, all with concrete 1:1 correspondences) and the most different from the known way: FNV-1a/xxHash's whole design is *one* accumulator; this design is explicitly seven.

# ASSUMPTION BROKEN

"the state is a single accumulator updated in place, one value."

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* Seven jars: each tuned with its own "hunger" (multiplier) and
       "rhythm" (rotate amount) -- seven independent accumulator lanes. */
    static const uint64_t MUL[7] = {
        0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
        0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL, 0x2545F4914F6CDD1DULL,
        0xD6E8FEB86659FD93ULL
    };
    static const int ROT[7] = { 13, 29, 41, 17, 53, 7, 37 };

    uint64_t h[7];
    const uint64_t OFFSET = 1469598103934665603ULL;
    for (int j = 0; j < 7; j++) {
        h[j] = OFFSET ^ ((uint64_t)(j + 1) * 0x9E3779B97F4A7C15ULL);
    }

    uint64_t W = 0;   /* the grey wax: overlapping impressions, not yet cooled */
    int lane = 0;     /* the forward-only wheel's current notch */

    for (size_t i = 0; i < len; i++) {
        /* the mark presses into the wax before the last shape cools */
        W = rotl64(W, 1) ^ (uint64_t)data[i];

        /* dip the wire into the wax; let this jar's swarm bite it,
           in its own count (multiplier) and rhythm (rotate amount) */
        h[lane] ^= W;
        h[lane]  = rotl64(h[lane], ROT[lane]);
        h[lane] *= MUL[lane];

        /* the wheel turns forward, never back */
        lane++;
        if (lane == 7) lane = 0;
    }

    /* the brine basin: freeze the seven jars' bites into six frost-flowers */
    uint64_t acc = h[0];
    for (int j = 1; j < 7; j++) {
        acc ^= h[j];
        acc  = rotl64(acc, ROT[j]);
        acc *= MUL[j];
        acc ^= acc >> 31;
    }

    /* wait for the frost-lace to stop spreading before lifting the token */
    acc ^= (uint64_t)len;
    acc ^= acc >> 33;
    acc *= 0xFF51AFD7ED558CCDULL;
    acc ^= acc >> 33;
    acc *= 0xC4CEB9FE1A85EC53ULL;
    acc ^= acc >> 33;

    /* only the tin sketch survives; the wax (h[], W) is scratch, discarded */
    return acc;
}
```

# PREDICTION

Reasoning: the baseline FNV-1a chain (`h = (h^byte) * prime`) is a *single* sequentially-dependent multiply chain — every byte's update must wait on the previous one, so it is latency-bound (~multiply latency, several cycles) per byte, which is exactly why real fast hashes (xxHash) use multiple lanes. This kernel's true serial dependency is only the cheap `W` chain (rotate+xor, ~1–2 cycle latency); each of the seven `h[lane]` multiply-chains is only revisited every 7th byte, so the CPU can keep several of those chains in flight and hide their multiply latency behind each other. That should let this design beat a purely-serial single-accumulator multiply chain on latency, even though it issues more total instructions per byte (extra rotate, extra branch for the wheel wraparound). Net: modest speedup expected, tempered by the non-power-of-2 (mod-7) branch and higher instruction count offsetting some of the ILP gain. Avalanche is expected to be strong (close to ~50%) because of seven differently-tuned lanes plus a six-step fold plus a full finalizer.

PREDICTION: speedup_vs_dp = 1.2

# MEASUREMENT

No tools were available in this session (per the environment note, this turn runs without `hash_bench`/`hash_contract` access), so I cannot execute the measurement myself. The prediction above is recorded before any measurement, as required; the external pipeline compiling with `gcc -O3 -march=native -fopenmp -lm` will report the actual throughput and avalanche score.

# VERDICT

Unverified pending the pipeline's run. The design is a literal, traceable translation of the "seven swarms + forward-only wheel" mechanism (seven tuned lanes replacing the single accumulator, combined by seven wax-mediated dips and six frost-flower fold steps, sealed by one careful brine finalizer). It should score well on avalanche given the multi-lane diversity and finalization, and I predict it is roughly on par with or modestly faster (~1.2x) than the single-accumulator FNV-1a baseline because it breaks the serial multiply-dependency chain that bottlenecks the known way — but this is a prediction from first-principles latency/throughput reasoning only, not a measured result, and should be treated as such until the pipeline reports real numbers.