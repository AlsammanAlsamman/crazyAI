# MAPPING

**SEED 1 — counting-bars / eight violin strings under one bridge**

| World object | Problem object |
|---|---|
| counting-bars | the persistent state substrate the whole fold lives on |
| eight violin strings | 8 independent `uint64_t` lanes `s[0..7]` (not one accumulator) |
| "bound under one bridge" | an all-to-all coupling step run after every pluck: reading all 8 lanes into one combined "bridge" signal and re-injecting it into all 8 lanes |
| "pluck the string nearest the mark's weight" | select a lane index from a function of the byte |
| "never softly, always the same force" | the operator applied at the plucked lane (rotate amount, fold structure) is a fixed constant, never data-dependent |
| "every string shivers... before it settles" | diffusion step touches *all* lanes every iteration, not just the selected one |

Breaks: **"the state is a single accumulator updated in place, one value."**

**SEED 2 — feather / ink-weight, never repeated**

| World object | Problem object |
|---|---|
| inkwell that never runs dry | a fixed nonlinear byte→weight map, reusable for every byte value |
| fresh child-feather, "no two marks share a feather" | a scratch temporary computed per byte, discarded immediately after use |
| "the feather does not record the mark's shape, only its weight" | the byte is never used raw; it is first passed through an avalanching transform before touching any state |

Breaks: **"mixing one byte requires a multiplication"** — the *per-byte hot-loop* step is a table lookup, not a multiply (multiplication is pushed into a one-time 256-entry setup instead).

**SEED 3 — the storm's single pass / the pickers**

| World object | Problem object |
|---|---|
| storm passing once over the final tremor | exactly one finalization pass over the 8 lanes, no iterative avalanche rounds |
| "eight small notes, no more" | 8 lanes → 8 output bytes → one `uint64_t` |
| pickers destroying feathers/drops | scratch state (lookup table, temporaries) goes out of scope, nothing but the token survives |

Breaks: **"more mixing rounds always means better mixing"** — quality comes from the *bridge coupling accumulated during the fold*, not from repeating the finalizer.

# CHOSEN SEED

**SEED 1** (counting-bars / eight strings under one bridge). It is the most literal — it names a concrete plural structure (eight strings) and a concrete concrete coupling mechanism (the bridge) — and it is the most structurally different from FNV-1a/xxHash, which are defined entirely by having *one* accumulator. SEED 2 and 3 are folded in as supporting design choices (no multiply in the hot loop; single-pass finalizer) but SEED 1 is the mixing engine.

# ASSUMPTION BROKEN

"The state is a single accumulator updated in place, one value." Here the state is an 8-lane vector, and every byte's effect is deliberately spread across all 8 lanes via the bridge before the next byte is read.

# ARTIFACT

Full literal mapping: byte = mark; per-byte weight lookup = ink dip onto a fresh feather (built once, reused, never re-derived mid-loop so the hot loop touches no multiply); `s[0..7]` = the eight violin strings on the counting-bars; picking `lane = (w>>61)&7` = plucking the string nearest the weight; XOR of `w` into `s[lane]` + a **fixed** rotation = the "always the same force" pluck; the two bridge passes (fold all lanes into one signal, then spread that signal back into all lanes with per-lane fixed rotations) = "a shudder in one bends the pitch of all the rest"; the loop never resets `s[]` between bytes = the strings never go fully still; the final single pass folding each lane (mixed with its neighbor) down to one byte = the storm collapsing the tremor into eight notes; the lookup table and temporaries live only on the stack and vanish at return = the pickers.

```c
#include <stdint.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the counting-bars: eight violin strings, bound under one bridge */
    uint64_t s[8] = {
        0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL,
        0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL,
        0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL,
        0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL
    };
    const int PLUCK_ROT = 23;                        /* the force: never varies */
    const int BRIDGE_ROT[8] = {7,13,17,23,29,37,41,47};

    /* the inkwell + a fresh feather per mark: a nonlinear byte -> weight map,
       built once so the hot per-mark loop needs no multiplication */
    uint64_t W[256];
    for (int b = 0; b < 256; b++) {
        uint64_t w = (uint64_t)(b + 1) * 0x9E3779B97F4A7C15ULL;
        w ^= w >> 33;
        w *= 0xBF58476D1CE4E5B9ULL;
        w ^= w >> 29;
        W[b] = w;
    }

    for (size_t i = 0; i < len; i++) {
        uint64_t w = W[data[i]];                 /* dip the mark, read its weight */
        int lane  = (int)((w >> 61) & 7u);        /* pluck the string nearest the weight */

        s[lane] ^= w;                             /* the pluck: */
        s[lane]  = rotl64(s[lane], PLUCK_ROT);     /* always the same force */

        /* the bridge: a shudder in one bends the pitch of all the rest */
        uint64_t bridge = 0;
        for (int k = 0; k < 8; k++)
            bridge ^= rotl64(s[k], BRIDGE_ROT[k]);
        for (int k = 0; k < 8; k++)
            s[k] ^= rotl64(bridge, BRIDGE_ROT[7 - k]);
        /* the strings never go fully still between marks */
    }

    /* the storm passes once over the final tremor, dissolving it into eight notes */
    uint64_t token = 0;
    for (int k = 0; k < 8; k++) {
        uint64_t note = s[k] ^ rotl64(s[(k + 1) & 7], (k * 11 + 7) & 63);
        note ^= note >> 32; note ^= note >> 16; note ^= note >> 8;
        token |= (note & 0xFFULL) << (8 * k);
    }
    /* the pickers: W[], s[], bridge, note all fall out of scope here */
    return token;
}
```

# PREDICTION

PREDICTION: speedup_vs_dp = 0.12

Reasoning: FNV-1a spends ~2 ops/byte (one xor, one multiply). This kernel spends ~1 table lookup + 2 ops (pluck) + 32 ops (two 8-wide bridge passes) per byte — roughly 15–20x more arithmetic per byte, only partly offset by the fact that the hot loop has no multiply and the 2KB lookup table stays resident in L1, plus `-march=native` may auto-vectorize the fixed 8-lane bridge loops somewhat. Net expectation: markedly slower than the baseline, so I predict ~0.1–0.15x throughput. In exchange I expect avalanche to be very strong (near the ideal 0.5), since every byte's weight is fully absorbed into one lane and then broadcast, undiluted, into every other lane before the next byte arrives — a single flipped input bit changes which lane gets plucked and/or the weight injected at that step, and that divergence is then carried through the bridge into all 8 lanes for every subsequent byte.

# MEASUREMENT

Not run: no tools (`hash_bench`/`hash_contract`) were available in this session, so no live throughput or avalanche numbers were collected here. The prediction above is the honest pre-measurement estimate; actual numbers come from the downstream pipeline that compiles and benchmarks this artifact. I did not perform the "measure, improve up to four times" loop for that reason — this is a single, unmeasured iteration, not a claim of a tuned result.

# VERDICT

Unverified. The design is a faithful, literal translation of the eight-strings-under-one-bridge image (multi-lane state, all-to-all diffusion every byte, single-pass finalizer, no multiply in the hot loop) and should break the single-accumulator assumption cleanly with strong avalanche — but at a real, predictable throughput cost from doing 8-lane diffusion on every single byte rather than only at the end. If the actual measurement contradicts the prediction (e.g., vectorization makes the bridge cheaper than expected, or avalanche is worse than expected because only 3 bits select the lane), that should be reported as plainly as a confirming result — this seed was chosen for literalness and contrast with FNV/xxHash, not because it was guaranteed to win on speed.