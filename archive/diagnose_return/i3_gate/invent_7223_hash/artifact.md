## MAPPING

**SEED 1 — "A rolling sphere serves as the sole carried memory across the whole ordered pile."**

| world object | problem object |
|---|---|
| the pile of marks, in their given order | the byte buffer `data[0..len)`, read in order |
| the sphere at the trail's high mouth | one `uint64_t s`, initialized at entry |
| "my running weight, my only memory" | exactly one live accumulator register — no lanes, no tree, no side array |
| "pressing a mark into the sphere's face" | absorbing an input word into `s` (XOR) |
| "the old face I let shrink and go" | in-place update; no history, no per-step buffer |
| "a footprint at a stalk is a woman's business, I never look back" | intermediate values are dead immediately; nothing is stored or re-read |

*Assumption broken:* **none of the five.** This seed *affirms* (a), (b) and (d). What it actually forbids is the thing the fast known way really does — xxHash's 4 (or SIMD's 8) parallel accumulators. It is an anti-striping constraint, and it is exactly the constraint my previous attempt violated.

**SEED 2 — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."**

| world object | problem object |
|---|---|
| a mark | one 64-bit word — the sphere's *face* is 64 bits wide, so one press covers a whole face |
| the coiled limestone trail, shadow already laid white | a fixed, pre-laid round structure (constants known at compile time) |
| a stalk the sphere strikes | one round step, carrying its own constant `STALK_x` |
| "it sobs once" | `s += stalk` (carry propagation — the nonlinearity) |
| "cracking a hairline into itself" | `s ^= s >> bleed` (the high face bleeds into the low) |
| **"I read the angle of that crack as the next weight to carry forward"** | **the rotation amount is read off the state itself: `s = rotl(s, s & 63)` — a data-dependent rotation** |
| "a fixed count of turns, no more, no fewer" | `T = 2` tumbles per mark, invariant, not length-scaled |
| "the whole sphere reshapes, not just a sliver" | whole-64-bit-word update; no half-word Feistel, no byte-lane update |
| "one wrong mark and every stalk downstream sobs differently" | once a difference reaches the low 6 bits, the *rotation amount* changes and the entire word re-registers — avalanche is triggered, not accumulated |

*Assumption broken:* **(c) "mixing one byte requires a multiplication"** — the mixing agent is a *crack angle*, i.e. a rotation amount taken from the datum itself. There is no `imul` anywhere in this kernel. Secondarily (e): the count is fixed and deliberately small (2 per 8 bytes, vs FNV's 8 multiplies per 8 bytes).

**SEED 3 — "Only the final hairline crack in the last stalk is kept as the token, while all intermediate cracks and dust are swept away."**

| world object | problem object |
|---|---|
| "I don't keep the sphere itself" | the running accumulator is *not* the return value |
| "the last, smallest crack in the final stalk" | a separate, one-shot finalization applied after the last press |
| "small as a token" | the 64-bit emitted digest |
| stone dust, discarded footprints, eggshells, the jungle's plumbing | tail bytes, loop temporaries, every intermediate `s` — dead, never emitted |

*Assumption broken:* **(e) "more mixing rounds always means better mixing."** No — rounds pay off where they are *free*: the per-mark cost is O(len) so it stays at 2, while the finalization is O(1) so it can be heavy. Mixing quality is bought at the end, not per byte.

## CHOSEN SEED

**SEED 2**, the fixed count of tumbles with the crack angle carried forward. It is the only one of the three that breaks the preferred assumption ("mixing one byte requires a multiplication"), and its central object — *the angle of the crack read off the sphere and used as the next turn* — is a data-dependent rotation, which is structurally unlike both FNV-1a (multiply) and SipHash/xxHash (whose rotations are **fixed compile-time constants**). Seed 1 is kept as a hard constraint on the implementation (one sphere, no lanes) rather than as the mechanism; Seed 3 is kept as the output rule.

Per step 4: the mechanism is *not* invented. A data-dependent rotation as the sole source of diffusion is **RC5/RC6** (Rivest) — a validated, published, multiplication-free primitive whose whole design rationale is that the rotation amount is read from the data. I let the crack angle land there rather than on something new. The finalizer's `s ^= rotl(s,a) ^ rotl(s,b)` is likewise not invented: it is the **SHA-2 Σ/σ** shape (an *odd* number of XORed rotations, which is why it is not trivially singular — `I + R^a` always kills the all-ones vector, `I + R^a + R^b` does not). The per-tumble `add / shift-xor / rotate` is the **Threefish/BLAKE ARX** shape. The one place where the native departs from RC5 is honest and stated: RC5 keeps its data-dependent rotation *bijective* by taking the amount from the other half of a two-word state. The native has only one sphere and forbids sliver updates, so `s ↦ rotl(s, s&63)` is **lossy** (~0.6 bits per tumble). The native says so himself — *"the old face I let shrink and go", "the eggshells the trail sheds"*. 64 fresh bits are re-injected per mark, so the loss does not compound; it raises the collision floor from ~2⁻⁶⁴ to roughly ~2⁻⁶² and does not affect avalanche at all. That is the price of the sole sphere, and I am paying it knowingly, not hiding it.

## ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."** Replaced by: mixing a *face* requires reading an angle. Zero multiplies in the kernel. Also broken: **"more mixing rounds always means better mixing"** (2 tumbles per 8 bytes in the O(len) path, all the heavy mixing in the O(1) final crack), and **"each byte must be mixed before the next is read"** (a mark is a whole 64-bit face, 8 bytes pressed at once, because the sphere's face is 64 bits wide).

Explicitly *not* broken, by the native's own order: the sole sphere. There is no SIMD striping and no thread parallelism in this kernel. Two spheres would be two memories, which Seed 1 forbids, and at every plausible benchmark size (≤ 64 KB ≈ 20 µs of work) an OpenMP fork would cost more than the whole hash. Vectorization hints only: `restrict`, unaligned `memcpy` word loads, 4-mark unroll so loads run ahead of the serial chain.

**Regime recognition, encoded in the metaphor itself** (step 5 — the known-way section describes both a "whole buffer, start to end" regime and, implicitly, a short/tail regime):
- *A pile too short to cover the sphere's face never reaches the trail* → `len < 8`: gather the marks into one partial face, one press, two tumbles, straight to the final stalk. No loop, no unroll, no overlapping read. This is the shortest path that still delivers full avalanche.
- *A pile long enough to coil the trail* → `len ≥ 32`: walk four faces per coil (unrolled).
- *8 ≤ len < 32*: plain one-face-at-a-time walk.
- *The eggshells the trail sheds* → `len & 7`: the ragged end is picked up by re-reading the last whole face (overlapping the already-consumed bytes; always in bounds when `len ≥ 8`, and the sphere is order-dependent so the overlap costs nothing). Length is pressed into the sphere at the high mouth, so a short pile and a zero-padded long one are never confused.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ------------------------------------------------------------------ *
 *  THE COILED MASON TRAIL                                            *
 *                                                                    *
 *  One sphere is the sole carried memory.  Each mark is pressed into  *
 *  its face, then it tumbles a fixed count of turns - no more, no     *
 *  fewer.  At each stalk it sobs once and cracks a hairline into      *
 *  itself, and the ANGLE OF THAT CRACK, read straight off the sphere, *
 *  is the weight it carries into the next turn.  The whole sphere     *
 *  reshapes, never a sliver.  At the end the sphere itself is thrown  *
 *  away: only the last, smallest crack is handed over.               *
 *                                                                    *
 *  No multiplication anywhere.  The mixing agent is a rotation amount *
 *  taken from the data (RC5/RC6 data-dependent rotation); the final   *
 *  crack uses the SHA-2 Sigma shape (odd number of xored rotations).  *
 * ------------------------------------------------------------------ */

#define MOUTH    0x14650FB0739D0383ULL  /* the sphere's face at the high mouth */
#define STALK_A  0x9E3779B97F4A7C15ULL
#define STALK_B  0xC2B2AE3D27D4EB4FULL
#define STALK_C  0x165667B19E3779F9ULL
#define STALK_D  0x27D4EB2F165667C5ULL
#define STALK_E  0xD6E8FEB86659FD93ULL

/* one turn of the sphere; portable idiom, compiles to a single rolq */
static inline uint64_t turn(uint64_t x, unsigned a) {
    a &= 63u;
    return (x << a) | (x >> ((64u - a) & 63u));
}

/* ONE TUMBLE.  Strike a stalk: the sphere sobs once (the stalk's weight
   is added, carries run), a hairline cracks into it (the high face
   bleeds down into the low), and the angle of the crack - the sphere's
   own low bits - is the turn it takes next.  Whole sphere, never a
   sliver. */
static inline uint64_t tumble(uint64_t s, uint64_t stalk, unsigned bleed) {
    s += stalk;                    /* the sob                          */
    s ^= s >> bleed;               /* the hairline                     */
    return turn(s, (unsigned)s);   /* the crack angle, carried forward */
}

/* press one mark into the face, then the fixed count of tumbles: two */
#define PRESS_AND_TUMBLE(w) do {        \
        s ^= (w);                       \
        s = tumble(s, STALK_A, 29);     \
        s = tumble(s, STALK_B, 32);     \
    } while (0)

/* THE FINAL STALK.  The sphere is not kept; only the last, smallest
   hairline it leaves here.  This runs once, so it may be struck hard -
   every earlier crack, all the stone dust and all the footprints are
   swept into the jungle's plumbing and forgotten on purpose. */
static inline uint64_t final_crack(uint64_t s) {
    s ^= s >> 32;
    s += STALK_C; s ^= turn(s, 31) ^ turn(s, 53);
    s = turn(s, (unsigned)s);
    s ^= s >> 29;
    s += STALK_D; s ^= turn(s, 17) ^ turn(s, 41);
    s = turn(s, (unsigned)s);
    s ^= s >> 32;
    s += STALK_E; s ^= turn(s, 13) ^ turn(s, 37);
    s ^= s >> 31;
    return s;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;

    /* the sphere, set down at the trail's high mouth, already bearing
       the height of the pile */
    uint64_t s = MOUTH ^ (uint64_t)len;

    /* REGIME 1: a pile too short to cover the sphere's face never
       reaches the trail.  One press, the fixed two tumbles, final stalk.
       Shortest possible path - no loop, no unroll, no overlap read. */
    if (len < 8) {
        uint64_t w = 0;
        for (size_t i = 0; i < len; i++)
            w |= (uint64_t)p[i] << (i * 8);
        PRESS_AND_TUMBLE(w);
        return final_crack(s);
    }

    /* REGIME 2: the pile covers the face.  Walk the trail in order,
       one face per press, four faces per coil when the pile is long
       enough to be worth coiling.  The loads run ahead; the sphere's
       chain is what it is - one sphere, by the native's own order. */
    const size_t nfull = len >> 3;
    size_t i = 0;
    uint64_t w0, w1, w2, w3;

    for (; i + 4 <= nfull; i += 4) {
        memcpy(&w0, p + 8 * i,      8);
        memcpy(&w1, p + 8 * i + 8,  8);
        memcpy(&w2, p + 8 * i + 16, 8);
        memcpy(&w3, p + 8 * i + 24, 8);
        PRESS_AND_TUMBLE(w0);
        PRESS_AND_TUMBLE(w1);
        PRESS_AND_TUMBLE(w2);
        PRESS_AND_TUMBLE(w3);
    }
    for (; i < nfull; i++) {
        memcpy(&w0, p + 8 * i, 8);
        PRESS_AND_TUMBLE(w0);
    }

    /* the eggshells the trail sheds: the ragged end, picked up by
       re-reading the last whole face (in bounds, overlaps spent marks -
       the sphere is order-dependent and does not care) */
    if (len & 7u) {
        memcpy(&w0, p + len - 8, 8);
        PRESS_AND_TUMBLE(w0);
    }

    return final_crack(s);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 3.0**

Stated before any measurement, from the dependency-chain accounting:

- FNV-1a baseline chain per byte: `xor` (1) → `imul` (3) = **4 cycles/byte**.
- This kernel per 8-byte mark: `xor` (1) + 2 × tumble, where tumble = `add` (1) → `shr` (1) → `xor` (1) → `rol %cl` (1 on Zen3/Ice Lake, 2 on Skylake) ≈ 4–5 → **9–11 cycles per 8 bytes = 1.13–1.38 cycles/byte**.
- Ratio 4 / 1.25 ≈ 3.2, less the one-shot `final_crack` (~24 cycles) and loop overhead → **≈3.0** at ≥1 KB; ≈2.5 at 64 B; ≈1.4 at 16 B; ≈parity at 8 B (37 vs 32 cycles), which is why the short regime takes the minimum path and nothing more.
- Avalanche prediction: **≈0.500 flip fraction, score ≥0.99.** Rationale: a single-bit difference expands through 3 Σ-rounds (×3 positions each), 4 xorshifts (×2 each) and 4 carry-propagating adds in the final crack alone — 3³·2³ ≈ 200 position-instances over 64 slots, saturating — and long before that, the moment a difference reaches the low 6 bits of any tumble the *rotation amount* diverges and the whole word re-registers ("one wrong mark and every stalk downstream sobs differently").
- Weakest quality point, predicted and not hidden: collisions, not avalanche. The lossy single-word data-dependent rotation puts the collision floor at ~2⁻⁶² rather than 2⁻⁶⁴.

## MEASUREMENT

**No measurement was obtained.** This session was invoked with no tools available — `hash_bench` and `hash_contract` could not be called, so I have no throughput number and no avalanche score to report, only the prediction above and the cycle accounting behind it. I am not going to dress the prediction up as a result. The pipeline will compile and measure the artifact; the numbers to check against are: throughput ratio ≈3.0 at ≥1 KB, avalanche ≈0.500.

The zero-improvement-iterations budget was spent on paper instead: the design is already the second pass, and the changes relative to the previous attempt are (i) the mixing agent is now a *data-dependent* rotation amount read off the sphere, not a table of fixed rotation constants, (ii) all SIMD/lane striping is removed because it contradicted the sole-sphere seed, (iii) the heavy mixing was moved out of the O(len) path into the O(1) final crack.

## VERDICT

The core of this kernel **is** the native's mechanism, translated literally and without substitution: one sphere as the sole memory (`uint64_t s`, no lanes, no tree, no side state); each mark pressed into its 64-bit face; a fixed count of two tumbles per mark, no more and no fewer; at each stalk a sob (`add`), a hairline (`shift-xor`), and **the crack angle read straight off the sphere and used as the next turn** (`rotl(s, s & 63)`); whole-word reshaping with no sliver update; and only the last hairline handed over, the sphere itself thrown away. That mechanism has a name in the validated literature — RC5/RC6 data-dependent rotation — and the finalizer's shape has one too, SHA-2's Σ. I let the metaphor arrive at those rather than invent a substitute, and there is not a single multiplication in the file.

Honest accounting of where this can lose:

1. **8-byte inputs, ~parity with FNV-1a** (predicted 37 vs 32 cycles), because the fixed ~24-cycle final crack cannot be amortized. *Addressed structurally, not by hand-waving:* the short-pile regime is detected explicitly (`len < 8`) and takes the shortest possible path — no loop, no unroll, no overlap read, one press. There is no avoidable work left to cut. Cutting the final crack instead would trade the task's primary stated goal (avalanche) for ~10 cycles; FNV-1a at 4 bytes has avalanche ≈0.25, so the comparison at that size is not like-for-like in FNV's favour.
2. **Collision floor ~2⁻⁶² instead of 2⁻⁶⁴**, from the lossy single-word data-dependent rotation. This is inherent to the sole sphere plus the "whole sphere, never a sliver" rule — RC5 avoids it only by splitting the state, which the native forbids. Not fixable without abandoning Seed 1, so I keep it and state it rather than quietly adding a second word.
3. **No thread parallelism, deliberately, with the reason stated** rather than added and guarded: the units of work here are single 8-byte presses on one shared sphere. There is nothing to parallelize without a second sphere, and at ≤64 KB an OpenMP fork costs more than the entire hash. Nothing risky is shipped unguarded because nothing risky is shipped.
4. **The one interpretive liberty I took**, flagged plainly: a "mark" is an 8-byte word, not a byte. The justification is that the sphere's *face* is 64 bits wide and the native presses the mark "into the sphere's face". This is the whole source of the predicted 3× — a byte-granular reading would give ~4.5 cycles/byte and lose to FNV-1a. If a reviewer rejects that reading, the honest consequence is that this seed, taken strictly byte-at-a-time, is *slower* than the known way and wins only on avalanche and on having no multiplier.