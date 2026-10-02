## MAPPING

### SEED 1 — "the stone never resets between marks, every fold carries the callus of all folds before"

| world object | problem object |
|---|---|
| the pile of marks, in given order | the byte buffer `data[0..len-1]`, read in order |
| one fold per mark | exactly one round per input chunk — never a second cleanup pass over the buffer |
| "never lifting it fully off the desk" | the state is never re-initialized mid-stream; no per-chunk fresh accumulator |
| "centuries of marks fold down just the same as a handful" | O(1) state, O(n) time, streaming — state size independent of `len` |

**Breaks:** *"more mixing rounds always means better mixing."* The native does **one** fold per mark and claims that is enough, because quality comes from non-resetting composition, not from round count. (It *affirms* the in-order/sequential assumption rather than breaking it.)

### SEED 2 — "each weight is pressed into the already-turned position, early marks bend how later marks land"

| world object | problem object |
|---|---|
| "turn it a quarter through the groove" | `ROTL64(x, 16)` — a quarter of a 64-bit face |
| "drop the weight into the same seated place" | `state += word` — addition, not substitution |
| "the stone's memory of the last press bends how deep this one goes" | the carry/rotation depth that a word achieves depends on the state it lands on — a state-dependent, *multiplication-free* mix |
| "a hair of chalk-dust difference ... the next fold multiplies it, and the one after multiplies it again" | per-round differential expansion through carries + rotation |

**Breaks:** *"mixing one byte requires a multiplication."* The bend is rotation-plus-carry, state-dependent; no multiply in the per-byte path.

### SEED 3 — "only the final seated number leaves the desk; every intermediate turn and all residue is swept away" ← chosen

| world object | problem object |
|---|---|
| the bone-colored **die**-stone (a many-*faced* body) | a **wide multi-lane state**: `v0,v1,v2,v3` — 256 bits, four faces |
| "the one that came inside the child" (a fixed, given stone) | fixed initial basis constants, identical every call |
| the desk's **numbered groove** | the index line; its length vs. the pile = the runtime regime check (`len >= 32`?) |
| cycle-light "the same starlight color no matter the hour" | position-independent round: same constants at every index, no counter/table |
| reading marks "as weights, as the physicist reads a flat thing about to be invested into three" | 8 flat bytes reinterpreted (little-endian `memcpy` load) as **one wide 64-bit weight** on one face |
| pressing each mark against the stone, nothing washed clean | absorb: `v_i += w_i` into the live lanes |
| one fold = quarter turn + seat + the previous press bending the depth | `v_i = ROTL64(v_i + w_i, 16)` then pairwise/crossed coupling of the faces |
| "no two folding-paths that started differently ever walk the same last step" | the block transition must be a **bijection** of the 256-bit state — differences can never collapse |
| the **wooden numbered keeps** at the desk's edge | the finalizer: a fixed external ladder the wide state is read off against |
| "that reading, and only that reading, is the token" | 256-bit state → **64-bit** output: a narrow squeeze |
| groove-dust, intermediate turns, chalk residue, swept away | 192 bits of state are *discarded*, never emitted — the hidden capacity |
| "two different piles essentially never seat the same way twice" | collision resistance claim |

**Breaks:** *"the state is a single accumulator updated in place, one value."* The desk holds strictly **more** than the token: a four-faced stone plus residue that is thrown away. State and output are different objects, of different widths.

## CHOSEN SEED

**SEED 3.** It is the one that breaks the preferred assumption, and it is also the most distant from FNV-1a/xxHash-as-stated, where the accumulator *is* the hash — one 64-bit value that is both the working state and the result. Seed 3 says the working body is many-faced and wide, and the 64-bit answer is only what you can *read off the edge of the desk* after folding stops. That is, literally, the **sponge discipline**: wide permuted state, absorb into part of it, squeeze a narrow output, discard the capacity. Per step 4 I let the mechanism land on validated technique rather than invent: the squeeze (`keep` merge + Murmur3/xxHash64-grade avalanche) and the short-input path are **xxHash64's own validated finalization and tail**, unchanged; the novelty is confined to the absorb round, where the native's "quarter turn, seat, let the last press bend the depth" gives a multiplication-free, provably bijective four-face fold.

Seeds 1 and 2 are folded in where they belong (one round per chunk; no multiply in the hot loop) — they do not break the single-accumulator assumption, so they are not the pick.

**Regime recognition (step 5).** The known_way section describes two regimes — "centuries of marks" vs. "a handful", with setup overhead on short piles. The native's own instrument encodes the test: he lays the pile along *the desk's numbered groove* and sees whether it fills it. In code: `len >= 32` (the groove = one full turn of four faces) takes the wide four-face path; a shorter pile never wakes the extra faces and is folded on a single face by the plain validated short path. Both paths share one tail and one set of keeps.

**Threads.** Declined, deliberately — and the metaphor is what declines them: there is exactly *one* stone, and the claim "two different piles pressed through **this same stone**" is void the moment you hand out a second stone per thread. Vectorization hints only (`restrict`, lane-symmetric body, single-instruction `rol`, 8-byte `memcpy` loads). At benchmark sizes a 64-bit-per-cycle-class hash is already near memory bandwidth; threads would buy nothing and would make the output depend on thread count.

## ASSUMPTION BROKEN

> **"the state is a single accumulator updated in place, one value."**

Replaced by: a 256-bit four-face state, absorbed into by a bijective multiplication-free round, squeezed once at the end to 64 bits with 192 bits swept off the desk.

Secondary breakage (SEED 2, free with the chosen round): **no multiplication in the per-byte path** — all multiplies are O(1), in the keeps.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the die-stone ------------------------------------------------------
   Four faces (256-bit state), never reset, folded once per 32-byte block.
   One fold = quarter turn of each face (ROTL 16), seat the next weights into
   the already-turned positions (+=), then let each face's callus bend its
   neighbour (pairwise, then crossed).  The whole block map is a BIJECTION of
   the 256-bit state, so differing fold-paths can never re-converge:
     given (v0',v1'): v1 = ROTR(v1' ^ v0', 23), v0 = v0' - v1   (and likewise
     for the (v2,v3) pair); the crossing and the turns invert trivially.
   No multiplication touches a data byte.  At the end the stone is read off
   against the wooden numbered keeps -- xxHash64's validated merge, tail and
   avalanche, unchanged -- and the other 192 bits are swept away.
   Regime check: a pile that does not fill the groove (len < 32) never wakes
   the extra faces and falls back to the plain single-face path.          */

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

#define HP1 11400714785074694791ULL
#define HP2 14029467366897019727ULL
#define HP3  1609587929392839161ULL
#define HP4  9650029242287828579ULL
#define HP5  2870177450012600261ULL

static inline uint64_t hld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t hld32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
/* one wooden keep */
static inline uint64_t hkeep(uint64_t v) {
    v *= HP2; v = ROTL64(v, 31); v *= HP1; return v;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    const unsigned char *const end  = p + len;
    uint64_t h;

    if (len >= 32) {                      /* the pile fills the groove */
        uint64_t v0 = HP1 + HP2;
        uint64_t v1 = HP2;
        uint64_t v2 = 0;
        uint64_t v3 = (uint64_t)0 - HP1;
        const unsigned char *const limit = end - 32;
        do {
            uint64_t w0 = hld64(p);
            uint64_t w1 = hld64(p +  8);
            uint64_t w2 = hld64(p + 16);
            uint64_t w3 = hld64(p + 24);

            /* quarter turn, weight seated into the turned position */
            v0 = ROTL64(v0 + w0, 16);
            v1 = ROTL64(v1 + w1, 16);
            v2 = ROTL64(v2 + w2, 16);
            v3 = ROTL64(v3 + w3, 16);

            /* the callus: each face bends its neighbour, then the pairs cross */
            v0 += v1;  v1 = ROTL64(v1, 23) ^ v0;
            v2 += v3;  v3 = ROTL64(v3, 23) ^ v2;
            v0 += v3;  v2 += v1;

            p += 32;
        } while (p <= limit);

        /* read the stone off the numbered keeps; sweep the rest away */
        h  = ROTL64(v0, 1) + ROTL64(v1, 7) + ROTL64(v2, 12) + ROTL64(v3, 18);
        h ^= hkeep(v0); h = h * HP1 + HP4;
        h ^= hkeep(v1); h = h * HP1 + HP4;
        h ^= hkeep(v2); h = h * HP1 + HP4;
        h ^= hkeep(v3); h = h * HP1 + HP4;
    } else {                              /* a handful: one face is enough */
        h = HP5;
    }

    h += (uint64_t)len;                   /* the count of turns is part of the reading */

    /* trailing marks, single face */
    while ((size_t)(end - p) >= 8) {
        h ^= hkeep(hld64(p));
        h  = ROTL64(h, 27) * HP1 + HP4;
        p += 8;
    }
    if ((size_t)(end - p) >= 4) {
        h ^= (uint64_t)hld32(p) * HP1;
        h  = ROTL64(h, 23) * HP2 + HP3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * HP5;
        h  = ROTL64(h, 11) * HP1;
        p += 1;
    }

    /* final avalanche */
    h ^= h >> 33; h *= HP2;
    h ^= h >> 29; h *= HP3;
    h ^= h >> 32;
    return h;
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 12**

Stated before any measurement, and no measurement has been taken. Reasoning, so the number is falsifiable rather than decorative:

- Baseline FNV-1a is a *serial* 64-bit multiply chain, one byte per link: ~4–5 cycles/byte ⇒ ≈ 0.7 GB/s at 3 GHz.
- The fold consumes 32 bytes per iteration in ~24 ALU uops with a 4-step critical path, so it is throughput-bound, not latency-bound: ~6 cycles/32 B ⇒ ~5 B/cycle ⇒ ≈ 15 GB/s. That alone is ~20× on buffers ≥ 1 KB, and slightly *better* than xxHash64 because the eight 64-bit multiplies per 32 bytes are gone from the hot loop.
- I predict 12 rather than 20 because any benchmark mix that includes short buffers pays the O(1) keeps-and-avalanche ladder (~6 multiplies, ~25 cycles) which FNV-1a does not pay; at len ≤ 16 the two are roughly level and the geometric mean gets dragged down.
- Avalanche: I predict ~0.50 flip fraction, bias well under 1% (score ≈ 0.95–1.0). The absorb is a proven bijection so no input difference can be annihilated, and the squeeze is Murmur3/xxHash64-grade.

Risk I will name and have already addressed, rather than ship: *the wide path could lose to the simpler path on short inputs.* It is guarded — `len >= 32` is a hard runtime regime check, and below it the kernel degenerates exactly to xxHash64's validated short-input path with no lane setup at all. The only residual cost at tiny `len` is the final avalanche, which cannot be dropped without destroying the metric the task asks for.

## MEASUREMENT

Not performed. `hash_bench` and `hash_contract` were unavailable in this session (no tools were exposed), so I have no throughput or avalanche number and I will not manufacture one. What I *could* verify without tools, I did, by hand:

- **Bijectivity of the block transition** (the native's "no two folding-paths ever walk the same last step"): inverted explicitly. From the post-block state, `v1 = ROTR(v1' ^ v0_mid, 23)`, `v0_mid = v0' − v3'`, `v0_pre = v0_mid − v1`, and symmetrically for `(v2,v3)`; each quarter turn and each `+= w_i` inverts trivially. So the map is a permutation of the 256-bit state for every fixed 32-byte block, and differential collapse is impossible — not merely unlikely.
- **Boundary arithmetic**: `len = 0` (no deref, returns `avalanche(HP5)`), `len = 31` (short path only), `len = 32` (exactly one block, empty tail), `len = 33`, `len = 64` (two blocks, empty tail) all traced by hand against the `do/while (p <= limit)` form.
- The unmodified reuse of xxHash64's merge/tail/avalanche means the finalization half of the construction is already SMHasher-validated upstream; only the absorb is new.

The prediction above is therefore the honest open claim, not a report.

## VERDICT

The native was describing a **sponge**, and he was describing it exactly. "Only the last seated number ever leaves the desk; the groove-dust and intermediate turns I sweep off and throw away" is capacity-versus-rate stated in furniture. Taken literally it forces three things the textbook one-accumulator hash does not have: a state wider than the output, a round that is a bijection so nothing can wash clean, and a separate narrow squeeze at the end. All three are real, validated design principles, and the mapping arrived at them without my importing them.

Two honest qualifications:

1. **The speed win is mostly not from the broken assumption.** It comes from reading 8 bytes per face instead of 1 byte per round — that is the "weights, not words" line in the seed, and the four faces then multiply it. The sponge discipline buys *quality* (provable non-collapse, a clean place to put the heavy mixing) rather than throughput. I would rather say that than claim the wide state is what made it fast.
2. **I did not take the metaphor's last step.** "The stone's own memory of the last press bends how deep this one goes" is most literally a *data-dependent rotation* (`ROTL64(v, v >> 58)`). I used a fixed quarter turn plus state-dependent carry propagation instead, for two reasons: a data-dependent rotate of the form `x ^ ROTL(x, k(x))` is not a bijection on 64 bits (`x` and `~x` both land on 0 under `x ^ ROTL(x,k)` for every `k`), which would violate the native's own strongest claim; and it would cost 3 AVX2 ops where `rol` costs 1. The fixed turn keeps the bijection the native insisted on. That is a real, named departure from the most literal reading, and if a measurement shows the avalanche score short of ~0.95 it is the first thing I would revisit — AVX2's `vpsllv/vpsrlv` do give per-lane variable rotates, so the literal version is implementable, just not provably non-collapsing.

Where it could still be worse than the known way: a benchmark consisting entirely of ≤ 8-byte keys, where FNV-1a's total absence of a finalizer wins on cost and the task's own avalanche requirement is simply unmet by the baseline. That trade I accept openly rather than guard away, because guarding it means returning a badly-avalanching hash. Threads were refused on the metaphor's own authority — one stone, one pile — and that happens to be the right engineering call too.