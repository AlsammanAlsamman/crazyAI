## MAPPING

I read all three seeds as one desk, then check each against the five silent assumptions.

**SEED 1 — "the die-stone never resets between marks; every fold carries the callus of all folds before it."**

| World object | Problem object |
|---|---|
| pile of marks, given order | `data[0..len-1]`, byte order fixed |
| die-stone | the accumulator |
| "never resets between marks" | loop-carried dependency: `h` at step *i* feeds step *i+1* |
| callus of the previous press | state is read-modify-write, never re-initialized |

Assumption broken: **none.** This seed *is* FNV-1a. It asserts the serial chain, i.e. it entrenches "each byte must be mixed before the next is read" and "single accumulator updated in place."

**SEED 2 — "each mark's weight is pressed into the already-turned position rather than onto a clean face, so early marks bend how later marks land."**

| World object | Problem object |
|---|---|
| mark's *weight*, not its word | byte as an integer, not a character |
| "already-turned position" | `h` is the multiplicand/addend, not a fresh register |
| "early marks bend how later ones land" | order-sensitivity; non-commutative mixing |
| chalk-dust hair's width multiplied twice over | avalanche by repeated multiply |

Assumption broken: **none of the five.** It's the justification for the multiply, i.e. it *defends* "mixing one byte requires a multiplication."

**SEED 3 — "only the stone's final seated number against the wooden keeps ever leaves the desk; every intermediate turn and residue is swept away and discarded."**

| World object | Problem object |
|---|---|
| the die-**stone** (a die has faces, not one face) | the internal state is a *multi-lane* object |
| "turn it a quarter through the chalk-bank groove" | rotate which lane is live: quarter turn → **4 faces → 4 lanes** |
| "drop the next mark's weight into the same seated place" | mark *i* lands on lane `i mod 4`; the seat is fixed, the stone moved |
| "never lifting it fully off the desk" | lanes live in registers across the whole loop, no spill |
| intermediate turns, groove-dust, chalk residue → **swept off and thrown away** | the 4-lane intermediate state is **unobservable**; nothing outside the loop may depend on its shape |
| wooden numbered **keeps** (plural) at the desk's edge | the finalizer: merge the 4 faces, then avalanche to one 64-bit token |
| "the final seated number, and only that, is the token" | the ABI is `uint64_t`; width of the *state* is unconstrained |
| "I keep folding until the stone has turned once for every mark, however long the pile runs — centuries fold down just like a handful" | one fold per 8-byte word, O(*len*), no per-size special-casing of the *fold* itself |
| a pile too short to complete one revolution of the stone | `len < 32`: the 4-face reading is meaningless → single-face fallback |
| "two different piles essentially never seat the same way" | collision resistance / avalanche ≈ 0.5 |

Assumption broken: **"the state is a single accumulator updated in place, one value."**

## CHOSEN SEED

**Seed 3.** Stated plainly: seeds 1 and 2 break *none* of the five assumptions — they are faithful descriptions of FNV-1a and give me nothing. Seed 3 is the only one that breaks anything, and it happens to break exactly the preferred one.

The lever is the sweeping-away. Because *only* the last seated number leaves the desk, no one can ever observe how wide the stone was or how many faces it turned through. The contract constrains the **output** to one 64-bit value; the native's discipline reveals that this says nothing about the **state**. A die-stone turned a quarter at a time has four faces; four faces receiving marks round-robin are four *independent* folding chains, and the "wooden keeps" are the stage that collapses them.

## ASSUMPTION BROKEN

> the state is a single accumulator updated in place, one value

Four lanes, one seat, quarter-turn rotation. FNV's serial multiply chain is ~4–5 cycles of pure latency per **byte**; the multiplier unit sits idle almost the whole time. Four independent chains fill those idle slots with no extra work — the throughput comes from the latency that was already being wasted, not from doing less mixing.

**Arriving at a validated technique rather than inventing one (step 4).** A quarter-turn stone with 4 faces, round-robin 8-byte words, `rotl`-and-sum merge, then a 3-stage shift-multiply avalanche *is* **xxHash64's stripe design** (and the ancestor of XXH3). I am not shipping a novel mixer; the metaphor lands on the known-good one, which passes SMHasher's avalanche and collision suites. That is the right outcome — I keep the validated constants and structure and claim only the derivation.

**SIMD, honestly evaluated and rejected.** The obvious next step is to put the four faces in one `__m256i`. AVX2 has no 64×64→64 multiply (`vpmullq` is AVX-512DQ only); emulating it costs 3 `vpmuludq` + shifts/adds, which is *slower* than four independent scalar `imul`s that already saturate the multiplier port. So the vectorization here is instruction-level (4 independent chains + 64-byte unroll + `memcpy` loads that compile to single `mov`s), not `immintrin.h`. **No threads**: one fold is ~8 bytes of work; at benchmark sizes the OpenMP fork cost exceeds the entire hash. Declining rather than adding a guard I'd have to hide behind.

**Two regimes, recognized in-world (step 5).** The known-way section carries a small-vs-large regime ("overhead if small"). The metaphor detects it without my help: *does the pile fill at least one full revolution of the stone?* Under 32 bytes the stone never completes a turn, so the four-face reading is nonsense — take the single-face short path (`KP5` basis, 8/4/1-byte folds, same finalizer). This is the guard for the one condition where my mechanism could lose.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the chalk-bank constants: the cycle-light is the same starlight color
   no matter the hour -- fixed, input-independent. */
#define KP1 11400714785074694791ULL
#define KP2 14029467366897019727ULL
#define KP3  1609587929392839161ULL
#define KP4  9650029242287828579ULL
#define KP5  2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
/* read a mark's weight off the desk's numbered groove (order as given) */
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* ONE FOLD: drop the mark's weight into the stone's already-turned seat,
   quarter-turn it, and let the callus of the last press bend this one.
   Nothing washes clean: acc enters and leaves. */
static inline uint64_t fold(uint64_t acc, uint64_t w) {
    acc += w * KP2;
    acc  = rotl64(acc, 31);
    acc *= KP1;
    return acc;
}
/* read one face against the wooden numbered keeps at the desk's edge */
static inline uint64_t keep(uint64_t h, uint64_t face) {
    h ^= fold(0, face);
    h  = h * KP1 + KP4;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *const end = data + len;
    uint64_t h;

    /* REGIME TEST: does the pile fill at least one full revolution of the
       stone?  Four faces, eight bytes a face = 32 marks. */
    if (len >= 32) {
        /* the four faces of the die-stone, seated differently to begin with
           so no two folding-paths start alike */
        uint64_t v1 = KP1 + KP2;
        uint64_t v2 = KP2;
        uint64_t v3 = 0ULL;
        uint64_t v4 = 0ULL - KP1;

        /* two revolutions per pass: same four faces, less groove overhead.
           The stone is never lifted fully off the desk -- v1..v4 stay in
           registers for the whole pile. */
        if (len >= 64) {
            const unsigned char *const lim64 = end - 64;
            while (p <= lim64) {
                v1 = fold(v1, ld64(p     ));  v2 = fold(v2, ld64(p +  8));
                v3 = fold(v3, ld64(p + 16));  v4 = fold(v4, ld64(p + 24));
                v1 = fold(v1, ld64(p + 32));  v2 = fold(v2, ld64(p + 40));
                v3 = fold(v3, ld64(p + 48));  v4 = fold(v4, ld64(p + 56));
                p += 64;
            }
        }
        {   /* len >= 32, so end - 32 is in-bounds */
            const unsigned char *const lim32 = end - 32;
            while (p <= lim32) {
                v1 = fold(v1, ld64(p     ));  v2 = fold(v2, ld64(p +  8));
                v3 = fold(v3, ld64(p + 16));  v4 = fold(v4, ld64(p + 24));
                p += 32;
            }
        }

        /* lift the stone: collapse the four faces against the keeps.
           Everything about how wide the stone was is swept away here. */
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = keep(h, v1);
        h = keep(h, v2);
        h = keep(h, v3);
        h = keep(h, v4);
    } else {
        /* FALLBACK: too short to turn the stone once -- press onto one face */
        h = KP5;
    }

    h += (uint64_t)len;

    /* the last few marks, pressed one face at a time */
    while (p + 8 <= end) {
        h ^= fold(0, ld64(p));
        h  = rotl64(h, 27) * KP1 + KP4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)ld32(p) * KP1;
        h  = rotl64(h, 23) * KP2 + KP3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * KP5;
        h  = rotl64(h, 11) * KP1;
        p++;
    }

    /* THE FINAL SEATED NUMBER -- the only thing that leaves the desk.
       A hair's width of chalk-dust is multiplied, and multiplied again,
       until no two folding-paths walk the same last step. */
    h ^= h >> 33;  h *= KP2;
    h ^= h >> 29;  h *= KP3;
    h ^= h >> 32;
    return h;
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 8.0

Stated before any measurement, and decomposed so it can be scored honestly:

- **Throughput, large buffers (≥ 1 KB).** FNV-1a is one 8-byte-free `xor` + one `imul` per **byte** on a serial chain: ≈ 4–5 cycles/byte, ≈ 0.2 B/cycle. This kernel does ~2 `imul` + `rotl` + `add` per 8 bytes on **4 independent** chains: critical path ~10 cycles per lane, four lanes in flight → ≈ 2.5 cycles per 8 bytes ≈ **3.2 B/cycle**. That is a **15–18×** ratio, capped by L1/L2 bandwidth (~10–14 GB/s measured territory) for multi-MB buffers.
- **Throughput, < 32 bytes.** Falls into the single-face path, which is the *same* shape as the baseline plus a fixed ~12-cycle finalizer: expect **0.8–1.3×**. This is the regime where I can lose, and losing by ~20% on a 4-byte key is the cost of the guard existing at all.
- **Blended.** If the harness sweeps sizes with small keys weighted equally, the harmonic reality lands near **6–9×**. Hence 8.0.
- **Avalanche.** Expect **0.49–0.51** (score near the ideal 0.5). Every input word reaches only one lane inside the loop, but the `rotl`-sum + four `keep` rounds + 3-stage shift-multiply finalizer is the xxHash64 finalizer, which passes SMHasher's strict-avalanche criterion. I expect *no* avalanche loss versus FNV-1a — and possibly a gain, since FNV-1a's low bits mix poorly (its last byte only ever touches bits via one multiply).

Falsifiable failure modes I am committing to in advance: (a) if avalanche comes back below 0.45, the 4-lane split leaked and the merge is too weak — that would be a real defeat for the mapping, not a tuning issue; (b) if large-buffer speedup is under 4×, the loads are the bottleneck and the whole latency argument was beside the point.

## MEASUREMENT

**Not measured.** No tools were available in this session — `hash_bench` and `hash_contract` could not be invoked, so I ran zero of the four permitted improvement rounds. Reporting this plainly rather than dressing up the prediction as a result: the numbers in PREDICTION are analytic (instruction latency × port throughput), not observed. The pipeline will compile and measure; the prediction above stands as filed and unrevised, with the two falsification conditions named.

The one thing I did verify by hand, since it needs no tool: **no undefined behavior in the size guards.** `end - 64` is computed only inside `if (len >= 64)`, `end - 32` only inside `if (len >= 32)`; the tail loops use `p + n <= end` rather than pointer subtraction. Loads go through `memcpy`, so unaligned and strict-aliasing-safe, and compile to single `mov`s at `-O3`. `len == 0` returns the finalized `KP5 + 0` without touching `data`.

## VERDICT

The native's desk does contain the multi-lane insight, but it is hidden in the *third* seed, not the first two. Seeds 1 and 2 are FNV-1a restated; anyone who stopped there would have re-derived the baseline and called it a translation. The load-bearing sentence is the sweeping-away: *only the final seated number leaves the desk.* That is a statement about the **observable**, and the standard solution silently over-reads it as a statement about the **state**. Once you notice that a die-**stone** turned a **quarter** at a time has four faces, and that nothing downstream can ever tell how wide it was, the single accumulator dissolves and FNV's wasted multiplier latency becomes free throughput.

What I claim: the derivation. What I do not claim: novelty. The mechanism lands on xxHash64's exact stripe structure and constants, and I kept them rather than inventing a mixer I'd have no way to validate — a validated known technique beats a novel untested one, and this one has been through SMHasher.

Where this can be worse than the known way, addressed rather than hand-waved: **buffers under 32 bytes**, where four lanes cannot be filled and the merge is pure overhead. Guarded — `len >= 32` selects the stone path, everything shorter takes the single-face path with a `KP5` basis. I did not ship the risk. Two further risks I dropped rather than guarded: **SIMD** (AVX2 has no `vpmullq`; emulation would be slower than the scalar version, so the honest move was to not pretend) and **OpenMP** (one fold is 8 bytes; thread setup dwarfs the whole hash at these sizes).

The residual weakness is real and worth naming: for keys in the 4–31 byte range — very common in hash tables — this is at best a wash and plausibly 10–20% slower than plain FNV-1a, because the strong finalizer is a fixed cost the baseline doesn't pay. The metaphor told me a handful of marks folds just like centuries, and on this one point the desk is wrong: a handful does *not* fold the same, and the guard exists precisely because I don't believe that line.