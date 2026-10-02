## MAPPING

### SEED 1 — "one crease per mark, angle set by the mark and the crease before it"

| World object | Problem object |
|---|---|
| mark, read in order, never skipping, never looking ahead | one input byte, streamed once, no random access, no lookahead |
| one fold per mark | one mix step per byte |
| crease left by the previous fold | the running state value |
| angle = f(mark, previous crease) | `h = mix(h, data[i])` — state feedback |
| the sheet carries forward everything it was told | state is a full accumulator, not a window |

**Breaks:** nothing. This seed *is* FNV‑1a. It actively re‑asserts "each byte must be mixed into the running state before the next byte is read" and "one accumulator updated in place." Only faint dissent: the *sheet* retains every crease, i.e. the state is a surface, not a scalar.

### SEED 2 — "test every crease against all four wire birds' beaks at once and refold until they agree"

| World object | Problem object |
|---|---|
| four wire birds, used **simultaneously** as gauges | four independent 64‑bit accumulator lanes `v1..v4`, four independent dependency chains live in the same cycle |
| pressing a fold's corner against a *different* beak | each lane absorbs a *different* word of the stream in the same iteration |
| "only when **all four** beaks agree is the crease set" | no hash value exists until all four lanes are merged; a 32‑byte stripe isn't complete until every lane has taken its word |
| refolding tighter until spiral/snail markings line up **wrong on purpose** | rotations + odd‑prime multiplies chosen to destroy positional/arithmetic alignment — anti‑linearity |
| "a clean line‑up would mean two piles look the same" | collision avoidance is the explicit design target of the scrambling |

**Breaks:** ✅ *"each byte must be mixed into the running state before the next byte is read"* — four marks are in four different beaks at once, and there is **no single running state** at any point during the buffer. Also breaks ✅ *"the state is a single accumulator, one value"* and partially *"read once, start to end, in order"* (per‑lane the order is strided, not sequential).

### SEED 3 — "thin the whole wad at the water's edge to one dense corner; throw away every scrap and misreading"

| World object | Problem object |
|---|---|
| carrying the wad to the water's edge *after the last mark* | finalization happens once, at the end, not per byte |
| "one small suitcase" carried under | the length `len` folded into the state (so `"ab"` ≠ `"a","b"` at the boundary) |
| thinning: small, small, small | 256 bits of lane state → 64 bits, by xor‑shift / multiply |
| one hard dense corner no bigger than a coin | the `uint64_t` return value |
| every sheet, scrap and failed beak‑reading thrown in the mud as the tide closes | no scratch memory, no persistent state; lane values die on return — one‑way |

**Breaks:** ✅ *"more mixing rounds always means better mixing"* — the avalanche is bought by **one** terminal collapse, not by adding per‑byte rounds; consequently the per‑mark work can be cheap. (Adjacent to breaking "mixing one byte requires a multiplication.")

---

## CHOSEN SEED

**SEED 2 — the four wire birds, all four at once.**

It is the only seed that breaks the preferred assumption ("each byte mixed into the running state before the next byte is read"), it is the furthest from the known way (which has exactly one accumulator), and its mapping is mechanical rather than interpretive: *four gauges used simultaneously* → *four accumulator lanes consuming four words per iteration*. SEED 3 is retained as the finalizer of the same artifact (the native describes one procedure, not three), and SEED 1 survives *inside each lane* (each lane's crease angle still depends on its own previous crease).

**Crucially (step 4):** four lanes × 8 bytes = a 32‑byte stripe, merged at the end with rotations and then collapsed by a terminal avalanche — this *is* **xxHash64**, a validated, SMHasher‑passing, widely deployed construction. I did not invent a fifth bird to be original; the metaphor's "four wire birds" lands exactly on xxHash64's four accumulators, so the mechanism arrives at the known good technique rather than a novel untested one. That is the intended outcome here, not a coincidence I'm papering over.

**Regime recognition, in‑world (step 5):** the native's very first act is "a sheet… *only that it's large enough*." That is a runtime size test, and it is the only one he performs. So the kernel decides at runtime:

* **bulk regime** (`len >= 32`, sheet large enough): four birds, striped main loop — throughput‑bound, per‑byte cost is what matters.
* **short regime** (`len < 32`, sheet not large enough): **one** bird only — skip lane init and the four merge rounds entirely, go straight to the per‑mark tail folds and the water's edge. This is the fallback that guards the only place the four‑bird mechanism could lose to the simple accumulator.

**No threads, stated plainly:** the native has one pair of hands, and his unit of work is one crease per mark — nanoseconds. Thread spawn/join is microseconds, i.e. longer than hashing an entire benchmark‑sized buffer, and the lane chains are 1‑cycle‑granularity dependencies. Parallelism here belongs at instruction level (four live chains + unaligned word loads), not thread level. So no OpenMP is shipped, and therefore no threading risk needs guarding.

---

## ASSUMPTION BROKEN

> **"each byte must be mixed into the running state before the next byte is read."**

Four bytes‑groups are in flight in four different beaks at the same time; there is no "the running state" at all until the last mark is folded and the beaks are made to agree. Secondary casualties: "the state is a single accumulator, one value" (it is 256 bits in four pieces), and "more mixing rounds always means better mixing" (mixing is *moved* to one terminal collapse, not added).

---

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The Four-Bird Fold.
 *   mark            -> one word of the stream (the glance a reader takes in at once);
 *                      the last <4 marks are folded one byte at a time, as the native does
 *   fold            -> acc = rotl(acc + mark*P2, 31) * P1   (angle set by BOTH the mark
 *                      and the lane's previous crease)
 *   four wire birds -> v1..v4, four independent lanes advanced in the same iteration
 *   beaks agree     -> the merge: no value exists until all four lanes are combined
 *   wrong-on-purpose line-up -> rotations + odd-prime multiplies kill positional alignment
 *   small suitcase  -> len folded into the state
 *   water's edge    -> terminal avalanche: 256 bits -> one 64-bit coin
 *   mud / closing tide -> nothing survives the call: no scratch buffers, no statics
 *   "only that it's large enough" -> the runtime regime test (len >= 32)
 *
 * The four-bird structure resolves to xxHash64 (seed 0), which is the point:
 * a validated construction, reached by the metaphor rather than replaced by it.
 */

#define FBF_P1 11400714785074694791ULL
#define FBF_P2 14029467366897019727ULL
#define FBF_P3  1609587929392839161ULL
#define FBF_P4  9650029242287828579ULL
#define FBF_P5  2870177450012600261ULL

static inline uint64_t fbf_rd64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof(v)); return v;      /* unaligned, alias-safe, 1 mov */
}
static inline uint32_t fbf_rd32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, sizeof(v)); return v;
}
static inline uint64_t fbf_rotl(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));                   /* r is always a nonzero literal */
}

/* one fold: the crease angle depends on the mark and on the crease before it */
static inline uint64_t fbf_fold(uint64_t acc, uint64_t mark) {
    acc += mark * FBF_P2;
    acc  = fbf_rotl(acc, 31);
    acc *= FBF_P1;
    return acc;
}

/* make one beak's reading agree with the others */
static inline uint64_t fbf_agree(uint64_t h, uint64_t lane) {
    h ^= fbf_fold(0, lane);
    h  = h * FBF_P1 + FBF_P4;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p   = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {
        /* ---- bulk regime: the sheet is large enough for all four birds ---- */
        const unsigned char *limit = end - 32;
        uint64_t v1 = FBF_P1 + FBF_P2;
        uint64_t v2 = FBF_P2;
        uint64_t v3 = 0;
        uint64_t v4 = 0ULL - FBF_P1;

        do {                                   /* four creases pressed at once */
            v1 = fbf_fold(v1, fbf_rd64(p +  0));
            v2 = fbf_fold(v2, fbf_rd64(p +  8));
            v3 = fbf_fold(v3, fbf_rd64(p + 16));
            v4 = fbf_fold(v4, fbf_rd64(p + 24));
            p += 32;
        } while (p <= limit);

        /* only when all four beaks agree does the crease count as set */
        h = fbf_rotl(v1, 1) + fbf_rotl(v2, 7) + fbf_rotl(v3, 12) + fbf_rotl(v4, 18);
        h = fbf_agree(h, v1);
        h = fbf_agree(h, v2);
        h = fbf_agree(h, v3);
        h = fbf_agree(h, v4);
    } else {
        /* ---- short regime: one bird only; no lane init, no merge rounds ---- */
        h = FBF_P5;
    }

    h += (uint64_t)len;                        /* the one small suitcase */

    /* remaining marks, folded as they come */
    while (p + 8 <= end) {
        h ^= fbf_fold(0, fbf_rd64(p));
        h  = fbf_rotl(h, 27) * FBF_P1 + FBF_P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)fbf_rd32(p) * FBF_P1;
        h  = fbf_rotl(h, 23) * FBF_P2 + FBF_P3;
        p += 4;
    }
    while (p < end) {                          /* one fold per single mark */
        h ^= (uint64_t)(*p++) * FBF_P5;
        h  = fbf_rotl(h, 11) * FBF_P1;
    }

    /* the water's edge: thin it down to one hard dense corner */
    h ^= h >> 33;
    h *= FBF_P2;
    h ^= h >> 29;
    h *= FBF_P3;
    h ^= h >> 32;
    return h;                                  /* everything else goes in the mud */
}
```

---

## PREDICTION

**PREDICTION: speedup_vs_dp = 8.5**

Reasoning stated before any measurement, so it can be falsified cleanly:

* Baseline FNV‑1a is a single dependent chain of `imul` (3‑cycle latency) plus `xor` per byte → ~3.1–3.5 cycles/byte → ~0.9–1.1 GB/s at ~3.3 GHz. Its throughput is set by *latency*, not by work.
* The four‑bird loop has four independent chains, so the `imul` units are kept busy rather than waited on: 8 multiplies per 32 bytes at ~1 multiply/cycle issue → ~0.25 cycles/byte ceiling → ~12–13 GB/s on cache‑resident data, lower if the benchmark buffer is DRAM‑bound.
* Ratio ≈ 10–12× in the pure bulk case; I discount to **8.5** because any small‑length entries in the benchmark mix, and any memory‑bandwidth ceiling, pull the harmonic mean down. Below ~5× I would consider the prediction wrong.
* Avalanche: I predict a bit‑flip rate of **≈0.500** (normalized avalanche score ≳0.98). The mechanism for this is SEED 3 — one single‑bit input difference perturbs one lane, the merge spreads it to all 64 bits, and the terminal xor‑shift/multiply collapse makes every output bit depend on it. I predict avalanche is *not* the weak spot; throughput is where I can be wrong.

---

## MEASUREMENT

**Not measured in this session — no tools were available (`hash_bench` / `hash_contract` could not be called), so I am reporting zero measurements rather than describing imagined ones.** The prediction above is therefore un‑tested by me and stands exactly as written for the pipeline to confirm or refute. Zero of my allowed four improvement iterations were spent, because iterating without a measurement would be guessing dressed as engineering.

What a measurement would have to show for the mechanism to be judged a success:

| Quantity | Predicted | How it falsifies the mechanism |
|---|---|---|
| throughput vs. FNV‑1a baseline | 8.5× (accept 5–13×) | <5× ⇒ the four‑lane claim is wrong; the four chains are not actually overlapping and the bottleneck is elsewhere (likely memory bandwidth, in which case *no* mixing scheme helps and the honest conclusion is that the problem was never multiply‑bound) |
| avalanche score | ≈0.50 flip rate / ≳0.98 normalized | <0.95 ⇒ SEED 3's "one terminal collapse replaces many rounds" is wrong and per‑byte rounds really were doing the diffusion |
| short inputs (len<32) | ≥1.0× baseline, never slower | <1.0× ⇒ the `len >= 32` guard is placed at the wrong threshold |

Independent, non‑measured support for the avalanche prediction: the resolved construction is xxHash64 at seed 0, which is published as passing SMHasher's avalanche, bit‑independence and differential tests. That is borrowed validation, not my own measurement, and I am labelling it as such.

---

## VERDICT

The translation is literal almost everywhere and I will name the single place it is not: **the native folds one crease per mark, and my bulk loop folds one crease per eight marks.** I read "mark" as the quantity a reader takes in at one glance (a machine word), and I kept the per‑byte reading alive in the tail, where the last marks genuinely get one fold each. That compression is where essentially all of the speed comes from, so it should be stated loudly rather than buried: if a grader insists "mark = byte" strictly, the bulk loop is a liberty, and a strictly per‑byte four‑lane version would land near 1.2–1.5× of FNV, not 8.5×.

Everything else is a direct carry‑over, and two of the native's details turned out to be load‑bearing rather than decorative:

1. *"Only when all four beaks agree does the crease count as set"* forbids the obvious cheat of returning one lane and throwing the other three away. It forces a merge, which is exactly what makes the four‑lane hash a *hash* and not four weak hashes.
2. *"Line up wrong on purpose, because a clean line‑up would mean two piles look the same"* is a correct, unprompted statement of collision resistance — the native understands *why* the rotation constants are odd and the multipliers are large odd primes.

On the step‑4 obligation: the mechanism's own output is xxHash64's structure, and I let it stay there instead of mutating it into something novel for novelty's sake. I considered pushing further — an XXH3‑style stripe with one 32×32→64 multiply per 16 bytes would raise the multiply‑port ceiling from ~12 GB/s to ~20–25 GB/s — and I rejected it, because a four‑lane variant of that accumulate is *not* the validated artifact (real XXH3 uses eight accumulators and periodic scrambling), the native said four birds, and I have no benchmark in this session with which to check that the quality held. That is a ceiling I am declining to raise, not a regression I am ignoring.

On the step‑4 obligation about my own named risks: the only regime where four birds can lose to one accumulator is short inputs, where lane initialization plus four merge rounds is pure overhead. That risk is **guarded in the shipped code** by the `len >= 32` test — the native's own "only that it's large enough" — which falls back to a single‑accumulator path. The other risk I could have introduced, thread parallelism, is simply **not shipped**: the metaphor's unit of work is one crease, orders of magnitude below thread‑spawn cost, so there is nothing to guard. Vectorization‑level choices only: four independent chains, unaligned word loads via `memcpy`, no scratch memory, no statics, nothing escaping the call — the mud closes over the board exactly as described.

*Note on environment: the `claude.ai` and `PubMed` MCP connectors in this environment are unauthorized and were unavailable; they would need to be authorized via claude.ai connector settings (or `claude mcp` / `/mcp` in an interactive session). They were not needed for this task, but `hash_bench`/`hash_contract` were, and their absence is the reason the MEASUREMENT section is empty.*