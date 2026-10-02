## MAPPING

### SEED 1 — "one crease per mark, angle set by both the mark and the crease before it"

| World object | Problem object |
|---|---|
| mark in the pile | one input byte `data[i]` |
| the pile, read in order, never skipping | the buffer, read sequentially once |
| one fold per mark | one state update per byte |
| fold angle ← mark **and** previous crease | `h = f(h, data[i])` — strict serial data dependency |
| crease left by previous fold | the single running accumulator |

**Assumption broken: none.** This seed *is* FNV-1a. It affirms "each byte must be mixed into the running state before the next byte is read" and "the state is a single accumulator updated in place."

### SEED 2 — "test every crease against all four wire birds' beaks at once, refold until they agree"

| World object | Problem object |
|---|---|
| the four wire birds | four independent 64-bit accumulator lanes `b1..b4`, each seeded differently ("a different beak") |
| "press **each** fold against **a different** bird's beak" | mark *i* is folded into lane *i mod 4* — four consecutive bytes-groups are in flight simultaneously |
| "at once" | the four lane updates of one 32-byte stripe are independent → 4-way instruction-level parallelism, no lane waits for another |
| "only when all four beaks agree does the crease count as set" | the stripe is not retired until all four lanes consumed their 8 bytes; a final **merge/agreement** step combines all four lanes into one value |
| "if they disagree I refold tighter" | the rotate-then-multiply inside each lane update (`rotl(acc,31) * P1`) — the fold is tightened, not loosened |
| "until spiral and snail-shell markings line up **wrong on purpose**, scrambled" | rotation amounts chosen so bit-fields deliberately *mis*-align between lanes and between rounds; "a clean line-up would mean two piles look the same" = that is a collision, stated in-world |
| the huge sheet, "pink or orange, doesn't matter, only that it's **large enough**" | the odd 64-bit prime constants: their particular value is irrelevant, their *magnitude/entropy* is not |
| "only that it's large enough" (of the sheet) | **runtime regime check**: if the pile won't fill a sheet (`len < 32`) you don't lay out the big sheet and you don't consult four birds — one bird only |

**Assumption broken: "each byte must be mixed into the running state before the next byte is read"** — and, simultaneously, **"the state is a single accumulator updated in place, one value."**

### SEED 3 — "thin the whole folded wad at the water's edge to one small dense corner, throw away every scrap"

| World object | Problem object |
|---|---|
| carrying the wad to the water's edge | the finalizer, applied *once*, after all input is consumed |
| thinning "the way a body thins going under **with one small suitcase**" | the length `len` is folded in at the finalize step — the suitcase that goes under with it |
| "one hard dense corner no bigger than a coin" | the 64-bit output: a fixed-width digest, all width collapsed |
| scraps, used sheets, bird-readings that didn't hold → the mud | registers/scratch discarded; the function is **pure**, nothing survives between calls to be compared |
| "if even one mark had been different … a different, unrecognizable corner" | the avalanche requirement itself |

**Assumption broken: "more mixing rounds always means better mixing."** The native does *not* mix hard per mark; per-mark folds are cheap, and *all* the strong mixing is concentrated in one final thinning. Partially also "the state is one value" (4 lanes → 1 coin).

## CHOSEN SEED

**SEED 2** — the four wire birds. It is the seed that breaks the preferred assumption ("each byte must be mixed into the running state before the next byte is read"), it is the most literal (four birds = four lanes, "agree" = merge, "a different beak" = a different seed), and it is maximally unlike the known way, whose defining feature is one accumulator on a serial critical path. SEED 3 is retained as the *finalizer* that SEED 2's merge feeds into, because the native uses both in one pass and they are not in conflict.

Honest tension: the preamble's "never looking ahead" belongs to SEED 1 and SEED 2 contradicts it — four birds pressed "at once" requires four marks in hand at once. I took SEED 2 literally and let it override.

## ASSUMPTION BROKEN

Primary: **each byte must be mixed into the running state before the next byte is read.** Four birds are consulted simultaneously; lane *k* never waits for lane *k−1*. The serial multiply chain of FNV-1a (xor+imul latency, ≈4 cycles per byte, ~0.25 B/cycle) becomes four interleaved chains over 32 bytes (≈5 cycles per stripe, ~6 B/cycle).

Secondary: **the state is a single accumulator** (four birds), and **more rounds is always better** (cheap per-mark folds, one hard thinning at the end).

Step-4 discipline: this mechanism, taken literally — 4×64-bit lanes over 32-byte stripes, a rotate-and-multiply round per lane, a rotate-weighted merge of the four lanes, length folded in, then an xor-shift-multiply finalizer — **is xxHash64**. I let the birds land on the validated, SMHasher-passing design rather than invent a new mixer. The constants, rotation amounts, merge and tail schedule below are the published xxHash64 ones; what the metaphor supplies is the *reason* for each part, including the `len < 32` regime fork and the one-bird fallback.

No OpenMP: the native is one folder, one sheet at a time, and the metaphor's own unit of work is a 32-byte sheet — far too small to pay a thread-dispatch cost, and threading would require redefining the digest as a tree hash, which the native never does. Per the instructions, vectorization/ILP only. No `immintrin.h` either: the 4-lane scalar form already reaches ~6 B/cycle, at or past single-core memory bandwidth for out-of-cache buffers, and AVX2 has no 64×64→low multiply (AVX512DQ only), so a vector path would be emulated, slower, and silently absent on machines without the ISA.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Four wire birds: four beaks, four different gauges.
   The sheet's colour does not matter, only that it is large enough:
   these constants matter only by magnitude/entropy, not by identity. */
#define BIRD_A 11400714785074694791ULL
#define BIRD_B 14029467366897019727ULL
#define BIRD_C  1609587929392839161ULL
#define BIRD_D  9650029242287828579ULL
#define COIN    2870177450012600261ULL

/* refold tighter: rotate so spiral and snail-shell markings line up WRONG on purpose */
static inline uint64_t refold(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* read marks off the pile (memcpy compiles to one unaligned load under -O3) */
static inline uint64_t mark64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t mark32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

/* one crease: angle set by the mark AND the crease before it (per bird, independently) */
static inline uint64_t crease(uint64_t acc, uint64_t m) {
    acc += m * BIRD_B;
    acc  = refold(acc, 31);
    acc *= BIRD_A;
    return acc;
}

/* "only when all four beaks agree does the crease count as set" */
static inline uint64_t agree(uint64_t h, uint64_t bird) {
    bird = crease(0, bird);
    h   ^= bird;
    h    = h * BIRD_A + BIRD_D;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p   = data;
    const unsigned char *end = data + len;
    uint64_t h;

    /* REGIME CHECK, in the native's own words: is the pile large enough
       to be worth laying out a huge sheet and consulting four birds? */
    if (len >= 32) {
        /* --- large regime: four birds, pressed at once, one sheet = 32 marks --- */
        const unsigned char *limit = end - 32;
        uint64_t b1 = BIRD_A + BIRD_B;
        uint64_t b2 = BIRD_B;
        uint64_t b3 = 0;
        uint64_t b4 = 0ULL - BIRD_A;
        do {
            b1 = crease(b1, mark64(p     ));
            b2 = crease(b2, mark64(p +  8));
            b3 = crease(b3, mark64(p + 16));
            b4 = crease(b4, mark64(p + 24));
            p += 32;
        } while (p <= limit);

        /* the four beaks are brought into agreement */
        h = refold(b1, 1) + refold(b2, 7) + refold(b3, 12) + refold(b4, 18);
        h = agree(h, b1);
        h = agree(h, b2);
        h = agree(h, b3);
        h = agree(h, b4);
    } else {
        /* --- small regime: fallback, one bird only, no sheet laid out --- */
        h = COIN;
    }

    /* the one small suitcase that goes under with it */
    h += (uint64_t)len;

    /* trimmed scraps: leftover marks, 8 then 4 then 1 at a time */
    while (p + 8 <= end) {
        h ^= crease(0, mark64(p));
        h  = refold(h, 27) * BIRD_A + BIRD_D;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)mark32(p) * BIRD_A;
        h  = refold(h, 23) * BIRD_B + BIRD_C;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * COIN;
        h  = refold(h, 11) * BIRD_A;
        p++;
    }

    /* the water's edge: thin it down, small, small, small, to one dense coin */
    h ^= h >> 33;
    h *= BIRD_B;
    h ^= h >> 29;
    h *= BIRD_C;
    h ^= h >> 32;
    return h;
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 10.0

Stated before any measurement, with the reasoning that produced it:

- Baseline FNV-1a is latency-bound on a single accumulator: `xor` (1 cycle) + 64-bit `imul` (3 cycles) = **~4 cycles/byte ≈ 0.25 B/cycle ≈ 0.8–0.9 GB/s** at ~3.5 GHz. It cannot be unrolled out of this; the dependency is real.
- Four birds: one stripe = 32 bytes, four independent chains of (`imul` 3 + `add` 1 + `rotl` 1 + `imul` 3) ≈ 8-cycle latency but 4-way overlapped and pipelined across iterations, bounded by ~16 µops/stripe at 4 µops/cycle ≈ **5–7 B/cycle ≈ 18–23 GB/s** in cache, clamped to single-core DRAM bandwidth (~10–15 GB/s) for buffers past L2.
- So the in-cache ratio is ~20–25×, the out-of-cache ratio ~12–15×, and anything the harness measures at or below ~64 bytes is ~1–3× (tail-dominated). I predict a reported figure around **10×**, i.e. I expect the harness to include small/medium sizes in whatever it aggregates. If it reports one large-buffer number only, I expect 15–25× and will call that a miss on the low side.
- Avalanche: ~**0.50** (near-ideal). This is published-xxHash64 mixing; the deliberate mis-alignment of rotations plus the three-stage xor-shift-multiply finalizer is what SMHasher's avalanche test already certifies. I would treat any score below 0.47 as a real failure, not a rounding artifact.

## MEASUREMENT

**Not obtained in this session — stated plainly rather than estimated and dressed up as data.** No tools were available to me here (`hash_bench` and `hash_contract` were listed but not callable), so I ran zero iterations of the four allowed improvement rounds and I have no measured throughput or avalanche number. The prediction above is therefore an unverified forecast, not a confirmed result. What I can report honestly:

- Compile-correctness checks I could do by inspection only: signature matches the fixed contract exactly (`uint64_t kernel(const unsigned char *data, size_t len)`, no added qualifiers); `len == 0` is handled (returns `avalanche(COIN + 0)`); `len` in 1..31 never touches the four-bird path; `limit = end - 32` is only formed when `len >= 32`; all reads go through `memcpy`, so no unaligned-access or strict-aliasing UB; no allocation, no scratch memory, no global state, so the function is reentrant and thread-safe.
- Endianness: like xxHash64, the digest value is little-endian-defined via `memcpy`. On a big-endian target the *value* differs; speed and avalanche do not. The contract asks for a hash, not a specific hash, so this is acceptable, but it is a real property worth naming.
- The one thing I most want measured and could not: whether the harness's reported throughput is in-cache or out-of-cache, because that alone moves the expected speedup between ~12× and ~25×.

## VERDICT

The four wire birds map, with no slack and no reinterpretation, onto four independent accumulator lanes, and "only when all four beaks agree does the crease count as set" maps onto a rotate-weighted merge of those lanes. Taken literally, that mechanism *is* xxHash64 — so the native's idea is not a curiosity, it is a rediscovery of the validated state of the art, and I let it arrive there instead of inventing a novel mixer I could not test. The assumption that breaks is the one that was worth breaking: FNV-1a's serial accumulator is a latency trap, not a necessity.

Risks I named, and what I did about each rather than leaving them stated-and-unaddressed:

1. *"Four lanes are overhead on short inputs."* Guarded: the `len >= 32` branch is the metaphor's own "only that it's large enough," and inputs below 32 bytes take a one-bird path with no lane setup and no merge. This is a genuine fallback to the simpler path, not a cosmetic check.
2. *"Thread parallelism could help on huge buffers."* Dropped, not shipped-and-hoped. The metaphor's unit of work is a 32-byte sheet and the native is a single folder; threading would force a tree-hash redefinition the native never performs, and the kernel is already at or past single-core bandwidth. Shipping an unmeasured OpenMP path would have violated the rule I was given.
3. *"SIMD should be faster."* Dropped for a concrete architectural reason, not taste: AVX2 lacks a 64×64→low multiply, so the round would be emulated and slower, and an AVX512DQ path would silently vanish on hardware without it.

Where this could still lose to the known way: inputs of roughly 1–16 bytes, where the four-bird path never runs and the tail loop plus the three-stage finalizer cost more total work than FNV-1a's handful of multiplies. I expect a *slowdown* there, plausibly 0.5–0.8×, and I am not hiding it behind the aggregate. If the harness weights tiny inputs heavily, the headline number will be poor and the honest conclusion would be that the mechanism is correct but mis-targeted — not that the measurement is wrong. The claim I am actually making is narrow and falsifiable: for buffers of 32 bytes and up, four birds beat one accumulator by roughly an order of magnitude at equal or better avalanche, and the first measurement taken should either confirm that band or refute me.