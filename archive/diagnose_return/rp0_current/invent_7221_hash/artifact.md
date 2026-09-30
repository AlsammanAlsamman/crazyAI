## 1. MAPPING

### SEED 1 — "one crease per mark, angle set by both the mark and the crease before it"

| World object | Problem object |
|---|---|
| huge sheet, "pink or orange, doesn't matter, only that it's **large enough**" | the buffer; colour = the content (irrelevant), **size is the only thing that changes strategy** — the native's own runtime regime test |
| the pile of marks, read in order, never skipping, never looking ahead | `data[0..len-1]`, each byte touched exactly once, in address order, **never read past `len`** (no over-read tricks) |
| one fold per mark | one mixing step per byte |
| the fold's *angle* | a rotation amount / the rotated state word |
| angle = f(this mark, previous crease) | `state = g(byte, state)` — a strict serial dependency chain |
| "the paper carries forward everything it has already been told" | the state is a full summary of the prefix |

**Assumption broken:** "mixing one byte requires a multiplication" — a *fold angle* is a rotation, not a product. It **affirms** the serial-dependency assumption, so it is the closest of the three to plain FNV-1a.

### SEED 2 — "test every crease against all four wire birds' beaks at once, refold until they agree"

| World object | Problem object |
|---|---|
| **four wire birds**, each a different gauge | **four independent 64-bit accumulator lanes**, each seeded with a different offset so no two gauges read alike |
| pressing one crease against all four beaks **at once** | every 32-byte stripe is measured by all four lanes **concurrently**; the four dependency chains never touch each other |
| "only when all four beaks agree does the crease count as set" | a stripe is complete only when all four lanes have absorbed their word; *agreement* is enforced once, at the merge, where all four must contribute |
| "if they disagree I refold tighter" | a per-lane `mergeRound` — an extra rotate×multiply stage before a lane joins the single value |
| spiral markings vs. snail-shell markings, made to line up **wrong on purpose** | deliberate inter-lane asymmetry: **distinct rotations (1, 7, 12, 18)** and an **ordered** merge |
| "a clean line-up would mean two different piles could end up looking the same" | a *symmetric* combination of lanes collides on lane-permuted input (swap the words at offsets 0 and 8 → same hash). Hence the asymmetry is load-bearing, not decoration |

**Assumptions broken:** "the state is a single accumulator updated in place, one value" **and** — the preferred one — **"each byte must be mixed into the running state before the next byte is read"**: byte *i+1* is pressed against a *different bird* than byte *i* and never waits for byte *i*'s reading to complete.

### SEED 3 — "thin the wad at the water's edge to one dense corner; throw away every scrap and misreading"

| World object | Problem object |
|---|---|
| carrying the wad to the water's edge | the finalization stage, after the last stripe |
| thinning to "one hard dense corner no bigger than a coin" | a 64-bit avalanche finalizer (xor-shift / multiply pairs) → the return value |
| "the way a body thins going under **with one small suitcase**" | the **length** goes under with it: `h += len` — so `"ab"` and `"ab\0"` cannot share a corner |
| every sheet, every scrap, every bad beak-reading thrown in the mud; tide closes over | **no global/static/scratch state survives the call**; nothing is carried between calls; pure function, no observable intermediates |
| "if even one mark had been different, the whole spiral would have folded down to a different, unrecognizable corner" | the avalanche requirement itself |

**Assumption broken:** "more mixing rounds always means better mixing" — the native does *one* short, exactly calibrated thinning at the end rather than piling rounds on every byte.

## 2. CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks the preferred assumption ("each byte must be mixed into the running state before the next byte is read"), and it is the most literal: *four* named physical instruments, used *simultaneously*, whose readings are reconciled only once at the end. Seeds 1 and 3 are the same ritual's other stages (the per-mark fold; the water's-edge thinning) and I keep them as the bulk round and the finalizer — but the **four birds** are the organ that makes this procedure structurally unlike FNV-1a.

**Honest note per step 4:** four independent 64-bit accumulators, fed 8 bytes each per 32-byte stripe, merged by distinct rotations and an ordered `mergeRound`, then thinned with the length folded in — that is the exact structural fingerprint of **xxHash64** (SMHasher-validated, deployed everywhere). So I let the four birds *arrive* at xxHash64 rather than inventing a new constant set. I specifically did **not** add the data-dependent rotation that SEED 1's "angle" literally invites: it would be an untested novelty, it costs a µop, and xxHash64 already satisfies "the new crease depends jointly on the mark and the crease before it."

## 3. ASSUMPTION BROKEN

Primary: **"each byte must be mixed into the running state before the next byte is read."** Four birds means four independent chains; consecutive bytes go to different birds. Secondary: **"the state is a single accumulator updated in place, one value"** (four lanes), and **"mixing one byte requires a multiplication"** (bytes are absorbed 8 at a time, so it is one multiply per *eight* bytes, not per byte).

**Regime recognition (step 5).** The native states it himself: *"only that it's large enough."* Four birds need enough sheet to hold four creases. Mapped: `len >= 32` → four-bird striped path; `len < 32` → single-chain path that skips the birds entirely and goes straight to the water's edge. This is the guard against my own stated risk (four-lane setup + a 4-stage merge is pure overhead on short inputs), with the simpler path as the fallback. No OpenMP: the metaphor has **one** folder with four birds, not four folders, and thread-launch overhead dwarfs a hash at any size a hash bench uses.

**Vectorization before threads:** the four independent chains *are* the vectorization — they give 4-way ILP over the 3-cycle `imul` latency. Explicit AVX2 cannot beat it, because AVX2 has no full 64×64→64 multiply (`vpmullq` is AVX-512DQ); emulating it costs 3 multiplies plus shifts. There are no stores in the kernel, so aliasing cannot constrain the loop and `restrict` would buy nothing.

## 4. ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the four birds' gauges, and the stone at the water's edge */
#define B1 11400714785074694791ULL
#define B2 14029467366897019727ULL
#define B3  1609587929392839161ULL
#define B4  9650029242287828579ULL
#define B5  2870177450012600261ULL

/* a crease: an angle, not a product */
static inline uint64_t crease(uint64_t x, int a) {
    return (x << a) | (x >> (64 - a));
}
static inline uint64_t mark8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;   /* one mov under -O3 */
}
static inline uint32_t mark4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, sizeof v); return v;
}
/* one fold: the new angle is set by the mark AND the crease before it */
static inline uint64_t fold(uint64_t acc, uint64_t m) {
    return crease(acc + m * B2, 31) * B1;
}
/* the beaks disagreed: refold tighter before this bird joins the wad */
static inline uint64_t refold(uint64_t h, uint64_t v) {
    return (h ^ fold(0, v)) * B1 + B4;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p   = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {                 /* "only that it's large enough" */
        const unsigned char *limit = end - 32;
        /* four birds, four different beaks - never seeded alike */
        uint64_t v1 = B1 + B2, v2 = B2, v3 = 0, v4 = (uint64_t)0 - B1;
        do {
            /* one crease, pressed against all four beaks at once;
               no bird waits on another bird's reading */
            v1 = fold(v1, mark8(p +  0));
            v2 = fold(v2, mark8(p +  8));
            v3 = fold(v3, mark8(p + 16));
            v4 = fold(v4, mark8(p + 24));
            p += 32;
        } while (p <= limit);        /* never skipping, never looking ahead */

        /* spiral and snail-shell lined up WRONG on purpose: distinct
           angles + an ordered merge, so no two piles fold the same way */
        h = crease(v1, 1) + crease(v2, 7) + crease(v3, 12) + crease(v4, 18);
        h = refold(h, v1);
        h = refold(h, v2);
        h = refold(h, v3);
        h = refold(h, v4);
    } else {
        h = B5;                      /* too small for the birds: one chain */
    }

    h += (uint64_t)len;              /* the one small suitcase goes under too */

    /* the last few marks, still strictly in order, still in bounds */
    while (end - p >= 8) {
        h ^= fold(0, mark8(p));
        h  = crease(h, 27) * B1 + B4;
        p += 8;
    }
    if (end - p >= 4) {
        h ^= (uint64_t)mark4(p) * B1;
        h  = crease(h, 23) * B2 + B3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * B5;
        h  = crease(h, 11) * B1;
        p++;
    }

    /* the water's edge: hold it down until it is small, small, small */
    h ^= h >> 33; h *= B2;
    h ^= h >> 29; h *= B3;
    h ^= h >> 32;
    return h;                        /* the coin. every scrap died above. */
}
```

## 5. PREDICTION

**PREDICTION: speedup_vs_dp = 12.0**

Reasoning stated before measurement: the FNV-1a baseline is a single chain of `xor`(1 cycle) + `imul`(3 cycles) **per byte** ≈ 4 cycles/byte ≈ 1 GB/s. The four-bird path runs four independent chains of ≈5 cycles each while consuming 32 bytes per iteration → ≈0.16 cycles/byte ≈ 20 GB/s theoretical, realistically 10–15 GB/s once load ports, loop overhead and (for large buffers) DRAM bandwidth bite. Secondary prediction: **avalanche ≈ 0.50 bit-flip probability, score ≥ 0.99** — xxHash64 passes SMHasher's avalanche and differential tests.

**Stated falsifiers, in advance:** (a) if the bench averages over many *small* buffers, speedup collapses toward ~2× — at `len=8` FNV costs ~32 cycles and my small path ~15; (b) if the bench is single-huge-buffer and DRAM-bound, speedup caps near the bandwidth ratio regardless of how good the kernel is; (c) if avalanche comes back below ~0.95 the mapping is wrong somewhere and I would suspect the lane merge, not the round.

## 6. MEASUREMENT

**Not measured — and I will not pretend otherwise.** This session was invoked with no tool access: `hash_bench` and `hash_contract` were unavailable, so the prediction above stands untested by me. The pipeline will compile and measure it. What I *can* assert without the bench, because it is structural rather than empirical:

- **In-bounds, in-order, no over-read.** `limit = end - 32` with `do {...} while (p <= limit)` consumes stripes only while a full 32 bytes remain; the tail uses `end - p >= 8 / >= 4 / p < end`. At `len == 32`: one stripe, zero tail. At `len == 0`: no loads at all, returns a fixed non-zero constant.
- **Determinism / purity.** No statics, no scratch, no globals, nothing carried between calls — the tide closes over everything but the return value.
- **Known-good value.** On a little-endian target this kernel is bit-identical to `XXH64(data, len, 0)`, so the avalanche claim is inherited from a widely validated implementation rather than asserted about a fresh design. (Portability caveat, honestly flagged: the hash value is little-endian-specific; the contract asks for a hash, not a byte-order-stable one.)

## 7. VERDICT

The metaphor was not decorative — it was a design document. "Four wire birds, pressed at once" is literally four independent accumulator lanes, and it is exactly the move that dissolves the assumption the problem statement flagged as the interesting one: consecutive bytes stop queueing behind one multiply chain. The metaphor then supplied, unprompted, three things a naive 4-lane sketch usually gets wrong: the **deliberate mis-alignment** of the spiral and snail-shell markings (per-lane distinct rotations + ordered merge — without it, swapping two 8-byte words collides), the **one small suitcase** (length folded in — without it, trailing-zero extensions collide), and the **water's-edge thinning** (a single calibrated finalizer instead of more rounds everywhere).

Where it could be worse than the known way, and what I did about it: the four-bird apparatus plus a four-stage merge is dead weight on short inputs. The native's own clause *"only that it's large enough"* is the guard, and it is implemented as `len >= 32` with a single-chain fallback — so the risk my own verdict names is addressed in the artifact, not just noted. I declined the one place the metaphor tempted me into novelty (SEED 1's data-dependent fold angle), because a validated technique beats an untested one; and I declined both explicit AVX2 (no 64-bit vector multiply below AVX-512DQ) and OpenMP (one folder, four birds — not four folders). If the bench reports a speedup near 12 with avalanche ≈ 0.5, the translation held. If it reports ~2×, the bench is small-buffer-dominated and the honest conclusion is that the birds had nothing to do — not that the kernel is wrong.