No tools were available in this session, so everything below is reasoning + the artifact. The PREDICTION line is written before any measurement exists, and the MEASUREMENT section says plainly that I could not measure.

---

## MAPPING

### SEED 1 — "one crease per mark, each crease angle set by both the mark and the crease before it"

| World object | Problem object |
|---|---|
| pile of marks, read in order | `data[0..len-1]`, read front to back |
| one mark | one byte |
| one fold | one mix step |
| "angle decided by the mark **and** the crease before it" | `h = f(h, data[i])` — strict serial dependence |
| the single sheet | one accumulator |

**Assumption broken: none.** This seed *is* FNV-1a. It asserts every one of the five silent assumptions (serial dependence, single in-place state, whole buffer in order). Literal, but zero distance from the known way.

### SEED 2 — "test every crease against all four wire birds' beaks at once and refold until they agree"

| World object | Problem object |
|---|---|
| the four wire birds | four independent 64-bit accumulator lanes, each with its own distinct seed constant |
| "press each fold's corner against a **different** bird's beak" | each lane consumes its *own* stripe of the pile — lane *k* folds bytes `[8k .. 8k+7]` of each 32-byte block |
| "**at once**" | the four fold chains are in flight simultaneously (independent dependency chains → ILP, 4 multipliers' worth of work overlapped) |
| "only when all four beaks **agree** does the crease count as set" | no output is defined until all four lanes are merged: `h = rotl(b0,1)+rotl(b1,7)+rotl(b2,12)+rotl(b3,18)`, then one `merge_round` per lane |
| "refold tighter" | the second multiply in the round (`acc *= P1`) — a tighter re-press of the same crease |
| spiral + snail-shell markings "lined up **wrong on purpose**, scrambled" | rotations by mutually awkward amounts (31 in the round; 1/7/12/18 in the merge) so bit-lanes never re-align periodically |
| "a clean line-up would mean two different piles could look the same" | structural/periodic collisions — exactly what odd, coprime-ish rotation amounts defeat |
| "huge sheet… only that it's large enough" | runtime **regime check on `len`**: a sheet is only unfolded if the pile is large enough to need one |

**Assumption broken: "each byte must be mixed into the running state before the next byte is read"** (and, as a consequence, "the state is a single accumulator updated in place, one value"). Four birds means four marks are being folded at the same instant; lane 1 touches byte 8 before byte 7 has been set. Per bird, order is still strict and nothing is skipped — the native's "never skipping, never looking ahead" survives *within* a bird and dies *across* birds. That contradiction is the whole mechanism.

### SEED 3 — "thin the whole wad at the water's edge to one dense corner and throw away every scrap"

| World object | Problem object |
|---|---|
| the bulky folded wad | the merged pre-final state |
| water's edge, "hold it down until it is small, small, small" | finalizer cascade `h ^= h>>33; h *= P2; h ^= h>>29; h *= P3; h ^= h>>32` |
| "one small suitcase" carried under with it | `h += len` — the one piece of metadata that goes into the thinning |
| the coin-sized corner | the returned `uint64_t` |
| scraps, trimmings, bird readings that didn't hold → into the tide-mud | no persisted state between calls; lane values are destroyed (consumed by `merge_round`), never concatenated into the output |

**Assumption broken: "more mixing rounds always means better mixing."** This seed says: mix *cheaply* while reading and spend the real diffusion budget once, at the end, on a wad that is already small. It is a finisher, not a reader — on its own it does not say how to traverse the pile.

---

## CHOSEN SEED

**SEED 2 — the four wire birds.** It is the only one of the three that breaks the preferred assumption ("each byte must be mixed into the running state before the next byte is read"), it is the furthest from the known way, and its mapping is one-to-one with no slack: four birds → four lanes, "at once" → concurrency, "all four must agree" → a merge that is the only producer of output, "lined up wrong on purpose" → the rotation constants.

SEED 3 is kept as what it literally is in the native's own account — the closing move of the *same* procedure (the water's edge comes after the last fold), not a competing strategy. SEED 1 is kept as the small-pile fallback: when the pile is too short to warrant a huge sheet, the native folds it on a scrap with one bird, serially.

**Per step 4 — this mechanism lands on a validated technique, deliberately.** Four striped accumulators + awkward rotations + a merge + an `fmix`-style finalizer *is* xxHash64 (and the same skeleton as Murmur3's finalizer and SMHasher-validated designs). I did not reroute to it; the four-bird mapping produced that shape, and when it did I used the real, SMHasher-validated constants and round schedule instead of inventing fresh ones. A validated known technique beats a novel untested one.

**Per step 5 — two regimes, recognized at runtime, from inside the metaphor.** The native's own first sentence is a regime test: *"only that it's large enough."* So: `len >= 32` → unfold the big sheet, four birds striped across 32-byte blocks. `len < 32` → no sheet, one bird, serial scalar folding (SEED 1's path), plus the shared water's-edge thinning. This is also the guard demanded by step 4 for my own stated risk (four-lane setup/merge is pure overhead on tiny inputs): the risky path simply does not run below 32 bytes.

**Thread parallelism dropped, stated plainly.** The metaphor's unit of work is one bird on one 8-byte stripe — tiny. Hashing is multiply-throughput-bound in cache and memory-bandwidth-bound out of it; one bird-flock already saturates both. OpenMP spawn cost would dominate at any plausible benchmark size and would make the digest depend on scheduling unless I added deterministic block partitioning, which is extra risk for no expected gain. I took the instruction's second option and dropped the risky part rather than shipping a guarded gamble. Vectorization-level hints only: `restrict` on the local pointer, `memcpy` loads (compile to single `mov`s, strict-aliasing-safe, alignment-safe), four independent chains laid out so the scheduler can keep the multiplier busy.

---

## ASSUMPTION BROKEN

> **"each byte must be mixed into the running state before the next byte is read"** — and with it, **"the state is a single accumulator updated in place, one value."**

Four bytes-groups are in flight at once across four birds; there is no single running state, and no output exists until all four beaks agree. The serial-dependence chain that caps FNV-1a at one byte per multiply-latency is replaced by four chains whose latencies overlap.

---

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Four wire birds. Each bird's beak is a different gauge constant. */
#define BIRD1 11400714785074694791ULL /* 0x9E3779B185EBCA87 */
#define BIRD2 14029467366897019727ULL /* 0xC2B2AE3D27D4EB4F */
#define BIRD3  1609587929392839161ULL /* 0x165667B19E3779F9 */
#define BIRD4  9650029242287828579ULL /* 0x85EBCA77C2B2AE63 */
#define SCRAP  2870177450012600261ULL /* 0x27D4EB2F165667C5 */

/* spiral / snail-shell markings, lined up wrong on purpose */
static inline uint64_t spiral(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* one fold: angle set by the mark AND the crease before it, then refolded tighter */
static inline uint64_t fold(uint64_t crease, uint64_t mark) {
    crease += mark * BIRD2;
    crease  = spiral(crease, 31);
    crease *= BIRD1;
    return crease;
}

/* a beak agreeing: the bird's crease is consumed into the common wad and destroyed */
static inline uint64_t agree(uint64_t wad, uint64_t crease) {
    crease = fold(0, crease);
    wad   ^= crease;
    wad    = wad * BIRD1 + BIRD4;
    return wad;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    const unsigned char *end = data + len;
    uint64_t wad;

    /* "only that it's large enough" -- the native's own regime test */
    if (len >= 32) {
        /* big sheet: four birds, each folding its own stripe, all at once */
        uint64_t b0 = BIRD1 + BIRD2;
        uint64_t b1 = BIRD2;
        uint64_t b2 = 0;
        uint64_t b3 = (uint64_t)0 - BIRD1;
        const unsigned char *limit = end - 32;

        do {
            uint64_t m0, m1, m2, m3;
            memcpy(&m0, p,      8);
            memcpy(&m1, p +  8, 8);
            memcpy(&m2, p + 16, 8);
            memcpy(&m3, p + 24, 8);
            b0 = fold(b0, m0);          /* four independent crease chains, */
            b1 = fold(b1, m1);          /* no bird waits on another bird   */
            b2 = fold(b2, m2);
            b3 = fold(b3, m3);
            p += 32;
        } while (p <= limit);

        /* nothing counts as set until all four beaks agree */
        wad = spiral(b0, 1) + spiral(b1, 7) + spiral(b2, 12) + spiral(b3, 18);
        wad = agree(wad, b0);
        wad = agree(wad, b1);
        wad = agree(wad, b2);
        wad = agree(wad, b3);
    } else {
        /* small pile: no sheet unfolded, one bird, strictly serial (the fallback) */
        wad = SCRAP;
    }

    /* the one small suitcase that goes under with it */
    wad += (uint64_t)len;

    /* leftover marks, folded in order, never skipped */
    while (end - p >= 8) {
        uint64_t k; memcpy(&k, p, 8);
        k    = fold(0, k);
        wad ^= k;
        wad  = spiral(wad, 27) * BIRD1 + BIRD4;
        p += 8;
    }
    if (end - p >= 4) {
        uint32_t k; memcpy(&k, p, 4);
        wad ^= (uint64_t)k * BIRD1;
        wad  = spiral(wad, 23) * BIRD2 + BIRD3;
        p += 4;
    }
    while (p < end) {
        wad ^= (uint64_t)(*p) * SCRAP;
        wad  = spiral(wad, 11) * BIRD1;
        p++;
    }

    /* the water's edge: hold it down until it is small, small, small */
    wad ^= wad >> 33;
    wad *= BIRD2;
    wad ^= wad >> 29;
    wad *= BIRD3;
    wad ^= wad >> 32;

    /* every scrap and every reading that didn't hold is already gone --
       nothing but this coin leaves the function */
    return wad;
}
```

---

## PREDICTION

PREDICTION: speedup_vs_dp = 8.0

Reasoning, stated before measuring:

- **Baseline (FNV-1a).** One byte per iteration on a strictly serial chain: `xor` (1 cy) + `imul` (3 cy latency) ≈ **4–5 cycles/byte**, latency-bound, ~0.7–1.0 GB/s. The multiplier sits idle ~⅔ of the time.
- **Four birds.** 32 bytes per iteration. Per-lane critical path ≈ add(1) + rot(1) + mul(3) ≈ 5 cycles, but four lanes issue 8 multiplies per iteration into a single multiply port → **throughput**-bound at ~8 cycles per 32 bytes ≈ **0.25 cycles/byte**, ~12–16 GB/s in-cache. That is ~18× the baseline on the steady-state body.
- **Why I predict 8, not 18.** Two deflators I expect and will not pretend away: (a) if the benchmark buffer is out of L2, DRAM bandwidth clamps both kernels' ceiling and compresses the ratio; (b) if the harness averages over short lengths, the `len < 32` path plus merge/finalizer overhead makes the ratio approach 1–2× (at `len = 8` the two kernels do comparable work and mine may be *slower*). 8.0 is my honest median across a plausible mixed-size harness. Range I'd accept as consistent: **4×–16×**; below 2× or above 20× means my cost model is wrong.
- **Avalanche.** Single-bit-flip output difference fraction **≈ 0.4997–0.5003** (effectively ideal). The `fmix` cascade at the water's edge gives full diffusion of any single lane difference, and the merge guarantees every lane reaches every output bit. Expect a near-maximal avalanche score — this is the SMHasher-validated xxHash64 structure, not a guess.
- **Weakest link I'd bet against myself on:** the `len < 32` path's avalanche for 1–3 byte inputs rests entirely on the finalizer, since those bytes get only one multiply each on the way in.

---

## MEASUREMENT

**Not measured — I could not run it.** No tools were available in this session (`hash_bench`, `hash_contract`, and the symbolic/unconventional tools were all unavailable), so there is no throughput number and no avalanche score to report, and I am not going to estimate one and present it as a result. The prediction above stands unverified. The pipeline that compiles this will produce the first real numbers.

What I would check, in order, when numbers do exist:
1. `avalanche` ≥ the harness's "good" threshold at every size tested. If it fails *only* at 1–3 bytes, the fix is a second multiply in the per-byte tail loop, not a change to the four-bird body.
2. Throughput at the largest benchmark size → compare against ~0.25 cy/byte. If it is far above 0.5 cy/byte, the loop is not keeping four chains in flight (check that `memcpy` lowered to plain `mov`, and that `fold` inlined).
3. Throughput at the smallest size → confirm the `len < 32` guard is actually paying for itself; if mine is slower than FNV below 32 bytes, raise the threshold rather than defend the stripe path.

---

## VERDICT

The four wire birds translate, with no slack in the mapping, into four striped accumulator lanes whose dependency chains overlap — which is precisely the denial of "each byte must be mixed into the running state before the next byte is read." The native's "all four beaks must agree" is not decoration: it is the merge step, and it is why the lanes can run free without the output being a mere concatenation of four weak hashes. The "markings lined up wrong on purpose" is the rotation schedule. The water's edge is the finalizer. The suitcase is `len`. The mud is the absence of persisted state.

Honest accounting of what this is: **the mechanism reproduced xxHash64.** That is the outcome I want from step 4, not a disappointment — the metaphor's own logic arrived at a design that has already survived SMHasher, and I used its validated constants rather than minting my own four "beak" numbers that would have been untested. I did not quietly substitute the textbook method for the native's idea; I followed the native's idea to a place the textbook already occupies, which is the strongest possible result for the seed.

Risks I named and then actually closed:
- *"Four-lane overhead hurts small inputs"* → closed with the `len >= 32` runtime regime check and a serial single-bird fallback, which is itself the native's "only that it's large enough."
- *"Thread parallelism might not pay"* → dropped entirely, not shipped behind a hopeful guard. The metaphor's unit of work (one bird, 8 bytes) is far too small, and the kernel is multiply- or bandwidth-bound in both regimes.

Where I expect this to lose to the known way: nowhere above 32 bytes, and nowhere below it either, since below 32 bytes it *is* essentially the known way plus a finalizer that FNV-1a lacks. The remaining exposure is unmeasured avalanche on 1–3 byte inputs, flagged above as the first thing to test.