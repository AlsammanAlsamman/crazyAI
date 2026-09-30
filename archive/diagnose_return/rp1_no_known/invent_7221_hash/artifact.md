## MAPPING

**SEED 1 — "one crease per mark, angle set by the mark *and* the crease before it"**

| World object | Problem object |
|---|---|
| pile of marks, read in order, never skipping | the byte buffer, consumed left to right |
| one mark | one 64-bit mark-word (8 bytes) in the big-sheet path; one byte in the small-pile path |
| one fold | one state-update step |
| "angle decided by the mark **and** the previous crease" | rotate amount = `((mark ^ crease) & 31) + 16` — a *data-dependent* rotation, not a constant |
| the crease left behind | the accumulator value carried into the next step |
| "the paper carries forward everything it has already been told" | the update is a chained, invertible-per-step function of all prior marks |

*Breaks:* "mixing one byte requires a multiplication" — partly; the nonlinearity here is a **data-dependent rotation**, the multiply is only the final press. It does **not** break the serial-dependence assumption; it affirms it. This seed alone ≈ FNV with a variable rotate.

**SEED 2 — "test every crease against all four wire birds' beaks at once, refold until they agree"**

| World object | Problem object |
|---|---|
| four wire birds | **four independent 64-bit accumulators** (four gauges, not one) |
| "press each fold's corner against a *different* beak" | mark *i* goes to lane *i mod 4* — consecutive marks never touch the same accumulator |
| each beak's own shape | each lane has its **own odd multiplier** (its own prime) |
| "only when all four beaks agree does the crease count as set" | the token is a function of **all four** lanes; no single lane is the hash |
| "refold tighter until the spiral and snail-shell markings line up **wrong on purpose**" | deliberate asymmetry: distinct primes per lane and distinct merge rotations, so no permutation of lanes/blocks can collide by symmetry ("two different piles looking the same") |
| the huge sheet, "only that it's large enough" | the 4-lane rig is only laid out when `len` is big enough to pay for it — a **runtime regime test** |

*Breaks:* **"each byte must be mixed into the running state before the next byte is read"** (four disjoint chains → byte *i+1* is folded while byte *i* is still in flight) **and** "the state is a single accumulator updated in place, one value".

**SEED 3 — "thin the whole wad at the water's edge to one dense corner; throw away every scrap and misreading"**

| World object | Problem object |
|---|---|
| carrying the wad to the water | the finalization step, run **once**, not per byte |
| "thins the way a body thins going under with one small suitcase" | 256 bits of lane state compressed to 64 bits |
| "hold it down until it is small, small, small" | three xor-shift/multiply rounds (full 64-bit avalanche) |
| the coin-sized hard corner | the returned `uint64_t` |
| every used sheet, trimmed scrap and failed reading thrown in the mud | no scratch buffer, no raw lane value leaks into the output |

*Breaks:* "more mixing rounds always means better mixing" — the per-mark work is deliberately *thin*, and all the heavy mixing is spent once, at the end, where it is cheap per byte.

## CHOSEN SEED

**SEED 2.** It is the only seed that breaks the preferred assumption, and it is the furthest from the known way (one accumulator, one strict chain). SEED 1 supplies the per-fold rule *inside* each bird's lane and SEED 3 supplies the water's-edge finalizer — they are the same native's same procedure, so I keep all three rather than amputating two of them.

## ASSUMPTION BROKEN

*each byte must be mixed into the running state before the next byte is read.* Four birds means four creases in progress at once: the dependency chain between adjacent marks is severed, and the only place the four meet is the water's edge. Secondary breaks: single-accumulator state (four), and per-byte heavy mixing (thin per mark, heavy once).

**Regime recognition, in-world:** "only that it's large enough" is the native's own size test. A small pile never gets the big sheet or the bird rig — it is folded against one beak, mark by mark (and byte by byte below 8), then thinned. Two paths, one runtime check.

**Thread parallelism declined:** the metaphor has one folder and one sheet; four beaks is lane-level, not thread-level, and hashing is bandwidth-bound, so OpenMP would add nondeterminism and fork cost for nothing. I take the vectorization-hint route instead (`restrict` on a local copy of the pointer so the contract is untouched, `memcpy` loads, 32-byte-blocked reads).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the four wire birds: each beak its own prime, its own merge angle ---- */
#define BEAK0 0x9E3779B97F4A7C15ULL
#define BEAK1 0xC2B2AE3D27D4EB4FULL
#define BEAK2 0x165667B19E3779F9ULL
#define BEAK3 0x27D4EB2F165667C5ULL
#define BASIS 1469598103934665603ULL            /* the blank sheet */

#define ROTC(x,c) (((x) << (c)) | ((x) >> (64 - (c))))   /* fixed beak angle, 1..63 */

static inline uint64_t read_mark(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;      /* one mark, read in order */
}

/* ONE FOLD.  The angle is decided by two things together -- the mark itself and
   the crease left by the fold before it -- and then the corner is pressed
   against this bird's beak.  Angle is forced into 16..47 so it is never 0/64. */
static inline uint64_t press(uint64_t crease, uint64_t mark, uint64_t beak) {
    unsigned a = (unsigned)((mark ^ crease) & 31u) + 16u;
    uint64_t bent = (crease << a) | (crease >> (64u - a));
    return (bent ^ mark) * beak;
}

/* THE WATER'S EDGE.  Thin the whole wad down to one hard dense corner no
   bigger than a coin; nothing else survives to be compared next time. */
static inline uint64_t thin_to_coin(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 32;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* "pink or orange, doesn't matter, only that it's LARGE ENOUGH":
       a small pile never gets the big sheet or the four-bird rig -- one beak,
       mark by mark, then straight to the water.  This is the fallback path. */
    if (len < 32) {
        uint64_t c = BASIS ^ (BEAK0 * (uint64_t)len);   /* thickness of the wad */
        size_t i = 0;
        for (; i + 8 <= len; i += 8) c = press(c, read_mark(p + i), BEAK1);
        for (; i < len; i++)         c = press(c, (uint64_t)p[i], BEAK1);
        return thin_to_coin(c);
    }

    /* four birds, four creases in progress at once */
    uint64_t l0 = BASIS ^ (BEAK0 * (uint64_t)len);
    uint64_t l1 = BASIS + BEAK1;
    uint64_t l2 = BASIS ^ BEAK2;
    uint64_t l3 = BASIS - BEAK3;

    size_t i = 0;
    const size_t blocks = len & ~(size_t)31;

    /* each successive mark is pressed against a DIFFERENT beak, cycling 0,1,2,3 */
    for (; i < blocks; i += 32) {
        const uint64_t m0 = read_mark(p + i);
        const uint64_t m1 = read_mark(p + i +  8);
        const uint64_t m2 = read_mark(p + i + 16);
        const uint64_t m3 = read_mark(p + i + 24);
        l0 = press(l0, m0, BEAK0);
        l1 = press(l1, m1, BEAK1);
        l2 = press(l2, m2, BEAK2);
        l3 = press(l3, m3, BEAK3);
    }

    /* the scraps trimmed off the edge -- still round-robin across the beaks */
    size_t rem = len - i;
    if (rem >= 8) { l0 = press(l0, read_mark(p + i), BEAK0); i += 8; rem -= 8; }
    if (rem >= 8) { l1 = press(l1, read_mark(p + i), BEAK1); i += 8; rem -= 8; }
    if (rem >= 8) { l2 = press(l2, read_mark(p + i), BEAK2); i += 8; rem -= 8; }
    if (rem)      { l3 = press(l3, read_mark(p + len - 8), BEAK3); } /* len>=32: in range */

    /* only when all four beaks agree does the crease count as set:
       distinct angles, so no two piles line up cleanly by symmetry. */
    uint64_t x = ROTC(l0,1) + ROTC(l1,7) + ROTC(l2,12) + ROTC(l3,18);
    return thin_to_coin(x);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 10**

Stated before any measurement, and falsifiable in detail:

- Per-lane critical path ≈ 8 cycles (xor → count → variable rotate → xor → imul), four independent lanes consuming 32 bytes ⇒ **~4 bytes/cycle ≈ 13–15 GB/s**, against FNV-1a's 4-cycle-per-byte chain ⇒ **~0.9 GB/s**. Compute-bound ratio ≈ 15×; capped by memory bandwidth on large buffers and by call overhead on small ones, hence **10** as the reported figure. Plausible outcome band: 6–18×.
- Avalanche: ~0.5 flipped output bits per flipped input bit (score at or near the tool's ideal). The three-round coin-thinning is a bijection with full 64-bit diffusion, and a single-lane delta cannot cancel in a sum of distinct rotations.

## MEASUREMENT

**Not performed — no tools are available in this session.** `hash_bench`, `hash_contract` and the symbolic/unconventional tools were all unavailable, so I used **zero** of my four allowed improvement rounds on measured evidence. I am reporting that plainly rather than inventing numbers. What I did verify by hand: rotate counts confined to 16..47 (no shift-by-0/64 UB), the `p + len - 8` tail read reachable only when `len >= 32`, all four multipliers odd (each press invertible, so no lane collapses to a constant), and no scratch allocation.

If the pipeline's numbers disagree with the prediction, the three failure modes and their fixes, in order:

1. **Throughput below ~5×** → `rol %cl` is costing more than modelled on this uarch. Fix: keep the angle data-dependent but move its derivation off the critical path (`a` from the mark alone), restoring a 5-cycle chain. Costs a little literalness in SEED 1, keeps SEED 2 intact.
2. **Avalanche below ideal** → the last mark absorbed gets only one press before the merge. Fix: press each lane once more against its own beak before the merge (four extra multiplies total, amortized to nothing).
3. **Small-input throughput weak** → raise the 32-byte threshold, or absorb the small pile as two overlapping 8-byte marks instead of a word loop plus a byte loop.

## VERDICT

The native's procedure is not a worse FNV — it is a **four-lane sponge with a thin absorb and one heavy squeeze**, which is the same structural insight that makes xxHash/wyhash fast, arrived at from the wrong end: four wire birds instead of four accumulators, and a body going under with a suitcase instead of `fmix64`. The part that is genuinely *not* the known way is the data-dependent fold angle (SEED 1's "the angle is decided by two things together"), where the nonlinearity comes from a mark-and-crease-dependent rotation rather than from the multiply. That is a real design choice, not a translation artifact, and it is the part most likely to cost throughput.

Stated risks, and how each is discharged rather than left hanging:

- *"Only helps if the pile is large."* Discharged in code: `len < 32` takes the single-beak path — the four-lane rig is never laid out for a buffer too small to fill one block. This is the metaphor's own "only that it's large enough", not a bolted-on heuristic.
- *"Thread parallelism might not pay."* Discharged by dropping it. The metaphor's unit of work is one folder at one sheet; OpenMP is not in the description and would not survive a bandwidth-bound benchmark. Vectorization hints only.
- *"The coin-thinning costs 2 extra multiplies that FNV doesn't pay."* Not guarded, and I say so openly: for `len <= 3` this kernel is very likely **slower** than the baseline by roughly a nanosecond per call. I keep the finalizer there anyway, deliberately, because the benchmark also scores avalanche and baseline FNV-1a on a 1–3 byte input diffuses almost nothing; guarding the finalizer away would trade the axis that matters for the axis that doesn't at that size. That is a declared trade with a named crossover (`len ≈ 4`), not a silent one.

Where the native is simply wrong: "refold tighter until they agree" implies rejection sampling — an unbounded retry loop. There is no retry in a hash, and I did not fake one; I translated "agreement" as the *closure* condition (the token depends on all four beaks, no lane escapes alone), which is what the four-way merge enforces. That is the one place my translation is an interpretation rather than a transcription, and it should be read as such.