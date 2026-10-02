## MAPPING

I take the three seeds one at a time and write out what each world-object is, literally, in the hash problem.

**SEED 1 — "one crease per mark, angle set by the mark *and* the crease before it"**

| World object | Problem object |
|---|---|
| huge sheet (pink/orange — colour irrelevant) | the state register; its initial constant is arbitrary |
| pile of marks, read in order, never skipping | the byte buffer `data[0..len)`, scanned once in order |
| one fold per mark | one state update per byte |
| fold angle = f(mark, previous crease) | `h = mix(h, data[i])` — each update depends on the *immediately preceding* update |
| the paper "carries forward everything it has been told" | the serial dependency chain |

Assumption broken: **none.** This seed is a restatement of the serial chain. It *affirms* "each byte must be mixed into the running state before the next byte is read" and "the state is a single accumulator updated in place." SEED 1 is FNV-1a. Useless as a departure.

**SEED 2 — "test every crease against all four wire birds' beaks at once; refold until they agree; markings must line up *wrong*"**

| World object | Problem object |
|---|---|
| four wire birds, each a *different* bird | four independent 64-bit accumulators `b1..b4` |
| pressing the same fold against a different beak each | one 32-byte stripe split into four 8-byte lanes, each lane folded into its own accumulator |
| "all four beaks **at once**" | the four updates are mutually independent → four parallel mul/rotate chains in flight, not one chain |
| "only when all four beaks agree does the crease count as set" | the crease (the hash) is not defined by any single accumulator; it exists only in the final agreement = the merge of all four |
| "refold tighter until the spiral and snail-shell markings line up **wrong on purpose, scrambled**" | the lanes must be deliberately *de*symmetrised: distinct seeds per lane and distinct rotations at merge (`rotl 1, 7, 12, 18`), so lanes cannot be permuted |
| "a clean line-up would mean two different piles could look the same" | symmetric lanes ⇒ stripe-permutation collisions. The asymmetry is a *collision requirement*, not decoration |
| "the sheet must be large enough" (the one property of the sheet he checks) | runtime regime check: four gauges only get laid out if `len >= 32`; otherwise the small-sheet path |

Assumptions broken: **"the state is a single accumulator updated in place, one value"** and — the preferred one — **"each byte must be mixed into the running state before the next byte is read."** Bytes 8–15 are read and folded without waiting for bytes 0–7 to reach the state.

**SEED 3 — "thin the wad at the water's edge to one dense corner; throw away every scrap and misreading"**

| World object | Problem object |
|---|---|
| carrying the reduced wad to the water | the finalizer, applied once after the loop |
| "thins the way a body thins going under **with one small suitcase**" | the length is mixed in at finalization (`h += len`) — the suitcase is `len`, carried down with the body |
| "hold it down until it is small, small, small" | exactly three xor-shift/multiply contractions — three presses, not a loop |
| "one hard dense corner no bigger than a coin" | the 64-bit return value, fully avalanched |
| scraps, trimmed sheets, bird-readings that didn't hold → mud, tide closes over | no state survives the call; no scratch buffer, no table, nothing carried between invocations |
| "if even one mark had been different the whole spiral folds to a different corner" | avalanche: one input bit flip ⇒ ~32 output bits flip |

Assumption broken: **"more mixing rounds always means better mixing."** He presses a *fixed, small* number of times and then stops; everything else gets thrown in the mud. It also breaks the whole-buffer-in-order assumption only weakly.

## CHOSEN SEED

**SEED 2 — the four wire birds.** It is the most literal (four birds ⇒ four accumulators, nothing to stretch), the most distant from the known way (the known way has exactly one accumulator and one dependency chain), and it is the only seed that breaks the preferred assumption. SEED 3 is kept as the finalizer, because in the world it is the same procedure — the native folds *and then* goes to the water; refusing the water's-edge step would be a less literal reading of the native, not a more literal one.

Critically, step 4 applies here: four decorrelated lanes over 32-byte stripes, merged with distinct rotations and a three-press finalizer **is** XXH64. I let the birds arrive there rather than inventing a private mixer. The native's "line up wrong on purpose" is the exact reason xxHash merges with `rotl 1/7/12/18` and lane-specific seeds; the "small suitcase" is `h += len`; "small, small, small" is the three-step avalanche. So the artifact is a validated, SMHasher-passing construction reached through the metaphor, not a novel untested one.

## ASSUMPTION BROKEN

Primary: *each byte must be mixed into the running state before the next byte is read* — four beaks read **at once**, so four 8-byte lanes advance concurrently and the serial mul-latency chain (≈3–5 cycles/byte in FNV-1a) is replaced by four independent chains issuing in parallel (≈4 multiplies per 32 bytes).

Secondary: *the state is a single accumulator* — the state is four birds plus their disagreement; and *more rounds is better* — the finalizer is three presses, fixed.

Regime handling, as the metaphor itself demands (`"only that it's large enough"`): the known_way section describes a whole-buffer in-order scan with no size qualifier, i.e. both the long-pile and the few-marks regime. The native checks the sheet before unfolding four gauges. So `len >= 32` takes the four-bird striped path; `len < 32` takes the single-sheet path (seed `P5`, no striping, no merge) and still goes to the water's edge. The 8/4/1-byte tail is the "scrap trimmed off." No OpenMP: at the sizes a hash benchmark actually uses, a thread's unit of work (one bird, 8 bytes per stripe) is far below any fork cost, and the metaphor's unit of work is a *crease*, not a *shift of workers* — thread parallelism would be me importing something the native never said. Vectorization-level (ILP + `restrict` + unaligned `memcpy` loads that gcc folds to single `mov`s) only.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The four birds are different birds: four distinct wire gauges. */
#define P1 11400714785074694791ULL
#define P2 14029467366897019727ULL
#define P3  1609587929392839161ULL
#define P4  9650029242287828579ULL
#define P5  2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* unaligned reads; gcc -O3 folds these to single loads */
static inline uint64_t rd8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t rd4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

/* one fold pressed against one beak: lane value folded into that bird */
static inline uint64_t beak(uint64_t acc, uint64_t lane) {
    acc += lane * P2;
    acc  = rotl64(acc, 31);
    acc *= P1;
    return acc;
}

/* "only when all four agree": a bird's reading folded into the agreement,
   deliberately scrambled so the birds cannot be interchanged */
static inline uint64_t agree(uint64_t h, uint64_t bird) {
    uint64_t v = beak(0, bird);
    h ^= v;
    h  = h * P1 + P4;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p   = data;
    const unsigned char * restrict end = data + len;
    uint64_t h;

    if (len >= 32) {
        /* the sheet is large enough: lay out all four birds */
        const unsigned char *limit = end - 32;
        uint64_t b1 = P1 + P2;
        uint64_t b2 = P2;
        uint64_t b3 = 0;
        uint64_t b4 = (uint64_t)0 - P1;

        /* four beaks at once: four independent chains, 32 bytes per pass */
        do {
            b1 = beak(b1, rd8(p +  0));
            b2 = beak(b2, rd8(p +  8));
            b3 = beak(b3, rd8(p + 16));
            b4 = beak(b4, rd8(p + 24));
            p += 32;
        } while (p <= limit);

        /* the birds disagree on purpose: distinct rotations, no symmetry */
        h = rotl64(b1, 1) + rotl64(b2, 7) + rotl64(b3, 12) + rotl64(b4, 18);
        h = agree(h, b1);
        h = agree(h, b2);
        h = agree(h, b3);
        h = agree(h, b4);
    } else {
        /* too few marks for four gauges: one sheet, one gauge */
        h = P5;
    }

    /* the one small suitcase carried under with the body */
    h += (uint64_t)len;

    /* scraps trimmed off the sheet */
    while (p + 8 <= end) {
        h ^= beak(0, rd8(p));
        h  = rotl64(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)rd4(p) * P1;
        h  = rotl64(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p++) * P5;
        h  = rotl64(h, 11) * P1;
    }

    /* the water's edge: small, small, small -> one coin. Three presses, then stop. */
    h ^= h >> 33; h *= P2;
    h ^= h >> 29; h *= P3;
    h ^= h >> 32;
    return h;
}
```

## PREDICTION

Stated before any measurement (no benchmark tool is reachable in this session, so this is a blind prediction and I will not retroactively adjust it):

**PREDICTION: speedup_vs_dp = 6.0**

Reasoning I commit to: FNV-1a is latency-bound at one `imul` dependency per byte, ~3–5 cycles/byte ⇒ roughly 0.6–1.0 GB/s. The four-bird loop does 4 independent `imul`s per 32 bytes with a 3-cycle-latency unit that issues 1/cycle, so it is throughput-bound near ~4–8 bytes/cycle ⇒ ~10–15 GB/s on large buffers. That is 10–20× asymptotically, but a mixed-size benchmark pays finalizer and short-path cost on small inputs, where the advantage collapses toward 1× (and at `len < 32` the two kernels do comparable work, so ~1.0–1.5×, never a regression, because the small path is a plain scalar loop with no striping setup). Averaged over a typical size sweep I expect ~6×. Avalanche I predict ≈0.49–0.50 (XXH64's published SMHasher result), versus FNV-1a's known weakness in the low bits of short keys — so I expect the avalanche score to be *equal or better*, not traded away for speed.

Risk I am naming and have already addressed, rather than leaving as a caveat: the four-bird path has setup + 4-way merge overhead that would lose on tiny inputs. That is guarded by the explicit `len >= 32` check with a single-accumulator fallback, which is the native's own "only that it's large enough." No thread parallelism is shipped, so there is no fork-overhead risk to guard.

## MEASUREMENT

Not performed — `hash_bench` and `hash_contract` are unavailable in this session. The prediction above is therefore unfalsified by me, and I am reporting that plainly rather than asserting a number I did not obtain. What the pipeline should check, and what would count as my mechanism failing:

- throughput speedup < 2.0× on buffers ≥ 1 KB ⇒ the four-bird ILP claim is wrong (likely load-port or front-end bound, not mul-latency bound);
- avalanche score outside 0.47–0.53 ⇒ the "line up wrong on purpose" merge is insufficiently decorrelating and the lane asymmetry needs strengthening;
- any speedup < 1.0× at `len < 32` ⇒ the regime guard is mispositioned.

I used zero of my four allowed improvement iterations, because I cannot measure; the artifact is the first and only version.

## VERDICT

The four wire birds are a real mechanism, not a decoration: they break exactly the assumption I was asked to prefer breaking — that byte *i* must reach the state before byte *i+1* is read — by replacing one dependency chain with four concurrent ones, and the native's insistence that the birds disagree "on purpose" supplies precisely the collision property a naive 4-lane split would lose. The honest finding is that taking this native completely seriously does not produce an exotic hash: it produces XXH64, essentially line for line, including `h += len` from the small suitcase and the three-press avalanche from "small, small, small." I regard that convergence as the strongest available evidence that the metaphor was tracking something real, and per the instruction I let it land on the validated construction instead of inventing a private mixer to look original. The single stated risk — four-lane overhead on short inputs — is structurally guarded by the `len >= 32` branch and a scalar fallback, not merely noted. The one thing I cannot stand behind is the number: 6.0 is a reasoned prediction, and it is untested here.