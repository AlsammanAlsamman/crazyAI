## MAPPING

### SEED 1 — the servant, the inflated house, the flat stone

| World object | Problem object |
|---|---|
| pile of marks "in their given order" | the input bytes `data[0..len-1]` |
| servant swatting the mouth to **hold silence** | *no state update is allowed while reading* — reading is decoupled from mixing |
| the flat stone at the riverbank | a fixed-size staging word/tile (a local `uint64_t`, zero-filled) that bytes are `memcpy`'d into |
| "the order never gets spoken aloud and lost" | order is carried **positionally** (byte offset → lane index → lane-specific constant), not by the sequence of state updates |
| laying all marks down *before* any heap is folded | load a whole block of bytes first, then mix |
| **Assumption broken** | "each byte must be mixed into the running state before the next byte is read" — and, as a consequence, "the whole buffer must be read once, start to end, in order": if position encodes order, traversal order is free. |

### SEED 2 — the musk deer's stride, uphill twist, downhill fold

| World object | Problem object |
|---|---|
| "heaps of **even count**" | the state is an even number of parallel lanes (8 × `uint64_t`) |
| "gathers ground beneath it in **fixed strides**" | fixed block stride: 64 bytes per heap, 128 bytes per main-loop step |
| uphill twist | `ROTL64` |
| downhill fold | `x ^= x >> k` / `b = rotl(b,r) ^ a` |
| "a doubling-back that eats its own trail" | ARX pair step `a += b; b = rotl(b,r1)^a; b += a; a = rotl(a,r2)^b` — the value re-consumes its own just-changed partner |
| "no heap keeps the shape it entered with" | every lane step is **invertible**, so no difference can die |
| "nothing stays still except the final stone at the shrine, where the running shape gets carried forward" | lane state persists across heaps (chaining), only the lanes survive |
| "a calculation grown too satisfied with itself is a pillar that brings the roof down" | **fixed points**: a pure xor/rot mix of all-zero (or all-equal) lanes collapses. Cured by adding a per-lane odd constant XOR a block counter |
| **Assumption broken** | "mixing one byte requires a multiplication" (ARX only on the per-byte path: zero multiplies per byte) and "the state is a single accumulator updated in place" |

### SEED 3 — the owl, the dreamer inside, the burning at the shrine

| World object | Problem object |
|---|---|
| the owl takes the pen, "calls the shape inside the dreamer inside" | a finalizer applied once at the end: nested invertible mix (`fmix64` — xor-shift, multiply, xor-shift, multiply, xor-shift) |
| "fixes it, small and unchanging, the size of any other token regardless of how many marks came before" | output is exactly 64 bits for any `len`; `len` is mixed in |
| intermediate heaps "burned the moment the next heap swallows them" | no scratch buffer kept; 8 registers of state, O(1) memory, streaming |
| "a token that can be walked backward to its marks is no token at all" | the 8→1 lane combine is deliberately lossy — the only non-invertible step in the whole design |
| **Assumption broken** | "more mixing rounds always means better mixing" — the owl acts **once**; the work is placed where it buys avalanche (end), not repeated per byte. |

## CHOSEN SEED

**SEED 2**, the musk deer's stride — with SEED 1's flat stone as its loader and SEED 3's owl as its finalizer (the native describes one working, not three).

Why: it is the most mechanically literal (every clause names an operation — stride, rotate, fold, carry-forward, fixed-point warning) and the most different from the known way, since it deletes the multiply from the per-byte path entirely and replaces the single accumulator with eight chained lanes plus the "search in squares" cross-lane fold.

Honest note on the preference in step 2: **SEED 2 does not break "read once, start to end, in order"** — the native explicitly keeps a carry chain ("the running shape gets carried forward into the next heap's racing"), so the buffer is still read once, forward. The seed that breaks that assumption is **SEED 1** (order fixed by position on the stone, so traversal order stops mattering). I did not choose SEED 1 as the primary because its mapping is thin on mechanism; I instead let SEED 1's positional ordering become the *loader* of SEED 2's kernel, which is where it does real work: bytes are absorbed into lane `i` chosen by offset, not by arrival into one accumulator.

## ASSUMPTION BROKEN

Primary: **"mixing one byte requires a multiplication"** — the per-byte path contains zero multiplications; mixing is add–rotate–xor only. This is not an invention: ARX round functions are the validated real-world answer here (SipHash, ChaCha, BLAKE2 all mix without a multiply), and the pair step the metaphor describes *is* an ARX quarter-round. Per step 4 I let the mechanism land on that known construction rather than a novel one.

Secondary: **"the state is a single accumulator updated in place"** — eight lanes, hypercube-folded. And **"more mixing rounds always means better mixing"** — one fold stage per heap during streaming, with full diffusion paid once at the end.

Multiplication survives in exactly one place: the owl's `fmix64`, two multiplies for the entire buffer, O(1) in `len`.

**Regime handling (step 5).** The known_way section spans two regimes (short keys vs. large buffers). The metaphor itself encodes the test: the deer's carry-forward only means anything once there is a *second* heap to carry into, so if the pile cannot make a pair of heaps (`len < 128`), the deer takes no stride at all — the marks go from the flat stone straight to the owl, through one accumulator. That is the small-input fallback, and it is also the guard for the risk my own verdict names (the 8-lane setup plus the end-of-run square folds are a fixed cost that would lose to FNV on short keys). No OpenMP: the metaphor's unit of work is a 64-byte heap carried forward serially, which is far below a thread's worth of work at any plausible benchmark size, and a threaded tree-combine would be a mechanism whose risk I cannot bound here — so it is dropped rather than shipped unguarded. Vectorization is left to `-march=native` on the 8-lane loops (uniform array form, variable per-lane rotates map to `vprolvq`/`vpsllvq`).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* "the owl takes the pen and calls the final shape inside the dreamer inside --
   fixes it, small and unchanging, the size of any other token."
   MurmurHash3 fmix64: a validated finalizer, O(1) for the whole buffer.
   These are the only two multiplies in the design. */
static inline uint64_t owl_fix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

/* the flat stone: one distinct odd mark per lane, so lanes can never
   become equal to each other and the all-zero state cannot persist
   ("a calculation grown too satisfied with itself"). */
static const uint64_t STONE[8] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
    0x94D049BB133111EBULL, 0x2545F4914F6CDD1DULL,
    0xD1B54A32D192ED03ULL, 0xA5CB3B6F2E1D4C87ULL,
    0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL
};
/* the uphill twist: a different climb for every lane */
static const unsigned TWIST[8] = { 7u, 11u, 17u, 23u, 29u, 37u, 43u, 53u };

/* one heap = 64 bytes laid on the stone, order fixed by position (lane i
   <- bytes 8i..8i+7), then raced: twist, then a stride-counter mark added
   so a heap of zeros still changes the running shape. Invertible in s. */
static inline void race(uint64_t *s, const unsigned char *p, uint64_t ctr) {
    int i;
    for (i = 0; i < 8; i++) {
        uint64_t w;
        memcpy(&w, p + (size_t)8 * i, 8);
        s[i] = ROTL64(s[i] ^ w, TWIST[i]) + (STONE[i] ^ ctr);
    }
}

/* "turning the shape ninety degrees and folding its corners into its own
   center" -- a hypercube butterfly over the 8 lanes. d = 1,2,4 are the
   three turns; three turns reach every lane from every lane. Each pair
   step is an ARX quarter-round: invertible, so nothing can collapse. */
static inline void fold_square(uint64_t *s, int d, unsigned r1, unsigned r2) {
    int i;
    for (i = 0; i < 8; i++) {
        uint64_t a, b;
        if (i & d) continue;
        a = s[i];
        b = s[i | d];
        a += b;  b = ROTL64(b, r1) ^ a;
        b += a;  a = ROTL64(a, r2) ^ b;
        s[i] = a;
        s[i | d] = b;
    }
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    const unsigned char *end = data + len;
    uint64_t s[8];
    uint64_t h, ctr;
    int i;

    /* REGIME TEST, in the metaphor's own terms: the carry-forward only
       means anything once there is a second heap to carry into. Too few
       marks for a pair of heaps -> the deer takes no stride; the marks go
       from the flat stone straight to the owl, through one accumulator.
       This is the small-input fallback that pays no 8-lane setup and no
       end-of-run square folds. */
    if (len < 128) {
        h = STONE[0] ^ ((uint64_t)len * 0xBF58476D1CE4E5B9ULL);
        while (p + 8 <= end) {
            uint64_t w;
            memcpy(&w, p, 8);
            h = ROTL64(h ^ w, 31) + 0x9E3779B97F4A7C15ULL;
            h ^= h >> 27;
            p += 8;
        }
        if (p < end) {                 /* the stone, zero-padded: position
                                          still encodes order, no byte loop */
            uint64_t w = 0;
            memcpy(&w, p, (size_t)(end - p));
            h = ROTL64(h ^ w, 31) + 0x9E3779B97F4A7C15ULL;
        }
        return owl_fix(h);
    }

    /* --- the deer's run: heaps of even count, fixed stride --- */
    for (i = 0; i < 8; i++) s[i] = STONE[i] ^ (uint64_t)len;
    ctr = 1;

    /* two heaps per step, each followed by a different ninety-degree turn,
       so the stride schedule needs no branch */
    while ((size_t)(end - p) >= 128) {
        race(s, p, ctr);
        fold_square(s, 1, 13u, 29u);
        race(s, p + 64, ctr + 1);
        fold_square(s, 2, 23u, 47u);
        p += 128;
        ctr += 2;
    }
    if ((size_t)(end - p) >= 64) {
        race(s, p, ctr);
        fold_square(s, 1, 13u, 29u);
        fold_square(s, 4, 19u, 41u);
        p += 64;
        ctr++;
    }
    /* the last partial heap: whole words to their own lanes, then the stone */
    i = 0;
    while (p + 8 <= end) {
        uint64_t w;
        memcpy(&w, p, 8);
        s[i] = ROTL64(s[i] ^ w, TWIST[i]) + (STONE[i] ^ ctr);
        p += 8; i++; ctr++;
    }
    if (p < end) {
        uint64_t w = 0;
        memcpy(&w, p, (size_t)(end - p));
        s[i] = ROTL64(s[i] ^ w, TWIST[i]) + (STONE[i] ^ ctr);
    }

    /* "searching in squares ... again and again, until even a single mark
       changed at the start has smeared itself across every square" --
       two full passes of the three turns: >= 6 ARX stages after the last
       byte is absorbed, every lane reachable from every lane twice over. */
    fold_square(s, 1, 13u, 29u);
    fold_square(s, 2, 23u, 47u);
    fold_square(s, 4, 19u, 41u);
    fold_square(s, 1, 41u, 19u);
    fold_square(s, 2, 47u, 23u);
    fold_square(s, 4, 29u, 13u);

    /* the heaps are burned: 8 lanes collapse to one token. This is the
       only non-invertible step in the design -- deliberately so. */
    h = (uint64_t)len ^ 0x6A09E667F3BCC909ULL;
    for (i = 0; i < 8; i++) h = ROTL64(h ^ s[i], 29) + s[i];
    return owl_fix(h);
}
```

Design corrections made *before* any measurement (not measurements, just caught by the metaphor's own warnings):
1. The first butterfly I wrote was `t_i = s_i ^ rotl(s_{i^d}, r)`. With equal lanes that collapses toward zero — the native's "pillar that will bring the roof down." Replaced with an invertible ARX quarter-round.
2. A pure rot/xor lane step has a fixed point at zero (a buffer of zero bytes would not change the state). Added `+ (STONE[i] ^ ctr)`, the per-lane odd mark and stride counter.
3. Lanes initially shared one rotation amount, so they were permutation-symmetric; distinct `TWIST[i]` breaks that.
4. Regime threshold started at 64 bytes (one heap). Hand-counting the fixed cost (eight lane inits plus six end folds ≈ 100+ ops) against the per-block saving put break-even near two heaps, so the threshold is 128 — which is also the metaphor's own condition for carry-forward to mean anything.

## PREDICTION

Stated before any measurement, from op counts and dependency-chain length only.

- Baseline `kernel` (FNV-1a) is latency-bound: `imul` (3 cycles) + `xor` per byte ≈ 4 cycles/byte ≈ 0.8 GB/s at 3 GHz.
- Mine on the striped path: ≈ 56 ops per 64 bytes ≈ 0.9 ops/byte, across 8 independent lanes, so ~4 ops/cycle retire → ≈ 0.25 cycles/byte ≈ 10–12 GB/s, falling back to memory bandwidth for buffers above L2.
- On the short path (< 128 B) the gain is far smaller — ~2–4× — because both paths are then dominated by a short serial chain plus the finalizer.
- Avalanche: I expect ≈ 0.50 (score near 1.0 on a normalized scale). The argument is structural, not hopeful: every state step up to the final 8→1 collapse is invertible, so a one-bit input difference **cannot** vanish; it is still present as a nonzero 8-lane difference, the six closing fold stages carry it into all eight lanes, and `fmix64` is a measured-good 64-bit avalanche. The weakest case would be a flip in the very last byte, which still gets ≥6 ARX stages plus the owl.

PREDICTION: speedup_vs_dp = 7.0

## MEASUREMENT

**Not measured — and I will not dress that up.** This session was invoked with no tools: `hash_bench`, `hash_contract` and the symbolic/unconventional tools are all unavailable, so there is no throughput number and no avalanche score from me. The prediction above is therefore an unvalidated forecast, made from op counts and dependency-chain arithmetic, and it is written to be falsifiable by the pipeline that compiles this:

- If `speedup_vs_dp` comes back below ~4, my ILP assumption for the 8-lane race is wrong (likely the per-lane variable rotate blocked vectorization, or the 128-byte loop is load-port-bound) — the fix would be to cut the per-heap fold to one stride and widen to 16-byte-per-lane loads, not to add rounds.
- If the avalanche score comes back materially below ~0.5, my invertibility argument has a hole — the first place to look is the final 8→1 collapse (`h = rotl(h ^ s[i], 29) + s[i]`), where a single-lane difference could partially cancel across the xor and the add. The fix is to `owl_fix` two independent 4-lane halves and combine them, not to add fold stages.
- The one thing I would *not* accept as a reason to change the design is the per-byte multiply coming back. That is the assumption under test.

## VERDICT

The native's method translates into a real, named thing: a multiply-free ARX stream hash with eight chained lanes, a three-turn hypercube butterfly for cross-lane diffusion, and a single `fmix64` at the end. It is not FNV and it is not xxHash — it has no multiply in the per-byte path at all, which is exactly the assumption the musk-deer seed breaks, and the construction it lands on (ARX quarter-rounds, hypercube folding, one-shot finalizer) is the validated family rather than something I invented for novelty.

Where it is honestly weaker than the known way: on short keys the eight-lane setup and the six closing folds are pure overhead, and FNV-1a would win. I did not leave that as a caveat — it is guarded in the code by `len < 128` with a single-accumulator fallback, and the threshold comes from the metaphor's own condition (no carry-forward without a second heap) rather than being bolted on. Thread parallelism is deliberately absent: the metaphor's unit of work is a 64-byte heap carried forward serially, which is nowhere near a thread's worth of work, and I will not ship a mechanism whose risk I cannot bound.

The claim that remains open is purely quantitative — whether the eight-lane ILP actually buys the ~7× I predicted. The avalanche claim rests on an argument I am willing to be held to: every step before the final collapse is invertible, so no input difference can die before the owl sees it. If the bench contradicts either, the fixes are the two named above, and neither of them is "put the multiply back."