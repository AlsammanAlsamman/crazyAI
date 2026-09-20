No tools were invoked for this task — the toolset available in this session (Docs/Gmail/Drive/Slack) has no bash/compiler/`hash_bench` access, and the system note confirms no tools are available here; the external pipeline will compile and measure the artifact below. I did not silently swap in a stock recipe without deriving it from the story — the derivation below is what *produces* the xxHash64-shaped kernel.

## MAPPING

**SEED 1** — "The servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded."

| World object | Problem object |
|---|---|
| Marks | Input bytes `data[0..len)` |
| Pile of marks, given order | The byte sequence and its positions |
| Servant + inflated house on a stick | A position-tagging mechanism that runs alongside processing |
| Swat against the mouth ("hold silence") | Withhold folding a byte/block into an accumulator until its position identity is fixed |
| Flat stone at the riverbank | A small set of fixed memory slots (lane accumulators) that each byte is eventually laid into |
| "order itself never gets spoken aloud and lost" | Order is recorded structurally (per-lane seed/index) instead of only existing implicitly as "whichever byte the single accumulator last touched" |

Breaks: **"the whole buffer must be read once, start to end, in order [through one accumulator]"** and, as a consequence, **"the state is a single accumulator updated in place, one value."** If order is fixed by tagging (lane identity) rather than by strict sequential threading, blocks can be folded through *independent* accumulators without losing the ability to reconstruct a well-defined, order-sensitive result at the end.

**SEED 2** — "Each heap races the musk deer's mountain stride… until no heap keeps the shape it entered with."

| World object | Problem object |
|---|---|
| Heap of even count | A fixed-size block of bytes (one 8-byte word per lane) |
| Musk deer's fixed stride | A fixed, repeatable per-block operation count (O(1) work per block) |
| Uphill twist | Bit rotation (`rotl`) |
| Downhill fold | Multiply-add combine of block into accumulator |
| Doubling back that eats its own trail | Self-referential multiply (`acc *= PRIME`) so output doesn't resemble input bytes |

Breaks: **"mixing one byte requires a multiplication"** only weakly — it still ends in a multiply, it just adds rotate+fold around it. This is structurally close to what xxHash/Murmur already do per block, so it is the *least novel* of the three relative to the known way.

**SEED 3** — "The owl calls the final folded shape inside the dreamer inside… every intermediate heap is burned at the shrine so the token cannot be walked back to its marks."

| World object | Problem object |
|---|---|
| Owl | Finalization function |
| Final folded shape | The combined accumulator after all blocks are folded |
| "inside the dreamer inside" | A nested (two-stage) avalanche mix applied to the finalized value |
| Sealed to one fixed size | Output is always 64 bits regardless of `len` |
| Burning intermediate heaps | Partial block states are never retained — only the running accumulator(s) persist |

This doesn't break any of the five listed assumptions — it describes exactly what the given baseline already does (one accumulator, non-invertible, fixed 64-bit output). Most literal, but not different from the known way.

## CHOSEN SEED

**SEED 1.** It's the only one of the three that breaks the target assumption ("read once, start to end, in order" through one accumulator), and its mapping (lane identity fixed by index, not by sequential position in a single accumulator) is direct and literal.

## ASSUMPTION BROKEN

"The whole buffer must be read once, start to end, in order" through **a single accumulator**. Replaced with: order is fixed structurally via per-lane seeding, so several accumulators can fold disjoint stripes of the buffer with independent (non-serially-dependent) update chains, then be recombined at the end.

Per the step-4 instruction: a well-known, validated technique already implements exactly this — **xxHash64's four-lane accumulator scheme** (independent lanes seeded from `PRIME1+PRIME2`, `PRIME2`, `0`, `-PRIME1`, each folding a distinct 8-byte stripe per 32-byte block, recombined with rotate-and-add). I let the mechanism arrive there rather than inventing a fifth lane count or a novel combiner. The finalizer is MurmurHash3's `fmix64`, matching SEED 3's "nested avalanche" literally and reusing another validated piece rather than a new one.

Vectorization-before-threads: the four lanes give **instruction-level parallelism** (breaking one long multiply-latency-bound dependency chain into four independent ones the CPU can overlap) without needing explicit SIMD intrinsics or OpenMP threads — the "heaps" here (8–32 bytes) are far too small per unit of work to justify thread spawn overhead, so no `#pragma omp` is used.

Risk guard: the four-lane setup only pays off once there's at least one full 32-byte block. Below that, per the story itself ("heaps of even count" — there's nothing to fix order *across* if there's only one heap), the kernel takes an explicit `len < 32` fallback branch to a plain single-accumulator FNV-1a-style loop with a light finalizer, so tiny inputs never pay lane-setup overhead for no benefit.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t read_le64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

#define PRIME1 11400714785074694791ULL
#define PRIME2 14029467366897019727ULL
#define PRIME3 1609587929392839161ULL
#define PRIME4 9650029242287828579ULL
#define PRIME5 2870177450012600261ULL

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char * const end = data + len;
    uint64_t h;

    /* Fallback: too small for the servant to have anything to fix order
       across (fewer than one full heap-of-four) -- plain single
       accumulator, walked mark by mark. */
    if (len < 32) {
        h = PRIME5 ^ (uint64_t)len;
        while (p < end) {
            h ^= (uint64_t)(*p++);
            h *= PRIME1;
            h ^= h >> 33;
        }
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }

    /* Fix each heap's identity (order) before any folding happens: four
       lanes, each seeded distinctly, so their position is recorded up
       front instead of depending on strict single-accumulator sequencing. */
    uint64_t acc0 = PRIME1 + PRIME2;
    uint64_t acc1 = PRIME2;
    uint64_t acc2 = 0;
    uint64_t acc3 = (uint64_t)0 - PRIME1;

    /* Each heap races its own mountain stride -- uphill twist (rotl),
       downhill fold (multiply-add), doubling back (self-multiply) -- with
       no data dependency between lanes, so the four chains overlap on the
       CPU instead of forming one long serial chain. */
    size_t nblocks = len / 32;
    for (size_t b = 0; b < nblocks; b++) {
        acc0 += read_le64(p +  0) * PRIME2; acc0 = rotl64(acc0, 31); acc0 *= PRIME1;
        acc1 += read_le64(p +  8) * PRIME2; acc1 = rotl64(acc1, 31); acc1 *= PRIME1;
        acc2 += read_le64(p + 16) * PRIME2; acc2 = rotl64(acc2, 31); acc2 *= PRIME1;
        acc3 += read_le64(p + 24) * PRIME2; acc3 = rotl64(acc3, 31); acc3 *= PRIME1;
        p += 32;
    }

    /* The four running heaps are folded onto the one final stone. */
    h = rotl64(acc0, 1) + rotl64(acc1, 7) + rotl64(acc2, 12) + rotl64(acc3, 18);
    h ^= h >> 29; h *= PRIME3;
    h ^= (uint64_t)len;

    /* Remaining tail marks, walked one at a time, in order, into the now
       single running shape. */
    while (p + 8 <= end) {
        uint64_t k = read_le64(p) * PRIME2;
        k = rotl64(k, 31); k *= PRIME1;
        h ^= k;
        h = rotl64(h, 27) * PRIME1 + PRIME4;
        p += 8;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * PRIME5;
        h = rotl64(h, 11) * PRIME1;
        p++;
    }

    /* The owl calls the final shape inside the dreamer inside: a nested
       avalanche finalizer sealing the token to one fixed 64-bit size.
       No intermediate heap (acc0..3, tail words) survives past here. */
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return h;
}
```

PREDICTION: speedup_vs_dp = 4.0

(Reasoning for that number: the baseline's serial `h ^= b; h *= PRIME` chain is latency-bound on the multiply — one dependent multiply per byte. The four-lane version keeps the same multiply-per-8-bytes-per-lane cost, but the four chains are independent, so a superscalar core can overlap them; xxHash64-family kernels are reported at roughly 3–6x FNV-1a scalar throughput on large buffers in published benchmarks, so 4x is a middle-of-the-road estimate. I expect this to hold only once buffers clear the 32-byte lane threshold; near/under that size I expect roughly parity with the baseline (~1x), since the fallback path does slightly *more* work per byte than raw FNV-1a — a light xor-shift after every byte — to keep avalanche reasonable at tiny sizes.)

## MEASUREMENT

Not run live in this session — the tool surface available to me here (Docs/Gmail/Drive/Slack) has no `hash_bench`/compiler access, matching the session note that no tools are available; the pipeline runs it after this response. Reporting that plainly rather than fabricating numbers. Expected outcome to compare against the prediction once measured:

- **Throughput**: ≥3x baseline for buffers well above 32 bytes; close to 1x for buffers under 32 bytes (fallback path dominates, and it does a bit more work per byte than the reference FNV-1a).
- **Avalanche**: should be strong (close to the ideal ~50% output-bit-flip rate for a single input-bit flip) on both paths, since the block round (multiply-rotate-multiply) and the MurmurHash3 `fmix64` finalizer are both independently validated (SMHasher) to diffuse well — the "corner survives untouched" failure mode the story warns about is exactly what `fmix64`'s three xor-shift/multiply rounds are designed to eliminate.
- If measurement instead shows the small-buffer path noticeably *slower* than the given baseline, the fix (my first of the four allowed improvement passes, held in reserve) would be to strip the extra `h ^= h >> 33` inside the fallback's per-byte loop and only avalanche once at the end, trading a little diffusion margin for parity with the baseline below 32 bytes.

## VERDICT

The mechanism is literally derived: lane index = fixed order ("the stone the servant makes you lay each mark on before it can be folded"), block round = the deer's stride, four-lane recombination + `fmix64` = the owl's nested finalization, no retained block state = the burned heaps. It resolves to xxHash64's known, validated four-accumulator construction rather than a novel one, per the step-4 instruction, and uses ILP-friendly independent scalar chains (vectorization-adjacent) instead of OpenMP threads, since 8–32-byte "heaps" are far too small a unit of work to amortize thread spawn.

Stated risk and its guard: the four-lane machinery only helps once the buffer has at least one full 32-byte block; below that it's pure overhead for no diffusion benefit. This is addressed, not left as a caveat — the kernel has an explicit `len < 32` branch that skips lane setup entirely and falls back to a single-accumulator loop. No unaddressed risk is shipped.