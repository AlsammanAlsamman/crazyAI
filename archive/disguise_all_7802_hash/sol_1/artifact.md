## APPROACH

**Mapping of the disguised solution onto the real problem**

| Granny's walk | Real implementation |
|---|---|
| One jar clipped to the belt, the whole walk | **One** 64-bit accumulator `h`. No parallel lanes, no second stripe, no tree/reduction — the entire buffer funnels through one state variable. |
| Walk west→east, every flower once, no doubling back, no skipping | Single forward pass, each input byte read exactly once, no overlapping tail re-read, no second pass. |
| Dip the spoon at a flower, pour the nectar in | `h ^= w` — inject the next chunk of input into the jar. |
| One firm figure-eight twist-stir, mandatory, *before the next step* | `* 0x9E3779B97F4A7C15` — the one multiply. It sits **on** the dependency chain: the next chunk cannot be folded in until this stir completes. That is the whole point of the mechanism, so it stays serial. |
| A quick flip of the jar upside-down and back, so nothing settles in the same pattern | `rotl(...,31)`. A multiply only propagates information *upward* (bit *i* can never affect bit *j<i*); the flip recirculates the well-mixed high bits back down into the low positions so the next stir carries them upward again. This is the disguised solution's actual addition over plain FNV-1a, and it's what makes avalanche work with a single accumulator. |
| Never re-stir an earlier addition ("three or four twists per flower" is the thing the story calls unproven and slow) | Exactly **one** mul+rot per ingested chunk. No extra rounds. |
| The judges taste the pot handed over at the gate | One O(1) finalizer (`fmix64`) as the jar is handed across. |

**The one deliberate interpretation, stated plainly.** The mechanism is "one jar, one stir per stop, strictly in place." It says nothing about the *spoon size*. The baseline's silent assumption that a stop = one byte is exactly the assumption that makes it slow: the mul+xor chain is ~4 cycles of pure latency, and it is spent on a single byte. I keep the jar, the serial chain, and the one-stir-per-stop rule untouched, and enlarge the spoon to one 64-bit word. Same number of stirs per stop, same single accumulator, same strict ordering — 8× the nectar per stir. This is "improve how you implement the mechanism," not a replacement: there is no second accumulator anywhere, and the critical path is still one dependent mul per stop.

**Why this must be faster.** Both baseline and this kernel are latency-bound on one chain; nothing else matters. Baseline: `xor(1) + imul(3)` = **4 cycles/byte**. This kernel: `xor(1) + imul(3) + rol(1)` = 5 cycles per 8 bytes = **0.625 cycles/byte**. Loads and loop overhead are independent of the chain and hide completely under it, so the unrolling is cosmetic. Theoretical ratio 6.4×.

**Why avalanche holds.** Every word passes through ≥1 mul+rot, and the rotate guarantees downward propagation that the multiply alone can never give. The weak case is the *final* word — flipping its bit 63 changes only bit 63 of the product — so the gate finalizer (`fmix64`, a bijection with measured full avalanche) is not optional; it converts any nonzero state difference, single-bit or not, into ~32 flipped output bits. `len` is mixed at both ends so zero-extension and length ambiguity can't collide.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* One jar. One stir per stop. West to east, once. */

#define JAR_M 0x9E3779B97F4A7C15ULL   /* the twist  */
#define JAR_R 31                       /* the flip   */

static inline uint64_t jar_rotl(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t jar_load64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);                 /* one mov; handles unaligned data */
    return v;
}

/* pour -> twist -> flip.  Exactly one of each, every single time. */
static inline uint64_t jar_stir(uint64_t h, uint64_t w) {
    return jar_rotl((h ^ w) * JAR_M, JAR_R);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    uint64_t h = 0x2545F4914F6CDD1DULL ^ (len * JAR_M);
    const unsigned char *p = data;
    size_t n = len >> 3;              /* number of full stops */
    size_t i = 0;

    /* The walk. The four stirs below are strictly sequential on h;
       only the loads and the counter run ahead. */
    for (; i + 4 <= n; i += 4) {
        h = jar_stir(h, jar_load64(p)); p += 8;
        h = jar_stir(h, jar_load64(p)); p += 8;
        h = jar_stir(h, jar_load64(p)); p += 8;
        h = jar_stir(h, jar_load64(p)); p += 8;
    }
    for (; i < n; i++) {
        h = jar_stir(h, jar_load64(p)); p += 8;
    }

    /* Last, partial flower: read only the bytes that exist, one stir. */
    size_t rem = len & 7;
    if (rem) {
        uint64_t t = 0;
        for (size_t k = 0; k < rem; k++)
            t |= (uint64_t)p[k] << (8 * k);
        h = jar_stir(h, t ^ ((uint64_t)rem << 56));   /* top byte is free */
    }

    /* At the gate: one last shake before the judges taste it. */
    h ^= (uint64_t)len;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}
```

## PREDICTION

PREDICTION: 6.0

(Speedup factor over the FNV-1a baseline on a large buffer. Derivation: 4.0 cycles/byte ÷ 0.625 cycles/byte = 6.4 theoretical; I shade to 6.0 for loop-boundary and tail/finalizer overhead. Absolute throughput expected ≈ 5–7 GB/s single-core at 3.5–4.5 GHz, versus ≈ 0.9–1.1 GB/s for the baseline. On short inputs (≤64 B) the constant-cost finalizer eats into this and I expect only ≈ 4×. I expect avalanche ≈ 0.50 ± 0.02 bit-flip fraction and to pass the contract check.)

## MEASUREMENT

**Not measured — and I am reporting that as a failure of this run, not glossing it.** This session was invoked with no tools available: `hash_bench` and `hash_contract` could not be called, so the "measure, then improve up to four times" loop never executed. The number above is a static-analysis prediction from instruction latencies, not an observation, and nothing below it has been validated on hardware.

What I would have checked, in order, and the specific risks that measurement would settle:

1. **Contract + avalanche first** (`hash_contract`). Risk: the final-word case. If the bench measures avalanche per-bit-position rather than averaged, and the finalizer under-covers the very last input word, the fix is a second `jar_stir` of the length at the gate before `fmix64` — cost ~5 cycles, zero effect on throughput.
2. **Large-buffer throughput** (`hash_bench`). If measured cycles/byte come in near 0.625, the kernel is at its structural floor and no further tuning of *this* mechanism helps — the chain is `xor→imul→rol` and none of the three can be removed without abandoning either the pour, the stir, or the flip.
3. **If it lands above ~0.8 cycles/byte**, the suspect is the compiler spilling or failing to hoist loads above the chain; remedy is `__builtin_prefetch` off-chain plus hand-scheduled loads into locals before the stir sequence — again without touching the mechanism.
4. **Small-input sweep** (8/16/32/64 B), where the 3-multiply finalizer is the dominant cost and the predicted 6× will not hold.

## VERDICT

The mechanism translates cleanly and, on paper, decisively beats the baseline: the story's insistence on *one jar, one stir, no backtracking* is precisely a serial dependency chain, and the baseline's only real sin is spending that chain on one byte at a time instead of eight. Keeping the jar and widening the spoon buys ~6× at zero cost to the structure. The story's added "flip" is not decoration — without the rotate, a single accumulator has no downward bit propagation at all and the hash would fail avalanche outright, so the disguised solution's one novel element is load-bearing.

Two honest caveats. First, **unmeasured**: the 6.0 is predicted, not observed. Second, the "one jar" constraint the story insists on is a genuine quality ceiling, and the grumbling beekeepers are right on the merits: a 64-bit state fed 8 bytes per round has no reserve, which is exactly why xxHash64 carries four accumulators (256 bits) and folds only at the end. This kernel is fast and avalanches well, but it is a *non-cryptographic* hash whose collision resistance is bounded by its 64-bit state; I would not claim it matches xxHash64's collision quality, and building it this way is a constraint I was handed, not one I would choose.