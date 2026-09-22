## APPROACH

The three disguised solutions map onto three real hashing strategies:

- **Solution 1** (walk the shelf in strict order, one bowl) = the given FNV‑1a baseline: a single accumulator, one multiply per byte, a fully serial dependency chain (`h` at byte *i* can't be computed until `h` at byte *i‑1* is done). On modern CPUs a 64‑bit multiply has ~3–5 cycles of *latency* but the port can *issue* a new multiply every cycle — FNV wastes almost all of that issue throughput because each multiply must wait for the previous one to finish.
- **Solution 3** (two pointers, meet in the middle) is just a 2‑lane special case of the next idea.
- **Solution 2** (several helpers, several independent bowls, poured together and hard‑mixed at the end) is the general and strictly better version of Solution 3, and it is exactly the trick real fast hashes (xxHash, MurmurHash, CityHash) use: split the buffer into several **independent accumulator lanes**. Nobody is "waiting on anybody" — the 4 multiply chains are data‑independent, so the CPU's out‑of‑order scheduler can have several multiplies in flight at once, hiding each other's latency. Each lane still folds in every byte it's responsible for exactly once (order within/between lanes doesn't matter for avalanche, only that *all* bytes get mixed and the final pour‑together mix is strong), so flipping any one input byte still changes every lane's contribution and — after the final hard "twist‑stir" (multiplicative avalanche finalizer) — flips ~half the output bits.

I implement the well‑known xxHash64 construction: 4 independent 64‑bit lanes consuming 32 bytes/iteration (so 1 multiply per 4 bytes instead of FNV's 1 multiply per byte, spread across 4 independent chains for ILP), a lane‑merge step, an 8‑/4‑/1‑byte tail, and a 3‑round xor‑shift‑multiply avalanche finalizer for the "hard final twist‑stir." No OpenMP threads: real OS‑thread spawn overhead would only pay off on very large buffers and I have no measurements here to safely pick a threshold, so I keep the win entirely from ILP (ports/ROB do the "hiring of helpers" for free, with no thread overhead), which is the safe, always‑correct part of Solution 2's idea.

## ARTIFACT

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    static const uint64_t P1 = 11400714785074694791ULL;
    static const uint64_t P2 = 14029467366897019727ULL;
    static const uint64_t P3 = 1609587929392839161ULL;
    static const uint64_t P4 = 9650029242287828579ULL;
    static const uint64_t P5 = 2870177450012600261ULL;

    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        /* Four independent "helpers", each with their own bowl (accumulator).
           These chains are data-independent, so the CPU can run their
           multiplies concurrently instead of waiting on one serial chain. */
        uint64_t v1 = P1 + P2;
        uint64_t v2 = P2;
        uint64_t v3 = 0;
        uint64_t v4 = (uint64_t)0 - P1;

        do {
            uint64_t k1, k2, k3, k4;
            memcpy(&k1, p,      8);
            memcpy(&k2, p + 8,  8);
            memcpy(&k3, p + 16, 8);
            memcpy(&k4, p + 24, 8);
            p += 32;

            v1 += k1 * P2; v1 = rotl64(v1, 31); v1 *= P1;
            v2 += k2 * P2; v2 = rotl64(v2, 31); v2 *= P1;
            v3 += k3 * P2; v3 = rotl64(v3, 31); v3 *= P1;
            v4 += k4 * P2; v4 = rotl64(v4, 31); v4 *= P1;
        } while (p <= limit);

        /* Pour all the mini-swirls together and give it hard final stirs. */
        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= P2; v1 = rotl64(v1, 31); v1 *= P1; h64 ^= v1; h64 = h64 * P1 + P4;
        v2 *= P2; v2 = rotl64(v2, 31); v2 *= P1; h64 ^= v2; h64 = h64 * P1 + P4;
        v3 *= P2; v3 = rotl64(v3, 31); v3 *= P1; h64 ^= v3; h64 = h64 * P1 + P4;
        v4 *= P2; v4 = rotl64(v4, 31); v4 *= P1; h64 ^= v4; h64 = h64 * P1 + P4;
    } else {
        h64 = P5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1;
        memcpy(&k1, p, 8);
        k1 *= P2; k1 = rotl64(k1, 31); k1 *= P1;
        h64 ^= k1;
        h64 = rotl64(h64, 27) * P1 + P4;
        p += 8;
    }

    if (p + 4 <= end) {
        uint32_t k1;
        memcpy(&k1, p, 4);
        h64 ^= (uint64_t)k1 * P1;
        h64 = rotl64(h64, 23) * P2 + P3;
        p += 4;
    }

    while (p < end) {
        h64 ^= (uint64_t)(*p) * P5;
        h64 = rotl64(h64, 11) * P1;
        p++;
    }

    /* Final hard twist-stir: full avalanche mix. */
    h64 ^= h64 >> 33;
    h64 *= P2;
    h64 ^= h64 >> 29;
    h64 *= P3;
    h64 ^= h64 >> 32;

    return h64;
}
```

PREDICTION: 3.5

## MEASUREMENT

No tools (including `hash_bench`) were available in this session, so I was not able to actually execute the benchmark or the avalanche/collision checker here — this run is reasoning-only, per the session instructions ("the pipeline will compile and measure it" downstream). I did not fabricate numbers.

What the prediction of ~3.5x is based on, not measurement:
- FNV‑1a issues 1 multiply per byte on a fully serial dependency chain — throughput is latency-bound (~3–5 cycles/byte on typical x86‑64).
- The 4‑lane xxHash64‑style kernel issues only 2 multiplies per 8 bytes per lane (1 mul per 4 bytes averaged over the 32‑byte block), across 4 mutually‑independent chains, so an out‑of‑order core can keep several multiplies in flight and approach 1 multiply issued per cycle rather than 1 completed every latency-cycles. Combined with 4x fewer total multiplies for the same data, a 3–6x throughput improvement on buffers ≥32 bytes is the typical, widely‑reported result for this exact construction versus byte‑at‑a‑time FNV. For very small inputs (<16–32 bytes) the win shrinks toward parity because setup/branch overhead dominates and the bulk 32‑byte loop never runs — this is the main risk to the 3.5x prediction being an overestimate if the benchmark is dominated by tiny inputs.

## VERDICT

Correctness: the construction is the standard, extensively validated xxHash64 algorithm (passes SMHasher-class avalanche/collision tests in its original form), so I'm confident it satisfies the avalanche requirement and the exact fixed `uint64_t kernel(const unsigned char*, size_t)` contract, using only `stdint.h`/`string.h` (no OpenMP threads used, so no thread-related correctness/overhead risk). It is a direct, faithful translation of Solution 2 (independent lanes, poured and hard‑stirred together at the end) rather than Solution 1 (verbatim baseline) or the narrower 2‑lane Solution 3.

Performance: predicted faster than the FNV‑1a baseline for any buffer of at least a few dozen bytes, with the advantage shrinking toward parity on very short inputs. This prediction is provisional — it is not backed by an actual `hash_bench` run in this session, and should be treated as a hypothesis for the downstream pipeline to confirm or refute, not a confirmed result.