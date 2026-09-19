# APPROACH

The FNV-1a reference violates none of the physics of the CPU, but it wastes them: one byte per iteration, one multiply per byte, and every multiply is data-dependent on the previous one (`h` is a single accumulator), so the CPU can never have more than one multiply in flight. Latency-bound, not throughput-bound.

I'm keeping the "mix with multiply+xor+shift, fold the whole buffer through an accumulator" idea (it's the right idea for avalanche), but breaking the three assumptions that make FNV-1a slow without buying it anything:

1. **Wider reads.** Load 8 bytes at a time (unaligned `memcpy`) instead of 1, cutting per-byte overhead ~8x.
2. **Multiple independent accumulators.** Use 4 parallel state lanes (like xxHash64's `v1..v4`) instead of 1, so 4 independent multiply chains can be in flight simultaneously — this is what actually removes the latency bottleneck, not more rounds per byte.
3. **A proper avalanche finalizer** (xor-shift-multiply x3) at the end instead of relying on the per-byte mix alone, so tiny/short inputs still avalanche well even though the main loop only triggers for ≥32 bytes.

This is functionally xxHash64 (seed=0): 4-lane 32-byte main loop, 8/4/1-byte tail handling, final avalanche mix. It's a well-proven design specifically for this problem (fast + strong avalanche), not something exotic.

On top of that, since OpenMP is available and free to use: for large buffers (≥1 MiB) I split the buffer into up to `omp_get_max_threads()` contiguous chunks (deterministically, by index — not by completion order), hash each chunk independently in parallel with a per-chunk seed derived from its index, then fold the partial hashes together with one more mixing round + avalanche. This turns hashing of big buffers into a bandwidth-parallel, multi-core operation instead of a single-core streaming pass — directly attacking the "whole buffer must be read once, in order, by one accumulator" assumption for the regime where it hurts most.

Small buffers skip the parallel path entirely (thread spawn/join overhead would dominate), so they just pay for the serial 4-lane algorithm, which is still much cheaper than FNV-1a's byte loop.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define XXH_PRIME64_1 0x9E3779B185EBCA87ULL
#define XXH_PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define XXH_PRIME64_3 0x165667B19E3779F9ULL
#define XXH_PRIME64_4 0x85EBCA77C2B2AE63ULL
#define XXH_PRIME64_5 0x27D4EB2F165667C5ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t xxh64_round(uint64_t acc, uint64_t input) {
    acc += input * XXH_PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= XXH_PRIME64_1;
    return acc;
}

static inline uint64_t xxh64_merge_round(uint64_t acc, uint64_t val) {
    val = xxh64_round(0, val);
    acc ^= val;
    acc = acc * XXH_PRIME64_1 + XXH_PRIME64_4;
    return acc;
}

/* Core serial xxHash64-style hash over a contiguous range with given seed. */
static uint64_t hash_range(const unsigned char *data, size_t len, uint64_t seed) {
    const unsigned char *p = data;
    const unsigned char *bEnd = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = bEnd - 32;
        uint64_t v1 = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
        uint64_t v2 = seed + XXH_PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - XXH_PRIME64_1;

        do {
            uint64_t k1, k2, k3, k4;
            memcpy(&k1, p, 8); p += 8;
            memcpy(&k2, p, 8); p += 8;
            memcpy(&k3, p, 8); p += 8;
            memcpy(&k4, p, 8); p += 8;
            v1 = xxh64_round(v1, k1);
            v2 = xxh64_round(v2, k2);
            v3 = xxh64_round(v3, k3);
            v4 = xxh64_round(v4, k4);
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed + XXH_PRIME64_5;
    }

    h64 += (uint64_t)len;

    while ((size_t)(bEnd - p) >= 8) {
        uint64_t k1;
        memcpy(&k1, p, 8);
        k1 = xxh64_round(0, k1);
        h64 ^= k1;
        h64 = rotl64(h64, 27) * XXH_PRIME64_1 + XXH_PRIME64_4;
        p += 8;
    }
    if ((size_t)(bEnd - p) >= 4) {
        uint32_t k1;
        memcpy(&k1, p, 4);
        h64 ^= (uint64_t)k1 * XXH_PRIME64_1;
        h64 = rotl64(h64, 23) * XXH_PRIME64_2 + XXH_PRIME64_3;
        p += 4;
    }
    while (p < bEnd) {
        h64 ^= (uint64_t)(*p) * XXH_PRIME64_5;
        h64 = rotl64(h64, 11) * XXH_PRIME64_1;
        p++;
    }

    h64 ^= h64 >> 33;
    h64 *= XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= XXH_PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}

#define PAR_THRESHOLD ((size_t)1 << 20)   /* 1 MiB: below this, go serial */
#define MIN_CHUNK     ((size_t)1 << 16)   /* 64 KiB minimum chunk size */
#define MAX_LANES     64

uint64_t kernel(const unsigned char *data, size_t len) {
    if (len < PAR_THRESHOLD) {
        return hash_range(data, len, 0);
    }

    int maxT = 1;
#ifdef _OPENMP
    maxT = omp_get_max_threads();
    if (maxT < 1) maxT = 1;
#endif

    size_t maxChunks = len / MIN_CHUNK;
    if (maxChunks < 1) maxChunks = 1;

    size_t T = (size_t)maxT;
    if (T > maxChunks) T = maxChunks;
    if (T > MAX_LANES) T = MAX_LANES;
    if (T < 1) T = 1;

    uint64_t partial[MAX_LANES];
    size_t base = len / T;

    {
        long long Ti = (long long)T;
        #pragma omp parallel for schedule(static) if(T > 1)
        for (long long i = 0; i < Ti; i++) {
            size_t start = (size_t)i * base;
            size_t end = ((size_t)i == T - 1) ? len : (start + base);
            uint64_t seed = XXH_PRIME64_1 * (uint64_t)(i + 1) ^ (uint64_t)len;
            partial[i] = hash_range(data + start, end - start, seed);
        }
    }

    uint64_t h64 = (uint64_t)len ^ XXH_PRIME64_5;
    for (size_t i = 0; i < T; i++) {
        uint64_t k1 = xxh64_round(0, partial[i]);
        h64 ^= k1;
        h64 = rotl64(h64, 27) * XXH_PRIME64_1 + XXH_PRIME64_4;
    }

    h64 ^= h64 >> 33;
    h64 *= XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= XXH_PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}
```

# PREDICTION: 4.5

# MEASUREMENT

No tools were available in this session (the environment note explicitly disabled tool calls, including `hash_bench`), so I could not run the actual benchmark or `hash_contract` verification myself. The prediction above is a reasoning-based estimate, not a measured result:

- For buffers ≥32 bytes and below the 1 MiB parallel threshold, this is architecturally xxHash64: 32 bytes consumed per loop iteration instead of 1, across 4 independent multiply/rotate chains, which is publicly benchmarked at roughly 5-10x FNV-1a's throughput on modern x86 (FNV-1a is latency-bound at one multiply's latency per byte; xxHash64 keeps ~4 multiplies in flight per iteration and processes 8x more bytes per loop body).
- For very small buffers (<16-32 bytes), the win shrinks since both algorithms are dominated by fixed overhead/finalization rather than loop throughput — I'd expect closer to parity or a modest 1.5-2x win there.
- For buffers ≥1 MiB, the OpenMP chunked path should scale close to `min(cores, len/64KiB)`, potentially giving a much larger speedup than 4.5x on a multi-core box, but this depends entirely on `hash_bench`'s size distribution and core count, which I don't know.
- 4.5x is a blended guess assuming the benchmark harness weights small/medium/large buffers roughly evenly; if it's dominated by tiny (<32B) keys the real number will be lower, if dominated by large (≥1MiB) buffers on a many-core machine it could be much higher.

I could not execute the four allowed measure-and-improve iterations because no measurement tool was reachable; the pipeline will need to supply the actual numbers.

# VERDICT

Unverified in this session — correctness (must match contract signature exactly, must produce a well-mixed 64-bit output) and the 4.5x speedup prediction both rest on the well-established behavior of the xxHash64 algorithm and general knowledge of multiply latency vs. throughput on x86, not on an actual run here. If the pipeline's `hash_contract`/`hash_bench` results come back, they should be treated as authoritative over this prediction; in particular the OpenMP parallel path (used only above 1 MiB) is the part most likely to need retuning (thresholds, chunk size) once real numbers are available.