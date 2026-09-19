## APPROACH

The FNV-1a reference does one multiply per **byte** with a single scalar accumulator — this serializes every step (each multiply depends on the previous one) and wastes the CPU's ability to issue multiple independent multiplies per cycle.

I challenge three of the baked-in assumptions:

1. **"State is a single accumulator"** → use 4 independent 64-bit lanes (like xxHash64) so the CPU's out-of-order engine can run several multiply/rotate chains in parallel instead of waiting on one dependency chain.
2. **"Mixing one byte requires a multiplication"** → consume 8 bytes per multiply in the main loop (word-at-a-time via `memcpy` to avoid unaligned-UB), so total multiplies drop ~8× for large buffers.
3. **"More mixing rounds is always better"** → fewer, wider, well-vetted rounds (xxHash64's round/mergeRound/avalanche functions) give a stronger, well-tested avalanche profile than repeated byte-at-a-time xor-multiply, using less total work.
4. **"Read the whole buffer once, in order, single-threaded"** → for very large buffers (≥1 MiB) split into 4 fixed, deterministic segments, hash each in parallel with OpenMP using a distinct per-segment seed, then fold the 4 sub-hashes in a fixed order. The segment count is fixed (not tied to `omp_get_max_threads()`), so the result is deterministic regardless of how many threads actually run.

This is essentially a from-scratch reimplementation of the well-known xxHash64 algorithm (public, well-documented mixing constants/rounds), plus a deterministic OpenMP fan-out for big inputs.

## ARTIFACT

```c
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define PRIME64_1 11400714785074694791ULL
#define PRIME64_2 14029467366897019727ULL
#define PRIME64_3 1609587929392839161ULL
#define PRIME64_4 9650029242287828579ULL
#define PRIME64_5 2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t read64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t mergeRound64(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

static inline uint64_t avalanche64(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

/* Core xxHash64-style mixer: word-at-a-time, 4 parallel lanes. */
static uint64_t xxh64_core(const unsigned char *data, size_t len, uint64_t seed) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - PRIME64_1;

        do {
            v1 = round64(v1, read64(p)); p += 8;
            v2 = round64(v2, read64(p)); p += 8;
            v3 = round64(v3, read64(p)); p += 8;
            v4 = round64(v4, read64(p)); p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = mergeRound64(h64, v1);
        h64 = mergeRound64(h64, v2);
        h64 = mergeRound64(h64, v3);
        h64 = mergeRound64(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = round64(0, read64(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t v;
        memcpy(&v, p, 4);
        h64 ^= (uint64_t)v * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    return avalanche64(h64);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* Deterministic fixed fan-out for large buffers only; small/medium
       buffers go straight through the single-threaded core, which is
       already far cheaper per byte than a byte-at-a-time multiply. */
    if (len >= (1u << 20)) {
        enum { NSEG = 4 };
        size_t seg_len = len / NSEG;
        seg_len -= seg_len % 32;           /* keep segment boundaries clean */
        if (seg_len == 0) seg_len = len;   /* defensive; unreachable at this size */

        size_t offs[NSEG], lens[NSEG];
        for (int i = 0; i < NSEG; i++) {
            offs[i] = (size_t)i * seg_len;
            lens[i] = (i == NSEG - 1) ? (len - offs[i]) : seg_len;
        }

        uint64_t hashes[NSEG];
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NSEG; i++) {
            uint64_t seed = (uint64_t)i * PRIME64_5 + 0x9E3779B97F4A7C15ULL;
            hashes[i] = xxh64_core(data + offs[i], lens[i], seed);
        }

        uint64_t acc = PRIME64_1 ^ (uint64_t)len;
        for (int i = 0; i < NSEG; i++) {
            acc ^= hashes[i];
            acc = rotl64(acc, 27) * PRIME64_1 + PRIME64_4;
        }
        return avalanche64(acc);
    }

    return xxh64_core(data, len, 0);
}
```

PREDICTION: 4.0

## MEASUREMENT

No `hash_bench`/`hash_contract` tool call was available in this session (the environment explicitly disabled tool use here and stated the pipeline will compile and measure the artifact afterward). I have not fabricated a benchmark number — the figure above is a stated *prediction*, not a measured result. Reasoning behind the 4.0 estimate: the main loop replaces 1 multiply/byte (FNV-1a, fully serial dependency chain) with 4 multiplies per 32 bytes across 4 independent lanes (~1 multiply per 8 bytes, and those multiplies are mutually independent so the CPU can pipeline them), which typically yields xxHash64-class throughput several times that of scalar FNV-1a on modern x86; the OpenMP fan-out should add further gains only on the largest buffers, where thread-pool dispatch overhead (~1–2 µs) is negligible relative to the work.

## VERDICT

Pending actual measurement by the pipeline. Expected outcome based on the algorithmic argument: a clear win on medium/large buffers (word-at-a-time + ILP across 4 lanes vs. byte-serial FNV-1a), roughly break-even to modestly faster on very small buffers (few bytes) where loop overhead dominates for both implementations, and correctness/avalanche quality should be at least as good as FNV-1a since the mixing rounds are the well-vetted xxHash64 rounds. If measurement shows the OpenMP path underperforming at the 1 MiB threshold (thread overhead not amortized), the fix would be to raise the threshold (e.g., to 4–8 MiB) rather than removing the parallel path — that would be iteration #1 of the allowed four.