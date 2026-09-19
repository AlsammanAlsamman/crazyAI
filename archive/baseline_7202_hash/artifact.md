# APPROACH

The FNV-1a reference has a single accumulator with a **strict sequential dependency chain**: each `h *= prime` cannot start until the previous multiply finishes, so the CPU can never issue more than one multiply per byte in flight — multiply latency (3–5 cycles) dominates, giving effectively ~1 byte per few cycles no matter how wide the pipeline is.

The standard solution's hidden assumptions can all be relaxed without hurting avalanche quality:
- **"one accumulator"** → use **4 independent accumulators** (lanes) that only get combined at the very end. This breaks the dependency chain into 4 chains the CPU can run concurrently (ILP), hiding multiply latency.
- **"mix one byte per step"** → consume **8 bytes per lane per step** (64-bit word) via an unaligned load, so 32 bytes are absorbed per loop iteration instead of 1.
- **"more rounds = better"** → a single 32-bit-strength round (multiply-add, rotate, multiply) per lane per word is already enough since the final **avalanche mix** (xor-shift/multiply x3) provides the full-bit diffusion at the end, so we don't need to over-mix every intermediate step.

This is exactly the structure xxHash64 uses, so I implement that algorithm (seed=0) directly against the given contract: 4-lane 32-byte-per-iteration main loop with `round()`, lane merge with `mergeRound()`, then word/dword/byte tail handling, then the standard 3-step avalanche finalizer. It is a well-studied, high-quality hash (passes SMHasher) so avalanche is not a concern, and its 4-way ILP is the core speed win over serial FNV-1a, especially on buffers of more than a few dozen bytes.

# ARTIFACT

```c
#include <stdint.h>
#include <string.h>

#define PRIME64_1 0x9E3779B185EBCA87ULL
#define PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define PRIME64_3 0x165667B19E3779F9ULL
#define PRIME64_4 0x85EBCA77C2B2AE63ULL
#define PRIME64_5 0x27D4EB2F165667C5ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t read64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t read32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline uint64_t xxh64_round(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t xxh64_mergeRound(uint64_t acc, uint64_t val) {
    val = xxh64_round(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *const end = data + len;
    uint64_t h64;
    const uint64_t seed = 0;

    if (len >= 32) {
        const unsigned char *const limit = end - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME64_1;

        do {
            v1 = xxh64_round(v1, read64(p)); p += 8;
            v2 = xxh64_round(v2, read64(p)); p += 8;
            v3 = xxh64_round(v3, read64(p)); p += 8;
            v4 = xxh64_round(v4, read64(p)); p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_mergeRound(h64, v1);
        h64 = xxh64_mergeRound(h64, v2);
        h64 = xxh64_mergeRound(h64, v3);
        h64 = xxh64_mergeRound(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = xxh64_round(0, read64(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    if (p + 4 <= end) {
        h64 ^= (uint64_t)read32(p) * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    while (p < end) {
        h64 ^= (uint64_t)(*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}
```

PREDICTION: 4.5

# MEASUREMENT

No `hash_bench` / `hash_contract` tool was reachable in this session (the environment note states no tools are available here), so I could not execute the benchmark or the avalanche/correctness checks myself. I'm reporting this plainly rather than fabricating numbers: the prediction above (~4.5x throughput vs. the byte-serial FNV-1a reference, largest on buffers ≳64 bytes, smaller-but-still-positive on tiny buffers where loop/tail overhead dominates) is based on reasoning about dependency-chain latency vs. 4-way ILP and 8×-wider reads per step, not on an actual run. The external pipeline mentioned in the instructions is expected to compile and measure this artifact; I have not seen those results.

# VERDICT

Correctness: this is the standard, widely-audited xxHash64 algorithm (SMHasher-clean, excellent avalanche by construction of its finalizer) applied unmodified to the given `(data, len)` contract with seed 0 — I'm confident it satisfies the mixing requirement and matches the exact function signature. Performance: unmeasured in this session — my 4.5x prediction is a reasoned estimate (serial multiply-latency-bound FNV-1a vs. 4-lane ILP-parallel, 32-bytes/iteration xxHash64), not a verified result, and should be treated as provisional until the pipeline's actual benchmark numbers come back.