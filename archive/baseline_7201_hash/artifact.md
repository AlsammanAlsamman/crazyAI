## APPROACH

The reference FNV-1a mixes one byte per multiply, with a strict serial dependency chain (each multiply must finish before the next XOR/multiply can start), so the CPU can never issue more than one multiply's worth of useful work per byte and the multiply latency (~3 cycles) dominates.

I break every "silent assumption" the prompt calls out:

- **Not one accumulator** — use 4 independent 64-bit lanes (`v1..v4`) so the multiply/rotate chains are independent and the out-of-order core can pipeline them (classic ILP trick used by xxHash64).
- **Not one byte at a time** — read 8 bytes (a full `uint64_t`) per mix step via `memcpy` (safe unaligned load), so each multiply now absorbs 8 bytes instead of 1: ~8x fewer multiplies for the same data.
- **Not "more rounds is always better"** — only 2 mixing passes per 32-byte block (one `round64` per lane, one `mergeround` per lane at the end) plus a 3-step finalization avalanche; this is the minimum SMHasher has shown to pass strict avalanche/bias tests, not more.
- Sequential single pass over the buffer is kept (still O(n), no restructuring needed, and required for correctness/determinism), with proper tail handling for 8/4/1-byte remainders so all lengths are handled exactly.

This is a scalar reimplementation of the well-known xxHash64 algorithm (five 64-bit primes, 4-lane 32-byte-per-iteration main loop, finalization avalanche). It is a widely validated, SMHasher-passing hash, and its structural win over FNV-1a (word-wide reads + 4-way ILP vs. byte-wide + single dependency chain) is well documented to give large throughput gains on buffers of more than a few dozen bytes, with only marginal extra overhead on tiny buffers.

## ARTIFACT

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

#define PRIME64_1 11400714785074694791ULL
#define PRIME64_2 14029467366897019727ULL
#define PRIME64_3 1609587929392839161ULL
#define PRIME64_4 9650029242287828579ULL
#define PRIME64_5 2870177450012600261ULL

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t mergeround(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

static inline uint64_t avalanche(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

static inline uint64_t read64le(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t read32le(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = PRIME64_1 + PRIME64_2;
        uint64_t v2 = PRIME64_2;
        uint64_t v3 = 0;
        uint64_t v4 = (uint64_t)0 - PRIME64_1;

        do {
            v1 = round64(v1, read64le(p)); p += 8;
            v2 = round64(v2, read64le(p)); p += 8;
            v3 = round64(v3, read64le(p)); p += 8;
            v4 = round64(v4, read64le(p)); p += 8;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = mergeround(h, v1);
        h = mergeround(h, v2);
        h = mergeround(h, v3);
        h = mergeround(h, v4);
    } else {
        h = PRIME64_5;
    }

    h += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = round64(0, read64le(p));
        h ^= k1;
        h = rotl64(h, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)read32le(p) * PRIME64_1;
        h = rotl64(h, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * PRIME64_5;
        h = rotl64(h, 11) * PRIME64_1;
        p++;
    }

    return avalanche(h);
}
```

PREDICTION: 4.5

## MEASUREMENT

No `hash_bench` / `hash_contract` tools were actually exposed in this session (the tool list I was given contains only Docs/Gmail/Drive/Slack tools, not the hashing or symbolic tools referenced in the task text), so I was not able to run the benchmark or contract check myself. Per the task's own fallback instructions ("No tools are available in this session... the pipeline will compile and measure it"), I'm submitting the artifact and prediction for the external pipeline to compile and measure; I have not observed real numbers and am not fabricating any.

## VERDICT

Unverified in this session — no measurement was performed, so this is a reasoned prediction, not a confirmed result. Structurally, this kernel should beat byte-wise FNV-1a on any buffer of more than a few dozen bytes: it does 8x fewer multiplies per byte (word-wide reads) and runs 4 independent multiply/rotate chains that an out-of-order core can overlap, versus FNV-1a's single strictly-serial multiply-per-byte chain. For very small inputs (well under 32 bytes) the extra branching (length checks, tail-handling cascade) gives it less of an edge, and it could even be marginally slower than the trivial FNV-1a loop there — if the pipeline's benchmark is dominated by tiny buffers, the real speedup may come in well below the 4.5x prediction. If given further iterations, next steps would be: measure the crossover point for small-buffer overhead and simplify the tail path if it's costly, and consider an OpenMP-chunked variant only if the benchmark uses buffers large enough (multi-MB) to amortize thread-spawn cost.