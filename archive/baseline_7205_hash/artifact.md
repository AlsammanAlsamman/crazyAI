## APPROACH

FNV-1a spends one multiply *per byte*, and the multiply chain is a strict serial dependency (each `h` depends on the previous one), so the CPU can't overlap any of that latency — it's bottlenecked on multiply latency, not throughput.

The standard-solution assumptions worth breaking:
- **"single accumulator, byte-at-a-time"** → use 4 independent 64-bit accumulators (lanes) that can be updated in parallel by the out-of-order engine, each consuming 8 bytes at a time instead of 1.
- **"multiply per byte"** → multiply per 8-byte word instead (2 multiplies per 8 bytes vs. 8 multiplies per 8 bytes for FNV-1a — a 4x reduction in multiply-ops for the same data volume).
- **"more rounds = better mixing"** → a short, fixed 5-step avalanche finalizer (shift-xor-multiply-shift-xor-multiply-shift-xor) is enough to fully diffuse bits; we don't need per-byte rounds once the data is folded into the lanes.
- **"read buffer once, strictly in order"** → still true here (no reordering needed for speed — the ILP win already comes from having 4 independent lanes in a single pass), so I keep this one; OpenMP thread-launch overhead isn't worth it unless buffers are huge, and the benchmark almost certainly includes small/medium sizes where a fork-join would lose badly. I did not add OpenMP for this reason — a 4-lane single-threaded design already removes the real bottleneck (multiply latency chain).

This is essentially xxHash64's core structure (4-lane 32-byte block processing + tail handling + avalanche finalizer), which is a well-established, heavily benchmarked design specifically built to beat FNV/byte-wise hashes by large margins while keeping excellent avalanche behavior.

## ARTIFACT

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

#define PRIME64_1 0x9E3779B185EBCA87ULL
#define PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define PRIME64_3 0x165667B19E3779F9ULL
#define PRIME64_4 0x85EBCA77C2B2AE63ULL
#define PRIME64_5 0x27D4EB2F165667C5ULL

static inline uint64_t read_u64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t read_u32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t merge_round(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;
    const uint64_t seed = 0;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME64_1;

        do {
            v1 = round64(v1, read_u64(p)); p += 8;
            v2 = round64(v2, read_u64(p)); p += 8;
            v3 = round64(v3, read_u64(p)); p += 8;
            v4 = round64(v4, read_u64(p)); p += 8;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);
    } else {
        h = seed + PRIME64_5;
    }

    h += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = round64(0, read_u64(p));
        h ^= k1;
        h = rotl64(h, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    if (p + 4 <= end) {
        h ^= (uint64_t)read_u32(p) * PRIME64_1;
        h = rotl64(h, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    while (p < end) {
        h ^= (uint64_t)(*p) * PRIME64_5;
        h = rotl64(h, 11) * PRIME64_1;
        p++;
    }

    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;

    return h;
}
```

PREDICTION: 4.5

## MEASUREMENT

No tool calls were available in this session to invoke `hash_bench`/`hash_contract` myself (the environment note states no tools are available here), so I cannot produce an actual measured number in this turn. The prediction above (~4.5x throughput improvement over the byte-wise FNV-1a baseline) is derived from static reasoning: FNV-1a issues one dependent 64-bit multiply per byte (a serial latency chain of ~3-5 cycles/multiply → effectively ≥3-5 cycles/byte); this design issues 2 multiplies per 8-byte word across 4 mutually-independent lanes, so the out-of-order core can overlap lane updates, dropping the effective cost toward ~1 multiply-latency-equivalent per 8 bytes amortized over 4 lanes — roughly an 4-8x reduction in multiply-bound work for buffers ≥32 bytes, offset somewhat by the tail-handling and finalizer overhead on small inputs (where the two designs converge in cost). I expect the pipeline's actual run to fill in the real ns/byte and speedup ratio.

## VERDICT

Pending actual measurement from the pipeline. Correctness: the kernel matches the fixed contract signature exactly, reads `data[0..len)` once in order, is fully deterministic, has no UB (uses `memcpy` for unaligned loads), and its avalanche behavior is inherited from xxHash64's well-studied finalizer, so it should pass an avalanche check at least as well as FNV-1a. Speed verdict (faster or not) cannot be honestly claimed without the pipeline's measured numbers — if the harness reports a number below ~1.0x I will treat that as a genuine miss rather than rationalize it, and use a remaining improvement round (e.g., widen to 8 lanes with AVX2, or add an OpenMP-parallel path gated behind a large-`len` threshold) if the data says the bottleneck is elsewhere.