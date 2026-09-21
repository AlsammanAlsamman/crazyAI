## APPROACH

The FNV-1a reference forces a strictly serial dependency chain: each byte's `h *= prime` cannot start until the previous byte's multiply-xor has finished, so throughput is bounded by multiply latency (~3 cycles/byte on modern x86 → ~0.33 bytes/cycle), no matter how wide the CPU's execution ports are. None of the "silent assumptions" in the prompt are actually required by the contract — only "hash the buffer, avalanche well, be fast" is required.

I break the single-accumulator serial chain into **four independent accumulator lanes** (classic xxHash64 structure), each consuming its own 8-byte stripe of a 32-byte block per iteration. Because the four lanes have no data dependency on each other, the CPU can execute their adds/rotates/multiplies with instruction-level parallelism, overlapping ~4 independent multiply-latency chains instead of serializing one. The four lanes are combined and cross-mixed at the end (`mergeRound`), and a final 3-round xor-shift-multiply avalanche finalizer (proven via extensive prior testing to hit near-ideal bit avalanche) is applied. Tail bytes (<32B remainder) are folded in via 8-byte/4-byte/1-byte passes, each with its own mix-and-rotate step, so avalanche quality doesn't degrade for odd-sized inputs. Reads are unaligned via `memcpy` (compiles to a single unaligned load on x86-64 with `-march=native`).

I did not add OpenMP threading: thread launch/join overhead (~microseconds) would dominate for the small-to-medium buffer sizes a hash-throughput benchmark typically sweeps, and would only pay off for very large (multi-MB+) inputs. The ILP-lane trick is a strict win at all sizes ≥32B, and does not lose anything below 32B since we just fall through to the tail loop (8 bytes/iteration vs FNV-1a's 1 byte/iteration).

## ARTIFACT

```c
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t read64LE(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t read32LE(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

#define PRIME64_1 0x9E3779B185EBCA87ULL
#define PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define PRIME64_3 0x165667B19E3779F9ULL
#define PRIME64_4 0x85EBCA77C2B2AE63ULL
#define PRIME64_5 0x27D4EB2F165667C5ULL

static inline uint64_t xxh_round(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t xxh_merge(uint64_t acc, uint64_t val) {
    val = xxh_round(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *bEnd = data + len;
    const uint64_t seed = 0ULL;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = bEnd - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME64_1;

        do {
            v1 = xxh_round(v1, read64LE(p)); p += 8;
            v2 = xxh_round(v2, read64LE(p)); p += 8;
            v3 = xxh_round(v3, read64LE(p)); p += 8;
            v4 = xxh_round(v4, read64LE(p)); p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh_merge(h64, v1);
        h64 = xxh_merge(h64, v2);
        h64 = xxh_merge(h64, v3);
        h64 = xxh_merge(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += (uint64_t)len;

    while ((size_t)(bEnd - p) >= 8) {
        uint64_t k1 = xxh_round(0, read64LE(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    if ((size_t)(bEnd - p) >= 4) {
        h64 ^= (uint64_t)read32LE(p) * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    while (p < bEnd) {
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

PREDICTION: 3.5

## MEASUREMENT

No benchmarking tools (`hash_bench`, `hash_contract`) were actually invocable in this session — the tool-calling interface returned none available, and I was instructed to reason, predict, and hand off the artifact rather than run anything myself. I have **not** measured this implementation; the numbers above are a reasoned estimate, not an observed result. Reporting this plainly rather than fabricating a MEASUREMENT trace: the pipeline referenced in the instructions is expected to compile and benchmark this kernel against the FNV-1a baseline separately.

Reasoning behind the 3.5 estimate: for large buffers (≫32B), 4-lane ILP should let throughput approach the multiply-throughput limit rather than multiply-latency limit — plausibly 5–8x over FNV-1a. For small buffers (<32B, common in many hash-benchmark sweeps), the win shrinks to roughly the 8-bytes-per-step vs 1-byte-per-step tail-loop ratio, plus one-time branch overhead, i.e. closer to 1.5–2x, and could even be marginally slower for very tiny (1–3 byte) inputs due to the extra branching before the tail loop. Averaged across a typical size sweep, ~3.5x seems like a fair, conservative point estimate rather than the best-case large-buffer number.

## VERDICT

Correctness: this is a direct, careful reimplementation of the well-established, extensively avalanche-tested xxHash64 algorithm (seed=0), so it should pass any avalanche/uniformity check the harness applies, and it satisfies the fixed contract signature exactly (`uint64_t kernel(const unsigned char *data, size_t len)`), with no aliasing/alignment UB (all multi-byte reads go through `memcpy`).

Performance claim: unverified in this session — I could not run `hash_bench` here, so I'm reporting that gap plainly rather than presenting the 3.5x prediction as a measured fact. The structural argument (breaking the single serial multiply-dependency chain into 4 independent ILP-friendly lanes) is sound and is the same argument that makes xxHash faster than FNV-1a in real-world benchmarks, but the actual number needs to come from the pipeline's run, not from me.