## APPROACH

Mapping the fable back to hashing: Solution 1 is exactly the given FNV-1a baseline — one accumulator, one multiply-xor per byte, strictly serial, "more twists = better" (extra rounds). Solution 3 (read everything first, mix later) doesn't help a hash: you still must touch every byte once and there's no benefit to decoupling "reading" from "mixing" when the mixing itself is cheap — it would just add a redundant pass over memory.

Solution 2 is the one that translates directly into real, standard hashing practice: **give several independent "stubby sticks" (accumulator lanes) their own charms at once, twist them independently, then fold the nubs together at the end.** The reason FNV-1a is slow isn't the ALU work — it's that each `h *= prime` has ~3-5 cycles of multiply latency, and the next byte's XOR/multiply *must* wait for it, so the whole 64-bit chain runs at multiply-latency speed regardless of how many multiply units the CPU has. Using 4 independent accumulators (like xxHash's v1..v4) lets the CPU issue 4 independent multiplies back-to-back with no dependency stall — same total work, but bound by multiply *throughput* instead of *latency*. This is a direct, well-known technique (xxHash64's actual design), not a novel gamble, so I'm implementing xxHash64's algorithm as the "several stubby sticks" solution, and additionally splitting very large buffers across OpenMP threads (independent segments hashed in parallel, folded together in deterministic index order) as the natural extension of "call over your stall-neighbors" to multiple cores.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
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

static inline uint64_t read_u64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint64_t xxh64_round(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t xxh64_mergeround(uint64_t acc, uint64_t val) {
    val = xxh64_round(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

/* xxHash64-style hash of one contiguous segment: 4 independent lanes
   (stubby sticks) twisted in parallel (ILP, not threads), folded together
   at the end, then a byte/word tail, then a final avalanche mix. */
static uint64_t xxh64_segment(const unsigned char *data, size_t len, uint64_t seed) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME64_1;

        do {
            v1 = xxh64_round(v1, read_u64(p)); p += 8;
            v2 = xxh64_round(v2, read_u64(p)); p += 8;
            v3 = xxh64_round(v3, read_u64(p)); p += 8;
            v4 = xxh64_round(v4, read_u64(p)); p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_mergeround(h64, v1);
        h64 = xxh64_mergeround(h64, v2);
        h64 = xxh64_mergeround(h64, v3);
        h64 = xxh64_mergeround(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = xxh64_round(0, read_u64(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t v32;
        memcpy(&v32, p, 4);
        h64 ^= (uint64_t)v32 * PRIME64_1;
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

uint64_t kernel(const unsigned char *data, size_t len) {
#ifdef _OPENMP
    /* For big buffers, hand chunks to different "stall-neighbors" (threads):
       hash each chunk independently, then fold the partial hashes together
       in fixed index order for determinism. */
    if (len >= (1u << 20)) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            int nc = nthreads > 64 ? 64 : nthreads;
            size_t base_chunk = len / (size_t)nc;
            if (base_chunk >= (1u << 16)) {
                uint64_t partial[64];
                #pragma omp parallel for schedule(static) num_threads(nc)
                for (int i = 0; i < nc; i++) {
                    size_t start = (size_t)i * base_chunk;
                    size_t clen = (i == nc - 1) ? (len - start) : base_chunk;
                    partial[i] = xxh64_segment(data + start, clen,
                                    (uint64_t)i * PRIME64_1 + 0x9E3779B185EBCA87ULL);
                }
                uint64_t h = (uint64_t)len ^ PRIME64_5;
                for (int i = 0; i < nc; i++) {
                    h ^= partial[i];
                    h = rotl64(h, 27) * PRIME64_1 + PRIME64_4;
                }
                h ^= h >> 33;
                h *= PRIME64_2;
                h ^= h >> 29;
                h *= PRIME64_3;
                h ^= h >> 32;
                return h;
            }
        }
    }
#endif
    return xxh64_segment(data, len, 0);
}
```

PREDICTION: 6

## MEASUREMENT

No `hash_bench` tool is available in this session (explicitly disabled per the task note); the pipeline compiles and measures this artifact after submission, so I cannot report actual numbers here. Reasoning basis for the prediction: FNV-1a's serial `xor+multiply` chain is bound by multiply latency (~3-5 cycles/byte-group), typically ~1-2 GB/s scalar throughput. xxHash64's 4-lane design processes 32 bytes/iteration with independent multiply chains, letting the CPU pipeline multiplies instead of stalling on latency — published xxHash64 benchmarks show ~10-15 GB/s single-threaded on modern x86, i.e. roughly 6-10x over naive FNV-1a for buffers large enough to hit the main loop; small buffers (<32 bytes) should be roughly on par or mildly faster due to fewer total operations. The OpenMP path only activates above 1 MiB with a real per-thread chunk, so it shouldn't hurt small/medium-size cases.

## VERDICT

Implementation is complete and, by construction, matches the fixed contract (`uint64_t kernel(const unsigned char*, size_t)`), reads the whole buffer, and is deterministic (thread partitioning is by fixed index, not completion order, so results don't depend on scheduling). Whether it beats FNV-1a as predicted, and by how much, awaits the pipeline's actual `hash_bench` run — pending confirmation, this stands as a reasoned but unverified prediction, not a measured result.