# APPROACH

FNV‑1a's bottleneck isn't the multiply cost per se — it's that every byte's update depends on the *previous* byte's state (`h ^= byte; h *= prime`), forming one long serial dependency chain. Even though a 64‑bit multiply has ~3‑5 cycle latency but 1/cycle throughput, FNV‑1a can never exploit that throughput because each iteration must wait for the previous multiply to finish. That's the load‑bearing assumption I'm breaking: "the state is a single accumulator" is not required for correctness or avalanche — only for simplicity. Also "more mixing rounds always means better mixing" is false past a saturation point (xxHash's SMHasher results confirm 1 round of multiply‑rotate‑multiply per lane already saturates avalanche).

Design (xxHash64‑style, since it's the well‑known fast/avalanche‑proven point in this design space):
1. **Four independent 64‑bit lanes**, each consuming a different 8‑byte word per 32‑byte block. Independent chains let the CPU pipeline multiple multiplies concurrently, hiding latency instead of serializing it — this is the main lever, not SIMD width.
2. Each lane: `v += word*P2; v = rotl(v,31); v *= P1` — one multiply-rotate-multiply per word, sufficient for full avalanche (proven by xxHash's SMHasher pass).
3. Lanes combined with rotations/XOR, tail bytes (8/4/1‑byte remainders) folded in with the standard xxHash tail mixing, then a 3‑step finalization avalanche (`xor-shift, mul, xor-shift, mul, xor-shift`) so short inputs and long inputs both fully avalanche.
4. For genuinely large buffers (≥1 MiB) split into a **fixed number of chunks (8, independent of `omp_get_num_threads()`)**, hash each chunk in parallel via OpenMP with a chunk-indexed seed, then fold the 8 partial hashes together in chunk-index order. Fixing the chunk count (not thread count) keeps the result deterministic regardless of how many OpenMP threads actually run — a hash must not depend on hardware/env at call time.
5. Small buffers skip OpenMP entirely (thread dispatch would dominate); they fall straight into the scalar tail path of the same block hasher.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

#define PRIME1 11400714785074694791ULL
#define PRIME2 14029467366897019727ULL
#define PRIME3 1609587929392839161ULL
#define PRIME4 9650029242287828579ULL
#define PRIME5 2870177450012600261ULL

static inline uint64_t read_u64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static uint64_t xxh64_block(const unsigned char *data, size_t len, uint64_t seed) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h64;

    if (len >= 32) {
        const unsigned char *limit = end - 32;
        uint64_t v1 = seed + PRIME1 + PRIME2;
        uint64_t v2 = seed + PRIME2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME1;

        do {
            v1 += read_u64(p) * PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1; p += 8;
            v2 += read_u64(p) * PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1; p += 8;
            v3 += read_u64(p) * PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1; p += 8;
            v4 += read_u64(p) * PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1; p += 8;
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME2; v1 = rotl64(v1, 31); v1 *= PRIME1; h64 ^= v1; h64 = h64 * PRIME1 + PRIME4;
        v2 *= PRIME2; v2 = rotl64(v2, 31); v2 *= PRIME1; h64 ^= v2; h64 = h64 * PRIME1 + PRIME4;
        v3 *= PRIME2; v3 = rotl64(v3, 31); v3 *= PRIME1; h64 ^= v3; h64 = h64 * PRIME1 + PRIME4;
        v4 *= PRIME2; v4 = rotl64(v4, 31); v4 *= PRIME1; h64 ^= v4; h64 = h64 * PRIME1 + PRIME4;
    } else {
        h64 = seed + PRIME5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = read_u64(p) * PRIME2;
        k1 = rotl64(k1, 31) * PRIME1;
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME1 + PRIME4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t v32;
        memcpy(&v32, p, 4);
        h64 ^= (uint64_t)v32 * PRIME1;
        h64 = rotl64(h64, 23) * PRIME2 + PRIME3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * PRIME5;
        h64 = rotl64(h64, 11) * PRIME1;
        p++;
    }

    h64 ^= h64 >> 33;
    h64 *= PRIME2;
    h64 ^= h64 >> 29;
    h64 *= PRIME3;
    h64 ^= h64 >> 32;

    return h64;
}

uint64_t kernel(const unsigned char *data, size_t len) {
#ifdef _OPENMP
    const size_t PAR_THRESHOLD = 1u << 20; /* 1 MiB */
    const int NCHUNKS = 8;
    if (len >= PAR_THRESHOLD) {
        uint64_t partial[8];
        size_t base = len / NCHUNKS;
        size_t rem = len % NCHUNKS;
        size_t offsets[9];
        offsets[0] = 0;
        for (int i = 0; i < NCHUNKS; i++) {
            size_t sz = base + (size_t)(i < (int)rem ? 1 : 0);
            offsets[i + 1] = offsets[i] + sz;
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NCHUNKS; i++) {
            partial[i] = xxh64_block(data + offsets[i], offsets[i + 1] - offsets[i],
                                      0x9E3779B97F4A7C15ULL + (uint64_t)i * PRIME5);
        }
        uint64_t h = PRIME5 + (uint64_t)len;
        for (int i = 0; i < NCHUNKS; i++) {
            h ^= partial[i];
            h = rotl64(h, 27) * PRIME1 + PRIME4;
        }
        h ^= h >> 33; h *= PRIME2; h ^= h >> 29; h *= PRIME3; h ^= h >> 32;
        return h;
    }
#endif
    return xxh64_block(data, len, 0);
}
```

# PREDICTION: 6

(predicted speedup factor over the FNV-1a baseline's throughput, averaged across the benchmark's size mix — driven mainly by processing 32 bytes/iteration across 4 independent multiply chains instead of 1 byte/iteration on a single serial chain, plus an OpenMP win on the ≥1 MiB tail of the distribution)

# MEASUREMENT

No tools were available in this session (the environment explicitly disabled tool access here), so `hash_bench`/`hash_contract` could not actually be invoked. I am reporting this plainly rather than fabricating numbers: **the prediction above is unverified.** Reasoning for the prediction, for the record: FNV‑1a is latency‑bound at roughly one multiply per byte with no ILP available (~1 byte per 3‑5 cycles ⇒ low single-digit GB/s on typical hardware); the 4‑lane xxHash64‑style block hasher processes 32 bytes per outer iteration across independent chains, which modern out‑of‑order cores can overlap, typically reaching well into double‑digit GB/s in published xxHash64 benchmarks — consistent with a ~5‑8x throughput gain on medium/large inputs, tempered downward on tiny inputs (a few bytes) where fixed overhead dominates and the two implementations converge.

# VERDICT

Implementation is complete and, by construction, matches the fixed `uint64_t kernel(const unsigned char*, size_t)` contract exactly, is deterministic (OpenMP chunking is keyed off a fixed chunk count, not runtime thread count, so results don't vary with `OMP_NUM_THREADS`), and follows a well-established, SMHasher-avalanche-validated mixing structure (xxHash64) rather than a novel untested one. However, I cannot claim a verified performance win here: no benchmarking tool was reachable in this session, so the PREDICTION stands unmeasured and should be treated as a hypothesis, not a result, until `hash_bench` is actually run against this artifact.