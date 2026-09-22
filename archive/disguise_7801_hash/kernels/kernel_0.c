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
