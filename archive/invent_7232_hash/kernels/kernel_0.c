#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define P1 11400714785074694791ULL
#define P2 14029467366897019727ULL
#define P3 1609587929392839161ULL
#define P4 9650029242287828579ULL
#define P5 2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

static inline uint64_t read_u64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t read_u32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * P2;
    acc = rotl64(acc, 31);
    acc *= P1;
    return acc;
}
static inline uint64_t merge_round(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * P1 + P4;
    return acc;
}

/* Fold one contiguous run of bytes into a single 64-bit digest under the given seed. */
static uint64_t hash_block(const unsigned char *p, size_t n, uint64_t seed) {
    const unsigned char *end = p + n;
    uint64_t h;
    if (n >= 32) {
        uint64_t v1 = seed + P1 + P2;
        uint64_t v2 = seed + P2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - P1;
        const unsigned char *limit = end - 32;
        do {
            v1 = round64(v1, read_u64(p)); p += 8;
            v2 = round64(v2, read_u64(p)); p += 8;
            v3 = round64(v3, read_u64(p)); p += 8;
            v4 = round64(v4, read_u64(p)); p += 8;
        } while (p <= limit);
        h = rotl64(v1,1) + rotl64(v2,7) + rotl64(v3,12) + rotl64(v4,18);
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);
    } else {
        h = seed + P5;
    }
    h += (uint64_t)n;
    while (p + 8 <= end) {
        h ^= round64(0, read_u64(p));
        h = rotl64(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)read_u32(p) * P1;
        h = rotl64(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * P5;
        h = rotl64(h, 11) * P1;
        p++;
    }
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const size_t MIN_PARALLEL = (size_t)1 << 20;   /* below this, one hand alone is faster */
    const size_t MIN_CHUNK    = (size_t)1 << 17;   /* smallest fair share for a hand */
    const int MAX_CHUNKS = 16;

    if (len < MIN_PARALLEL) {
        return hash_block(data, len, 0);
    }

    int nchunks = (int)(len / MIN_CHUNK);
    if (nchunks > MAX_CHUNKS) nchunks = MAX_CHUNKS;
    if (nchunks < 2) {
        return hash_block(data, len, 0);
    }

    uint64_t partial[MAX_CHUNKS];
    size_t chunk = len / (size_t)nchunks;

    #pragma omp parallel for num_threads(nchunks) schedule(static)
    for (int i = 0; i < nchunks; i++) {
        size_t start = (size_t)i * chunk;
        size_t stop  = (i == nchunks - 1) ? len : start + chunk;
        uint64_t seed = P5 ^ ((uint64_t)i * P3 + 0x9E3779B97F4A7C15ULL);
        partial[i] = hash_block(data + start, stop - start, seed);
    }

    uint64_t h = (uint64_t)len ^ P1;
    for (int i = 0; i < nchunks; i++) {
        h ^= partial[i];
        h = rotl64(h, 29) * P2 + P4;
        h ^= h >> 31;
    }
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}
