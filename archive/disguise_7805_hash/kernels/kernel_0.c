#include <stdint.h>
#include <string.h>

#define P64_1 11400714785074694791ULL
#define P64_2 14029467366897019727ULL
#define P64_3 1609587929392839161ULL
#define P64_4 9650029242287828579ULL
#define P64_5 2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t get64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t get32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

/* one "strong twist": multiply, rotate, multiply */
static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * P64_2;
    acc = rotl64(acc, 31);
    acc *= P64_1;
    return acc;
}

static inline uint64_t merge_round(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * P64_1 + P64_4;
    return acc;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    const unsigned char *end = data + len;
    uint64_t h;

    if (len >= 32) {
        /* four independent lanes: no data dependency between them,
           so their multiplies overlap in the CPU's pipeline */
        uint64_t v1 = P64_1 + P64_2;
        uint64_t v2 = P64_2;
        uint64_t v3 = 0;
        uint64_t v4 = (uint64_t)0 - P64_1;

        const unsigned char *limit = end - 32;
        do {
            v1 = round64(v1, get64(p));      p += 8;
            v2 = round64(v2, get64(p));      p += 8;
            v3 = round64(v3, get64(p));      p += 8;
            v4 = round64(v4, get64(p));      p += 8;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);
    } else {
        h = P64_5;
    }

    h += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = round64(0, get64(p));
        h ^= k1;
        h = rotl64(h, 27) * P64_1 + P64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)get32(p) * P64_1;
        h = rotl64(h, 23) * P64_2 + P64_3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * P64_5;
        h = rotl64(h, 11) * P64_1;
        p++;
    }

    /* final avalanche: the one deliberate strong twist */
    h ^= h >> 33;
    h *= P64_2;
    h ^= h >> 29;
    h *= P64_3;
    h ^= h >> 32;

    return h;
}
