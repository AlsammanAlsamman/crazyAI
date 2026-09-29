#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* One jar. One stir per stop. West to east, once. */

#define JAR_M 0x9E3779B97F4A7C15ULL   /* the twist  */
#define JAR_R 31                       /* the flip   */

static inline uint64_t jar_rotl(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t jar_load64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);                 /* one mov; handles unaligned data */
    return v;
}

/* pour -> twist -> flip.  Exactly one of each, every single time. */
static inline uint64_t jar_stir(uint64_t h, uint64_t w) {
    return jar_rotl((h ^ w) * JAR_M, JAR_R);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    uint64_t h = 0x2545F4914F6CDD1DULL ^ (len * JAR_M);
    const unsigned char *p = data;
    size_t n = len >> 3;              /* number of full stops */
    size_t i = 0;

    /* The walk. The four stirs below are strictly sequential on h;
       only the loads and the counter run ahead. */
    for (; i + 4 <= n; i += 4) {
        h = jar_stir(h, jar_load64(p)); p += 8;
        h = jar_stir(h, jar_load64(p)); p += 8;
        h = jar_stir(h, jar_load64(p)); p += 8;
        h = jar_stir(h, jar_load64(p)); p += 8;
    }
    for (; i < n; i++) {
        h = jar_stir(h, jar_load64(p)); p += 8;
    }

    /* Last, partial flower: read only the bytes that exist, one stir. */
    size_t rem = len & 7;
    if (rem) {
        uint64_t t = 0;
        for (size_t k = 0; k < rem; k++)
            t |= (uint64_t)p[k] << (8 * k);
        h = jar_stir(h, t ^ ((uint64_t)rem << 56));   /* top byte is free */
    }

    /* At the gate: one last shake before the judges taste it. */
    h ^= (uint64_t)len;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}
