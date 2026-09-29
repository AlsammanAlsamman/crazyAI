#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* pairwise-distinct, popcount-32 secrets */
#define S0 0xa0761d6478bd642fULL
#define S1 0xe7037ed1a0b428dbULL
#define S2 0x8ebc6af09c88c6e3ULL
#define S3 0x589965cc75374cc3ULL
#define S4 0x1d8e4e27c47d124fULL

static inline uint64_t ld8(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t ld4(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }

/* THE CORKSCREW: one widening multiply (low->high carry propagation),
   folded high^low (high bits dragged back down). One twist, thorough. */
static inline uint64_t corkscrew(uint64_t x, uint64_t y){
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)x * (__uint128_t)y;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t xl=(uint32_t)x, xh=x>>32, yl=(uint32_t)y, yh=y>>32;
    uint64_t ll=xl*yl, lh=xl*yh, hl=xh*yl, hh=xh*yh;
    uint64_t mid = lh + hl;
    uint64_t carry = (mid < lh) ? (1ULL<<32) : 0ULL;
    uint64_t lo = ll + (mid << 32);
    uint64_t c2 = (lo < ll);
    uint64_t hi = hh + (mid >> 32) + carry + c2;
    return lo ^ hi;
#endif
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    uint64_t h = S0 ^ (uint64_t)len;      /* the plain lump, salted with length */

    if (len >= 32) {
        size_t n = len;
        do {
            /* bytes 16..31 pre-folded OFF the critical path (parallel mul unit) */
            uint64_t t = corkscrew(ld8(p + 16) ^ S3, ld8(p + 24) ^ S4);
            /* ONE chained corkscrew swallows all 32 bytes */
            h = corkscrew(ld8(p) ^ h ^ S1, ld8(p + 8) ^ t ^ S2);
            p += 32; n -= 32;
        } while (n >= 32);
        if (n) {                           /* 1..31 left: one overlapping end twist */
            const unsigned char *q = data + len - 16;
            h = corkscrew(ld8(q) ^ h ^ S3, ld8(q + 8) ^ S4);
        }
    } else if (len >= 16) {
        h = corkscrew(ld8(p) ^ h ^ S1, ld8(p + 8) ^ S2);
        if (len > 16) {
            const unsigned char *q = data + len - 16;
            h = corkscrew(ld8(q) ^ h ^ S3, ld8(q + 8) ^ S4);
        }
    } else if (len >= 8) {
        h = corkscrew(ld8(p) ^ h ^ S1, ld8(data + len - 8) ^ S2);
    } else if (len >= 4) {
        h = corkscrew(ld4(p) ^ h ^ S1, ld4(data + len - 4) ^ S2);
    } else if (len) {
        uint64_t a = ((uint64_t)p[0] << 16) |
                     ((uint64_t)p[len >> 1] << 8) |
                      (uint64_t)p[len - 1];
        h = corkscrew(a ^ h ^ S1, S2);
    }

    /* one final thorough pull before handing it over (full 64-bit avalanche) */
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 29; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 32;
    return h;
}
