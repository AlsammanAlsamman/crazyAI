#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the four wire birds: distinct odd "beaks" ---- */
#define B0 0xa0761d6478bd642fULL
#define B1 0xe7037ed1a0b428dbULL
#define B2 0x8ebc6af09c88c6e3ULL
#define B3 0x589965cc75374cc3ULL
#define B4 0x1d8e4e27c47d124fULL
#define B5 0xeb44accab455d165ULL

/* one fold: two things pressed together into one crease.
   full 128-bit product, folded hi^lo -- the "tighter" refold. */
static inline uint64_t fold2(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
#else
    uint64_t lo = a * b;
    uint64_t a0 = a & 0xffffffffULL, a1 = a >> 32;
    uint64_t b0 = b & 0xffffffffULL, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    uint64_t hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return lo ^ hi;
#endif
}

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* unaligned-safe, strict-aliasing-safe reads (gcc -O3 folds these to MOVs) */
static inline uint64_t rd8(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t rd4(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }
static inline uint64_t rd3(const unsigned char *p, size_t k) {
    return ((uint64_t)p[0] << 16) | ((uint64_t)p[k >> 1] << 8) | (uint64_t)p[k - 1];
}

/* the water's edge: thin the wad to one dense corner, with len as the suitcase */
static inline uint64_t thin(uint64_t h, uint64_t len) {
    h = fold2(h ^ B0, len ^ B1);
    h ^= h >> 32; h *= B3;
    h ^= h >> 29; h *= B4;
    h ^= h >> 32;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    size_t n = len;
    uint64_t h;

    if (n >= 64) {
        /* four birds, each holding its own crease; len enters all four up front */
        uint64_t v0 = B0 ^ (uint64_t)len;
        uint64_t v1 = B1 ^ (uint64_t)len;
        uint64_t v2 = B2 ^ (uint64_t)len;
        uint64_t v3 = B3 ^ (uint64_t)len;
        do {
            __builtin_prefetch(p + 384, 0, 0);
            uint64_t w0 = rd8(p),      w1 = rd8(p + 8);
            uint64_t w2 = rd8(p + 16), w3 = rd8(p + 24);
            uint64_t w4 = rd8(p + 32), w5 = rd8(p + 40);
            uint64_t w6 = rd8(p + 48), w7 = rd8(p + 56);
            /* each crease = this mark folded against the crease before it;
               four beaks pressed in the same breath -> four independent chains */
            v0 = fold2(v0 ^ w0, B4 ^ w1) ^ w1;
            v1 = fold2(v1 ^ w2, B5 ^ w3) ^ w3;
            v2 = fold2(v2 ^ w4, B0 ^ w5) ^ w5;
            v3 = fold2(v3 ^ w6, B1 ^ w7) ^ w7;
            p += 64; n -= 64;
        } while (n >= 64);
        /* only when all four beaks agree does the crease count as set;
           prime rotations keep the spirals deliberately misaligned */
        h = fold2(rotl64(v0, 17) ^ v1 ^ B2, rotl64(v2, 43) ^ v3 ^ B3)
            ^ (v0 + v1 + v2 + v3);
    } else {
        /* GUARD: for short pulls the four birds are never set up at all */
        h = B0 ^ (uint64_t)len;
    }

    while (n >= 16) {
        uint64_t a = rd8(p), b = rd8(p + 8);
        h = fold2(h ^ a, B2 ^ b) ^ b;
        p += 16; n -= 16;
    }
    if (n) {
        uint64_t a, b;
        if (n >= 8)      { a = rd8(p);       b = rd8(p + n - 8); }
        else if (n >= 4) { a = rd4(p);       b = rd4(p + n - 4); }
        else             { a = rd3(p, n);    b = (uint64_t)n;    }
        h = fold2(h ^ a ^ B4, b ^ B5) ^ b;
    }

    return thin(h, (uint64_t)len);
}
