#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Four jars on the belt: distinct nonzero starters / stir secrets. */
#define S0 0xa0761d6478bd642fULL
#define S1 0xe7037ed1a0b428dbULL
#define S2 0x8ebc6af09c88c6e3ULL
#define S3 0x589965cc75374cc3ULL
#define S4 0x1d8e4e27c47d124fULL

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;          /* one mov at -O3 */
}
static inline uint64_t ld32(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return (uint64_t)v;
}

/* ONE hard figure-eight twist-stir: widening multiply, halves folded together.
   Absorbs two 64-bit operands for a single multiply instruction. */
static inline uint64_t stir(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t a0 = (uint32_t)a, a1 = a >> 32;
    uint64_t b0 = (uint32_t)b, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    uint64_t hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (a * b) ^ hi;
#endif
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    size_t n = len;

    /* Four jars, each starting from its own nonzero base. */
    uint64_t A = S0, B = S1, C = S2, D = S3;

    /* The walk: one pass, round-robin A,B,C,D; each pour gets its own single
       stir while the other three jars sit untouched (4 independent chains). */
    while (n >= 64) {
        A = stir(ld64(p +  0) ^ S1, ld64(p +  8) ^ A);
        B = stir(ld64(p + 16) ^ S2, ld64(p + 24) ^ B);
        C = stir(ld64(p + 32) ^ S3, ld64(p + 40) ^ C);
        D = stir(ld64(p + 48) ^ S4, ld64(p + 56) ^ D);
        p += 64; n -= 64;
    }
    /* Keep going round the jars for whole 16-byte spoonfuls that remain. */
    if (n >= 16) { A = stir(ld64(p) ^ S1, ld64(p + 8) ^ A); p += 16; n -= 16; }
    if (n >= 16) { B = stir(ld64(p) ^ S2, ld64(p + 8) ^ B); p += 16; n -= 16; }
    if (n >= 16) { C = stir(ld64(p) ^ S3, ld64(p + 8) ^ C); p += 16; n -= 16; }

    /* Last partial spoonful (0..15 bytes) goes into D. Overlapping reads stay
       strictly inside the buffer; nothing is read when len == 0. */
    uint64_t x, y;
    if (n >= 8)      { x = ld64(p);            y = ld64(p + n - 8); }
    else if (n >= 4) { x = ld32(p);            y = ld32(p + n - 4); }
    else if (n != 0) { x = ((uint64_t)p[0] << 16) |
                           ((uint64_t)p[n >> 1] << 8) |
                            (uint64_t)p[n - 1];       y = 0; }
    else             { x = 0;                  y = 0; }
    D = stir(x ^ S4, y ^ D);

    /* At the east gate: pour all four jars into the big jar, one final hard
       twist (fold + full 64-bit avalanche finisher). */
    uint64_t h = stir(A ^ S1, B ^ S2) ^ stir(C ^ S3, D ^ S4) ^ (uint64_t)len;
    h ^= h >> 33; h *= 0xc2b2ae3d27d4eb4fULL;
    h ^= h >> 29; h *= 0x165667b19e3779f9ULL;
    h ^= h >> 32;
    return h;
}
