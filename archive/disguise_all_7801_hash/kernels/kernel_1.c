#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Mixing secrets: high-quality 64-bit constants (wyhash default secret),
   each with balanced popcount and no small factors. */
#define K0 0xa0761d6478bd642fULL
#define K1 0xe7037ed1a0b428dbULL
#define K2 0x8ebc6af09c88c6e3ULL
#define K3 0x589965cc75374cc3ULL

/* ---- "labeled saucers": pure loads, no state dependency ---- */
static inline uint64_t rd8(const unsigned char *p) {
    uint64_t v; __builtin_memcpy(&v, p, 8); return v;
}
static inline uint64_t rd4(const unsigned char *p) {
    uint32_t v; __builtin_memcpy(&v, p, 4); return (uint64_t)v;
}
static inline uint64_t rd3(const unsigned char *p, size_t k) {
    /* touches only p[0], p[k>>1], p[k-1] : never out of bounds for 1<=k<=3 */
    return (((uint64_t)p[0]) << 16) | (((uint64_t)p[k >> 1]) << 8) | (uint64_t)p[k - 1];
}

/* ---- "one hard twist": 64x64 -> 128, both halves kept ---- */
static inline void mum(uint64_t *a, uint64_t *b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)(*a) * (__uint128_t)(*b);
    *a = (uint64_t)r;
    *b = (uint64_t)(r >> 64);
#else
    uint64_t ha = *a >> 32, la = (uint32_t)*a, hb = *b >> 32, lb = (uint32_t)*b;
    uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb;
    uint64_t t = rl + (rm0 << 32), c = t < rl;
    uint64_t lo = t + (rm1 << 32); c += lo < t;
    uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
    *a = lo; *b = hi;
#endif
}
static inline uint64_t mix(uint64_t a, uint64_t b) { mum(&a, &b); return a ^ b; }

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    uint64_t seed = mix(K0, K1);          /* constant-folded at compile time */
    uint64_t a, b;

    if (len <= 16) {
        if (len >= 4) {
            size_t d = (len >> 3) << 2;
            a = (rd4(p) << 32)           | rd4(p + d);
            b = (rd4(p + len - 4) << 32) | rd4(p + len - 4 - d);
        } else if (len > 0) {
            a = rd3(p, len); b = 0;
        } else {
            a = 0; b = 0;
        }
    } else {
        size_t i = len;
        if (i > 48) {
            /* three independent lanes: 48 bytes absorbed per 3 multiplies,
               critical path = ONE multiply per lane per block */
            uint64_t s1 = seed, s2 = seed;
            do {
                uint64_t w0 = rd8(p),      w1 = rd8(p + 8);
                uint64_t w2 = rd8(p + 16), w3 = rd8(p + 24);
                uint64_t w4 = rd8(p + 32), w5 = rd8(p + 40);
                seed = mix(w0 ^ K1, w1 ^ seed);
                s1   = mix(w2 ^ K2, w3 ^ s1);
                s2   = mix(w4 ^ K3, w5 ^ s2);
                p += 48; i -= 48;
            } while (i > 48);
            seed ^= s1 ^ s2;              /* saucers collapse into the one blob */
        }
        while (i > 16) {
            seed = mix(rd8(p) ^ K1, rd8(p + 8) ^ seed);
            i -= 16; p += 16;
        }
        a = rd8(p + i - 16);              /* overlapping tail, always in bounds */
        b = rd8(p + i - 8);
    }

    a ^= K1; b ^= seed;
    mum(&a, &b);
    return mix(a ^ K0 ^ len, b ^ K1);    /* finalizer: length folded in */
}
