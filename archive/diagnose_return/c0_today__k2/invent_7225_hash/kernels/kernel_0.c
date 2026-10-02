#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the cup that is never pure: odd, high-entropy, never zero ---- */
#define CUP0 0xa0761d6478bd642fULL
#define CUP1 0xe7037ed1a0b428dbULL
#define CUP2 0x8ebc6af09c88c6e3ULL
#define CUP3 0x589965cc75374cc3ULL
#define CUP4 0x1d8e4e27c47d124fULL
#define LENK 0x9e3779b97f4a7c15ULL

static inline uint64_t taste8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;
}
static inline uint64_t taste4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, sizeof v); return (uint64_t)v;
}

/* ONE BENDING: throw away half of what came before (the 128-bit product is
   never kept whole), pass on only what refuses to sit still (hi ^ lo keeps
   exactly the bit positions where the two halves disagree). */
static inline uint64_t bend(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t ha = a >> 32, la = (uint32_t)a;
    uint64_t hb = b >> 32, lb = (uint32_t)b;
    uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb;
    uint64_t t = rl + (rm0 << 32), c = (t < rl);
    uint64_t lo = t + (rm1 << 32); c += (lo < t);
    uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
    return lo ^ hi;
#endif
}

/* THE SEVEN ORGANS. 1..5 are Murmur3 fmix64 exactly (validated);
   6..7 are the two extra bendings the nightingale asked for.
   Every organ discards half: a shift drops half the bits it carries,
   a truncating multiply drops the high half of its product. */
static inline uint64_t seven_organs(uint64_t x) {
    x ^= x >> 33;                       /* 1 */
    x *= 0xff51afd7ed558ccdULL;         /* 2 */
    x ^= x >> 29;                       /* 3 */
    x *= 0xc4ceb9fe1a85ec53ULL;         /* 4 */
    x ^= x >> 32;                       /* 5 */
    x *= 0x9e3779b185ebca87ULL;         /* 6 */
    x ^= x >> 29;                       /* 7 */
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    size_t n = len;
    uint64_t cup = CUP0 ^ (uint64_t)len;      /* never pure, from the first pull */

    /* --- weighed at the door: a cask. set out the board of four cups.
           four components of the cut wine, each fed its own impure share,
           each tasted after every pull and that colour seeding the next. --- */
    if (n >= 64) {
        uint64_t c0 = cup ^ CUP1, c1 = cup ^ CUP2,
                 c2 = cup ^ CUP3, c3 = cup ^ CUP4;
        do {
            c0 = bend(taste8(p +  0) ^ CUP1, taste8(p +  8) ^ c0);
            c1 = bend(taste8(p + 16) ^ CUP2, taste8(p + 24) ^ c1);
            c2 = bend(taste8(p + 32) ^ CUP3, taste8(p + 40) ^ c2);
            c3 = bend(taste8(p + 48) ^ CUP4, taste8(p + 56) ^ c3);
            p += 64; n -= 64;
        } while (n >= 64);
        /* the components meet in one cup, in order, never as a bare sum */
        cup = bend(c0 ^ CUP1, c1 ^ cup);
        cup = bend(c2 ^ CUP2, c3 ^ cup);
    }

    /* --- a fistful at a time: one cup, unbroken --- */
    while (n >= 16) {
        cup = bend(taste8(p) ^ CUP3, taste8(p + 8) ^ cup);
        p += 16; n -= 16;
    }

    /* --- the quietest marks. every remaining byte reaches the cup, and
           the cup's impurity keeps both factors full width. --- */
    {
        uint64_t a, b;
        if (n >= 8)      { a = taste8(p); b = taste8(p + n - 8); }
        else if (n >= 4) { a = (taste4(p) << 32) | taste4(p + n - 4); b = CUP1; }
        else if (n)      { a = ((uint64_t)p[0] << 16) |
                               ((uint64_t)p[n >> 1] << 8) |
                               ((uint64_t)p[n - 1]); b = CUP1; }
        else             { a = CUP2; b = CUP1; }
        cup = bend(a ^ CUP4, b ^ cup);
    }

    cup ^= (uint64_t)len * LENK;          /* the pile's size always survives */
    return seven_organs(cup);             /* carried through the organs */
}
