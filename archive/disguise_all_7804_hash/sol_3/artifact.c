#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Odd, high-entropy mixing constants (wyhash-family). */
#define KP0 0xa0761d6478bd642fULL
#define KP1 0xe7037ed1a0b428dbULL
#define KP2 0x8ebc6af09c88c6e3ULL
#define KP3 0x589965cc75374cc3ULL
#define KP4 0x1d8e4e27c47d124fULL
#define KP5 0xeb44accab455d165ULL

static inline uint64_t hk_r8(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint32_t hk_r4(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return v; }
static inline uint64_t hk_rotl(uint64_t x,int r){ return (x<<r)|(x>>(64-r)); }

/* "twist-stir": 64x64 -> 128 multiply, folded to 64 bits by XOR of halves. */
static inline uint64_t hk_fold(uint64_t a, uint64_t b){
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t al=a&0xffffffffULL, ah=a>>32, bl=b&0xffffffffULL, bh=b>>32;
    uint64_t ll=al*bl, lh=al*bh, hl=ah*bl, hh=ah*bh;
    uint64_t cross=(ll>>32)+(lh&0xffffffffULL)+hl;
    uint64_t hi=(lh>>32)+(cross>>32)+hh;
    uint64_t lo=(cross<<32)|(ll&0xffffffffULL);
    return lo ^ hi;
#endif
}

/* final firm twist-stir: MurmurHash3-class finalizer (full avalanche). */
static inline uint64_t hk_final(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t A = KP0 ^ (uint64_t)len;   /* bowl of the walker going forward  */
    uint64_t B = KP1 + (uint64_t)len;   /* bowl of the walker going backward */

    if (len < 16) {                      /* short shelf: the two meet immediately */
        if (len >= 8) {
            A ^= hk_r8(data);
            B ^= hk_r8(data + len - 8);
        } else if (len >= 4) {
            A ^= (uint64_t)hk_r4(data);
            B ^= (uint64_t)hk_r4(data + len - 4);
        } else if (len) {
            A ^= ((uint64_t)data[0] << 16) |
                 ((uint64_t)data[len >> 1] << 8) |
                 ((uint64_t)data[len - 1]);
            B ^= (uint64_t)data[len - 1] * KP5;
        }
        return hk_final(hk_fold(A ^ KP2, B ^ KP3) ^ (A + B));
    }

    const unsigned char *p = data;        /* forward walker  */
    const unsigned char *q = data + len;  /* backward walker (exclusive) */
    size_t rem = len;                     /* bytes still between them: q - p */

    /* Both walk toward the middle, 32 bytes each per step. */
    while (rem >= 64) {
        uint64_t a0 = hk_r8(p),      a1 = hk_r8(p + 8);
        uint64_t a2 = hk_r8(p + 16), a3 = hk_r8(p + 24);
        q -= 32;
        uint64_t b0 = hk_r8(q),      b1 = hk_r8(q + 8);
        uint64_t b2 = hk_r8(q + 16), b3 = hk_r8(q + 24);
        A = hk_rotl(A, 29) + hk_fold(a0 ^ KP1, a1 ^ KP2)
                           + hk_fold(a2 ^ KP3, a3 ^ KP4);
        B = hk_rotl(B, 31) + hk_fold(b0 ^ KP4, b1 ^ KP3)
                           + hk_fold(b2 ^ KP2, b3 ^ KP1);
        p += 32;
        rem -= 64;
    }
    /* 16 bytes each per step as they close in. */
    while (rem >= 32) {
        uint64_t a0 = hk_r8(p), a1 = hk_r8(p + 8);
        q -= 16;
        uint64_t b0 = hk_r8(q), b1 = hk_r8(q + 8);
        A = hk_rotl(A, 29) + hk_fold(a0 ^ KP1, a1 ^ KP2);
        B = hk_rotl(B, 31) + hk_fold(b0 ^ KP4, b1 ^ KP3);
        p += 16;
        rem -= 32;
    }

    /* The meeting: one last overlapping scoop each, covering every leftover jar.
       In-bounds for all len >= 16 because p and q are symmetric about the middle. */
    A = hk_rotl(A, 29) + hk_fold(hk_r8(p)      ^ KP1, hk_r8(p + 8)  ^ KP2);
    B = hk_rotl(B, 31) + hk_fold(hk_r8(q - 16) ^ KP4, hk_r8(q - 8)  ^ KP3);

    /* Pour the two bowls together, one final firm twist-stir. */
    return hk_final(hk_fold(A ^ KP2, B ^ KP3) ^ (A + B));
}
