#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define KA 0x9E3779B185EBCA87ULL
#define KB 0xC2B2AE3D27D4EB4FULL
#define KC 0x165667B19E3779F9ULL
#define KD 0x27D4EB2F165667C5ULL
#define KE 0x85EBCA77C2B2AE63ULL
#define KF 0x9FB21C651E98DF25ULL
#define KG 0xD6E8FEB86659FD93ULL
#define KH 0xA0761D6478BD642FULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t rd8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint64_t rd4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, 4); return (uint64_t)v;
}

/* terminal avalanche (fmix64) */
static inline uint64_t fmix(uint64_t h) {
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 32;
    return h;
}

#if defined(__SIZEOF_INT128__)
static inline uint64_t mulfold(uint64_t x, uint64_t y) {
    __uint128_t p = (__uint128_t)x * (__uint128_t)y;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
}
#else
static inline uint64_t mulfold(uint64_t x, uint64_t y) {
    uint64_t xl = (uint32_t)x, xh = x >> 32, yl = (uint32_t)y, yh = y >> 32;
    uint64_t ll = xl * yl, lh = xl * yh, hl = xh * yl, hh = xh * yh;
    uint64_t cross = (ll >> 32) + (uint32_t)lh + hl;
    uint64_t hi = hh + (cross >> 32) + (lh >> 32);
    uint64_t lo = (cross << 32) | (uint32_t)ll;
    return lo ^ hi;
}
#endif

/* --- the four different "hand motions": one application per word, cycled by position --- */
#define TWIST(a, v)   do { (a) = rotl64((a) ^ (v), 31) * KA; } while (0)
#define STRETCH(a, v) do { uint64_t t_ = (a) + (v); t_ ^= t_ >> 29; \
                           (a) = rotl64(t_, 21) + KB; } while (0)
#define BRAID(a, v)   do { (a) = (rotl64((a), 27) ^ rotl64((v), 17)) * KC; } while (0)
#define SHEAR(a, v)   do { uint64_t t_ = rotl64((a), 41) + ((v) ^ KD); \
                           (a) = t_ ^ (t_ >> 37); } while (0)

uint64_t kernel(const unsigned char *data, size_t len)
{
    if (len <= 16) {                                  /* short cards: one shaped fold */
        if (len == 0) return fmix(KA);
        if (len < 4) {
            uint64_t c = ((uint64_t)data[0] << 16)
                       | ((uint64_t)data[len >> 1] << 8)
                       | ((uint64_t)data[len - 1])
                       | ((uint64_t)len << 24);
            return fmix((c ^ KB) * KE);
        }
        if (len < 8) {
            uint64_t w = (rd4(data) << 32) | rd4(data + len - 4);
            return fmix((w ^ KB ^ ((uint64_t)len * KD)) * KE);
        }
        if (len == 8) return fmix((rd8(data) ^ KB) * KE);
        {
            uint64_t lo = rd8(data) ^ KB;
            uint64_t hi = rd8(data + len - 8) ^ KC;
            uint64_t m  = mulfold(lo, hi) + rotl64(lo, 31) + rotl64(hi, 47)
                        + (uint64_t)len * KD;
            return fmix(m);
        }
    }

    {
        const unsigned char *p = data, *end = data + len;
        uint64_t a0 = KA ^ (uint64_t)len, a1 = KB, a2 = KC, a3 = KD;

        if (len > 64) {                               /* main pass: 64 B, 8 positions */
            uint64_t a4 = KE, a5 = KF, a6 = KG, a7 = KH;
            do {
                TWIST  (a0, rd8(p     ));
                STRETCH(a1, rd8(p +  8));
                BRAID  (a2, rd8(p + 16));
                SHEAR  (a3, rd8(p + 24));
                TWIST  (a4, rd8(p + 32));
                STRETCH(a5, rd8(p + 40));
                BRAID  (a6, rd8(p + 48));
                SHEAR  (a7, rd8(p + 56));
                p += 64;
            } while ((size_t)(end - p) >= 64);
            a0 ^= rotl64(a4, 29);
            a1 ^= rotl64(a5, 31);
            a2 ^= rotl64(a6, 33);
            a3 ^= rotl64(a7, 35);
        }

        while ((size_t)(end - p) >= 32) {              /* 4 positions */
            TWIST  (a0, rd8(p     ));
            STRETCH(a1, rd8(p +  8));
            BRAID  (a2, rd8(p + 16));
            SHEAR  (a3, rd8(p + 24));
            p += 32;
        }
        {
            unsigned i = 0;
            while ((size_t)(end - p) >= 8) {            /* keep cycling the motions */
                uint64_t v = rd8(p);
                switch (i & 3u) {
                    case 0:  TWIST  (a0, v); break;
                    case 1:  STRETCH(a1, v); break;
                    case 2:  BRAID  (a2, v); break;
                    default: SHEAR  (a3, v); break;
                }
                p += 8; ++i;
            }
        }
        if (p != end) {                                 /* len >= 17 here: in bounds */
            uint64_t v = rd8(end - 8);
            BRAID(a1, v);
        }

        {
            uint64_t m = mulfold(a0 ^ KE, a1 ^ KF) + mulfold(a2 ^ KG, a3 ^ KH)
                       + rotl64(a0, 13) + rotl64(a1, 27)
                       + rotl64(a2, 41) + rotl64(a3, 55)
                       + (uint64_t)len * KD;
            return fmix(m);
        }
    }
}
