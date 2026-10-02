#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the eight strings' resting pitches ---- */
static const uint64_t ACC_INIT[8] = {
    0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0x85EBCA77C2B2AE63ULL,
    0x27D4EB2F165667C5ULL, 0xD6E8FEB86659FD93ULL,
    0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL
};
/* ---- the first feather drawn from the inkwell ---- */
static const uint64_t FEATHER0[8] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
    0x94D049BB133111EBULL, 0xDA942042E4DD58B5ULL,
    0x8CB92BA72F3D8DD7ULL, 0xC4CEB9FE1A85EC53ULL,
    0xFF51AFD7ED558CCDULL, 0xB5026F5AA96619E9ULL
};
/* ---- the inkwell never runs dry: a fresh drop per lane per stripe ---- */
static const uint64_t FSTEP[8] = {
    0x6A09E667F3BCC909ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

static inline uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }
static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

/* the feather takes the weight, not the shape: 32x32 -> 64 of the inked word */
static inline uint64_t weigh(uint64_t k) {
    return (uint64_t)(uint32_t)k * (uint64_t)(uint32_t)(k >> 32);
}

/* one chord of the storm: 64x64 -> 128, folded to 64 */
static inline uint64_t fold128(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32, bl = (uint32_t)b, bh = b >> 32;
    uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint64_t cross = (ll >> 32) + (uint32_t)lh + hl;
    uint64_t hi = hh + (lh >> 32) + (cross >> 32);
    uint64_t lo = (cross << 32) | (uint32_t)ll;
    return lo ^ hi;
#endif
}

/* THE STORM: exactly one pass, hands off until the last mark is plucked.
   (rrmxmx -- validated strong 64-bit finalizer) */
static inline uint64_t storm(uint64_t h, uint64_t len) {
    h ^= rotl64(h, 49) ^ rotl64(h, 24);
    h *= 0x9FB21C651E98DF25ULL;
    h ^= (h >> 35) + len;
    h *= 0x9FB21C651E98DF25ULL;
    h ^= h >> 28;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t h;

    /* ---- the bars hold the count: which regime is this pile? ---- */
    if (len < 64) {
        /* the pile cannot reach across all eight strings: pluck what it reaches,
           with overlapping reads, and hand it straight to the storm. */
        uint64_t a, b;
        if (len >= 16) {
            h  = (uint64_t)len * ACC_INIT[0];
            h += fold128(rd64(p)           ^ FEATHER0[0], rd64(p + 8)         ^ FEATHER0[1]);
            h += fold128(rd64(p + len - 16) ^ FEATHER0[2], rd64(p + len - 8)  ^ FEATHER0[3]);
            if (len > 32) {
                h += fold128(rd64(p + 16)       ^ FEATHER0[4], rd64(p + 24)       ^ FEATHER0[5]);
                h += fold128(rd64(p + len - 32) ^ FEATHER0[6], rd64(p + len - 24) ^ FEATHER0[7]);
            }
            return storm(h, (uint64_t)len);
        }
        if (len >= 8)      { a = rd64(p);  b = rd64(p + len - 8); }
        else if (len >= 4) { a = rd32(p);  b = rd32(p + len - 4); }
        else if (len)      { a = ((uint64_t)p[0] << 16) | ((uint64_t)p[len >> 1] << 8)
                                 | (uint64_t)p[len - 1];
                             b = a + (uint64_t)len; }
        else               { a = 0; b = 0; }
        h = (uint64_t)len * ACC_INIT[0] + fold128(a ^ FEATHER0[0], b ^ FEATHER0[1]);
        return storm(h, (uint64_t)len);
    }

    /* ---- the pile reaches all eight strings: the bridge path ---- */
    {
        const unsigned char *end = p + len;
        const size_t nstripes = len >> 6;
        uint64_t acc[8], w[8];
        size_t s;
        int i;

#if defined(__AVX2__)
        {
            __m256i av0 = _mm256_loadu_si256((const __m256i *)(const void *)(ACC_INIT));
            __m256i av1 = _mm256_loadu_si256((const __m256i *)(const void *)(ACC_INIT + 4));
            __m256i wv0 = _mm256_loadu_si256((const __m256i *)(const void *)(FEATHER0));
            __m256i wv1 = _mm256_loadu_si256((const __m256i *)(const void *)(FEATHER0 + 4));
            const __m256i tv0 = _mm256_loadu_si256((const __m256i *)(const void *)(FSTEP));
            const __m256i tv1 = _mm256_loadu_si256((const __m256i *)(const void *)(FSTEP + 4));

            for (s = 0; s < nstripes; s++) {
                const unsigned char *q = p + (s << 6);
                __m256i d0 = _mm256_loadu_si256((const __m256i *)(const void *)q);
                __m256i d1 = _mm256_loadu_si256((const __m256i *)(const void *)(q + 32));
                __m256i k0 = _mm256_xor_si256(d0, wv0);     /* dip in the inkwell */
                __m256i k1 = _mm256_xor_si256(d1, wv1);
                wv0 = _mm256_add_epi64(wv0, tv0);           /* a fresh feather, always */
                wv1 = _mm256_add_epi64(wv1, tv1);
                /* the pluck: one constant force, never varying */
                av0 = _mm256_add_epi64(av0,
                      _mm256_add_epi64(_mm256_mul_epu32(k0, _mm256_srli_epi64(k0, 32)),
                                       _mm256_shuffle_epi32(d0, 0x4E))); /* the bridge */
                av1 = _mm256_add_epi64(av1,
                      _mm256_add_epi64(_mm256_mul_epu32(k1, _mm256_srli_epi64(k1, 32)),
                                       _mm256_shuffle_epi32(d1, 0x4E)));
            }
            _mm256_storeu_si256((__m256i *)(void *)acc,       av0);
            _mm256_storeu_si256((__m256i *)(void *)(acc + 4), av1);
            _mm256_storeu_si256((__m256i *)(void *)w,         wv0);
            _mm256_storeu_si256((__m256i *)(void *)(w + 4),   wv1);
        }
#else
        for (i = 0; i < 8; i++) { acc[i] = ACC_INIT[i]; w[i] = FEATHER0[i]; }
        for (s = 0; s < nstripes; s++) {
            const unsigned char *q = p + (s << 6);
            for (i = 0; i < 8; i++) {
                uint64_t dv = rd64(q + 8 * i);
                uint64_t kv = dv ^ w[i];
                w[i]      += FSTEP[i];
                acc[i ^ 1] += dv;          /* the bridge: the whole instrument answers */
                acc[i]     += weigh(kv);
            }
        }
#endif
        /* the last, short handful of marks: one more stripe, overlapping,
           on a feather no earlier mark has touched */
        if (len & 63) {
            const unsigned char *q = end - 64;
            for (i = 0; i < 8; i++) {
                uint64_t dv = rd64(q + 8 * i);
                uint64_t kv = dv ^ w[i];
                acc[i ^ 1] += dv;
                acc[i]     += weigh(kv);
            }
        }

        /* the storm reads the final shivering pattern, once, and writes eight notes */
        h  = (uint64_t)len * ACC_INIT[0];
        h += fold128(acc[0] ^ FEATHER0[0], acc[1] ^ FEATHER0[1]);
        h += fold128(acc[2] ^ FEATHER0[2], acc[3] ^ FEATHER0[3]);
        h += fold128(acc[4] ^ FEATHER0[4], acc[5] ^ FEATHER0[5]);
        h += fold128(acc[6] ^ FEATHER0[6], acc[7] ^ FEATHER0[7]);
        return storm(h, (uint64_t)len);
    }
}
