#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- anchors: where each wire is fused into the riverbed (all odd) ---- */
#define A0 0x9E3779B185EBCA87ULL
#define A1 0xC2B2AE3D27D4EB4FULL
#define A2 0x165667B19E3779F9ULL
#define A3 0x85EBCA77C2B2AE63ULL
#define A4 0x27D4EB2F165667C5ULL
#define A5 0xA0761D6478BD642FULL
#define A6 0xE7037ED1A0B428DBULL
#define A7 0x8EBC6AF09C88C6E3ULL

static const uint64_t cave_anchor[8] = { A0, A1, A2, A3, A4, A5, A6, A7 };

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

/* two wires crossing: 64x64 -> 128, folded back to 64 */
static inline uint64_t fold128(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    unsigned __int128 pr = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)pr ^ (uint64_t)(pr >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32, bl = (uint32_t)b, bh = b >> 32;
    uint64_t ll = al * bl, hl = ah * bl, lh = al * bh, hh = ah * bh;
    uint64_t cross = (ll >> 32) + (uint32_t)hl + lh;
    uint64_t upper = (hl >> 32) + (cross >> 32) + hh;
    uint64_t lower = (cross << 32) | (uint32_t)ll;
    return lower ^ upper;
#endif
}

/* the shadows settle, then stop chittering: exactly two multiply rounds */
static inline uint64_t settle(uint64_t h) {
    h ^= h >> 33;
    h *= 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 29;
    h *= 0x9E3779B185EBCA87ULL;
    h ^= h >> 32;
    return h;
}

static inline uint64_t rd8(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t rd4(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* ---- small pile: never reaches the wires; the nearest prisoner alone ---- */
static uint64_t cave_small(const unsigned char *p, size_t len) {
    uint64_t h = (uint64_t)len * A0 + A7;
    if (len == 0) return settle(h ^ A1);
    if (len >= 32) {
        h += fold128(rd8(p)        ^ A1, rd8(p + 8)        ^ A2);
        h += fold128(rd8(p + 16)   ^ A3, rd8(p + 24)       ^ A4);
        h += fold128(rd8(p+len-32) ^ A5, rd8(p + len - 24) ^ A6);
        h += fold128(rd8(p+len-16) ^ A7, rd8(p + len - 8)  ^ A0);
    } else if (len >= 16) {
        h += fold128(rd8(p)        ^ A1, rd8(p + 8)       ^ A2);
        h += fold128(rd8(p+len-16) ^ A3, rd8(p + len - 8) ^ A4);
    } else if (len >= 8) {
        h += fold128(rd8(p) ^ A1, rd8(p + len - 8) ^ A2);
    } else if (len >= 4) {
        uint64_t x = ((uint64_t)rd4(p) << 32) | (uint64_t)rd4(p + len - 4);
        h += fold128(x ^ A1, ((uint64_t)len * A2) ^ A3);
    } else {
        uint64_t x = ((uint64_t)p[0] << 16) | ((uint64_t)p[len >> 1] << 24)
                   | ((uint64_t)p[len - 1]) | ((uint64_t)len << 8);
        h += fold128(x ^ A1, ((uint64_t)len * A2) ^ A3);
    }
    return settle(h);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    if (len < 64) return cave_small(data, len);   /* guarded fallback: small-pile path */

    const unsigned char * restrict p = data;
    size_t nstripe = len >> 6;                    /* one stripe = one knowing wall */

    /* the chain of prisoners sharpening the pillar */
    uint64_t p0 = A0 ^ (uint64_t)len, p1 = A2 + (uint64_t)len, p2 = A4, p3 = A6;
    uint64_t out[8];

#if defined(__AVX2__)
    const __m256i anchA = _mm256_set_epi64x((long long)A3,(long long)A2,(long long)A1,(long long)A0);
    const __m256i anchB = _mm256_set_epi64x((long long)A7,(long long)A6,(long long)A5,(long long)A4);
    __m256i acc0 = anchA, acc1 = anchB;           /* the wires, anchored */

#define CAVE_STRIPE(PP)                                                        \
    do {                                                                       \
        const unsigned char *q = (PP);                                         \
        __m256i d0 = _mm256_loadu_si256((const __m256i *)(q));                 \
        __m256i d1 = _mm256_loadu_si256((const __m256i *)(q + 32));            \
        uint64_t mk = rd8(q) ^ rd8(q + 40);       /* fed to the nearest one */  \
        uint64_t b0 = p0, b1 = p1, b2 = p2, b3 = p3;  /* last pass's angles */  \
        p0 = rotl64(b0 + mk, 29);                                              \
        p1 = rotl64(b1 ^ b0, 17);                 /* angle, not the mark */     \
        p2 = b2 + rotl64(b1, 41);                                              \
        p3 = rotl64(b3 ^ b2, 7) + b3;             /* the last one's angle */    \
        __m256i av = _mm256_set1_epi64x((long long)p3);                        \
        __m256i k0 = _mm256_xor_si256(d0, _mm256_add_epi64(anchA, av));        \
        __m256i k1 = _mm256_xor_si256(d1, _mm256_add_epi64(anchB, av));        \
        __m256i x0 = _mm256_mul_epu32(k0, _mm256_shuffle_epi32(k0, 0x31));     \
        __m256i x1 = _mm256_mul_epu32(k1, _mm256_shuffle_epi32(k1, 0x31));     \
        acc0 = _mm256_add_epi64(acc0,                                          \
                 _mm256_add_epi64(x0, _mm256_shuffle_epi32(d0, 0x4E)));        \
        acc1 = _mm256_add_epi64(acc1,                                          \
                 _mm256_add_epi64(x1, _mm256_shuffle_epi32(d1, 0x4E)));        \
    } while (0)

    for (size_t s = 0; s < nstripe; s++) { CAVE_STRIPE(p); p += 64; }
    if (len & 63) CAVE_STRIPE(data + len - 64);   /* led down again, overlapping */

    _mm256_storeu_si256((__m256i *)out,       acc0);
    _mm256_storeu_si256((__m256i *)(out + 4), acc1);
#undef CAVE_STRIPE

#else   /* no wide flood in this pit: same algebra, one wire at a time */
    uint64_t acc[8];
    for (int j = 0; j < 8; j++) acc[j] = cave_anchor[j];

#define CAVE_STRIPE(PP)                                                        \
    do {                                                                       \
        const unsigned char *q = (PP);                                         \
        uint64_t w[8]; memcpy(w, q, 64);                                       \
        uint64_t mk = w[0] ^ w[5];                                             \
        uint64_t b0 = p0, b1 = p1, b2 = p2, b3 = p3;                           \
        p0 = rotl64(b0 + mk, 29);                                              \
        p1 = rotl64(b1 ^ b0, 17);                                              \
        p2 = b2 + rotl64(b1, 41);                                              \
        p3 = rotl64(b3 ^ b2, 7) + b3;                                          \
        for (int j = 0; j < 8; j++) {                                          \
            uint64_t k = w[j] ^ (cave_anchor[j] + p3);                         \
            acc[j ^ 1] += w[j];                                                \
            acc[j] += (uint64_t)(uint32_t)k * (uint64_t)(uint32_t)(k >> 32);   \
        }                                                                      \
    } while (0)

    for (size_t s = 0; s < nstripe; s++) { CAVE_STRIPE(p); p += 64; }
    if (len & 63) CAVE_STRIPE(data + len - 64);
    memcpy(out, acc, sizeof(out));
#undef CAVE_STRIPE
#endif

    /* the silhouette: which lean, which stand straight, which cross another */
    uint64_t sil = (uint64_t)len * A0;
    sil += fold128(out[0] ^ A1, out[1] ^ A2);
    sil += fold128(out[2] ^ A3, out[3] ^ A4);
    sil += fold128(out[4] ^ A5, out[5] ^ A6);
    sil += fold128(out[6] ^ A7, out[7] ^ A0);
    sil += fold128(p0 ^ p2, p1 ^ p3);   /* the pillar's own last angle, crossed */
    return settle(sil);                 /* read once, kept alone */
}
