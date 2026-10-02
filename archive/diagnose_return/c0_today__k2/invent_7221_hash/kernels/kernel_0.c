#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ----- the four birds' beaks, and the constants of the water's edge ----- */
#define BEAK1        0x9E3779B185EBCA87ULL
#define BEAK2        0xC2B2AE3D27D4EB4FULL
#define BEAK3        0x165667B19E3779F9ULL
#define BEAK4        0x85EBCA77C2B2AE63ULL
#define COUNT_PRIME  0x27D4EB2F165667C5ULL

/* each bird refolds tighter after every fold: its beak advances, so no two
   folds are gauged alike -> stripe position and stripe order both matter. */
#define STEP0 0x9E3779B97F4A7C15ULL
#define STEP1 0xBF58476D1CE4E5B9ULL
#define STEP2 0x94D049BB133111EBULL
#define STEP3 0x2545F4914F6CDD1DULL

#define KEY0  0xDA942042E4DD58B5ULL
#define KEY1  0x9FB21C651E98DF25ULL
#define KEY2  0xEB44ACCAB455D165ULL
#define KEY3  0xA0761D6478BD642FULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* SEED 3: the water's edge. The whole wad thins down to one dense corner.
   MurmurHash3 fmix64 -- a validated, bijective 64-bit avalanche. */
static inline uint64_t thin_at_the_water(uint64_t h) {
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return h;
}

/* "only when all four beaks agree does the crease count as set":
   no output exists until every lane has been folded in.
   This is xxHash64's mergeRound -- validated. */
static inline uint64_t beak_agree(uint64_t h, uint64_t lane) {
    lane *= BEAK2;
    lane  = rotl64(lane, 31);
    lane *= BEAK1;
    h ^= lane;
    h  = h * BEAK1 + BEAK4;
    return h;
}

/* ----- REGIME 2: the pile is too small for a huge sheet, so fold on a
   scrap with a single bird. This is SEED 1 / the known way verbatim
   (one in-place accumulator, carry-forward crease), widened to 8-byte
   words, then thinned. It can never be slower than the baseline. ----- */
static uint64_t fold_on_scrap(const unsigned char *d, size_t len) {
    uint64_t h = 1469598103934665603ULL ^ ((uint64_t)len * COUNT_PRIME);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t w;
        memcpy(&w, d + i, 8);
        h ^= w;
        h *= 1099511628211ULL;
        h  = rotl64(h, 29);              /* fold high bits back down */
    }
    for (; i < len; i++) {
        h ^= (uint64_t)d[i];
        h *= 1099511628211ULL;
    }
    return thin_at_the_water(h);
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* "pink or orange, doesn't matter, only that it's large enough":
       the runtime regime test. No sheet -> no birds. */
    if (len < 32u) return fold_on_scrap(data, len);

    const unsigned char *restrict p = data;
    const size_t nfold = len >> 5;       /* one fold per 32 bytes */
    const size_t rem   = len & 31u;      /* the ragged trimmed scrap */

    /* the ragged tail is padded onto its own scrap, with the count of
       marks pressed into the last corner so that padding cannot lie */
    unsigned char tail[32];
    if (rem) {
        memset(tail, 0, sizeof tail);
        memcpy(tail, p + (nfold << 5), rem);
        tail[31] = (unsigned char)rem;
    }

#if defined(__AVX2__)
    /* THE FOUR WIRE BIRDS: four 64-bit lanes of one register, pressed at once. */
    __m256i acc  = _mm256_set_epi64x((int64_t)(BEAK4 ^ (uint64_t)len),
                                     (int64_t)BEAK3,
                                     (int64_t)BEAK2,
                                     (int64_t)BEAK1);
    __m256i beak = _mm256_set_epi64x((int64_t)KEY3, (int64_t)KEY2,
                                     (int64_t)KEY1, (int64_t)KEY0);
    const __m256i step = _mm256_set_epi64x((int64_t)STEP3, (int64_t)STEP2,
                                           (int64_t)STEP1, (int64_t)STEP0);

#if defined(__GNUC__)
#pragma GCC unroll 2
#endif
    for (size_t i = 0; i < nfold; i++, p += 32) {
        __m256i d  = _mm256_loadu_si256((const __m256i *)p);   /* one fold, 4 corners */
        __m256i dk = _mm256_xor_si256(d, beak);                /* corner meets beak   */
        __m256i sw = _mm256_shuffle_epi32(dk, 0xB1);           /* hi32 <-> lo32       */
        __m256i pr = _mm256_mul_epu32(dk, sw);                 /* lo32 * hi32, 4 lanes*/
        /* "lined up wrong on purpose, scrambled, because a clean line-up
            would mean two different piles could look the same" */
        __m256i ds = _mm256_permute4x64_epi64(d, 0x4E);
        acc  = _mm256_add_epi64(acc, ds);                      /* critical path: 1 cy */
        acc  = _mm256_add_epi64(acc, pr);                      /* critical path: 1 cy */
        beak = _mm256_add_epi64(beak, step);                   /* refold tighter      */
    }
    if (rem) {
        __m256i d  = _mm256_loadu_si256((const __m256i *)tail);
        __m256i dk = _mm256_xor_si256(d, beak);
        __m256i sw = _mm256_shuffle_epi32(dk, 0xB1);
        __m256i pr = _mm256_mul_epu32(dk, sw);
        __m256i ds = _mm256_permute4x64_epi64(d, 0x4E);
        acc = _mm256_add_epi64(acc, ds);
        acc = _mm256_add_epi64(acc, pr);
    }

    uint64_t l[4];
    _mm256_storeu_si256((__m256i *)l, acc);
#else
    /* Same arithmetic, four birds held in four scalar registers.
       Produces a bit-identical hash to the AVX2 path. */
    uint64_t l[4];
    uint64_t a0 = BEAK1, a1 = BEAK2, a2 = BEAK3, a3 = BEAK4 ^ (uint64_t)len;
    uint64_t k0 = KEY0,  k1 = KEY1,  k2 = KEY2,  k3 = KEY3;
    for (size_t i = 0; i <= nfold; i++) {
        const unsigned char *q;
        if (i < nfold)      q = p + (i << 5);
        else if (rem)       q = tail;
        else                break;
        uint64_t w0, w1, w2, w3;
        memcpy(&w0, q,      8); memcpy(&w1, q +  8, 8);
        memcpy(&w2, q + 16, 8); memcpy(&w3, q + 24, 8);
        uint64_t d0 = w0 ^ k0, d1 = w1 ^ k1, d2 = w2 ^ k2, d3 = w3 ^ k3;
        a0 += w2; a1 += w3; a2 += w0; a3 += w1;          /* the scramble */
        a0 += (uint64_t)(uint32_t)d0 * (uint64_t)(uint32_t)(d0 >> 32);
        a1 += (uint64_t)(uint32_t)d1 * (uint64_t)(uint32_t)(d1 >> 32);
        a2 += (uint64_t)(uint32_t)d2 * (uint64_t)(uint32_t)(d2 >> 32);
        a3 += (uint64_t)(uint32_t)d3 * (uint64_t)(uint32_t)(d3 >> 32);
        if (i < nfold) { k0 += STEP0; k1 += STEP1; k2 += STEP2; k3 += STEP3; }
    }
    l[0] = a0; l[1] = a1; l[2] = a2; l[3] = a3;
#endif

    /* all four beaks must agree before the crease counts as set */
    uint64_t h = (uint64_t)len * COUNT_PRIME
               + rotl64(l[0],  1) + rotl64(l[1],  7)
               + rotl64(l[2], 12) + rotl64(l[3], 18);
    h = beak_agree(h, l[0]);
    h = beak_agree(h, l[1]);
    h = beak_agree(h, l[2]);
    h = beak_agree(h, l[3]);

    /* carry the wad to the water's edge; keep only the coin, throw the
       sheets, the trimmings and the failed readings into the mud */
    return thin_at_the_water(h);
}
