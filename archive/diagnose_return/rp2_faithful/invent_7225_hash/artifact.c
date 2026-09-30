#include <stdint.h>
#include <stddef.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROTL64(x, r) (((uint64_t)(x) << (r)) | ((uint64_t)(x) >> (64 - (r))))

/* the cup is never pure: every cup starts already tainted */
static const uint64_t WINE[16] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL, 0x2545F4914F6CDD1DULL,
    0xD1B54A32D192ED03ULL, 0xA5CB3B6D4CF0E9C7ULL, 0x8EBC6AF09C88C6E3ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL, 0x85EBCA77C2B2AE63ULL, 0x9E3779B185EBCA87ULL,
    0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL, 0xEB44ACCAB455D37DULL, 0xD6E8FEB86659FD93ULL
};
/* each cup sits at its own tilt; all odd, so a tilt visits every bit */
static const unsigned TILT[16] = { 11,23,37,53, 13,29,41,59, 17,31,43,61, 19,7,47,5 };

/* ---- SEED 2: the seven organs. bend, throw away half, keep what refuses to sit still ---- */
static inline uint64_t organs7(uint64_t h)
{
    h ^= h >> 32; h = ROTL64(h, 27);   /* organ 1: discards 32 -- half of the whole   */
    h ^= h >> 16; h = ROTL64(h, 13);   /* organ 2: half of what came before           */
    h ^= h >>  8; h = ROTL64(h, 41);   /* organ 3                                     */
    h += h << 19;                      /* organ 4: the organ that COMPENSATES (carry) */
    h ^= h >>  4; h = ROTL64(h,  7);   /* organ 5                                     */
    h ^= h >>  2; h = ROTL64(h, 53);   /* organ 6                                     */
    h ^= h >>  1; h = ROTL64(h, 31);   /* organ 7 -- the ladder is spent              */
    return h;                          /* small enough to knot into a collar          */
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *__restrict p = data;
    uint64_t h;
    size_t i, n;

    /* === regime 1: a small pile -- one drinker, one cup, one unbroken pour === */
    if (len < 32) {
        uint64_t c = WINE[0];
        for (i = 0; i < len; i++)
            c = ROTL64(c + (uint64_t)p[i], 11);
        c = ROTL64(c + (uint64_t)len, 31);
        return organs7(c);
    }

#if defined(__AVX2__)
    /* === regime 3: a caravan -- the whole tavern, sixteen cups, four poured at once === */
    if (len >= 256) {
        const __m256i T0L = _mm256_setr_epi64x(11,23,37,53);
        const __m256i T0R = _mm256_setr_epi64x(53,41,27,11);
        const __m256i T1L = _mm256_setr_epi64x(13,29,41,59);
        const __m256i T1R = _mm256_setr_epi64x(51,35,23, 5);
        const __m256i T2L = _mm256_setr_epi64x(17,31,43,61);
        const __m256i T2R = _mm256_setr_epi64x(47,33,21, 3);
        const __m256i T3L = _mm256_setr_epi64x(19, 7,47, 5);
        const __m256i T3R = _mm256_setr_epi64x(45,57,17,59);
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(const void *)(WINE +  0));
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(const void *)(WINE +  4));
        __m256i a2 = _mm256_loadu_si256((const __m256i *)(const void *)(WINE +  8));
        __m256i a3 = _mm256_loadu_si256((const __m256i *)(const void *)(WINE + 12));
        uint64_t cups[16];
        int k;

        n = len & ~(size_t)15;
        for (i = 0; i < n; i += 16) {
            __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(p + i));
            /* each mark poured into its own cup: byte j -> cup j, still one by one */
            a0 = _mm256_add_epi64(a0, _mm256_cvtepu8_epi64(b));
            a1 = _mm256_add_epi64(a1, _mm256_cvtepu8_epi64(_mm_srli_si128(b,  4)));
            a2 = _mm256_add_epi64(a2, _mm256_cvtepu8_epi64(_mm_srli_si128(b,  8)));
            a3 = _mm256_add_epi64(a3, _mm256_cvtepu8_epi64(_mm_srli_si128(b, 12)));
            /* the tasted colour becomes the colour waiting: each cup at its own tilt */
            a0 = _mm256_or_si256(_mm256_sllv_epi64(a0,T0L), _mm256_srlv_epi64(a0,T0R));
            a1 = _mm256_or_si256(_mm256_sllv_epi64(a1,T1L), _mm256_srlv_epi64(a1,T1R));
            a2 = _mm256_or_si256(_mm256_sllv_epi64(a2,T2L), _mm256_srlv_epi64(a2,T2R));
            a3 = _mm256_or_si256(_mm256_sllv_epi64(a3,T3L), _mm256_srlv_epi64(a3,T3R));
        }
        _mm256_storeu_si256((__m256i *)(void *)(cups +  0), a0);
        _mm256_storeu_si256((__m256i *)(void *)(cups +  4), a1);
        _mm256_storeu_si256((__m256i *)(void *)(cups +  8), a2);
        _mm256_storeu_si256((__m256i *)(void *)(cups + 12), a3);

        for (; i < len; i++)                        /* leftover marks -> the first cup */
            cups[0] = ROTL64(cups[0] + (uint64_t)p[i], 11);
        h = cups[0];
        for (k = 1; k < 16; k++)                    /* the cups poured into one another */
            h = ROTL64(h + cups[k], TILT[k]);
        h = ROTL64(h + (uint64_t)len, 31);          /* the pile ends: its tally is a mark */
        return organs7(h);
    }
#endif

    /* === regime 2: the bar -- eight cups, the pour goes down the row === */
    {
        uint64_t c0 = WINE[0], c1 = WINE[1], c2 = WINE[2], c3 = WINE[3];
        uint64_t c4 = WINE[4], c5 = WINE[5], c6 = WINE[6], c7 = WINE[7];

        n = len & ~(size_t)7;
        for (i = 0; i < n; i += 8) {
            c0 = ROTL64(c0 + (uint64_t)p[i + 0], 11);
            c1 = ROTL64(c1 + (uint64_t)p[i + 1], 23);
            c2 = ROTL64(c2 + (uint64_t)p[i + 2], 37);
            c3 = ROTL64(c3 + (uint64_t)p[i + 3], 53);
            c4 = ROTL64(c4 + (uint64_t)p[i + 4], 13);
            c5 = ROTL64(c5 + (uint64_t)p[i + 5], 29);
            c6 = ROTL64(c6 + (uint64_t)p[i + 6], 41);
            c7 = ROTL64(c7 + (uint64_t)p[i + 7], 59);
        }
        for (; i < len; i++)
            c0 = ROTL64(c0 + (uint64_t)p[i], 11);

        h = c0;
        h = ROTL64(h + c1, 23); h = ROTL64(h + c2, 37); h = ROTL64(h + c3, 53);
        h = ROTL64(h + c4, 13); h = ROTL64(h + c5, 29); h = ROTL64(h + c6, 41);
        h = ROTL64(h + c7, 59);
        h = ROTL64(h + (uint64_t)len, 31);
        return organs7(h);
    }
}
