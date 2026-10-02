#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the markings printed on the sheet: compile-time only, nothing retained ---- */
#define FOLD_P1 0x9E3779B185EBCA87ULL
#define FOLD_P2 0xC2B2AE3D27D4EB4FULL
#define FOLD_P3 0x165667B19E3779F9ULL
#define FOLD_P4 0x85EBCA77C2B2AE63ULL
#define FOLD_P5 0x27D4EB2F165667C5ULL

static inline uint64_t fold_rotl(uint64_t x, uint64_t n)
{
    unsigned s = (unsigned)(n & 63);
    return (x << s) | (x >> ((64u - s) & 63));
}

/* 32x32 -> 64 press, the NH/UMAC primitive: one multiply per 8-byte corner */
static inline uint64_t m32(uint64_t a, uint64_t b)
{
    return (uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b;
}

/* ---- the water's edge: thin the wad small, small, small, to one coin ----
   each bird's reading passes through its own multiply and rotate before the
   final avalanche, so a difference confined to one bird's high bits still
   reaches the low bits of the coin. Nothing here is kept. */
static inline uint64_t waters_edge(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                                   uint64_t len)
{
    uint64_t h = len * FOLD_P5;              /* the one small suitcase */
    h = (h ^ a) * FOLD_P1; h = fold_rotl(h, 29);
    h = (h ^ b) * FOLD_P2; h = fold_rotl(h, 31);
    h = (h ^ c) * FOLD_P3; h = fold_rotl(h, 37);
    h = (h ^ d) * FOLD_P4;
    h ^= h >> 33; h *= FOLD_P2;               /* small */
    h ^= h >> 29; h *= FOLD_P3;               /* smaller */
    h ^= h >> 32;                             /* one hard dense corner */
    return h;
}

/* ---- scrap path (SEED 1, literal): the pile is too small for the big sheet,
   so no birds are set up. One crease per mark, the angle set by the mark AND
   the crease left by the fold before it. ---- */
static uint64_t scrap_fold(const unsigned char *p, size_t len)
{
    uint64_t h = 0x1BD11BDAA9FC1A22ULL ^ ((uint64_t)len * FOLD_P5);
    uint64_t crease = 0xCBF29CE484222325ULL;
    size_t i;
    for (i = 0; i < len; i++) {
        uint64_t mark = p[i];
        crease = fold_rotl(crease + mark + FOLD_P3, crease ^ mark);
        h = (h ^ crease) * FOLD_P1;
    }
    h ^= h >> 33; h *= FOLD_P2;
    h ^= h >> 29; h *= FOLD_P3;
    h ^= h >> 32;
    return h;
}

#if defined(__AVX2__)
/* ---- one fold of the sheet: four corners, four birds, all four beaks at once,
   then refold tighter by exactly how much they disagree. ---- */
static inline void beak_press(__m256i v, __m256i K, const __m256i KB,
                              __m256i *G, __m256i *C)
{
    __m256i hi = _mm256_srli_epi64(v, 32);
    __m256i a  = _mm256_add_epi64(v, K);    /* low 32 bits: corner + sheet marking */
    __m256i b  = _mm256_add_epi64(hi, KB);  /* low 32 bits: corner + this bird's beak */
    __m256i pr = _mm256_xor_si256(_mm256_mul_epu32(a, b), v); /* the mark itself survives */
    __m256i g  = _mm256_add_epi64(*G, pr);  /* four gauge readings, independent */

    /* all four beaks at once: lane j = bird j against bird j+1.
       all four agree  <=>  d is zero in every lane. */
    __m256i s = _mm256_permute4x64_epi64(g, 0x39);
    __m256i d = _mm256_xor_si256(g, s);

    /* refold tighter, by exactly the disagreement. d == 0 is the identity:
       a crease the birds already agree on gets no extra folding at all. */
    __m256i c = _mm256_xor_si256(*C, d);
#if defined(__AVX512VL__) && defined(__AVX512F__)
    *C = _mm256_rolv_epi64(c, d);
#else
    {
        __m256i sh = _mm256_and_si256(d, _mm256_set1_epi64x(63));
        *C = _mm256_or_si256(
                 _mm256_sllv_epi64(c, sh),
                 _mm256_srlv_epi64(c, _mm256_sub_epi64(_mm256_set1_epi64x(64), sh)));
    }
#endif
    *G = g;
}
#endif /* __AVX2__ */

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* the native's own regime test: is the sheet large enough for this pile? */
    if (len < 32)
        return scrap_fold(data, len);

#if defined(__AVX2__)
    {
        /* four beaks, four distinct sheet-marking advances (all odd, all distinct
           in their low 32 bits, so position is scrambled on purpose) */
        const __m256i KB  = _mm256_setr_epi64x((long long)0x2545F4914F6CDD1DULL,
                                               (long long)0x9E3779B97F4A7C15ULL,
                                               (long long)0xBF58476D1CE4E5B9ULL,
                                               (long long)0x94D049BB133111EBULL);
        const __m256i DEL = _mm256_setr_epi64x((long long)0x0A2127A3B5E0D1F7ULL,
                                               (long long)0x1B3C5D7F91A3B5C7ULL,
                                               (long long)0x2C4D6E8FA1B3C5D9ULL,
                                               (long long)0x3D5E7F91B3C5D7EBULL);
        const __m256i D2 = _mm256_add_epi64(DEL, DEL);

        /* the sheet is already doubled, so one press sets a crease in each ply */
        __m256i G0 = _mm256_setr_epi64x((long long)0x1BD11BDAA9FC1A22ULL,
                                        (long long)0xCBF29CE484222325ULL,
                                        (long long)0x84222325CBF29CE4ULL,
                                        (long long)0xA9FC1A221BD11BDAULL);
        __m256i C0 = _mm256_setr_epi64x((long long)0x6A09E667F3BCC908ULL,
                                        (long long)0xBB67AE8584CAA73BULL,
                                        (long long)0x3C6EF372FE94F82BULL,
                                        (long long)0xA54FF53A5F1D36F1ULL);
        __m256i K0 = _mm256_setr_epi64x((long long)0x510E527FADE682D1ULL,
                                        (long long)0x9B05688C2B3E6C1FULL,
                                        (long long)0x1F83D9ABFB41BD6BULL,
                                        (long long)0x5BE0CD19137E2179ULL);
        __m256i G1 = _mm256_xor_si256(G0, KB);
        __m256i C1 = _mm256_add_epi64(C0, DEL);
        __m256i K1 = _mm256_add_epi64(K0, DEL);

        const unsigned char *restrict p = data;
        size_t n = len;

        while (n >= 64) {
            __m256i v0 = _mm256_loadu_si256((const __m256i *)(const void *)(p));
            __m256i v1 = _mm256_loadu_si256((const __m256i *)(const void *)(p + 32));
            beak_press(v0, K0, KB, &G0, &C0);
            beak_press(v1, K1, KB, &G1, &C1);
            K0 = _mm256_add_epi64(K0, D2);   /* the markings advance with position */
            K1 = _mm256_add_epi64(K1, D2);
            p += 64; n -= 64;
        }
        if (n >= 32) {
            beak_press(_mm256_loadu_si256((const __m256i *)(const void *)p),
                       K0, KB, &G0, &C0);
            p += 32; n -= 32;
        }
        if (n) {
            /* the scrap trimmed off: fold the last 32 marks again, with the
               trim amount pressed into the markings so the overlap is not free */
            __m256i vt = _mm256_loadu_si256((const __m256i *)(const void *)(data + len - 32));
            beak_press(vt, _mm256_add_epi64(K1, _mm256_set1_epi64x((long long)n)),
                       KB, &G1, &C1);
        }

        /* carry the whole wad to the water's edge: gauges crossed with the other
           ply's creases, so neither the readings nor the refoldings can be ignored */
        {
            __m256i W = _mm256_add_epi64(_mm256_xor_si256(G0, C1),
                                         _mm256_xor_si256(G1, C0));
            uint64_t w[4];
            _mm256_storeu_si256((__m256i *)(void *)w, W);
            return waters_edge(w[0], w[1], w[2], w[3], (uint64_t)len);
        }
    }
#else
    /* same mechanism, four birds by hand (no SIMD available) */
    {
        uint64_t G[4] = { 0x1BD11BDAA9FC1A22ULL, 0xCBF29CE484222325ULL,
                          0x84222325CBF29CE4ULL, 0xA9FC1A221BD11BDAULL };
        uint64_t C[4] = { 0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
                          0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL };
        uint64_t K[4] = { 0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
                          0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL };
        const uint64_t KB[4] = { 0x2545F4914F6CDD1DULL, 0x9E3779B97F4A7C15ULL,
                                 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL };
        const uint64_t DL[4] = { 0x0A2127A3B5E0D1F7ULL, 0x1B3C5D7F91A3B5C7ULL,
                                 0x2C4D6E8FA1B3C5D9ULL, 0x3D5E7F91B3C5D7EBULL };
        size_t off = 0;
        uint64_t w[4], g[4], d[4];
        int j;
        while (off + 32 <= len) {
            memcpy(w, data + off, 32);
            for (j = 0; j < 4; j++)
                g[j] = G[j] + (m32(w[j] + K[j], (w[j] >> 32) + KB[j]) ^ w[j]);
            for (j = 0; j < 4; j++) d[j] = g[j] ^ g[(j + 1) & 3];
            for (j = 0; j < 4; j++) {
                C[j] = fold_rotl(C[j] ^ d[j], d[j]);
                G[j] = g[j];
                K[j] += DL[j];
            }
            off += 32;
        }
        if (off < len) {
            uint64_t rem = (uint64_t)(len - off);
            memcpy(w, data + len - 32, 32);
            for (j = 0; j < 4; j++)
                g[j] = G[j] + (m32(w[j] + K[j] + rem, (w[j] >> 32) + KB[j]) ^ w[j]);
            for (j = 0; j < 4; j++) d[j] = g[j] ^ g[(j + 1) & 3];
            for (j = 0; j < 4; j++) { C[j] = fold_rotl(C[j] ^ d[j], d[j]); G[j] = g[j]; }
        }
        return waters_edge(G[0] ^ C[1], G[1] ^ C[0],
                           G[2] ^ C[3], G[3] ^ C[2], (uint64_t)len);
    }
#endif
}
