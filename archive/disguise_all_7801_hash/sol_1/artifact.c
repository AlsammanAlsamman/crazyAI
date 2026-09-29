#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define HP      0x100000001b3ULL          /* the twist */
#define HBASIS  0xcbf29ce484222325ULL     /* the warm blob you start with */

/* g_R[m] = HP^(64-m) : the composed twists, precomputed once */
static uint64_t g_R[64];
static uint64_t g_Q;                      /* HP^64 */
#if defined(__AVX2__)
static __m256i g_WL[16], g_WH[16];        /* lo/hi 32-bit halves of g_R[0..63] */
#endif

static void kern_init(void)
{
    uint64_t p = 1;
    for (int m = 63; m >= 0; --m) { p *= HP; g_R[m] = p; }
    g_Q = g_R[0];
#if defined(__AVX2__)
    for (int i = 0; i < 16; ++i) {
        uint64_t w0 = g_R[4*i+0], w1 = g_R[4*i+1], w2 = g_R[4*i+2], w3 = g_R[4*i+3];
        g_WL[i] = _mm256_set_epi64x((long long)(w3 & 0xFFFFFFFFULL),
                                    (long long)(w2 & 0xFFFFFFFFULL),
                                    (long long)(w1 & 0xFFFFFFFFULL),
                                    (long long)(w0 & 0xFFFFFFFFULL));
        g_WH[i] = _mm256_set_epi64x((long long)(w3 >> 32), (long long)(w2 >> 32),
                                    (long long)(w1 >> 32), (long long)(w0 >> 32));
    }
#endif
}
__attribute__((constructor)) static void kern_ctor(void) { kern_init(); }

static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

#if defined(__AVX2__)
static inline uint64_t hsum_epu64(__m256i v)
{
    __m128i s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
    return (uint64_t)_mm_cvtsi128_si64(s);
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    if (__builtin_expect(g_Q == 0, 0)) kern_init();   /* safety net; ctor normally did it */

    uint64_t h = HBASIS;
    size_t i = 0;

#if defined(__AVX2__)
    const __m256i m32 = _mm256_set1_epi64x(0xFFFFFFFFLL);

    while (len - i >= 64) {
        const unsigned char *p = data + i;
        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
        __m256i b0 = _mm256_setzero_si256(), b1 = _mm256_setzero_si256();
        for (int k = 0; k < 16; k += 2) {
            __m256i v0 = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128((int)ld32(p + 4*k)));
            __m256i v1 = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128((int)ld32(p + 4*k + 4)));
            a0 = _mm256_add_epi64(a0, _mm256_mul_epu32(v0, g_WL[k]));
            b0 = _mm256_add_epi64(b0, _mm256_mul_epu32(v0, g_WH[k]));
            a1 = _mm256_add_epi64(a1, _mm256_mul_epu32(v1, g_WL[k+1]));
            b1 = _mm256_add_epi64(b1, _mm256_mul_epu32(v1, g_WH[k+1]));
        }
        uint64_t sl = hsum_epu64(_mm256_add_epi64(a0, a1));
        uint64_t sh = hsum_epu64(_mm256_add_epi64(b0, b1));
        h = h * g_Q + sl + (sh << 32);            /* the one blob advances */
        i += 64;
    }

    {
        size_t r = len - i;                        /* r <= 63 */
        if (r) {
            const unsigned char *p = data + i;
            const uint64_t *W = g_R + (64 - r);    /* W[j] = HP^(r-j) */
            uint64_t S = 0;
            size_t j = 0;
            if (r >= 4) {
                __m256i a = _mm256_setzero_si256(), b = _mm256_setzero_si256();
                for (; j + 4 <= r; j += 4) {
                    __m256i w = _mm256_loadu_si256((const __m256i *)(const void *)(W + j));
                    __m256i v = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128((int)ld32(p + j)));
                    a = _mm256_add_epi64(a, _mm256_mul_epu32(v, _mm256_and_si256(w, m32)));
                    b = _mm256_add_epi64(b, _mm256_mul_epu32(v, _mm256_srli_epi64(w, 32)));
                }
                S = hsum_epu64(a) + (hsum_epu64(b) << 32);
            }
            for (; j < r; ++j) S += (uint64_t)p[j] * W[j];
            h = h * g_R[64 - r] + S;
        }
    }
#else
    for (; i + 8 <= len; i += 8) {
        const unsigned char *p = data + i;
        uint64_t s0 = (uint64_t)p[0]*g_R[56] + (uint64_t)p[1]*g_R[57];
        uint64_t s1 = (uint64_t)p[2]*g_R[58] + (uint64_t)p[3]*g_R[59];
        uint64_t s2 = (uint64_t)p[4]*g_R[60] + (uint64_t)p[5]*g_R[61];
        uint64_t s3 = (uint64_t)p[6]*g_R[62] + (uint64_t)p[7]*g_R[63];
        h = h * g_R[56] + ((s0 + s1) + (s2 + s3));
    }
    for (; i < len; ++i) h = (h + data[i]) * HP;   /* press, twist, next */
#endif

    /* "give it a handful of extra twists at the end" */
    h ^= (uint64_t)len;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 29; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 32;
    return h;
}
