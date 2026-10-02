#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define GOLD 0x9E3779B97F4A7C15ULL   /* the inkwell's never-ending drip */

/* eight odd drops, one per string */
static const uint64_t KW[8] = {
    0xa0761d6478bd642fULL, 0xe7037ed1a0b428dbULL,
    0x8ebc6af09c88c6e3ULL, 0x589965cc75374cc3ULL,
    0x1d8e4e27c47d124fULL, 0xeb44accab455d165ULL,
    0xc9bc5a84fe4bbfd4ULL, 0xd6e8feb86659fd93ULL
};

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* 128-bit multiply folded to 64 -- used ONLY by the storm */
static inline uint64_t mum(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
#else
    uint64_t lo = a * b;
    uint64_t a0 = a & 0xffffffffULL, a1 = a >> 32;
    uint64_t b0 = b & 0xffffffffULL, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = p10 + (p00 >> 32) + (p01 & 0xffffffffULL);
    uint64_t hi  = p11 + (mid >> 32) + (p01 >> 32);
    return lo ^ hi;
#endif
}

static inline uint64_t rd8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

/* ONE pluck: fresh feather, nearest string, then the bridge shudders on */
static inline void pluck(uint64_t *v, uint64_t q, uint64_t w, uint64_t c) {
    unsigned j = (unsigned)(q & 7u);
    uint64_t f = w ^ (KW[j] + c);                 /* no two marks share a feather */
    v[j] = rotl64(v[j] ^ f, 23) + f;              /* same force, every time, no multiply */
    v[(j + 1u) & 7u] += rotl64(v[j], 13);         /* bound under one bridge */
}

/* THE STORM: a single pass across the eight strings, each read exactly once,
   dissolved into eight notes (64 bits). All avalanche is bought here. */
static inline uint64_t storm(const uint64_t *v, uint64_t len) {
    uint64_t h = len ^ GOLD;
    h = mum(h ^ v[0], v[1] ^ KW[0]);
    h = mum(h ^ v[2], v[3] ^ KW[1]);
    h = mum(h ^ v[4], v[5] ^ KW[2]);
    h = mum(h ^ v[6], v[7] ^ KW[3]);
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;     /* splitmix64 finaliser */
    h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t v[8];
    unsigned k;
    for (k = 0; k < 8; k++) v[k] = KW[k] + GOLD * (uint64_t)(k + 1u);

    size_t   i   = 0;
    uint64_t q   = 0;
    uint64_t ctr = GOLD;

    /* ---- regime check: is the pile long enough to span all eight strings? ---- */
    if (len >= 64) {
#if defined(__AVX2__)
        __m256i A = _mm256_loadu_si256((const __m256i *)&v[0]);
        __m256i B = _mm256_loadu_si256((const __m256i *)&v[4]);
        const __m256i KA = _mm256_setr_epi64x((long long)KW[0], (long long)KW[1],
                                              (long long)KW[2], (long long)KW[3]);
        const __m256i KB = _mm256_setr_epi64x((long long)KW[4], (long long)KW[5],
                                              (long long)KW[6], (long long)KW[7]);
        const __m256i G  = _mm256_set1_epi64x((long long)GOLD);
        __m256i C        = _mm256_set1_epi64x((long long)GOLD);

        for (; i + 64 <= len; i += 64) {
            __m256i x0 = _mm256_loadu_si256((const __m256i *)(p + i));
            __m256i x1 = _mm256_loadu_si256((const __m256i *)(p + i + 32));
            __m256i f0 = _mm256_xor_si256(x0, _mm256_add_epi64(KA, C)); /* feathers */
            __m256i f1 = _mm256_xor_si256(x1, _mm256_add_epi64(KB, C));
            A = _mm256_xor_si256(A, f0);                                /* plucks   */
            B = _mm256_xor_si256(B, f1);
            A = _mm256_or_si256(_mm256_slli_epi64(A, 23),
                                _mm256_srli_epi64(A, 41));
            A = _mm256_add_epi64(A, B);                                 /* bridge   */
            B = _mm256_or_si256(_mm256_slli_epi64(B, 31),
                                _mm256_srli_epi64(B, 33));
            B = _mm256_xor_si256(B, A);
            B = _mm256_permute4x64_epi64(B, _MM_SHUFFLE(2, 1, 0, 3));   /* travels  */
            C = _mm256_add_epi64(C, G);
        }
        _mm256_storeu_si256((__m256i *)&v[0], A);
        _mm256_storeu_si256((__m256i *)&v[4], B);
        q   = (uint64_t)(i >> 3);
        ctr = GOLD + (uint64_t)(i >> 6) * GOLD;
#else
        /* no wide strings available: pluck them one at a time, same machine */
        for (; i + 8 <= len; i += 8, q++, ctr += GOLD)
            pluck(v, q, rd8(p + i), ctr);
#endif
    }

    /* short pile, and the tail of a long one: plucked by hand */
    for (; i + 8 <= len; i += 8, q++, ctr += GOLD)
        pluck(v, q, rd8(p + i), ctr);

    if (i < len) {                      /* 1..7 remaining marks, no over-read */
        size_t r = len - i, b;
        uint64_t t = (uint64_t)r << 56;
        for (b = 0; b < r; b++) t |= (uint64_t)p[i + b] << (8u * b);
        pluck(v, q, t, ctr);
    }

    return storm(v, (uint64_t)len);     /* the storm, once; then the pickers */
}
