#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#define KP1 0x9E3779B185EBCA87ULL
#define KP2 0xC2B2AE3D27D4EB4FULL
#define KP32 0x9E3779B1ULL

/* 256 bytes of "twist" material: sliding window (0..22), final stripe (16..23),
   scramble (24..31), merge/short paths (0..15). */
static const uint64_t SEC[32] = {
    0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL, 0x85EBCA77C2B2AE63ULL,
    0x27D4EB2F165667C5ULL, 0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL,
    0x8EBC6AF09C88C6E3ULL, 0x589965CC75374CC3ULL, 0x1D8E4E27C47D124FULL, 0xEB44ACCAB455D165ULL,
    0x4D6D2C1A5B7A6E7BULL, 0xF1BBCDCBFA53E0ABULL, 0xB5026F5AA96619E9ULL, 0x2545F4914F6CDD1DULL,
    0x9FB21C651E98DF25ULL, 0x30BE9C1D71F0A5D7ULL, 0x6A09E667F3BCC909ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL, 0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL, 0xCBBB9D5DC1059ED8ULL, 0x629A292A367CD507ULL,
    0x9159015A3070DD17ULL, 0x152FECD8F70E5939ULL, 0x67332667FFC00B31ULL, 0x8EB44A8768581511ULL
};

static inline uint64_t rd8(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t rd4(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

/* 128-bit multiply, folded: the "hard twist" that spreads every bit. */
static inline uint64_t mul_fold(uint64_t a, uint64_t b)
{
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32, bl = (uint32_t)b, bh = b >> 32;
    uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint64_t cr = (ll >> 32) + (uint32_t)lh + (uint32_t)hl;
    uint64_t lo = (ll & 0xFFFFFFFFULL) | (cr << 32);
    uint64_t hi = hh + (lh >> 32) + (hl >> 32) + (cr >> 32);
    return lo ^ hi;
#endif
}

/* Bijective finalizer (moremur): single-bit-in => full avalanche out. */
static inline uint64_t moremur(uint64_t x)
{
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    return x;
}

static inline uint64_t mix16(const unsigned char *p, const uint64_t *s)
{
    return mul_fold(rd8(p) ^ s[0], rd8(p + 8) ^ s[1]);
}

#if defined(__AVX2__)
static inline void acc_stripe(__m256i *a0, __m256i *a1,
                              const unsigned char *q, const uint64_t *k)
{
    __m256i d0 = _mm256_loadu_si256((const __m256i *)q);
    __m256i d1 = _mm256_loadu_si256((const __m256i *)(q + 32));
    __m256i x0 = _mm256_xor_si256(d0, _mm256_loadu_si256((const __m256i *)k));
    __m256i x1 = _mm256_xor_si256(d1, _mm256_loadu_si256((const __m256i *)(k + 4)));
    __m256i p0 = _mm256_mul_epu32(x0, _mm256_srli_epi64(x0, 32));
    __m256i p1 = _mm256_mul_epu32(x1, _mm256_srli_epi64(x1, 32));
    __m256i s0 = _mm256_shuffle_epi32(d0, _MM_SHUFFLE(1, 0, 3, 2)); /* lane i <-> i^1 */
    __m256i s1 = _mm256_shuffle_epi32(d1, _MM_SHUFFLE(1, 0, 3, 2));
    *a0 = _mm256_add_epi64(*a0, _mm256_add_epi64(p0, s0));
    *a1 = _mm256_add_epi64(*a1, _mm256_add_epi64(p1, s1));
}

static inline __m256i scramble256(__m256i a, const uint64_t *s)
{
    __m256i pr, lo, hi;
    a = _mm256_xor_si256(a, _mm256_srli_epi64(a, 47));
    a = _mm256_xor_si256(a, _mm256_loadu_si256((const __m256i *)s));
    pr = _mm256_set1_epi64x((long long)KP32);
    lo = _mm256_mul_epu32(a, pr);
    hi = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), pr);
    return _mm256_add_epi64(lo, _mm256_slli_epi64(hi, 32)); /* == a * KP32 mod 2^64 */
}
#else
static inline void acc_stripe_s(uint64_t *acc, const unsigned char *q, const uint64_t *k)
{
    int i;
    for (i = 0; i < 8; i++) {
        uint64_t d = rd8(q + 8 * i);
        uint64_t x = d ^ k[i];
        acc[i ^ 1] += d;
        acc[i] += (uint64_t)(uint32_t)x * (uint64_t)(uint32_t)(x >> 32);
    }
}
static inline void scramble_s(uint64_t *acc, const uint64_t *s)
{
    int i;
    for (i = 0; i < 8; i++) {
        uint64_t a = acc[i];
        a ^= a >> 47;
        a ^= s[i];
        acc[i] = a * KP32;
    }
}
#endif

/* Requires len >= 128. Returns the pre-finalized merge of 8 lanes. */
static uint64_t long_core(const unsigned char *p, size_t len)
{
    uint64_t acc[8];
    const size_t nb = (len - 1) >> 6;   /* full stripes, last one handled separately */
    const size_t nblocks = nb >> 4;
    const size_t rem = nb & 15;
    size_t b, j;
    uint64_t h;

#if defined(__AVX2__)
    __m256i a0 = _mm256_loadu_si256((const __m256i *)(SEC + 8));
    __m256i a1 = _mm256_loadu_si256((const __m256i *)(SEC + 12));
    for (b = 0; b < nblocks; b++) {
        const unsigned char *q = p + (b << 10);
        for (j = 0; j < 16; j++) acc_stripe(&a0, &a1, q + (j << 6), SEC + j);
        a0 = scramble256(a0, SEC + 24);
        a1 = scramble256(a1, SEC + 28);
    }
    {
        const unsigned char *q = p + (nblocks << 10);
        for (j = 0; j < rem; j++) acc_stripe(&a0, &a1, q + (j << 6), SEC + j);
    }
    acc_stripe(&a0, &a1, p + len - 64, SEC + 16);
    _mm256_storeu_si256((__m256i *)acc, a0);
    _mm256_storeu_si256((__m256i *)(acc + 4), a1);
#else
    for (j = 0; j < 8; j++) acc[j] = SEC[8 + j];
    for (b = 0; b < nblocks; b++) {
        const unsigned char *q = p + (b << 10);
        for (j = 0; j < 16; j++) acc_stripe_s(acc, q + (j << 6), SEC + j);
        scramble_s(acc, SEC + 24);
    }
    {
        const unsigned char *q = p + (nblocks << 10);
        for (j = 0; j < rem; j++) acc_stripe_s(acc, q + (j << 6), SEC + j);
    }
    acc_stripe_s(acc, p + len - 64, SEC + 16);
#endif

    h  = (uint64_t)len * KP1;
    h += mul_fold(acc[0] ^ SEC[0], acc[1] ^ SEC[1]);
    h += mul_fold(acc[2] ^ SEC[2], acc[3] ^ SEC[3]);
    h += mul_fold(acc[4] ^ SEC[4], acc[5] ^ SEC[5]);
    h += mul_fold(acc[6] ^ SEC[6], acc[7] ^ SEC[7]);
    return h;
}

#define PAR_MIN   ((size_t)1 << 20)
#define PAR_CHUNKS 16

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t h;

    if (len <= 16) {                      /* one charm, or a pinch of them */
        uint64_t a, b;
        if (len >= 8) { a = rd8(data); b = rd8(data + len - 8); }
        else if (len >= 4) { a = (uint64_t)rd4(data); b = (uint64_t)rd4(data + len - 4); }
        else if (len > 0) {
            uint64_t c1 = data[0], c2 = data[len >> 1], c3 = data[len - 1];
            a = (c1 << 48) | (c2 << 24) | c3;
            b = ((uint64_t)len << 32) ^ ((c3 << 16) | (c2 << 8) | c1);
        } else { a = SEC[6]; b = SEC[7]; }
        h  = (uint64_t)len * KP1;
        h += (a ^ SEC[0]) + rotl64(b ^ SEC[1], 27);
        h += mul_fold(a ^ SEC[2], (b ^ SEC[3]) + ((uint64_t)len << 40));
        return moremur(h);
    }

    if (len <= 128) {                     /* a few nubs, from both ends inward */
        int i = (int)((len - 1) >> 5);    /* 0..3 */
        h = (uint64_t)len * KP1;
        for (; i >= 0; i--) {
            h += mix16(data + 16 * i, SEC + 4 * i);
            h += mix16(data + len - 16 * ((size_t)i + 1), SEC + 4 * i + 2);
        }
        return moremur(h);
    }

    if (len >= PAR_MIN) {                 /* call over the stall-neighbours */
        uint64_t sub[PAR_CHUNKS];
        size_t base = (len / PAR_CHUNKS) & ~(size_t)63;   /* >= 65536, stripe-aligned */
        int c;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (c = 0; c < PAR_CHUNKS; c++) {
            size_t off = (size_t)c * base;
            size_t l = (c == PAR_CHUNKS - 1) ? (len - off) : base;
            sub[c] = long_core(data + off, l);
        }
        h = (uint64_t)len * KP1;          /* ordered combine => deterministic */
        for (c = 0; c < PAR_CHUNKS; c++)
            h += mul_fold(sub[c] ^ SEC[c], (KP2 * (uint64_t)(c + 1)) ^ SEC[c + 8]);
        return moremur(h);
    }

    return moremur(long_core(data, len));
}
