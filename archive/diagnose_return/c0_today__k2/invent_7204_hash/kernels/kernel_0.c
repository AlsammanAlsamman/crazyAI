#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- counting-bars: exact unsigned integer arithmetic only ---- */
#define XP64_1 0x9E3779B185EBCA87ULL
#define XP64_2 0xC2B2AE3D27D4EB4FULL
#define XP64_3 0x165667B19E3779F9ULL
#define XP64_4 0x85EBCA77C2B2AE63ULL
#define XP64_5 0x27D4EB2F165667C5ULL
#define XP32_1 0x9E3779B1ULL
#define XP32_2 0x85EBCA77ULL
#define XP32_3 0xC2B2AE3DULL

/* ---- the inkwell that never runs dry: 24 fixed drops (192 bytes).
       High-entropy fixed secret; any such secret works, no bit-compat claimed. ---- */
static const uint64_t SEC[24] = {
    0xbe4ba423396cfeb8ULL, 0x1cad21f72c81017cULL, 0xdb979083e96dd4deULL, 0x1f67b3b7a4a44072ULL,
    0x78e5c0cc4ee679cbULL, 0x217cffcc7dd05a82ULL, 0x8e2443f7744608b8ULL, 0x4c263a81e69035e0ULL,
    0xcb00c391bb52283cULL, 0xa32e531b8b65d088ULL, 0x4ef90da297486471ULL, 0xd8acdea946ef1938ULL,
    0x3f349ce33f76faa8ULL, 0x1d4f0bc7c7bbdcf9ULL, 0x3159b4cd4be0518aULL, 0x647378d9c97e9fc8ULL,
    0xc3ebd33483acc5eaULL, 0xeb6313faffa081c5ULL, 0x49daf0b751dd0d17ULL, 0x9e68d429265516d3ULL,
    0xfca1477d58be162bULL, 0xce31d07ad1b8f88fULL, 0x280416958f3acb45ULL, 0x7e404bbbcafbd7afULL
};

static inline uint64_t ld64(const unsigned char *p){ uint64_t v; memcpy(&v,p,sizeof v); return v; }
static inline uint64_t ld32(const unsigned char *p){ uint32_t v; memcpy(&v,p,sizeof v); return (uint64_t)v; }

/* 64x64 -> 128, folded: the one place strong diffusion is spent */
static inline uint64_t fold128(uint64_t a, uint64_t b)
{
#if defined(__SIZEOF_INT128__)
    __uint128_t const r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t const al=a&0xFFFFFFFFULL, ah=a>>32, bl=b&0xFFFFFFFFULL, bh=b>>32;
    uint64_t const ll=al*bl, lh=al*bh, hl=ah*bl, hh=ah*bh;
    uint64_t const cross=(ll>>32)+(lh&0xFFFFFFFFULL)+hl;
    uint64_t const upper=(lh>>32)+(cross>>32)+hh;
    uint64_t const lower=(cross<<32)|(ll&0xFFFFFFFFULL);
    return lower ^ upper;
#endif
}

/* ---- the storm: ONE pass, eight notes ---- */
static inline uint64_t storm(uint64_t h){ h ^= h>>37; h *= XP64_3; h ^= h>>32; return h; }

/* ---- the pluck: nstripes x 64 bytes onto the eight strings.
       stripe s uses secret bytes [secoff+8s, secoff+8s+64). ---- */
static inline void absorb(uint64_t *restrict acc, const unsigned char *restrict p,
                          size_t nstripes, size_t secoff)
{
    const unsigned char *const sb = (const unsigned char *)SEC + secoff;
#if defined(__AVX2__)
    __m256i a0 = _mm256_loadu_si256((const __m256i *)(const void *)acc);
    __m256i a1 = _mm256_loadu_si256((const __m256i *)(const void *)(acc + 4));
    for (size_t s = 0; s < nstripes; s++) {
        const unsigned char *const in = p  + 64*s;
        const unsigned char *const sk = sb +  8*s;
        __m256i const d0 = _mm256_loadu_si256((const __m256i *)(const void *)in);
        __m256i const d1 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 32));
        __m256i const k0 = _mm256_loadu_si256((const __m256i *)(const void *)sk);
        __m256i const k1 = _mm256_loadu_si256((const __m256i *)(const void *)(sk + 32));
        __m256i const x0 = _mm256_xor_si256(d0, k0);          /* fresh feather */
        __m256i const x1 = _mm256_xor_si256(d1, k1);
        __m256i const p0 = _mm256_mul_epu32(x0, _mm256_srli_epi64(x0, 32)); /* ink-weight */
        __m256i const p1 = _mm256_mul_epu32(x1, _mm256_srli_epi64(x1, 32));
        __m256i const w0 = _mm256_shuffle_epi32(d0, _MM_SHUFFLE(1,0,3,2));  /* neighbour shivers */
        __m256i const w1 = _mm256_shuffle_epi32(d1, _MM_SHUFFLE(1,0,3,2));
        a0 = _mm256_add_epi64(_mm256_add_epi64(a0, w0), p0);
        a1 = _mm256_add_epi64(_mm256_add_epi64(a1, w1), p1);
    }
    _mm256_storeu_si256((__m256i *)(void *)acc,       a0);
    _mm256_storeu_si256((__m256i *)(void *)(acc + 4), a1);
#else
    for (size_t s = 0; s < nstripes; s++) {
        const unsigned char *const in = p  + 64*s;
        const unsigned char *const sk = sb +  8*s;
        for (int j = 0; j < 8; j++) {                 /* 8 independent strings */
            uint64_t const v = ld64(in + 8*j);
            uint64_t const k = v ^ ld64(sk + 8*j);
            acc[j ^ 1] += v;                          /* the neighbour shivers */
            acc[j]     += (uint64_t)(uint32_t)k * (uint64_t)(uint32_t)(k >> 32);
        }
    }
#endif
}

/* ---- the bridge: bijective, so no state entropy is lost.
       scramble each string, then rotate the strings so the pairing graph
       becomes a connected 8-cycle: every pluck reaches every string. ---- */
static inline void bridge(uint64_t *restrict acc)
{
    uint64_t t[8];
    for (int j = 0; j < 8; j++) {
        uint64_t a = acc[j];
        a ^= a >> 47;
        a ^= SEC[16 + j];
        a *= XP32_1;
        t[j] = a;
    }
    for (int j = 0; j < 8; j++) acc[j] = t[(j + 3) & 7];
}

/* ---- regime 2: too few marks to reach across all eight strings.
       One string, overlapping end-reads, then the same single storm. ---- */
static uint64_t short_path(const unsigned char *data, size_t len)
{
    uint64_t a, b;
    uint64_t seed = SEC[0] ^ ((uint64_t)len * XP64_2);
    if (len >= 16) {
        size_t i = 0, n = len;
        while (n > 16) {                              /* 16 marks per pluck */
            seed = fold128(ld64(data + i) ^ SEC[1] ^ seed, ld64(data + i + 8) ^ SEC[2]);
            i += 16; n -= 16;
        }
        a = ld64(data + len - 16);
        b = ld64(data + len -  8);
    } else if (len >= 8) {
        a = ld64(data);          b = ld64(data + len - 8);
    } else if (len >= 4) {
        a = ld32(data);          b = ld32(data + len - 4);
    } else if (len > 0) {
        a = ((uint64_t)data[0] << 16) | ((uint64_t)data[len >> 1] << 8) | (uint64_t)data[len - 1];
        b = 0;
    } else { a = 0; b = 0; }
    uint64_t h = fold128(a ^ SEC[3] ^ seed, b ^ SEC[4]);
    h += (uint64_t)len * XP64_1;
    return storm(h);
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* the native counts the marks first and picks his regime */
    if (len < 64) return short_path(data, len);

    uint64_t acc[8] = { XP32_3, XP64_1, XP64_2, XP64_3, XP64_4, XP32_2, XP64_5, XP32_1 };

    size_t const nbStripes = (len - 1) / 64;     /* every stripe but the last */
    size_t const nbBlocks  = nbStripes / 16;     /* 1024-byte blocks */

    for (size_t b = 0; b < nbBlocks; b++) {
        absorb(acc, data + b * 1024, 16, 0);
        bridge(acc);                             /* once per 1024 bytes */
    }
    size_t const rem = nbStripes - nbBlocks * 16;
    if (rem) absorb(acc, data + nbBlocks * 1024, rem, 0);

    /* the last mark is never skipped: final stripe read from the very end,
       with its own window of the inkwell */
    absorb(acc, data + len - 64, 1, 120);

    /* ---- the storm passes once, and only once ---- */
    uint64_t h = (uint64_t)len * XP64_1;
    h += fold128(acc[0] ^ SEC[1], acc[1] ^ SEC[2]);
    h += fold128(acc[2] ^ SEC[3], acc[3] ^ SEC[4]);
    h += fold128(acc[4] ^ SEC[5], acc[5] ^ SEC[6]);
    h += fold128(acc[6] ^ SEC[7], acc[7] ^ SEC[8]);
    return storm(h);
}
