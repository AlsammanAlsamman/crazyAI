#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define HP0 0x9E3779B185EBCA87ULL
#define HP1 0xC2B2AE3D27D4EB4FULL
#define HP2 0x165667B19E3779F9ULL
#define HP3 0x85EBCA77C2B2AE63ULL
#define HP4 0x27D4EB2F165667C5ULL
#define HP5 0xD6E8FEB86659FD93ULL
#define K32 0x9E3779B1u            /* odd -> multiply is a bijection mod 2^64 */

static inline uint64_t rd8(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t rd4(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }

/* 64x64 -> 128 multiply, folded: strong 2-input mixer, ~1 multiply */
static inline uint64_t mum(uint64_t a, uint64_t b)
{
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t al=a&0xffffffffULL, ah=a>>32, bl=b&0xffffffffULL, bh=b>>32;
    uint64_t ll=al*bl, lh=al*bh, hl=ah*bl, hh=ah*bh;
    uint64_t mid=(ll>>32)+(lh&0xffffffffULL)+(hl&0xffffffffULL);
    uint64_t lo=(ll&0xffffffffULL)|(mid<<32);
    uint64_t hi=hh+(lh>>32)+(hl>>32)+(mid>>32);
    return lo ^ hi;
#endif
}

/* final "hard twist-stir": MurmurHash3 fmix64 -- full 64-bit avalanche */
static inline uint64_t fin64(uint64_t h)
{
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33; return h;
}

#if defined(__AVX2__)
/* one helper's twist-stir on 4 lanes at once: acc = ((acc+v) ^ ((acc+v)>>32)) * K  */
static inline __m256i vround(__m256i acc, __m256i v, __m256i k)
{
    __m256i t  = _mm256_add_epi64(acc, v);
    t          = _mm256_xor_si256(t, _mm256_srli_epi64(t, 32));
    __m256i lo = _mm256_mul_epu32(t, k);                          /* lo32(t)*K */
    __m256i hi = _mm256_mul_epu32(_mm256_srli_epi64(t, 32), k);   /* hi32(t)*K */
    return _mm256_add_epi64(lo, _mm256_slli_epi64(hi, 32));       /* == t*K mod 2^64 */
}
#endif

static uint64_t hcore(const unsigned char *p, size_t len, uint64_t seed)
{
    uint64_t h;

    if (len <= 16) {                                  /* one bowl is enough */
        uint64_t a, b;
        if (len >= 8)      { a = rd8(p);              b = rd8(p + len - 8); }
        else if (len >= 4) { a = rd4(p);              b = rd4(p + len - 4); }
        else if (len)      { a = ((uint64_t)p[0]<<16) | ((uint64_t)p[len>>1]<<8)
                                 | (uint64_t)p[len-1]; b = 0; }
        else               { a = 0;                   b = 0; }
        h = mum(a ^ HP2 ^ seed, b ^ HP3 ^ (uint64_t)len);

    } else if (len < 256) {                           /* two helpers, scalar */
        const unsigned char *q = p; size_t n = len;
        uint64_t h0 = HP0 ^ seed, h1 = HP1 ^ (uint64_t)len;
        while (n >= 32) {
            h0 = mum(h0 ^ rd8(q),      HP2 ^ rd8(q + 8));
            h1 = mum(h1 ^ rd8(q + 16), HP3 ^ rd8(q + 24));
            q += 32; n -= 32;
        }
        if (n > 16) h0 = mum(h0 ^ rd8(q), HP2 ^ rd8(q + 8));
        h1 = mum(h1 ^ rd8(p + len - 16), HP3 ^ rd8(p + len - 8)); /* overlapping tail */
        h  = h0 ^ h1;

    } else {
#if defined(__AVX2__)
        const unsigned char *q = p; size_t n = len;
        const __m256i k  = _mm256_set1_epi64x((long long)(uint64_t)K32);
        __m256i a0 = _mm256_set_epi64x((long long)(HP0 ^ seed), (long long)HP1,
                                       (long long)HP2,          (long long)HP3);
        __m256i a1 = _mm256_set_epi64x((long long)HP4, (long long)HP5,
                                       (long long)(HP0 + (uint64_t)len),
                                       (long long)(HP1 ^ (uint64_t)len));
        __m256i a2 = _mm256_set_epi64x((long long)(HP2 ^ seed), (long long)HP3,
                                       (long long)HP4,          (long long)HP5);
        __m256i a3 = _mm256_set_epi64x((long long)HP1, (long long)HP0,
                                       (long long)(HP5 ^ (uint64_t)len),
                                       (long long)(HP4 + (uint64_t)len));
        while (n >= 128) {                            /* 16 lanes stirring at once */
            a0 = vround(a0, _mm256_loadu_si256((const __m256i *)(q      )), k);
            a1 = vround(a1, _mm256_loadu_si256((const __m256i *)(q +  32)), k);
            a2 = vround(a2, _mm256_loadu_si256((const __m256i *)(q +  64)), k);
            a3 = vround(a3, _mm256_loadu_si256((const __m256i *)(q +  96)), k);
            q += 128; n -= 128;
        }
        {                                             /* branch-free overlapping tail */
            const unsigned char *e = p + len - 128;
            a0 = vround(a0, _mm256_loadu_si256((const __m256i *)(e      )), k);
            a1 = vround(a1, _mm256_loadu_si256((const __m256i *)(e +  32)), k);
            a2 = vround(a2, _mm256_loadu_si256((const __m256i *)(e +  64)), k);
            a3 = vround(a3, _mm256_loadu_si256((const __m256i *)(e +  96)), k);
        }
        a0 = vround(a0, _mm256_permute4x64_epi64(a2, 0x4E), k);   /* pour together */
        a1 = vround(a1, _mm256_permute4x64_epi64(a3, 0x1B), k);
        a0 = vround(a0, _mm256_permute4x64_epi64(a1, 0xB1), k);
        {
            uint64_t l[4];
            _mm256_storeu_si256((__m256i *)l, a0);
            h = mum(l[0] ^ HP0, l[1] ^ HP1) ^ mum(l[2] ^ HP2, l[3] ^ HP3);
        }
#else
        const unsigned char *q = p; size_t n = len;
        uint64_t h0 = HP0 ^ seed, h1 = HP1, h2 = HP2, h3 = HP3 ^ (uint64_t)len;
        while (n >= 64) {
            h0 = mum(h0 ^ rd8(q     ), HP4 ^ rd8(q +  8));
            h1 = mum(h1 ^ rd8(q + 16), HP5 ^ rd8(q + 24));
            h2 = mum(h2 ^ rd8(q + 32), HP0 ^ rd8(q + 40));
            h3 = mum(h3 ^ rd8(q + 48), HP1 ^ rd8(q + 56));
            q += 64; n -= 64;
        }
        {
            const unsigned char *e = p + len - 64;
            h0 = mum(h0 ^ rd8(e     ), HP4 ^ rd8(e +  8));
            h1 = mum(h1 ^ rd8(e + 16), HP5 ^ rd8(e + 24));
            h2 = mum(h2 ^ rd8(e + 32), HP0 ^ rd8(e + 40));
            h3 = mum(h3 ^ rd8(e + 48), HP1 ^ rd8(e + 56));
        }
        h = mum(h0 ^ h1 ^ HP2, h2 ^ h3 ^ HP3);
#endif
    }
    return fin64(mum(h ^ (uint64_t)len ^ HP4, HP5 ^ seed));
}

uint64_t kernel(const unsigned char *data, size_t len)
{
#if defined(_OPENMP)
    if (len >= (size_t)(1u << 21)) {                  /* extra helpers for the cellar */
        enum { NP = 8 };
        size_t blk = (len / NP) & ~(size_t)127;       /* depends only on len -> deterministic */
        if (blk >= 4096) {
            uint64_t part[NP];
            int i;
            #pragma omp parallel for schedule(static)
            for (i = 0; i < NP; i++) {
                size_t off = (size_t)i * blk;
                size_t l   = (i == NP - 1) ? (len - off) : blk;
                part[i] = hcore(data + off, l, HP0 * (uint64_t)(i + 1));
            }
            {
                uint64_t h = HP1;
                for (i = 0; i < NP; i++)
                    h = mum(h ^ part[i] ^ HP2, HP3 + (uint64_t)i);
                return fin64(mum(h ^ (uint64_t)len, HP5));
            }
        }
    }
#endif
    return hcore(data, len, 0);
}
