#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- keys ---- */
static const uint64_t KS[16] = {
    0xa0761d6478bd642fULL, 0xe7037ed1a0b428dbULL, 0x8ebc6af09c88c6e3ULL,
    0x589965cc75374cc3ULL, 0x1d8e4e27c47d124fULL, 0xff51afd7ed558ccdULL,
    0xc4ceb9fe1a85ec53ULL, 0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL,
    0x94d049bb133111ebULL, 0x2545f4914f6cdd1dULL, 0xd6e8feb86659fd93ULL,
    0xca62c1d6b1a9e38dULL, 0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
    0xc0ac29b7c97c50ddULL
};

static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t rd32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }
static inline uint64_t rotl64(uint64_t x, int r){ return (x<<r)|(x>>(64-r)); }

/* 128-bit product folded to 64 bits */
static inline uint64_t mum(uint64_t a, uint64_t b){
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t ha=a>>32, la=(uint32_t)a, hb=b>>32, lb=(uint32_t)b;
    uint64_t rh=ha*hb, m0=ha*lb, m1=hb*la, rl=la*lb;
    uint64_t t=rl+(m0<<32); uint64_t c=(t<rl);
    uint64_t lo=t+(m1<<32); c+=(lo<t);
    uint64_t hi=rh+(m0>>32)+(m1>>32)+c;
    return lo^hi;
#endif
}

/* scalar tree node: combine two cups into one, twist-stirred */
static inline uint64_t snode(uint64_t x, uint64_t y, uint64_t k){
    uint64_t a = x ^ k;
    uint64_t b = y ^ rotl64(k, 32);
    return mum(a, b) ^ (a + rotl64(b, 32));
}

static inline uint64_t finish(uint64_t h, uint64_t len){
    h ^= len * 0x9e3779b97f4a7c15ULL;
    h ^= mum(h ^ 0xa0761d6478bd642fULL, h + 0xe7037ed1a0b428dbULL);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 29; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 32;
    return h;
}

/* ---- tiny inputs (len < 16) ---- */
static uint64_t hash_small(const unsigned char *p, size_t len){
    uint64_t a, b;
    if (len >= 8)      { a = rd64(p);            b = rd64(p + len - 8); }
    else if (len >= 4) { a = rd32(p);            b = rd32(p + len - 4); }
    else if (len)      { a = ((uint64_t)p[0]<<16) | ((uint64_t)p[len>>1]<<8) | (uint64_t)p[len-1]; b = 0x9e3779b9ULL; }
    else               { a = 0; b = 0; }
    return finish(snode(a, b, KS[0]), (uint64_t)len);
}

/* ---- medium inputs: explicit pairwise tree over 16-byte chunks ---- */
static uint64_t hash_medium(const unsigned char *p, size_t len){
    uint64_t buf[20];
    size_t n = (len + 15) >> 4;            /* 1..16 chunks for len < 256 */
    for (size_t i = 0; i < n; i++) {
        size_t off = (i + 1 < n) ? (i << 4) : (len - 16);   /* last chunk overlaps */
        buf[i] = snode(rd64(p + off), rd64(p + off + 8), KS[i & 15]);
    }
    size_t m = n, r = 0;
    while (m > 1) {                        /* pair up the cups, round after round */
        size_t h2 = m >> 1;
        for (size_t j = 0; j < h2; j++)
            buf[j] = snode(buf[2*j], buf[2*j+1], KS[(j + 5*r) & 15]);
        if (m & 1) { buf[h2] = buf[m-1]; m = h2 + 1; } else m = h2;
        r++;
    }
    return finish(buf[0], (uint64_t)len);
}

#if defined(__AVX2__)
/* ---- 4-lane vector node: Granny plus three helpers, one twist-stir ---- */
static const uint64_t VKS[8][4] = {
    { 0xa0761d6478bd642fULL, 0xe7037ed1a0b428dbULL, 0x8ebc6af09c88c6e3ULL, 0x589965cc75374cc3ULL },
    { 0x1d8e4e27c47d124fULL, 0xff51afd7ed558ccdULL, 0xc4ceb9fe1a85ec53ULL, 0x9e3779b97f4a7c15ULL },
    { 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL, 0x2545f4914f6cdd1dULL, 0xd6e8feb86659fd93ULL },
    { 0xca62c1d6b1a9e38dULL, 0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL, 0xc0ac29b7c97c50ddULL },
    { 0x9ae16a3b2f90404fULL, 0xc3a5c85c97cb3127ULL, 0xb492b66fbe98f273ULL, 0x9ddfea08eb382d69ULL },
    { 0x85ebca6bc2b2ae63ULL, 0xcc9e2d51b873593dULL, 0x1b873593cc9e2d51ULL, 0xe9846af9b1a615d3ULL },
    { 0x2b7e151628aed2a7ULL, 0x3243f6a8885a308dULL, 0x13198a2e03707345ULL, 0xa4093822299f31d0ULL },
    { 0x082efa98ec4e6c89ULL, 0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL, 0xc0ac29b7c97c50ddULL }
};
#define VK(i) _mm256_loadu_si256((const __m256i *)VKS[i])

static inline __m256i vnode(__m256i x, __m256i y, __m256i k){
    __m256i a = _mm256_xor_si256(x, k);
    __m256i b = _mm256_add_epi64(y, k);
    __m256i d = _mm256_xor_si256(a, _mm256_shuffle_epi32(b, 0xB1));
    __m256i p = _mm256_mul_epu32(d, _mm256_srli_epi64(d, 32)); /* lo32*hi32 */
    __m256i s = _mm256_add_epi64(a, b);                        /* keep all bits */
    return _mm256_xor_si256(p, _mm256_shuffle_epi32(s, 0xB1));
}

/* 256-byte block -> one cup: 8 vectors -> 4 -> 2 -> 1 */
static inline __m256i vblock(const unsigned char *q){
    __m256i v0 = _mm256_loadu_si256((const __m256i *)(q +   0));
    __m256i v1 = _mm256_loadu_si256((const __m256i *)(q +  32));
    __m256i v2 = _mm256_loadu_si256((const __m256i *)(q +  64));
    __m256i v3 = _mm256_loadu_si256((const __m256i *)(q +  96));
    __m256i v4 = _mm256_loadu_si256((const __m256i *)(q + 128));
    __m256i v5 = _mm256_loadu_si256((const __m256i *)(q + 160));
    __m256i v6 = _mm256_loadu_si256((const __m256i *)(q + 192));
    __m256i v7 = _mm256_loadu_si256((const __m256i *)(q + 224));
    __m256i a0 = vnode(v0, v1, VK(0));
    __m256i a1 = vnode(v2, v3, VK(1));
    __m256i a2 = vnode(v4, v5, VK(2));
    __m256i a3 = vnode(v6, v7, VK(3));
    __m256i b0 = vnode(a0, a1, VK(4));
    __m256i b1 = vnode(a2, a3, VK(5));
    return vnode(b0, b1, VK(6));
}

static uint64_t hash_big(const unsigned char *p, size_t len){
    size_t nb = len >> 8, rem = len & 255;
    __m256i st[64]; unsigned dp[64]; int top = 0;
    const unsigned char *q = p;
    for (size_t i = 0; i < nb; i++, q += 256) {
        __m256i v = vblock(q);
        unsigned d = 0;                            /* streaming pairwise merge */
        while (top > 0 && dp[top-1] == d) { top--; v = vnode(st[top], v, VK(7 - (d & 1))); d++; }
        st[top] = v; dp[top] = d; top++;
    }
    if (rem) {                                     /* overlapping final block */
        __m256i v = vblock(p + len - 256);
        unsigned d = 0;
        while (top > 0 && dp[top-1] == d) { top--; v = vnode(st[top], v, VK(7 - (d & 1))); d++; }
        st[top] = v; dp[top] = d; top++;
    }
    __m256i acc = st[--top];
    while (top > 0) { top--; acc = vnode(st[top], acc, VK(6)); }
    uint64_t l[4];
    _mm256_storeu_si256((__m256i *)l, acc);
    uint64_t x = snode(l[0], l[1], KS[2]);
    uint64_t y = snode(l[2], l[3], KS[5]);
    return finish(snode(x, y, KS[9]), (uint64_t)len);
}
#else
/* scalar 4-wide fallback: 64-byte block -> one cup */
static inline uint64_t sblock(const unsigned char *q){
    uint64_t w0 = snode(rd64(q +  0), rd64(q +  8), KS[0]);
    uint64_t w1 = snode(rd64(q + 16), rd64(q + 24), KS[1]);
    uint64_t w2 = snode(rd64(q + 32), rd64(q + 40), KS[2]);
    uint64_t w3 = snode(rd64(q + 48), rd64(q + 56), KS[3]);
    return snode(snode(w0, w1, KS[4]), snode(w2, w3, KS[5]), KS[6]);
}
static uint64_t hash_big(const unsigned char *p, size_t len){
    size_t nb = len >> 6, rem = len & 63;
    uint64_t st[64]; unsigned dp[64]; int top = 0;
    const unsigned char *q = p;
    for (size_t i = 0; i < nb; i++, q += 64) {
        uint64_t v = sblock(q); unsigned d = 0;
        while (top > 0 && dp[top-1] == d) { top--; v = snode(st[top], v, KS[8 + (d & 3)]); d++; }
        st[top] = v; dp[top] = d; top++;
    }
    if (rem) {
        uint64_t v = sblock(p + len - 64); unsigned d = 0;
        while (top > 0 && dp[top-1] == d) { top--; v = snode(st[top], v, KS[8 + (d & 3)]); d++; }
        st[top] = v; dp[top] = d; top++;
    }
    uint64_t acc = st[--top];
    while (top > 0) { top--; acc = snode(st[top], acc, KS[12]); }
    return finish(acc, (uint64_t)len);
}
#endif

uint64_t kernel(const unsigned char *data, size_t len){
    if (len < 16) return hash_small(data, len);
#if defined(__AVX2__)
    if (len >= 256) return hash_big(data, len);
#else
    if (len >= 64)  return hash_big(data, len);
#endif
    return hash_medium(data, len);
}
