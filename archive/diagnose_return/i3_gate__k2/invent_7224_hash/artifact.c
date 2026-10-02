#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define RIVERBANK_AVX2 1
#endif

#define ROTR64(x,n) (((x) >> (n)) | ((x) << (64 - (n))))

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

/* the owl: seals the folded shape to one fixed size, one way only
   (MurmurHash3 fmix64 - the only multiplications in the whole kernel) */
static inline uint64_t owl_seal(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32;
    return x;
}

/* one musk-deer stride on a column: uphill twist (rot), downhill fold (xor),
   doubling-back that eats its own trail (add).  BLAKE2b's G, message words
   removed - it is a permutation, not a compression. */
#define STRIDE(a,b,c,d) do {               \
    a += b; d ^= a; d = ROTR64(d, 32);     \
    c += d; b ^= c; b = ROTR64(b, 24);     \
    a += b; d ^= a; d = ROTR64(d, 16);     \
    c += d; b ^= c; b = ROTR64(b, 63);     \
} while (0)

static const uint64_t IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};
#define RC(i) (IV[(i) & 7] ^ (0x9e3779b97f4a7c15ULL * (uint64_t)((i) + 1)))

/* ---- regime B: not enough ground for a stride -> walk the riverbank ---- */
/* one column, 32-byte steps.  also the portable path when AVX2 is absent. */
static uint64_t riverbank_walk(const unsigned char *data, size_t len) {
    uint64_t s0 = IV[0] ^ (uint64_t)len, s1 = IV[1], s2 = IV[2], s3 = IV[3];
    const unsigned char *p = data;
    size_t n = len;

    while (n >= 32) {                        /* silence: 4 pure loads, 4 xors */
        s0 ^= ld64(p);      s1 ^= ld64(p + 8);
        s2 ^= ld64(p + 16); s3 ^= ld64(p + 24);
        STRIDE(s0, s1, s2, s3);
        p += 32; n -= 32;
    }
    if (n) {                                 /* the flat stone: pad, then lay */
        unsigned char stone[32];
        memset(stone, 0, sizeof stone);
        memcpy(stone, p, n);
        stone[n] = 0x01;
        s0 ^= ld64(stone);      s1 ^= ld64(stone + 8);
        s2 ^= ld64(stone + 16); s3 ^= ld64(stone + 24);
        STRIDE(s0, s1, s2, s3);
    }
    /* turn ninety degrees twice: roles rotate so no lane keeps its direction */
    STRIDE(s0, s1, s2, s3);
    STRIDE(s0, s2, s3, s1);
    STRIDE(s0, s3, s1, s2);

    uint64_t h = (s0 + ROTR64(s1, 11)) ^ (s2 + ROTR64(s3, 37));
    return owl_seal(h ^ (uint64_t)len);
}

#ifdef RIVERBANK_AVX2
/* ---- regime A: twin 4x4 grids, 256-byte strides ---- */
#define V_ADD(a,b)   _mm256_add_epi64(a, b)
#define V_XOR(a,b)   _mm256_xor_si256(a, b)
#define V_ROTR(x,n)  _mm256_or_si256(_mm256_srli_epi64(x, (n)),           \
                                     _mm256_slli_epi64(x, 64 - (n)))
#define V_ROTR32(x)  _mm256_shuffle_epi32(x, 0xB1)
#define V_LOAD(p)    _mm256_loadu_si256((const __m256i *)(const void *)(p))
#define V4(w,x,y,z)  _mm256_set_epi64x((long long)(z), (long long)(y),     \
                                       (long long)(x), (long long)(w))

#define V_STRIDE(a,b,c,d) do {                                 \
    a = V_ADD(a,b); d = V_XOR(d,a); d = V_ROTR32(d);           \
    c = V_ADD(c,d); b = V_XOR(b,c); b = V_ROTR(b,24);          \
    a = V_ADD(a,b); d = V_XOR(d,a); d = V_ROTR(d,16);          \
    c = V_ADD(c,d); b = V_XOR(b,c); b = V_ROTR(b,63);          \
} while (0)

/* turning the shape ninety degrees, and turning it back */
#define V_TURN(b,c,d) do {                           \
    b = _mm256_permute4x64_epi64(b, 0x39);           \
    c = _mm256_permute4x64_epi64(c, 0x4E);           \
    d = _mm256_permute4x64_epi64(d, 0x93);           \
} while (0)
#define V_UNTURN(b,c,d) do {                         \
    b = _mm256_permute4x64_epi64(b, 0x93);           \
    c = _mm256_permute4x64_epi64(c, 0x4E);           \
    d = _mm256_permute4x64_epi64(d, 0x39);           \
} while (0)

/* columns in every direction, then diagonals: one full race of the heap */
#define V_ROUND(a,b,c,d) do {                        \
    V_STRIDE(a,b,c,d);                               \
    V_TURN(b,c,d);                                   \
    V_STRIDE(a,b,c,d);                               \
    V_UNTURN(b,c,d);                                 \
} while (0)

static uint64_t riverbank_grid(const unsigned char *restrict data, size_t len) {
    __m256i a0 = V4(RC(0),  RC(1),  RC(2),  RC(3));
    __m256i a1 = V4(RC(4),  RC(5),  RC(6),  RC(7));
    __m256i a2 = V4(RC(8),  RC(9),  RC(10), RC(11));
    __m256i a3 = V4(RC(12), RC(13), RC(14), RC(15));
    __m256i b0 = V4(RC(16), RC(17), RC(18), RC(19));
    __m256i b1 = V4(RC(20), RC(21), RC(22), RC(23));
    __m256i b2 = V4(RC(24), RC(25), RC(26), RC(27));
    __m256i b3 = V4(RC(28), RC(29), RC(30), RC(31));

    const unsigned char *p = data;
    size_t n = len;

    while (n >= 256) {
        /* the servant holds silence: eight pure loads, eight xors, no chain */
        a0 = V_XOR(a0, V_LOAD(p));         a1 = V_XOR(a1, V_LOAD(p + 32));
        a2 = V_XOR(a2, V_LOAD(p + 64));    a3 = V_XOR(a3, V_LOAD(p + 96));
        b0 = V_XOR(b0, V_LOAD(p + 128));   b1 = V_XOR(b1, V_LOAD(p + 160));
        b2 = V_XOR(b2, V_LOAD(p + 192));   b3 = V_XOR(b3, V_LOAD(p + 224));
        V_ROUND(a0, a1, a2, a3);           /* two heaps race side by side */
        V_ROUND(b0, b1, b2, b3);
        p += 256; n -= 256;
    }
    if (n) {
        unsigned char stone[256];
        memset(stone, 0, sizeof stone);
        memcpy(stone, p, n);
        stone[n] = 0x01;
        a0 = V_XOR(a0, V_LOAD(stone));        a1 = V_XOR(a1, V_LOAD(stone + 32));
        a2 = V_XOR(a2, V_LOAD(stone + 64));   a3 = V_XOR(a3, V_LOAD(stone + 96));
        b0 = V_XOR(b0, V_LOAD(stone + 128));  b1 = V_XOR(b1, V_LOAD(stone + 160));
        b2 = V_XOR(b2, V_LOAD(stone + 192));  b3 = V_XOR(b3, V_LOAD(stone + 224));
        V_ROUND(a0, a1, a2, a3);
        V_ROUND(b0, b1, b2, b3);
    }

    /* the shrine: the last heap has nowhere left to diffuse, so spend the
       rounds here instead of spending them on every block */
    __m256i L = _mm256_set1_epi64x((long long)(uint64_t)len);
    a0 = V_XOR(a0, L); b0 = V_XOR(b0, L);
    V_ROUND(a0, a1, a2, a3); V_ROUND(b0, b1, b2, b3);
    V_ROUND(a0, a1, a2, a3); V_ROUND(b0, b1, b2, b3);
    a0 = V_XOR(a0, b0); a1 = V_XOR(a1, b1);
    a2 = V_XOR(a2, b2); a3 = V_XOR(a3, b3);
    V_ROUND(a0, a1, a2, a3);
    V_ROUND(a0, a1, a2, a3);

    /* fold the corners into the centre; the heaps are burned */
    uint64_t t[16];
    _mm256_storeu_si256((__m256i *)(void *)(t + 0),  a0);
    _mm256_storeu_si256((__m256i *)(void *)(t + 4),  a1);
    _mm256_storeu_si256((__m256i *)(void *)(t + 8),  a2);
    _mm256_storeu_si256((__m256i *)(void *)(t + 12), a3);

    uint64_t h = (uint64_t)len;
    for (int i = 0; i < 16; i += 4)
        h ^= (t[i] + ROTR64(t[i + 1], 11)) ^ (t[i + 2] + ROTR64(t[i + 3], 37));
    return owl_seal(h);
}
#endif /* RIVERBANK_AVX2 */

uint64_t kernel(const unsigned char *data, size_t len) {
#ifdef RIVERBANK_AVX2
    if (len >= 256)                     /* enough ground for a full stride */
        return riverbank_grid(data, len);
#endif
    return riverbank_walk(data, len);   /* otherwise walk the riverbank */
}
