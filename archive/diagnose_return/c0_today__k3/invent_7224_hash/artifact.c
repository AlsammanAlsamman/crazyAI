#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* nothing-up-my-sleeve constants: SHA-512 / BLAKE2b / SHA-384 IVs */
static const uint64_t IVC[16] = {
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL, 0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL, 0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL,
    0xCBBB9D5DC1059ED8ULL, 0x629A292A367CD507ULL, 0x9159015A3070DD17ULL, 0x152FECD8F70E5939ULL,
    0x67332667FFC00B31ULL, 0x8EB44A8768581511ULL, 0xDB0C2E0D64F98FA7ULL, 0x47B5481DBEFA4FA4ULL
};
#define GOLD 0x9E3779B97F4A7C15ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) { return (x << r) | (x >> (64u - r)); }
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* the owl's seal: splitmix64 finalizer -- the dreamer inside the dreamer */
static inline uint64_t fmix64(uint64_t z) {
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27; z *= 0x94D049BB133111EBULL;
    z ^= z >> 31; return z;
}

/* the musk deer's stride: uphill twist (rotl), downhill fold (xor),
   doubling-back that eats its own trail (state <- f(state)).  No multiply. */
#define QR(a,b,c,d) do {                              \
    (a) += (b); (d) ^= (a); (d) = rotl64((d), 32);     \
    (c) += (d); (b) ^= (c); (b) = rotl64((b), 24);     \
    (a) += (b); (d) ^= (a); (d) = rotl64((d), 16);     \
    (c) += (d); (b) ^= (c); (b) = rotl64((b), 63);     \
} while (0)

/* searching the result in squares: column round, then the ninety-degree
   turn that folds the corners into the centre (ChaCha's diagonals). */
#define DR(s) do {                            \
    QR((s)[0],(s)[4],(s)[ 8],(s)[12]);        \
    QR((s)[1],(s)[5],(s)[ 9],(s)[13]);        \
    QR((s)[2],(s)[6],(s)[10],(s)[14]);        \
    QR((s)[3],(s)[7],(s)[11],(s)[15]);        \
    QR((s)[0],(s)[5],(s)[10],(s)[15]);        \
    QR((s)[1],(s)[6],(s)[11],(s)[12]);        \
    QR((s)[2],(s)[7],(s)[ 8],(s)[13]);        \
    QR((s)[3],(s)[4],(s)[ 9],(s)[14]);        \
} while (0)

/* diagonal fold of the 4x4 stone into one column, then the seal */
static inline uint64_t seal(uint64_t *s, size_t len) {
    uint64_t a, b, c, d;
    s[0] ^= (uint64_t)len;
    s[7] += (uint64_t)len;
    DR(s); DR(s); DR(s);                 /* mixing concentrated at the shrine */
    a = s[0] ^ s[5] ^ s[10] ^ s[15];
    b = s[1] ^ s[6] ^ s[11] ^ s[12];
    c = s[2] ^ s[7] ^ s[ 8] ^ s[13];
    d = s[3] ^ s[4] ^ s[ 9] ^ s[14];
    QR(a, b, c, d);
    return fmix64((a ^ rotl64(c, 32)) + (b ^ rotl64(d, 19)));
}

/* ---- regime: a partial heap (16 <= len < 128): the single-column stone ---- */
static uint64_t small_square(const unsigned char *restrict d, size_t len) {
    uint64_t a = IVC[0] ^ (uint64_t)len;
    uint64_t b = IVC[1];
    uint64_t c = IVC[2];
    uint64_t e = IVC[3] ^ ((uint64_t)len << 32);
    size_t i = 0;
    while (len - i >= 32) {                     /* lay 4 marks, then race once */
        a ^= ld64(d + i);      b ^= ld64(d + i + 8);
        c ^= ld64(d + i + 16); e ^= ld64(d + i + 24);
        QR(a, b, c, e); QR(b, c, e, a);
        i += 32;
    }
    if (len - i >= 16) { a ^= ld64(d + i); b ^= ld64(d + i + 8); }
    c ^= ld64(d + len - 16);                    /* overlapping tail window */
    e ^= ld64(d + len - 8);
    QR(a, b, c, e); QR(b, c, e, a); QR(c, e, a, b); QR(e, a, b, c);
    return fmix64((a ^ rotl64(c, 32)) + (b ^ rotl64(e, 19)));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    uint64_t s[16];
    size_t nb, b;
    int i;

    /* ---- at the riverbank: can this pile fill a heap at all? ---- */
    if (len < 16) {                 /* a handful -- straight to the owl */
        uint64_t x;
        if (len >= 8) {
            x = (ld64(p) ^ IVC[0]) + rotl64(ld64(p + len - 8) ^ IVC[1], 32);
        } else if (len >= 4) {
            x = ((uint64_t)ld32(p) << 32) | (uint64_t)ld32(p + len - 4);
        } else if (len) {
            x = (uint64_t)p[0] | ((uint64_t)p[len >> 1] << 8)
              | ((uint64_t)p[len - 1] << 16);
        } else {
            x = 0;
        }
        return fmix64(x ^ ((uint64_t)len * GOLD));
    }
    if (len < 128) return small_square(p, len);

    for (i = 0; i < 16; i++) s[i] = IVC[i];
    s[0] ^= (uint64_t)len;
    nb = len >> 7;

#if defined(__AVX2__)
    {
        const __m256i r24 = _mm256_setr_epi8(5,6,7,0,1,2,3,4, 13,14,15,8,9,10,11,12,
                                             5,6,7,0,1,2,3,4, 13,14,15,8,9,10,11,12);
        const __m256i r16 = _mm256_setr_epi8(6,7,0,1,2,3,4,5, 14,15,8,9,10,11,12,13,
                                             6,7,0,1,2,3,4,5, 14,15,8,9,10,11,12,13);
        const __m256i inc = _mm256_setr_epi64x(0, 0, 0, 1);
        __m256i ctr = _mm256_setzero_si256();
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(s + 0));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(s + 4));
        __m256i v2 = _mm256_loadu_si256((const __m256i *)(s + 8));
        __m256i v3 = _mm256_loadu_si256((const __m256i *)(s + 12));
#define VR32(x) _mm256_shuffle_epi32((x), 0xB1)
#define VR24(x) _mm256_shuffle_epi8((x), r24)
#define VR16(x) _mm256_shuffle_epi8((x), r16)
#define VR63(x) _mm256_or_si256(_mm256_slli_epi64((x),63), _mm256_srli_epi64((x),1))
#define VQR(a,b,c,d) do {                                                   \
    a = _mm256_add_epi64(a,b); d = _mm256_xor_si256(d,a); d = VR32(d);      \
    c = _mm256_add_epi64(c,d); b = _mm256_xor_si256(b,c); b = VR24(b);      \
    a = _mm256_add_epi64(a,b); d = _mm256_xor_si256(d,a); d = VR16(d);      \
    c = _mm256_add_epi64(c,d); b = _mm256_xor_si256(b,c); b = VR63(b);      \
} while (0)
        for (b = 0; b < nb; b++) {
            const unsigned char *q = p + (b << 7);
            /* silence: the whole heap is laid on the stone before any racing */
            v0 = _mm256_xor_si256(v0, _mm256_loadu_si256((const __m256i *)(q +  0)));
            v1 = _mm256_xor_si256(v1, _mm256_loadu_si256((const __m256i *)(q + 32)));
            v2 = _mm256_xor_si256(v2, _mm256_loadu_si256((const __m256i *)(q + 64)));
            v3 = _mm256_xor_si256(v3, _mm256_loadu_si256((const __m256i *)(q + 96)));
            ctr = _mm256_add_epi64(ctr, inc);   /* the stone that never ossifies */
            v3  = _mm256_add_epi64(v3, ctr);
            VQR(v0, v1, v2, v3);                            /* columns */
            v1 = _mm256_permute4x64_epi64(v1, 0x39);        /* ninety degrees */
            v2 = _mm256_permute4x64_epi64(v2, 0x4E);
            v3 = _mm256_permute4x64_epi64(v3, 0x93);
            VQR(v0, v1, v2, v3);                            /* diagonals */
            v1 = _mm256_permute4x64_epi64(v1, 0x93);
            v2 = _mm256_permute4x64_epi64(v2, 0x4E);
            v3 = _mm256_permute4x64_epi64(v3, 0x39);
        }
        _mm256_storeu_si256((__m256i *)(s +  0), v0);
        _mm256_storeu_si256((__m256i *)(s +  4), v1);
        _mm256_storeu_si256((__m256i *)(s +  8), v2);
        _mm256_storeu_si256((__m256i *)(s + 12), v3);
#undef VQR
#undef VR63
#undef VR16
#undef VR24
#undef VR32
    }
#else
    for (b = 0; b < nb; b++) {
        const unsigned char *q = p + (b << 7);
        for (i = 0; i < 16; i++) s[i] ^= ld64(q + 8 * i);
        s[15] += (uint64_t)(b + 1);
        DR(s);
    }
#endif

    if ((nb << 7) < len) {                 /* ragged end: overlapping last heap */
        const unsigned char *q = p + len - 128;
        for (i = 0; i < 16; i++) s[i] ^= ld64(q + 8 * i);
        s[15] += (uint64_t)(nb + 1);
        DR(s);
    }
    return seal(s, len);
}
