#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define KERNEL_AVX2 1
#endif

/* "flip + one plain quarter-turn": fixed, data-independent rotations.
   All odd => coprime to 64 => full 64-position orbit, no short cycles. */
#define ROT_A 33u
#define ROT_B 23u
#define ROT_C 41u
#define ROT_D 13u

#define KC0 0x243F6A8885A308D3ULL
#define KC1 0x13198A2E03707344ULL
#define KC2 0xA4093822299F31D0ULL
#define KC3 0x082EFA98EC4E6C89ULL

static inline uint64_t k_rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

static inline uint64_t k_ld64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

/* splitmix64 finalizer: bijective, ~0.5 avalanche. O(1), not per byte. */
static inline uint64_t k_final(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

#ifdef KERNEL_AVX2
static inline __m256i k_rot256(__m256i v, const int r) {
    return _mm256_or_si256(_mm256_slli_epi64(v, r),
                           _mm256_srli_epi64(v, 64 - r));
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *p = data;
    size_t n = len;
    uint64_t h0 = KC0, h1 = KC1, h2 = KC2, h3 = KC3;
    uint64_t h;

#ifdef KERNEL_AVX2
    if (n >= 128) {
        __m256i a0 = _mm256_set1_epi64x((long long)KC0);
        __m256i a1 = _mm256_set1_epi64x((long long)KC1);
        __m256i a2 = _mm256_set1_epi64x((long long)KC2);
        __m256i a3 = _mm256_set1_epi64x((long long)KC3);
        uint64_t t[16];
        int i;

        do {
            __m256i v0 = _mm256_loadu_si256((const __m256i *)(p +  0));
            __m256i v1 = _mm256_loadu_si256((const __m256i *)(p + 32));
            __m256i v2 = _mm256_loadu_si256((const __m256i *)(p + 64));
            __m256i v3 = _mm256_loadu_si256((const __m256i *)(p + 96));
            /* fixed flip, then overlay: xor (pure) / add (blended, carries) */
            a0 = _mm256_xor_si256(k_rot256(a0, ROT_A), v0);
            a1 = _mm256_add_epi64(k_rot256(a1, ROT_B), v1);
            a2 = _mm256_xor_si256(k_rot256(a2, ROT_C), v2);
            a3 = _mm256_add_epi64(k_rot256(a3, ROT_D), v3);
            p += 128;
            n -= 128;
        } while (n >= 128);

        _mm256_storeu_si256((__m256i *)&t[0],  a0);
        _mm256_storeu_si256((__m256i *)&t[4],  a1);
        _mm256_storeu_si256((__m256i *)&t[8],  a2);
        _mm256_storeu_si256((__m256i *)&t[12], a3);
        for (i = 0; i < 16; i++)
            h0 = k_rotl64(h0, ROT_A) ^ t[i];
    }
#endif

    while (n >= 32) {
        h0 = k_rotl64(h0, ROT_A) ^ k_ld64(p +  0);
        h1 = k_rotl64(h1, ROT_B) + k_ld64(p +  8);
        h2 = k_rotl64(h2, ROT_C) ^ k_ld64(p + 16);
        h3 = k_rotl64(h3, ROT_D) + k_ld64(p + 24);
        p += 32;
        n -= 32;
    }

    h = k_rotl64(h0, 11) ^ k_rotl64(h1, 27)
      ^ k_rotl64(h2, 43) ^ k_rotl64(h3, 59);

    while (n >= 8) {
        h = k_rotl64(h, ROT_A) ^ k_ld64(p);
        p += 8;
        n -= 8;
    }
    if (n) {                       /* 1..7 bytes, no over-read */
        uint64_t w = 0;
        memcpy(&w, p, n);
        h = k_rotl64(h, ROT_C) ^ w;
    }

    h ^= (uint64_t)len;
    return k_final(h);
}
