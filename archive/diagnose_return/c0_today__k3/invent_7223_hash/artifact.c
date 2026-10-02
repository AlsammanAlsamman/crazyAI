#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROTL(x, b) (((uint64_t)(x) << (b)) | ((uint64_t)(x) >> (64 - (b))))

/* reading the angle of the crack: add-rotate-xor only, no multiplication.
   This is SipHash's round -- the validated multiply-free mixer. */
#define SIPROUND(v0, v1, v2, v3) do {                       \
    v0 += v1; v1 = ROTL(v1, 13); v1 ^= v0; v0 = ROTL(v0, 32); \
    v2 += v3; v3 = ROTL(v3, 16); v3 ^= v2;                  \
    v0 += v3; v3 = ROTL(v3, 21); v3 ^= v0;                  \
    v2 += v1; v1 = ROTL(v1, 17); v1 ^= v2; v2 = ROTL(v2, 32); \
} while (0)

static inline uint64_t load64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    size_t n = len;

    /* the sphere set at the trail's high mouth: one body, four facets */
    uint64_t v0 = 0x736f6d6570736575ULL;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL;

#if defined(__AVX2__)
    /* REGIME TEST: can the sphere complete a full coil?  (>= 32 marks) */
    if (n >= 32) {
        __m256i s = _mm256_set_epi64x((long long)v3, (long long)v2,
                                      (long long)v1, (long long)v0);
        do {
            /* press 32 marks into the sphere's face, then let it go */
            s = _mm256_xor_si256(s, _mm256_loadu_si256((const __m256i *)p));

            /* THREE TURNS -- no more, no fewer.  Each turn ends on a stalk,
               and the sphere cracks itself there.  No multiplication. */

            /* turn 1: spin about the axis (23), struck in with a carry */
            s = _mm256_add_epi64(s, _mm256_or_si256(_mm256_slli_epi64(s, 23),
                                                    _mm256_srli_epi64(s, 41)));
            /* turn 2: spin the other arc (41), struck in as a hairline */
            s = _mm256_xor_si256(s, _mm256_or_si256(_mm256_slli_epi64(s, 41),
                                                    _mm256_srli_epi64(s, 23)));
            /* turn 3: roll along the coil -- the far side of the body comes
               round to the face, so facets do not stay separate tracks */
            {
                __m256i a = _mm256_permute4x64_epi64(s, 0x93); /* lane i <- lane i-1 */
                s = _mm256_add_epi64(s, _mm256_or_si256(_mm256_slli_epi64(s, 13),
                                                        _mm256_srli_epi64(a, 51)));
            }
            p += 32; n -= 32;
        } while (n >= 32);

        {   /* the sphere, unchanged in identity, handed to the hand trail */
            uint64_t t[4];
            _mm256_storeu_si256((__m256i *)t, s);
            v0 = t[0]; v1 = t[1]; v2 = t[2]; v3 = t[3];
        }
    }
#endif

    /* the hand trail: small pile, or the tail a coil cannot hold.
       One word per press, one tumble each -- validated SipHash-1-x. */
    while (n >= 8) {
        uint64_t m = load64(p);
        v3 ^= m; SIPROUND(v0, v1, v2, v3); v0 ^= m;
        p += 8; n -= 8;
    }
    {
        uint64_t b = ((uint64_t)len) << 56;
        switch (n) {
            case 7: b |= (uint64_t)p[6] << 48; /* fall through */
            case 6: b |= (uint64_t)p[5] << 40; /* fall through */
            case 5: b |= (uint64_t)p[4] << 32; /* fall through */
            case 4: b |= (uint64_t)p[3] << 24; /* fall through */
            case 3: b |= (uint64_t)p[2] << 16; /* fall through */
            case 2: b |= (uint64_t)p[1] <<  8; /* fall through */
            case 1: b |= (uint64_t)p[0];       /* fall through */
            case 0: break;
        }
        v3 ^= b; SIPROUND(v0, v1, v2, v3); v0 ^= b;
    }

    /* the last, smallest crack in the final stalk -- the only token kept.
       Everything else (every intermediate face) is swept away on purpose. */
    v2 ^= 0xffULL;
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;
}
