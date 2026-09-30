#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ============ THE NARROW COIL: one sphere, one face ============ */

/* the sphere turns on the trail: a rotation -- never a multiply */
static inline uint64_t sturn(uint64_t x, unsigned r) {
    return (x << (r & 63u)) | (x >> ((64u - r) & 63u));
}

/* the angle of the hairline crack, read off the sphere's high face.
   Forced odd (1..63) so it can never coincide with the coil's even pitch:
   s ^ rot(s,pitch) ^ rot(s,angle) is then always a weight-3 rotation
   polynomial, hence a bijection -- the sphere loses no marks. */
static inline unsigned cangle(uint64_t s) {
    return (unsigned)((((s) >> 58) & 31u) << 1) | 1u;
}

#define PITCH 26u                 /* the coil's own pitch: fixed, even */

/* A STALK: the sphere sobs once, cracks a hairline into itself at the angle
   it carried here, reads the new angle to carry forward, and lets the old
   face shrink and go. */
#define STALK(s, a) do {                                  \
    (s) ^= sturn((s), PITCH) ^ sturn((s), (a));           \
    (a)  = cangle((s));                                   \
    (s) ^= (s) >> 28;                                     \
} while (0)

/* THE LAST, SMALLEST CRACK: the sphere is not kept, only this token.
   The shrink distance is itself read from the crack (bits 60..63 survive a
   right xorshift of >=17, so the step stays invertible). */
#define FROUND(s, a, pit) do {                            \
    (s) ^= sturn((s), (pit)) ^ sturn((s), (a));           \
    (s) ^= (s) >> (17u + (unsigned)((s) >> 60));          \
    (a)  = cangle((s));                                   \
} while (0)

static inline uint64_t last_crack(uint64_t s, unsigned a, uint64_t len) {
    s ^= sturn(len, 40);                  /* the tally of marks, at the final stalk */
    FROUND(s, a, 14u);
    FROUND(s, a, 22u);
    FROUND(s, a, 30u);
    FROUND(s, a,  6u);
    FROUND(s, a, 18u);
    return s;
}

#if defined(__AVX2__)
/* ============ THE WIDE COIL: one sphere, four faces ============ */

#define VTURN(x, r) _mm256_or_si256(_mm256_slli_epi64((x), (r)),           \
                                    _mm256_srli_epi64((x), 64 - (r)))

static inline __m256i vturnv(__m256i x, __m256i r) {   /* per-face crack angle */
    return _mm256_or_si256(
        _mm256_sllv_epi64(x, r),
        _mm256_srlv_epi64(x, _mm256_sub_epi64(_mm256_set1_epi64x(64), r)));
}
static inline __m256i vangle(__m256i s) {
    return _mm256_or_si256(
        _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(s, 58),
                                           _mm256_set1_epi64x(31)), 1),
        _mm256_set1_epi64x(1));
}

/* ONE TURN of the wide coil: a full turn's worth of marks (32) is pressed
   onto the sphere's four faces at once, then each face cracks at its own
   carried angle and its old face shrinks and goes. */
#define WTURN(S, A, q) do {                                               \
    __m256i W_ = _mm256_loadu_si256((const __m256i *)(q));                \
    (S) = _mm256_xor_si256((S), W_);                                      \
    (S) = _mm256_xor_si256((S), _mm256_xor_si256(VTURN((S), 26),          \
                                                 vturnv((S), (A))));      \
    (A) = vangle((S));                                                    \
    (S) = _mm256_xor_si256((S), _mm256_srli_epi64((S), 28));              \
} while (0)

/* THE STALK after a fixed count of turns (one full revolution, four faces):
   every face cracks into the others.  I + P_neighbour + P_opposite has odd
   weight in the group algebra, so this too is a bijection -- no face is lost.
   This is what makes the four faces ONE sphere and not four accumulators. */
static inline __m256i wstalk(__m256i S) {
    __m256i u = _mm256_shuffle_epi32(S, _MM_SHUFFLE(1, 0, 3, 2)); /* neighbours */
    __m256i v = _mm256_permute2x128_si256(S, S, 0x01);            /* opposites  */
    return _mm256_xor_si256(S, _mm256_xor_si256(u, v));
}
#endif

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    size_t n = len;
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;  /* sphere at the high mouth */
    unsigned a = 29u;                                    /* the mouth's own angle    */

#if defined(__AVX2__)
    /* REGIME TEST, in world terms: does the pile fill whole turns of the WIDE
       coil?  If not, the wide coil is not worth walking -- take the narrow one. */
    if (n >= 64) {
        __m256i S = _mm256_xor_si256(
            _mm256_set_epi64x((int64_t)0x452821E638D01377ULL,
                              (int64_t)0xBE5466CF34E90C6CULL,
                              (int64_t)0xC0AC29B7C97C50DDULL,
                              (int64_t)0x9E3779B97F4A7C15ULL),
            _mm256_set1_epi64x((int64_t)len));
        __m256i A = _mm256_set_epi64x(45, 19, 7, 29);    /* odd mouth angles */
        uint64_t f[4];

        while (n >= 128) {           /* a fixed count of turns: four, then a stalk */
            WTURN(S, A, p);
            WTURN(S, A, p + 32);
            WTURN(S, A, p + 64);
            WTURN(S, A, p + 96);
            S = wstalk(S);
            p += 128; n -= 128;
        }
        while (n >= 32) {            /* the coil's last short revolutions */
            WTURN(S, A, p);
            S = wstalk(S);
            p += 32; n -= 32;
        }
        _mm256_storeu_si256((__m256i *)f, S);   /* the sphere is set down: */
        s = sturn(f[0], 1) ^ sturn(f[1], 17)    /* its four faces crack into one */
          ^ sturn(f[2], 33) ^ sturn(f[3], 49);
        a = cangle(s);
    }
#endif

    while (n >= 8) {                 /* the narrow coil: one turn = 8 marks */
        uint64_t w;
        memcpy(&w, p, 8);
        s ^= w;                      /* press the turn's marks into the face */
        STALK(s, a);
        p += 8; n -= 8;
    }
    while (n != 0) {                 /* the last marks, pressed one by one */
        s ^= (uint64_t)(*p++);
        STALK(s, a);
        n--;
    }
    return last_crack(s, a, (uint64_t)len);  /* hand over only the hairline */
}
