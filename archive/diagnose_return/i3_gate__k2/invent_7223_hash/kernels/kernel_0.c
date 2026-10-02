#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the sphere's four faces: the sole carried memory (SEED 1) ----
   no table, no history, no allocation anywhere in this file.        */
#define K0 0x9E3779B97F4A7C15ULL
#define K1 0xBF58476D1CE4E5B9ULL
#define K2 0x94D049BB133111EBULL
#define K3 0xD6E8FEB86659FD93ULL

/* a tumble: one turn of the sphere (compiles to a single ROL) */
#define ROTL(x,r) (((x) << (r)) | ((x) >> (64 - (r))))

/* a stalk: the sphere cracks into itself and the crack's angle is carried
   forward.  add + rotate + xor only -- no multiplication (SEED 2).      */
#define STALK(A,B,R) do { (A) ^= (B); (B) = ROTL((B),(R)); (A) += (B); } while (0)

/* the last stalk, read as the token (SEED 3): twelve stalks = three full
   turns of the coil, so every face is struck three times as target and
   three times as source.  Jenkins' SpookyHash ShortEnd chain.          */
#define TRAIL(a,b,c,d) do {                     \
    STALK(d,c,15); STALK(a,d,52);               \
    STALK(b,a,26); STALK(c,b,51);               \
    STALK(d,c,28); STALK(a,d, 9);               \
    STALK(b,a,47); STALK(c,b,54);               \
    STALK(d,c,32); STALK(a,d,25);               \
    STALK(b,a,63); STALK(c,b,41);               \
} while (0)

/* one mark pressed into one face, then its fixed count of tumbles.
   x ^ rotl(x,47) alone would be rank-63 (2-to-1); the preceding
   += rotl(x,29) supplies carries and nonlinearity, so no bit is lost. */
#define LANE(v,w) do { (v) ^= (w); (v) += ROTL((v),29); (v) ^= ROTL((v),47); } while (0)

#if defined(__AVX2__)
/* the same stalk, walked by four courses at once */
#define VLANE(v,w) do {                                                     \
    __m256i t_  = _mm256_xor_si256((v), (w));                               \
    __m256i r1_ = _mm256_or_si256(_mm256_slli_epi64(t_,29),                 \
                                  _mm256_srli_epi64(t_,35));                \
    t_ = _mm256_add_epi64(t_, r1_);                                         \
    __m256i r2_ = _mm256_or_si256(_mm256_slli_epi64(t_,47),                 \
                                  _mm256_srli_epi64(t_,17));                \
    (v) = _mm256_xor_si256(t_, r2_);                                        \
} while (0)
#endif

static inline uint64_t ld64(const unsigned char *p)
{
    uint64_t v;
    memcpy(&v, p, 8);          /* single MOVQ at -O3; endian-stable per machine */
    return v;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;   /* vectorization hint, contract untouched */
    uint64_t h0 = K0, h1 = K1, h2 = K2, h3 = K3 ^ (uint64_t)len;

    /* ---- regime 1: a pile I can hold in one hand ---------------------
       pressed straight onto the four faces; the braided course's
       fold-and-finish epilogue is never paid at this size.             */
    if (len < 32) {
        unsigned char pad[32];
        if (len) memcpy(pad, p, len);
        memset(pad + len, 0, 32 - len);
        h0 ^= ld64(pad);        h1 ^= ld64(pad +  8);
        h2 ^= ld64(pad + 16);   h3 ^= ld64(pad + 24);
        TRAIL(h0, h1, h2, h3);
        return h2;                      /* the last, smallest crack */
    }

#if defined(__AVX2__)
    /* ---- regime 3: a pile taller than I am, river high ----
       eight courses, 64 bytes a turn.                        */
    if (len >= 256) {
        static const uint64_t vk[8] = { K0, K1, K2, K3,
                                        K0^K1, K1^K2, K2^K3, K3^K0 };
        __m256i v0 = _mm256_loadu_si256((const __m256i *)vk);
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(vk + 4));
        uint64_t L[8];
        size_t nb = len >> 6, i;

        for (i = 0; i < nb; i++) {
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(p));
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(p + 32));
            VLANE(v0, w0);
            VLANE(v1, w1);
            p += 64;
        }
        if (len & 63) {                 /* the overlapping last course */
            const unsigned char *q = data + len - 64;
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(q));
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(q + 32));
            VLANE(v0, w0);
            VLANE(v1, w1);
        }
        _mm256_storeu_si256((__m256i *)L,       v0);
        _mm256_storeu_si256((__m256i *)(L + 4), v1);
        h0 ^= L[0]; h1 ^= L[1]; h2 ^= L[2]; h3 ^= L[3];
        h0 += L[4]; h1 += L[5]; h2 += L[6]; h3 += L[7];
        TRAIL(h0, h1, h2, h3);
        return h2;
    }
#endif

    /* ---- regime 2: the four braided courses, 32 bytes a turn ---- */
    {
        uint64_t v0 = K0, v1 = K1, v2 = K2, v3 = K3;
        size_t nb = len >> 5, i;

        for (i = 0; i < nb; i++) {
            LANE(v0, ld64(p     ));
            LANE(v1, ld64(p +  8));
            LANE(v2, ld64(p + 16));
            LANE(v3, ld64(p + 24));
            p += 32;
        }
        if (len & 31) {                 /* overlapping last course: no byte loop */
            const unsigned char *q = data + len - 32;
            LANE(v0, ld64(q     ));
            LANE(v1, ld64(q +  8));
            LANE(v2, ld64(q + 16));
            LANE(v3, ld64(q + 24));
        }
        h0 ^= v0; h1 ^= v1; h2 ^= v2; h3 ^= v3;
        TRAIL(h0, h1, h2, h3);
        return h2;
    }
}
