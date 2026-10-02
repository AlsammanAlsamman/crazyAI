#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ===================================================================
   THE FOUR WIRE BIRDS.  v0,v1,v2,v3 are the four beaks.  Every crease
   (one message word) is pressed against ALL FOUR at once: the word is
   xored in, then one round carries it through all four words before the
   next mark may be read.  Folds only -- add, rotate, xor.  No stretch,
   no multiply.  The rotation amounts are the spiral and snail markings
   deliberately scrambled so that no two piles can line up.
   This round is the SipRound (Aumasson-Bernstein).
   =================================================================== */
#define ROTL(x, b) ((uint64_t)(((x) << (b)) | ((x) >> (64 - (b)))))

#define BIRDS(v0, v1, v2, v3)                                              \
    do {                                                                   \
        v0 += v1; v1 = ROTL(v1, 13); v1 ^= v0; v0 = ROTL(v0, 32);          \
        v2 += v3; v3 = ROTL(v3, 16); v3 ^= v2;                             \
        v0 += v3; v3 = ROTL(v3, 21); v3 ^= v0;                             \
        v2 += v1; v1 = ROTL(v1, 17); v1 ^= v2; v2 = ROTL(v2, 32);          \
    } while (0)

#define IV0 0x736f6d6570736575ULL   /* "somepseu" */
#define IV1 0x646f72616e646f6dULL   /* "dorandom" */
#define IV2 0x6c7967656e657261ULL   /* "lygenera" */
#define IV3 0x7465646279746573ULL   /* "tedbytes" */

#define K0  0x0706050403020100ULL
#define K1  0x0f0e0d0c0b0a0908ULL

/* "only that it's large enough" -- the regime test. Below this, one
   sheet; at or above it, four dealt sheets. 256 is the exact analytic
   break-even, so the four-sheet path is never the slower choice. */
#define SHEET_MIN ((size_t)256)

/* -------------------------------------------------------------------
   ONE SHEET, STRICTLY SERIAL: canonical SipHash-2-4.
   One crease per mark, each crease's angle set by the mark and by the
   state the previous crease left; two refolds per crease until the four
   beaks agree; then the water's edge -- the length rides in as the one
   small suitcase (top byte of the last fold), four thinning rounds with
   no new marks, and the four birds collapse to one coin.
   ------------------------------------------------------------------- */
static uint64_t fold_one_sheet(const unsigned char *in, size_t inlen,
                               uint64_t k0, uint64_t k1)
{
    uint64_t v0 = IV0 ^ k0;
    uint64_t v1 = IV1 ^ k1;
    uint64_t v2 = IV2 ^ k0;
    uint64_t v3 = IV3 ^ k1;
    uint64_t m, b = ((uint64_t)inlen) << 56;     /* the small suitcase */
    size_t   nw = inlen >> 3, left = inlen & 7u, i;

    for (i = 0; i < nw; i++) {                   /* never skip, never look ahead */
        memcpy(&m, in + (i << 3), 8);
        v3 ^= m;
        BIRDS(v0, v1, v2, v3);
        BIRDS(v0, v1, v2, v3);
        v0 ^= m;
    }
    in += (nw << 3);
    switch (left) {                              /* every trimmed scrap still folded in */
    case 7: b |= (uint64_t)in[6] << 48; /* fall through */
    case 6: b |= (uint64_t)in[5] << 40; /* fall through */
    case 5: b |= (uint64_t)in[4] << 32; /* fall through */
    case 4: b |= (uint64_t)in[3] << 24; /* fall through */
    case 3: b |= (uint64_t)in[2] << 16; /* fall through */
    case 2: b |= (uint64_t)in[1] <<  8; /* fall through */
    case 1: b |= (uint64_t)in[0];       /* fall through */
    default: break;
    }
    v3 ^= b;
    BIRDS(v0, v1, v2, v3);
    BIRDS(v0, v1, v2, v3);
    v0 ^= b;

    v2 ^= 0xffULL;                               /* hold it down at the water's edge */
    BIRDS(v0, v1, v2, v3);
    BIRDS(v0, v1, v2, v3);
    BIRDS(v0, v1, v2, v3);
    BIRDS(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;                    /* one hard dense corner */
}

#if defined(__AVX2__)
/* The four birds, each now holding its gauge reading for all four
   sheets at once: one AVX2 register per bird, one lane per sheet. */
#define VROTL(x, b) _mm256_or_si256(_mm256_slli_epi64((x), (b)),            \
                                    _mm256_srli_epi64((x), 64 - (b)))
#define VR32(x)     _mm256_shuffle_epi32((x), 0xB1)
#define VR16(x)     _mm256_shuffle_epi8((x), r16)
#define VBIRDS(V0, V1, V2, V3)                                             \
    do {                                                                   \
        V0 = _mm256_add_epi64(V0, V1); V1 = VROTL(V1, 13);                 \
        V1 = _mm256_xor_si256(V1, V0); V0 = VR32(V0);                      \
        V2 = _mm256_add_epi64(V2, V3); V3 = VR16(V3);                      \
        V3 = _mm256_xor_si256(V3, V2);                                     \
        V0 = _mm256_add_epi64(V0, V3); V3 = VROTL(V3, 21);                 \
        V3 = _mm256_xor_si256(V3, V0);                                     \
        V2 = _mm256_add_epi64(V2, V1); V1 = VROTL(V1, 17);                 \
        V1 = _mm256_xor_si256(V1, V2); V2 = VR32(V2);                      \
    } while (0)
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t v0[4], v1[4], v2[4], v3[4], coin[4];
    unsigned char wad[64];
    size_t nblk, bulk, tail, i;
    int j;

    /* ---- regime test: is the sheet large enough to be worth dealing? */
    if (len < SHEET_MIN)
        return fold_one_sheet(p, len, K0, K1);

    /* ---- four sheets, each with its own differently-scrambled markings */
    for (j = 0; j < 4; j++) {
        uint64_t s = 0x9E3779B97F4A7C15ULL * (uint64_t)(j + 1);
        uint64_t t = 0xBF58476D1CE4E5B9ULL * (uint64_t)(j + 1);
        v0[j] = IV0 ^ (K0 ^ s);
        v1[j] = IV1 ^ (K1 ^ t);
        v2[j] = IV2 ^ (K0 ^ s);
        v3[j] = IV3 ^ (K1 ^ t);
    }

    nblk = len >> 5;            /* 32 bytes dealt per step: one word per sheet */
    bulk = nblk << 5;

#if defined(__AVX2__)
    {
        const __m256i r16 = _mm256_setr_epi8(
            6, 7, 0, 1, 2, 3, 4, 5, 14, 15, 8, 9, 10, 11, 12, 13,
            6, 7, 0, 1, 2, 3, 4, 5, 14, 15, 8, 9, 10, 11, 12, 13);
        __m256i V0 = _mm256_loadu_si256((const __m256i *)v0);
        __m256i V1 = _mm256_loadu_si256((const __m256i *)v1);
        __m256i V2 = _mm256_loadu_si256((const __m256i *)v2);
        __m256i V3 = _mm256_loadu_si256((const __m256i *)v3);
        __m256i M;

        for (i = 0; i + 2 <= nblk; i += 2) {
            M  = _mm256_loadu_si256((const __m256i *)(p + (i << 5)));
            V3 = _mm256_xor_si256(V3, M);
            VBIRDS(V0, V1, V2, V3);
            V0 = _mm256_xor_si256(V0, M);
            M  = _mm256_loadu_si256((const __m256i *)(p + ((i + 1) << 5)));
            V3 = _mm256_xor_si256(V3, M);
            VBIRDS(V0, V1, V2, V3);
            V0 = _mm256_xor_si256(V0, M);
        }
        for (; i < nblk; i++) {
            M  = _mm256_loadu_si256((const __m256i *)(p + (i << 5)));
            V3 = _mm256_xor_si256(V3, M);
            VBIRDS(V0, V1, V2, V3);
            V0 = _mm256_xor_si256(V0, M);
        }
        _mm256_storeu_si256((__m256i *)v0, V0);
        _mm256_storeu_si256((__m256i *)v1, V1);
        _mm256_storeu_si256((__m256i *)v2, V2);
        _mm256_storeu_si256((__m256i *)v3, V3);
    }
#else
    for (i = 0; i < nblk; i++) {
        uint64_t m[4];
        memcpy(m, p + (i << 5), 32);
        for (j = 0; j < 4; j++) v3[j] ^= m[j];
        for (j = 0; j < 4; j++) BIRDS(v0[j], v1[j], v2[j], v3[j]);
        for (j = 0; j < 4; j++) v0[j] ^= m[j];
    }
#endif

    /* ---- each sheet thinned to its own coin (SipHash-1-3 finalisation) */
    {
        uint64_t lb = ((uint64_t)(nblk << 3)) << 56;   /* that sheet's suitcase */
        for (j = 0; j < 4; j++) {
            uint64_t a = v0[j], b = v1[j], c = v2[j], d = v3[j];
            d ^= lb;
            BIRDS(a, b, c, d);
            a ^= lb;
            c ^= 0xffULL;
            BIRDS(a, b, c, d);
            BIRDS(a, b, c, d);
            BIRDS(a, b, c, d);
            coin[j] = a ^ b ^ c ^ d;
        }
    }

    /* ---- the water's edge: the whole reduced wad -- four coins and the
           scraps the dealing could not cover -- folded one last time,
           with the true total length as the suitcase.  Nothing else
           survives; every sheet and every intermediate reading is gone. */
    memcpy(wad, coin, 32);
    tail = len - bulk;                              /* < 32 */
    if (tail) memcpy(wad + 32, p + bulk, tail);
    return fold_one_sheet(wad, 32 + tail,
                          K0 ^ (uint64_t)len, K1 ^ (uint64_t)nblk);
}
