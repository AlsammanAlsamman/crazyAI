#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the Weighing Hall -------------------------------------------------
   A grey wax trough of 16 cells, kept at blood heat.  Marks are PRESSED in
   (add: material displaces, carries ride upward; never xor -- the slate
   wheel turns forward only, no bite can be walked off).  After each press
   the warm wax flows sideways ONE step -- not enough to cool flat.  When
   the pile ends: one brine plunge, held until the lace stops spreading,
   then six frost-flowers are read off and sketched to tin.               */

static inline uint64_t rotl64(uint64_t x, int n){ return (x << n) | (x >> (64 - n)); }
static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v, p, 8); return v; }

/* crystallisation: Murmur3 fmix64 (validated finaliser, paid once) */
static inline uint64_t crystallize(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32; return x;
}

/* ---- narrow channel: a pile too small to warm the whole trough -------- */
#define NSPREAD do {               \
        c1 += rotl64(c0, 13);      \
        c3 += rotl64(c2, 29);      \
        c0 += rotl64(c3, 41);      \
        c2 += rotl64(c1, 53);      \
    } while (0)

static uint64_t narrow_channel(const unsigned char *data, size_t len){
    uint64_t c0 = 0x243f6a8885a308d3ULL, c1 = 0x13198a2e03707344ULL;
    uint64_t c2 = 0xa4093822299f31d0ULL, c3 = 0x082efa98ec4e6c89ULL;
    const unsigned char * restrict p = data;
    size_t rem = len;

    c0 += (uint64_t)len;                       /* the pile is weighed first */

    while (rem >= 32){
        c0 += rd64(p);      c1 += rd64(p + 8);
        c2 += rd64(p + 16); c3 += rd64(p + 24);
        NSPREAD;
        p += 32; rem -= 32;
    }
    while (rem >= 8){ c0 += rd64(p); NSPREAD; p += 8; rem -= 8; }
    {   /* scrape the last marks in, with a pad stop so 0-bytes still count */
        uint64_t t = 0;
        if (rem) memcpy(&t, p, rem);
        t += (uint64_t)1 << (8 * rem);         /* rem is 0..7 */
        c1 += t; NSPREAD;
    }
    { int i; for (i = 0; i < 4; i++) NSPREAD; } /* short plunge */

    {   /* six frost-flowers, no two alike */
        uint64_t f = c0
                   + rotl64(c1, 17) + rotl64(c2, 31) + rotl64(c3, 47)
                   + rotl64(c0,  5) + rotl64(c2, 53);
        return crystallize(f);
    }
}

#if !defined(__AVX2__)
/* ---- the full trough, scalar ----------------------------------------- */
static const uint64_t WAX[16] = {
    0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL, 0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
    0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL, 0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
    0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL, 0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
    0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL, 0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
};
/* four shears: two independent, two coupling; each writes a group it does
   not read, so every step is invertible and the loops vectorise.          */
static inline void spread16(uint64_t * restrict s){
    static const int RA[4] = {13,29,41,53}, RB[4] = { 7,19,37,47};
    static const int RC[4] = {11,23,43,59}, RD[4] = {17,31, 5,61};
    int i;
    for (i = 0; i < 4; i++) s[ 4 + i] += rotl64(s[ 0 + ((i + 1) & 3)], RA[i]);
    for (i = 0; i < 4; i++) s[12 + i] += rotl64(s[ 8 + ((i + 2) & 3)], RB[i]);
    for (i = 0; i < 4; i++) s[ 0 + i] += rotl64(s[12 + ((i + 3) & 3)], RC[i]);
    for (i = 0; i < 4; i++) s[ 8 + i] += rotl64(s[ 4 + ((i + 1) & 3)], RD[i]);
}
#endif

uint64_t kernel(const unsigned char *data, size_t len){
    /* regime: can this pile fill the trough at all? */
    if (len < 128) return narrow_channel(data, len);

#if defined(__AVX2__)
    {
        const __m256i L1 = _mm256_setr_epi64x(13,29,41,53), Q1 = _mm256_setr_epi64x(51,35,23,11);
        const __m256i L2 = _mm256_setr_epi64x( 7,19,37,47), Q2 = _mm256_setr_epi64x(57,45,27,17);
        const __m256i L3 = _mm256_setr_epi64x(11,23,43,59), Q3 = _mm256_setr_epi64x(53,41,21, 5);
        const __m256i L4 = _mm256_setr_epi64x(17,31, 5,61), Q4 = _mm256_setr_epi64x(47,33,59, 3);
#define ROLV(x,l,q) _mm256_or_si256(_mm256_sllv_epi64((x),(l)), _mm256_srlv_epi64((x),(q)))
/* one step of warm wax flowing sideways: four invertible shears */
#define SPREAD do {                                                                        \
        B = _mm256_add_epi64(B, _mm256_permute4x64_epi64(ROLV(A,L1,Q1), 0x39));            \
        D = _mm256_add_epi64(D, _mm256_permute4x64_epi64(ROLV(C,L2,Q2), 0x4E));            \
        A = _mm256_add_epi64(A, _mm256_permute4x64_epi64(ROLV(D,L3,Q3), 0x93));            \
        C = _mm256_add_epi64(C, _mm256_permute4x64_epi64(ROLV(B,L4,Q4), 0x39));            \
    } while (0)

        __m256i A = _mm256_setr_epi64x((long long)0x243f6a8885a308d3ULL,(long long)0x13198a2e03707344ULL,
                                       (long long)0xa4093822299f31d0ULL,(long long)0x082efa98ec4e6c89ULL);
        __m256i B = _mm256_setr_epi64x((long long)0x452821e638d01377ULL,(long long)0xbe5466cf34e90c6cULL,
                                       (long long)0xc0ac29b7c97c50ddULL,(long long)0x3f84d5b5b5470917ULL);
        __m256i C = _mm256_setr_epi64x((long long)0x9216d5d98979fb1bULL,(long long)0xd1310ba698dfb5acULL,
                                       (long long)0x2ffd72dbd01adfb7ULL,(long long)0xb8e1afed6a267e96ULL);
        __m256i D = _mm256_setr_epi64x((long long)0xba7c9045f12c7f99ULL,(long long)0x24a19947b3916cf7ULL,
                                       (long long)0x0801f2e2858efc16ULL,(long long)0x636920d871574e69ULL);
        const unsigned char * restrict p = data;
        size_t n = len >> 7, r = len & 127, i;

        A = _mm256_add_epi64(A, _mm256_set1_epi64x((long long)len));   /* weigh the pile */

        for (i = 0; i < n; i++, p += 128){
            A = _mm256_add_epi64(A, _mm256_loadu_si256((const __m256i *)(const void *)(p      )));
            B = _mm256_add_epi64(B, _mm256_loadu_si256((const __m256i *)(const void *)(p +  32)));
            C = _mm256_add_epi64(C, _mm256_loadu_si256((const __m256i *)(const void *)(p +  64)));
            D = _mm256_add_epi64(D, _mm256_loadu_si256((const __m256i *)(const void *)(p +  96)));
            SPREAD;                       /* ONE step: never cools flat */
        }
        {   unsigned char pad[128];
            memset(pad, 0, sizeof pad);
            if (r) memcpy(pad, p, r);
            pad[r] = 0x01;                /* r < 128 */
            A = _mm256_add_epi64(A, _mm256_loadu_si256((const __m256i *)(const void *)(pad      )));
            B = _mm256_add_epi64(B, _mm256_loadu_si256((const __m256i *)(const void *)(pad +  32)));
            C = _mm256_add_epi64(C, _mm256_loadu_si256((const __m256i *)(const void *)(pad +  64)));
            D = _mm256_add_epi64(D, _mm256_loadu_si256((const __m256i *)(const void *)(pad +  96)));
            SPREAD;
        }
        for (i = 0; i < 12; i++) SPREAD;  /* the brine plunge, held */

        {   uint64_t o[16]; uint64_t f;
            _mm256_storeu_si256((__m256i *)(void *)(o     ), A);
            _mm256_storeu_si256((__m256i *)(void *)(o +  4), B);
            _mm256_storeu_si256((__m256i *)(void *)(o +  8), C);
            _mm256_storeu_si256((__m256i *)(void *)(o + 12), D);
            f = o[0]
              + rotl64(o[ 5], 19) + rotl64(o[10], 37) + rotl64(o[15], 52)
              + rotl64(o[ 3], 29) + rotl64(o[ 9],  7);      /* six flowers */
            return crystallize(f);
        }
#undef SPREAD
#undef ROLV
    }
#else
    {
        uint64_t s[16];
        const unsigned char * restrict p = data;
        size_t n = len >> 7, r = len & 127, i; int j;

        memcpy(s, WAX, sizeof s);
        s[0] += (uint64_t)len;

        for (i = 0; i < n; i++, p += 128){
            for (j = 0; j < 16; j++) s[j] += rd64(p + 8 * j);
            spread16(s);
        }
        {   unsigned char pad[128];
            memset(pad, 0, sizeof pad);
            if (r) memcpy(pad, p, r);
            pad[r] = 0x01;
            for (j = 0; j < 16; j++) s[j] += rd64(pad + 8 * j);
            spread16(s);
        }
        for (i = 0; i < 12; i++) spread16(s);
        {   uint64_t f = s[0]
                       + rotl64(s[ 5], 19) + rotl64(s[10], 37) + rotl64(s[15], 52)
                       + rotl64(s[ 3], 29) + rotl64(s[ 9],  7);
            return crystallize(f);
        }
    }
#endif
}
