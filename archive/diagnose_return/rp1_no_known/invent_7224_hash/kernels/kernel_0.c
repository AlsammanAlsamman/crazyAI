#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------------------------------------------------------- *
 *  The riverbank, the mountain path, and the owl.                  *
 *  No multiplication touches a byte; the only two multiplies in    *
 *  the whole working are the owl's seal, done once.                *
 * ---------------------------------------------------------------- */

#define ROTL64(x, r) (((uint64_t)(x) << (r)) | ((uint64_t)(x) >> (64 - (r))))

/* the sixteen flat stones at the riverbank */
static const uint64_t RIVER[16] = {
    0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL,
    0x2545f4914f6cdd1dULL, 0xc2b2ae3d27d4eb4fULL, 0x165667b19e3779f9ULL,
    0x85ebca77c2b2ae63ULL, 0x27d4eb2f165667c5ULL, 0x6a09e667f3bcc909ULL,
    0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL,
    0x5be0cd19137e2179ULL
};

/* the owl takes the pen: fixed size, fixed cost, not invertible back to marks */
static inline uint64_t owl_seal(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32;
    return x;
}

/* the deer's stride: uphill twist, downhill fold (one constant pair per row) */
static inline void deer_stride(uint64_t *st) {
    int j;
    for (j = 0; j < 4; j++) { st[j]      = ROTL64(st[j], 23);      st[j]      ^= st[j]      >> 29; }
    for (j = 0; j < 4; j++) { st[4 + j]  = ROTL64(st[4 + j], 37);  st[4 + j]  ^= st[4 + j]  >> 17; }
    for (j = 0; j < 4; j++) { st[8 + j]  = ROTL64(st[8 + j], 13);  st[8 + j]  ^= st[8 + j]  >> 41; }
    for (j = 0; j < 4; j++) { st[12 + j] = ROTL64(st[12 + j], 47); st[12 + j] ^= st[12 + j] >> 11; }
}

/* searching in squares: turn the shape ninety degrees, fold the corners
   into its own centre.  Bijective on the whole 1024-bit state.          */
static inline void sq_mix(uint64_t *st) {
    uint64_t t[4];
    int j;
    for (j = 0; j < 4; j++) t[j] = st[4  + ((j + 1) & 3)];
    for (j = 0; j < 4; j++) st[4  + j] = t[j];
    for (j = 0; j < 4; j++) t[j] = st[8  + ((j + 2) & 3)];
    for (j = 0; j < 4; j++) st[8  + j] = t[j];
    for (j = 0; j < 4; j++) t[j] = st[12 + ((j + 3) & 3)];
    for (j = 0; j < 4; j++) st[12 + j] = t[j];

    for (j = 0; j < 4; j++) { st[j]     += st[4 + j];  st[12 + j] ^= st[j];     st[12 + j] = ROTL64(st[12 + j], 32); }
    for (j = 0; j < 4; j++) { st[8 + j] += st[12 + j]; st[4 + j]  ^= st[8 + j]; st[4 + j]  = ROTL64(st[4 + j], 24); }
    for (j = 0; j < 4; j++) { st[j]     += st[4 + j];  st[12 + j] ^= st[j];     st[12 + j] = ROTL64(st[12 + j], 16); }
    for (j = 0; j < 4; j++) { st[8 + j] += st[12 + j]; st[4 + j]  ^= st[8 + j]; st[4 + j]  = ROTL64(st[4 + j], 21); }
}

/* ---- regime 1: the pile is too small to fill a heap.  No mountain path,
        no grid, no square rounds: four stones at the riverbank, then the owl. */
static inline uint64_t riverbank(const unsigned char *data, size_t len) {
    uint64_t s[4];
    size_t i = 0, k = 0;
    s[0] = RIVER[0]  ^ (uint64_t)len;
    s[1] = RIVER[5];
    s[2] = RIVER[10] ^ ROTL64((uint64_t)len, 32);
    s[3] = RIVER[15];
    while (i + 8 <= len) {
        uint64_t w, v;
        memcpy(&w, data + i, 8);
        v = ROTL64(s[k] ^ w, 29);
        v ^= v >> 31;
        s[k] = v;
        k = (k + 1) & 3;
        s[k] += v;                    /* the stone carried forward */
        i += 8;
    }
    if (i < len) {
        uint64_t w = 0, v;
        size_t b = 0;
        while (i < len) { w |= (uint64_t)data[i] << (b * 8); i++; b++; }
        v = ROTL64(s[k] ^ w, 41);
        v ^= v >> 23;
        s[k] = v;
        s[(k + 1) & 3] += v;
    }
    {
        uint64_t x = (s[0] + ROTL64(s[1], 13)) ^ (s[2] + ROTL64(s[3], 29));
        return owl_seal(x ^ ROTL64(s[1] ^ s[3], 47) ^ (uint64_t)len);
    }
}

#if defined(__AVX2__)
#define VROTL(x, r) _mm256_or_si256(_mm256_slli_epi64((x), (r)), _mm256_srli_epi64((x), 64 - (r)))
#define VXSR(x, s)  _mm256_xor_si256((x), _mm256_srli_epi64((x), (s)))
#endif

uint64_t kernel(const unsigned char *data, size_t len) {
    uint64_t st[16];
    size_t i, nb, rem;
    const unsigned char *p0, *p1, *mid;

    /* the deer needs enough ground for one stride */
    if (len < 128) return riverbank(data, len);

    for (i = 0; i < 16; i++) st[i] = RIVER[i];
    st[0]  ^= (uint64_t)len;
    st[15] ^= ROTL64((uint64_t)len, 32);

    nb  = len >> 7;                   /* heaps of even count: 128 marks each */
    rem = len - (nb << 7);
    p0  = data;                       /* cursor at the riverbank            */
    p1  = data + len - (nb << 6);     /* cursor at the midstream stone      */
    mid = data + (nb << 6);           /* the marks left in the middle       */

#if defined(__AVX2__)
    {
        __m256i A = _mm256_loadu_si256((const __m256i *)(st + 0));
        __m256i B = _mm256_loadu_si256((const __m256i *)(st + 4));
        __m256i C = _mm256_loadu_si256((const __m256i *)(st + 8));
        __m256i D = _mm256_loadu_si256((const __m256i *)(st + 12));
        for (i = 0; i < nb; i++) {
            __m256i h0 = _mm256_loadu_si256((const __m256i *)(p0));
            __m256i h1 = _mm256_loadu_si256((const __m256i *)(p0 + 32));
            __m256i t0 = _mm256_loadu_si256((const __m256i *)(p1));
            __m256i t1 = _mm256_loadu_si256((const __m256i *)(p1 + 32));
            p0 += 64; p1 += 64;
            /* every mark laid on its own stone: order is placement, not time */
            A = _mm256_xor_si256(A, h0);
            B = _mm256_xor_si256(B, h1);
            C = _mm256_xor_si256(C, t0);
            D = _mm256_xor_si256(D, t1);
            /* the deer's stride */
            A = VROTL(A, 23); A = VXSR(A, 29);
            B = VROTL(B, 37); B = VXSR(B, 17);
            C = VROTL(C, 13); C = VXSR(C, 41);
            D = VROTL(D, 47); D = VXSR(D, 11);
            /* turn ninety degrees */
            B = _mm256_permute4x64_epi64(B, 0x39);
            C = _mm256_permute4x64_epi64(C, 0x4E);
            D = _mm256_permute4x64_epi64(D, 0x93);
            /* fold the corners into the centre */
            A = _mm256_add_epi64(A, B); D = _mm256_xor_si256(D, A); D = VROTL(D, 32);
            C = _mm256_add_epi64(C, D); B = _mm256_xor_si256(B, C); B = VROTL(B, 24);
            A = _mm256_add_epi64(A, B); D = _mm256_xor_si256(D, A); D = VROTL(D, 16);
            C = _mm256_add_epi64(C, D); B = _mm256_xor_si256(B, C); B = VROTL(B, 21);
        }
        _mm256_storeu_si256((__m256i *)(st + 0),  A);
        _mm256_storeu_si256((__m256i *)(st + 4),  B);
        _mm256_storeu_si256((__m256i *)(st + 8),  C);
        _mm256_storeu_si256((__m256i *)(st + 12), D);
    }
#else
    for (i = 0; i < nb; i++) {
        uint64_t w[16];
        int j;
        memcpy(w,     p0, 64);
        memcpy(w + 8, p1, 64);
        p0 += 64; p1 += 64;
        for (j = 0; j < 16; j++) st[j] ^= w[j];
        deer_stride(st);
        sq_mix(st);
    }
#endif

    /* the marks left in the middle, each still on its own stone */
    {
        size_t k = 0;
        while (k + 8 <= rem) {
            uint64_t w;
            size_t idx = (k >> 3) & 15;
            memcpy(&w, mid + k, 8);
            st[idx] = ROTL64(st[idx] ^ w, 29);
            st[idx] ^= st[idx] >> 31;
            st[(idx + 7) & 15] += st[idx];
            k += 8;
        }
        if (k < rem) {
            uint64_t w = 0;
            size_t b = 0;
            while (k < rem) { w |= (uint64_t)mid[k] << (b * 8); k++; b++; }
            st[15] = ROTL64(st[15] ^ w, 41);
            st[15] ^= st[15] >> 23;
            st[0]  += st[15];
        }
    }

    /* the owl calls the shape inside the dreamer inside: two turns, no more */
    sq_mix(st);
    sq_mix(st);

    /* burn the heaps: sixteen stones down to one token, one way only */
    {
        uint64_t a0 = (st[0]  + ROTL64(st[1],  13)) ^ (st[2]  + ROTL64(st[3],  29));
        uint64_t a1 = (st[4]  + ROTL64(st[5],  41)) ^ (st[6]  + ROTL64(st[7],   7));
        uint64_t a2 = (st[8]  + ROTL64(st[9],  19)) ^ (st[10] + ROTL64(st[11], 53));
        uint64_t a3 = (st[12] + ROTL64(st[13], 23)) ^ (st[14] + ROTL64(st[15], 37));
        uint64_t x  = (a0 + ROTL64(a1, 17)) ^ (a2 + ROTL64(a3, 43));
        return owl_seal(x ^ (uint64_t)len);
    }
}
