/* "Eight strings under one bridge": wide-state ARX sponge.
 *   per byte  : one add of a position-unique constant  (no multiply, no round)
 *   per block : one butterfly pluck, 8 x 64-bit state  (constant force)
 *   once      : one storm pass + irreversible 512->64 fold (the only multiplies)
 * Portable scalar path and AVX2 path compute the identical function.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROTL64(x,n) (((x) << (n)) | ((x) >> (64 - (n))))

/* the eight strings' resting pitches */
static const uint64_t KIV[8] = {
    0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
    0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
    0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL,
    0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL
};
/* the inkwell that never runs dry: strides giving a fresh, never-repeated
   ink-weight F[lane] = (block+1)*KSTR[lane] for every mark */
static const uint64_t KSTR[8] = {
    0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL,
    0x85EBCA77C2B2AE63ULL, 0x2545F4914F6CDD1DULL,
    0xD6E8FEB86659FD93ULL, 0xA24BAED4963EE407ULL
};

/* the storm: the sky's dissolving pigment. One pass. The only multiplies in
   the kernel. (MurmurHash3 fmix64 - validated, not invented.) */
static inline uint64_t storm64(uint64_t h) {
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 32;
    return h;
}

static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* REGIME 2: a pile too small to span all eight strings never reaches the
   bridge. Two strings, no storm pass, one fmix64. */
static uint64_t short_pile(const unsigned char *d, size_t len) {
    uint64_t a = KIV[0] ^ (uint64_t)len;
    uint64_t b = KIV[5] + ROTL64((uint64_t)len, 29);
    if (len >= 16) {
        size_t i = 0;
        do {
            a += ld64(d + i)     + KSTR[0]; a = ROTL64(a, 29); a ^= b;
            b += ld64(d + i + 8) + KSTR[5]; b = ROTL64(b, 41); b ^= a;
            i += 16;
        } while (i + 16 <= len);
        if (i < len) {                      /* last 16 bytes, fresh feathers */
            a += ld64(d + len - 16) + KSTR[2]; a = ROTL64(a, 31); a ^= b;
            b += ld64(d + len - 8)  + KSTR[6]; b = ROTL64(b, 43); b ^= a;
        }
    } else if (len >= 8) {
        a += ld64(d)           + KSTR[0]; a = ROTL64(a, 29); a ^= b;
        b += ld64(d + len - 8) + KSTR[5]; b = ROTL64(b, 41); b ^= a;
    } else if (len >= 4) {
        a += (uint64_t)ld32(d)           + KSTR[1]; a = ROTL64(a, 27); a ^= b;
        b += (uint64_t)ld32(d + len - 4) + KSTR[6]; b = ROTL64(b, 43); b ^= a;
    } else if (len > 0) {
        uint64_t t = (uint64_t)d[0];
        t |= (uint64_t)d[len >> 1] << 8;
        t |= (uint64_t)d[len - 1] << 16;
        a += t + KSTR[3]; a = ROTL64(a, 25); a ^= b;
    }
    return storm64(a ^ ROTL64(b, 27));
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * __restrict p = data;
    uint64_t s[8];

    /* the count goes onto the counting-bars first: regime check */
    if (len < 64) return short_pile(p, len);

#if defined(__AVX2__)
    {
    #define VROTL(x,n) _mm256_or_si256(_mm256_slli_epi64((x),(n)), _mm256_srli_epi64((x),64-(n)))
    #define PERM1(x)   _mm256_permute4x64_epi64((x), 0x39)   /* lane i <- i+1 */
    #define PERM3(x)   _mm256_permute4x64_epi64((x), 0x93)   /* lane i <- i+3 */
    /* one pluck: constant force, whole instrument answers, feathers advance */
    #define PLUCK(MA,MB,R1,R2) do {                                            \
            __m256i tA = _mm256_add_epi64(_mm256_add_epi64(A,(MA)), FA);        \
            __m256i tB = _mm256_add_epi64(_mm256_add_epi64(B,(MB)), FB);        \
            A = _mm256_xor_si256(PERM1(tA), VROTL(tB,(R1)));                   \
            B = _mm256_add_epi64(PERM3(tB), VROTL(tA,(R2)));                   \
            FA = _mm256_add_epi64(FA, SA); FB = _mm256_add_epi64(FB, SB);      \
        } while (0)

        const __m256i L = _mm256_set1_epi64x((long long)len);
        __m256i A  = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)KIV), L);
        __m256i B  = _mm256_add_epi64(_mm256_loadu_si256((const __m256i*)(KIV+4)), L);
        const __m256i SA = _mm256_loadu_si256((const __m256i*)KSTR);
        const __m256i SB = _mm256_loadu_si256((const __m256i*)(KSTR+4));
        __m256i FA = SA, FB = SB;
        size_t i = 0;

        for (; i + 64 <= len; i += 64) {
            __m256i MA = _mm256_loadu_si256((const __m256i*)(p + i));
            __m256i MB = _mm256_loadu_si256((const __m256i*)(p + i + 32));
            PLUCK(MA, MB, 23, 40);
        }
        if (i < len) {                       /* tail: padded, never doubling back */
            unsigned char pad[64];
            memset(pad, 0, 64);
            memcpy(pad, p + i, len - i);
            PLUCK(_mm256_loadu_si256((const __m256i*)pad),
                  _mm256_loadu_si256((const __m256i*)(pad + 32)), 29, 47);
        }
        {   /* THE STORM: one pass, hands off the marks, no message at all */
            const __m256i Z = _mm256_setzero_si256();
            PLUCK(Z, Z, 17, 31); PLUCK(Z, Z, 23, 40); PLUCK(Z, Z, 29, 47);
            PLUCK(Z, Z, 37, 53); PLUCK(Z, Z, 11, 19); PLUCK(Z, Z, 43, 59);
        }
        _mm256_storeu_si256((__m256i*)s,       A);
        _mm256_storeu_si256((__m256i*)(s + 4), B);
    #undef PLUCK
    #undef PERM3
    #undef PERM1
    #undef VROTL
    }
#else
    {
        uint64_t f[8], m[8];
        int k; size_t i = 0;
        for (k = 0; k < 8; k++) { s[k] = KIV[k]; f[k] = KSTR[k]; }
        for (k = 0; k < 4; k++) s[k]     ^= (uint64_t)len;
        for (k = 4; k < 8; k++) s[k]     += (uint64_t)len;

    #define PLUCK(M,R1,R2) do {                                                \
            uint64_t tA[4], tB[4]; int j;                                      \
            for (j = 0; j < 4; j++) tA[j] = s[j]   + (M)[j]   + f[j];          \
            for (j = 0; j < 4; j++) tB[j] = s[j+4] + (M)[j+4] + f[j+4];        \
            for (j = 0; j < 4; j++) s[j]   = tA[(j+1)&3] ^ ROTL64(tB[j],(R1)); \
            for (j = 0; j < 4; j++) s[j+4] = tB[(j+3)&3] + ROTL64(tA[j],(R2)); \
            for (j = 0; j < 8; j++) f[j] += KSTR[j];                           \
        } while (0)

        for (; i + 64 <= len; i += 64) {
            for (k = 0; k < 8; k++) m[k] = ld64(p + i + 8*(size_t)k);
            PLUCK(m, 23, 40);
        }
        if (i < len) {
            unsigned char pad[64];
            memset(pad, 0, 64);
            memcpy(pad, p + i, len - i);
            for (k = 0; k < 8; k++) m[k] = ld64(pad + 8*(size_t)k);
            PLUCK(m, 29, 47);
        }
        {   /* THE STORM */
            uint64_t z[8]; for (k = 0; k < 8; k++) z[k] = 0;
            PLUCK(z, 17, 31); PLUCK(z, 23, 40); PLUCK(z, 29, 47);
            PLUCK(z, 37, 53); PLUCK(z, 11, 19); PLUCK(z, 43, 59);
        }
    #undef PLUCK
    }
#endif

    /* the pickers: irreversible 512 -> 64, nothing of the order survives */
    {
        uint64_t h = (s[0] + ROTL64(s[1], 11)) ^ (s[2] + ROTL64(s[3], 23));
        h += (s[4] + ROTL64(s[5], 37)) ^ (s[6] + ROTL64(s[7], 53));
        h ^= (uint64_t)len;
        return storm64(h);          /* eight small notes, no more */
    }
}
