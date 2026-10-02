#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* the sphere's four faces at the trail's high mouth */
#define SPHERE_A 0x736f6d6570736575ULL
#define SPHERE_B 0x646f72616e646f6dULL
#define SPHERE_C 0x6c7967656e657261ULL
#define SPHERE_D 0x7465646279746573ULL
#define GOLD     0x9E3779B97F4A7C15ULL

static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

/* ---- the last, smallest crack in the final stalk: O(1), no multiply instruction.
   Every step is a bijection:
     x ^= rotl(x,a)^rotl(x,b)  -> 3 rotation terms = odd weight = unit in GF(2)[t]/(t+1)^64
     x += x << k               -> x * (2^k+1), odd multiplier
     x ^= x >> k               -> unitriangular over GF(2)                                  */
static inline uint64_t crack64(uint64_t x) {
    x ^= rotl64(x, 25) ^ rotl64(x, 47);
    x += x << 13;
    x ^= x >> 31;
    x ^= rotl64(x, 13) ^ rotl64(x, 41);
    x += x << 7;
    x ^= x >> 23;
    x ^= rotl64(x, 29) ^ rotl64(x, 53);
    x += x << 19;
    x ^= x >> 33;
    return x;
}

/* one tumble: it swells where it struck (add = the sob), the old face shrinks and goes,
   then the faces trade cracks so the WHOLE sphere reshapes, not a sliver.
   Invertible as a map on (a,b,c,d): b=b'-R(c'), a=a'-R(b), d=d'-R(a'), c=c'-R(d).        */
#define TUMBLE()                                                     \
    do {                                                             \
        a += a <<  7;  b += b << 11;  c += c << 13;  d += d << 17;   \
        a ^= a >> 29;  b ^= b >> 31;  c ^= c >> 23;  d ^= d >> 19;   \
        a += rotl64(b, 29);  c += rotl64(d, 17);                     \
        b += rotl64(c, 41);  d += rotl64(a, 11);                     \
    } while (0)

/* one tread of the mason trail: press 32 marks into the face, then tumble once */
#define TREAD(P)                                                     \
    do {                                                             \
        uint64_t m0, m1, m2, m3;                                     \
        memcpy(&m0, (P) +  0, 8);  memcpy(&m1, (P) +  8, 8);         \
        memcpy(&m2, (P) + 16, 8);  memcpy(&m3, (P) + 24, 8);         \
        a ^= m0; b ^= m1; c ^= m2; d ^= m3;                          \
        TUMBLE();                                                    \
    } while (0)

/* the final stalk: SipRound-shaped cross-face tumbles (ARX, no multiply) */
#define FINAL_TUMBLE()                                               \
    do {                                                             \
        for (int r_ = 0; r_ < 2; r_++) {                             \
            a += b; b = rotl64(b, 13); b ^= a; a = rotl64(a, 32);    \
            c += d; d = rotl64(d, 16); d ^= c;                       \
            a += d; d = rotl64(d, 21); d ^= a;                       \
            c += b; b = rotl64(b, 17); b ^= c; c = rotl64(c, 32);    \
        }                                                            \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *__restrict p = data;
    size_t n = len;

    /* --- regime 1: the pile is a single mark or a handful; one stalk only.
           (guards the stated small-input risk: no tread, no cross-face rounds) --- */
    if (n <= 8) {
        uint64_t w = 0;
        for (size_t k = 0; k < n; k++) w = (w << 8) | (uint64_t)p[k];
        return crack64(w + (GOLD ^ ((uint64_t)len << 32)));
    }
    if (n <= 16) {
        uint64_t w0, w1;
        memcpy(&w0, p, 8);
        memcpy(&w1, p + n - 8, 8);          /* step back onto the last full tread */
        return crack64(w0 + rotl64(w1, 32) + (GOLD ^ ((uint64_t)len << 32)));
    }

    uint64_t a = SPHERE_A ^ (uint64_t)len;
    uint64_t b = SPHERE_B ^ rotl64((uint64_t)len, 21);
    uint64_t c = SPHERE_C ^ rotl64((uint64_t)len, 43);
    uint64_t d = SPHERE_D ^ ((uint64_t)len << 1);

    /* --- regime 2: the pile does not reach the first landing: one tread, by hand --- */
    if (n <= 32) {
        uint64_t w0, w1, w2, w3;
        memcpy(&w0, p,         8);  memcpy(&w1, p + 8,      8);
        memcpy(&w2, p + n - 16, 8); memcpy(&w3, p + n - 8,  8);
        a ^= w0; b ^= w1; c ^= w2; d ^= w3;
        TUMBLE();
        FINAL_TUMBLE();
        return crack64((a ^ c) + (b ^ d) + (uint64_t)len);
    }

    /* --- regime 4: the pile wraps the full coil: the wide sphere rolls --- */
#if defined(__AVX2__)
    if (n > 160) {
        __m256i S = _mm256_set_epi64x((long long)d, (long long)c,
                                      (long long)b, (long long)a);
        while (n > 160) {                       /* four treads to a landing */
            __m256i m;
            m = _mm256_loadu_si256((const __m256i *)(p +  0));
            S = _mm256_xor_si256(S, m);
            S = _mm256_add_epi64(S, _mm256_slli_epi64(S,  7));
            S = _mm256_shuffle_epi32(S, 0x39);  /* a quarter turn of the face */
            m = _mm256_loadu_si256((const __m256i *)(p + 32));
            S = _mm256_xor_si256(S, m);
            S = _mm256_add_epi64(S, _mm256_slli_epi64(S, 11));
            S = _mm256_shuffle_epi32(S, 0x39);
            m = _mm256_loadu_si256((const __m256i *)(p + 64));
            S = _mm256_xor_si256(S, m);
            S = _mm256_add_epi64(S, _mm256_slli_epi64(S, 13));
            S = _mm256_shuffle_epi32(S, 0x39);
            m = _mm256_loadu_si256((const __m256i *)(p + 96));
            S = _mm256_xor_si256(S, m);
            S = _mm256_add_epi64(S, _mm256_slli_epi64(S, 17));
            S = _mm256_shuffle_epi32(S, 0x39);
            /* the coil turn: old face shrinks and goes, then faces trade cracks */
            S = _mm256_xor_si256(S, _mm256_srli_epi64(S, 31));
            S = _mm256_add_epi64(S, _mm256_permute4x64_epi64(S, 0x39));
            p += 128; n -= 128;
        }
        uint64_t lane[4];
        _mm256_storeu_si256((__m256i *)lane, S);
        a = lane[0]; b = lane[1]; c = lane[2]; d = lane[3];
    }
#endif

    /* --- regime 3: plain treads down the trail --- */
    while (n > 32) { TREAD(p); p += 32; n -= 32; }
    TREAD(data + len - 32);                 /* step back onto the last full tread */

    FINAL_TUMBLE();
    return crack64((a ^ c) + (b ^ d) + (uint64_t)len);
}
