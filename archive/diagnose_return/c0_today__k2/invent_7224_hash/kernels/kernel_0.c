#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ================= riverbank: placement-only reads =================
   The servant's swat: during the load nothing is computed. A mark's
   order is carried by WHERE it lands (lane j, word offset), never by
   WHEN it is folded.                                                */
static inline uint64_t rotl64(uint64_t x, unsigned r) { return (x << r) | (x >> (64u - r)); }
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* ================= the owl: murmur3 fmix64 ========================= */
static inline uint64_t owl(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

/* 16 distinct nothing-up-my-sleeve stones: BLAKE2b IV ++ SHA-512 K[0..7].
   None zero -> the all-zero grid is not a fixed point ("never let it ossify"). */
static const uint64_t SHRINE[16] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL, /* row A */
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL, /* row B */
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, /* row C */
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL  /* row D */
};

/* ===== the shrine stone: fold the 4x4 square along its four diagonals.
   Every one of the 16 words appears exactly once -- no corner survives
   untouched -- then the dreamer inside the dreamer seals one token.   */
static inline uint64_t shrine_fold(const uint64_t *S, size_t len) {
    uint64_t d0 = S[0] ^ S[ 5] ^ S[10] ^ S[15];
    uint64_t d1 = S[1] ^ S[ 6] ^ S[11] ^ S[12];
    uint64_t d2 = S[2] ^ S[ 7] ^ S[ 8] ^ S[13];
    uint64_t d3 = S[3] ^ S[ 4] ^ S[ 9] ^ S[14];
    uint64_t h  = d0 + rotl64(d1, 13);
    h ^= rotl64(d2, 29);
    h += rotl64(d3, 47);
    h ^= (uint64_t)len;
    return owl(owl(h));
}

/* ---- scalar racing path (4 heaps, one SipRound each per 8-byte word) ---- */
#define SROUND() do { int j_; for (j_ = 0; j_ < 4; j_++) {                               \
    A[j_] += B[j_]; B[j_] = rotl64(B[j_],13); B[j_] ^= A[j_]; A[j_] = rotl64(A[j_],32);  \
    C[j_] += D[j_]; D[j_] = rotl64(D[j_],16); D[j_] ^= C[j_];                            \
    A[j_] += D[j_]; D[j_] = rotl64(D[j_],21); D[j_] ^= A[j_];                            \
    C[j_] += B[j_]; B[j_] = rotl64(B[j_],17); B[j_] ^= C[j_]; C[j_] = rotl64(C[j_],32);  \
} } while (0)

/* the 90-degree turn: lanes slide, so each heap's next race is run with a
   neighbour's shape -- cross-lane diffusion for the price of a shuffle. */
#define STURN() do { uint64_t t_;                                 \
    t_=B[0]; B[0]=B[1]; B[1]=B[2]; B[2]=B[3]; B[3]=t_;            \
    t_=C[0]; C[0]=C[2]; C[2]=t_; t_=C[1]; C[1]=C[3]; C[3]=t_;     \
    t_=D[3]; D[3]=D[2]; D[2]=D[1]; D[1]=D[0]; D[0]=t_;            \
} while (0)

#define SSTRIPE(q) do { int j_; uint64_t m_[4];                   \
    for (j_=0;j_<4;j_++) m_[j_] = ld64((q) + 8*j_);               \
    for (j_=0;j_<4;j_++) D[j_] ^= m_[j_];                         \
    SROUND();                                                      \
    for (j_=0;j_<4;j_++) A[j_] ^= m_[j_];                         \
} while (0)

#if defined(__AVX2__)
#define VROT(x,r)  _mm256_or_si256(_mm256_slli_epi64((x),(r)), _mm256_srli_epi64((x), 64-(r)))
#define VROT32(x)  _mm256_shuffle_epi32((x), 0xB1)
#define VROUND() do {                                                                   \
    vA = _mm256_add_epi64(vA,vB); vB = VROT(vB,13); vB = _mm256_xor_si256(vB,vA); vA = VROT32(vA); \
    vC = _mm256_add_epi64(vC,vD); vD = VROT(vD,16); vD = _mm256_xor_si256(vD,vC);        \
    vA = _mm256_add_epi64(vA,vD); vD = VROT(vD,21); vD = _mm256_xor_si256(vD,vA);        \
    vC = _mm256_add_epi64(vC,vB); vB = VROT(vB,17); vB = _mm256_xor_si256(vB,vC); vC = VROT32(vC); \
} while (0)
#define VTURN() do {                                       \
    vB = _mm256_permute4x64_epi64(vB, 0x39);  /* <<1 */     \
    vC = _mm256_permute4x64_epi64(vC, 0x4E);  /* <<2 */     \
    vD = _mm256_permute4x64_epi64(vD, 0x93);  /* <<3 */     \
} while (0)
#define VSTRIPE(q) do {                                                   \
    __m256i m_ = _mm256_loadu_si256((const __m256i *)(const void *)(q));  \
    vD = _mm256_xor_si256(vD, m_);                                        \
    VROUND();                                                             \
    vA = _mm256_xor_si256(vA, m_);                                        \
} while (0)
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * const p0 = data;

    /* ===== regime test, in the native's own terms: are there enough marks
       to make even one heap of even count? If not, no racing happens --
       the marks go from the stone straight to the owl. (Validated XXH3-style
       first/last overlapping ladder; every byte read at least once.)      */
    if (len < 32) {
        uint64_t a, b, x;
        if (len >= 16) {
            a = ld64(p0)            ^ rotl64(ld64(p0 + 8), 23);
            b = ld64(p0 + len - 16) ^ rotl64(ld64(p0 + len - 8), 41);
        } else if (len >= 8) {
            a = ld64(p0);
            b = ld64(p0 + len - 8);
        } else if (len >= 4) {
            a = (uint64_t)ld32(p0);
            b = (uint64_t)ld32(p0 + len - 4);
        } else if (len) {
            a = ((uint64_t)p0[0] << 16) | ((uint64_t)p0[len >> 1] << 8) | (uint64_t)p0[len - 1];
            b = 0;
        } else {
            a = 0; b = 0;
        }
        x = (a + 0x6a09e667f3bcc908ULL)
          ^ rotl64(b + 0xbb67ae8584caa73bULL, 32)
          ^ ((uint64_t)len * 0x9e3779b97f4a7c15ULL);
        x = owl(x);
        /* guard: the nested dreamer's extra multiplies are pure overhead on
           a handful of marks -- only pay for them once there are >= 16.    */
        if (len >= 16) x = owl(x ^ (uint64_t)len);
        return x;
    }

    /* ===== the racing path: 4 heaps abreast, 32-byte deer stride ===== */
    {
        uint64_t S[16];
        size_t rem = len;
        const unsigned char *p = p0;

#if defined(__AVX2__)
        __m256i vA = _mm256_loadu_si256((const __m256i *)(const void *)(SHRINE + 0));
        __m256i vB = _mm256_loadu_si256((const __m256i *)(const void *)(SHRINE + 4));
        __m256i vC = _mm256_loadu_si256((const __m256i *)(const void *)(SHRINE + 8));
        __m256i vD = _mm256_loadu_si256((const __m256i *)(const void *)(SHRINE + 12));
        int r_;

        while (rem >= 64) { VSTRIPE(p); VSTRIPE(p + 32); VTURN(); p += 64; rem -= 64; }
        if (rem >= 32)    { VSTRIPE(p); VTURN(); p += 32; rem -= 32; }
        /* the tail heap is read overlapping the marks already read: legal
           only because order lives in placement, not in reading sequence. */
        if (rem)          { VSTRIPE(p0 + len - 32); VTURN(); }

        vD = _mm256_xor_si256(vD, _mm256_set1_epi64x((long long)(uint64_t)len));
        vC = _mm256_xor_si256(vC, _mm256_set1_epi64x((long long)0xffULL));
        for (r_ = 0; r_ < 3; r_++) { VROUND(); VTURN(); }

        _mm256_storeu_si256((__m256i *)(void *)(S +  0), vA);
        _mm256_storeu_si256((__m256i *)(void *)(S +  4), vB);
        _mm256_storeu_si256((__m256i *)(void *)(S +  8), vC);
        _mm256_storeu_si256((__m256i *)(void *)(S + 12), vD);
#else
        uint64_t A[4], B[4], C[4], D[4];
        int j, r_;
        for (j = 0; j < 4; j++) { A[j] = SHRINE[j]; B[j] = SHRINE[4+j]; C[j] = SHRINE[8+j]; D[j] = SHRINE[12+j]; }

        while (rem >= 64) { SSTRIPE(p); SSTRIPE(p + 32); STURN(); p += 64; rem -= 64; }
        if (rem >= 32)    { SSTRIPE(p); STURN(); p += 32; rem -= 32; }
        if (rem)          { SSTRIPE(p0 + len - 32); STURN(); }

        for (j = 0; j < 4; j++) { D[j] ^= (uint64_t)len; C[j] ^= 0xffULL; }
        for (r_ = 0; r_ < 3; r_++) { SROUND(); STURN(); }

        for (j = 0; j < 4; j++) { S[j] = A[j]; S[4+j] = B[j]; S[8+j] = C[j]; S[12+j] = D[j]; }
#endif
        return shrine_fold(S, len);
    }
}
