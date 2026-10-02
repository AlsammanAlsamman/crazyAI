#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the trail is coiled, so every turn is circular ---------------- */
#define TURN(x, r) (((uint64_t)(x) << (r)) | ((uint64_t)(x) >> (64 - (r))))

/* the mason's stalk angles, set before the first mark is laid: compile-time
   constants, so the trail costs no memory traffic.  Chosen so the composed
   diffusion polynomial (1+z^13+z^37)(1+z^19+z^47) has 9 distinct exponents
   mod 64 - nothing cancels.                                            */
#define A1 13
#define B1 37
#define A2 19
#define B2 47

/* the sphere's eight faces, as the mason left them */
#define C0 0x736f6d6570736575ULL
#define C1 0x646f72616e646f6dULL
#define C2 0x6c7967656e657261ULL
#define C3 0x7465646279746573ULL
#define C4 0x9e3779b97f4a7c15ULL
#define C5 0xbf58476d1ce4e5b9ULL
#define C6 0x94d049bb133111ebULL
#define C7 0xd6e8feb86659fd93ULL

static inline uint64_t ld64(const unsigned char *q) {
    uint64_t v; memcpy(&v, q, 8); return v;
}

/* one strike against a stalk: the sphere cracks a hairline into itself along
   the stalk's angle.  XOR of an odd number of rotations is always a bijection
   on 64 bits (1+z^a+z^b is a unit in GF(2)[z]/(z+1)^64), so nothing is lost
   anywhere down the trail - no multiplier needed to stay injective.      */
static inline uint64_t stalk(uint64_t x, unsigned a, unsigned b) {
    return x ^ TURN(x, a) ^ TURN(x, b);
}

/* the sphere takes the mark's weight and carries it forward (this add is the
   only nonlinearity on the whole trail - carry is what a rotation can never
   fake), then tumbles a fixed count of two turns.  No more, no fewer.    */
static inline uint64_t tumble(uint64_t face, uint64_t mark) {
    uint64_t x = face + mark;
    x = stalk(x, A1, B1);
    x = stalk(x, A2, B2);
    return x;
}

/* the final stalk, struck harder because it is struck only once: SipRound,
   verbatim, the validated ARX round this metaphor arrives at.            */
#define SIPROUND(v0,v1,v2,v3) do {                                  \
    v0 += v1; v1 = TURN(v1,13); v1 ^= v0; v0 = TURN(v0,32);         \
    v2 += v3; v3 = TURN(v3,16); v3 ^= v2;                           \
    v0 += v3; v3 = TURN(v3,21); v3 ^= v0;                           \
    v2 += v1; v1 = TURN(v1,17); v1 ^= v2; v2 = TURN(v2,32);         \
} while (0)

#if defined(__AVX2__)
#define VTURN(x, r) _mm256_or_si256(_mm256_slli_epi64((x),(r)),           \
                                    _mm256_srli_epi64((x),64-(r)))
#define VTUMBLE(x, m) do {                                                 \
    __m256i t_ = _mm256_add_epi64((x), (m));                               \
    t_   = _mm256_xor_si256(_mm256_xor_si256(t_, VTURN(t_,A1)),            \
                            VTURN(t_,B1));                                 \
    (x)  = _mm256_xor_si256(_mm256_xor_si256(t_, VTURN(t_,A2)),            \
                            VTURN(t_,B2));                                 \
} while (0)
#endif

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    uint64_t f0 = C0, f1 = C1, f2 = C2, f3 = C3;
    size_t i = 0;

#if defined(__AVX2__)
    /* THE WIDE TRAIL.  Eight faces, 64 marks a roll.  Splitting the sphere
       and gathering it again is a real cost, so it is only set up when the
       pile is big enough to repay it; otherwise the narrow trail below runs
       as the fallback.  Runtime regime check, the native counting the pile
       at the trail's mouth.                                             */
    if (len >= 256) {
        __m256i s0 = _mm256_set_epi64x((long long)C3, (long long)C2,
                                       (long long)C1, (long long)C0);
        __m256i s1 = _mm256_set_epi64x((long long)C7, (long long)C6,
                                       (long long)C5, (long long)C4);
        for (; i + 64 <= len; i += 64) {
            __m256i m0 = _mm256_loadu_si256((const __m256i *)(p + i));
            __m256i m1 = _mm256_loadu_si256((const __m256i *)(p + i + 32));
            VTUMBLE(s0, m0);
            VTUMBLE(s1, m1);
        }
        uint64_t g[8];
        _mm256_storeu_si256((__m256i *)g,       s0);
        _mm256_storeu_si256((__m256i *)(g + 4), s1);
        f0 = g[0] ^ TURN(g[4], 13);     /* gather the eight faces back */
        f1 = g[1] ^ TURN(g[5], 29);     /* onto four, offset so no two */
        f2 = g[2] ^ TURN(g[6], 47);     /* faces can cancel each other */
        f3 = g[3] ^ TURN(g[7],  7);
    }
#endif

    /* THE NARROW TRAIL.  Four faces, 32 marks a roll.  The faces never
       consult each other here - the old face shrinks and goes - which is
       exactly what lets four tumbles overlap in flight.                 */
    for (; i + 32 <= len; i += 32) {
        f0 = tumble(f0, ld64(p + i));
        f1 = tumble(f1, ld64(p + i +  8));
        f2 = tumble(f2, ld64(p + i + 16));
        f3 = tumble(f3, ld64(p + i + 24));
    }

    /* the sphere is turned onto a new face only while marks remain: a pile
       of fewer than 32 uses as many faces as it has words, and no more.  */
    if (i + 8 <= len) { f0 = tumble(f0, ld64(p + i)); i += 8; }
    if (i + 8 <= len) { f1 = tumble(f1, ld64(p + i)); i += 8; }
    if (i + 8 <= len) { f2 = tumble(f2, ld64(p + i)); i += 8; }

    /* the eggshells the trail sheds: 0..7 trailing marks, and the size of
       the pile itself, pressed into the top of the same last face.       */
    {
        uint64_t tail = 0;
        size_t r = len - i, k;
        for (k = 0; k < r; k++) tail |= (uint64_t)p[i + k] << (8 * k);
        tail |= (uint64_t)(len & 0xff) << 56;
        f3 = tumble(f3, tail);
    }

    /* THE FINAL STALK.  The sphere itself is not kept; only the last,
       smallest crack it leaves here is handed over.                      */
    f0 ^= (uint64_t)len;
    f2 ^= 0xffULL;
    SIPROUND(f0, f1, f2, f3);
    SIPROUND(f0, f1, f2, f3);
    SIPROUND(f0, f1, f2, f3);
    SIPROUND(f0, f1, f2, f3);
    return f0 ^ f1 ^ f2 ^ f3;
}
