#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ===================== THE DESK ======================================
 * mark              -> one input byte, in its given order
 * numbered groove   -> the flat buffer, read in place, never copied
 * bone die-stone    -> FOUR coupled 64-bit seats (256 bits of orientation)
 * quarter turn      -> a 4-cycle permutation of the four seats
 * "drop the weight
 *  into the seated
 *  place"           -> XOR of the mark-weights into the currently seated faces
 * "the stone's memory
 *  of the last press
 *  bends how deep
 *  this one goes"   -> the CARRY CHAIN of a 64-bit add: existing depth
 *                      decides how far the new impression travels
 * "no two folding
 *  paths converge"  -> every step below is a BIJECTION of the 256-bit state,
 *                      so two equal-length piles can never seat alike;
 *                      all collisions are made at the read, nowhere else
 * wooden keeps      -> the one 64-bit reading, taken once, at the end
 * ==================================================================== */

#define SEAT0 0x9E3779B97F4A7C15ULL
#define SEAT1 0xBF58476D1CE4E5B9ULL
#define SEAT2 0x94D049BB133111EBULL
#define SEAT3 0xD6E8FEB86659FD93ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* the keeps: the only reading that ever leaves the desk */
static inline uint64_t keeps1(uint64_t x) {
    x ^= x >> 32;
    x *= 0xD6E8FEB86659FD93ULL;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 32;
    return x;
}

/* read the four seated faces off as one number; the pile's length is
   pressed in last, so piles of different length never read alike */
static inline uint64_t keeps4(uint64_t l0, uint64_t l1, uint64_t l2,
                              uint64_t l3, size_t len) {
    uint64_t a = l0 * 0x87C37B91114253D5ULL;   /* four independent multiplies */
    uint64_t b = l1 * 0x4CF5AD432745937FULL;   /* -> issued in parallel, off   */
    uint64_t c = l2 * 0xFF51AFD7ED558CCDULL;   /*    the folding chain         */
    uint64_t d = l3 * 0xC4CEB9FE1A85EC53ULL;
    uint64_t x = rotl64(a + b, 29) ^ (c + rotl64(d, 41));
    x += (uint64_t)len * SEAT0;
    return keeps1(x);
}

/* ---------- the wide folding stone: piles that fill a revolution ----- */
#if defined(__AVX2__)
/* one fold: drop 32 marks' weights into the four seats, press (carries
   bend the depth), settle, then turn the stone a quarter.  Each of the
   four steps is invertible, so the whole fold is a bijection. */
#define STONE_FOLD(P)                                                        \
    do {                                                                     \
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(P));   \
        st = _mm256_xor_si256(st, v);                    /* drop weights   */ \
        __m256i q = _mm256_shuffle_epi32(st, 0x4E);      /* far seats round*/ \
        __m256i s = _mm256_add_epi64(st, q);             /* THE PRESS      */ \
        st = _mm256_blend_epi32(st, s, 0x33);            /* seats 0,2 take */ \
        st = _mm256_or_si256(_mm256_slli_epi64(st, 27),                       \
                             _mm256_srli_epi64(st, 37)); /* settle deeper  */ \
        st = _mm256_permute4x64_epi64(st, 0x93);         /* QUARTER TURN   */ \
    } while (0)
#else
#define STONE_FOLD(P)                                                        \
    do {                                                                     \
        uint64_t w0, w1, w2, w3, tt;                                          \
        memcpy(&w0, (const void *)(P), 8);                                    \
        memcpy(&w1, (const void *)((const unsigned char *)(P) + 8), 8);        \
        memcpy(&w2, (const void *)((const unsigned char *)(P) + 16), 8);       \
        memcpy(&w3, (const void *)((const unsigned char *)(P) + 24), 8);       \
        l0 ^= w0; l1 ^= w1; l2 ^= w2; l3 ^= w3;                               \
        l0 += l1; l2 += l3;                                                   \
        l0 = rotl64(l0, 27); l1 = rotl64(l1, 27);                             \
        l2 = rotl64(l2, 27); l3 = rotl64(l3, 27);                             \
        tt = l3; l3 = l2; l2 = l1; l1 = l0; l0 = tt;                          \
    } while (0)
#endif

static uint64_t fold_revolutions(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    size_t n = len;                     /* len >= 32 guaranteed by caller */
#if defined(__AVX2__)
    __m256i st = _mm256_set_epi64x((int64_t)SEAT3, (int64_t)SEAT2,
                                   (int64_t)SEAT1, (int64_t)SEAT0);
    while (n >= 32) { STONE_FOLD(p); p += 32; n -= 32; }
    if (n) STONE_FOLD(data + len - 32); /* last revolution re-seats overlap */
    uint64_t t[4];
    _mm256_storeu_si256((__m256i *)(void *)t, st);
    return keeps4(t[0], t[1], t[2], t[3], len);
#else
    uint64_t l0 = SEAT0, l1 = SEAT1, l2 = SEAT2, l3 = SEAT3;
    while (n >= 32) { STONE_FOLD(p); p += 32; n -= 32; }
    if (n) STONE_FOLD(data + len - 32);
    return keeps4(l0, l1, l2, l3, len);
#endif
}

/* ---------- the narrow press: a handful that cannot fill a revolution */
static uint64_t press_handful(const unsigned char *data, size_t len) {
    uint64_t h = SEAT0 ^ ((uint64_t)len * SEAT3);
    if (len >= 8) {
        const unsigned char *p = data;
        size_t n = len;
        while (n >= 8) {
            uint64_t w; memcpy(&w, p, 8);
            h ^= w; h = rotl64(h, 29); h *= SEAT1;
            p += 8; n -= 8;
        }
        if (n) {                               /* safe: len >= 8 */
            uint64_t w; memcpy(&w, data + len - 8, 8);
            h ^= w; h = rotl64(h, 31); h *= SEAT2;
        }
    } else if (len) {                          /* not even one seat full */
        uint64_t w = 0;
        memcpy(&w, data, len);
        h ^= w; h = rotl64(h, 31); h *= SEAT2;
    }
    return keeps1(h);
}

/* the native's one question of the pile: does it fill a revolution? */
uint64_t kernel(const unsigned char *data, size_t len) {
    if (len >= 32) return fold_revolutions(data, len);
    return press_handful(data, len);
}
