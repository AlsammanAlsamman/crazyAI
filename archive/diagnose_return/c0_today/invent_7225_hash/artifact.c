#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ================= THE CUP THAT IS NEVER PURE =========================
   One pull absorbs 8 marks. The cup's memory is a polynomial remainder
   over a non-trivial generator: a "pure" polynomial (x^n) would hold no
   memory, a cut one remembers every drop. No multiplication anywhere in
   the pour.                                                             */

#if defined(__SSE4_2__) && defined(__x86_64__)
#  include <immintrin.h>
#  define CUP_TABLE 0
#  define POUR8(c,v) ((uint64_t)_mm_crc32_u64((uint64_t)(c),(uint64_t)(v)))
#elif defined(__ARM_FEATURE_CRC32)
#  include <arm_acle.h>
#  define CUP_TABLE 0
#  define POUR8(c,v) ((uint64_t)__crc32d((uint32_t)(c),(uint64_t)(v)))
#else
#  define CUP_TABLE 1
#endif

#if CUP_TABLE
/* Portable cup: reflected CRC-64/XZ (ECMA-182). Table build is
   deterministic and idempotent, so a concurrent first call writes
   identical bytes and is harmless.                                      */
static uint64_t cup_tbl[256];
static int cup_ready;
static void cup_fill(void)
{
    for (unsigned i = 0; i < 256u; i++) {
        uint64_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xC96C5795D7870F42ULL & (uint64_t)-(int64_t)(c & 1u));
        cup_tbl[i] = c;
    }
    cup_ready = 1;
}
static inline uint64_t pour8(uint64_t c, uint64_t v)
{
    for (int k = 0; k < 8; k++) { c = cup_tbl[(unsigned char)(c ^ v)] ^ (c >> 8); v >>= 8; }
    return c;
}
#  define POUR8(c,v) pour8((uint64_t)(c),(uint64_t)(v))
#endif

#define CUP_A  0x2AD7D2FBULL      /* the cup is never pure: non-zero seed */
#define CUP_B  0x8F1BBCDCULL
#define CUP_C  0xCA62C1D6ULL
#define GOLDEN 0x9E3779B97F4A7C15ULL

static inline uint64_t ror64(uint64_t x, unsigned r)
{
    return (x >> r) | (x << (64u - r));
}

/* ================= THE SEVEN ORGANS ===================================
   The cup's last color is carried through seven organs in turn. Each
   organ bends, or discards half of what it received and keeps only what
   refuses to sit still. Seven is fixed, not grown: every organ is lossy,
   so more bending is not more mixing. The count was settled at the
   garden door against the 0.5-avalanche note.

   Organs 1-5 are exactly nasam (Pelle Evensen); organs 6-7 are exactly
   the tail of rrmxmx. Both are published, avalanche-tested mixers --
   nothing here is invented.                                             */
static inline uint64_t seven_organs(uint64_t x)
{
    x ^= ror64(x, 25) ^ ror64(x, 47);   /* organ 1: discard, keep what moves */
    x *= 0x9E6C63D0676A9A99ULL;         /* organ 2: bend                     */
    x ^= (x >> 23) ^ (x >> 51);         /* organ 3: discard half             */
    x *= 0x9E6D62D06F6A9A9BULL;         /* organ 4: bend                     */
    x ^= (x >> 23) ^ (x >> 51);         /* organ 5: discard half             */
    x *= 0x9FB21C651E98DF25ULL;         /* organ 6: bend                     */
    x ^= (x >> 28) ^ (x >> 47);         /* organ 7: discard half             */
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;
    uint64_t x;

#if CUP_TABLE
    if (!cup_ready) cup_fill();
#endif

    /* He weighs the pile in his hand before the first pour. */
    if (len >= 192u) {
        /* HEAVY PILE: the tavern's row of three cups, poured side by side.
           Three independent pours hide the cup's 3-cycle settling time.   */
        const size_t third = (len / 24u) * 8u;         /* multiple of 8 */
        const unsigned char *q0 = p;
        const unsigned char *q1 = p + third;
        const unsigned char *q2 = p + 2u * third;
        uint64_t a = CUP_A, b = CUP_B, c = CUP_C;
        const size_t n = third >> 3;
        size_t i, rem;
        const unsigned char *r;

        for (i = 0; i < n; i++) {
            uint64_t v0, v1, v2;
            memcpy(&v0, q0 + (i << 3), 8);
            memcpy(&v1, q1 + (i << 3), 8);
            memcpy(&v2, q2 + (i << 3), 8);
            a = POUR8(a, v0);
            b = POUR8(b, v1);
            c = POUR8(c, v2);
        }

        /* what the three cups could not divide evenly goes to the first */
        r   = p + 3u * third;
        rem = len - 3u * third;
        while (rem >= 8u) {
            uint64_t v; memcpy(&v, r, 8);
            a = POUR8(a, v); r += 8; rem -= 8u;
        }
        if (rem) {                        /* last drops, never re-read */
            uint64_t v = 0; memcpy(&v, r, rem);
            a = POUR8(a, v);
        }

        b = POUR8(b, c);                  /* third cup poured into the second */
        x = (a & 0xFFFFFFFFULL) | (b << 32);
        x ^= ror64(c, 17);
    } else {
        /* LIGHT PILE: two cups, drop by drop. Fallback path -- no row of
           cups to set up, no loop a short pile cannot amortise.           */
        uint64_t a = CUP_A, b = CUP_B;
        size_t rem = len;
        while (rem >= 16u) {
            uint64_t v0, v1;
            memcpy(&v0, p, 8);
            memcpy(&v1, p + 8, 8);
            a = POUR8(a, v0);
            b = POUR8(b, v1);
            p += 16; rem -= 16u;
        }
        if (rem >= 8u) {
            uint64_t v; memcpy(&v, p, 8);
            a = POUR8(a, v); p += 8; rem -= 8u;
        }
        if (rem) {
            uint64_t v = 0; memcpy(&v, p, rem);
            b = POUR8(b, v);
        }
        x = (a & 0xFFFFFFFFULL) | (b << 32);
    }

    x ^= (uint64_t)len * GOLDEN;          /* the pile's own weight */
    return seven_organs(x);
}
