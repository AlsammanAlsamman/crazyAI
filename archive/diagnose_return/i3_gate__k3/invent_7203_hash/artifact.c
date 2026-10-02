#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__SSE4_2__) && (defined(__x86_64__) || defined(_M_X64))
#  include <immintrin.h>
#  define CHAIN_HW 1
#else
#  define CHAIN_HW 0
#endif

/* =====================================================================
   THE PIT.  Three objects, nothing else.

   grind()   one chained prisoner's stroke.  He never looks at the mark;
             it only changes the angle of what he was already sharpening,
             and he passes the ANGLE, not the mark, onward.  A 32-stage
             shift register whose top stage carries backward into the low
             taps on every stroke, so a mark dropped at the very start
             still trembles in the hand of the last one sharpening.
             NO MULTIPLICATION.  NO BRANCH.  NO TABLE.

   ANCHOR[]  where the eight wire shapes are fused into the riverbed,
             past where any traveler has ever fused its source.

   silhouette()  the single shadow the wires throw against the sun once
             the flood has drained: which lean, which stand straight,
             WHICH CROSS ANOTHER.  Read once.  Kept alone.
   ===================================================================== */

static const uint32_t ANCHOR[8] = {
    0x243F6A88u, 0x85A308D3u, 0xB7E15162u, 0x8AED2A6Au,
    0x9E3779B9u, 0x7F4A7C15u, 0x6A09E667u, 0xBB67AE85u
};

#if CHAIN_HW
/* the chain, in hardware: one stroke, one cycle of throughput */
#  define GRIND64(a, m)  ((uint32_t)_mm_crc32_u64((uint64_t)(uint32_t)(a), (uint64_t)(m)))
#  define GRIND8(a, b)   ((uint32_t)_mm_crc32_u8 ((uint32_t)(a), (unsigned char)(b)))
#else
/* the same chain, by hand: shifts and xors only, still no multiply */
static inline uint32_t sr_grind(uint32_t a, uint64_t m) {
    uint64_t s = ((uint64_t)a ^ m) ^ 0x9E3779B97F4A7C15ULL;
    s ^= s << 13;          /* the angle passes forward down the line  */
    s ^= s >> 7;           /* and carries backward through it         */
    s ^= s << 17;
    return (uint32_t)s ^ (uint32_t)(s >> 32);
}
#  define GRIND64(a, m)  sr_grind((uint32_t)(a), (uint64_t)(m))
#  define GRIND8(a, b)   sr_grind((uint32_t)(a), (uint64_t)(unsigned char)(b) | 0x100ULL)
#endif

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

/* two wires crossing: the only place in the kernel a multiply is allowed */
static inline uint64_t cross(uint64_t u, uint64_t v) {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)u * (__uint128_t)v;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
#else
    uint64_t ul = (uint32_t)u, uh = u >> 32, vl = (uint32_t)v, vh = v >> 32;
    uint64_t ll = ul * vl, lh = ul * vh, hl = uh * vl, hh = uh * vh;
    uint64_t mid = lh + hl + (ll >> 32);
    uint64_t lo  = (ll & 0xFFFFFFFFu) | (mid << 32);
    uint64_t hi  = hh + (mid >> 32);
    return lo ^ hi;
#endif
}

/* the token: one projection of the eight wires, drawn once, kept alone */
static uint64_t silhouette(const uint32_t *w, uint64_t len) {
    uint64_t a = (((uint64_t)w[0] << 32) ^ (uint64_t)w[1]) ^ 0x452821E638D01377ULL;
    uint64_t b = (((uint64_t)w[2] << 32) ^ (uint64_t)w[3]) ^ 0xBE5466CF34E90C6CULL;
    uint64_t c = (((uint64_t)w[4] << 32) ^ (uint64_t)w[5]) ^ 0xC0AC29B7C97C50DDULL;
    uint64_t d = (((uint64_t)w[6] << 32) ^ (uint64_t)w[7]) ^ 0x3F84D5B5B5470917ULL;

    uint64_t lean  = cross(a, b);                 /* which lean          */
    uint64_t stand = cross(c, d);                 /* which stand straight*/
    uint64_t z     = cross(lean ^ len, stand ^ 0x9E3779B97F4A7C15ULL);

    z ^= z >> 33; z *= 0xFF51AFD7ED558CCDULL;     /* the shadow settles  */
    z ^= z >> 29; z *= 0xC4CEB9FE1A85EC53ULL;
    z ^= z >> 32;
    return z;
}

/* one pass of the flood: eight wires bent, each by the angle its own
   chain ground on this pass.  Nothing is read back out. */
#define PASS(q)                                                        \
    do {                                                               \
        w0 = GRIND64(w0, ld64((q) +  0)); w1 = GRIND64(w1, ld64((q) +  8)); \
        w2 = GRIND64(w2, ld64((q) + 16)); w3 = GRIND64(w3, ld64((q) + 24)); \
        w4 = GRIND64(w4, ld64((q) + 32)); w5 = GRIND64(w5, ld64((q) + 40)); \
        w6 = GRIND64(w6, ld64((q) + 48)); w7 = GRIND64(w7, ld64((q) + 56)); \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    size_t n = len;

    uint32_t w0 = ANCHOR[0], w1 = ANCHOR[1], w2 = ANCHOR[2], w3 = ANCHOR[3];
    uint32_t w4 = ANCHOR[4], w5 = ANCHOR[5], w6 = ANCHOR[6], w7 = ANCHOR[7];

    /* ---- REGIME ONE: the deep flood.  The water reaches every anchor,
       so it rises in wide passes and bends all eight wires at once.     */
    if (n >= 256) {
        do {
            __builtin_prefetch(p + 512, 0, 0);
            PASS(p);  PASS(p + 64);  PASS(p + 128);  PASS(p + 192);
            p += 256; n -= 256;
        } while (n >= 256);
    }
    while (n >= 64) { PASS(p); p += 64; n -= 64; }

    /* ---- REGIME TWO: the shallow pile.  The water never floods the
       riverbed, so the marks are led down the line one by one to the
       nearest prisoner.  This is also the tail of regime one.           */
    {
        uint32_t w[8];
        unsigned k = 0;
        w[0]=w0; w[1]=w1; w[2]=w2; w[3]=w3; w[4]=w4; w[5]=w5; w[6]=w6; w[7]=w7;

        while (n >= 8) { w[k]      = GRIND64(w[k],      ld64(p)); p += 8; n -= 8; k++; }
        while (n)      { w[k & 7u] = GRIND8 (w[k & 7u], *p++);           n--;    k++; }

        /* the water goes down; the angle carries backward through the
           line, so the last one sharpening holds every mark.            */
        {
            uint32_t carry = w[7];
            int i;
            for (i = 0; i < 8; i++) { carry = GRIND64(w[i], carry); w[i] = carry; }
        }

        /* climb to the rim.  Read the shadow once.  Keep nothing else.   */
        return silhouette(w, (uint64_t)len);
    }
}
