#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__SSE4_2__)
#  include <immintrin.h>
#  define CHAINED_PRISONERS 1
#else
#  define CHAINED_PRISONERS 0
#endif

/* the fixed anchors in the riverbed: the wires are seeded from these, never from data */
#define A0 0xa0761d6478bd642fULL
#define A1 0xe7037ed1a0b428dbULL
#define A2 0x8ebc6af09c88c6e3ULL
#define A3 0x2d358dccaa6c78a5ULL
#define A4 0x8bb84b93962eacc9ULL
#define A5 0x4b33a62ed433d4a3ULL

static inline uint64_t rd8(const unsigned char *p){ uint64_t v; memcpy(&v,p,sizeof v); return v; }
static inline uint64_t rd4(const unsigned char *p){ uint32_t v; memcpy(&v,p,sizeof v); return (uint64_t)v; }

static inline uint64_t rotl64(uint64_t x, unsigned r){
    r &= 63u;                       /* r == 0 is well defined under this idiom */
    return (x << r) | (x >> ((64u - r) & 63u));
}

/* two wires crossing in the silhouette: 64x64 -> 128, folded (wyhash's wymix) */
static inline uint64_t cross(uint64_t x, uint64_t y){
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)x * (__uint128_t)y;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t xl = (uint32_t)x, xh = x >> 32, yl = (uint32_t)y, yh = y >> 32;
    uint64_t lo = x * y;
    uint64_t hi = xh * yh + ((xl * yh) >> 32) + ((xh * yl) >> 32);
    return lo ^ hi;
#endif
}

/* THE SILHOUETTE: read once, after the flood has drained and the shadows settled.
   `lean` are rotated lanes, `straight` are raw lanes, `crossing` is the term that
   crosses another.  Constant cost, three multiplies, independent of len.        */
static inline uint64_t silhouette(uint64_t lean, uint64_t straight,
                                  uint64_t crossing, uint64_t len)
{
    uint64_t h = cross(lean ^ A3, straight ^ A4);
    h = cross(h ^ crossing ^ A5, len ^ A2);
    h ^= h >> 32; h *= A0; h ^= h >> 29;
    return h;
}

/* REGIME 1: the pile is too small to raise the flood at all.  No descent, no
   chain, no wires -- the marks are pressed straight onto the shadow-stone.      */
static uint64_t pressed_flat(const unsigned char *p, size_t len)
{
    uint64_t A = 0, B = 0, C = 0;
    if (len >= 16)      { A = rd8(p); B = rd8(p + 8);       C = rd8(p + len - 8) ^ rotl64(A, 27); }
    else if (len >= 8)  { A = rd8(p); B = rd8(p + len - 8); C = rotl64(A, 32) ^ B; }
    else if (len >= 4)  { A = rd4(p); B = rd4(p + len - 4); C = (A << 32) ^ B; }
    else if (len > 0)   { A = p[0];   B = p[len >> 1];      C = p[len - 1]; }
    return silhouette(A, B, C, (uint64_t)len);
}

#if !CHAINED_PRISONERS
/* a pit with no chained prisoners: each one sharpens by add-rotate-xor instead
   of by shift register.  Same ring, same wires, same single read.               */
static inline uint64_t sharpen(uint64_t s, uint64_t m){
    s += m;
    s ^= rotl64(s, 23) ^ rotl64(s, 47);
    return s;
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;

    /* count the pile against the depth of the pit before descending */
    if (len < 24) return pressed_flat(p, len);

    /* the prisoners, chained in a ring; each holds an angle, not a mark */
    uint64_t a = (uint32_t)(0x9E3779B9u ^ (uint32_t)len);
    uint64_t b = (uint32_t)(0x85EBCA77u ^ (uint32_t)(len >> 32));
    uint64_t c = (uint32_t)0xC2B2AE3Du;

    /* the wires, anchored in the riverbed -- data never seeds them */
    uint64_t w0 = A0, w1 = A1, w2 = A2;

    const unsigned char *last = p + len - 24;   /* the final full pass */

    for (;;) {
        uint64_t m0 = rd8(p), m1 = rd8(p + 8), m2 = rd8(p + 16);

        /* Each mark is fed singly to the nearest prisoner.  What travels down the
           chain is the angle -- the previous pass's state -- never the mark; and
           the last one's angle carries backward to the first, closing the ring,
           so a mark dropped at the very start still trembles in the last hand.  */
#if CHAINED_PRISONERS
        uint64_t na = _mm_crc32_u64(a, m0 ^ c);
        uint64_t nb = _mm_crc32_u64(b, m1 ^ a);
        uint64_t nc = _mm_crc32_u64(c, m2 ^ b);
#else
        uint64_t na = sharpen(a, m0 ^ c);
        uint64_t nb = sharpen(b, m1 ^ a);
        uint64_t nc = sharpen(c, m2 ^ b);
#endif
        a = na; b = nb; c = nc;

        /* The flood rises and bends each anchored wire by exactly the angle the
           last chained prisoner ground into the pillar on this pass.            */
        uint64_t angle = (nc << 32) ^ (nb << 16) ^ na;
        w0 = rotl64(w0 ^ angle, (unsigned)(nc & 63u));   /* bent BY the angle */
        w1 = rotl64(w1 + angle, 23);
        w2 ^= rotl64(angle, 47);

        if (p == last) break;                 /* the water recedes only here */
        p += 24;
        if (p > last) p = last;               /* the last marks overlap into one full pass */
    }

    /* climb to the rim and read the silhouette, once */
    return silhouette(w0 ^ rotl64(w1, 19),
                      w2 ^ rotl64(w0, 41) ^ (a << 32) ^ b,
                      (c << 32) ^ (a ^ b ^ c) ^ rotl64(w1 ^ w2, 7),
                      (uint64_t)len);
}
