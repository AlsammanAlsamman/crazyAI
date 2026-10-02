#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------- the seven hungers: distinct odd 64-bit constants, high bit-entropy ---------- */
#define H0 0x9E3779B185EBCA87ULL
#define H1 0xC2B2AE3D27D4EB4FULL
#define H2 0x165667B19E3779F9ULL
#define H3 0x85EBCA77C2B2AE63ULL
#define H4 0x27D4EB2F165667C5ULL
#define H5 0xFF51AFD7ED558CCDULL
#define H6 0xC4CEB9FE1A85EC53ULL

/* one dip of the wire = seven jars x one 8-byte mark-group */
#define STRIPE 56

/* the slate wheel: forward only, never back */
static inline uint64_t rotl64(uint64_t x, unsigned r) { return (x << r) | (x >> (64 - r)); }

/* the wax: eight marks pressed at once, never resolved one at a time */
static inline uint64_t ld64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* the brine basin: six frost-flowers, no two alike.
   Ends on a spread (xor-shift), never on a press (a trailing multiply would
   leave output bit 0 a deterministic function of input bit 0).            */
static inline uint64_t frost(uint64_t x) {
    x *= 0xFF51AFD7ED558CCDULL; x ^= x >> 33;   /* flowers 1, 2 */
    x *= 0xC4CEB9FE1A85EC53ULL; x ^= x >> 29;   /* flowers 3, 4 */
    x *= 0x9E3779B97F4A7C15ULL; x ^= x >> 32;   /* flowers 5, 6 */
    return x;
}

/* SMALL REGIME: the pile cannot cover the trough floor, so the jars stay shut
   and the native presses the marks by hand straight into the brine.
   One accumulator, serial - the simple path, deliberately.                  */
static uint64_t handpress(const unsigned char *p, size_t len) {
    uint64_t h = H4 + (uint64_t)len * H0;
    size_t i = 0;
    while (len - i >= 8) { h = rotl64(h + ld64(p + i), 31) * H1; i += 8; }
    if    (len - i >= 4) { h = rotl64(h + ld32(p + i), 29) * H2; i += 4; }
    while (i < len)      { h = rotl64(h + p[i],        23) * H3; i += 1; }
    return frost(h);
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the native eyes the pile: which regime is this? */
    if (len < STRIPE) return handpress(data, len);

    const unsigned char *restrict p = data;
    size_t n = len;

    /* seven jars, each opened on its own hunger */
    uint64_t a0 = H0, a1 = H1, a2 = H2, a3 = H3, a4 = H4, a5 = H5, a6 = H6;

    /* LARGE REGIME: dip the wire once per stripe; all seven swarms bite at once.
       Each lane update  a = rotl(a + w, R) * H  is a bijection in a, so no input
       difference can ever be annihilated in the loop - all the nonlinearity the
       output needs is paid once, later, in the brine.                          */
    while (n >= STRIPE) {
        uint64_t w0 = ld64(p +  0), w1 = ld64(p +  8), w2 = ld64(p + 16);
        uint64_t w3 = ld64(p + 24), w4 = ld64(p + 32), w5 = ld64(p + 40);
        uint64_t w6 = ld64(p + 48);
        a0 = rotl64(a0 + w0, 23) * H0;
        a1 = rotl64(a1 + w1, 29) * H1;
        a2 = rotl64(a2 + w2, 31) * H2;
        a3 = rotl64(a3 + w3, 37) * H3;
        a4 = rotl64(a4 + w4, 41) * H4;
        a5 = rotl64(a5 + w5, 43) * H5;
        a6 = rotl64(a6 + w6, 47) * H6;
        p += STRIPE; n -= STRIPE;
    }

    /* the pile ended mid-trough: one last dip over the FINAL 56 bytes.
       Safe (len >= 56) and branch-free in the tail length; the overlap is
       harmless because len itself is folded in below. Counts and hungers are
       permuted so the last dip is never a repeat of the one before it.      */
    if (n) {
        const unsigned char *q = data + len - STRIPE;
        a0 = rotl64(a0 + ld64(q +  0), 47) * H1;
        a1 = rotl64(a1 + ld64(q +  8), 43) * H2;
        a2 = rotl64(a2 + ld64(q + 16), 41) * H3;
        a3 = rotl64(a3 + ld64(q + 24), 37) * H4;
        a4 = rotl64(a4 + ld64(q + 32), 31) * H5;
        a5 = rotl64(a5 + ld64(q + 40), 29) * H6;
        a6 = rotl64(a6 + ld64(q + 48), 23) * H0;
    }

    /* read the wheel's resting notch, then plunge once.
       Each jar is multiplied by its own hunger before summing, so cancellation
       between lanes is not an accident waiting to happen.                   */
    uint64_t h = (uint64_t)len * H0;
    h += rotl64(a0, 11) * H1;
    h += rotl64(a1, 17) * H2;
    h += rotl64(a2, 23) * H3;
    h += rotl64(a3, 29) * H4;
    h += rotl64(a4, 37) * H5;
    h += rotl64(a5, 43) * H6;
    h += rotl64(a6, 53) * H0;

    /* only the tin sketch survives */
    return frost(h);
}
