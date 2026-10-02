#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ===================== THE WEIGHING HALL =====================
   W        : the grey wax trough - one shared running impression of
              every mark so far ("only the sum of every collision")
   v0..v6   : seven bite-patterns on the dipped wire, one per jar,
              each tuned to its own hunger and its own rhythm
   rotl     : the spinning slate wheel - forward only, fixed notches
   brine()  : one plunge, six frost-flowers, frozen to the fixpoint
   return   : the tin sketch; wax, wire and swarms are all remelted
   ============================================================= */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));            /* never turns back */
}

static inline uint64_t load64(const unsigned char *p) {
    uint64_t w; memcpy(&w, p, sizeof w); return w;  /* one mark, read whole */
}

/* the brine: six frost-flowers, no two alike. Stafford mix13 (the
   validated splitmix64 finalizer) is the fixpoint where the frost-lace
   stops spreading - stop there, not one round further. */
static inline uint64_t brine(uint64_t x, uint64_t len) {
    x ^= len * 0x9E3779B97F4A7C15ULL;              /* flower 1: pile depth */
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;      /* flowers 2, 3 */
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;      /* flowers 4, 5 */
    x ^= x >> 31;                                  /* flower 6 */
    return x;
}

/* one trickle: the thin-pile regime and the end-of-pile remainder.
   A single accumulator, strictly in order - the old way, kept exactly
   where the native keeps it: for piles too thin to feed seven jars. */
static inline uint64_t trickle(uint64_t h, const unsigned char *p, size_t r) {
    while (r >= 8) {
        uint64_t w = load64(p);
        h = rotl64(h ^ w, 31) + w;
        p += 8; r -= 8;
    }
    if (r) {
        uint64_t k = (uint64_t)r * 0x9E3779B97F4A7C15ULL;
        for (size_t i = 0; i < r; i++) k ^= (uint64_t)p[i] << (8u * (unsigned)i);
        h = rotl64(h ^ k, 23) + k;
    }
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;
    size_t r = len;

    /* weigh the pile in the hand: too thin to reach all seven jars? */
    if (r < 56)
        return brine(trickle(0x9E3779B97F4A7C15ULL, p, r), (uint64_t)len);

    uint64_t W  = 0x6A09E667F3BCC908ULL;           /* the wax */
    uint64_t v0 = 0x9E3779B185EBCA87ULL, v1 = 0xC2B2AE3D27D4EB4FULL,
             v2 = 0x165667B19E3779F9ULL, v3 = 0x85EBCA77C2B2AE63ULL,
             v4 = 0x27D4EB2F165667C5ULL, v5 = 0xD6E8FEB86659FD93ULL,
             v6 = 0xA0761D6478BD642FULL;           /* seven hungers */

    do {   /* one dip: seven marks pressed, seven swarms bite the wire */
        W += load64(p +  0); v0 = rotl64(v0 ^ W, 13) + W;
        W += load64(p +  8); v1 = rotl64(v1 ^ W, 29) + W;
        W += load64(p + 16); v2 = rotl64(v2 ^ W, 41) + W;
        W += load64(p + 24); v3 = rotl64(v3 ^ W,  7) + W;
        W += load64(p + 32); v4 = rotl64(v4 ^ W, 53) + W;
        W += load64(p + 40); v5 = rotl64(v5 ^ W, 19) + W;
        W += load64(p + 48); v6 = rotl64(v6 ^ W, 37) + W;
        W = rotl64(W, 23);                         /* the wheel notches on */
        p += 56; r -= 56;
    } while (r >= 56);

    /* lift the wire: all seven bite-patterns and the wax onto one strand */
    uint64_t h = rotl64(v0,  1) + rotl64(v1,  9) + rotl64(v2, 17)
               + rotl64(v3, 25) + rotl64(v4, 33) + rotl64(v5, 41)
               + rotl64(v6, 49);
    h ^= W * 0x9FB21C651E98DF25ULL;                /* the wax's own witness */
    h  = trickle(h, p, r);                         /* the pile's last marks */
    return brine(h, (uint64_t)len);                /* plunge once, sketch to tin */
}
