#include <stdint.h>
#include <stddef.h>

/* the bend: the brick-incense on its short chain, swung w places.  1 <= w <= 63 */
static inline uint64_t swing(uint64_t x, unsigned w) {
    return (x << w) | (x >> (64u - w));
}

/* the small cup: this mark's own strokes, counted as pebbles.  yields 1..9 */
#define PEBBLES(b) ((unsigned)__builtin_popcount((unsigned)(unsigned char)(b)) + 1u)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;

    /* the fire, thrown onto the wall's six fixed cracks: six places, six slots */
    uint64_t s0 = 0x9E3779B97F4A7C15ULL;
    uint64_t s1 = 0xBF58476D1CE4E5B9ULL;
    uint64_t s2 = 0x94D049BB133111EBULL;
    uint64_t s3 = 0xFF51AFD7ED558CCDULL;
    uint64_t s4 = 0xC4CEB9FE1A85EC53ULL;
    uint64_t s5 = 0x165667B19E3779F9ULL;

    size_t n = len;

    /* WIDE PILE: march the marks six abreast, one knot per sailor per rank.
       every slot is bent by its own mark's pebble-count, then takes its
       neighbour's shadow from the rank before - tension travels down the
       chain at finite speed, so after six ranks the whole line hangs together,
       while the six bends themselves stay independent and pipeline. */
    while (n >= 6) {
        unsigned w0 = PEBBLES(p[0]), w1 = PEBBLES(p[1]), w2 = PEBBLES(p[2]);
        unsigned w3 = PEBBLES(p[3]), w4 = PEBBLES(p[4]), w5 = PEBBLES(p[5]);
        uint64_t t0 = swing(s0 ^ p[0], w0) + s5;
        uint64_t t1 = swing(s1 ^ p[1], w1) + s0;
        uint64_t t2 = swing(s2 ^ p[2], w2) + s1;
        uint64_t t3 = swing(s3 ^ p[3], w3) + s2;
        uint64_t t4 = swing(s4 ^ p[4], w4) + s3;
        uint64_t t5 = swing(s5 ^ p[5], w5) + s4;
        s0 = t0; s1 = t1; s2 = t2; s3 = t3; s4 = t4; s5 = t5;
        p += 6; n -= 6;
    }

    /* A HANDFUL: fewer than six left, so walk them single file past the fire -
       here each knot feels its neighbour at once.  No setup, no prologue. */
    if (n > 0) s0 = swing(s0 ^ p[0], PEBBLES(p[0])) + s5;
    if (n > 1) s1 = swing(s1 ^ p[1], PEBBLES(p[1])) + s0;
    if (n > 2) s2 = swing(s2 ^ p[2], PEBBLES(p[2])) + s1;
    if (n > 3) s3 = swing(s3 ^ p[3], PEBBLES(p[3])) + s2;
    if (n > 4) s4 = swing(s4 ^ p[4], PEBBLES(p[4])) + s3;

    /* THE CLAY PRINT: the shadow has gone still; press it to the wall and read
       it across the six fixed cracks, each crack at its own fixed height.
       The chain's own length is a property of the chain, so it prints too. */
    uint64_t h = (swing(s0,  7) + swing(s1, 19))
               ^ (swing(s2, 31) + swing(s3, 43))
               ^ (swing(s4, 53) + swing(s5, 61))
               ^ (uint64_t)len;

    /* THE STREAM: let the water run over the stub until only ridges remain.
       One wash, not many - and it is the only place a wheel is turned. */
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 32;
    return h;
}
