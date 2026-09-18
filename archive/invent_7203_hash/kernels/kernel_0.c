#include <stdint.h>

#define NWIRE 64

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    /* r is constructed to always lie in [1,63] -- no shift-by-0/64 UB */
    return (x << r) | (x >> (64 - r));
}

/* the "settling": a strong two-round avalanche mixer, applied once,
   per wire, at readout time -- never repeated, never applied mid-buffer */
static inline uint64_t bend(uint64_t v) {
    v ^= v >> 30;
    v *= 0xBF58476D1CE4E5B9ULL;
    v ^= v >> 27;
    v *= 0x94D049BB133111EBULL;
    v ^= v >> 31;
    return v;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the chained line of prisoners: ONE accumulator, sharpened mark by
       mark, in the given order -- this is the pillar and its angle */
    uint64_t A = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        A ^= (uint64_t)data[i];
        A *= 1099511628211ULL;
    }
    /* A is the angle ground into the pillar on this pass. It is never
       returned directly -- "a token pulled while the water is still up
       is worthless." We only use A after the loop above has fully
       finished, i.e. after every mark has gone through. */

    /* the flood: rises once, after every mark has gone through, and
       bends every one of the 64 anchored wires by exactly that same
       angle A. The wires never move themselves and are never touched
       during the loop above -- each is anchored at its own fixed,
       input-independent position. */
    uint64_t hash = 0;
    const uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;
    for (unsigned i = 0; i < NWIRE; i++) {
        uint64_t anchor = GOLDEN * (uint64_t)(i + 1);   /* wire i's own
                                                            fixed anchor */
        unsigned r = (i % 63) + 1;                       /* 1..63 */
        uint64_t bent = rotl64(anchor ^ A, r);           /* the SAME
                                                            angle A bends
                                                            every wire */
        bent = bend(bent);                               /* let the
                                                            shivering
                                                            settle */
        /* the silhouette: read once whether this wire leans one way */
        hash |= (bent >> 63) << i;
    }
    return hash;                    /* the shadow-shape, kept alone */
}
