#include <stdint.h>
#include <stddef.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the rope: 8 sailors standing between fire and wall */
    uint64_t chain[8];
    static const uint64_t INIT[8] = {
        0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
        0x94D049BB133111EBULL, 0x2545F4914F6CDD1DULL,
        0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
        0x27D4EB2F165667C5ULL, 0x85EBCA6B85EBCA6BULL
    };
    static const int ROT[8] = {7,13,19,23,29,31,37,41};

    for (int j = 0; j < 8; j++)
        chain[j] = INIT[j] + (uint64_t)len * 0x9E3779B1ULL;

    /* comb is always the shadow of the WHOLE chain, kept exact and
       updated incrementally: never just the newest knot's shadow */
    uint64_t comb = 0;
    for (int j = 0; j < 8; j++) comb ^= rotl64(chain[j], ROT[j]);

    for (size_t i = 0; i < len; i++) {
        int idx = (int)(i & 7);
        uint64_t old = chain[idx];

        /* tie one more knot onto this sailor's spot in the rope;
           every other sailor's knot is left exactly as it was */
        uint64_t nw = old + (((uint64_t)data[i] + 0x9E3779B97F4A7C15ULL)
                              ^ rotl64(old, 11));
        chain[idx] = nw;

        /* the fire recasts the shadow of the whole row: remove the old
           lane's contribution, add the new one, so comb is always the
           full recombination of all 8 knots, not just this one */
        comb ^= rotl64(old, ROT[idx]) ^ rotl64(nw, ROT[idx]);
    }

    /* wait for the shadow to settle, press the clay, harden it in the
       stream -- one finalizing avalanche pass, nothing else kept */
    uint64_t h = comb ^ (uint64_t)len;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return h;
}
