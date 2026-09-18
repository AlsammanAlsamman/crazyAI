#include <stdint.h>
#include <stddef.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* Seven jars: each tuned with its own "hunger" (multiplier) and
       "rhythm" (rotate amount) -- seven independent accumulator lanes. */
    static const uint64_t MUL[7] = {
        0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
        0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL, 0x2545F4914F6CDD1DULL,
        0xD6E8FEB86659FD93ULL
    };
    static const int ROT[7] = { 13, 29, 41, 17, 53, 7, 37 };

    uint64_t h[7];
    const uint64_t OFFSET = 1469598103934665603ULL;
    for (int j = 0; j < 7; j++) {
        h[j] = OFFSET ^ ((uint64_t)(j + 1) * 0x9E3779B97F4A7C15ULL);
    }

    uint64_t W = 0;   /* the grey wax: overlapping impressions, not yet cooled */
    int lane = 0;     /* the forward-only wheel's current notch */

    for (size_t i = 0; i < len; i++) {
        /* the mark presses into the wax before the last shape cools */
        W = rotl64(W, 1) ^ (uint64_t)data[i];

        /* dip the wire into the wax; let this jar's swarm bite it,
           in its own count (multiplier) and rhythm (rotate amount) */
        h[lane] ^= W;
        h[lane]  = rotl64(h[lane], ROT[lane]);
        h[lane] *= MUL[lane];

        /* the wheel turns forward, never back */
        lane++;
        if (lane == 7) lane = 0;
    }

    /* the brine basin: freeze the seven jars' bites into six frost-flowers */
    uint64_t acc = h[0];
    for (int j = 1; j < 7; j++) {
        acc ^= h[j];
        acc  = rotl64(acc, ROT[j]);
        acc *= MUL[j];
        acc ^= acc >> 31;
    }

    /* wait for the frost-lace to stop spreading before lifting the token */
    acc ^= (uint64_t)len;
    acc ^= acc >> 33;
    acc *= 0xFF51AFD7ED558CCDULL;
    acc ^= acc >> 33;
    acc *= 0xC4CEB9FE1A85EC53ULL;
    acc ^= acc >> 33;

    /* only the tin sketch survives; the wax (h[], W) is scratch, discarded */
    return acc;
}
