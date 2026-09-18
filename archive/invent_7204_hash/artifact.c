#include <stdint.h>

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len) {
    /* the counting-bars: eight violin strings, bound under one bridge */
    uint64_t s[8] = {
        0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL,
        0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL,
        0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL,
        0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL
    };
    const int PLUCK_ROT = 23;                        /* the force: never varies */
    const int BRIDGE_ROT[8] = {7,13,17,23,29,37,41,47};

    /* the inkwell + a fresh feather per mark: a nonlinear byte -> weight map,
       built once so the hot per-mark loop needs no multiplication */
    uint64_t W[256];
    for (int b = 0; b < 256; b++) {
        uint64_t w = (uint64_t)(b + 1) * 0x9E3779B97F4A7C15ULL;
        w ^= w >> 33;
        w *= 0xBF58476D1CE4E5B9ULL;
        w ^= w >> 29;
        W[b] = w;
    }

    for (size_t i = 0; i < len; i++) {
        uint64_t w = W[data[i]];                 /* dip the mark, read its weight */
        int lane  = (int)((w >> 61) & 7u);        /* pluck the string nearest the weight */

        s[lane] ^= w;                             /* the pluck: */
        s[lane]  = rotl64(s[lane], PLUCK_ROT);     /* always the same force */

        /* the bridge: a shudder in one bends the pitch of all the rest */
        uint64_t bridge = 0;
        for (int k = 0; k < 8; k++)
            bridge ^= rotl64(s[k], BRIDGE_ROT[k]);
        for (int k = 0; k < 8; k++)
            s[k] ^= rotl64(bridge, BRIDGE_ROT[7 - k]);
        /* the strings never go fully still between marks */
    }

    /* the storm passes once over the final tremor, dissolving it into eight notes */
    uint64_t token = 0;
    for (int k = 0; k < 8; k++) {
        uint64_t note = s[k] ^ rotl64(s[(k + 1) & 7], (k * 11 + 7) & 63);
        note ^= note >> 32; note ^= note >> 16; note ^= note >> 8;
        token |= (note & 0xFFULL) << (8 * k);
    }
    /* the pickers: W[], s[], bridge, note all fall out of scope here */
    return token;
}
