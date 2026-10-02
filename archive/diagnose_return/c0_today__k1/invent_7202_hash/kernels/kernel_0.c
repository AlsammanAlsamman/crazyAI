#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------- *
 *  SAILOR-CHAIN HASH
 *  six knots on a closed chain, hung between the fire and the wall;
 *  each mark's pebble-count swings the incense, bending where its
 *  shadow lands; one wash in the stream at the end, nothing kept.
 *  No multiplication anywhere inside the walk.
 * ---------------------------------------------------------------- */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* the stream: wash the clay stub until only the hardened ridges remain.
   SplitMix64 / Murmur3-style fmix64 - a published, validated avalanche
   stage. ONE wash, never more: the old print is thrown away, not stacked. */
static inline uint64_t ridges(uint64_t z) {
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27; z *= 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z;
}

/* SHADOW[b] = (b | 0x100) << (6 * popcount(b))
   the cup's fill (popcount) swings the incense six paces per pebble;
   the 0x100 marker is the knot's own body. Because every entry lies in
   [256,511] before the swing - exactly one octave - v<<s == v'<<s'
   forces v==v' and s==s', so distinct marks NEVER land identically.
   Max shift 48, max bit touched 56: nothing is ever shifted off the wall. */
static uint64_t SHADOW[256];
static int      SHADOW_LIT = 0;

static void light_the_fire(void) {
    for (unsigned b = 0; b < 256u; ++b)
        SHADOW[b] = (uint64_t)(b | 0x100u)
                  << (6u * (unsigned)__builtin_popcount(b));
    SHADOW_LIT = 1;            /* idempotent: a race rewrites equal values */
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    if (!SHADOW_LIT) light_the_fire();

    const unsigned char * __restrict p = data;
    const uint64_t      * __restrict T = SHADOW;

    /* ---- weigh the pile in the hand: which regime is this? ----
       a pile too short to carry the chain once around all six cracks
       never gets the chain; it walks on a single knot.            */
    if (len < 48) {
        uint64_t h = 0x9E3779B97F4A7C15ULL
                   ^ ((uint64_t)len << 32) ^ (uint64_t)len;
        for (size_t i = 0; i < len; ++i) {
            uint64_t s = T[p[i]];
            h = rotl64(h ^ s, 23) + s;     /* xor, rotate, add - no multiply */
        }
        return ridges(h);
    }

    /* ---- the six fixed cracks: the only six places the token holds ---- */
    uint64_t L0 = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;
    uint64_t L1 = 0xC2B2AE3D27D4EB4FULL;
    uint64_t L2 = 0x165667B19E3779F9ULL;
    uint64_t L3 = 0x27D4EB2F165667C5ULL;
    uint64_t L4 = 0x85EBCA77C2B2AE63ULL;
    uint64_t L5 = 0xD6E8FEB86659FD93ULL;

    /* Each knot takes its own mark AND its neighbour's hang: the whole line
       hangs together. Lanes 0..4 read the neighbour's PRE-pass value (free,
       no extra move); lane 5 closes the loop onto the already-updated lane 0,
       so information runs round the ring and every lane feels every mark.
       The update matrix is invertible over GF(2): a difference, once
       introduced, can never die out before the fold. */
    size_t i = 0;
    for (; i + 12 <= len; i += 12) {
        L0 = rotl64(L0,  7) ^ L1 ^ T[p[i +  0]];
        L1 = rotl64(L1, 11) ^ L2 ^ T[p[i +  1]];
        L2 = rotl64(L2, 19) ^ L3 ^ T[p[i +  2]];
        L3 = rotl64(L3, 31) ^ L4 ^ T[p[i +  3]];
        L4 = rotl64(L4, 43) ^ L5 ^ T[p[i +  4]];
        L5 = rotl64(L5, 59) ^ L0 ^ T[p[i +  5]];
        L0 = rotl64(L0,  7) ^ L1 ^ T[p[i +  6]];
        L1 = rotl64(L1, 11) ^ L2 ^ T[p[i +  7]];
        L2 = rotl64(L2, 19) ^ L3 ^ T[p[i +  8]];
        L3 = rotl64(L3, 31) ^ L4 ^ T[p[i +  9]];
        L4 = rotl64(L4, 43) ^ L5 ^ T[p[i + 10]];
        L5 = rotl64(L5, 59) ^ L0 ^ T[p[i + 11]];
    }
    for (; i + 6 <= len; i += 6) {
        L0 = rotl64(L0,  7) ^ L1 ^ T[p[i + 0]];
        L1 = rotl64(L1, 11) ^ L2 ^ T[p[i + 1]];
        L2 = rotl64(L2, 19) ^ L3 ^ T[p[i + 2]];
        L3 = rotl64(L3, 31) ^ L4 ^ T[p[i + 3]];
        L4 = rotl64(L4, 43) ^ L5 ^ T[p[i + 4]];
        L5 = rotl64(L5, 59) ^ L0 ^ T[p[i + 5]];
    }

    /* the last few marks, each still on its own knot */
    switch (len - i) {
        case 5: L4 = rotl64(L4, 43) ^ L5 ^ T[p[i + 4]];  /* fall through */
        case 4: L3 = rotl64(L3, 31) ^ L4 ^ T[p[i + 3]];  /* fall through */
        case 3: L2 = rotl64(L2, 19) ^ L3 ^ T[p[i + 2]];  /* fall through */
        case 2: L1 = rotl64(L1, 11) ^ L2 ^ T[p[i + 1]];  /* fall through */
        case 1: L0 = rotl64(L0,  7) ^ L1 ^ T[p[i + 0]];  /* fall through */
        default: break;
    }

    /* ---- the smoke thins to one thread; press the clay to the wall ----
       read the settled shadow against the six fixed cracks. '+' and '^'
       alternate so the fold is not linear over GF(2); each lane enters
       injectively, so a difference in any one lane survives the press. */
    uint64_t h = L0 ^ rotl64(L1, 13);
    h = (h + rotl64(L2, 29)) ^ rotl64(L3, 43);
    h = (h ^ rotl64(L4, 17)) + rotl64(L5, 53);
    h ^= (uint64_t)len;

    /* ---- the stream. nothing else is kept. ---- */
    return ridges(h);
}
