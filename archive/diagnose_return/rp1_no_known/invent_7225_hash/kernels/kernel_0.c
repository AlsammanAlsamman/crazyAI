#include <stdint.h>
#include <stddef.h>

/* "The Cup and the Seven Organs"
 *
 *   the cup is never pure : state mixed in two algebras at once
 *                           (Z/2^64 addition-with-carry, and rotation)
 *                           -- no multiply is spent per mark.
 *   one pull per mark     : c = rotl64(c + byte, R), each byte once, in order.
 *   the tavern keeper     : hefts the pile first -- < 64 marks go to one cup,
 *     hefts the pile        >= 64 are dealt to the body's seven cups,
 *                           one mark to each organ in turn, still front to back.
 *   the seven organs      : z *= C_j   (the bending)
 *                           z ^= z>>s_j (throws away half; keeps only the bits
 *                                        that refuse to sit still)
 *   the garden door       : 3 organs answer a pile of 16 or fewer, 7 otherwise.
 */

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t z;
    int organs;

    if (len < 64u) {
        /* ---- small pile: one cup, one unbroken pour ---- */
        uint64_t c = 0x9e3779b97f4a7c15ULL ^ (uint64_t)len;
        size_t i;
        for (i = 0; i < len; i++)
            c = rotl64(c + (uint64_t)p[i], 11);
        z = c;
        organs = (len <= 16u) ? 3 : 7;
    } else {
        /* ---- large pile: seven cups, one per organ ----
           seven independent 2-cycle chains; the buffer is still walked
           strictly front to back, so every cache line is consumed whole. */
        uint64_t c0 = 0x9e3779b97f4a7c15ULL;
        uint64_t c1 = 0xbf58476d1ce4e5b9ULL;
        uint64_t c2 = 0x94d049bb133111ebULL;
        uint64_t c3 = 0x2545f4914f6cdd1dULL;
        uint64_t c4 = 0xd6e8feb86659fd93ULL;
        uint64_t c5 = 0xa3b195354a39b70dULL;
        uint64_t c6 = 0x1b03738712fad5c9ULL;
        uint64_t x, y;
        size_t i = 0, r;

        for (; i + 7 <= len; i += 7) {
            c0 = rotl64(c0 + (uint64_t)p[i + 0],  7);
            c1 = rotl64(c1 + (uint64_t)p[i + 1], 11);
            c2 = rotl64(c2 + (uint64_t)p[i + 2], 13);
            c3 = rotl64(c3 + (uint64_t)p[i + 3], 17);
            c4 = rotl64(c4 + (uint64_t)p[i + 4], 19);
            c5 = rotl64(c5 + (uint64_t)p[i + 5], 23);
            c6 = rotl64(c6 + (uint64_t)p[i + 6], 29);
        }
        r = len - i;                      /* 0..6 last drops */
        if (r > 0) c0 = rotl64(c0 + (uint64_t)p[i + 0],  7);
        if (r > 1) c1 = rotl64(c1 + (uint64_t)p[i + 1], 11);
        if (r > 2) c2 = rotl64(c2 + (uint64_t)p[i + 2], 13);
        if (r > 3) c3 = rotl64(c3 + (uint64_t)p[i + 3], 17);
        if (r > 4) c4 = rotl64(c4 + (uint64_t)p[i + 4], 19);
        if (r > 5) c5 = rotl64(c5 + (uint64_t)p[i + 5], 23);

        /* the seven colours are married; each cup enters exactly once under
           its own rotation, so the marriage is injective in every cup. */
        x = c0 ^ rotl64(c1,  9) ^ rotl64(c2, 18) ^ rotl64(c3, 27);
        y = c4 ^ rotl64(c5, 36) ^ rotl64(c6, 45) ^ (uint64_t)len;
        z = rotl64(x, 23) + y;
        organs = 7;
    }

    /* ---- the organs: bend, then throw away half ---- */
    z *= 0xff51afd7ed558ccdULL; z ^= z >> 33;   /* 1 */
    z *= 0xc4ceb9fe1a85ec53ULL; z ^= z >> 29;   /* 2 */
    z *= 0xbf58476d1ce4e5b9ULL; z ^= z >> 32;   /* 3 */
    if (organs == 3) return z;                  /* the door, reached early */
    z *= 0x94d049bb133111ebULL; z ^= z >> 31;   /* 4 */
    z *= 0xd6e8feb86659fd93ULL; z ^= z >> 30;   /* 5 */
    z *= 0xa3b195354a39b70dULL; z ^= z >> 27;   /* 6 */
    z *= 0x1b03738712fad5c9ULL; z ^= z >> 33;   /* 7 */
    return z;
}
