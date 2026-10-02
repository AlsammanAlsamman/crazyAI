#include <stdint.h>
#include <stddef.h>

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* ================= the inkwell that never runs dry ================= */
static inline uint64_t ink_drop(uint64_t *s) {          /* splitmix64 */
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ===== eight feathers x 256 ink-weights: 16 KiB, L1-resident =====
   the feather records not the mark's shape but its weight: a byte is
   never multiplied, only used as an index.  (= simple tabulation /
   Zobrist hashing)                                                  */
static uint64_t FEATHER[8][256] __attribute__((aligned(64)));
static int feathers_inked = 0;

static void ink_the_feathers(void) {
    uint64_t s = 0x243F6A8885A308D3ULL;                 /* pi */
    for (int f = 0; f < 8; ++f)
        for (int v = 0; v < 256; ++v)
            FEATHER[f][v] = ink_drop(&s);
    feathers_inked = 1;
}
#if defined(__GNUC__)
__attribute__((constructor)) static void ink_before_main(void) { ink_the_feathers(); }
#endif

/* ===== the storm: ONE pass over the final tremor, no more =====
   Murmur3 fmix64 -- the only two multiplications in the kernel.     */
static inline uint64_t storm_pass(uint64_t h) {
    h ^= h >> 33;  h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;  h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return h;
}

/* one 8-byte block = eight plucks, one per string, same force each.
   feather index (J+l)&7 is a Latin square: inside a 64-byte
   super-block no two marks ever share a feather.                    */
#define BLOCK(J)                                                      \
    do {                                                              \
        const unsigned char *q = p + 8 * (J);                         \
        s0 += FEATHER[((J) + 0) & 7][q[0]];                           \
        s1 += FEATHER[((J) + 1) & 7][q[1]];                           \
        s2 += FEATHER[((J) + 2) & 7][q[2]];                           \
        s3 += FEATHER[((J) + 3) & 7][q[3]];                           \
        s4 += FEATHER[((J) + 4) & 7][q[4]];                           \
        s5 += FEATHER[((J) + 5) & 7][q[5]];                           \
        s6 += FEATHER[((J) + 6) & 7][q[6]];                           \
        s7 += FEATHER[((J) + 7) & 7][q[7]];                           \
    } while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    if (!feathers_inked) ink_the_feathers();            /* belt + braces */

    const unsigned char *p = data;

    /* ---- the bars report the count before the first pluck: a short
       pile does not get eight strings strung across the landscape ---- */
    if (len < 32) {
        uint64_t h = 0x9E3779B97F4A7C15ULL + (uint64_t)len;
        for (size_t i = 0; i < len; ++i)
            h = ROTL64(h, 13) + FEATHER[i & 7][p[i]];   /* one string */
        return storm_pass(h);                           /* one storm   */
    }

    /* ---- eight strings, never let go still ---- */
    uint64_t s0 = 0x243F6A8885A308D3ULL + (uint64_t)len;
    uint64_t s1 = 0x13198A2E03707344ULL + (uint64_t)len;
    uint64_t s2 = 0xA4093822299F31D0ULL + (uint64_t)len;
    uint64_t s3 = 0x082EFA98EC4E6C89ULL + (uint64_t)len;
    uint64_t s4 = 0x452821E638D01377ULL + (uint64_t)len;
    uint64_t s5 = 0xBE5466CF34E90C6CULL + (uint64_t)len;
    uint64_t s6 = 0xC0AC29B7C97C50DDULL + (uint64_t)len;
    uint64_t s7 = 0x3F84D5B5B5470917ULL + (uint64_t)len;

    size_t n = len;

    /* ---- the long landscape: 64 marks a breath, never doubling back ---- */
    while (n >= 64) {
        BLOCK(0); BLOCK(1); BLOCK(2); BLOCK(3);
        BLOCK(4); BLOCK(5); BLOCK(6); BLOCK(7);
        /* the shudder finishes its circuit of the bridge: the top
           strings' tremor comes back around onto the bottom ones.
           all amounts odd -> full 64-step orbit per lane.            */
        s0 = ROTL64(s0,  7); s1 = ROTL64(s1, 13);
        s2 = ROTL64(s2, 19); s3 = ROTL64(s3, 29);
        s4 = ROTL64(s4, 37); s5 = ROTL64(s5, 43);
        s6 = ROTL64(s6, 53); s7 = ROTL64(s7, 59);
        p += 64; n -= 64;
    }

    /* ---- whole blocks left (at most 7) ---- */
    size_t j = 0;
    while (n >= 8) {
        s0 += FEATHER[(j + 0) & 7][p[0]];
        s1 += FEATHER[(j + 1) & 7][p[1]];
        s2 += FEATHER[(j + 2) & 7][p[2]];
        s3 += FEATHER[(j + 3) & 7][p[3]];
        s4 += FEATHER[(j + 4) & 7][p[4]];
        s5 += FEATHER[(j + 5) & 7][p[5]];
        s6 += FEATHER[(j + 6) & 7][p[6]];
        s7 += FEATHER[(j + 7) & 7][p[7]];
        p += 8; n -= 8; ++j;
    }

    /* ---- the last marks (at most 7), each still its own feather ---- */
    if (n) {
        uint64_t t = 0;
        for (size_t k = 0; k < n; ++k)
            t = ROTL64(t, 9) + FEATHER[k & 7][p[k]];
        s7 += t;
    }

    /* ---- the storm reads the whole shivering pattern and dissolves
       it into eight small notes, no more ---- */
    uint64_t a = s0 + ROTL64(s1, 17);
    uint64_t b = s2 + ROTL64(s3, 23);
    uint64_t c = s4 + ROTL64(s5, 31);
    uint64_t d = s6 + ROTL64(s7, 41);
    a ^= ROTL64(b, 13);
    c ^= ROTL64(d, 47);
    uint64_t h = a + ROTL64(c, 29) + (uint64_t)len;
    return storm_pass(h);
}
