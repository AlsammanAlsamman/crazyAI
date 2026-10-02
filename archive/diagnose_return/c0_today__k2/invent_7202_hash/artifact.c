#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ================================================================= *
 *  The six fixed cracks in the wall — the same six every time.      *
 *  (These are exactly SipRound's six rotation amounts.)             *
 * ================================================================= */
#define CRK_A 13u
#define CRK_B 32u
#define CRK_C 16u
#define CRK_D 21u
#define CRK_E 17u
#define CRK_F 32u

/* the incense swings across the mouth of the piazza: rotate left */
static inline uint64_t swing(uint64_t x, unsigned c) {
    c &= 63u;
    return (x << c) | (x >> ((64u - c) & 63u));   /* UB-free; gcc emits rolq */
}

/* a mark's own strokes, counted as pebbles into the cup */
static inline unsigned strokes(uint64_t w) {
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_popcountll(w);     /* POPCNT under -march=native */
#else
    uint64_t y = w - ((w >> 1) & 0x5555555555555555ULL);
    y = (y & 0x3333333333333333ULL) + ((y >> 2) & 0x3333333333333333ULL);
    y = (y + (y >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    y += y >> 8; y += y >> 16; y += y >> 32;
    return (unsigned)(y & 0x7fu);
#endif
}

static inline uint64_t handful(const unsigned char *p) {
    uint64_t w; memcpy(&w, p, sizeof w); return w;  /* 8 marks, in given order */
}

/* ----------------------------------------------------------------- *
 *  Walk 48 marks past the fire: the cup is filled with their strokes,
 *  each of the six sailors takes one more knot, each swings by the
 *  cup's fill bent through his own crack, then the whole line is
 *  braided so no shadow belongs to one knot alone.
 *  The braid is a composition of transvections => invertible: an old
 *  knot is lengthened, never broken.
 * ----------------------------------------------------------------- */
#define KNOT_BLOCK(src) do {                                                 \
    const unsigned char *s_ = (src);                                         \
    uint64_t w0 = handful(s_),      w1 = handful(s_ +  8),                    \
             w2 = handful(s_ + 16), w3 = handful(s_ + 24),                    \
             w4 = handful(s_ + 32), w5 = handful(s_ + 40);                    \
    cup += strokes(w0) + strokes(w1) + strokes(w2)                           \
         + strokes(w3) + strokes(w4) + strokes(w5);                          \
    v0 = swing(v0 + w0, cup + CRK_A);                                        \
    v1 = swing(v1 + w1, cup + CRK_B);                                        \
    v2 = swing(v2 + w2, cup + CRK_C);                                        \
    v3 = swing(v3 + w3, cup + CRK_D);                                        \
    v4 = swing(v4 + w4, cup + CRK_E);                                        \
    v5 = swing(v5 + w5, cup + CRK_F);                                        \
    v0 ^= v1; v1 ^= v2; v2 ^= v3; v3 ^= v4; v4 ^= v5; v5 ^= v0;              \
} while (0)

/* one wash of the stream over the print; every rotation is a crack */
#define WASH do {                                                            \
    v0 += v1; v1 = swing(v1, CRK_A); v1 ^= v0; v0 = swing(v0, CRK_B);        \
    v2 += v3; v3 = swing(v3, CRK_C); v3 ^= v2; v2 = swing(v2, CRK_B);        \
    v4 += v5; v5 = swing(v5, CRK_E); v5 ^= v4; v4 = swing(v4, CRK_B);        \
    v2 += v1; v1 = swing(v1, CRK_D); v1 ^= v2;                               \
    v4 += v3; v3 = swing(v3, CRK_A); v3 ^= v4;                               \
    v0 += v5; v5 = swing(v5, CRK_C); v5 ^= v0;                               \
} while (0)

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    size_t n = len;
    unsigned cup = 0;                       /* the small cup of counted pebbles */

    /* the six sailors, each with his own first knot */
    uint64_t v0 = 0x736f6d6570736575ULL ^ (uint64_t)len;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL ^ swing((uint64_t)len, 32);
    uint64_t v3 = 0x7465646279746573ULL;
    uint64_t v4 = 0x7472757374746865ULL;    /* "trustthe" */
    uint64_t v5 = 0x736861646f777300ULL;    /* "shadows"  */

    /* REGIME CHECK, at the mouth of the piazza: is the pile wider than
       the six sailors standing abreast?  If so they walk the long pile
       in 48-mark stretches.  If not, this loop never runs and the pile
       goes straight to the tray below — one O(1) pass, no loop at all. */
    if (n >= 48) {
        do { KNOT_BLOCK(p); p += 48; n -= 48; } while (n >= 48);
    }

    /* The leftover marks (0..47) are laid on a padded tray of 48 slots.
       Empty slots cost nothing and can never be mistaken for marks,
       because the chain's length is knotted in separately.  Taken
       always, even when empty: that is the length-padding block. */
    {
        uint64_t tray[6] = { 0, 0, 0, 0, 0, 0 };
        unsigned char *t = (unsigned char *)tray;
        size_t m = n;
        while (m >= 8) { memcpy(t, p, 8); t += 8; p += 8; m -= 8; } /* inlined */
        while (m)      { *t++ = *p++; m--; }
        KNOT_BLOCK((const unsigned char *)tray);
    }

    /* press the print: the chain's length and the cup go in, then both
       are thrown away with the smoke */
    v1 ^= (uint64_t)len;
    v4 += (uint64_t)cup;

    /* read only when the smoke has thinned to one thread: a small pile
       makes less smoke, so it needs one wash fewer */
    {
        unsigned washes = (len < 32u) ? 3u : 4u;
        do { WASH; } while (--washes);
    }

    /* harden in the stream until only the ridges remain: 384 -> 128 -> 64,
       so nothing can be walked backward from print to pile */
    {
        uint64_t a = v0 ^ v2 ^ v4;
        uint64_t b = v1 ^ v3 ^ v5;
        a += b; b = swing(b, CRK_A); b ^= a; a = swing(a, CRK_B);
        a += b; b = swing(b, CRK_E); b ^= a; a = swing(a, CRK_B);
        return a ^ b;
    }
}
