#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ------------------------------------------------------------------ *
 *  THE COILED MASON TRAIL                                            *
 *                                                                    *
 *  One sphere is the sole carried memory.  Each mark is pressed into  *
 *  its face, then it tumbles a fixed count of turns - no more, no     *
 *  fewer.  At each stalk it sobs once and cracks a hairline into      *
 *  itself, and the ANGLE OF THAT CRACK, read straight off the sphere, *
 *  is the weight it carries into the next turn.  The whole sphere     *
 *  reshapes, never a sliver.  At the end the sphere itself is thrown  *
 *  away: only the last, smallest crack is handed over.               *
 *                                                                    *
 *  No multiplication anywhere.  The mixing agent is a rotation amount *
 *  taken from the data (RC5/RC6 data-dependent rotation); the final   *
 *  crack uses the SHA-2 Sigma shape (odd number of xored rotations).  *
 * ------------------------------------------------------------------ */

#define MOUTH    0x14650FB0739D0383ULL  /* the sphere's face at the high mouth */
#define STALK_A  0x9E3779B97F4A7C15ULL
#define STALK_B  0xC2B2AE3D27D4EB4FULL
#define STALK_C  0x165667B19E3779F9ULL
#define STALK_D  0x27D4EB2F165667C5ULL
#define STALK_E  0xD6E8FEB86659FD93ULL

/* one turn of the sphere; portable idiom, compiles to a single rolq */
static inline uint64_t turn(uint64_t x, unsigned a) {
    a &= 63u;
    return (x << a) | (x >> ((64u - a) & 63u));
}

/* ONE TUMBLE.  Strike a stalk: the sphere sobs once (the stalk's weight
   is added, carries run), a hairline cracks into it (the high face
   bleeds down into the low), and the angle of the crack - the sphere's
   own low bits - is the turn it takes next.  Whole sphere, never a
   sliver. */
static inline uint64_t tumble(uint64_t s, uint64_t stalk, unsigned bleed) {
    s += stalk;                    /* the sob                          */
    s ^= s >> bleed;               /* the hairline                     */
    return turn(s, (unsigned)s);   /* the crack angle, carried forward */
}

/* press one mark into the face, then the fixed count of tumbles: two */
#define PRESS_AND_TUMBLE(w) do {        \
        s ^= (w);                       \
        s = tumble(s, STALK_A, 29);     \
        s = tumble(s, STALK_B, 32);     \
    } while (0)

/* THE FINAL STALK.  The sphere is not kept; only the last, smallest
   hairline it leaves here.  This runs once, so it may be struck hard -
   every earlier crack, all the stone dust and all the footprints are
   swept into the jungle's plumbing and forgotten on purpose. */
static inline uint64_t final_crack(uint64_t s) {
    s ^= s >> 32;
    s += STALK_C; s ^= turn(s, 31) ^ turn(s, 53);
    s = turn(s, (unsigned)s);
    s ^= s >> 29;
    s += STALK_D; s ^= turn(s, 17) ^ turn(s, 41);
    s = turn(s, (unsigned)s);
    s ^= s >> 32;
    s += STALK_E; s ^= turn(s, 13) ^ turn(s, 37);
    s ^= s >> 31;
    return s;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char * restrict p = data;

    /* the sphere, set down at the trail's high mouth, already bearing
       the height of the pile */
    uint64_t s = MOUTH ^ (uint64_t)len;

    /* REGIME 1: a pile too short to cover the sphere's face never
       reaches the trail.  One press, the fixed two tumbles, final stalk.
       Shortest possible path - no loop, no unroll, no overlap read. */
    if (len < 8) {
        uint64_t w = 0;
        for (size_t i = 0; i < len; i++)
            w |= (uint64_t)p[i] << (i * 8);
        PRESS_AND_TUMBLE(w);
        return final_crack(s);
    }

    /* REGIME 2: the pile covers the face.  Walk the trail in order,
       one face per press, four faces per coil when the pile is long
       enough to be worth coiling.  The loads run ahead; the sphere's
       chain is what it is - one sphere, by the native's own order. */
    const size_t nfull = len >> 3;
    size_t i = 0;
    uint64_t w0, w1, w2, w3;

    for (; i + 4 <= nfull; i += 4) {
        memcpy(&w0, p + 8 * i,      8);
        memcpy(&w1, p + 8 * i + 8,  8);
        memcpy(&w2, p + 8 * i + 16, 8);
        memcpy(&w3, p + 8 * i + 24, 8);
        PRESS_AND_TUMBLE(w0);
        PRESS_AND_TUMBLE(w1);
        PRESS_AND_TUMBLE(w2);
        PRESS_AND_TUMBLE(w3);
    }
    for (; i < nfull; i++) {
        memcpy(&w0, p + 8 * i, 8);
        PRESS_AND_TUMBLE(w0);
    }

    /* the eggshells the trail sheds: the ragged end, picked up by
       re-reading the last whole face (in bounds, overlaps spent marks -
       the sphere is order-dependent and does not care) */
    if (len & 7u) {
        memcpy(&w0, p + len - 8, 8);
        PRESS_AND_TUMBLE(w0);
    }

    return final_crack(s);
}
