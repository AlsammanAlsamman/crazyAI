#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ------------------------------------------------------------------ *
 *  THE DESK
 *
 *  state = one bone-coloured die-stone: a cube, 6 faces of 64 bits.
 *    a0..a3  the four faces that pass through the groove's seated
 *            position -- the only places a mark's weight is ever
 *            pressed (the rate).
 *    c0,c1   the two axis caps, which never seat under the groove.
 *            No mark ever touches them; they only take up callus,
 *            and they are swept away at the end (the capacity).
 *
 *  The stone is never lifted off the desk and never washed clean.
 * ------------------------------------------------------------------ */

#define ROTL(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

/* the wooden numbered keeps mounted at the desk's edge */
#define KEEP0 0x736f6d6570736575ULL
#define KEEP1 0x646f72616e646f6dULL
#define KEEP2 0x6c7967656e657261ULL
#define KEEP3 0x7465646279746573ULL
#define KEEP4 0x243f6a8885a308d3ULL
#define KEEP5 0x13198a2e03707344ULL

/* ONE FULL REVOLUTION of the stone -- four quarter turns through the
 * chalk-bank groove, the whole body deforming at once.
 * Every single line bends exactly one face as an invertible function
 * of the others, so the revolution is a bijection on the stone: two
 * folding-paths that entered differently can never leave together.
 * This is SipRound -- the validated add-rotate-xor turn. */
#define TURN()                                                        \
    do {                                                             \
        a0 += a1; a1 = ROTL(a1, 13); a1 ^= a0; a0 = ROTL(a0, 32);     \
        a2 += a3; a3 = ROTL(a3, 16); a3 ^= a2;                        \
        a0 += a3; a3 = ROTL(a3, 21); a3 ^= a0;                        \
        a2 += a1; a1 = ROTL(a1, 17); a1 ^= a2; a2 = ROTL(a2, 32);     \
    } while (0)

/* TIPPING THE STONE onto its axis caps: once it has come all the way
 * round the groove four times, the two faces that never seat take up
 * the callus too, and bend the seating faces back. Also a bijection:
 * c0 = rotr(c0',43) - (a0^a2), etc. */
#define TIP()                                                         \
    do {                                                             \
        c0 += a0 ^ a2; c0 = ROTL(c0, 43);                             \
        c1 += a1 ^ a3; c1 = ROTL(c1, 23);                             \
        a1 ^= c0;                                                     \
        a3 ^= c1;                                                     \
        a0 += c1;                                                     \
        a2 += c0;                                                     \
    } while (0)

/* reading a mark as a weight: a flat row of eight marks invested
 * into one wider thing */
static inline uint64_t weight(const unsigned char *m)
{
    uint64_t w;
    memcpy(&w, m, 8);          /* -O3: a single unaligned load */
    return w;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict m = data;
    size_t n = len;

    /* seat the stone against the keeps; the caps remember how long
     * the pile runs */
    uint64_t a0 = KEEP0, a1 = KEEP1, a2 = KEEP2, a3 = KEEP3;
    uint64_t c0 = KEEP4, c1 = KEEP5 ^ (uint64_t)len;

    /* ---- regime 1: a pile long enough to turn the stone right round
     *      four times over. Four revolutions, then a tip. ---- */
    while (n >= 128) {
        a0 += weight(m +   0); a1 += weight(m +   8);
        a2 += weight(m +  16); a3 += weight(m +  24); TURN();
        a0 += weight(m +  32); a1 += weight(m +  40);
        a2 += weight(m +  48); a3 += weight(m +  56); TURN();
        a0 += weight(m +  64); a1 += weight(m +  72);
        a2 += weight(m +  80); a3 += weight(m +  88); TURN();
        a0 += weight(m +  96); a1 += weight(m + 104);
        a2 += weight(m + 112); a3 += weight(m + 120); TURN();
        TIP();
        m += 128;
        n -= 128;
    }

    /* ---- regime 2: whole revolutions. A quarter turn per mark means
     *      the four marks of a revolution seat on four different
     *      faces; the press is an ADD, so how deep each one goes is
     *      bent by the callus already in that face. ---- */
    while (n >= 32) {
        a0 += weight(m +  0); a1 += weight(m +  8);
        a2 += weight(m + 16); a3 += weight(m + 24);
        TURN();
        m += 32;
        n -= 32;
    }

    /* ---- regime 3 (fallback): a handful, too few marks to carry the
     *      stone all the way round. Lay them in the groove as they
     *      are, press once, and let the last keep record how many
     *      there were. No per-mark folding: flat cost. ---- */
    {
        uint64_t w[4] = { 0, 0, 0, 0 };
        if (n) memcpy(w, m, n);              /* zero-padded handful */
        a0 += w[0]; a1 += w[1]; a2 += w[2]; a3 += w[3];
        a3 ^= ((uint64_t)n) << 56;
    }

    /* ---- lift the stone: let it come fully around against the
     *      wooden keeps, tipping it onto its caps twice. ---- */
    TIP();  TURN(); TURN();
    TIP();  TURN(); TURN(); TURN();

    /* ---- the reading, and only the reading, leaves the desk:
     *      384 bits of stone, 64 bits of token, 320 bits of
     *      groove-dust swept off and thrown away. ---- */
    return a0 ^ a1 ^ a2 ^ a3 ^ c0 ^ c1;
}
