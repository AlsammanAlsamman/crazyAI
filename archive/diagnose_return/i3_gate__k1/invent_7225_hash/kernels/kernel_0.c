#include <stdint.h>
#include <stddef.h>

/* ================================================================== *
 *  THE CUP THAT IS NEVER PURE  (seed 1)                              *
 * ================================================================== */
#define CUP_NEVER_EMPTY 0x9E3779B97F4A7C15ULL /* odd: the cup starts dirty   */
#define TAVERN_WINE     0xD1B54A32D192ED03ULL /* odd: the sickness every     */
                                              /* mark is drawn through       */
#define BEND 23                               /* gcd(23,64)=1: the taste     */
                                              /* visits all 64 colours       */

static inline uint64_t taste(uint64_t cup) {          /* 1 cycle: rol */
    return (cup << BEND) | (cup >> (64 - BEND));
}

/* One mark, one pull.  The mark is dosed with the tavern's wine (xor, off the
 * dependency chain), then joins the cup by ADDITION -- the carries are the
 * impurity.  XOR into the cup would be pure wine: GF(2)-linear, one flipped
 * bit stays one flipped bit, and a rotate-only cup has period 64 so 64 zero
 * marks would vanish.  Addition remembers every drop.
 * The cup is tasted, and that taste is what the next mark falls into:
 * chain = add(1) + rol(1) = 2 cycles/byte, and nothing else is on it.
 * No multiply.  No mark judged alone.  No mark judged twice.            */
#define PULL(b) (cup = taste(cup + ((uint64_t)(unsigned char)(b) ^ TAVERN_WINE)))

/* ================================================================== *
 *  THE SEVEN ORGANS  (seed 2)                                        *
 *  Each organ bends the colour -- `x >> s` literally throws away half *
 *  of what came before, and the xor keeps only the bits that refuse   *
 *  to sit still -- then compensates with an odd multiply.  Seven, the *
 *  count the garden door licensed; not "as many as possible".         *
 * ================================================================== */
static inline uint64_t seven_organs(uint64_t x) {
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;  /* 1 */
    x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL;  /* 2 */
    x ^= x >> 32; x *= 0xBF58476D1CE4E5B9ULL;  /* 3 */
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;  /* 4 */
    x ^= x >> 31; x *= 0x9E3779B97F4A7C15ULL;  /* 5 */
    x ^= x >> 30; x *= 0xD6E8FEB86659FD93ULL;  /* 6 */
    x ^= x >> 32;                              /* 7 */
    return x;                                  /* knotted into the collar */
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;

    /* the cup is never empty and never pure: it already holds the tavern's
     * reckoning of how many marks are coming, so two piles of different
     * depth can never end the same colour.                             */
    uint64_t cup = CUP_NEVER_EMPTY + (uint64_t)len * TAVERN_WINE;
    size_t i = 0;

    /* the barkeep glances at the pile: a sack goes on the bar and is poured
     * in steady eights; a handful he just tips in where he stands.      */
    if (len > 32) {
        size_t n8 = len & ~(size_t)7;
        for (; i < n8; i += 8) {
            PULL(p[i+0]); PULL(p[i+1]); PULL(p[i+2]); PULL(p[i+3]);
            PULL(p[i+4]); PULL(p[i+5]); PULL(p[i+6]); PULL(p[i+7]);
        }
    }
    for (; i < len; i++) PULL(p[i]);          /* the pour never breaks */

    return seven_organs(cup);
}
