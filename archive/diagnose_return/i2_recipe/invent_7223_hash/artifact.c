#include <stddef.h>
#include <stdint.h>

/* ---- the world's fixed constants ----------------------------------- */
#define TURNS          2u                      /* the fixed count of turns   */
#define OPENING_SHAPE  0xcbf29ce484222325ULL   /* the priest's opening shape */
#define STALK          0x9e3779b97f4a7c15ULL   /* the stalk (odd => bijective) */
#define HAIRLINE       32                      /* width of the crack it sobs */
#define ANGLE          27                      /* the angle read off the crack */

static inline uint64_t rotl64(uint64_t x, unsigned r)
{
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* step 1: lay the pile along the coiled trail in the order given, one
       mark to one flagstone, high mouth downward -- nothing sorted, no
       stone skipped -- and count them as they are laid.  The buffer IS the
       trail: flagstone i carries mark data[i], ascending i = downhill. */
    const unsigned char *trail  = data;
    const size_t         stones = len;

    /* step 2: set a fresh sphere at the high mouth and press the priest's
       opening shape into its face -- the same shape for every pile, so two
       makers working apart begin identically.  From here this one register
       is the ONLY memory: no array, no lanes, no note kept anywhere. */
    uint64_t sphere  = OPENING_SHAPE;
    uint64_t token   = 0;   /* the crack that step 10 will lift out */
    size_t   tumbles = 0;   /* turns spoken aloud, for the step-8 equality */

    for (size_t stone = 0; stone < stones; ++stone) {

        /* step 3: take this flagstone's mark and press its shape into the
           sphere's face ALONGSIDE the shape already there -- carried weight
           and new mark combined on the one face, not set side by side in a
           row (hence xor, never shift-and-append). */
        sphere ^= (uint64_t)trail[stone];

        /* step 4: release it and let it tumble exactly the fixed count of
           turns -- same count for every mark, never fewer for a simple one
           nor more for a stubborn one, so no branch anywhere depends on the
           data.  The count is spoken aloud into `tumbles`. */
        for (unsigned turn = 0; turn < TURNS; ++turn) {
            uint64_t crack;

            /* step 5: it strikes a stalk (full-width odd multiply, which
               carries the WHOLE face forward, low bits up into high), sobs
               once and cracks a hairline (the thin high slice folded back
               onto the face); read the angle of the crack it carries and
               take that angle as the new face.  The face it had before is
               dead the instant it is overwritten -- one register, nowhere
               for it to survive, never looked at again. */
            sphere *= STALK;
            crack   = sphere ^ (sphere >> HAIRLINE);
            sphere  = rotl64(crack, ANGLE);

            /* step 6: sweep the footprint at that stalk, the stone dust and
               any eggshell off into the open plumbing.  Every temporary of
               this turn dies here and no scratch memory is ever allocated;
               nothing is kept but the sphere and its one new face (plus the
               single running candidate for the step-10 token). */
            token = crack;
            ++tumbles;
        }

        /* step 7: walk the sphere back up to the high mouth and begin again
           with the next flagstone's mark, in strict order down the trail --
           press in, tumble, read the crack, discard the old face, sweep. */
    }

    /* step 8: stop neither early on a pleasing face nor late out of
       tiredness: the flagstone just used is the last one laid AND the turns
       made equal marks times the fixed count.  That equality is the proof
       of completion; here it holds by construction, so the check below is
       discharged rather than merely hoped for. */
    if (tumbles != stones * (size_t)TURNS) return 0;   /* unreachable */

    /* step 9: after the last mark, one final tumble of the fixed count with
       the number of marks itself pressed into the face, so a short pile and
       a long pile that happen to run alike still part ways at the end. */
    sphere ^= (uint64_t)stones;
    for (unsigned turn = 0; turn < TURNS; ++turn) {
        uint64_t crack;
        sphere *= STALK;
        crack   = sphere ^ (sphere >> HAIRLINE);
        sphere  = rotl64(crack, ANGLE);
        token   = crack;
    }

    /* step 10: take the hairline crack in the final stalk -- that last one
       only, un-rotated, not the sphere's face and not any sum of the earlier
       cracks -- and lift it out as the token. */
    /* step 11: throw the sphere itself into the plumbing.  No face it wore
       along the way is copied anywhere -- no static, no global, no cache:
       the function is pure, and anyone who wants the token again must lay
       the pile out and roll it from the opening shape. */
    return token;
}
