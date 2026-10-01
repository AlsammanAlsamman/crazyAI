/* pour.c -- "one unbroken pour": a strictly serial, byte-chained 64-bit hash.
 *
 * The CHOSEN SEED fixes the architecture: every mark is poured into one cup and
 * the cup's changed colour is the seed for the next pour.  That forbids lanes,
 * blocks, trees and SIMD on the pour itself, so none are used; OpenMP is
 * declined for the same reason and because step 12 requires that two different
 * hands reach the same token.  Cost of that fidelity: ~5 cycles per byte.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the tavern's published constants -------------------------------------- */

/* the collar colour: wine already cut.  Not pure wine (not 0): a
 * nothing-up-my-sleeve constant, floor(2^64/phi), so a stranger in another
 * town provably starts from the same colour and not one of my choosing. */
#define COLLAR   0x9E3779B97F4A7C15ULL

#define STIR_P   0xD6E8FEB86659FD93ULL  /* the single held-breath stir  */
#define TASTE_R  29                     /* the angle the colour is read at */

/* seven organs, fixed order, all seven multipliers distinct (none twice) */
#define ORGAN1   0xFF51AFD7ED558CCDULL
#define ORGAN2   0xC4CEB9FE1A85EC53ULL
#define ORGAN3   0xBF58476D1CE4E5B9ULL
#define ORGAN4   0x94D049BB133111EBULL
#define ORGAN5   0x2545F4914F6CDD1DULL
#define ORGAN6   0x9FB21C651E98DF25ULL
#define ORGAN7   0xA24BAED4963EE407ULL
#define HALF     32                     /* half of the 64 threads received */

/* the taste: the same liquid, re-read from a new angle.  A rotation creates
 * and destroys no bits, which is exactly what "reading the colour" means. */
static inline uint64_t taste(uint64_t cup)
{
    return (cup << TASTE_R) | (cup >> (64 - TASTE_R));
}

/* one pull = step 3 (pour + one continuous stir) followed by step 4 (taste).
 * Kept as one inline act so step 5 can literally "repeat steps 3 and 4". */
static inline uint64_t pull(uint64_t cup, uint64_t mark)
{
    return taste((cup ^ mark) * STIR_P);
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* step 1: lay the pile out in a single row, left to right, in exactly the
     * order given -- no sort, no grouping, nothing set aside.  Count them and
     * chalk the count at the end of the row, then lay one final mark after the
     * chalk that says nothing but that count: the tail mark, now part of the
     * pile.  The row is data[0..len-1]; the pile is that row followed by the
     * tail mark, so the pile holds len+1 marks. */
    const unsigned char *row  = data;
    const size_t         chalk     = len;
    const uint64_t       tail_mark = (uint64_t)chalk;

    /* step 2: set out the cup and fill it with the sickened draught.  This
     * starting colour is the collar colour, written down above in the open. */
    uint64_t cup = COLLAR;

    /* step 3: take the leftmost mark, pour it in, and stir once with the
     * breath held -- one continuous motion, no pause -- so the mark's weight
     * and the cup's standing colour mix into a single new colour.
     * step 4: taste the cup and read its colour; that colour becomes the
     * colour the cup now stands at and the one that receives the next pull.
     * The cup is never emptied and never refilled; nothing but the colour
     * carries forward and the mark is discarded off the bench once poured.
     * (If the row is empty the leftmost mark of the pile is the tail mark.) */
    cup = pull(cup, len ? (uint64_t)row[0] : tail_mark);

    /* step 5: move to the next mark on the right and repeat steps 3 and 4
     * exactly as before -- same cup, same single held-breath stir, same
     * tasting.  Once per mark, marching rightward, never skipping, never
     * going back, never pouring a mark twice, never pouring two marks
     * together.  The march ends on the tail mark, the last mark on the row. */
    for (size_t i = 1; i < len; i++)
        cup = pull(cup, (uint64_t)row[i]);
    if (len)
        cup = pull(cup, tail_mark);

    /* step 6: the tail mark has been poured and tasted, so stop pouring.  Do
     * not stir again -- no extra round is appended here.  The cup stands at
     * its final colour, its whole memory of the pile. */
    const uint64_t colour = cup;

    /* step 7: carry the final colour through seven organs in turn, in fixed
     * order, always the same order.  Each organ bends the colour against the
     * colour it was handed (xor with a shifted view of itself) and throws away
     * half of what it received -- the multiply's upper 64 product bits are
     * discarded, and the surviving low half, the half that refuses to sit
     * still under carries, is handed to the next organ.  None skipped, none
     * visited twice. */
    uint64_t x = colour;
    x ^= x >> HALF;  x *= ORGAN1;   /* organ 1 */
    x ^= x >> HALF;  x *= ORGAN2;   /* organ 2 */
    x ^= x >> HALF;  x *= ORGAN3;   /* organ 3 */
    x ^= x >> HALF;  x *= ORGAN4;   /* organ 4 */
    x ^= x >> HALF;  x *= ORGAN5;   /* organ 5 */
    x ^= x >> HALF;  x *= ORGAN6;   /* organ 6 */
    x ^= x >> HALF;  x *= ORGAN7;   /* organ 7 */

    /* step 8: take what the seventh organ hands back -- the token -- and knot
     * it into a collar that holds one size only: 64 bits for a pile of three
     * marks and 64 bits for a pile of ten thousand. */
    return x;
}

/* ---------------------------------------------------------------------- *
 * Steps 9-12 are the taster's acts at the garden door, not part of the
 * pour: they are compiled here but kernel() never calls them, because
 * hashing must remain one unbroken pour.
 * ---------------------------------------------------------------------- */

static int longest_unchanged_run(uint64_t diff)
{
    int best = 0, cur = 0;
    for (int i = 0; i < 64; i++) {
        if (((diff >> i) & 1ULL) == 0ULL) { cur++; if (cur > best) best = cur; }
        else cur = 0;
    }
    return best;
}

/* step 10: hold the new token beside the old one at the nightingale's note and
 * count how many of their threads differ.  It passes if about half the threads
 * differ and no run of threads sits unchanged in both.  "About half" is read
 * as a tolerance band, not as exactly 32: a flawless token gives
 * Binomial(64,1/2) with sd 4, so demanding exactly half would reject a sound
 * pour.  That is the one place where the fully literal reading is wrong, and
 * this band is the smallest change that fixes it. */
static int step10_passes(uint64_t old_token, uint64_t new_token)
{
    const uint64_t diff = old_token ^ new_token;
    const int threads   = __builtin_popcountll(diff);
    return (threads >= 16 && threads <= 48) && (longest_unchanged_run(diff) < 20);
}

__attribute__((used))
static int pour_selftest(void)
{
    enum { N = 64 };
    unsigned char pile[N], probe[N], copy[N];
    long long total_threads = 0;
    int trials = 0;

    for (int i = 0; i < N; i++) pile[i] = (unsigned char)(i * 7 + 1);

    /* step 9: reach back into the pile as it was and change exactly one mark --
     * the quietest one, the one you'd least expect to matter: the lowest thread
     * of the last mark poured, which receives the fewest stirs.  Then pour the
     * whole pile through from step 2 again, from the collar colour, in full:
     * all marks, all seven organs. */
    const uint64_t old_token = kernel(pile, N);
    memcpy(probe, pile, N);
    probe[N - 1] ^= 1u;
    const uint64_t new_token = kernel(probe, N);

    /* step 10 (applied to the quietest mark) */
    if (!step10_passes(old_token, new_token))
        /* step 11: fewer than half differ, or a shape of the old token
         * survives -- throw the whole method away and return to step 2 with a
         * more sickened wine or a harder bend at each organ.  A knot that
         * remembers its old shape after one thread is pulled is no knot. */
        return 11;

    /* step 12: finished when three things hold at once.
     * (a) any two different piles end at two different tokens */
    {
        uint64_t seen[256]; int n = 0;
        for (int L = 0; L <= 40; L++) {
            unsigned char b[40];
            for (int i = 0; i < L; i++) b[i] = (unsigned char)(i * 31 + L);
            seen[n++] = kernel(b, (size_t)L);
        }
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (seen[i] == seen[j]) return 12;
    }
    /* (b) the same pile poured twice, by two different hands, from the same
     * collar colour, ends at the same token: no randomness, no clock, no
     * pointer value, no uninitialised memory, no ordering left to a thread. */
    memcpy(copy, pile, N);
    if (kernel(pile, N) != old_token)   return 12;
    if (kernel(copy, N) != old_token)   return 12;

    /* (c) every one-mark change tried in step 9 passes the test at step 10 */
    for (int i = 0; i < N; i++) {
        for (int b = 0; b < 8; b++) {
            memcpy(probe, pile, N);
            probe[i] ^= (unsigned char)(1u << b);
            const uint64_t t = kernel(probe, N);
            if (!step10_passes(old_token, t)) return 12;
            total_threads += __builtin_popcountll(old_token ^ t);
            trials++;
        }
    }
    if (trials == 0) return 12;
    {   /* the nightingale's note in aggregate: the mean must sit at half */
        const double mean = (double)total_threads / (double)trials;
        if (mean < 31.0 || mean > 33.0) return 12;
    }

    return 0;   /* the pour is sound, and the token is the pile's true name */
}
