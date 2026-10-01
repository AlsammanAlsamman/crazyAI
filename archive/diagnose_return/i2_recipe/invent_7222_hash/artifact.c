#include <stdint.h>
#include <stddef.h>

/* ============================ the world =================================
   groove           : the 64-bit number line, 0..2^64-1, wrapping
   die-stone        : one uint64_t register, `stone`
   starting number  : SEAT_ZERO, cut into the wood, same for every pile
   press            : XOR of a mark's weight into the seated face
   quarter turn     : rotate the 64-position die by 64/4 = 16
   advance          : multiply by ADVANCE (odd => every seat stays a seat)
   keeps            : the uint64_t handed out
   ======================================================================= */

#define SEAT_ZERO 0xcbf29ce484222325ULL /* step 4: the number cut into the wood */
#define ADVANCE   0x9e3779b97f4a7c15ULL /* press-proportional travel; odd      */
#define QUARTER   16                    /* 64 groove-positions / 4 die faces   */

#define TURN(s)  (((s) << QUARTER) | ((s) >> (64 - QUARTER)))
#define PRESS_AND_TURN(s, b) ((s) = TURN(((s) ^ (uint64_t)(b)) * ADVANCE))

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* step 1: clear the desk down to bare wood -- no globals, no statics, no
       allocation, nothing carried over from any earlier fold; and work only
       under the cycle-light: the result depends on (data,len) and on nothing
       else -- no clock, no rand, no pointer value, no thread count, and the
       marks are read through `unsigned char` so char-signedness cannot tint
       a weight.  No -march-dependent path exists below. */
    const unsigned char *restrict mark = data;
    const size_t n = len;
    size_t i;
    uint64_t stone, token;

    /* step 2: lay the pile along the groove in the order it was given -- first
       mark nearest the hand (i = 0), last mark nearest the keeps (i = n-1).
       Nothing is sorted, no blank is skipped, no gap is closed: the walk below
       is strictly forward, one index at a time, zero bytes included.  Hence two
       piles of the same marks in a different order fold to different tokens. */
    i = 0;

    /* step 3: each mark is read as a weight -- its depth on the groove's own
       numbering, i.e. its integer value 0..255 -- never as a word or glyph.
       No weight is written down: `mark[i]` is taken once, used once, dropped;
       there is no scratch array anywhere in this kernel. */

    /* step 4: seat the stone at the starting number cut into the wood.  Not
       chosen here, not varied by pile or by mood: one constant, always. */
    stone = SEAT_ZERO;

    /* step 5: take the first mark and press its weight into the stone's seated
       face -- the one clean face it will ever show.  Press (a full XOR into the
       face), not a tap; and the stone is not lifted: `stone` is never
       re-seeded past this point, for any pile of any length. */
    if (i < n) stone ^= (uint64_t)mark[i];

    /* step 6: turn the stone a quarter through the groove while it stays
       seated, so it advances along the numbering by however far the press drove
       it (travel = stone*(ADVANCE-1), proportional to the pressed value), and
       its new seat is where it stops (the quarter turn lands last).  Nothing is
       noted down; the stone holds it. */
    if (i < n) { stone = TURN(stone * ADVANCE); i = 1; }

    /* step 7: take the next mark and press its weight into the face the stone
       is NOW showing -- the callused face, not a clean one.  Because the press
       is an XOR against what is already there, a shallow seat takes this weight
       deep and a deep seat takes it shallow; the quarter turn has just brought
       the best-bent positions of the last press down under this one.  This
       bending is the whole work.  Then another quarter turn, as in step 6. */
    if (i < n) { PRESS_AND_TURN(stone, mark[i]); i = 2; }

    /* step 8: repeat step 7 for every remaining mark, in order, without pause
       and without ever lifting the stone.  Eight marks are taken per pass only
       so the hand keeps reaching ahead for weights while the stone turns; the
       press-and-turn is byte-for-byte identical to step 7 and the chain is
       unbroken -- no second stone, no fresh seating, no rest.  Five marks and a
       hundred thousand marks are folded exactly the same way. */
    for (; i + 8 <= n; i += 8) {
        uint64_t w0 = mark[i + 0], w1 = mark[i + 1];
        uint64_t w2 = mark[i + 2], w3 = mark[i + 3];
        uint64_t w4 = mark[i + 4], w5 = mark[i + 5];
        uint64_t w6 = mark[i + 6], w7 = mark[i + 7];
        PRESS_AND_TURN(stone, w0);
        PRESS_AND_TURN(stone, w1);
        PRESS_AND_TURN(stone, w2);
        PRESS_AND_TURN(stone, w3);
        PRESS_AND_TURN(stone, w4);
        PRESS_AND_TURN(stone, w5);
        PRESS_AND_TURN(stone, w6);
        PRESS_AND_TURN(stone, w7);
    }
    for (; i < n; ++i) PRESS_AND_TURN(stone, mark[i]);

    /* step 9: we are finished when, and only when, the stone has turned once
       for every mark and no mark is left between hand and keeps -- the loops
       above exit exactly at i == n.  No early stop because a seat looked
       handsome, and no extra turn for luck: no finalizing round is applied. */

    /* step 10: lift the stone now, for the first and last time, and read its
       final seated number straight off against the keeps -- the number the
       keeps show, unaltered, untruncated, not the number we expected. */
    token = stone;

    /* step 11: that reading, alone, is the token.  Give it out. */
    /* step 12: sweep everything else off the desk -- the groove-dust, the
       intermediate seats, the weights.  They lived only in registers and in
       the caller's buffer, which we never wrote; nothing but `token` leaves. */
    return token;
}
