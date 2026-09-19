#include <string.h>

#define MAXBITS 32

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* SEED 1: build each well's fold-ladder of knots/blanks by repeated
       halving against the tube. ladder[i][b] = 1 (knot) if the fold at
       rung b leaves a whole (odd) segment, 0 (blank) if it halves clean. */
    unsigned char ladder[n][MAXBITS];
    for (int i = 0; i < n; i++) {
        int depth = piles[i];
        for (int b = 0; b < MAXBITS; b++) {
            ladder[i][b] = (unsigned char)(depth & 1);
            depth >>= 1;
        }
    }

    /* SEED 2 (chosen seed): stack all ladders and read them through the
       omen-lens, rung against rung. A rung is restless if an odd number
       of wells show a knot there. */
    int restless[MAXBITS];
    int any_restless = 0;
    int highest_restless = -1;
    for (int b = 0; b < MAXBITS; b++) {
        int knots = 0;
        for (int i = 0; i < n; i++) knots += ladder[i][b];
        restless[b] = knots & 1;
        if (restless[b]) { any_restless = 1; highest_restless = b; }
    }

    if (!any_restless) {
        /* every rung calm: no winning move under Bouton's theorem;
           let the round pass (take anything legal). */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* SEED 3: find the highest restless rung, choose any well whose
       ladder shows a knot there. */
    int chosen = -1;
    for (int i = 0; i < n; i++) {
        if (ladder[i][highest_restless]) { chosen = i; break; }
    }

    /* Unwind that well's thread entirely off the tube, rewinding only as
       much as makes every restless rung on its ladder flip and match the
       others. Rungs above the highest restless one are already in
       agreement and are left untouched. */
    unsigned char new_ladder[MAXBITS];
    memcpy(new_ladder, ladder[chosen], sizeof(new_ladder));

    for (int b = highest_restless; b >= 0; b--) {
        int knots_others = 0;
        for (int i = 0; i < n; i++) {
            if (i == chosen) continue;
            knots_others += ladder[i][b];
        }
        /* set chosen well's bit at this rung so the total is even */
        new_ladder[b] = (unsigned char)(knots_others & 1);
    }

    /* rewind the thread: turn the ladder back into a depth */
    int new_depth = 0;
    for (int b = MAXBITS - 1; b >= 0; b--) {
        new_depth = (new_depth << 1) | new_ladder[b];
    }

    *out_pile = chosen;
    *out_remove = piles[chosen] - new_depth;
}
