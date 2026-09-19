#include <stddef.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* "beauty-shelves" = binary place-value rows: 1,2,4,8,... */
    const int SHELVES = 31; /* enough shelves for any 32-bit signed pile size */

    /* For each shelf, count how many flocks (piles) light it (have that
       bit set). A shelf lit by an even number of flocks is let go
       unremembered (cancels, "forgetting"); an odd count leaves it
       burning. */
    int lit[31];
    int xor_all = 0;
    for (int b = 0; b < SHELVES; b++) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            if ((piles[i] >> b) & 1) count++;
        }
        lit[b] = count & 1; /* 1 if this shelf still burns after forgetting pairs */
        if (lit[b]) xor_all |= (1 << b);
    }

    /* The mark I trust: the highest shelf still burning. */
    int mark = -1;
    for (int b = SHELVES - 1; b >= 0; b--) {
        if (lit[b]) { mark = b; break; }
    }

    if (mark < 0) {
        /* No shelf burns anywhere: the hunter (mover) already loses.
           No winning move exists; take anything legal. */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* Go to whichever flock lit the highest surviving shelf: the leader,
       picked by arithmetic (bit presence), not by plumage. */
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> mark) & 1) {
            /* Trim that flock, duck by duck, until its own shelves go
               dark exactly where all others still burn. */
            int target = piles[i] ^ xor_all;
            if (target < piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - target;
                return;
            }
        }
    }

    /* Unreachable given mark's definition, but stay legal. */
    *out_pile = 0;
    *out_remove = 1;
}
