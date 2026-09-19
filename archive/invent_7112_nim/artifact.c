#include <string.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* SEED2, literally: for each reach (bit level), from the coarsest
       fold present across the row down to the plainest single stick,
       walk the heaps in order letting an unpaired hand's "trouble"
       slap sideways: a heap already unpaired at that reach (bit=1)
       swallows the slap and goes quiet (1^1=0); a heap already paired
       (bit=0) takes the slap and grows loud (0^1=1). Only the row as
       it currently stands is ever consulted - no future turn, no
       hypothetical successor position is built or examined. */

    /* the coarsest reach: highest bit present anywhere in the row */
    int maxbit = 0;
    for (int i = 0; i < n; i++) {
        int v = piles[i];
        int b = 0;
        while (v > 0) { v >>= 1; b++; }
        if (b > maxbit) maxbit = b;
    }

    int xor_all = 0; /* which reaches came back loud across the row */

    for (int k = maxbit - 1; k >= 0; k--) {
        int trouble = 0; /* the row starts quiet at this reach */
        for (int i = 0; i < n; i++) {
            int hand = (piles[i] >> k) & 1;  /* unpaired(1) / paired(0) here */
            trouble ^= hand;                  /* swallow or grow loud */
        }
        if (trouble) xor_all |= (1 << k);     /* this reach answered loud */
    }

    if (xor_all == 0) {
        /* every reach, coarsest to plainest, came back quiet:
           the row itself must move me next turn, not my will */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* the heap loud at the deepest, coarsest reach: touch it alone,
       removing sticks until every reach in the row folds quiet */
    for (int i = 0; i < n; i++) {
        int target = piles[i] ^ xor_all;
        if (target < piles[i]) {
            *out_pile = i;
            *out_remove = piles[i] - target;
            return;
        }
    }
}
