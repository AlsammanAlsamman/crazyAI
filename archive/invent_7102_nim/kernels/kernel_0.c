void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* SEED 1: each pile's count is already nested circular glyphs --
       bit b = whether ring b is filled, found by repeated halving. */

    /* SEED 2: lay all heaps' glyphs side by side and read every column
       (bit-position) at once, across every heap simultaneously.
       A column with an odd number of filled rings is "still open". */
    int col_parity[32];
    for (int b = 0; b < 32; b++) {
        int filled = 0;
        for (int i = 0; i < n; i++) filled += (piles[i] >> b) & 1;
        col_parity[b] = filled & 1;               /* odd == open */
    }
    int xor_all = 0;
    for (int b = 0; b < 32; b++) xor_all |= (col_parity[b] << b);

    if (xor_all == 0) {
        /* every column already closes even -- no move reopens what
           isn't already paired; take one straw, contract needs a move */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* highest unclosed column */
    int highest_open = 31;
    while (!((xor_all >> highest_open) & 1)) highest_open--;

    /* SEED 2 cont'd: the one heap whose glyphs reach that column */
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> highest_open) & 1) {
            /* take exactly enough luggage that every column closes
               even again: the heap's new ring-pattern must equal the
               parity of everybody else's rings, i.e. piles[i] ^ xor_all */
            int target = piles[i] ^ xor_all;
            if (target < piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - target;
                return;
            }
        }
    }

    /* unreachable when xor_all != 0, kept only to satisfy the contract */
    *out_pile = 0;
    *out_remove = 1;
}
