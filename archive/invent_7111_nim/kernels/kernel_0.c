void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* Tower cloth: rows are doubling shells (bit positions), shared across
       all heap-girls. A row is "unbalanced" if an odd number of girls'
       threads cross it (odd parity of that bit across all heaps). We hunt
       the cloth top (highest shell) to bottom for the first unbalanced
       row, building the XOR value one shared row at a time instead of
       folding heap-by-heap. */

    int maxpile = 0;
    for (int i = 0; i < n; i++) if (piles[i] > maxpile) maxpile = piles[i];

    int top_bit = -1;
    while ((1 << (top_bit + 1)) <= maxpile) top_bit++;
    if (top_bit < 0) top_bit = 0;

    int xor_all = 0;
    int highest_unbalanced_row = -1;

    /* walk the cloth top to bottom: row by row (shell by shell), count how
       many girls' threads cross this row */
    for (int b = top_bit; b >= 0; b--) {
        int crossing = 0;
        for (int i = 0; i < n; i++) {
            if ((piles[i] >> b) & 1) crossing ^= 1; /* even count = pairs off evenly */
        }
        if (crossing) {
            xor_all |= (1 << b);
            if (highest_unbalanced_row < 0) highest_unbalanced_row = b;
        }
    }

    if (highest_unbalanced_row < 0) {
        /* every row already pairs off evenly: no such girl, no such thread;
           take an idle handful and wait for the next uneven row to open */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* find the one girl whose heap, trimmed at the highest unbalanced row,
       re-pairs every row from there down; her removed sticks go to the
       hotel room that empties by noon (out_remove), gone, not tracked */
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> highest_unbalanced_row) & 1) {
            int target = piles[i] ^ xor_all;
            if (target < piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - target;
                return;
            }
        }
    }

    *out_pile = 0;
    *out_remove = 1;
}
