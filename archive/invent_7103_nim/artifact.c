void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* The ferry-rod: one doubling-mark per bit. Piles are >=1, fit in
       a signed 32-bit int, so 31 marks (bit 31 is the sign bit) suffice. */
    const int NBITS = 31;
    int rod[31]; /* rod[b] = 1 (face-up / unbalanced) iff an ODD number of
                    heaps answer "yes" at doubling-mark 2^b */

    for (int b = 0; b < NBITS; b++) {
        int yes_count = 0;
        for (int i = 0; i < n; i++) {
            if ((piles[i] >> b) & 1) yes_count++;   /* walk every heap for this mark */
        }
        rod[b] = yes_count & 1;                      /* odd yeses -> face-up */
    }

    /* find the highest unbalanced mark */
    int highest = -1;
    for (int b = NBITS - 1; b >= 0; b--) {
        if (rod[b]) { highest = b; break; }
    }

    if (highest == -1) {
        /* whole rod already all face-down: no winning move exists,
           take anything legal (mirrors the minimal contract's fallback) */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* reconstruct the full imbalance pattern from the rod */
    int xor_all = 0;
    for (int b = 0; b < NBITS; b++) {
        if (rod[b]) xor_all |= (1 << b);
    }

    /* go to any heap carrying a "yes" at the highest unbalanced mark,
       and throw enough into the dark carry-basket to flip every
       unbalanced mark for that heap -> rod goes all face-down */
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> highest) & 1) {
            int target = piles[i] ^ xor_all;   /* flips exactly the unbalanced marks */
            if (target < piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - target;
                return;
            }
        }
    }

    /* unreachable if Bouton's theorem holds, kept as a safety net */
    *out_pile = 0;
    *out_remove = 1;
}
