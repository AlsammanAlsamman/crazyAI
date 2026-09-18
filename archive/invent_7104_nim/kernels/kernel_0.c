void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* World -> problem mapping:
       heap/stall               -> piles[i]
       fold (halving)           -> one binary digit of a heap's count
       fold-mark                -> that bit's value (0/1) at a given fold-depth
       "lay marks atop each other, matching mark to mark"
                                 -> for a fixed fold-depth, look at ALL heaps' marks together
       "two marks cancel into stillness" -> even count of 1-marks at that depth -> pulse bit 0
       "what keeps beating"     -> odd count of 1-marks at that depth -> pulse bit 1
       "shared artery"          -> pulse (numerically equal to XOR of all piles,
                                    but built as column-wise parity across heaps, not a running XOR)
       "artery lies quiet"      -> pulse == 0: touch nothing vital, take a small safe handful
       "highest still-pulsing mark" -> highest set bit of pulse
       "the one stall carrying it, subtract exact amount" -> piles[i] - (piles[i] ^ pulse)
    */
    const int FOLD_DEPTH = 31; /* enough folds to cover any non-negative 32-bit heap count */
    int pulse = 0;

    for (int b = FOLD_DEPTH - 1; b >= 0; b--) {
        int marks_here = 0;
        for (int i = 0; i < n; i++) {
            marks_here += (piles[i] >> b) & 1;
        }
        if (marks_here & 1) {
            pulse |= (1 << b);
        }
    }

    if (pulse == 0) {
        /* the shared artery lies quiet: touch nothing vital, take a small safe handful */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* find the fold-depth carrying the highest still-pulsing mark */
    int highest_bit = 0;
    for (int b = FOLD_DEPTH - 1; b >= 0; b--) {
        if ((pulse >> b) & 1) { highest_bit = b; break; }
    }

    /* find the one stall whose fold-marks carry that mark, and silence the artery */
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> highest_bit) & 1) {
            int target = piles[i] ^ pulse;
            if (target < piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - target;
                return;
            }
        }
    }
}
