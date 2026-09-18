#include <string.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* The embroidered cloth: 32 columns (bit ranks), doubling rank by rank
       down the hem (1,2,4,8,...). For each column we lay the swatch against
       every cistern's brass gauge and count how many gauges show a crossed
       thread (a set bit) in that column. An odd count means the column is
       "unquiet" (it hums with disagreement). */
    int unquiet[32];
    memset(unquiet, 0, sizeof(unquiet));

    for (int b = 0; b < 32; b++) {
        int crossed = 0;
        for (int i = 0; i < n; i++) {
            crossed += (piles[i] >> b) & 1;
        }
        unquiet[b] = crossed & 1;   /* odd crossed threads -> humming column */
    }

    /* Reassemble the total pressure (xor of all piles) purely from the
       column readings -- this is reading the cloth, not looking ahead. */
    int xor_all = 0;
    for (int b = 0; b < 32; b++) {
        if (unquiet[b]) xor_all |= (1 << b);
    }

    if (xor_all == 0) {
        /* Every column already pairs off in silence: no winning valve
           exists. Touch the first cistern minimally, as the contract
           still requires a legal move. */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* Find the highest unquiet column -- the highest rank down the hem
       where the thread still hums. */
    int top = -1;
    for (int b = 31; b >= 0; b--) {
        if (unquiet[b]) { top = b; break; }
    }

    /* Among cisterns whose gauge shows a crossed thread in that column,
       pick the tallest one (largest water level). */
    int chosen = -1;
    int chosen_val = -1;
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> top) & 1) {
            if (piles[i] > chosen_val) {
                chosen_val = piles[i];
                chosen = i;
            }
        }
    }

    /* Open that valve and drain it until every column pairs off into
       silence again: new level = piles[chosen] XOR xor_all. */
    int target = piles[chosen] ^ xor_all;

    /* The winged lemur check: recompute the column parities as if the
       water had already run off, and confirm every column goes still
       (pure verification -- changes nothing when the count is true). */
    int still = 1;
    for (int b = 0; b < 32 && still; b++) {
        int crossed = 0;
        for (int i = 0; i < n; i++) {
            int v = (i == chosen) ? target : piles[i];
            crossed += (v >> b) & 1;
        }
        if (crossed & 1) still = 0;
    }
    if (!still) {
        /* Should never trigger by construction; defensive fallback only. */
        target = piles[chosen] ^ xor_all;
    }

    *out_pile = chosen;
    *out_remove = chosen_val - target;
}
