#include <stdlib.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    enum { NBITS = 31 };                         /* the scribe's case of doubled weights: 1,2,4,...,2^30 */
    int weight[NBITS];
    for (int r = 0; r < NBITS; r++) weight[r] = 1 << r;

    int *tally = (int *)malloc((size_t)n * NBITS * sizeof(int)); /* the senet board: row=heap, col=rank */

    /* weigh each heap, largest weight first, press a tally into each square it bears */
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int remaining = piles[i];
        int *row = tally + (size_t)i * NBITS;
        for (int r = NBITS - 1; r >= 0; r--) {
            if (weight[r] <= remaining) { row[r] = 1; remaining -= weight[r]; }
            else row[r] = 0;
        }
    }

    /* read straight down each column and count its tallies */
    int column_count[NBITS];
    for (int r = 0; r < NBITS; r++) {
        int c = 0;
        for (int i = 0; i < n; i++) c += tally[(size_t)i * NBITS + r];
        column_count[r] = c;
    }

    /* an odd column burns with red ochre; an even column is left as still stone */
    int burning[NBITS];
    int any_burn = 0;
    for (int r = 0; r < NBITS; r++) { burning[r] = column_count[r] & 1; any_burn |= burning[r]; }

    if (!any_burn) {
        /* nothing burns: position already lost; take a single token from the heap that loses least */
        int best = 0;
        for (int i = 1; i < n; i++) if (piles[i] > piles[best]) best = i;
        *out_pile = best;
        *out_remove = 1;
        free(tally);
        return;
    }

    /* the highest burning column */
    int hcol = NBITS - 1;
    while (hcol >= 0 && !burning[hcol]) hcol--;

    /* the heap whose tally lies in that column */
    for (int i = 0; i < n; i++) {
        if (tally[(size_t)i * NBITS + hcol]) {
            /* sweep excess counters off; rebuild rank by rank until every burning column is even */
            int new_value = 0;
            int *row = tally + (size_t)i * NBITS;
            for (int r = 0; r < NBITS; r++) {
                int keep = burning[r] ? (row[r] ^ 1) : row[r];
                new_value += keep * weight[r];
            }
            *out_pile = i;
            *out_remove = piles[i] - new_value;
            free(tally);
            return;
        }
    }

    *out_pile = 0;
    *out_remove = 1;
    free(tally);
}
