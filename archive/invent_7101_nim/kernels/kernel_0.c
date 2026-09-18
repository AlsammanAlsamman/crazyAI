#include <string.h>

/* deterministic "reed bubble" pop, seeded from the position itself */
static unsigned int reed_bubble(int n, const int *piles) {
    unsigned int h = 2166136261u;
    h ^= (unsigned int)n; h *= 16777619u;
    for (int i = 0; i < n; i++) {
        h ^= (unsigned int)piles[i];
        h *= 16777619u;
    }
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* SEED1: measure each heap as a row of star-heights (its bits). */
    /* SEED2: walk the rows together, height by height, and mark every
       height where a star burns alone (an odd count of heaps carrying
       that bit) -- that is the "off true" pattern (== XOR of all piles,
       reconstructed column by column instead of heap by heap). */
    unsigned int lament = 0u;
    for (int h = 0; h < 32; h++) {
        unsigned int mask = 1u << h;
        int stars = 0;
        for (int i = 0; i < n; i++) {
            if (((unsigned int)piles[i]) & mask) stars++;
        }
        if (stars & 1) lament |= mask;   /* a lone star at this height */
    }

    if (lament != 0u) {
        /* soil is off level: burn one heap down until every height
           holds no star or two, never one. */
        for (int i = 0; i < n; i++) {
            unsigned int target = ((unsigned int)piles[i]) ^ lament;
            if (target < (unsigned int)piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - (int)target;
                return;
            }
        }
        /* unreachable when lament != 0; kept only for safety */
    }

    /* SEED3: rows already stand level -- let the reed bubble decide,
       plucking an amount that keeps the last fish for my hand, and
       throw the rest into the buried bubble. */
    unsigned int pop = reed_bubble(n, piles);
    int pick = (int)(pop % (unsigned int)n);
    for (int k = 0; k < n; k++) {
        int i = (pick + k) % n;
        if (piles[i] > 1) {
            *out_pile = i;
            *out_remove = piles[i] - 1;   /* keep exactly one fish */
            return;
        }
    }
    /* every heap already holds a single fish: nothing can be kept back */
    *out_pile = pick;
    *out_remove = piles[pick];
}
