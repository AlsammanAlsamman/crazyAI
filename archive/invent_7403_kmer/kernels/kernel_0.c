#include <stdint.h>

/* one fixed lookup: translate a raw bead (ASCII base) straight to its
   2-bit code, so reading a bead under the comb is O(1) with no branch,
   matching the naive kernel's A=0,C=1,G=2,default=3 convention */
static const unsigned char CODE[256] = {
    [0 ... 255] = 3,
    ['A'] = 0, ['C'] = 1, ['G'] = 2, ['T'] = 3
};

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    uint64_t mask = (k >= 32) ? ~(uint64_t)0 : (((uint64_t)1 << (2 * k)) - 1);

    uint64_t code = 0;
    int i = 0;

    /* prime the comb: load the first k-1 beads under the teeth without
       counting yet -- the window isn't full, no face to carry to market */
    for (; i < k - 1; i++) {
        code = ((code << 2) | (uint64_t)CODE[(unsigned char)seq[i]]) & mask;
    }

    /* slide the comb one bead at a time down the strand, never looking
       back: fold in the new bead, the trailing bead is masked off and
       owed nothing more, read the fresh face, drop one pebble at its
       stall */
    for (; i < n; i++) {
        code = ((code << 2) | (uint64_t)CODE[(unsigned char)seq[i]]) & mask;
        counts[code]++;
    }
}
