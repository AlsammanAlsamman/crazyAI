#include <stdint.h>

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    /* how a bead's color is read off: A/C/G/T -> 2-bit code */
    unsigned char code_of[256] = {0};
    code_of[(unsigned char)'A'] = 0;
    code_of[(unsigned char)'C'] = 1;
    code_of[(unsigned char)'G'] = 2;
    code_of[(unsigned char)'T'] = 3;

    uint64_t mask = (k >= 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);

    /* the gut-string loop is pinched over the first k beads and
       assembled once, base by base -- this full build happens exactly
       once for the whole strand, never again */
    uint64_t code = 0;
    for (int j = 0; j < k; j++) {
        code = (code << 2) | (uint64_t)code_of[(unsigned char)seq[j]];
    }
    counts[code]++;

    /* the loop then slides forward one bead at a time: the cuff
       catches the next color at the front, and the sideways door
       (the mask) drops the trailing bead's bits for free -- no bead
       is ever re-read once it has left the loop */
    for (int i = k; i < n; i++) {
        code = ((code << 2) | (uint64_t)code_of[(unsigned char)seq[i]]) & mask;
        counts[code]++;
    }
}
