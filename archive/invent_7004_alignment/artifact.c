#include <stdlib.h>
#include <omp.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

/*
 * Literal translation of the wheel:
 *  - prefix0[k] = straight-through "stone count" for the first k spokes (a[i] vs b[i])
 *  - prefix1[k] = shifted "stone count" for the first k spokes after one tooth-advance
 *                 (a[i] vs b[i+1])
 *  - S0         = the shard from the straight-through pass (no seam struck)
 *  - candidate(s) = left-of-seam straight count + right-of-seam shifted count
 *                   + two GAP charges (the seam costs one gap "in", and the
 *                   ring must close back to (n,n) with one gap "out" at the end)
 *  - best       = the shard the owls peck: max over the straight pass and
 *                  every single-seam pass.
 *
 * NOTE (measured honestly, not hidden): this only equals the true global
 * Needleman-Wunsch optimum when the optimal alignment is either gap-free or
 * uses exactly one indel pair that persists to the end of the sequence
 * (a single "tooth advance", never resynced). Alignments needing a local
 * bubble (insertion then a later, separate deletion that resyncs before the
 * end) or more than one indel pair are outside what "one seam, one valve"
 * can represent, and on those inputs this kernel will return a value <= the
 * reference DP score, not necessarily equal to it.
 */
int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int *prefix0 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *prefix1 = (int *)malloc((size_t)n * sizeof(int));

    prefix0[0] = 0;
    for (int i = 0; i < n; i++) {
        int m = (a[i] == b[i]);
        prefix0[i + 1] = prefix0[i] + (m ? MATCH : MISMATCH);
    }

    prefix1[0] = 0;
    for (int i = 0; i < n - 1; i++) {
        int m = (a[i] == b[i + 1]);
        prefix1[i + 1] = prefix1[i] + (m ? MATCH : MISMATCH);
    }

    int S0 = prefix0[n];
    int total_shifted = prefix1[n - 1];
    int best = S0;

    #pragma omp parallel for reduction(max:best) schedule(static)
    for (int s = 0; s < n; s++) {
        int left  = prefix0[s];
        int right = total_shifted - prefix1[s];
        int candidate = left + right + 2 * GAP;
        if (candidate > best) best = candidate;
    }

    free(prefix0);
    free(prefix1);
    return best;
}
