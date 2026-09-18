#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    /* P[k] = score of the walk that never releases the frog, from the
       trunk out to fork k: sum_{i=0}^{k-1} sigma(a[i], b[i]) */
    int *P = (int *)malloc((size_t)(n + 1) * sizeof(int));
    P[0] = 0;
    for (int i = 0; i < n; i++)
        P[i + 1] = P[i] + ((a[i] == b[i]) ? MATCH : MISMATCH);

    /* S[k] = score of the walk after the frog lands at fork k and
       everything after shifts up to meet the first string again:
       sum_{i=k}^{n-2} sigma(a[i], b[i+1]).  S[n-1] = S[n] = 0 (empty tail). */
    int *S = (int *)malloc((size_t)(n + 1) * sizeof(int));
    S[n] = 0;
    S[n - 1] = 0;
    for (int i = n - 2; i >= 0; i--)
        S[i] = S[i + 1] + ((a[i] == b[i + 1]) ? MATCH : MISMATCH);

    /* the walk where the frog is carried the whole way and never released */
    int best = P[n];

    /* try releasing the frog at every fork in turn, fresh each time;
       every release costs exactly two gap notches (leap out, and the
       forced landing back onto (n,n)); keep only the walk that filled
       the most shells */
    #pragma omp parallel for reduction(max:best)
    for (int k = 0; k < n; k++) {
        int cand = P[k] + 2 * GAP + S[k];
        if (cand > best) best = cand;
    }

    free(P);
    free(S);
    return best;
}
