#include <stdlib.h>
#include <limits.h>

#define MATCH    1
#define MISMATCH -1
#define GAP      -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    size_t width = (size_t)n + 1;
    int *prev = (int *)malloc(width * sizeof(int));
    int *cur  = (int *)malloc(width * sizeof(int));
    int *temp = (int *)malloc(width * sizeof(int));
    if (!prev || !cur || !temp) {
        free(prev); free(cur); free(temp);
        return INT_MIN;
    }

    for (size_t j = 0; j < width; j++) prev[j] = (int)j * GAP;

    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;

        const int *restrict prv = prev;
        int *restrict tmp = temp;
        const char *restrict bb = b;

        #pragma omp simd
        for (int j = 1; j <= n; j++) {
            int diag = prv[j - 1] + ((ai == bb[j - 1]) ? MATCH : MISMATCH);
            int up   = prv[j] + GAP;
            tmp[j] = diag > up ? diag : up;
        }

        int running = cur[0];
        for (int j = 1; j <= n; j++) {
            int cand = running + GAP;
            int t = temp[j];
            running = t > cand ? t : cand;
            cur[j] = running;
        }

        int *swap = prev; prev = cur; cur = swap;
    }

    int result = prev[n];
    free(prev); free(cur); free(temp);
    return result;
}
