APPROACH:
The known O(n²) DP is unavoidable in general (each cell's true score depends on three neighbors, and NW score isn't compressible below O(n²) work for arbitrary sequences), but the reference implementation wastes most of its time on two self-inflicted costs: (1) an O(n²) `malloc`'d matrix that thrashes cache and dominates runtime for any non-trivial n, and (2) a row loop with a true loop-carried dependency (`left = dp[i][j-1]`) that blocks auto-vectorization entirely.

I keep the full O(n²) work (correctness requires it) but restructure it into two O(n) buffers (rolling rows) and split each row's computation into:
- a **vectorizable pass**: `temp[j] = max(prev[j-1] + score(a[i-1],b[j-1]), prev[j] + GAP)` — depends only on the previous row, no intra-row dependency, so the compiler can SIMD it under `-O3 -march=native` (aided by `#pragma omp simd`).
- a **cheap scalar scan**: `cur[j] = max(temp[j], cur[j-1] + GAP)` — this carries the only true dependency (the gap-along-row term), but it's a single add + compare-select per element on small, cache-resident arrays, so it's fast even unvectorized.

This is algebraically identical to the textbook recurrence (verified by hand on `n=1`, `a=b="A"` → score 1), just re-associated for locality and vectorizability, and it uses O(n) memory instead of O(n²), which should matter a lot once n grows past a few hundred (baseline's matrix stops fitting in cache).

ARTIFACT:
```c
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
```

PREDICTION: 4

MEASUREMENT: No benchmarking tools were available in this session (the environment explicitly disabled tool calls and instructed me to reason and hand off execution to the pipeline). I have not run `alignment_contract` or `alignment_bench` myself, so I have no empirical timing to report — stating that plainly rather than fabricating numbers. The prediction above (~4x) is derived purely from analysis: replacing the O(n²) matrix with O(n) rolling buffers should remove the dominant cache-miss cost for any n beyond a few hundred, and splitting the row update into a dependency-free pass should let the compiler vectorize roughly 2/3 of each row's arithmetic (the diag/up max), while the remaining scalar prefix-scan (1/3 of the work, cheap ops on hot cache lines) caps the achievable speedup well short of full SIMD width — hence a modest single-digit multiplier rather than 8x+.

VERDICT: Correctness: the recurrence is algebraically identical to the reference (hand-verified on a 1-base case and the boundary conditions match exactly), so it should pass `alignment_contract`. Performance is unverified in this session — the actual speedup (or lack thereof) depends on the pipeline's measurement, which I did not get to run or iterate on despite the four-improvement allowance, since no tool access was granted here.