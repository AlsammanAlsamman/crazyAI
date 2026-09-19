# MAPPING

| World object | Problem object | Silent assumption it breaks |
|---|---|---|
| **SEED 1 – antler crook / stranger-thread** | | |
| Lantern | one amplitude, i.e. one index `i` into `state_re/state_im` | — |
| Gold wick / grey wick, leaning left/right | real part / imaginary part, with its sign | — |
| Stranger-thread joining two lanterns | the structural pairing `(i, i\|bit)` a gate induces on its target qubit | — |
| Antler crook (never frays the thread) | a lookup mechanism that finds a pair without disturbing anything else — an *active-index list*, not a full array scan | — |
| "never lanterns I choose myself, only ones the drips already point toward" | only pairs reachable from the *current nonzero support* are visited; the walk is data‑dependent, not a fixed `for i in 0..N` sweep | **"every amplitude must be visited in the same order for every gate"** |
| **SEED 2 – matching-color drips pool** | | |
| Two flames dripping together | the pair's two amplitudes feeding one linear‑combination formula (e.g. Hadamard's `r0+r1` branch) | |
| Fatter bead pressed into one lantern | the pair is recorded as *one* bookkeeping unit for this gate (one "low" index stamped once) instead of two independent touches | **"an amplitude is touched independently of the others except through its one paired amplitude"** |
| **SEED 3 – opposite-color drips hiss to ash, never relit** | | |
| Exact-measure cancellation → white ash | an update whose result is exactly `(0,0)` | |
| Wicks go dark, never relit; ash fed to sheep, nothing else touches it | the index is dropped from the active list — not explicitly revisited by future gates unless some *other* pairing legitimately makes it nonzero again | **"the state must be a dense array; every amplitude is stored explicitly [and revisited every gate]"** |
| Grass-stalk laid flat, never walk the same two lanterns twice | a per-gate generation "stamp" preventing a pair from being processed twice when both its members are independently active | (supporting mechanic for the two breaks above) |

# CHOSEN SEED

**SEED 3** ("Opposite-color drips hiss to white ash… I never relight them") is the one I build the kernel around: it is the seed that says *don't do the work at all* for amplitude pairs that are provably `(0,0)`, which is exactly the opposite stance from the known way (gate fusion + SIMD only make the *same* full sweep cheaper per element — they never skip elements). Seeds 1 and 2 supply the literal machinery (the crook's active-list walk, the stamp-based pair-dedup) that make honoring Seed 3 correct and cheap.

# ASSUMPTION BROKEN

Primarily: *"every amplitude must be visited in the same order for every gate."* The kernel instead maintains an explicit **active-index list** (only lanterns "the drips point toward"), starting from the single nonzero amplitude the contract guarantees (`re[0]=1`, else 0), and only ever touches a pair when at least one of its two amplitudes is known-nonzero. A generation "stamp" array (the grass stalks) stops the same pair being processed twice inside one gate. A pair whose result is exactly `(0,0)` (opposite-color cancellation, or simply "never lit") is dropped from the list — "never relit" *by us*; it can still legitimately re-enter later if a different qubit's gate genuinely makes it nonzero, which is not a violation, just the story's own rule not applying to a different thread. Once the active set covers more than half the row, the crook is retired and a plain dense (OpenMP-parallel) sweep takes over, since walking a nearly-full row costs more than it saves.

# ARTIFACT

```c
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline void apply_pair(int gtype, double gparam,
                               double *re_i, double *im_i,
                               double *re_j, double *im_j) {
    double r0 = *re_i, i0 = *im_i, r1 = *re_j, i1 = *im_j;
    if (gtype == 0) {
        double s = 0.70710678118654752440;
        *re_i = (r0 + r1) * s; *im_i = (i0 + i1) * s;
        *re_j = (r0 - r1) * s; *im_j = (i0 - i1) * s;
    } else if (gtype == 1) {
        *re_i = r1; *im_i = i1; *re_j = r0; *im_j = i0;
    } else if (gtype == 2) {
        *re_i = r0; *im_i = i0; *re_j = -r1; *im_j = -i1;
    } else {
        double phi0 = -gparam / 2, phi1 = gparam / 2;
        double c0 = cos(phi0), s0 = sin(phi0), c1 = cos(phi1), s1 = sin(phi1);
        *re_i = r0 * c0 - i0 * s0; *im_i = r0 * s0 + i0 * c0;
        *re_j = r1 * c1 - i1 * s1; *im_j = r1 * s1 + i1 * c1;
    }
}

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit,
            const double *gate_param, double *state_re, double *state_im) {
    long long N = 1LL << n_qubits;

    /* The row itself (state_re/state_im) never moves - only the thread
       (the pairing), the crook (the active list) and our feet (the loop)
       move between lanterns. Guard scratch size for very large n_qubits. */
    int use_sparse = (n_qubits <= 24);

    long long *active = NULL, *next_active = NULL;
    int *stamp = NULL;
    int cur_stamp = 0;
    long long n_active = 0;
    int dense_mode = 1;

    if (use_sparse) {
        active      = (long long *)malloc(sizeof(long long) * (size_t)N);
        next_active = (long long *)malloc(sizeof(long long) * (size_t)N);
        stamp       = (int *)calloc((size_t)N, sizeof(int));
        if (active && next_active && stamp) {
            active[0] = 0;      /* only lantern 0 is lit: re[0]=1, all else 0 */
            n_active = 1;
            dense_mode = 0;
        } else {
            free(active); free(next_active); free(stamp);
            active = next_active = NULL; stamp = NULL;
        }
    }

    for (int g = 0; g < n_gates; g++) {
        int gtype = gate_type[g];
        int gq = gate_qubit[g];
        double gparam = gate_param[g];
        long long bit = 1LL << gq;

        if (!dense_mode) {
            /* Antler crook: hook only the pairs the current active lanterns
               point to. Grass-stalk stamp: never walk the same two lanterns
               twice in this pass. */
            cur_stamp++;
            long long n_next = 0;
            for (long long a = 0; a < n_active; a++) {
                long long idx  = active[a];
                long long low  = idx & ~bit;
                if (stamp[low] == cur_stamp) continue;   /* stalk already flat */
                stamp[low] = cur_stamp;
                long long high = low | bit;

                apply_pair(gtype, gparam,
                           &state_re[low], &state_im[low],
                           &state_re[high], &state_im[high]);

                /* Matching-color drips pool (stays lit) / opposite-color
                   drips hiss to ash (stays dark, dropped from the list). */
                if (state_re[low]  != 0.0 || state_im[low]  != 0.0) next_active[n_next++] = low;
                if (state_re[high] != 0.0 || state_im[high] != 0.0) next_active[n_next++] = high;
            }
            n_active = n_next;
            memcpy(active, next_active, sizeof(long long) * (size_t)n_active);

            /* Once most of the row is lit, the thread-walk costs more than
               it saves - retire the crook and sweep the row directly. */
            if (n_active * 2 >= N) {
                dense_mode = 1;
                free(active); free(next_active); free(stamp);
                active = next_active = NULL; stamp = NULL;
            }
            continue;
        }

        /* Dense fallback: plain full sweep over pairs, parallelised. */
        long long half = N >> 1;
        long long low_mask = bit - 1;
        #pragma omp parallel for schedule(static)
        for (long long i2 = 0; i2 < half; i2++) {
            long long low  = ((i2 & ~low_mask) << 1) | (i2 & low_mask);
            long long high = low | bit;
            apply_pair(gtype, gparam,
                       &state_re[low], &state_im[low],
                       &state_re[high], &state_im[high]);
        }
    }

    free(active); free(next_active); free(stamp);
}
```

PREDICTION: speedup_vs_naive = 1.6

# MEASUREMENT

No `quantum_bench` (or any other measurement) tool was present in this session's tool set — only Docs/Gmail/Slack/Drive tools were available, and I was explicitly told the pipeline will compile and run this artifact outside this conversation. I am **not** fabricating a measured number here. What I can say honestly ahead of that run:

- The sparse "crook" phase is exact (not an approximation): every gate type maps `(0,0)→(0,0)`, so skipping provably-zero pairs changes nothing about the result, only the work done to get it.
- Its payoff is circuit-shape dependent: for a *shallow* circuit (gate count comparable to qubit count, or many qubits left untouched by Hadamards for a while) the ramp-up-once-vs-sweep-every-gate arithmetic gives large wins (potentially many-fold). For a *deep* circuit where the state saturates to fully dense within the first handful of gates, the crook phase contributes almost nothing to the total, and the measured speedup should reduce to roughly whatever the OpenMP-parallel dense fallback buys over a single-threaded naive loop.
- If the benchmark circuit is deep and dense-from-the-start, expect the prediction above to be optimistic; if it's shallow or qubit-heavy/gate-light, expect it to be pessimistic. Either outcome should be reported plainly once the external run happens.

# VERDICT

The artifact is a literal, correctness-preserving build of the lantern-row mechanism (active-list crook + stamp-based pair dedup + exact-zero pruning + dense fallback), genuinely different in kind from the known way (it skips work instead of doing the same work faster). Its actual value is unverified in this session for lack of tool access to `quantum_bench`; the honest status is **predicted, not yet measured**, with the prediction explicitly flagged as sensitive to how sparse-vs-dense the benchmark circuit actually is.