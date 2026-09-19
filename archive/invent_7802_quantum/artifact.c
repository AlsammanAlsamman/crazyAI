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
