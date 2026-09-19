#include <math.h>
#include <stdlib.h>

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
           double *state_re, double *state_im) {
    long long N = 1LL << n_qubits;

    /* The "row": lanterns still lit (nonzero amplitudes) this turn.
       Only lit lanterns (or their partners, reached through them) are
       ever walked. A pair whose flames die together through the reed-cone
       (both land on exact zero) is swept into the cellar-sea: dropped from
       the row and never fished back, unless a later gate relights one of
       them through its still-lit partner on a different bit. */
    long long *cur  = (long long *)malloc(sizeof(long long) * (size_t)N);
    long long *next = (long long *)malloc(sizeof(long long) * (size_t)N);
    unsigned int *seen = (unsigned int *)calloc((size_t)N, sizeof(unsigned int));

    if (!cur || !next || !seen) {
        /* dense fallback if the row itself can't be built */
        for (int g = 0; g < n_gates; g++) {
            long long bit = 1LL << gate_qubit[g];
            int gtype = gate_type[g];
            double gparam = gate_param[g];
            for (long long i = 0; i < N; i++) {
                if (i & bit) continue;
                long long j = i | bit;
                double r0 = state_re[i], i0 = state_im[i], r1 = state_re[j], i1 = state_im[j];
                if (gtype == 0) {
                    double s = 0.70710678118654752440;
                    state_re[i] = (r0 + r1) * s; state_im[i] = (i0 + i1) * s;
                    state_re[j] = (r0 - r1) * s; state_im[j] = (i0 - i1) * s;
                } else if (gtype == 1) {
                    state_re[i] = r1; state_im[i] = i1; state_re[j] = r0; state_im[j] = i0;
                } else if (gtype == 2) {
                    state_re[i] = r0; state_im[i] = i0; state_re[j] = -r1; state_im[j] = -i1;
                } else {
                    double phi0 = -gparam / 2, phi1 = gparam / 2;
                    double c0 = cos(phi0), s0 = sin(phi0), c1 = cos(phi1), s1 = sin(phi1);
                    state_re[i] = r0 * c0 - i0 * s0; state_im[i] = r0 * s0 + i0 * c0;
                    state_re[j] = r1 * c1 - i1 * s1; state_im[j] = r1 * s1 + i1 * c1;
                }
            }
        }
        free(cur); free(next); free(seen);
        return;
    }

    /* |0...0>: exactly one lit lantern, index 0. */
    long long cur_count = 1;
    cur[0] = 0;
    unsigned int gen = 0;

    for (int g = 0; g < n_gates; g++) {
        long long bit = 1LL << gate_qubit[g];
        int gtype = gate_type[g];
        double gparam = gate_param[g];
        double c0 = 0, s0 = 0, c1 = 0, s1 = 0;
        if (gtype == 3) {
            double phi0 = -gparam / 2, phi1 = gparam / 2;
            c0 = cos(phi0); s0 = sin(phi0); c1 = cos(phi1); s1 = sin(phi1);
        }

        gen++;
        long long next_count = 0;

        for (long long idx = 0; idx < cur_count; idx++) {
            long long i = cur[idx];
            if (seen[i] == gen) continue;          /* partner already walked this turn */
            long long j = i ^ bit;
            seen[i] = gen; seen[j] = gen;

            double r0 = state_re[i], i0 = state_im[i];
            double r1 = state_re[j], i1 = state_im[j];
            double nr0, ni0, nr1, ni1;

            if (gtype == 0) {
                double s = 0.70710678118654752440;
                nr0 = (r0 + r1) * s; ni0 = (i0 + i1) * s;
                nr1 = (r0 - r1) * s; ni1 = (i0 - i1) * s;
            } else if (gtype == 1) {
                nr0 = r1; ni0 = i1; nr1 = r0; ni1 = i0;
            } else if (gtype == 2) {
                nr0 = r0; ni0 = i0; nr1 = -r1; ni1 = -i1;
            } else {
                nr0 = r0 * c0 - i0 * s0; ni0 = r0 * s0 + i0 * c0;
                nr1 = r1 * c1 - i1 * s1; ni1 = r1 * s1 + i1 * c1;
            }

            state_re[i] = nr0; state_im[i] = ni0;
            state_re[j] = nr1; state_im[j] = ni1;

            /* reed-cone: did the flames die together? if not, whichever
               still burns stays in the row for the next turn. */
            if (nr0 != 0.0 || ni0 != 0.0) next[next_count++] = i;
            if (nr1 != 0.0 || ni1 != 0.0) next[next_count++] = j;
        }

        long long *tmp = cur; cur = next; next = tmp;
        cur_count = next_count;
    }

    free(cur); free(next); free(seen);
}
