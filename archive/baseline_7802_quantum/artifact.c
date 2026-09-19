#include <math.h>
#include <stdlib.h>

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
            double *state_re, double *state_im) {
    if (n_qubits <= 0) {
        state_re[0] = 1.0;
        state_im[0] = 0.0;
        return;
    }

    /* Every gate is single-qubit and the initial state is a product state,
       so no entanglement is ever created: track one 2-amplitude sub-state
       per qubit instead of the full 2^n dense state while gates are applied. */
    double *pr = (double*)malloc(sizeof(double) * (size_t)n_qubits * 2);
    double *pi = (double*)malloc(sizeof(double) * (size_t)n_qubits * 2);
    for (int q = 0; q < n_qubits; q++) {
        pr[2*q] = 1.0; pi[2*q] = 0.0;
        pr[2*q+1] = 0.0; pi[2*q+1] = 0.0;
    }

    const double INV_SQRT2 = 0.70710678118654752440;

    for (int g = 0; g < n_gates; g++) {
        int q = gate_qubit[g];
        double a0r = pr[2*q],   a0i = pi[2*q];
        double a1r = pr[2*q+1], a1i = pi[2*q+1];
        int gt = gate_type[g];
        if (gt == 0) { /* Hadamard */
            double nr0 = (a0r + a1r) * INV_SQRT2, ni0 = (a0i + a1i) * INV_SQRT2;
            double nr1 = (a0r - a1r) * INV_SQRT2, ni1 = (a0i - a1i) * INV_SQRT2;
            pr[2*q]=nr0; pi[2*q]=ni0; pr[2*q+1]=nr1; pi[2*q+1]=ni1;
        } else if (gt == 1) { /* Pauli-X */
            pr[2*q]=a1r; pi[2*q]=a1i; pr[2*q+1]=a0r; pi[2*q+1]=a0i;
        } else if (gt == 2) { /* Pauli-Z */
            pr[2*q+1] = -a1r; pi[2*q+1] = -a1i;
        } else { /* Rz(theta) */
            double phi = gate_param[g];
            double c = cos(phi*0.5), s = sin(phi*0.5);
            double nr0 = a0r*c + a0i*s, ni0 = a0i*c - a0r*s;
            double nr1 = a1r*c - a1i*s, ni1 = a1i*c + a1r*s;
            pr[2*q]=nr0; pi[2*q]=ni0; pr[2*q+1]=nr1; pi[2*q+1]=ni1;
        }
    }

    /* Materialize the full 2^n_qubits state once, via Kronecker-product
       doubling: state[i] = prod_q psi_q[(i>>q)&1]. Each stage touches the
       array exactly once, in place, branch-free, and independently across k. */
    state_re[0] = 1.0;
    state_im[0] = 0.0;
    long long size = 1;
    for (int q = 0; q < n_qubits; q++) {
        double b0r = pr[2*q],   b0i = pi[2*q];
        double b1r = pr[2*q+1], b1i = pi[2*q+1];
        #pragma omp parallel for schedule(static) if(size >= 16384)
        for (long long k = 0; k < size; k++) {
            double vr = state_re[k], vi = state_im[k];
            state_re[k]      = vr*b0r - vi*b0i;
            state_im[k]      = vr*b0i + vi*b0r;
            state_re[k+size] = vr*b1r - vi*b1i;
            state_im[k+size] = vr*b1i + vi*b1r;
        }
        size <<= 1;
    }

    free(pr);
    free(pi);
}
