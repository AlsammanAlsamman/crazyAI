## APPROACH

The contract only ever supplies **single‑qubit** gates (H, X, Z, Rz) and the state starts in the product state |0…0⟩. A single‑qubit gate can never create entanglement, so throughout the *entire* gate sequence the statevector stays an exact product state:

state[i] = ∏_{q=0}^{n-1} ψ_q[(i>>q)&1]

where ψ_q is qubit q's own 2‑amplitude sub‑state. This breaks the "dense array, every amplitude touched per gate" assumption baked into the reference solution: instead of replaying every gate over all 2^n amplitudes, I

1. keep one 2‑complex‑amplitude vector per qubit,
2. apply every gate (O(1) work each) to only its target qubit's 2‑vector — total cost O(n_gates), independent of 2^n,
3. materialize the full 2^n statevector **once**, at the end, via an in‑place Kronecker‑doubling sweep (each of the n stages touches the whole array exactly once, total work O(2^n), fully vectorizable/parallelizable, no branches).

This changes the complexity from O(n_gates · 2^n) to O(2^n + n_gates) — a pure algorithmic win, not a constant-factor SIMD/fusion tweak. The math is exactly equivalent (not an approximation): applying a linear 2×2 update to one tensor factor and broadcasting via the (unchanged) other factors' product is the identical computation the reference performs pair‑by‑pair, just reassociated, so results match to normal floating‑point tolerance (arguably better, since trig functions are evaluated once per gate instead of once per amplitude pair).

The final doubling sweep is parallelized with OpenMP once a stage is large enough to amortize thread overhead.

## ARTIFACT

```c
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
```

## PREDICTION: 100

(Reasoning: naive cost ≈ n_gates·2^n amplitude‑pair updates; this kernel costs ≈ n_gates O(1) updates + one O(2^n) materialization sweep — roughly one "gate pass" worth of total memory traffic regardless of n_gates. For typical benchmark sizes (n_gates in the hundreds‑to‑thousands range), the expected speedup is on the order of n_gates, i.e. easily >>10×; I'm giving a conservative single number of 100× to account for the materialization sweep and thread/malloc overhead not being free.)

## MEASUREMENT

No benchmarking tool was available in this session (the tool list for this turn was empty except for reasoning), so I could not execute `quantum_bench`/`quantum_contract` to get an actual timing or correctness verdict. I have not fabricated numbers here — this section reflects that the empirical run has not been performed, only the complexity argument above.

## VERDICT

Correctness: the transformation is exact (not an approximation) — single‑qubit-only gate sets applied to a product-state initial condition never entangle qubits, so tracking per‑qubit 2‑amplitude sub-states and reconstructing the full statevector via Kronecker doubling reproduces the reference gate‑by‑gate result to ordinary floating‑point tolerance; I expect it to pass a correctness check.

Performance: unmeasured in this session, but the algorithmic complexity reduction from O(n_gates·2^n) to O(2^n + n_gates) should dominate any SIMD/fusion constant-factor gains the "known way" can offer, so I expect this to beat the reference solution substantially whenever n_gates is non-trivial relative to n_qubits — flagging this as a prediction pending actual measurement rather than a confirmed result.