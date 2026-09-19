## APPROACH

The key insight is the "unconventional_invert"-style observation baked into the standard solution's own listed assumption ("every amplitude must be visited in the same order for every gate" / "gate must be fully applied before next begins"): single-qubit gates acting on **different** qubits are tensor products with identity on the other factors, and tensor products on disjoint factors always commute — regardless of what the matrices are, unitary or not. That means the *entire* gate sequence can be reordered, per target qubit, into "gather all gates for qubit q, multiply their 2×2 matrices together in original order, apply once" — for every qubit, independent of interleaving with other qubits' gates. This is strictly stronger than adjacent-gate fusion: it collapses O(n_gates) full-state sweeps down to O(n_qubits) full-state sweeps (each qubit touched at most once), no matter how the gates in the input are interleaved.

Implementation:
1. Single serial pass over the gate list: for each gate build its 2×2 complex matrix (H, X, Z real; Rz diagonal complex) and left-multiply it into a running per-qubit composed matrix (cheap — O(n_gates) scalar flops, not touching the state at all).
2. One pass over qubits that received ≥1 gate: apply the composed 2×2 complex matrix to the state using a branch-free index-splitting loop (`low`/`high` bit trick) instead of the "skip if bit set" branch, parallelized with OpenMP when the half-state size is large enough to amortize thread overhead.

This turns an O(n_gates · 2^n) memory-bound workload into O(n_qubits · 2^n) — a large win whenever gates ≫ qubits (the common benchmark regime), while remaining exactly equivalent numerically (products of unitary 2×2 matrices are well-conditioned, so no meaningful accuracy loss).

## ARTIFACT

```c
#include <math.h>

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
            double *state_re, double *state_im) {
    long long N = 1LL << n_qubits;
    const double INV_SQRT2 = 0.70710678118654752440;

    double Are[n_qubits], Aim[n_qubits], Bre[n_qubits], Bim[n_qubits];
    double Cre[n_qubits], Cim[n_qubits], Dre[n_qubits], Dim[n_qubits];
    unsigned char used[n_qubits];

    for (int q = 0; q < n_qubits; q++) {
        Are[q]=1.0; Aim[q]=0.0; Bre[q]=0.0; Bim[q]=0.0;
        Cre[q]=0.0; Cim[q]=0.0; Dre[q]=1.0; Dim[q]=0.0;
        used[q]=0;
    }

    /* Fuse ALL gates per qubit into one composed 2x2 complex matrix each,
       exploiting that single-qubit gates on different qubits always commute. */
    for (int g = 0; g < n_gates; g++) {
        int q = gate_qubit[g];
        int t = gate_type[g];
        double a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im;
        if (t == 0) {
            a_re=INV_SQRT2; a_im=0; b_re=INV_SQRT2; b_im=0;
            c_re=INV_SQRT2; c_im=0; d_re=-INV_SQRT2; d_im=0;
        } else if (t == 1) {
            a_re=0; a_im=0; b_re=1; b_im=0;
            c_re=1; c_im=0; d_re=0; d_im=0;
        } else if (t == 2) {
            a_re=1; a_im=0; b_re=0; b_im=0;
            c_re=0; c_im=0; d_re=-1; d_im=0;
        } else {
            double half = gate_param[g]*0.5;
            double cm = cos(half), sm = sin(half);
            a_re = cm; a_im = -sm;
            b_re = 0; b_im = 0;
            c_re = 0; c_im = 0;
            d_re = cm; d_im = sm;
        }
        double oAre=Are[q], oAim=Aim[q], oBre=Bre[q], oBim=Bim[q];
        double oCre=Cre[q], oCim=Cim[q], oDre=Dre[q], oDim=Dim[q];

        /* M_new = G * M_old */
        Are[q] = a_re*oAre - a_im*oAim + b_re*oCre - b_im*oCim;
        Aim[q] = a_re*oAim + a_im*oAre + b_re*oCim + b_im*oCre;
        Bre[q] = a_re*oBre - a_im*oBim + b_re*oDre - b_im*oDim;
        Bim[q] = a_re*oBim + a_im*oBre + b_re*oDim + b_im*oDre;
        Cre[q] = c_re*oAre - c_im*oAim + d_re*oCre - d_im*oCim;
        Cim[q] = c_re*oAim + c_im*oAre + d_re*oCim + d_im*oCre;
        Dre[q] = c_re*oBre - c_im*oBim + d_re*oDre - d_im*oDim;
        Dim[q] = c_re*oBim + c_im*oBre + d_re*oDim + d_im*oDre;

        used[q]=1;
    }

    /* Apply each qubit's composed matrix exactly once. */
    for (int q = 0; q < n_qubits; q++) {
        if (!used[q]) continue;
        long long bit = 1LL << q;
        long long lowmask = bit - 1;
        long long half = N >> 1;
        double a_re=Are[q], a_im=Aim[q], b_re=Bre[q], b_im=Bim[q];
        double c_re=Cre[q], c_im=Cim[q], d_re=Dre[q], d_im=Dim[q];
        double * restrict re = state_re;
        double * restrict im = state_im;

        #pragma omp parallel for schedule(static) if(half > 65536)
        for (long long k = 0; k < half; k++) {
            long long low = k & lowmask;
            long long high = k >> q;
            long long i = (high << (q+1)) | low;
            long long j = i | bit;
            double r0 = re[i], i0 = im[i], r1 = re[j], i1 = im[j];
            re[i] = a_re*r0 - a_im*i0 + b_re*r1 - b_im*i1;
            im[i] = a_re*i0 + a_im*r0 + b_re*i1 + b_im*r1;
            re[j] = c_re*r0 - c_im*i0 + d_re*r1 - d_im*i1;
            im[j] = c_re*i0 + c_im*r0 + d_re*i1 + d_im*r1;
        }
    }
}
```

PREDICTION: 12

## MEASUREMENT

No `quantum_bench`/`quantum_contract` tools were available in this session (the tool list was provided but explicitly disabled: "No tools are available in this session"), so I could not run the actual benchmark or contract check myself. The prediction above (~12× speedup vs. the naive gate-by-gate sweep) is a reasoning-based estimate for a benchmark with gates ≫ qubits (e.g. hundreds of gates over a few dozen qubits), driven by collapsing O(n_gates) full-state passes to O(n_qubits) full-state passes, further helped by the branch-free split loop and OpenMP. For adversarial inputs with n_gates ≈ n_qubits (e.g. one gate per qubit, no repeats), the expected speedup shrinks toward ~1–2× (just the branch-free loop and OpenMP gains, since fusion has nothing to compress). The actual measurement must come from the pipeline's run of `quantum_bench`/`quantum_contract` against this artifact.

## VERDICT

Correctness: the transformation is an exact algebraic identity (disjoint-qubit tensor factors commute), not an approximation, so results should match the reference within normal floating-point tolerance — modulo a caveat: composing many matrices before applying could plausibly accumulate rounding differently than the reference's gate-by-gate order, which is worth double-checking against the actual tolerance once measured, especially for long Rz-heavy chains on a single qubit. Performance: this should not be slower than the naive/fusion baseline in any case (worst case ≈ same work, plus cheap composition overhead) and should be substantially faster whenever the gate count exceeds the qubit count, which is the typical benchmark shape. I cannot confirm the numeric speedup or pass/fail against tolerance without the pipeline actually executing `quantum_bench`/`quantum_contract` — reporting the prediction as an estimate, not a measured result.