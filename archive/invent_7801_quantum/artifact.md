# MAPPING

**SEED 1 — tide-bell / rope bridge**

| World object | Problem object |
|---|---|
| the row of lanterns | the statevector `state_re/state_im`, indexed 0..2^n-1 |
| a "district" whose rope bridge locks level | a target qubit's whole set of gates in the circuit |
| the tide-bell ringing | the moment all of a qubit's gates have been seen and its net effect is known |
| "flames hold still enough to be read" | the state array is only touched once that net effect is fixed, not after every gate |

Breaks: *"a gate must be fully applied to the whole state before the next gate starts."*

**SEED 2 — the Arbiter's rod**

| World object | Problem object |
|---|---|
| the rod | the bit-test/index-generation logic that finds a pair (i, i\|bit) |
| "warms only over the two it has chosen" | only genuine (i,j) pairs differing in the target bit are touched, everything else skipped |
| reaching for an unchosen pair, "closing on empty air" | indexing bugs / redundant double-visits the reference code already avoids with `if (i & bit) continue` |

Breaks: *"every amplitude must be visited in the same order for every gate"* (rod lets you generate pairs by a flat index `p`, not sequential `i`). This mapping is close to what the naive kernel already does — least novel of the three.

**SEED 3 — mirror-caps, cancellation vs. debt**

| World object | Problem object |
|---|---|
| a lantern | one qubit's *accumulated* single-qubit unitary (not a state amplitude) |
| bright wick / shadow wick | the real / imaginary parts of that accumulated 2×2 matrix's entries |
| the rod pairing two lanterns and warming | fusing two consecutive gates *on the same qubit* into one composite matrix (2×2 · 2×2) |
| "same-leaning wicks feed the receiving flame taller, giving flame gutters, debt grows" | ordinary matrix composition: the fused matrix is non-trivial, must still be applied to the state |
| "bright matches shadow, equal height, mirror-caps flash white, both wicks die, never relit" | the fused matrix reduces exactly to the identity (e.g. X·X, Z·Z, H·H, Rz(θ)·Rz(-θ)) — that qubit needs **zero** passes over the state, permanently, for the gates already folded in |
| carrying the dead wicks to the cellar, "paid is paid" | once folded into an identity matrix, those gates never touch memory again |

Breaks: *"the positive and negative parts of an amplitude are separate quantities, never merged"* — here it's stronger: whole **gates** (not just re/im pieces) are algebraically merged/cancelled *before* any amplitude is ever touched, and — because single-qubit gates on different qubits always commute — this fusion can happen across the *entire* gate sequence, not just consecutive same-qubit runs.

# CHOSEN SEED

SEED 3. It is the most literal (bright/shadow = real/imag of a 2×2 matrix, mirror-caps = matrix multiplication, flash-white = exact identity) and the most different from the "known way": the known way only fuses *consecutive* same-qubit gates; SEED 3's "the rod decides, I do nothing before, everything within" licenses fusing **all** gates on a qubit across the whole circuit (legal because disjoint-qubit gates commute exactly), collapsing the number of full-state sweeps from `n_gates` down to at most `n_qubits`.

# ASSUMPTION BROKEN

"A gate must be fully applied to the whole state before the next gate starts" — replaced by: fold every gate into its target qubit's running 2×2 matrix first (touches no memory), then sweep the state at most once per qubit, skipping qubits whose net matrix is the identity.

# ARTIFACT

```c
#include <math.h>
#include <stdlib.h>

typedef struct { double re, im; } cplx;
typedef struct { cplx a, b, c, d; } mat2;

static inline cplx cmul(cplx x, cplx y) {
    cplx r; r.re = x.re*y.re - x.im*y.im; r.im = x.re*y.im + x.im*y.re; return r;
}
static inline cplx cadd(cplx x, cplx y) { cplx r; r.re = x.re+y.re; r.im = x.im+y.im; return r; }

static mat2 gate_matrix(int gtype, double param) {
    mat2 m = {{0,0},{0,0},{0,0},{0,0}};
    if (gtype == 0) {              /* Hadamard */
        double s = 0.70710678118654752440;
        m.a.re = s; m.b.re = s; m.c.re = s; m.d.re = -s;
    } else if (gtype == 1) {       /* Pauli-X */
        m.b.re = 1; m.c.re = 1;
    } else if (gtype == 2) {       /* Pauli-Z */
        m.a.re = 1; m.d.re = -1;
    } else {                       /* Rz(param) */
        double phi0 = -param/2.0, phi1 = param/2.0;
        m.a.re = cos(phi0); m.a.im = sin(phi0);
        m.d.re = cos(phi1); m.d.im = sin(phi1);
    }
    return m;
}

/* result = G * old  (apply old first, then G) */
static inline mat2 mat2_mul(mat2 G, mat2 old) {
    mat2 r;
    r.a = cadd(cmul(G.a, old.a), cmul(G.b, old.c));
    r.b = cadd(cmul(G.a, old.b), cmul(G.b, old.d));
    r.c = cadd(cmul(G.c, old.a), cmul(G.d, old.c));
    r.d = cadd(cmul(G.c, old.b), cmul(G.d, old.d));
    return r;
}

static void apply_matrix(int n, int q, mat2 m, double *re, double *im) {
    long long bit  = 1LL << q;
    long long N    = 1LL << n;
    long long half = N >> 1;
    long long low_mask = bit - 1;
    double ar=m.a.re, ai=m.a.im, br=m.b.re, bi=m.b.im;
    double cr=m.c.re, ci=m.c.im, dr=m.d.re, di=m.d.im;
    #pragma omp parallel for simd schedule(static) if(half > 4096)
    for (long long p = 0; p < half; p++) {
        long long low  = p & low_mask;
        long long high = p >> q;
        long long i = (high << (q + 1)) | low;
        long long j = i | bit;
        double r0 = re[i], i0 = im[i], r1 = re[j], i1 = im[j];
        re[i] = ar*r0 - ai*i0 + br*r1 - bi*i1;
        im[i] = ar*i0 + ai*r0 + br*i1 + bi*r1;
        re[j] = cr*r0 - ci*i0 + dr*r1 - di*i1;
        im[j] = cr*i0 + ci*r0 + dr*i1 + di*r1;
    }
}

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
            double *state_re, double *state_im) {
    mat2 *M = (mat2 *)malloc(sizeof(mat2) * (size_t)n_qubits);
    for (int q = 0; q < n_qubits; q++) {
        mat2 id = {{1,0},{0,0},{0,0},{1,0}};
        M[q] = id;
    }
    /* Fold every gate into its qubit's running 2x2 matrix. Single-qubit
       gates on different qubits commute exactly, so interleaving across
       qubits in the original list is irrelevant to the final state -
       only per-qubit order matters, and that is preserved here. */
    for (int g = 0; g < n_gates; g++) {
        int q = gate_qubit[g];
        mat2 G = gate_matrix(gate_type[g], gate_param[g]);
        M[q] = mat2_mul(G, M[q]);
    }
    const double eps = 1e-12;
    for (int q = 0; q < n_qubits; q++) {
        mat2 m = M[q];
        int is_identity =
            fabs(m.a.re - 1.0) < eps && fabs(m.a.im) < eps &&
            fabs(m.d.re - 1.0) < eps && fabs(m.d.im) < eps &&
            fabs(m.b.re) < eps && fabs(m.b.im) < eps &&
            fabs(m.c.re) < eps && fabs(m.c.im) < eps;
        if (!is_identity) apply_matrix(n_qubits, q, m, state_re, state_im);
    }
    free(M);
}
```

PREDICTION: speedup_vs_naive = 60

# MEASUREMENT

No tools were available in this session (`quantum_bench`/`quantum_contract` could not be invoked here), so I cannot report an actual measured number. Reporting that plainly rather than inventing one: the 60× above is a derivation, not a measurement — it assumes the benchmark uses on the order of tens of gates per qubit (fusion turns `n_gates` full-state sweeps into ≤`n_qubits` sweeps, at cost only O(n_gates) of cheap 2×2 matrix multiplies that never touch the 2^n-sized arrays), plus modest OpenMP/SIMD gain on the sweeps that remain. The external pipeline that compiles and runs this kernel should supply the real number.

# VERDICT

Correctness argument: since single-qubit gates acting on different qubits commute exactly as tensor factors, folding all gates targeting a qubit (in their original relative order) into one running 2×2 matrix and applying it once — skipping entirely when that matrix is the identity — produces the same final state as the literal reference gate-by-gate loop, within floating-point tolerance. This is a legitimate, literal translation of the "mirror-caps merge or cancel, dead wicks never relight" seed, and it is a genuinely different algorithm from the known way (which only fuses *consecutive* same-qubit gates) rather than a relabeling of it. Whether the predicted ~60× actually holds is unverified in this session and must be checked by the pipeline; if the benchmark uses very few gates per qubit (ratio near 1), the honest expectation is a much smaller speedup, dominated by the identity-skip and light parallel/SIMD gains alone.