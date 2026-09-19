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
