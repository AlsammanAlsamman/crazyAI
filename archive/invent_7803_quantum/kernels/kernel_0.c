#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* a lantern-column's fused strength: 2x2 complex matrix, rows (r00,r01;r10,r11) */
typedef struct { double r00,i00,r01,i01,r10,i10,r11,i11; } cmat2;

static inline cmat2 mat_identity(void) {
    cmat2 m = {1,0, 0,0, 0,0, 1,0};
    return m;
}

static inline cmat2 gate_matrix(int gtype, double gparam) {
    cmat2 m;
    if (gtype == 0) { /* Hadamard */
        double s = 0.70710678118654752440;
        m.r00=s; m.i00=0; m.r01=s; m.i01=0; m.r10=s; m.i10=0; m.r11=-s; m.i11=0;
    } else if (gtype == 1) { /* Pauli-X */
        m.r00=0; m.i00=0; m.r01=1; m.i01=0; m.r10=1; m.i10=0; m.r11=0; m.i11=0;
    } else if (gtype == 2) { /* Pauli-Z */
        m.r00=1; m.i00=0; m.r01=0; m.i01=0; m.r10=0; m.i10=0; m.r11=-1; m.i11=0;
    } else { /* Rz(theta) */
        double phi0 = -gparam*0.5, phi1 = gparam*0.5;
        m.r00=cos(phi0); m.i00=sin(phi0);
        m.r01=0; m.i01=0; m.r10=0; m.i10=0;
        m.r11=cos(phi1); m.i11=sin(phi1);
    }
    return m;
}

/* R = G * M  (fold gate G onto the qubit's running accumulator M, G applied last) */
static inline cmat2 mat_mul_left(cmat2 g, cmat2 m) {
    cmat2 r;
    r.r00 = g.r00*m.r00 - g.i00*m.i00 + g.r01*m.r10 - g.i01*m.i10;
    r.i00 = g.r00*m.i00 + g.i00*m.r00 + g.r01*m.i10 + g.i01*m.r10;
    r.r01 = g.r00*m.r01 - g.i00*m.i01 + g.r01*m.r11 - g.i01*m.i11;
    r.i01 = g.r00*m.i01 + g.i00*m.r01 + g.r01*m.i11 + g.i01*m.r11;
    r.r10 = g.r10*m.r00 - g.i10*m.i00 + g.r11*m.r10 - g.i11*m.i10;
    r.i10 = g.r10*m.i00 + g.i10*m.r00 + g.r11*m.i10 + g.i11*m.r10;
    r.r11 = g.r10*m.r01 - g.i10*m.i01 + g.r11*m.r11 - g.i11*m.i11;
    r.i11 = g.r10*m.i01 + g.i10*m.r01 + g.r11*m.i11 + g.i11*m.r11;
    return r;
}

/* Gram-Schmidt re-orthonormalize the two rows (Hermitian inner product) so the
   fused matrix stays unitary - the "wait for the draft to steady, a stray
   gust tips the pan into a false balance" safeguard against drift when many
   gates are folded into one accumulator. */
static inline cmat2 mat_renorm(cmat2 m) {
    double n0 = sqrt(m.r00*m.r00 + m.i00*m.i00 + m.r01*m.r01 + m.i01*m.i01);
    if (n0 > 1e-300) { m.r00/=n0; m.i00/=n0; m.r01/=n0; m.i01/=n0; }
    double pr = m.r10*m.r00 + m.i10*m.i00 + m.r11*m.r01 + m.i11*m.i01;
    double pi = m.i10*m.r00 - m.r10*m.i00 + m.i11*m.r01 - m.r11*m.i01;
    m.r10 -= pr*m.r00 - pi*m.i00; m.i10 -= pr*m.i00 + pi*m.r00;
    m.r11 -= pr*m.r01 - pi*m.i01; m.i11 -= pr*m.i01 + pi*m.r01;
    double n1 = sqrt(m.r10*m.r10 + m.i10*m.i10 + m.r11*m.r11 + m.i11*m.i11);
    if (n1 > 1e-300) { m.r10/=n1; m.i10/=n1; m.r11/=n1; m.i11/=n1; }
    return m;
}

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
            double *state_re, double *state_im) {
    if (n_qubits <= 0) return;

    /* Walk the whole row once with an ear to the glass: fold every gate into
       its target qubit's running matrix, keyed by pitch (qubit id), not by
       neighbourhood in the gate sequence. */
    cmat2 *acc = (cmat2*)malloc(sizeof(cmat2) * (size_t)n_qubits);
    char  *touched = (char*)calloc((size_t)n_qubits, sizeof(char));
    for (int q = 0; q < n_qubits; q++) acc[q] = mat_identity();

    for (int g = 0; g < n_gates; g++) {
        int q = gate_qubit[g];
        cmat2 gm = gate_matrix(gate_type[g], gate_param[g]);
        acc[q] = mat_mul_left(gm, acc[q]);
        touched[q] = 1;
        if ((g % 3) == 2) acc[q] = mat_renorm(acc[q]); /* every third pair, let it steady */
    }
    for (int q = 0; q < n_qubits; q++) if (touched[q]) acc[q] = mat_renorm(acc[q]); /* final check, end to end */

    /* One sweep per touched qubit only - the row (state array) itself never
       moves, only the measured "dust" (matrix) travelled until now. */
    long long N = 1LL << n_qubits;
    for (int q = 0; q < n_qubits; q++) {
        if (!touched[q]) continue;
        cmat2 m = acc[q];
        long long bit = 1LL << q;
        #pragma omp parallel for schedule(static) if(N > 4096)
        for (long long i = 0; i < N; i++) {
            if (i & bit) continue;
            long long j = i | bit;
            double r0 = state_re[i], i0 = state_im[i];
            double r1 = state_re[j], i1 = state_im[j];
            state_re[i] = m.r00*r0 - m.i00*i0 + m.r01*r1 - m.i01*i1;
            state_im[i] = m.r00*i0 + m.i00*r0 + m.r01*i1 + m.i01*r1;
            state_re[j] = m.r10*r0 - m.i10*i0 + m.r11*r1 - m.i11*i1;
            state_im[j] = m.r10*i0 + m.i10*r0 + m.r11*i1 + m.i11*r1;
        }
    }
    free(acc);
    free(touched);
}
