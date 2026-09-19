#include <math.h>
#include <stdlib.h>

static inline void cmul(double ar,double ai,double br,double bi,double *cr,double *ci){
    *cr = ar*br - ai*bi;
    *ci = ar*bi + ai*br;
}

/* M := G * M  (2x2 complex matrices, row-major: idx 0=00,1=01,2=10,3=11) */
static inline void accumulate(double *Mre, double *Mim, const double *Gre, const double *Gim){
    double t1r,t1i,t2r,t2i;
    double r00,i00,r01,i01,r10,i10,r11,i11;

    cmul(Gre[0],Gim[0],Mre[0],Mim[0], &t1r,&t1i);
    cmul(Gre[1],Gim[1],Mre[2],Mim[2], &t2r,&t2i);
    r00=t1r+t2r; i00=t1i+t2i;

    cmul(Gre[0],Gim[0],Mre[1],Mim[1], &t1r,&t1i);
    cmul(Gre[1],Gim[1],Mre[3],Mim[3], &t2r,&t2i);
    r01=t1r+t2r; i01=t1i+t2i;

    cmul(Gre[2],Gim[2],Mre[0],Mim[0], &t1r,&t1i);
    cmul(Gre[3],Gim[3],Mre[2],Mim[2], &t2r,&t2i);
    r10=t1r+t2r; i10=t1i+t2i;

    cmul(Gre[2],Gim[2],Mre[1],Mim[1], &t1r,&t1i);
    cmul(Gre[3],Gim[3],Mre[3],Mim[3], &t2r,&t2i);
    r11=t1r+t2r; i11=t1i+t2i;

    Mre[0]=r00; Mim[0]=i00; Mre[1]=r01; Mim[1]=i01;
    Mre[2]=r10; Mim[2]=i10; Mre[3]=r11; Mim[3]=i11;
}

static inline int is_identity(const double *Mre, const double *Mim){
    const double eps = 1e-9;
    return fabs(Mre[0]-1.0)<eps && fabs(Mim[0])<eps &&
           fabs(Mre[1])<eps     && fabs(Mim[1])<eps &&
           fabs(Mre[2])<eps     && fabs(Mim[2])<eps &&
           fabs(Mre[3]-1.0)<eps && fabs(Mim[3])<eps;
}

static void apply_qubit_matrix(int n, int q, const double *Mre, const double *Mim,
                                double *re, double *im){
    long long bit  = 1LL << q;
    long long N    = 1LL << n;
    long long half = N >> 1;
    long long low_mask = bit - 1;

    double a_re=Mre[0], a_im=Mim[0], b_re=Mre[1], b_im=Mim[1];
    double c_re=Mre[2], c_im=Mim[2], d_re=Mre[3], d_im=Mim[3];

    #pragma omp parallel for schedule(static) if(half >= 65536)
    for (long long k = 0; k < half; k++) {
        long long low  = k & low_mask;
        long long high = (k & ~low_mask) << 1;
        long long i = high | low;
        long long j = i | bit;

        double r0=re[i], i0=im[i], r1=re[j], i1=im[j];

        double nri = a_re*r0 - a_im*i0 + b_re*r1 - b_im*i1;
        double nii = a_re*i0 + a_im*r0 + b_re*i1 + b_im*r1;
        double nrj = c_re*r0 - c_im*i0 + d_re*r1 - d_im*i1;
        double nij = c_re*i0 + c_im*r0 + d_re*i1 + d_im*r1;

        re[i]=nri; im[i]=nii; re[j]=nrj; im[j]=nij;
    }
}

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
            double *state_re, double *state_im) {
    if (n_qubits <= 0) return;

    double *Mre = (double*) malloc(sizeof(double) * 4 * n_qubits);
    double *Mim = (double*) malloc(sizeof(double) * 4 * n_qubits);

    for (int q = 0; q < n_qubits; q++) {
        double *r = &Mre[4*q], *m = &Mim[4*q];
        r[0]=1.0; r[1]=0.0; r[2]=0.0; r[3]=1.0;
        m[0]=0.0; m[1]=0.0; m[2]=0.0; m[3]=0.0;
    }

    const double s = 0.70710678118654752440;

    /* SEED-2 pass: walk the row of lanterns (gates), and for each one hold
       its matrix against the shadow flame already accumulated on its own
       wick-fork (qubit). Matched -> fuse (multiply in). This never touches
       the state array. */
    for (int g = 0; g < n_gates; g++) {
        int q = gate_qubit[g];
        int gt = gate_type[g];
        double Gre[4], Gim[4];

        if (gt == 0) {          /* Hadamard */
            Gre[0]=s;  Gre[1]=s;  Gre[2]=s;  Gre[3]=-s;
            Gim[0]=0;  Gim[1]=0;  Gim[2]=0;  Gim[3]=0;
        } else if (gt == 1) {   /* Pauli-X */
            Gre[0]=0;  Gre[1]=1;  Gre[2]=1;  Gre[3]=0;
            Gim[0]=0;  Gim[1]=0;  Gim[2]=0;  Gim[3]=0;
        } else if (gt == 2) {   /* Pauli-Z */
            Gre[0]=1;  Gre[1]=0;  Gre[2]=0;  Gre[3]=-1;
            Gim[0]=0;  Gim[1]=0;  Gim[2]=0;  Gim[3]=0;
        } else {                /* Rz(theta) */
            double theta = gate_param[g];
            double ch = cos(theta/2.0), sh = sin(theta/2.0);
            Gre[0]=ch;  Gre[1]=0;  Gre[2]=0;  Gre[3]=ch;
            Gim[0]=-sh; Gim[1]=0;  Gim[2]=0;  Gim[3]=sh;
        }

        accumulate(&Mre[4*q], &Mim[4*q], Gre, Gim);
    }

    /* SEED-3: wick-forks whose coin never warms (net identity) are
       discarded for good -- no sweep. Everything else gets exactly one
       visit, dish and all. */
    for (int q = 0; q < n_qubits; q++) {
        if (!is_identity(&Mre[4*q], &Mim[4*q])) {
            apply_qubit_matrix(n_qubits, q, &Mre[4*q], &Mim[4*q], state_re, state_im);
        }
    }

    free(Mre);
    free(Mim);
}
