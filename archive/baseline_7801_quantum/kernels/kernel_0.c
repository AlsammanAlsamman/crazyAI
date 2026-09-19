#include <math.h>

static void apply_single(int n, int gtype, int gq, double gparam, double *re, double *im) {
    long long bit = 1LL << gq;
    long long block = bit << 1;
    long long N = 1LL << n;
    long long n_blocks = N / block;

    if (gtype == 0) {
        double s = 0.70710678118654752440;
        #pragma omp parallel for collapse(2) schedule(static) if(N > (1LL<<18))
        for (long long b = 0; b < n_blocks; b++) {
            for (long long off = 0; off < bit; off++) {
                long long i = b*block + off, j = i + bit;
                double r0=re[i], i0=im[i], r1=re[j], i1=im[j];
                re[i]=(r0+r1)*s; im[i]=(i0+i1)*s;
                re[j]=(r0-r1)*s; im[j]=(i0-i1)*s;
            }
        }
    } else if (gtype == 1) {
        #pragma omp parallel for collapse(2) schedule(static) if(N > (1LL<<18))
        for (long long b = 0; b < n_blocks; b++) {
            for (long long off = 0; off < bit; off++) {
                long long i = b*block + off, j = i + bit;
                double r0=re[i], i0=im[i], r1=re[j], i1=im[j];
                re[i]=r1; im[i]=i1; re[j]=r0; im[j]=i0;
            }
        }
    } else if (gtype == 2) {
        #pragma omp parallel for collapse(2) schedule(static) if(N > (1LL<<18))
        for (long long b = 0; b < n_blocks; b++) {
            for (long long off = 0; off < bit; off++) {
                long long j = b*block + off + bit;
                re[j] = -re[j]; im[j] = -im[j];
            }
        }
    } else {
        double phi0 = -gparam/2, phi1 = gparam/2;
        double c0=cos(phi0), s0=sin(phi0), c1=cos(phi1), s1=sin(phi1);
        #pragma omp parallel for collapse(2) schedule(static) if(N > (1LL<<18))
        for (long long b = 0; b < n_blocks; b++) {
            for (long long off = 0; off < bit; off++) {
                long long i = b*block + off, j = i + bit;
                double r0=re[i], i0=im[i], r1=re[j], i1=im[j];
                re[i]=r0*c0-i0*s0; im[i]=r0*s0+i0*c0;
                re[j]=r1*c1-i1*s1; im[j]=r1*s1+i1*c1;
            }
        }
    }
}

static void apply_matrix(int n, int gq,
        double m00r,double m00i,double m01r,double m01i,
        double m10r,double m10i,double m11r,double m11i,
        double *re, double *im) {
    long long bit = 1LL << gq;
    long long block = bit << 1;
    long long N = 1LL << n;
    long long n_blocks = N / block;
    #pragma omp parallel for collapse(2) schedule(static) if(N > (1LL<<18))
    for (long long b = 0; b < n_blocks; b++) {
        for (long long off = 0; off < bit; off++) {
            long long i = b*block + off, j = i + bit;
            double r0=re[i], i0=im[i], r1=re[j], i1=im[j];
            re[i] = m00r*r0 - m00i*i0 + m01r*r1 - m01i*i1;
            im[i] = m00r*i0 + m00i*r0 + m01r*i1 + m01i*r1;
            re[j] = m10r*r0 - m10i*i0 + m11r*r1 - m11i*i1;
            im[j] = m10r*i0 + m10i*r0 + m11r*i1 + m11i*r1;
        }
    }
}

/* result (M) := G * M  (complex 2x2 matrix product) */
static void mat_mul(
    double g00r,double g00i,double g01r,double g01i,
    double g10r,double g10i,double g11r,double g11i,
    double *m00r,double *m00i,double *m01r,double *m01i,
    double *m10r,double *m10i,double *m11r,double *m11i) {
    double a00r=*m00r,a00i=*m00i,a01r=*m01r,a01i=*m01i;
    double a10r=*m10r,a10i=*m10i,a11r=*m11r,a11i=*m11i;
    double n00r = g00r*a00r-g00i*a00i + g01r*a10r-g01i*a10i;
    double n00i = g00r*a00i+g00i*a00r + g01r*a10i+g01i*a10r;
    double n01r = g00r*a01r-g00i*a01i + g01r*a11r-g01i*a11i;
    double n01i = g00r*a01i+g00i*a01r + g01r*a11i+g01i*a11r;
    double n10r = g10r*a00r-g10i*a00i + g11r*a10r-g11i*a10i;
    double n10i = g10r*a00i+g10i*a00r + g11r*a10i+g11i*a10r;
    double n11r = g10r*a01r-g10i*a01i + g11r*a11r-g11i*a11i;
    double n11i = g10r*a01i+g10i*a01r + g11r*a11i+g11i*a11r;
    *m00r=n00r; *m00i=n00i; *m01r=n01r; *m01i=n01i;
    *m10r=n10r; *m10i=n10i; *m11r=n11r; *m11i=n11i;
}

static void gate_matrix(int gtype, double gparam,
    double *g00r,double *g00i,double *g01r,double *g01i,
    double *g10r,double *g10i,double *g11r,double *g11i) {
    if (gtype==0) {
        double s=0.70710678118654752440;
        *g00r=s; *g00i=0; *g01r=s; *g01i=0;
        *g10r=s; *g10i=0; *g11r=-s; *g11i=0;
    } else if (gtype==1) {
        *g00r=0; *g00i=0; *g01r=1; *g01i=0;
        *g10r=1; *g10i=0; *g11r=0; *g11i=0;
    } else if (gtype==2) {
        *g00r=1; *g00i=0; *g01r=0; *g01i=0;
        *g10r=0; *g10i=0; *g11r=-1; *g11i=0;
    } else {
        double phi0=-gparam/2, phi1=gparam/2;
        *g00r=cos(phi0); *g00i=sin(phi0); *g01r=0; *g01i=0;
        *g10r=0; *g10i=0; *g11r=cos(phi1); *g11i=sin(phi1);
    }
}

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
           double *state_re, double *state_im) {
    int g = 0;
    while (g < n_gates) {
        int q = gate_qubit[g];
        int run_start = g;
        int run_len = 1;
        g++;
        while (g < n_gates && gate_qubit[g] == q) { run_len++; g++; }

        if (run_len == 1) {
            apply_single(n_qubits, gate_type[run_start], q, gate_param[run_start], state_re, state_im);
        } else {
            double m00r=1,m00i=0,m01r=0,m01i=0,m10r=0,m10i=0,m11r=1,m11i=0;
            for (int k = run_start; k < run_start+run_len; k++) {
                double g00r,g00i,g01r,g01i,g10r,g10i,g11r,g11i;
                gate_matrix(gate_type[k], gate_param[k], &g00r,&g00i,&g01r,&g01i,&g10r,&g10i,&g11r,&g11i);
                mat_mul(g00r,g00i,g01r,g01i,g10r,g10i,g11r,g11i, &m00r,&m00i,&m01r,&m01i,&m10r,&m10i,&m11r,&m11i);
            }
            apply_matrix(n_qubits, q, m00r,m00i,m01r,m01i,m10r,m10i,m11r,m11i, state_re, state_im);
        }
    }
}
