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
