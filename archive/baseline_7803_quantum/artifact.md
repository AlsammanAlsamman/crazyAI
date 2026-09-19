# APPROACH

The reference "known way" only fuses *consecutive* same-qubit gates. But single-qubit gates acting on *different* qubits always commute (they act on disjoint tensor factors: `(A⊗I)(I⊗B) = (I⊗B)(A⊗I)`), regardless of gate type or intervening gates on other qubits. This means the entire gate sequence can be re-grouped by target qubit — not just consecutive runs — and each qubit's full sequence of gates (H/X/Z/Rz, in their original relative order) collapsed into **one** 2×2 complex matrix via matrix multiplication.

This turns the sweep cost from **O(n_gates · 2^n)** into **O(n_qubits · 2^n)** — at most one full-state pass per qubit that was ever touched, independent of how many gates were applied to it. The fusion bookkeeping itself is O(n_gates) with tiny constant work (2×2 complex matrix multiplies), negligible next to the O(2^n) sweeps.

On top of the algorithmic win:
- Each per-qubit sweep uses the block/offset indexing (`base`, `off`) so that for the common case (many blocks) the outer loop is OpenMP-parallel and the inner loop is `omp simd`-vectorizable over contiguous memory; for the opposite case (few, large blocks — high qubit index) parallelism is applied on the inner loop instead, so all qubit positions get good parallel/SIMD utilization.
- Unused qubits are skipped entirely (no sweep at all).
- Small-N cases skip OpenMP via an `if()` clause to avoid thread-spawn overhead.

# ARTIFACT

```c
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline void cmul(double are,double aim,double bre,double bim,double *rre,double *rim){
    *rre = are*bre - aim*bim;
    *rim = are*bim + aim*bre;
}

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
           double *state_re, double *state_im) {
    if (n_qubits <= 0) return;
    long long N = 1LL << n_qubits;

    double *M = (double*)malloc(sizeof(double)*8*(size_t)n_qubits);
    unsigned char *used = (unsigned char*)calloc((size_t)n_qubits,1);
    if (!M || !used) { free(M); free(used); return; }

    for (int q=0;q<n_qubits;q++){
        double *m = M+8*q;
        m[0]=1;m[1]=0;m[2]=0;m[3]=0;
        m[4]=0;m[5]=0;m[6]=1;m[7]=0;
    }

    /* Fuse all gates per qubit into one 2x2 complex matrix, in temporal order.
       Gates on different qubits commute, so cross-qubit ordering does not matter. */
    for (int g=0; g<n_gates; g++){
        int t = gate_type[g];
        int q = gate_qubit[g];
        double g00re,g00im,g01re,g01im,g10re,g10im,g11re,g11im;
        if (t==0){
            double s=0.70710678118654752440;
            g00re=s;g00im=0.0; g01re=s;g01im=0.0;
            g10re=s;g10im=0.0; g11re=-s;g11im=0.0;
        } else if (t==1){
            g00re=0.0;g00im=0.0; g01re=1.0;g01im=0.0;
            g10re=1.0;g10im=0.0; g11re=0.0;g11im=0.0;
        } else if (t==2){
            g00re=1.0;g00im=0.0; g01re=0.0;g01im=0.0;
            g10re=0.0;g10im=0.0; g11re=-1.0;g11im=0.0;
        } else {
            double th=gate_param[g];
            double c=cos(th*0.5), s=sin(th*0.5);
            g00re=c; g00im=-s; g01re=0.0; g01im=0.0;
            g10re=0.0; g10im=0.0; g11re=c; g11im=s;
        }
        double *m = M+8*q;
        double m00re=m[0],m00im=m[1],m01re=m[2],m01im=m[3];
        double m10re=m[4],m10im=m[5],m11re=m[6],m11im=m[7];
        double t1re,t1im,t2re,t2im;
        double n00re,n00im,n01re,n01im,n10re,n10im,n11re,n11im;

        cmul(g00re,g00im,m00re,m00im,&t1re,&t1im);
        cmul(g01re,g01im,m10re,m10im,&t2re,&t2im);
        n00re=t1re+t2re; n00im=t1im+t2im;

        cmul(g00re,g00im,m01re,m01im,&t1re,&t1im);
        cmul(g01re,g01im,m11re,m11im,&t2re,&t2im);
        n01re=t1re+t2re; n01im=t1im+t2im;

        cmul(g10re,g10im,m00re,m00im,&t1re,&t1im);
        cmul(g11re,g11im,m10re,m10im,&t2re,&t2im);
        n10re=t1re+t2re; n10im=t1im+t2im;

        cmul(g10re,g10im,m01re,m01im,&t1re,&t1im);
        cmul(g11re,g11im,m11re,m11im,&t2re,&t2im);
        n11re=t1re+t2re; n11im=t1im+t2im;

        m[0]=n00re;m[1]=n00im;m[2]=n01re;m[3]=n01im;
        m[4]=n10re;m[5]=n10im;m[6]=n11re;m[7]=n11im;
        used[q]=1;
    }

    double * __restrict re = state_re;
    double * __restrict im = state_im;

    for (int q=0;q<n_qubits;q++){
        if (!used[q]) continue;
        double *m = M+8*q;
        const double a00re=m[0],a00im=m[1],a01re=m[2],a01im=m[3];
        const double a10re=m[4],a10im=m[5],a11re=m[6],a11im=m[7];
        long long bit = 1LL << q;
        long long step = bit<<1;
        long long nblocks = N/step;

        if (nblocks >= 8) {
            #pragma omp parallel for schedule(static) if(N > (1LL<<16))
            for (long long base = 0; base < N; base += step){
                long long i0 = base, j0 = base+bit;
                #pragma omp simd
                for (long long off=0; off<bit; off++){
                    long long i = i0+off, j = j0+off;
                    double r0=re[i], iv0=im[i], r1=re[j], iv1=im[j];
                    double nr0 = a00re*r0 - a00im*iv0 + a01re*r1 - a01im*iv1;
                    double ni0 = a00re*iv0 + a00im*r0 + a01re*iv1 + a01im*r1;
                    double nr1 = a10re*r0 - a10im*iv0 + a11re*r1 - a11im*iv1;
                    double ni1 = a10re*iv0 + a10im*r0 + a11re*iv1 + a11im*r1;
                    re[i]=nr0; im[i]=ni0; re[j]=nr1; im[j]=ni1;
                }
            }
        } else {
            for (long long base=0; base<N; base+=step){
                long long i0 = base, j0 = base+bit;
                #pragma omp parallel for simd schedule(static) if(N > (1LL<<16))
                for (long long off=0; off<bit; off++){
                    long long i = i0+off, j = j0+off;
                    double r0=re[i], iv0=im[i], r1=re[j], iv1=im[j];
                    double nr0 = a00re*r0 - a00im*iv0 + a01re*r1 - a01im*iv1;
                    double ni0 = a00re*iv0 + a00im*r0 + a01re*iv1 + a01im*r1;
                    double nr1 = a10re*r0 - a10im*iv0 + a11re*r1 - a11im*iv1;
                    double ni1 = a10re*iv0 + a10im*r0 + a11re*iv1 + a11im*r1;
                    re[i]=nr0; im[i]=ni0; re[j]=nr1; im[j]=ni1;
                }
            }
        }
    }

    free(M);
    free(used);
}
```

PREDICTION: 10

# MEASUREMENT

No tools (including `quantum_bench`) were available in this session to execute — I could not run the benchmark myself. The number above is a pre-measurement estimate only, based on algorithmic analysis: for typical benchmark workloads where `n_gates` substantially exceeds `n_qubits` (e.g. random circuits with many gates per qubit), replacing O(n_gates) full-state sweeps with at most O(n_qubits) sweeps should dominate runtime; a 10x figure is a conservative middle estimate — the actual speedup could be much larger for deep circuits (ratio ≈ n_gates/n_qubits) or much smaller (close to 1x, just SIMD/OpenMP gains) for shallow circuits where each qubit is touched only once or twice.

# VERDICT

Correctness: the fusion relies on the standard, exact fact that single-qubit gates on different qubits commute as tensor-product operators, and same-qubit gates are composed via ordinary 2×2 complex matrix multiplication in their original temporal order — this is mathematically exact (not an approximation), so results should match the reference to floating-point rounding only, likely *tighter* than the naive sequential approach since fewer passes over the data mean less accumulated rounding.

Performance: cannot be confirmed numerically in this session since no benchmarking tool was reachable; the report above is an honest analytical prediction, not a measured result. If the harness runs this code, I'd expect a clear win on gate-dense circuits and a wash-to-modest win on shallow ones, with no case where it should be slower than the naive baseline (fusion overhead is O(n_gates) scalar work, dwarfed by any O(2^n) sweep).