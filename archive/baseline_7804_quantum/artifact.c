#include <math.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline void cmul(double xr, double xi, double yr, double yi, double *or_, double *oi_) {
    *or_ = xr * yr - xi * yi;
    *oi_ = xr * yi + xi * yr;
}

static void apply_matrix(int n_qubits, long long bit,
                          double ar, double ai, double br, double bi,
                          double cr, double ci, double dr, double di,
                          double *re, double *im) {
    long long N = 1LL << n_qubits;
    long long step = bit << 1;
    long long nblocks = N / step;

    if (bit >= 4) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= (1LL<<20))
        #endif
        for (long long b = 0; b < nblocks; b++) {
            long long i = b * step;
            long long j = i + bit;
            long long k = 0;
            __m256d arv = _mm256_set1_pd(ar), aiv = _mm256_set1_pd(ai);
            __m256d brv = _mm256_set1_pd(br), biv = _mm256_set1_pd(bi);
            __m256d crv = _mm256_set1_pd(cr), civ = _mm256_set1_pd(ci);
            __m256d drv = _mm256_set1_pd(dr), div = _mm256_set1_pd(di);
            for (; k + 4 <= bit; k += 4) {
                __m256d r0 = _mm256_loadu_pd(&re[i + k]);
                __m256d x0 = _mm256_loadu_pd(&im[i + k]);
                __m256d r1 = _mm256_loadu_pd(&re[j + k]);
                __m256d x1 = _mm256_loadu_pd(&im[j + k]);

                __m256d nri = _mm256_add_pd(
                    _mm256_sub_pd(_mm256_mul_pd(arv, r0), _mm256_mul_pd(aiv, x0)),
                    _mm256_sub_pd(_mm256_mul_pd(brv, r1), _mm256_mul_pd(biv, x1)));
                __m256d nxi = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(arv, x0), _mm256_mul_pd(aiv, r0)),
                    _mm256_add_pd(_mm256_mul_pd(brv, x1), _mm256_mul_pd(biv, r1)));
                __m256d nrj = _mm256_add_pd(
                    _mm256_sub_pd(_mm256_mul_pd(crv, r0), _mm256_mul_pd(civ, x0)),
                    _mm256_sub_pd(_mm256_mul_pd(drv, r1), _mm256_mul_pd(div, x1)));
                __m256d nxj = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(crv, x0), _mm256_mul_pd(civ, r0)),
                    _mm256_add_pd(_mm256_mul_pd(drv, x1), _mm256_mul_pd(div, r1)));

                _mm256_storeu_pd(&re[i + k], nri);
                _mm256_storeu_pd(&im[i + k], nxi);
                _mm256_storeu_pd(&re[j + k], nrj);
                _mm256_storeu_pd(&im[j + k], nxj);
            }
            for (; k < bit; k++) {
                double r0 = re[i + k], x0 = im[i + k], r1 = re[j + k], x1 = im[j + k];
                re[i + k] = ar * r0 - ai * x0 + br * r1 - bi * x1;
                im[i + k] = ar * x0 + ai * r0 + br * x1 + bi * r1;
                re[j + k] = cr * r0 - ci * x0 + dr * r1 - di * x1;
                im[j + k] = cr * x0 + ci * r0 + dr * x1 + di * r1;
            }
        }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= (1LL<<20))
        #endif
        for (long long b = 0; b < nblocks; b++) {
            long long base = b * step;
            for (long long k = 0; k < bit; k++) {
                long long i = base + k, j = i + bit;
                double r0 = re[i], x0 = im[i], r1 = re[j], x1 = im[j];
                re[i] = ar * r0 - ai * x0 + br * r1 - bi * x1;
                im[i] = ar * x0 + ai * r0 + br * x1 + bi * r1;
                re[j] = cr * r0 - ci * x0 + dr * r1 - di * x1;
                im[j] = cr * x0 + ci * r0 + dr * x1 + di * r1;
            }
        }
    }
}

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
            double *state_re, double *state_im) {
    int g = 0;
    while (g < n_gates) {
        int q = gate_qubit[g];
        double ar = 1, ai = 0, br = 0, bi = 0, cr = 0, ci = 0, dr = 1, di = 0; /* identity */

        while (g < n_gates && gate_qubit[g] == q) {
            double gar, gai, gbr, gbi, gcr, gci, gdr, gdi;
            int gt = gate_type[g];
            if (gt == 0) {
                double s = 0.70710678118654752440;
                gar = s; gai = 0; gbr = s; gbi = 0; gcr = s; gci = 0; gdr = -s; gdi = 0;
            } else if (gt == 1) {
                gar = 0; gai = 0; gbr = 1; gbi = 0; gcr = 1; gci = 0; gdr = 0; gdi = 0;
            } else if (gt == 2) {
                gar = 1; gai = 0; gbr = 0; gbi = 0; gcr = 0; gci = 0; gdr = -1; gdi = 0;
            } else {
                double phi0 = -gate_param[g] * 0.5, phi1 = gate_param[g] * 0.5;
                gar = cos(phi0); gai = sin(phi0); gbr = 0; gbi = 0;
                gcr = 0; gci = 0; gdr = cos(phi1); gdi = sin(phi1);
            }

            double t1r, t1i, t2r, t2i, nar, nai, nbr, nbi, ncr, nci, ndr, ndi;
            cmul(gar, gai, ar, ai, &t1r, &t1i); cmul(gbr, gbi, cr, ci, &t2r, &t2i);
            nar = t1r + t2r; nai = t1i + t2i;
            cmul(gar, gai, br, bi, &t1r, &t1i); cmul(gbr, gbi, dr, di, &t2r, &t2i);
            nbr = t1r + t2r; nbi = t1i + t2i;
            cmul(gcr, gci, ar, ai, &t1r, &t1i); cmul(gdr, gdi, cr, ci, &t2r, &t2i);
            ncr = t1r + t2r; nci = t1i + t2i;
            cmul(gcr, gci, br, bi, &t1r, &t1i); cmul(gdr, gdi, dr, di, &t2r, &t2i);
            ndr = t1r + t2r; ndi = t1i + t2i;

            ar = nar; ai = nai; br = nbr; bi = nbi; cr = ncr; ci = nci; dr = ndr; di = ndi;
            g++;
        }

        int is_identity = (fabs(ar - 1.0) < 1e-15 && fabs(ai) < 1e-15 &&
                            fabs(dr - 1.0) < 1e-15 && fabs(di) < 1e-15 &&
                            fabs(br) < 1e-15 && fabs(bi) < 1e-15 &&
                            fabs(cr) < 1e-15 && fabs(ci) < 1e-15);
        if (!is_identity) {
            apply_matrix(n_qubits, 1LL << q, ar, ai, br, bi, cr, ci, dr, di, state_re, state_im);
        }
    }
}
