#include <string.h>
#include <stdlib.h>

/* The mountain-blueprint trough: a terrain built once, kept forever, one
   terrace for every pairing of digit-weights a stone could hold. Water
   poured to height a against height b settles at the mark a*b -- we only
   read that mark, we never reckon the multiply fresh. */
static int LUT[256][256];
static int lut_built = 0;
static void build_lut(void) {
    if (lut_built) return;
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++)
            LUT[a][b] = a * b;
    lut_built = 1;
}

/* A quantity is a row of digit-pebbles laid by weight of hill: hi is the
   hill-ward (heaviest) pebble, lo the lesser one laid beside it. */
typedef struct { unsigned char hi, lo; signed char sign; } Pebbles;

static void quantize(const double *M, size_t N, Pebbles *P, double *scale_out) {
    double maxabs = 0.0;
    for (size_t t = 0; t < N; t++) {
        double v = M[t] < 0 ? -M[t] : M[t];
        if (v > maxabs) maxabs = v;
    }
    if (maxabs == 0.0) maxabs = 1.0;
    const double inv = 65535.0 / maxabs;
    #pragma omp parallel for
    for (long long t = 0; t < (long long)N; t++) {
        double v = M[t];
        double av = (v < 0 ? -v : v) * inv;
        unsigned int q = (unsigned int)(av + 0.5);
        if (q > 65535u) q = 65535u;
        P[t].hi = (unsigned char)((q >> 8) & 0xFFu);  /* hill-ward, heaviest */
        P[t].lo = (unsigned char)(q & 0xFFu);          /* the lesser pebble */
        P[t].sign = (v < 0.0) ? -1 : 1;
    }
    *scale_out = maxabs;
}

void kernel(int n, const double *A, const double *B, double *C) {
    build_lut();
    size_t N = (size_t)n * (size_t)n;

    Pebbles *PA = (Pebbles *)malloc(N * sizeof(Pebbles));
    Pebbles *PB = (Pebbles *)malloc(N * sizeof(Pebbles));
    double scaleA = 1.0, scaleB = 1.0;
    quantize(A, N, PA, &scaleA);
    quantize(B, N, PB, &scaleB);

    /* the value of one grain of rust, translated back out of the trough */
    const double unit = 256.0 * (scaleA / 65535.0) * (scaleB / 65535.0);

    memset(C, 0, N * sizeof(double));

    const int BI = 64, BK = 64, BJ = 256; /* hill-sized blocks / basements */

    #pragma omp parallel
    {
        double *wire = (double *)malloc((size_t)BJ * sizeof(double));
        #pragma omp for schedule(dynamic)
        for (int ib = 0; ib < n; ib += BI) {
            int imax = ib + BI < n ? ib + BI : n;
            for (int kb = 0; kb < n; kb += BK) {
                int kmax = kb + BK < n ? kb + BK : n;
                for (int jb = 0; jb < n; jb += BJ) {
                    int jmax = jb + BJ < n ? jb + BJ : n;
                    int jlen = jmax - jb;
                    for (int i = ib; i < imax; i++) {
                        memcpy(wire, C + (size_t)i * n + jb, (size_t)jlen * sizeof(double));
                        for (int k = kb; k < kmax; k++) {
                            const Pebbles pa = PA[(size_t)i * n + k];
                            const int ah = pa.hi, al = pa.lo;
                            const double s = unit * (double)pa.sign;
                            const Pebbles *rowB = PB + (size_t)k * n;
                            for (int j = jb; j < jmax; j++) {
                                const Pebbles pb = rowB[j];
                                /* pour the pebble-shape into the trough; read
                                   where the water settles -- no fresh multiply.
                                   the least pebble's corner of the terrace was
                                   never carved fine enough: spilled, thrown away. */
                                int acc = LUT[ah][pb.hi] * 256 + LUT[ah][pb.lo] + LUT[al][pb.hi];
                                double prod = (double)acc * s * (double)pb.sign;
                                /* rust trickles onto the wire: waves thicken,
                                   not erase; the whole sum over k gathers here */
                                wire[j - jb] += prod;
                            }
                        }
                        memcpy(C + (size_t)i * n + jb, wire, (size_t)jlen * sizeof(double));
                    }
                }
            }
        }
        free(wire);
    }

    free(PA);
    free(PB);
}
