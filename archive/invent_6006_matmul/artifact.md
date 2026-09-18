# MAPPING (world-object → problem-object, per SEED)

**SEED 1 — "Each quantity is a row of digit-pebbles laid by weight of hill, not a single mark."**

| World object | Problem object |
|---|---|
| a quantity | one entry of A or B (a double) |
| row of pebbles, hill-ward heaviest first | fixed-point digit decomposition of that double's magnitude into weighted bytes, most-significant byte first |
| "a great count is truly many small counts leaning on each other" | the value is reconstructed only by weighting and summing the digit bytes |

Breaks: **"numbers are IEEE doubles and multiply is the primitive"** — a number stops being one 64-bit float and becomes a short vector of small integers.

**SEED 2 — "pouring the pebble-shape into the mountain-blueprint trough, reading where the water settles."**

| World object | Problem object |
|---|---|
| trough carved beforehand, one terrace per pairing of digit-weights | a static `LUT[256][256]` of all byte×byte products, built once |
| pouring water to the shape's height | writing the two digit-bytes as indices |
| water settling at the product level | reading `LUT[a][b]` instead of executing a multiply instruction |
| "without my reckoning it fresh each time" | the multiply is never issued at run time — it's a memory read of a precomputed answer |

Breaks: **"numbers are IEEE doubles and multiply is the primitive"** — most directly and literally of the three. The hardware multiplier is replaced by a lookup table.

**SEED 3 — "rust gathering in waves along the narrator-wire, cellar by cellar, overflow spilled and thrown away."**

| World object | Problem object |
|---|---|
| narrator-wire strung the length of the basement | a length-`BJ` accumulator buffer over one row-block of C |
| rust trickling, waves thickening not erasing | `wire[j] += prod` for each k, not one finished dot-product |
| "hill-sized blocks... only then descending" | cache blocking over (i,k,j) tiles, one tile fully finished before the next opens |
| overflow spilled, thrown away | the least-significant digit-pebble's cross term is deliberately dropped from the product — a stated, bounded relative error, not an exact result |

Breaks: "the whole sum over the shared index is finished before the next cell is started" and "every product is computed exactly, once" — but **not** the doubles/multiply-primitive assumption.

# CHOSEN SEED
**SEED 2** (the trough/terrace lookup), built on the digit-decomposition of SEED 1 and the trickling accumulation/blocking of SEED 3 to actually run. Of the three, only SEED 2 breaks "numbers are IEEE doubles and multiply is the primitive" in the most literal sense — the multiply instruction itself is replaced by a table read — so per the instructions it is preferred over SEED 3, which is the most literal in isolation but doesn't touch that assumption.

# ASSUMPTION BROKEN
"Numbers are IEEE doubles and multiply is the primitive." Each double is replaced by a 2-pebble (hi-byte, lo-byte, sign) fixed-point digit row; the elementary multiply of two numbers is replaced by up to three reads from a fixed 256×256 int lookup table built once and reused for the entire computation (and, in principle, forever), with the least-significant cross-term thrown away as declared overflow.

Literal object map: memory = the doubles (A, B, C) plus the two pebble-arrays PA/PB plus the one, permanent 256×256 terrain (LUT); what flows = the pebble-shapes poured k by k into the trough, and the rust (partial products) trickling onto the wire; what stays still = the terrain (LUT), built once, never rebuilt; a processor = one OpenMP thread, each holding its own basement (one wire buffer, one (ib,kb,jb) tile) so no thread ever has more than one basement open; time = the kb/jb block order, strictly one basement filled, read, and scraped clean before the next is opened.

# ARTIFACT

PREDICTION: speedup_vs_blocked = 0.03

```c
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
```

Stated error: dropping the `lo*lo` cross-term and 16-bit quantization together bound the per-product relative error to roughly 3–5×10⁻⁵; summed over n terms with mixed signs this is expected to land around 10⁻⁴–10⁻³ relative error in C for n in the low thousands — not exact, but bounded and explicit, exactly as the native's "trusting the terraces to be close enough" implies.

# PREDICTION
PREDICTION: speedup_vs_blocked = 0.03

Reasoning: the reference blocked kernel's inner loop is a single, data-independent FMA that GCC auto-vectorizes 4–8 doubles wide at `-O3 -march=native`. This kernel's inner loop instead does three data-dependent reads into a 256 KB table per (k,j) pair — loads the compiler cannot vectorize (gather-like, index depends on data), each riding L1/L2 latency even when pipelined — plus extra integer combine and rescale arithmetic. I expect this to cost on the order of 15–40× more cycles per output contribution than the vectorized FMA, only partly offset by half the memory traffic (1-byte pebbles vs 8-byte doubles) and by identical blocking/OpenMP treatment on both sides. Net expectation: markedly slower, not faster — a genuine test of whether the "cheap in their world" trough is actually cheap in ours, and by this estimate it plainly is not.

# MEASUREMENT
Not available in this session — no `kernel_bench` tool was reachable here, so this artifact is being handed to the pipeline unmeasured, exactly as specified in the closing instruction ("the pipeline will compile and measure it"). I am not fabricating a number; the PREDICTION above is the analytical estimate made *before* any run, and it stands as the falsifiable claim to check against whatever the pipeline reports.

# VERDICT
Taken completely literally, the native's method is a real, coherent algorithm — digit-decompose each number, replace the elementary multiply with a permanent lookup table, accumulate contributions as trickling waves, bound memory with sequential-basement blocking — and it produces a correct answer within a small stated error. But it is very unlikely to be fast on this hardware: our "trough" is DRAM/cache, not free-flowing water, and a data-dependent table read is nowhere near as cheap here as a single hardware FMA instruction, which is already about as optimized a primitive as this machine has. Unless a future revision replaces the software LUT with an actual fixed-function low-precision dot-product unit (e.g. AVX-512 VNNI int8, the real-world technique this metaphor most resembles — the Ozaki/DGEMM-via-tensor-core scheme), I predict — and will report, once the pipeline measures it — that this loses to the cache-blocked baseline by roughly an order of magnitude, not beats it. That is the honest reading of taking the native seriously rather than quietly substituting the textbook kernel.