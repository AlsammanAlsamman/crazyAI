# MAPPING (world-object → problem-object, per SEED)

**SEED 1 — "A quantity is a welded limb-chain of ones-, tens-, and hundreds-limbs rather than one indivisible mark."**

| World object | Problem object |
|---|---|
| a quantity / soul | one operand of a multiplication (a `double`) |
| limb-chain (ones-limb welded to tens-limb welded to hundreds-limb) | the value's scaled 56-bit fixed-point mantissa, split into 7 byte-sized limbs |
| the weld itself | the base-256 positional encoding that lets the limbs be re-summed exactly |
| Remaking (grafting flesh onto a boiler-driven debtor) | `frexp`/scale step that turns a floating value into an integer limb-chain plus a separate exponent |

Breaks: **"numbers are IEEE doubles and multiply is the primitive"** — a double stops being one atomic unit and becomes a chain of small integers.

**SEED 2 — "The Table of Grafts supplies every digit-pair's product as a memorized match looked up, never reckoned fresh."**

| World object | Problem object |
|---|---|
| Table of Grafts (boxwood board, every digit 0–9 × every digit 0–9) | a precomputed `GRAFT[256][256]` byte-product lookup table (base-256 "digits" instead of base-10) |
| "carved, inked, and waiting" | built once, before the hot loop, read-only thereafter |
| "a match looked up, not thought through anew" | the inner numeric step is a memory load, never a hardware `*` |

Breaks: **"numbers are IEEE doubles and multiply is the primitive"** — directly and most decisively: the multiply instruction is never issued in the numeric core; a limb×limb product is a table lookup.

**SEED 3 — "The ledgers are cut into slab-width plots... wax strip... scraped clean and discarded after each plot is banked."**

| World object | Problem object |
|---|---|
| debtor's ledger (downward columns) / thief's ledger (across rows) | A (walked along k per row i) / B (walked along j per row k) |
| shared rung / index-spine | the shared reduction index k |
| slab-width plot | an `(ii,jj,kk)` cache-blocking tile |
| wax accumulator-strip, scraped after each plot | transient stack/register state (limb bytes, partial products) discarded after each product is banked |
| third ledger | C, written once per plot |

This is essentially ordinary cache blocking — it doesn't break any of the listed assumptions in a new way (blocking is already part of "the known way").

# CHOSEN SEED

SEED 2 (Table of Grafts), built on the necessary substrate of SEED 1 (limb decomposition), with SEED 3 supplying the outer block/scaffold loop that any correct GEMM needs anyway. SEED 2 is the seed that most directly and literally breaks the assumption the instructions single out, and it is maximally different from "the known way" (OpenBLAS's register-tiled FMA microkernel): OpenBLAS pushes *more* multiplies per cycle through the FPU; this scheme tries to do zero hardware multiplies in the numeric core at all.

# ASSUMPTION BROKEN

"Numbers are IEEE doubles and multiply is the primitive." Here a double is unwelded into 7 base-256 limbs of a 56-bit fixed-point mantissa (SEED 1); every limb×limb term is supplied by a prebuilt 256×256 lookup table rather than a hardware multiply (SEED 2); the limb-products are carried into a 128-bit integer accumulator (the wax strip) and only converted back to a double once fully "banked."

# ARTIFACT — full literal object mapping

- **Ledgers lying flat / memory**: A, B in row-major DRAM (unchanged); the grafting-slab itself = the cache-resident tile of A,B,C plus the GRAFT table.
- **Shared rung**: index k.
- **Walking armature / crab-legs / processor**: one CPU core executing the inner loop; each OpenMP thread is its own armature walking its own `(ii,jj)` plot.
- **Table of Grafts**: static `GRAFT[256][256]`, built once, read-only.
- **Wax accumulator-strip**: the `unsigned __int128 strip`/`row` that builds one product's limb-chain and is discarded (scoped out) the instant that product is banked into the running double sum — nothing but the final sum survives, matching "none of it is truth, only the road to truth."
- **Slab-width plots**: the `PLOT×PLOT×PLOT` block loop.
- **Third ledger**: C, written once its plot is settled.
- **Time**: loop iterations; each "step" of the armature = one limb-pair lookup + carry.

```c
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- The Table of Grafts: every digit-pair's product, carved once, looked up forever ---- */
static uint16_t GRAFT[256][256];
static int graft_ready = 0;

static void build_graft_table(void) {
    if (graft_ready) return;
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++)
            GRAFT[a][b] = (uint16_t)(a * b);      /* memorized match, built ahead of time */
    graft_ready = 1;
}

/* unweld one double into its 7 byte-limbs (a 56-bit fixed-point mantissa)
 * plus a base-2 exponent and sign -- "a quantity is a limb-chain, not one mark". */
static inline void unweld(double v, uint8_t limb[7], int *exp2, int *sgn) {
    if (v == 0.0) { memset(limb, 0, 7); *exp2 = 0; *sgn = 0; return; }
    *sgn = (v < 0.0) ? -1 : 1;
    int e;
    double m = frexp(fabs(v), &e);                /* m in [0.5,1) */
    double scaled = m * 72057594037927936.0;       /* m * 2^56 */
    uint64_t M = (uint64_t)(scaled + 0.5);
    if (M >> 56) { M >>= 1; e++; }                 /* guard the rare round-up to 2^56 */
    for (int p = 0; p < 7; p++) { limb[p] = (uint8_t)(M & 0xFF); M >>= 8; }
    *exp2 = e;
}

/* multiply two doubles by unwelding them into limbs, walking every limb-pair
 * to the Table of Grafts, and carrying the looked-up grafts onto a wax
 * accumulator-strip -- no hardware '*' ever touches x or y. */
static inline double graft_multiply(double x, double y) {
    uint8_t bx[7], by[7];
    int ex, ey, sx, sy;
    unweld(x, bx, &ex, &sx);
    unweld(y, by, &ey, &sy);
    if (sx == 0 || sy == 0) return 0.0;

    unsigned __int128 strip = 0;                   /* the wax accumulator-strip */
    for (int p = 0; p < 7; p++) {
        unsigned __int128 row = 0;
        for (int q = 0; q < 7; q++)
            row += (unsigned __int128)GRAFT[bx[p]][by[q]] << (8 * q);
        strip += row << (8 * p);                   /* overflow-limb carried sideways */
    }                                               /* strip is scraped bare on return */

    double mag = ldexp((double)strip, ex + ey - 112);
    return (sx * sy < 0) ? -mag : mag;
}

#ifndef PLOT
#define PLOT 48   /* the slab's reach: a dozen-ish rows married to a dozen-ish columns */
#endif

void kernel(int n, const double *A, const double *B, double *C) {
    build_graft_table();
    memset(C, 0, (size_t)n * n * sizeof(double));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < n; ii += PLOT) {
        for (int jj = 0; jj < n; jj += PLOT) {
            int i_max = ii + PLOT < n ? ii + PLOT : n;
            int j_max = jj + PLOT < n ? jj + PLOT : n;
            for (int kk = 0; kk < n; kk += PLOT) {
                int k_max = kk + PLOT < n ? kk + PLOT : n;
                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        double sum = C[(size_t)i * n + j];
                        for (int k = kk; k < k_max; k++) {
                            double a = A[(size_t)i * n + k];
                            double b = B[(size_t)k * n + j];
                            sum += graft_multiply(a, b);   /* banked into the third ledger's cell */
                        }
                        C[(size_t)i * n + j] = sum;
                    }
                }
            }
        }
    }
}
```

Correctness note (stated error): `unweld` scales the mantissa to 56 bits (finer than a double's 53-bit mantissa), and `strip = Mx*My` is computed **exactly** via lookups+shifts+adds (no rounding anywhere in the limb arithmetic). The only rounding is the final `(double)strip` → double cast, i.e. at most the ordinary ~1 ULP rounding a correctly-rounded multiply would also incur. So accuracy should match native multiply to essentially the same tolerance; the k-reduction itself uses plain double addition, same as the baseline.

PREDICTION: speedup_vs_blocked = 0.01

Reasoning behind that number: each single scalar product now costs two `frexp`/`ldexp` library calls, 14 byte extractions, and 49 data-dependent table lookups (each a scalar memory load into a 128 KB table — not vectorizable with ordinary loads, and gather instructions are themselves too high-latency to fix this) plus `__int128` carry adds — on the order of a few hundred cycles of scalar, barely-pipelined work per multiply-add, replacing what `-O3 -march=native` turns into one vectorized FMA lane (4–8 multiply-adds per instruction, sub-cycle amortized) in the cache-blocked baseline. I expect roughly 50–200× slowdown; 0.01 (≈100×) is my point estimate.

# MEASUREMENT

Not executed in this session — no tool access was available here (explicitly stated in the task setup), so `kernel_bench` was not run by me. The prediction above is a reasoned estimate only, not a fabricated measurement. I'm flagging this plainly rather than inventing a number: the external pipeline is expected to compile and benchmark this kernel and report the real figure.

# VERDICT

Predicted, not verified: this literal translation of "Table of Grafts" almost certainly loses badly against the cache-blocked baseline (and far more badly against OpenBLAS), because it trades a single-cycle-amortized, SIMD-width hardware FMA for dozens of scalar, latency-bound table lookups per multiply-add. This is an honest expected failure of the literal mapping, not a disguised win — reported as such rather than quietly substituted with a textbook GEMM optimization. The exercise still produced something the "known way" never would: an exact-in-the-mantissa, multiply-free scalar product built entirely from table lookups and carries, which is the genuinely novel (if slow) artifact the SEED demanded.