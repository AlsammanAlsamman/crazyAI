# MAPPING

| World object | Problem object | Silent assumption it targets |
|---|---|---|
| **Seed 1** – a number is a row of rung-coins on a stalk, never one indivisible mark | An IEEE double's mantissa is exploded into an array of 8‑bit limbs (a little "bignum" of 7 bytes covering the 53‑bit mantissa) instead of being treated as one 64‑bit word | "numbers are IEEE doubles and multiply is the primitive" (prerequisite for breaking it) |
| **Seed 2** – each rung‑against‑rung product comes from a memorized strawberry‑chart, never freshly reckoned | A static 256×256 lookup table `BYTE_MUL[a][b]` of byte‑products, consulted instead of issuing a hardware multiply for every limb pair | "numbers are IEEE doubles and multiply is the primitive" — **directly and squarely** |
| **Seed 3** – partial rung‑products sit in carry‑baskets that overflow and cascade downstream, settling in one basin | A 14‑slot accumulator array, summed by limb weight, then carry‑propagated low→high into a single 106‑bit product, which is then Horner‑folded into one rounded double | "every product is computed exactly, once" / "one product is one problem, many products are many problems" |

# CHOSEN SEED

**Seed 2** — the strawberry‑chart lookup. It is the only one of the three that, on its own, breaks the assumption the task specifically flags ("numbers are IEEE doubles and multiply is the primitive"), and it is the most literal possible reading: children reciting a memorized times‑table maps to nothing else so cleanly as "replace the CPU's `fmul` with a table lookup." Seeds 1 and 3 are load‑bearing scaffolding this needs (you can't look up a digit‑product without first having digits, and you can't get a final number back without carrying) — I use all three in the artifact, but Seed 2 is the defining, chosen mechanism.

# ASSUMPTION BROKEN

"numbers are IEEE doubles and multiply is the primitive" — the primitive operation inside every one of the n³ scalar multiplications is no longer `a*b`; it is 49 memory lookups into a shared 64 KiB table plus an explicit carry cascade.

**World→computation mapping, literally:**
- eggshell houses = the two matrices A, B (still, resident RAM).
- lattice of stalks, one per shared index = the k‑loop; this is what *flows* through a cell.
- coin‑row pair on a stalk = the operand pair (A[i][k], B[k][j]).
- rung = one byte position of the mantissa decomposition.
- strawberry‑chart = the static `BYTE_MUL[256][256]` table — one shared, precomputed resource every stalk consults, never recomputed.
- carry‑basket = `basket[pos]`, weighted accumulator per limb position; "overflow tumbles downstream" = the explicit low→high carry‑propagation loop.
- basin at the bottom of the cell = the settled 106‑bit product, Horner‑folded into one rounded double, then added into the per‑(i,j) running sum `basin`.
- "cut the stalks, pitch the baskets in the river, no paddling upstream" = all scratch (`Alimb/Blimb/basket/settled`) is stack‑local and discarded per (i,k,j) triple — no memoization.
- "the shrinking man in his bubble watches the paces go by" = the serial carry‑propagation chain inside one scalar multiply — a latency‑bound critical path that cannot be hidden by parallelism.
- processor = one OpenMP thread owning a slice of output rows i; time = wall‑clock spent both across cells (parallel) and inside the byte‑cascade (serial, per multiply).

# ARTIFACT

```c
#include <string.h>
#include <math.h>
#include <stdint.h>

/* the strawberry-chart: a memorized rung-product table, shared by every stalk */
static uint16_t BYTE_MUL[256][256];
static int BYTE_MUL_READY = 0;

static void init_strawberry_chart(void) {
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++)
            BYTE_MUL[a][b] = (uint16_t)(a * b);
    BYTE_MUL_READY = 1;
}

/* multiply two nonnegative finite doubles by decomposing each into a
 * row of 8-bit rung-coins, looking up every rung-against-rung product
 * on the strawberry-chart, and letting the partial products cascade
 * through carry-baskets into one settled basin. */
static inline double lut_mul_mag(double a, double b) {
    if (a == 0.0 || b == 0.0) return 0.0;

    int ea, eb;
    double ma = frexp(a, &ea);   /* a = ma * 2^ea, ma in [0.5,1) */
    double mb = frexp(b, &eb);

    /* the coin-row: exact 53-bit integer mantissa, read off (not multiplied) */
    uint64_t Ma = (uint64_t)ldexp(ma, 53);
    uint64_t Mb = (uint64_t)ldexp(mb, 53);

    uint8_t Alimb[7], Blimb[7];
    for (int i = 0; i < 7; i++) {
        Alimb[i] = (uint8_t)(Ma & 0xFF); Ma >>= 8;
        Blimb[i] = (uint8_t)(Mb & 0xFF); Mb >>= 8;
    }

    /* the lattice of stalks: one carry-basket per output rung position */
    uint32_t basket[14];
    memset(basket, 0, sizeof(basket));

    for (int p = 0; p < 7; p++) {
        if (!Alimb[p]) continue;
        for (int q = 0; q < 7; q++) {
            if (!Blimb[q]) continue;
            basket[p + q] += BYTE_MUL[Alimb[p]][Blimb[q]]; /* strawberry-chart lookup */
        }
    }

    /* overflow tumbles downstream, rung by rung */
    uint8_t settled[14];
    uint32_t carry = 0;
    for (int pos = 0; pos < 14; pos++) {
        uint32_t v = basket[pos] + carry;
        settled[pos] = (uint8_t)(v & 0xFF);
        carry = v >> 8;
    }

    /* the basin: fold the settled rungs, most significant first, into one double */
    double D = 0.0;
    for (int pos = 13; pos >= 0; pos--)
        D = ldexp(D, 8) + (double)settled[pos];

    return ldexp(D, ea + eb - 106);
}

static inline double lut_mul(double a, double b) {
    if (!isfinite(a) || !isfinite(b)) return a * b; /* stalks don't cover nan/inf */
    double r = lut_mul_mag(fabs(a), fabs(b));
    int neg = (a < 0.0) != (b < 0.0);
    return neg ? -r : r;
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (!BYTE_MUL_READY) init_strawberry_chart();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double basin = 0.0; /* the basin at the bottom of this cell */
            for (int k = 0; k < n; k++) {
                double coin_a = A[(size_t)i * n + k];
                double coin_b = B[(size_t)k * n + j];
                basin += lut_mul(coin_a, coin_b);
            }
            C[(size_t)i * n + j] = basin;
        }
    }
}
```

Correctness caveat, stated honestly: this is exact for the mantissa product (106 bits are fully retained through the carry cascade), but the final 106‑bit→double fold uses an iterative Horner reduction rather than a single correctly‑rounded step, so results can differ from the true round‑to‑nearest‑even product by up to ~1 ULP (double‑rounding). Non‑finite operands fall back to native multiply.

PREDICTION: speedup_vs_blocked = 0.02

Reasoning for the number: each scalar multiply now costs two `frexp` calls, up to 49 table lookups (with unpredictable data‑dependent branches, unlike the perfectly regular access pattern hardware `fmul`/FMA units pipeline at ~1 per cycle with AVX throughput of 4–8 doubles/cycle), a 14‑step serial carry chain, and a 14‑step serial Horner fold, plus two more `ldexp` calls — roughly 100+ instructions with a real serial‑latency floor, replacing one pipelined FMA. OpenMP parallelism across rows can claw back maybe an order of magnitude on a multi‑core box, but not the ~2–3 orders of magnitude lost per multiply. I expect this kernel to land around 20–100× slower than the cache‑blocked baseline, hence a predicted speedup of ≈0.02 (≈50× slower) — and correspondingly nowhere near OpenBLAS.

# MEASUREMENT

Not run: no tool access is available in this session (per the environment reminder, `kernel_bench` etc. cannot be invoked here). This artifact and prediction are handed off for the downstream pipeline to compile with `gcc -O3 -march=native -fopenmp -lm` and benchmark against the cache‑blocked baseline and OpenBLAS; I have not fabricated a number for this section.

# VERDICT

Taken completely literally, Seed 2 is a real, well‑defined, and (by my hand‑trace) correct-to‑within‑~1‑ULP way to compute a double product without ever issuing a hardware multiply on the operands themselves — it genuinely breaks "multiply is the primitive." But nothing about replacing a single pipelined FMA with dozens of data‑dependent table lookups and two serial carry/fold chains should make this *fast*; the honest expectation, stated before any measurement, is a substantial slowdown rather than a win. This is a case where taking the metaphor seriously produces a working, interesting artifact but not (as far as I can reason without measuring) a faster one — that predicted failure should stand as the reported result unless the pipeline's actual numbers say otherwise.