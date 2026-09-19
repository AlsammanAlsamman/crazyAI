# MAPPING

| World object (native's description) | Problem object (WHT kernel) | Assumption it breaks |
|---|---|---|
| **Seed 1 —** row of wicks in a trough; knight-bent piece (2‑forward‑1‑sideways) walks the row in one fixed pass, touching every wick exactly once | The array of `n` positions, visited in **stage/stride order**: for a fixed gap `len` (1,2,4,…,n/2), every index is grouped with the partner `len` away and every index is touched exactly once per stage | **"the input order cannot be rearranged"** — instead of scanning `j=0..n-1` for one fixed `k`, the traversal groups elements by binary distance (`len`), a completely different fixed order than natural index order |
| **Seed 2 —** owl‑headed oracle raises/lowers her bladder‑stick; raised → paint a solid stripe, lowered → leave raw/un‑drunk | For each visited pair `(u,v)` the outcome is **both** kept: the "raised" branch is `u+v`, the "lowered" branch is `u−v` — computed together from the same two operands | **"combining two contributions always means an ordinary addition"** — every pairing yields one addition *and* one subtraction, produced from a single pair of reads, not an independent ± decision per output |
| **Seed 3 —** each finished row is carried off as a page by a fresh bird; wicks are doused and restarted with one flip added, "the way a counter climbs," until no bird‑flight repeats one already in the loft | Each completed stage (`len` doubling: 1→2→4→…) is one "page" = the whole array's intermediate state; the next stage is built **directly from that page**, and there are exactly `log2(n)` stages, each run once, no repeats | **"there is no way to reuse work between different outputs"** — every output shares the sums computed in earlier stages instead of being recomputed independently from the raw input each time |

# CHOSEN SEED
**Seed 1** — the knight's fixed creeping traversal of the wick row.

It is the most literal: "visit every wick once, in a fixed non‑sequential order" translates directly and unambiguously into "loop over the array once per stage, grouping indices by a fixed stride, touching each index exactly once" — no invented machinery needed. It is also the most different in *shape* from the textbook presentation of the "known way," which is normally described top‑down/recursively (split the array in half, recurse, combine). The knight never recurses or splits conceptually — it makes one flat, iterative, fixed‑stride pass over the whole row per stage. That dictates an **iterative, bottom‑up, in‑place butterfly** rather than a recursive one, even though the arithmetic is the same butterfly operation (Seeds 2 and 3 fill in what happens *at* each stop and *between* stages, respectively).

# ASSUMPTION BROKEN
"The input order cannot be rearranged" (chosen seed) — plus, as a consequence of adopting the same structure for all stages, "each output is one independent sum" and "there is no way to reuse work between different outputs" are broken too: outputs are built from shared intermediate pages, not recomputed from scratch.

# ARTIFACT

```c
#include <string.h>
#include <omp.h>

/* out[k] = sum_j in[j] * (-1)^popcount(j & k), computed as an iterative,
 * in-place butterfly: one flat pass over the wick-row per stage (Seed 1),
 * each visited pair producing both a "raised" (+) and "lowered" (-)
 * outcome from the same two operands (Seed 2), each stage's page built
 * directly from the previous page instead of raw input (Seed 3). */
void kernel(int n, const double *in, double *out) {
    memcpy(out, in, (size_t)n * sizeof(double));

    for (int len = 1; len < n; len <<= 1) {
        int step = len << 1;
        int nblocks = n / step;

        #pragma omp parallel for schedule(static)
        for (int b = 0; b < nblocks; b++) {
            double * restrict p = out + (size_t)b * step;
            double * restrict q = p + len;
            #pragma omp simd
            for (int j = 0; j < len; j++) {
                double u = p[j];
                double v = q[j];
                p[j] = u + v;
                q[j] = u - v;
            }
        }
    }
}
```

Correctness check by hand (n=4, in = [i0,i1,i2,i3]):
- stage len=1: out → [i0+i1, i0−i1, i2+i3, i2−i3]
- stage len=2: out → [i0+i1+i2+i3, i0−i1+i2−i3, i0+i1−i2−i3, i0−i1−i2+i3]

which matches `out[k]=Σ in[j]·(−1)^popcount(j&k)` for k=0..3 exactly (verified term‑by‑term).

Algorithmic cost: `log2(n)` stages, `n/2` add/sub pairs each ⇒ **O(n log n)** total work, vs. O(n²) for the naive double loop — the same complexity class as the standard FWHT, reached here via the flat stride‑walk (iterative) shape dictated literally by the knight's single fixed pass, rather than a recursive split.

PREDICTION: speedup_vs_naive = 200

# MEASUREMENT
No `wht_bench` tool call was available in this session (tools were disabled for this run — explicitly noted in the environment reminder), so no live measurement was taken. The prediction above is based purely on the operation‑count ratio (≈ n / log₂n for a representative benchmark size in the low thousands, discounted for thread/vectorization overhead on the memory‑bound small‑`len` stages) and is offered as the pre‑registered number for the pipeline to check against its actual `wht_bench` run.

# VERDICT
Not independently verified this session — pending the pipeline's own compile/measure pass against `wht_bench`. Algorithmically the kernel is a straightforward, correctness‑checked O(n log n) in‑place butterfly (hand‑verified for n=2 and n=4 against the reference formula), so I expect it to pass correctness and land in the same complexity class as the naive→fast ratio predicted above; I am not claiming the exact 200× figure is confirmed, only that it is the honest estimate made before any measurement, as required.