## MAPPING

| World object | Problem object | Which assumption it breaks |
|---|---|---|
| **SEED 1 — lattice-and-clay casting** | | |
| strand / riverbed | the DNA sequence `seq[0..n)` | — |
| bead | one base (one char of `seq`) | — |
| wet lattice frame, cut once over exactly k beads | the k-window `seq[i..i+k)` | *"every base contributes independently to the window it starts"* — the frame is a single fixed-size stencil applied once, not k independent per-base contributions computed separately |
| clay taking the exact seam, crest | the packed 2-bit code for that window | *"code must be built fully from scratch, one base at a time"* — the crest is cast as one solid object, not accreted base-by-base each time |
| **SEED 2 — desert mounds & wildflower seeds** | | |
| flat desert, "locked part," horizon permits no new makes | the pre-zeroed `counts` array (fixed size `4^k`, no allocation allowed) | — |
| pressing a crest into the sand to test seam match | indexing `counts[code]` | *"the count table can only be updated one position's result at a time"* — matching many crests against the desert is framed as independent, order-free lookups, not a strictly sequential position-by-position update |
| planting a fresh mound vs. adding a seed | creating vs. incrementing a bucket | — |
| rain claying the ground so mounds never drift/merge | guarantee that distinct codes never collide (each thread's/telling's tally is safe to keep separate) | *"the whole sequence must be read once, start to end, in order"* — implies independent tallies (e.g. per chunk/thread) can be kept apart and only reconciled later ("rain crosses at intervals"), i.e. order-independent partial counting |
| **SEED 3 — the riverbed shadow that dissolves the beads** | | |
| "I do not look back at the beads once the crest is drawn" | old raw bases are never re-read for the next window | *"each k-mer's code must be built fully from scratch, one base at a time, before it can be counted"* — directly broken: only **one new bead** is examined per step, the other k−1 are never revisited |
| "shadow uses them up right there... nothing of the strand lingers but its casting" | raw window bytes are consumed; only the running code survives | *"one window is examined, then discarded, before the next begins"* — broken: windows overlap and share state; nothing is independently re-examined |
| "the clay returns wet to the wall for the next stretch" | the same code register is reused/updated in place for the next window | same as above, most literally — this is a **rolling code register**, not a fresh buffer per window |

## CHOSEN SEED

**SEED 3 — the riverbed shadow that dissolves each stretch of beads the instant its crest is cast.**

It is the most literal and the most different from the known way: it doesn't just describe *how* to pack a window into a code (SEED 1) or *how* to file the result (SEED 2, already given by the contract's flat pre-zeroed array) — it makes an explicit, load-bearing claim about *state reuse*: "I do not look back at the beads," and "the clay returns wet to the wall for the next stretch." Taken literally, that forbids re-deriving each k-mer's code from its k raw bases (the naive method's O(k) inner loop) and instead demands a single running register that is *shifted forward by one bead* and reused, never rebuilt.

## ASSUMPTION BROKEN

"each k-mer's code must be built fully from scratch, one base at a time, before it can be counted" — broken by keeping one running 2-bit-packed register across the whole pass and updating it with exactly one new base per step (classic rolling k-mer encode), instead of re-reading all k bases at every position. This also breaks "one window is examined, then discarded, before the next begins," since windows now literally share (overlapping) state rather than being independently reconstructed.

SEED 2's imagery ("mounds never drift or merge") is used as a secondary, literal justification for a further engineering step: independent partial tallies (private desert plots) can be kept per thread and only reconciled at the end, without any risk of two different seams merging into one hollow — i.e., OpenMP with private histograms and a final reduction.

## ARTIFACT

```c
#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline int base_code(char c) {
    switch (c) { case 'A': return 0; case 'C': return 1; case 'G': return 2; default: return 3; }
}

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || k > 31 || n < k) return;

    uint64_t mask = (1ULL << (2 * k)) - 1ULL;
    uint64_t table_size = 1ULL << (2 * k);
    long long total_windows = (long long)n - k + 1;

#ifdef _OPENMP
    /* Only fan out to private "deserts" when there are enough windows to
       amortize the setup, and each private table is small enough that
       duplicating it per thread doesn't blow up memory (SEED 2: mounds
       kept apart, reconciled only when the rain comes). */
    int nthreads = omp_get_max_threads();
    if (nthreads > 1 && total_windows > 4LL * k && table_size <= (1ULL << 24)) {
        if ((long long)nthreads > total_windows) nthreads = (int)total_windows;
        uint64_t **partial = (uint64_t **)malloc(sizeof(uint64_t *) * nthreads);
        for (int t = 0; t < nthreads; t++)
            partial[t] = (uint64_t *)calloc((size_t)table_size, sizeof(uint64_t));

        #pragma omp parallel num_threads(nthreads)
        {
            int t = omp_get_thread_num();
            long long chunk = (total_windows + nthreads - 1) / nthreads;
            long long start = (long long)t * chunk;
            long long end = start + chunk;
            if (end > total_windows) end = total_windows;
            if (start < end) {
                uint64_t *local = partial[t];
                uint64_t code = 0;
                /* Prime the running crest with the k-1 beads just before
                   this thread's first window; after that we never look
                   back at a raw bead again. */
                for (long long j = 0; j < k - 1; j++)
                    code = ((code << 2) | (uint64_t)base_code(seq[start + j])) & mask;
                for (long long w = start; w < end; w++) {
                    code = ((code << 2) | (uint64_t)base_code(seq[w + k - 1])) & mask;
                    local[code]++;
                }
            }
        }
        for (int t = 0; t < nthreads; t++) {
            uint64_t *local = partial[t];
            for (uint64_t c = 0; c < table_size; c++) counts[c] += local[c];
            free(local);
        }
        free(partial);
        return;
    }
#endif
    {
        uint64_t code = 0;
        int i;
        for (i = 0; i < k - 1; i++)
            code = ((code << 2) | (uint64_t)base_code(seq[i])) & mask;
        for (i = k - 1; i < n; i++) {
            code = ((code << 2) | (uint64_t)base_code(seq[i])) & mask;
            counts[code]++;
        }
    }
}
```

PREDICTION: speedup_vs_naive = 10

## MEASUREMENT

Not run: `kmer_bench`/`kmer_contract` are not actually reachable in this session (the environment note explicitly states no tools are available here), so I cannot report a real measured number. Reporting one anyway would be fabrication, so this section stays honest rather than invented.

Analytical reasoning behind the prediction, stated plainly since I can't back it with a run: the naive kernel pays O(k) switch/shift/or work per window on top of the same random-access `counts[code]++`. The rolling version collapses that to O(1) amortized per window (one new base read per step), so the sequential part alone should save roughly a small-integer factor tied to k (not a full ×k, since the dominant cost for large tables is the cache-unfriendly scatter-increment into `counts`, which both versions pay equally). The OpenMP layer on top should add a further roughly linear-in-core-count factor as long as `4^k` fits the private-table memory budget, degrading if the table is too large or memory-bandwidth bound. Ten was chosen as a conservative combination of a modest (~2–3×) sequential gain and a moderate (~3–4×) parallel gain, not an optimistic ceiling.

## VERDICT

Provisional, pending an actual run: the literal translation of "I do not look back at the beads" into a rolling 2-bit register is a real, well-founded algorithmic change (O(nk) → O(n) construction cost) and is a standard, credible speedup for this exact problem — it is not a fantasy translation. The SEED-2-motivated OpenMP/private-histogram layer is a legitimate but riskier addition (memory duplication, reduction overhead) that should be verified, and reduced to the sequential-only path if it doesn't pay off, once `kmer_bench` is actually available to check the prediction against reality.