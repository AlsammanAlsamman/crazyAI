# MAPPING

**Seed 1 — the sliding oar-frame**

| World object | Problem object |
|---|---|
| strand | the DNA sequence `seq` |
| bead | one base (one character, 2-bit "color") |
| oar-frame cut to hold exactly k beads | the current k-mer window, held as a packed 2k-bit code register |
| mast-peg, bead pinned and named by color (shell/flower-plant/plank-grain/dent-poison) | one 2-bit slot in the code, one of the 4 base values A/C/G/T |
| reading the frame left to right | extracting the window's current packed code |
| trailing bead drops off the near end, into the barrel | the oldest base's 2 bits are shifted out (masked off) |
| new bead drawn onto the far peg | the next base's 2 bits are shifted in |
| sliding the frame one bead forward | `code = ((code << 2) | new_base) & mask` |
| frame runs past the last bead, stop | loop while `i + k <= n` |

Breaks: *"each k-mer's code must be built fully from scratch, one base at a time, before it can be counted."* Only the very first window is built from scratch; every later window is *derived* from the previous one by dropping one bead and drawing one bead — the inner `for(j=0..k)` re-encode loop disappears almost entirely.

**Seed 2 — the name-plank matched against the courtyard**

| World object | Problem object |
|---|---|
| name-plank | the packed code for one window |
| courtyard, planks standing in a row | the address space of all 4^k possible codes |
| walking the row, comparing notch-patterns (barrage-stand method) | a linear search for a matching code |

Taken completely literally this seed asks for an O(#distinct planks) scan per window to find "the matching plank" — but the contract already gives free O(1) addressing (`counts[code]`). Literalizing this seed would make things *slower* than naive, not faster, and it doesn't cleanly attack any of the four listed naive assumptions in a helpful direction. Rejected.

**Seed 3 — the tally-notch and the kindling pile**

| World object | Problem object |
|---|---|
| foot-notch | `counts[code]++` |
| duplicate discarded to kindling | the code isn't stored again, only the counter changes |

This just restates "counting = increment a counter, don't store duplicates" — which is exactly what naive's `counts[code]++` already does. It breaks none of the four listed assumptions differently from naive. Rejected as a distinguishing target.

# CHOSEN SEED
Seed 1 (the sliding oar-frame). It is the most literal (every clause maps onto one concrete rolling-code operation) and the most different from the known way (it removes the dominant O(k) inner loop entirely, rather than just renaming the same work).

# ASSUMPTION BROKEN
"Each k-mer's code must be built fully from scratch, one base at a time, before it can be counted." Also weakens "one window is examined, then discarded, before the next begins" — the window's *state* now persists across positions; only 2 bits leave and 2 bits enter per step.

# ARTIFACT
Object mapping used literally in code: bead = base (`lut[seq[i]]`, 2 bits); oar-frame = the `code` register (2k bits, held in a `uint64_t`); mast-pegs = the bit-positions inside `code`; sliding the frame = `code = ((code<<2)|new)&mask`; tally-notch = `counts[code]++` / atomic add; courtyard = the `counts[]` array itself (addressed directly, so "matching" is free). Multiple crews (OpenMP threads) each work their own stretch of dock rail (contiguous range of window-start positions), each building only its own first oar-frame from scratch, then sliding it to the end of its stretch; their tallies are added together at the end (thread-local courtyards) when that's cheap, or notched directly with an atomic when the courtyard is too big to duplicate per crew.

```c
#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Every "bead" (base) is read once into a 2-bit color via this peg-table. */
static void build_lut(unsigned char lut[256]) {
    for (int i = 0; i < 256; i++) lut[i] = 3; /* default: dent-poison / T */
    lut[(unsigned char)'A'] = 0; /* shell */
    lut[(unsigned char)'C'] = 1; /* flower-plant */
    lut[(unsigned char)'G'] = 2; /* plank-grain */
    lut[(unsigned char)'T'] = 3; /* dent-poison */
}

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    unsigned char lut[256];
    build_lut(lut);

    const long long total_windows = (long long)n - (long long)k + 1;
    const uint64_t mask      = (k >= 32) ? ~0ULL : ((1ULL << (2 * k)) - 1);
    const uint64_t num_codes = (k >= 32) ? ~0ULL : (1ULL << (2 * k));

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if ((long long)nthreads > total_windows) nthreads = (int)total_windows;
    if (nthreads < 1) nthreads = 1;
#endif

    /* Give each crew its own courtyard (thread-local histogram) only when
       that courtyard is cheap enough to build (<=32MB); otherwise share
       one courtyard and use an atomic tally-notch. */
    int use_local = (nthreads > 1) && (num_codes <= (1ULL << 22));

#ifdef _OPENMP
    if (use_local) {
        uint64_t *local = (uint64_t *)calloc((size_t)nthreads * (size_t)num_codes, sizeof(uint64_t));
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            uint64_t *my = local + (size_t)tid * (size_t)num_codes;
            long long chunk = (total_windows + nthreads - 1) / nthreads;
            long long start = (long long)tid * chunk;
            long long end = start + chunk;
            if (end > total_windows) end = total_windows;
            if (start < end) {
                uint64_t code = 0;
                for (int j = 0; j < k; j++)
                    code = (code << 2) | lut[(unsigned char)seq[start + j]];
                my[code]++;
                for (long long i = start + 1; i < end; i++) {
                    code = ((code << 2) | lut[(unsigned char)seq[i + k - 1]]) & mask;
                    my[code]++;
                }
            }
        }
        for (int t = 0; t < nthreads; t++) {
            uint64_t *my = local + (size_t)t * (size_t)num_codes;
            for (uint64_t c = 0; c < num_codes; c++) counts[c] += my[c];
        }
        free(local);
        return;
    }

    if (nthreads > 1) {
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            long long chunk = (total_windows + nthreads - 1) / nthreads;
            long long start = (long long)tid * chunk;
            long long end = start + chunk;
            if (end > total_windows) end = total_windows;
            if (start < end) {
                uint64_t code = 0;
                for (int j = 0; j < k; j++)
                    code = (code << 2) | lut[(unsigned char)seq[start + j]];
                __atomic_fetch_add(&counts[code], 1, __ATOMIC_RELAXED);
                for (long long i = start + 1; i < end; i++) {
                    code = ((code << 2) | lut[(unsigned char)seq[i + k - 1]]) & mask;
                    __atomic_fetch_add(&counts[code], 1, __ATOMIC_RELAXED);
                }
            }
        }
        return;
    }
#endif

    /* Single oar-frame, single strand-walk: build the first window from
       scratch once, then slide it one bead at a time for the rest. */
    uint64_t code = 0;
    for (int j = 0; j < k; j++) code = (code << 2) | lut[(unsigned char)seq[j]];
    counts[code]++;
    for (long long i = 1; i < total_windows; i++) {
        code = ((code << 2) | lut[(unsigned char)seq[i + k - 1]]) & mask;
        counts[code]++;
    }
}
```

PREDICTION: speedup_vs_naive = 18

# MEASUREMENT
`kmer_bench`/`kmer_contract` are not present among the tools actually available to me in this session (my tool set here is limited to Docs/Gmail/Drive/Slack), matching the note in the prompt that no measurement tools are available this turn. I have not executed or benchmarked this kernel — the reasoning and code above are offered for the external pipeline to compile and measure; I have not silently substituted a "measured" number for the prediction.

# VERDICT
Predicted, not verified. Expectation: for the naive kernel the per-window cost is O(k) (a full re-encode), while the rolling window collapses that to O(1) amortized (one shift/or/and), so the single-thread algorithmic factor alone should scale roughly with k (bigger for larger k-mers, negligible for k≈1–2). On top of that, splitting the strand into contiguous per-thread stretches (each rebuilding only its own first window) should multiply by close to the core count when the histogram is small enough to keep thread-local (avoiding atomic contention), and by a smaller, contention-limited factor when it must fall back to atomics for large k. 18x reflects a mid-size k (roughly 8–16) on a multi-core machine; the true number should be reported by `kmer_bench`, not assumed — if `k` is very small or `n` is small enough that chunk-setup and thread-launch overhead dominate, the real speedup could come in well under this prediction, and that should be reported plainly rather than reconciled after the fact.