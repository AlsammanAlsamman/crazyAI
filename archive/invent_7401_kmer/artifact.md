Since this task is pure algorithm design/coding (no HPC cluster, no doc requested), I'll go straight to the work. No shell/bash tool is exposed in this session, so I can't check for a `brainny` CLI here — skipping that silently per the standing instruction (nothing to report).

## MAPPING

**SEED 1 — the sliding window of k eyes**

| World object | Problem object |
|---|---|
| rail of iron pegs, one notch per bead | `seq[0..n-1]`, one array slot per base |
| bead resting in a notch | a single base (A/C/G/T) at one position |
| frame with exactly k open eyes, nailed still | a 2k-bit register `code` holding the current k-mer, width fixed at compile/run start |
| right side pulling a new bead in | `code = (code<<2) \| base_code(new_base)` |
| left side spitting the oldest bead into the trough | `code &= mask` — the bits beyond position k*2 simply fall off the top, the old base is never looked at again |
| beads in the trough burnt, no memory kept of the strand | no re-read of any base once it has scrolled out of the k-wide register |

Assumption broken: *"each k-mer's code must be built fully from scratch, one base at a time, before it can be counted"* and *"every base contributes independently to the window it starts."* Here each base is read from the rail exactly once, ever, and slides through every window it belongs to for free.

**SEED 2 — the carved board of hooks**

| World object | Problem object |
|---|---|
| board of hooks, one per 4-bead combination across k slots | `counts[]`, one slot per code in `[0, 4^k)` |
| carved once, standing still | the array is allocated & zeroed once, addressed directly, never rebuilt/rehashed/resorted |

Assumption broken: none new — this is just a restatement of the contract's own `counts` array (direct-indexed table), which naive already assumes. Least novel of the three.

**SEED 3 — the blind knot and the trough-timed reading**

| World object | Problem object |
|---|---|
| a walker's own cord hanging on a hook | one thread's private tally for one code |
| knot tied blind, by feel, never touched again | a non-atomic, write-only increment — never read back mid-scan |
| "I do not read a hook until the whole strand has passed" | no thread ever inspects/merges partial counts during the scan |
| walking the board once at the end, hook by hook | one final reduction pass, summing every thread's private board into the real `counts[]` |

Assumption broken: *"the count table can only be updated one position's result at a time."* This licenses each parallel walker to keep its own private board and update it contention-free, deferring all reading to a single end-of-strand reduction.

## CHOSEN SEED

**Seed 1** — most literal (bead↔base, window↔register, trough↔discarded high bits map one-to-one with no leftover metaphor) and most different from the known way, since it directly collapses the naive method's O(n·k) re-encoding into O(n). Seed 3 is folded in as the natural way to let several walkers work the same rail at once without fighting over the board — it doesn't change the core artifact, it lets it scale.

## ASSUMPTION BROKEN

Primarily: *"each k-mer's code must be built fully from scratch, one base at a time"* and *"every base contributes independently to the window it starts."* Secondarily (via Seed 3, used only for the parallel extension): *"the count table can only be updated one position's result at a time."*

## ARTIFACT

```c
#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline int base_code(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        default:  return 3; /* 'T' and anything else */
    }
}

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    const long long num_windows = (long long)n - k + 1;
    if (num_windows <= 0) return;

    const int bits = 2 * k;
    const uint64_t mask = (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
    const uint64_t num_buckets = (bits >= 64) ? 0ULL : (1ULL << bits);

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    if (max_threads < 1) max_threads = 1;

    /* Cap total private-board memory to ~512MB so large k can't blow up RAM. */
    uint64_t bytes_per_board = (bits < 64) ? num_buckets * (uint64_t)sizeof(uint64_t)
                                            : ((uint64_t)1 << 62);
    long long budget_threads = (long long)((512ULL * 1024 * 1024) /
                                            (bytes_per_board ? bytes_per_board : 1ULL));
    if (budget_threads < 1) budget_threads = 1;

    int nthreads = max_threads;
    if ((long long)nthreads > budget_threads) nthreads = (int)budget_threads;
    if ((long long)nthreads > num_windows) nthreads = (int)(num_windows < 1 ? 1 : num_windows);
    if (nthreads < 1) nthreads = 1;

    if (nthreads == 1 || bits >= 64) {
        /* One rail, one window, one board: pure rolling update. */
        uint64_t code = 0;
        for (int j = 0; j < k; j++) code = (code << 2) | (uint64_t)base_code(seq[j]);
        counts[code & mask]++;
        for (long long i = 1; i < num_windows; i++) {
            code = ((code << 2) | (uint64_t)base_code(seq[i + k - 1])) & mask;
            counts[code]++;
        }
        return;
    }

#ifdef _OPENMP
    uint64_t **boards = (uint64_t **)malloc(sizeof(uint64_t *) * (size_t)nthreads);
    for (int t = 0; t < nthreads; t++)
        boards[t] = (uint64_t *)calloc((size_t)num_buckets, sizeof(uint64_t));

    #pragma omp parallel num_threads(nthreads)
    {
        int t = omp_get_thread_num();
        uint64_t *board = boards[t];

        long long chunk = (num_windows + nthreads - 1) / nthreads;
        long long start = (long long)t * chunk;
        long long end = start + chunk;
        if (end > num_windows) end = num_windows;

        if (start < end) {
            uint64_t code = 0;
            for (int j = 0; j < k; j++) code = (code << 2) | (uint64_t)base_code(seq[start + j]);
            board[code & mask]++;
            for (long long i = start + 1; i < end; i++) {
                code = ((code << 2) | (uint64_t)base_code(seq[i + k - 1])) & mask;
                board[code]++;
            }
        }
    }

    /* Walk every board, hook by hook, exactly once, after the strand is spent. */
    for (uint64_t code = 0; code < num_buckets; code++) {
        uint64_t sum = 0;
        for (int t = 0; t < nthreads; t++) sum += boards[t][code];
        counts[code] = sum;
    }

    for (int t = 0; t < nthreads; t++) free(boards[t]);
    free(boards);
#endif
}
```

Reasoning-only iteration log (no measurement tool was reachable in this session, so these are design passes, not benchmarked ones): v0 = single rolling window, no parallelism → removes the O(k) re-encode. v1 (final, above) = added OpenMP with private per-thread boards + a memory budget guard, since naive read-modify-write into one shared 4^k table would otherwise force atomics or serialize threads.

PREDICTION: speedup_vs_naive = 12

## MEASUREMENT

Not performed — `kmer_bench` and the other tools were not reachable in this session (explicitly disabled per the task's runtime note). The prediction above is therefore an estimate only: naive spends O(n·k) work re-deriving each code from scratch, this kernel spends O(n) for encoding plus a final O(4^k · nthreads) reduction; on top of the algorithmic ~k-fold reduction in encoding work, multi-core scaling is expected to be partial because `counts[code]++` is a random-access, memory-bound operation regardless of version, capping realized speedup below the naive product of (k × cores). The actual number must come from the external pipeline compiling and running this kernel against `kmer_contract`.

## VERDICT

Unverified. The artifact is a literal, honest translation of Seed 1 (rolling bead through a fixed window) with Seed 3's blind-knot/private-board idea used only to make that translation safe under parallelism — it is not silently swapped back to the textbook rolling-hash writeup, though the resulting code is admittedly close to what a rolling-hash k-mer counter looks like, because that *is* what "slide the window, drop the old bead, never reread it" literally compiles to. The stated speedup is a prediction, not a measurement, and should be reported as such until the pipeline actually runs `kmer_bench`.