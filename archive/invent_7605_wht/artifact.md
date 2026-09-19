# MAPPING

| World object (native's ritual) | Problem object (WHT kernel) | Assumption it strains |
|---|---|---|
| **SEED 1** — bone-colored plank | the input buffer `in[0..n-1]` in memory | — |
| curve laid along the plank | sequence values, indexed in natural (unpermuted) order | — |
| tree of dried genders asking a round-side-up question | the recursive binary (bit 0/1) decomposition of an index | — |
| places where that tree stops | the `n` leaves of that decomposition = the `n` array positions | — |
| nails, "they do not move again" | the indices `j = 0..n-1`, fixed for the whole run, never bit-reversed or permuted | affects (d) — but *affirms* it rather than breaking it: order really is never rearranged |
| **SEED 2** — token-room | one output index `k`, `0 ≤ k < n` | |
| "cluster of nails" a room stands for | the bit-pattern of `k` itself, read as which nails are claimed | |
| "every possible cluster… empty to full" | `k` ranges exhaustively `0..n-1` | breaks (c)-adjacent framing: rooms are not opaque black boxes, they are a *lattice* over the same fixed nail set — the raw material for reuse exists |
| "throw the room's flood-water out and never revisit it" | no state persists between iterations of `k` | *affirms* (b): the ritual explicitly refuses the reuse its own lattice structure would allow |
| **SEED 3** — token | the multiplier applied to `in[j]` inside room `k` | |
| sun-side-up | `+1` (nail `j` belongs to `k`'s cluster: `popcount(j&k)` even) | |
| iron-blade-down, winter-cold | `-1` (doesn't belong: `popcount(j&k)` odd) | |
| "never half-turned, never warming toward kemmer" | the multiplier is a hard `±1`, no fractional/interpolated weight, chosen per (nail,room) pair by table/hardware lookup, not accumulated bit-by-bit | **breaks (e)**: subtraction is a first-class, directly-selected sign, not something built out of a chain of ordinary additions and a separate special-cased subtract |
| "flood… settles flat" | the multiply-accumulate reduction over `j` completes | |
| "score that single number… before it dries" | `out[k]` written exactly once | |

# CHOSEN SEED

SEED 3 — the sun-side-up / iron-blade-down token rule.

It needs no interpretive stretching (the mapping token→±1 is exact, not analogical), and it is the most different in *shape* from the known way: the FWHT butterfly never materializes a full sign for a (nail, room) pair — it gets subtraction "for free" as the second half of a paired add/subtract across `log n` recursive stages, reusing each partial sum for two future outputs. SEED 3's ritual instead insists on an independent, explicit, freshly-chosen token for every single nail in every single room, discarded the moment that room is scored. That is the literal opposite of butterfly reuse.

# ASSUMPTION BROKEN

(e) "combining two contributions always means an ordinary addition." The naive reference buries the sign inside a slow, branchy `popcount` loop that is logically just computing "should this be added or subtracted" the hard way. SEED 3 makes that choice a single, hardware-resolved token (`parity` via a POPCNT instruction) multiplied in directly — subtraction becomes a first-class per-element sign-select, not an emergent side effect of repeated addition.

Honesty check on the others: SEED 1 and SEED 2 supply the fixed-index/independent-room scaffolding around this, but SEED 2's own text ("never revisit") means assumption (b) is *not* actually broken by this artifact — the ritual, read literally, forbids the cross-output reuse that would buy asymptotic speedup. So this stays O(n²), by design of fidelity to the text, not by oversight.

# ARTIFACT

```c
#include <immintrin.h>
#include <omp.h>

void kernel(int n, const double *in, double *out) {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        __m256d vsum = _mm256_setzero_pd();
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            /* one token per nail, sun-side-up (+1) or iron-blade-down (-1),
               read directly off the hardware -- never a bit-by-bit count */
            int p0 = __builtin_parity((unsigned)((j + 0) & k));
            int p1 = __builtin_parity((unsigned)((j + 1) & k));
            int p2 = __builtin_parity((unsigned)((j + 2) & k));
            int p3 = __builtin_parity((unsigned)((j + 3) & k));
            double s0 = p0 ? -1.0 : 1.0;
            double s1 = p1 ? -1.0 : 1.0;
            double s2 = p2 ? -1.0 : 1.0;
            double s3 = p3 ? -1.0 : 1.0;

            __m256d vin   = _mm256_loadu_pd(&in[j]);
            __m256d vsign = _mm256_set_pd(s3, s2, s1, s0);
            /* let the flood settle: multiply-accumulate, not a running
               chain of separate additions */
            vsum = _mm256_add_pd(vsum, _mm256_mul_pd(vsign, vin));
        }

        double buf[4];
        _mm256_storeu_pd(buf, vsum);
        double sum = buf[0] + buf[1] + buf[2] + buf[3];

        for (; j < n; j++) {
            int p = __builtin_parity((unsigned)(j & k));
            sum += (p ? -1.0 : 1.0) * in[j];
        }

        /* score the single number onto the room, once, and never
           revisit it -- no state carried to the next k */
        out[k] = sum;
    }
}
```

Object accounting: plank → `in[]`; nails → the fixed loop variable `j`, never permuted; token-room → one iteration of the parallel `k` loop, independent and discarded (`omp parallel for`, no cross-`k` state); token → the `±1.0` selected per nail by `__builtin_parity`; "flood settles flat" → the `vsum`/`sum` reduction; "score before it dries" → the single write `out[k] = sum`.

# PREDICTION

Removing the naive reference's branchy `while`-loop `popcount` (up to `log2 n` iterations with data-dependent branches per element) in favor of one hardware parity instruction should cut per-element cost sharply on its own; 4-wide AVX2 accumulation adds a further partial factor on the reduction; `omp parallel for` over the `n` fully independent rooms adds a roughly core-count factor, capped by memory bandwidth since every room re-reads all of `in[]` (true O(n²) traffic, unchanged in complexity — the ritual's "never revisit" forbids removing that). I expect a solid constant-factor win, not an asymptotic one: it should *not* widen with growing `n` the way a real FWHT would.

PREDICTION: speedup_vs_naive = 12

# MEASUREMENT

No tool calls were available in this session (the harness explicitly disabled `wht_bench` here and instructed me to reason and hand off the artifact for the pipeline to compile/measure). I am not fabricating a number — this section is intentionally empty of an actual measured result. The prediction above was recorded before any measurement, as required, and stands to be checked against `wht_bench` by the pipeline.

# VERDICT

Not independently verified in this session — reported honestly as unmeasured rather than assumed correct. Structurally, the artifact is faithful to the text at the cost of staying O(n²): SEED 2/SEED 1 supply the raw lattice/fixed-index structure that *could* support the FWHT's cross-output reuse, but SEED 3's "throw the flood-water out and never revisit it" was taken literally and that reuse was deliberately not built in, so any real speedup here comes only from (1) replacing a branchy software popcount with one hardware parity op and (2) SIMD + multicore parallelism over genuinely independent rooms — a constant-factor engineering win, not the `log n` factor the known FWHT gets by recursively reusing partial sums. If the pipeline's measurement shows this speedup shrinking toward 1 at large `n` (memory-bandwidth bound, O(n²) traffic dominating), that would be the expected, honest signature of having followed the ritual's explicit no-reuse clause rather than quietly swapping in the butterfly.