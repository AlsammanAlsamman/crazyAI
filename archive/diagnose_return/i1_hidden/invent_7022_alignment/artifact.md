# MAPPING

**SEED 1 — "Two knot-cords laid crosswise over a sand-tray mark every place their symbols meet."**

| World object | Problem object |
|---|---|
| north cord, "the horse's length" | string `a`, indexed `i = 1..n` |
| east cord, "the mountain's width" | string `b`, indexed `j = 1..n` |
| sand-tray | the `(n+1)×(n+1)` DP lattice |
| furrow-point / junction | cell `(i,j)` |
| symbol facing symbol | `s(i,j) = (a[i-1]==b[j-1]) ? +1 : -1` |
| height of mound at a junction | `H(i,j)` |
| near corner / far corner | `H(0,0)=0` / `H(n,n)` = answer |

**SEED 2 — "A single ink-worm walks the eight desert winds and curls back across its own trail to test a junction twice."**

| World object | Problem object |
|---|---|
| ink-worm, **one cell wide**, a *stitch* (long body, unit width) | one SIMD register: 1 lane deep, 16 (AVX2) / 32 (AVX-512) cells long |
| the worm's body laid on the tray | a contiguous run of one **anti-diagonal** `i+j = d` |
| "hooded memory kneeling at each crossing in turn" (rejected) | the scalar `for i { for j }` cell-at-a-time DP |
| walking the winds | advancing `d → d+1`: the whole body moves one step SE |
| *curling back across ground already crossed* | re-reading its **own previous body** at lane offset ±1 — the N, W, NW predecessors are three offset loads of diagonals `d-1`, `d-1`, `d-2` |
| "tests a junction from another face" | the three-way `max` |
| "keeps no ledger apart from the sand"; beaten mound thrown away | only 3 rolling diagonals are kept; no matrix, no traceback |
| the whole trail, never lifted, never broken into separate glances | one unbroken wavefront sweep, `d = 0 … 2n` |

**SEED 3 — "The worm crosses a furrow sideways, spending no sand, to let one cord slip a step ahead of the other."**

| World object | Problem object |
|---|---|
| sideways crossing, spending nothing | letting the `a`-pointer and `b`-pointer differ by one index — a **base-address offset**, which costs zero instructions |
| one cord slipping ahead | `i` advances while `j` does not (the gap move) |
| "the one permitted stumble" | `-2` gap penalty branch of the recurrence |

# CHOSEN SEED

**Seed 2.** It is the only one whose objects are *shaped*: the worm is explicitly **one cell wide** and yet a *body* — that is a vector register, not a pointer. Seeds 1 and 3 describe data layout and one arithmetic branch; Seed 2 describes the processor. Taking "one worm, one cell wide, whose whole body settles at once" literally forces the anti-diagonal wavefront, because that is the only orientation on this lattice along which a one-cell-wide body has no internal dependency — every junction under the body can be judged in the same instant.

Literal mapping of the machine:
- **memory** = the sand: three rolling `int16` diagonals (`D2`,`D1`,`D0`) plus a padded copy of `a` and a **reversed** copy of `b` (reversing `b` makes both symbol-cords run the same way under the worm's body, so both are sequential byte loads).
- **what flows** = the worm (the vector), sweeping `d = 0..2n`.
- **what stays still** = the two cords (the sequences) and the sand already dropped.
- **the processor** = the worm's body; one instruction settles 16/32 junctions.
- **time** = the diagonal index `d`; exactly `2n+1` ticks, and never two glances at one junction.

# ASSUMPTION BROKEN

> "One pair of positions is judged at a time."

Broken literally: at tick `d` the worm settles **16 (AVX2) or 32 (AVX-512BW) distinct `(i,j)` pairs simultaneously**, and the gap move ("crossing sideways, spending nothing") is implemented as a *pointer offset*, not an operation — so the "stumble" costs zero instructions, exactly as the native says it costs no sand.

# ARTIFACT

```c
/* Ink-worm wavefront Needleman-Wunsch score.
   match=+1, mismatch=-1, gap=-2, global, equal-length.
   Contract: int kernel(int n, const char *a, const char *b); */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__SSE2__) || defined(__AVX2__) || defined(__AVX512BW__)
#include <immintrin.h>
#endif

#define NW_PAD 64   /* slack so the worm's body may overhang the live range */

/* ---- exact row DP: used for tiny n (setup would dominate) and as a safety net ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int sbuf[1026];
    int *base, *prev, *cur, *t;
    int i, j, r;
    if (n <= 0) return 0;
    if ((long)(n + 1) * 2 <= 1026) base = sbuf;
    else base = (int *)malloc((size_t)(n + 1) * 2 * sizeof(int));
    if (!base) return 0;
    prev = base; cur = base + (n + 1);
    for (j = 0; j <= n; ++j) prev[j] = -2 * j;
    for (i = 1; i <= n; ++i) {
        const char ai = a[i - 1];
        cur[0] = -2 * i;
        for (j = 1; j <= n; ++j) {
            int s = prev[j - 1] + ((ai == b[j - 1]) ? 1 : -1);
            int u = prev[j] - 2;
            int l = cur[j - 1] - 2;
            if (u > s) s = u;
            if (l > s) s = l;
            cur[j] = s;
        }
        t = prev; prev = cur; cur = t;
    }
    r = prev[n];
    if (base != sbuf) free(base);
    return r;
}

/* ---- 32-bit wavefront: only for n so large that 16-bit headroom (|H| <= 2n) fails ---- */
static int nw_diag32(int n, const char *a, const char *b)
{
    const int m = n + NW_PAD;
    int32_t *buf, *D2, *D1, *D0, *t;
    char *A, *Br;
    int d, i, res;
    void *mem = malloc((size_t)3 * m * sizeof(int32_t) + (size_t)2 * m + 64);
    if (!mem) return nw_rows(n, a, b);
    buf = (int32_t *)mem;
    memset(buf, 0, (size_t)3 * m * sizeof(int32_t));
    D2 = buf; D1 = buf + m; D0 = buf + 2 * m;
    A = (char *)(buf + 3 * m); Br = A + m;
    memcpy(A, a, (size_t)n); memset(A + n, 1, NW_PAD);
    for (i = 0; i < n; ++i) Br[i] = b[n - 1 - i];
    memset(Br + n, 2, NW_PAD);
    D0[0] = 0;
    t = D2; D2 = D1; D1 = D0; D0 = t;
    for (d = 1; d <= 2 * n; ++d) {
        int lo, hi; const int k0 = n - d;
        if (d <= n) { lo = 1; hi = d - 1; } else { lo = d - n; hi = n; }
        for (i = lo; i <= hi; ++i) {
            int s = D2[i - 1] + ((A[i - 1] == Br[i + k0]) ? 1 : -1);
            int u = D1[i - 1] - 2;
            int l = D1[i] - 2;
            if (u > s) s = u;
            if (l > s) s = l;
            D0[i] = s;
        }
        if (d <= n) { D0[0] = -2 * d; D0[d] = -2 * d; }
        t = D2; D2 = D1; D1 = D0; D0 = t;
    }
    res = (int)D1[n];
    free(mem);
    return res;
}

/* ---- the worm: 16-bit wavefront ---- */
static int nw_diag16(int n, const char *a, const char *b)
{
    const int m = n + NW_PAD;
    int16_t *buf, *D2, *D1, *D0, *t;
    char *A, *Br;
    int d, i, res;
    void *mem = malloc((size_t)3 * m * sizeof(int16_t) + (size_t)2 * m + 64);
    if (!mem) return nw_rows(n, a, b);
    buf = (int16_t *)mem;
    memset(buf, 0, (size_t)3 * m * sizeof(int16_t));
    D2 = buf; D1 = buf + m; D0 = buf + 2 * m;
    A = (char *)(buf + 3 * m); Br = A + m;
    /* both cords laid so the worm's body reads them in the same grain */
    memcpy(A, a, (size_t)n); memset(A + n, 1, NW_PAD);
    for (i = 0; i < n; ++i) Br[i] = b[n - 1 - i];
    memset(Br + n, 2, NW_PAD);

    D0[0] = 0;                                   /* near corner */
    t = D2; D2 = D1; D1 = D0; D0 = t;

    for (d = 1; d <= 2 * n; ++d) {
        int lo, hi; const int k0 = n - d;
        if (d <= n) { lo = 1; hi = d - 1; } else { lo = d - n; hi = n; }
        if (hi >= lo) {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            const __m512i vp1 = _mm512_set1_epi16(1);
            const __m512i vm1 = _mm512_set1_epi16(-1);
            const __m512i v2  = _mm512_set1_epi16(2);
            for (i = lo; i <= hi; i += 32) {
                __m256i ca = _mm256_loadu_si256((const __m256i *)(A + (i - 1)));
                __m256i cb = _mm256_loadu_si256((const __m256i *)(Br + (i + k0)));
                __mmask32 kq = _mm256_cmpeq_epi8_mask(ca, cb);
                __m512i sc = _mm512_mask_blend_epi16(kq, vm1, vp1);
                __m512i dg = _mm512_add_epi16(
                        _mm512_loadu_si512((const void *)(D2 + (i - 1))), sc);
                __m512i up = _mm512_loadu_si512((const void *)(D1 + (i - 1)));
                __m512i le = _mm512_loadu_si512((const void *)(D1 + i));
                __m512i gp = _mm512_sub_epi16(_mm512_max_epi16(up, le), v2);
                _mm512_storeu_si512((void *)(D0 + i), _mm512_max_epi16(dg, gp));
            }
#elif defined(__AVX2__)
            const __m256i vm1 = _mm256_set1_epi16(-1);
            const __m256i v2  = _mm256_set1_epi16(2);
            for (i = lo; i <= hi; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(A + (i - 1)));
                __m128i cb = _mm_loadu_si128((const __m128i *)(Br + (i + k0)));
                /* -1 where the symbols agree, 0 where they do not */
                __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                /* +1 on agreement, -1 otherwise */
                __m256i sc = _mm256_sub_epi16(vm1, _mm256_add_epi16(eq, eq));
                __m256i dg = _mm256_add_epi16(
                        _mm256_loadu_si256((const __m256i *)(D2 + (i - 1))), sc);
                __m256i up = _mm256_loadu_si256((const __m256i *)(D1 + (i - 1)));
                __m256i le = _mm256_loadu_si256((const __m256i *)(D1 + i));
                __m256i gp = _mm256_sub_epi16(_mm256_max_epi16(up, le), v2);
                _mm256_storeu_si256((__m256i *)(D0 + i), _mm256_max_epi16(dg, gp));
            }
#else
            for (i = lo; i <= hi; ++i) {   /* auto-vectorizable: no carried dep */
                int s = D2[i - 1] + ((A[i - 1] == Br[i + k0]) ? 1 : -1);
                int u = D1[i - 1] - 2;
                int l = D1[i] - 2;
                if (u > s) s = u;
                if (l > s) s = l;
                D0[i] = (int16_t)s;
            }
#endif
        }
        /* the two tray edges, written after the body so overhang cannot spoil them */
        if (d <= n) { D0[0] = (int16_t)(-2 * d); D0[d] = (int16_t)(-2 * d); }
        t = D2; D2 = D1; D1 = D0; D0 = t;
    }
    res = (int)D1[n];                            /* far corner */
    free(mem);
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 24)     return nw_rows(n, a, b);     /* guard: setup would dominate */
    if (n <= 12000) return nw_diag16(n, a, b);   /* guard: |H| <= 2n < 32767 */
    return nw_diag32(n, a, b);                   /* guard: 16-bit headroom gone */
}
```

Why it is exact, not approximate: it is the full `n²` Needleman–Wunsch lattice, merely traversed in anti-diagonal order instead of row order. No band, no heuristic, no early exit. The three predecessors of a live cell `(i,j)` — `D2[i-1]`, `D1[i-1]`, `D1[i]` — are always live cells of diagonals `d-2`, `d-1`, `d-1`, so the deliberate body-overhang past `hi` (which writes junk into slots outside the live range) can never feed a live cell; the two edge cells are rewritten after the body passes.

# PREDICTION

PREDICTION: speedup_vs_dp = 10.0

Reasoning stated before measuring: the reference row DP carries a `add → max` dependency through `cur[j-1]`, so it cannot beat ≈2.5–3 cycles/cell. The wavefront body is 5 loads + 1 store + ~8 ALU ops per 16 cells ⇒ load-port bound at ≈4 cycles/16 cells ≈ 0.25 cycles/cell, plus ~20n cycles of per-diagonal bookkeeping (≈8% at n=1000). 2.75/0.28 ≈ 10. On an AVX-512BW machine I would expect 15–18.

# MEASUREMENT

**Not performed — reported plainly as a failure of this run, not glossed.** `alignment_bench` and `alignment_contract` were not reachable in this session (no tools were exposed), so every number above is an a-priori estimate and nothing here has been compiled, checked against the reference, or timed. The prediction stands as written and un-tuned; I did not get the four allowed improvement rounds, so what is shipped is round 0.

What I would measure first, in order: (1) exact-match against the reference over random and adversarial inputs (all-equal, all-different, `n` = 24, 25, 31, 32, 33, 12000, 12001) — a mismatch here invalidates everything; (2) cycles/cell at n = 256, 1024, 4096 to see where the per-diagonal bookkeeping stops mattering; (3) whether the AVX-512 branch actually compiled in.

# VERDICT

The native's image survives translation almost without residue, and two things fell out of taking it literally that I would not have written otherwise:

1. **"One cell wide" forced the anti-diagonal.** A one-lane-deep body has zero internal dependency only when laid across `i+j = d`. Row-order DP is the "hooded memory kneeling at each crossing in turn" the native explicitly rejects — and it is exactly the shape that cannot be vectorized without Farrar's lazy-F patch.
2. **"Crosses sideways, spending no sand" became a pointer offset.** Reversing `b` so both cords run with the worm's grain turns the gap move into a difference of base addresses. The stumble genuinely costs zero instructions, which is a stronger statement than the native's metaphor even required.

Where I did **not** follow the native, and say so: he describes a worm that *doubles back in time* — revisiting junctions, kicking mounds flat, finishing only when no junction anywhere can be improved. That is Bellman–Ford relaxation to a fixpoint, and it would be several times slower than one sweep. I realized the doubling-back **spatially instead of temporally**: the worm re-reads its own trail at lane offsets ±1 (the three loads), and the anti-diagonal order already delivers the fixpoint in a single pass because the lattice is a DAG. This is a deliberate substitution; a reader who wanted the literal iterative worm would get a correct but much slower kernel.

Stated risks, each guarded in the shipped code rather than left as prose:
- **16-bit saturation.** `|H| ≤ 2n`, so the int16 worm is only used for `n ≤ 12000`; beyond that a 32-bit wavefront takes over.
- **Small-`n` overhead.** The wavefront pays one `malloc`, two `O(n)` cord copies and `2n` diagonal setups; below `n = 24` that dominates `n²`, so a stack-buffer row DP is used instead.
- **No AVX2 on the target.** The `#else` branch is a dependency-free scalar loop GCC vectorizes on its own; it never fails to compile.
- **OpenMP deliberately omitted.** A thread-parallel wavefront needs a barrier per diagonal — `2n` barriers at ~1 µs each is ~8 ms at n=4096 against ~2 ms of serial work, i.e. a guaranteed loss. Making it pay requires tiled wavefronts, which I did not have measurement budget to validate, so I shipped nothing rather than ship an unmeasured mechanism whose own arithmetic says it loses.

The honest headline: the mechanism is sound and the mapping is faithful, but the speedup number is a prediction that nobody has yet held a stopwatch to.