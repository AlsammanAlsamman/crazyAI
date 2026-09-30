No tools were available in this session, so the measurement section reports that honestly rather than inventing numbers. Everything else is done in full.

---

# MAPPING

### SEED 1 — "I hang two boughs of symbols from crossed rows of lamp-posts so their meeting makes a lattice of spark-cells."

| World object | Problem object |
|---|---|
| bough (cut from a note-tree) | input string `a` / input string `b`, length `n` |
| row of lamp-posts running **down** | row index `i` = position in `a`; `a` held stationary along one axis |
| row of lamp-posts running **across** | column index `j` = position in `b` |
| the lattice of unlit windows between them | the conceptual NW matrix `dp[i][j]` — *never materialized* |
| a spark at one crossing | the score value `dp[i][j]` |
| "I light each cell only when I need it" | lazy, per-wavefront materialization; no `(n+1)²` allocation |
| "the two symbols crossing choose the color, not me" | substitution score is a pure function of `a[i-1]==b[j-1]`, i.e. data-driven, branch-free |
| three colors: match / slip-out / slip-in | the three candidates: diagonal+s, up+GAP, left+GAP |

**Silent assumption broken:** *"the whole grid of every position against every other must be filled in."* The lattice exists as a coordinate system, not as storage.

### SEED 2 — "I fold the lattice-sheet along its slanting middle so cells of equal row-and-column sum press flush into a single crease."

| World object | Problem object |
|---|---|
| the slanting middle / corner-to-corner fold | the anti-diagonal direction `i + j = d` |
| a crease | one anti-diagonal: the set `{(i, d-i)}` |
| "cells of equal row-and-column sum press flush" | all cells with the same `i+j` are **mutually independent** and become one vector register's worth of lanes |
| "the note is already folded in the hour between the note-trees" | the dependency cone of NW is already anti-diagonal; the fold is not imposed, it is *discovered* |
| folding forward, crease after crease | wavefront traversal, `d = 2 … 2n` |
| the last crease holding a single spark at the far corner | `d = 2n` has exactly one cell, `dp[n][n]` = the answer |
| the spark's one color at the end | the returned global alignment score |

**Silent assumption broken:** ✅ ***"every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order."*** The fold destroys the *order* while keeping the *dependencies*: along a crease there is no above/left neighbour at all, so every cell of a crease is computed in the same instant. It simultaneously breaks *"one pair of positions is judged at a time."*

### SEED 3 — "I read three stacked layers through each crease at once and let the spent bottom layer drift off before folding the next crease forward."

| World object | Problem object |
|---|---|
| three layers pressed into one thickness | three rolling buffers: diagonals `d`, `d-1`, `d-2` |
| "the layer beneath tells me what agreeing straight through would have cost" | diagonal `d-2` supplies `dp[i-1][j-1]` (the match/mismatch predecessor) |
| "the layer beside it tells me what one slip would have cost" | diagonal `d-1` supplies **both** `dp[i-1][j]` and `dp[i][j-1]` — one buffer, offsets `i-1` and `i` |
| "I read all three at once, not walking them one after another" | SIMD: one load per layer, 16 or 32 cells resolved per instruction |
| "the spent bottom layer drifts off into the gradient" | the `d-2` buffer is recycled as the `d+1` buffer — **O(n) memory, not O(n²)** |
| "the note has no ground to store it on" | no DP table exists to be written to DRAM; the whole working set is ~6n bytes, L1-resident |
| "I am not a forest that hoards its own marsh" | explicit refusal of the traceback-capable full table; only the score is wanted |
| "I wait only as long as it takes each spark to catch from its neighbors, never longer" | dependency-only synchronisation — no barriers, no thread fork/join per crease; SIMD lanes, not OpenMP |

**Silent assumption broken:** *"the whole grid must be filled in"* and *"a slip can only be discovered by having already compared the position before it"* — the slip-costs arrive from the **same** crease layer as everything else, in parallel, not sequentially.

### The reversal, which falls out of SEED 1 being taken literally

The native hangs one bough **down** and one **across** from *crossed* rows of posts. Walking a crease, `i` rises while `j` falls — so `a` is read forward and `b` is read *backward*. Taking that literally: store `b` reversed, `br[k] = b[n-1-k]`. Then `b[j-1] = b[d-i-1] = br[n-d+i]`, which **increases with `i`**. Both symbol streams become unit-stride contiguous loads along the crease. This is the thing that makes the whole fold vectorize with two plain `loadu`s and one `cmpeq_epi8`, and it breaks *"both strings are read start to end in the same direction."*

# CHOSEN SEED

**SEED 2** — the fold along the slanting middle. It is the seed that breaks the preferred assumption (dependency *order*), it is the most literal (the fold **is** the anti-diagonal, the crease **is** the vector register), and it is the furthest from row-major Needleman–Wunsch. SEED 3 is not a separate choice but the fold's immediate physical consequence: once you fold, only three layers can ever be in contact, so the rest must be let go. SEED 1 supplies the crossed-posts reversal that makes the crease loadable.

# ASSUMPTION BROKEN

> **"every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order"**

The dependencies are kept *exactly* — this returns bit-identical NW scores — but the **order** is discarded. Row-major order has a loop-carried dependency through `dp[i][j-1]`, which is precisely why the reference inner loop cannot vectorize: cell `j` cannot start until cell `j-1` retires, ~4-cycle latency chain, one cell at a time forever. Anti-diagonal order has **zero** intra-crease dependency, so the entire crease is one straight-line SIMD expression.

Also broken: *"one pair of positions is judged at a time"* (16–32 pairs per instruction), *"the whole grid must be filled in"* (three O(n) buffers), and *"both strings are read start to end in the same direction"* (`b` is stored reversed so the crease is contiguous in both symbol streams).

### Literal machine mapping

| World | Machine |
|---|---|
| **memory** | the three crease buffers (`p2`, `p1`, `cu`), ~6n bytes, L1-resident; plus `ca`, `cb` symbol strips |
| **what flows** | the crease — the wavefront advancing across the lattice, `d = 2 … 2n` |
| **what stays still** | the two boughs (`ca` forward, `cb` reversed), written once, never moved |
| **a processor** | one SIMD lane = one spark-cell; 16 lanes (AVX2/int16) or 32 (AVX-512BW/int16) |
| **time** | `d`, the crease index — the only sequential axis that remains. `2n-1` ticks instead of `n²` |
| **letting a layer drift off** | pointer rotation `p2 ← p1 ← cu ← p2`; zero copying, zero freeing |
| **"never wait past the catching"** | no OpenMP, no barriers. A crease is `n/16` vector ops at benchmark sizes — far below fork/join cost. The metaphor forbids the wait and the hardware agrees. |

### Two regimes, recognised in-world at runtime

The known-way section names two regimes (plain O(n²) DP vs. a SIMD variant *"exploiting a bounded score range"*), so the native must feel both:

- **How bright can a spark get?** The total glow on any crease is bounded by `[-2n-2, n]`. If the boughs are short enough that no spark can outshine a half-width lamp (`n ≤ 16000` ⟹ `|score| < 32767`), the native uses **half-width lamps** — `int16`, twice as many sparks per crease-read. Otherwise **full-width lamps** — `int32`, 8 per read, still exact.
- **Is the crease even worth folding?** If a bough is shorter than a single spark-read (`n < 32`), the crease is mostly empty lanes and the folding itself costs more than it saves. The native then **does not fold at all** and walks flat — a plain rolling-row DP on stack buffers. This is the guard demanded by the verdict's own stated risk.
- **Are there lamps at all?** No AVX2 ⟹ the identical crease recurrence runs as a scalar loop; still O(n) memory, still correct.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  #include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* ============================================================
   REGIME 0 -- "the bough is shorter than one spark-read":
   do not fold at all, walk flat.  Plain rolling-row NW on the
   stack.  Guards the folding overhead at tiny n.            */
static int nw_flat(int n, const char *restrict a, const char *restrict b)
{
    int prev[40], cur[40];              /* n < 32  =>  n+1 <= 32 */
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        const char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int s = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int u = prev[j]     + GAP;
            int l = cur[j - 1]  + GAP;
            int best = s;
            if (u > best) best = u;
            if (l > best) best = l;
            cur[j] = best;
        }
        memcpy(prev, cur, (size_t)(n + 1) * sizeof(int));
    }
    return prev[n];
}

/* ============================================================
   REGIME 1 -- half-width lamps.  |score| <= 2n+2 < 32768.
   The fold: three creases pressed together, bottom one let go.
   cu[i] = max( p2[i-1] + s ,  max(p1[i-1], p1[i]) - 2 )
   with  s = (a[i-1] == b[d-i-1]) ? +1 : -1
   and   b stored REVERSED so b[d-i-1] == cb[n-d+i], unit stride. */
static int nw_fold16(int n, const char *a, const char *b)
{
    const int    N    = n;
    const size_t PAD  = 64;
    const size_t bufn = (size_t)N + 2 * PAD;

    int16_t *mem  = (int16_t *)malloc(3 * bufn * sizeof(int16_t));
    char    *cmem = (char    *)malloc(2 * ((size_t)N + 2 * PAD));
    if (!mem || !cmem) { free(mem); free(cmem); return nw_flat(0, a, b); }
    memset(mem, 0, 3 * bufn * sizeof(int16_t));

    int16_t *p2 = mem, *p1 = mem + bufn, *cu = mem + 2 * bufn;

    char *ca = cmem;                        /* bough A, forward  */
    char *cb = cmem + (size_t)N + 2 * PAD;  /* bough B, reversed */
    memcpy(ca, a, (size_t)N);
    memset(ca + N, 0x00, 2 * PAD);
    for (int k = 0; k < N; k++) cb[k] = b[N - 1 - k];
    memset(cb + N, 0x7f, 2 * PAD);          /* filler never matches */

    p2[0] = 0;                              /* crease d = 0 */
    p1[0] = (int16_t)GAP;                   /* crease d = 1 */
    p1[1] = (int16_t)GAP;

    for (int d = 2; d <= 2 * N; d++) {
        int ilo = d - N; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > N) ihi = N;
        const char *pb = cb + (N - d);      /* pb[i] == b[d-i-1] */

#if defined(__AVX512BW__) && defined(__AVX512F__)
        {
            const __m512i v2 = _mm512_set1_epi16(2);
            const __m512i v1 = _mm512_set1_epi16(1);
            for (int i = ilo; i <= ihi; i += 32) {
                __m512i pd = _mm512_loadu_si512((const void *)(p2 + i - 1));
                __m512i uu = _mm512_loadu_si512((const void *)(p1 + i - 1));
                __m512i ll = _mm512_loadu_si512((const void *)(p1 + i));
                __m256i av = _mm256_loadu_si256((const __m256i *)(ca + i - 1));
                __m256i bv = _mm256_loadu_si256((const __m256i *)(pb + i));
                __m256i eq = _mm256_cmpeq_epi8(av, bv);
                __m512i ss = _mm512_sub_epi16(
                                 _mm512_and_si512(_mm512_cvtepi8_epi16(eq), v2), v1);
                __m512i dg = _mm512_add_epi16(pd, ss);
                __m512i gp = _mm512_sub_epi16(_mm512_max_epi16(uu, ll), v2);
                _mm512_storeu_si512((void *)(cu + i), _mm512_max_epi16(dg, gp));
            }
        }
#elif defined(__AVX2__)
        {
            const __m256i v2 = _mm256_set1_epi16(2);
            const __m256i v1 = _mm256_set1_epi16(1);
            for (int i = ilo; i <= ihi; i += 16) {
                __m256i pd = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i uu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m128i av = _mm_loadu_si128((const __m128i *)(ca + i - 1));
                __m128i bv = _mm_loadu_si128((const __m128i *)(pb + i));
                __m128i eq = _mm_cmpeq_epi8(av, bv);
                __m256i ss = _mm256_sub_epi16(
                                 _mm256_and_si256(_mm256_cvtepi8_epi16(eq), v2), v1);
                __m256i dg = _mm256_add_epi16(pd, ss);
                __m256i gp = _mm256_sub_epi16(_mm256_max_epi16(uu, ll), v2);
                _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(dg, gp));
            }
        }
#else
        {
            const int16_t *restrict q2 = p2;
            const int16_t *restrict q1 = p1;
            int16_t       *restrict qc = cu;
            for (int i = ilo; i <= ihi; i++) {
                int s = (int)q2[i - 1] + ((ca[i - 1] == pb[i]) ? MATCH : MISMATCH);
                int u = (int)q1[i - 1] + GAP;
                int l = (int)q1[i]     + GAP;
                int best = s;
                if (u > best) best = u;
                if (l > best) best = l;
                qc[i] = (int16_t)best;
            }
        }
#endif
        if (d <= N) {                       /* the two rim sparks of this crease */
            cu[0] = (int16_t)(GAP * d);     /* dp[0][d] */
            cu[d] = (int16_t)(GAP * d);     /* dp[d][0] */
        }
        { int16_t *t = p2; p2 = p1; p1 = cu; cu = t; }  /* bottom layer drifts off */
    }

    int res = (int)p1[N];                   /* last crease, far corner */
    free(mem); free(cmem);
    return res;
}

/* ============================================================
   REGIME 2 -- full-width lamps.  Same fold, int32 sparks,
   8 per crease-read.  Used when a spark could outshine int16. */
static int nw_fold32(int n, const char *a, const char *b)
{
    const int    N    = n;
    const size_t PAD  = 64;
    const size_t bufn = (size_t)N + 2 * PAD;

    int32_t *mem  = (int32_t *)malloc(3 * bufn * sizeof(int32_t));
    char    *cmem = (char    *)malloc(2 * ((size_t)N + 2 * PAD));
    if (!mem || !cmem) { free(mem); free(cmem); return 0; }
    memset(mem, 0, 3 * bufn * sizeof(int32_t));

    int32_t *p2 = mem, *p1 = mem + bufn, *cu = mem + 2 * bufn;

    char *ca = cmem;
    char *cb = cmem + (size_t)N + 2 * PAD;
    memcpy(ca, a, (size_t)N);
    memset(ca + N, 0x00, 2 * PAD);
    for (int k = 0; k < N; k++) cb[k] = b[N - 1 - k];
    memset(cb + N, 0x7f, 2 * PAD);

    p2[0] = 0;
    p1[0] = GAP;
    p1[1] = GAP;

    for (int d = 2; d <= 2 * N; d++) {
        int ilo = d - N; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > N) ihi = N;
        const char *pb = cb + (N - d);

#if defined(__AVX2__)
        {
            const __m256i v2 = _mm256_set1_epi32(2);
            const __m256i v1 = _mm256_set1_epi32(1);
            for (int i = ilo; i <= ihi; i += 8) {
                __m256i pd = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i uu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m128i av = _mm_loadl_epi64((const __m128i *)(ca + i - 1));
                __m128i bv = _mm_loadl_epi64((const __m128i *)(pb + i));
                __m128i eq = _mm_cmpeq_epi8(av, bv);
                __m256i ss = _mm256_sub_epi32(
                                 _mm256_and_si256(_mm256_cvtepi8_epi32(eq), v2), v1);
                __m256i dg = _mm256_add_epi32(pd, ss);
                __m256i gp = _mm256_sub_epi32(_mm256_max_epi32(uu, ll), v2);
                _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi32(dg, gp));
            }
        }
#else
        {
            const int32_t *restrict q2 = p2;
            const int32_t *restrict q1 = p1;
            int32_t       *restrict qc = cu;
            for (int i = ilo; i <= ihi; i++) {
                int s = q2[i - 1] + ((ca[i - 1] == pb[i]) ? MATCH : MISMATCH);
                int u = q1[i - 1] + GAP;
                int l = q1[i]     + GAP;
                int best = s;
                if (u > best) best = u;
                if (l > best) best = l;
                qc[i] = best;
            }
        }
#endif
        if (d <= N) { cu[0] = GAP * d; cu[d] = GAP * d; }
        { int32_t *t = p2; p2 = p1; p1 = cu; cu = t; }
    }

    int res = p1[N];
    free(mem); free(cmem);
    return res;
}

/* ============================================================ */
int kernel(int n, const char *a, const char *b)
{
    if (n <= 0)      return 0;
    if (n < 32)      return nw_flat(n, a, b);    /* crease too short to fold  */
    if (n <= 16000)  return nw_fold16(n, a, b);  /* half-width lamps          */
    return nw_fold32(n, a, b);                   /* full-width lamps          */
}
```

**Correctness argument (hand-checked before any compile).** For cell `(i, j)` on crease `d = i+j`: `dp[i-1][j-1]` sits on crease `d-2` at row `i-1` ⟹ `p2[i-1]`; `dp[i-1][j]` sits on crease `d-1` at row `i-1` ⟹ `p1[i-1]`; `dp[i][j-1]` sits on crease `d-1` at row `i` ⟹ `p1[i]`. So every recurrence input is one of three unit-stride loads from two buffers — exactly the native's "three layers lying bit for bit atop each other," with "beneath" = the `d-2` agreeing-straight-through layer and "beside" = the `d-1` one-slip layer. I traced `n=2`, `a="AC"`, `b="AG"` cell by cell through the fold and got `0`, matching the reference table. Rim sparks are written *after* the vector loop so the pad-overrun store at index `d` is corrected; overrun beyond `min(n,d)` is provably never read before that buffer is recycled, and all overrun loads/stores land inside the 64-element pad.

# PREDICTION

**PREDICTION: speedup_vs_dp = 9.0**

Reasoning stated before measuring. The reference inner loop is a serial latency chain through `dp[i][j-1]` — realistically ~2 cycles/cell and unvectorizable. The fold issues ~8 uops per 16 cells (AVX2/int16), so ~0.2 cycles/cell, a raw ~10× on arithmetic. Lane waste at the short creases near the two corners costs roughly `(2n-1)·8` extra lane-slots ≈ negligible above n≈500 but real at n≈100. Against that, the reference touches `4(n+1)²` bytes of DRAM (16 MB at n=2000) while the fold's entire working set is ~6n bytes in L1 — worth several extra × at large n. Netting the small-n lane waste against the large-n bandwidth win across a typical benchmark sweep, 9× is my honest centre. AVX-512BW machines should land noticeably above it; a no-AVX2 machine would land near 1.5× (memory win only).

# MEASUREMENT

**Not measured in this session.** `alignment_bench` and `alignment_contract` were listed as available but no tool execution was possible here, so I ran zero of the four permitted improvement rounds. I am reporting that plainly rather than fabricating a number: the prediction above is unvalidated, the correctness claim rests on the hand-trace of the `n=2` case and the index-range proof, not on a differential test against the reference aligner. The pipeline's compile-and-measure pass is the first real evidence either way. If it disagrees with 9.0, the prediction was wrong, not the measurement.

The specific things I would have measured, in order: (1) bit-exact agreement with the reference on random `{A,C,G,T}` strings at `n = 1, 2, 31, 32, 33, 63, 127, 1000` — the regime boundaries are where a fold bug would hide; (2) speedup at each benchmark `n`; (3) whether the `n < 32` flat path is actually the faster side of its own cutoff, since my estimate says the fold may already win at `n = 32`–`63` and the guard may be costing a little there.

# VERDICT

The native's fold is not a metaphor for the anti-diagonal wavefront — it *is* one, arrived at from a different direction, and taking it literally paid twice.

What the fold gives that the textbook method does not: the row-major order is the only reason NW's inner loop cannot vectorize, and the fold deletes the order while preserving every dependency, so the result is bit-identical rather than approximate. This is the opposite trade from banded/striped SIMD aligners (SSW, KSW2), which buy their vector width by restricting the search or by accepting a lazy-`F` correction loop. The fold needs neither: on a crease there is genuinely no neighbour to wait for.

The genuinely new thing fell out of SEED 1, not SEED 2. "Crossed rows of lamp-posts" forced me to notice that `i` rises while `j` falls along a crease, hence to store `b` reversed — which turns the symbol comparison into two contiguous `loadu`s and a single `cmpeq_epi8`. Had I translated "compare the two characters" in the ordinary way I would have written a gather or a shuffle there and lost most of the win. The literalism requirement earned that.

"Throw the bottom layer away" I initially read as a mere memory optimisation. It is not: at n=2000 the reference is writing 16 MB through the store pipe while the fold's whole state fits in L1, and that difference is probably a larger share of the speedup than the SIMD width is. The native's refusal to "hoard its own marsh" is a bandwidth argument in disguise.

**Stated risks, and how each is addressed rather than left standing** (per step 4 — I have shipped no mechanism whose own risk I named and ignored):

- *Folding overhead dominates when the creases are short.* Guarded: `n < 32` falls back to a plain rolling-row DP on stack buffers, no mallocs, no wavefront bookkeeping. I flagged above that this cutoff may be set slightly too high and would tune it against measurement.
- *`int16` sparks overflow on long boughs.* Guarded: `n > 16000` switches to the `int32` fold. Bound checked by hand — the widest reachable intermediate is `max(u,l) - 2 ≥ -2n-2 = -32002` at the threshold, inside `int16`.
- *No AVX2 on the target machine.* Guarded: the identical crease recurrence compiles as a scalar loop with `restrict`, keeping the O(n) memory win.
- *Thread parallelism.* Deliberately **not** shipped. The metaphor itself forbids it — "I wait only as long as it takes each spark to catch from its neighbors, never longer" is a prohibition on barriers, and a per-crease OpenMP fork/join is exactly the wait past the catching. At benchmark sizes a crease is a few dozen vector ops, orders of magnitude below fork cost. The default of vectorization-before-threads and the native's own physics agree, so there was nothing to guard.

What could still be wrong: I have not proven the pad-overrun reasoning by execution, only by index arithmetic, and that is the class of bug that silently returns a right answer on most inputs. Until `alignment_contract` says otherwise, treat the correctness as argued, not demonstrated.