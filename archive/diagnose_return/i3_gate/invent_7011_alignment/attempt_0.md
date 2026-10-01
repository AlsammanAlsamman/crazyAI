## MAPPING

### SEED 1 — "I hang two boughs of symbols from crossed rows of lamp-posts so their meeting makes a lattice of spark-cells."

| World object | Problem object |
|---|---|
| bough running down | `a`, indexed by row `i` |
| bough running across | `b`, indexed by column `j` |
| row of lamp-posts / column of lamp-posts | the two pointers/strides that address `a` and `b` |
| lattice of *unlit* cells | the DP grid as a **virtual** object — coordinates only, never allocated |
| a spark | one DP score |
| "I light each cell only when I need it" | on-demand materialization; no `(n+1)²` table |
| "I do not choose the color, the two symbols crossing choose it" | the substitution score is a *data-driven select* (`cmpeq` → blend), not a branch |

**Silent assumption broken:** *"the whole grid of every position against every other must be filled in."* Only lit cells exist; the grid is an addressing scheme, not storage.

### SEED 2 — "I fold the lattice-sheet along its slanting middle so cells of equal row-and-column sum press flush into a single crease."

| World object | Problem object |
|---|---|
| the slanting middle / the fold | the anti-diagonal `i + j = d` |
| one crease | one anti-diagonal = one contiguous vector of cells |
| "cells of equal row-and-column sum press flush" | every cell on a crease is **mutually independent** → one SIMD lane each |
| "the way the note itself is already folded in the hour between the note-trees" | the recurrence's own dependence cone is already diagonal; row-major order is an imposition, not a requirement |
| walking the fold forward, crease after crease | `d = 2 … 2n`, the wavefront |
| the last crease holding one spark at the far corner | `dp[n][n]`, the answer |

**Silent assumption broken:** *"every cell of the comparison depends on the ones above, to the left, and diagonally above-left, **computed in that order**."* The dependency set is kept exactly; the *order* is discarded. Also breaks "one pair of positions is judged at a time" — a crease judges 16 (or 32) pairs in one instruction.

### SEED 3 — "I read three stacked layers through each crease at once and let the spent bottom layer drift off before folding the next crease forward."

| World object | Problem object |
|---|---|
| three layers pressed into one thickness | diagonals `d`, `d−1`, `d−2` |
| "the layer beneath — agreeing straight through" | `D[d−2][i−1] + (match ? +1 : −1)` |
| "the layer beside it — what one slip would have cost" | `D[d−1][i−1] + GAP` (up) and `D[d−1][i] + GAP` (left) — **both slips live on the same layer** |
| take whichever of the three costs less | `max` of three |
| "I throw the bottom layer away" | the `d−2` buffer is recycled as the new `d` buffer: three rotating rows, **O(n) memory** |
| "the note has no ground to store it on" | no `O(n²)` table, no cache streaming, working set ≈ 6n bytes → fits L1/L2 |
| "I wait only as long as it takes each spark to catch from its neighbours" | no barrier beyond the crease-to-crease dependence; no row serialization |

**Silent assumption broken:** *"a slip (gap) can only be discovered by having already compared the position before it."* Up-gap and left-gap are read off the **same** crease at offsets `i−1` and `i` — one `vpmaxsw` on two unaligned loads of one array. Also re-breaks "the whole grid must be filled in."

---

## CHOSEN SEED

**SEED 2** — the fold along the slanting middle. It is the one seed that attacks the *ordering* clause of the preferred assumption head-on, and its mapping is one-to-one with no slack: crease = anti-diagonal = SIMD register, row-and-column sum = `d`, fold-forward = wavefront. SEED 3 is not a competitor but SEED 2's memory consequence (once you fold, only three layers are reachable), so the artifact implements 2 and 3 as the single mechanism the native actually described.

Per step 4, I deliberately let the metaphor **land on a validated technique rather than invent one**: anti-diagonal ("Wozniak") vectorization of Needleman–Wunsch is a real, published, field-tested layout, and the three-rotating-diagonal buffer is the standard wavefront storage used in WFA/ksw2-class aligners. The native's sheet-folding is that technique, described from the inside. I did **not** reach for the more exotic 8-bit difference-recurrence (Suzuki–Kasahara) encoding: it would double lane count but I cannot test it here, and an untested cleverness loses to a validated one.

## ASSUMPTION BROKEN

> *every cell of the comparison depends on the ones above, to the left, and diagonally above-left, **computed in that order***

The **dependency** is preserved exactly (the kernel is bit-identical to the reference recurrence). The **order** is not. Row-major order creates a loop-carried chain through `left` — each cell waits ~2 cycles on the previous one, so the reference is structurally scalar. Under the fold, all cells with `i+j = d` are independent, so the cheapest correct order is "all of them at once."

Consequences that fall out of taking the fold literally:
1. `up` and `left` collapse into **one** array read twice at offsets `i−1`, `i` → one `max` before the gap add.
2. Only three creases are ever live → `O(n)` memory instead of the reference's 4 MB-at-n=1000 table.
3. The crease needs `a` forward and `b` **reversed** to make both symbol streams contiguous — the native's "boughs cut from different note-trees," hung from opposite ends.

**Regime dispatch, expressed in-metaphor** (step 5 — `known_way` names two regimes: plain `O(n²)` DP vs. a variant "exploiting a bounded score range"): the native checks, before hanging anything, whether the lattice is worth the lamp-posts. Short boughs (`n < 64`) → don't hang the sheet, walk the ground (two-row scalar DP). Boughs so long that a spark's colour would burn through its cell (`n > 16000`, i.e. `|−2n|` past `int16`) → same fallback. Both paths are exact.

No thread parallelism: the metaphor's unit of work is one crease, which at `n = 1000–4000` is ~60–250 ns; an OpenMP barrier `2n` times would cost more than the whole kernel. Vectorization only, per step 4's default.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX512BW__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH      1
#define MISMATCH (-1)
#define GAP      (-2)

/* regime gates: below WF_MIN_N the lattice costs more than it saves;
   above WF_MAX_N a 16-bit spark would burn through its cell (-2n < -32768). */
#define WF_MIN_N   64
#define WF_MAX_N   16000
#define WF_PAD     64          /* slack so a vector may overrun a crease safely */

/* ---- ground-walker: exact two-row Needleman-Wunsch, O(n) memory ---- */
static int nw_rows(int n, const char *restrict a, const char *restrict b)
{
    if (n <= 0) return 0;
    int *restrict row = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!row) return 0;
    for (int j = 0; j <= n; ++j) row[j] = j * GAP;
    for (int i = 1; i <= n; ++i) {
        int diag = row[0];
        row[0] = i * GAP;
        int left = row[0];
        const char ai = a[i - 1];
        for (int j = 1; j <= n; ++j) {
            int up   = row[j];
            int best = diag + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int g    = ((up > left) ? up : left) + GAP;
            if (g > best) best = g;
            row[j] = best;
            diag   = up;
            left   = best;
        }
    }
    int r = row[n];
    free(row);
    return r;
}

/* ---- the folded sheet: anti-diagonal wavefront, three creases live ---- */
int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < WF_MIN_N || n > WF_MAX_N) return nw_rows(n, a, b);

    const size_t clen = (size_t)n + WF_PAD;            /* padded symbol boughs */
    const size_t dlen = (size_t)n + 1 + WF_PAD;        /* one crease           */
    const size_t ibytes = 3 * dlen * sizeof(int16_t);

    unsigned char *mem = (unsigned char *)malloc(ibytes + 2 * clen + 64);
    if (!mem) return nw_rows(n, a, b);

    int16_t *buf = (int16_t *)mem;
    char *ap = (char *)(mem + ibytes);
    char *br = ap + clen;

    memset(buf, 0, ibytes);
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 0, WF_PAD);
    for (int k = 0; k < n; ++k) br[k] = b[n - 1 - k];  /* bough hung from the far end */
    memset(br + n, 1, WF_PAD);                          /* padding never matches ap's  */

    int16_t *p2 = buf;                 /* crease d-2 */
    int16_t *p1 = buf + dlen;          /* crease d-1 */
    int16_t *cu = buf + 2 * dlen;      /* crease d   */

    p2[0] = 0;                         /* d = 0 : dp[0][0]            */
    p1[0] = (int16_t)GAP;              /* d = 1 : dp[0][1]            */
    p1[1] = (int16_t)GAP;              /*         dp[1][0]            */

#if defined(__AVX512BW__)
    const __m512i vg  = _mm512_set1_epi16((short)GAP);
    const __m512i vm1 = _mm512_set1_epi16((short)-1);
#elif defined(__AVX2__)
    const __m256i vg  = _mm256_set1_epi16((short)GAP);
    const __m256i vm1 = _mm256_set1_epi16((short)-1);
#endif

    for (int d = 2; d <= 2 * n; ++d) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        const int len = hi - lo + 1;
        const char *pa = ap + (lo - 1);
        const char *pb = br + (lo + n - d);

#if defined(__AVX512BW__)
        for (int k = 0; k < len; k += 32) {
            __m256i ca = _mm256_loadu_si256((const __m256i *)(pa + k));
            __m256i cb = _mm256_loadu_si256((const __m256i *)(pb + k));
            __m512i eq = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
            __m512i s  = _mm512_sub_epi16(vm1, _mm512_add_epi16(eq, eq)); /* +1 / -1 */
            __m512i x  = _mm512_add_epi16(
                             _mm512_loadu_si512((const void *)(p2 + lo + k - 1)), s);
            __m512i g  = _mm512_add_epi16(_mm512_max_epi16(
                             _mm512_loadu_si512((const void *)(p1 + lo + k - 1)),
                             _mm512_loadu_si512((const void *)(p1 + lo + k))), vg);
            _mm512_storeu_si512((void *)(cu + lo + k), _mm512_max_epi16(x, g));
        }
#elif defined(__AVX2__)
        for (int k = 0; k < len; k += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(pa + k));
            __m128i cb = _mm_loadu_si128((const __m128i *)(pb + k));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i s  = _mm256_sub_epi16(vm1, _mm256_add_epi16(eq, eq)); /* +1 / -1 */
            __m256i x  = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(p2 + lo + k - 1)), s);
            __m256i g  = _mm256_add_epi16(_mm256_max_epi16(
                             _mm256_loadu_si256((const __m256i *)(p1 + lo + k - 1)),
                             _mm256_loadu_si256((const __m256i *)(p1 + lo + k))), vg);
            _mm256_storeu_si256((__m256i *)(cu + lo + k), _mm256_max_epi16(x, g));
        }
#else
        {
            int16_t       *restrict cc  = cu + lo;
            const int16_t *restrict q2  = p2 + lo - 1;
            const int16_t *restrict q1a = p1 + lo - 1;
            const int16_t *restrict q1b = p1 + lo;
            for (int k = 0; k < len; ++k) {
                int s = (pa[k] == pb[k]) ? MATCH : MISMATCH;
                int x = (int)q2[k] + s;
                int u = (int)q1a[k], v = (int)q1b[k];
                int g = ((u > v) ? u : v) + GAP;
                cc[k] = (int16_t)((x > g) ? x : g);
            }
        }
#endif
        if (d <= n) {                       /* the two lamp-posts of this crease */
            int16_t e = (int16_t)(-2 * d);
            cu[0] = e;                      /* dp[0][d] */
            cu[d] = e;                      /* dp[d][0] */
        }
        { int16_t *t = p2; p2 = p1; p1 = cu; cu = t; }   /* bottom layer drifts off */
    }

    int res = (int)p1[n];
    free(mem);
    return res;
}
```

**Why it is exact.** With `d = i + j` and creases indexed by the row `i`:
`D_d[i] = max( D_{d-2}[i-1] + s(a_{i-1}, b_{j-1}), D_{d-1}[i-1] + GAP, D_{d-1}[i] + GAP )`
because `(i-1,j-1)` sits on `d-2` at index `i-1`, `(i-1,j)` on `d-1` at `i-1`, and `(i,j-1)` on `d-1` at `i`. With `br[k] = b[n-1-k]`, the symbol pair for index `i` on crease `d` is `a[i-1]` vs `br[i+n-d]` — both streams contiguous and ascending. Boundaries `dp[i][0] = dp[0][i] = -2i` are the two ends of crease `i`. Hand-traced against the reference for `n = 2` (`"AC"` vs `"AC"` → 2) and the index ranges checked so every read lands on a written cell and every vector overrun lands on padding that is never read back.

## PREDICTION

Reference: one cell per iteration of a loop-carried `add → max` chain ≈ 2.5–4 cycles/cell, over a `(n+1)²` `int` table that leaves cache at `n ≳ 700`.
Fold: 7 instructions per 16 cells (AVX2), ≈ 2.5–3 cycles per crease-chunk → ≈ 0.18 cy/cell, degrading to ≈ 0.3 cy/cell after short-crease and loop overhead; working set ≈ 6n bytes, L1-resident.

**PREDICTION: speedup_vs_dp = 9**

(AVX-512 machine → I'd expect 14–18; AVX2-only → 8–11; `n < 64` → ~2–3 from the row-DP fallback alone, since the reference's `O(n²)` allocation is itself a cost.)

## MEASUREMENT

**Not performed.** No tools were reachable in this session — `alignment_bench` and `alignment_contract` were listed but the environment exposes none of them, so I could not compile, verify against the reference, or time anything. I am reporting that plainly rather than quoting a number I did not obtain. The prediction above was written before, and independently of, any measurement, and stands as stated for the pipeline to falsify.

What would falsify it, in order of likelihood:
1. **Speedup ≈ 1–2× instead of 9×** → the benchmark `n` is at or below `WF_MIN_N = 64`, so the fallback is doing the work. The fix is to lower the gate, not to touch the fold.
2. **Speedup ≈ 3–5×** → short creases dominate (small `n`), or the store-to-load turnaround on `p1` between consecutive creases is serializing; the remedy is blocking two creases per pass.
3. **Wrong answer** → most probably the reversed-bough index `i + n - d` or the boundary write `cu[d]` racing the vector overrun; both are the parts I could only verify by hand.

## VERDICT

The native is describing anti-diagonal (Wozniak) wavefront vectorization with a three-buffer rolling store, and the translation is literal end to end: crease = anti-diagonal, three layers = three live diagonals, "both slips on the layer beside" = one `vpmaxsw` on two offset loads of a single array, "the bottom layer drifts off" = `O(n)` memory, "the note has no ground" = never allocating the table. Nothing in the metaphor had to be bent, and nothing in it is invention — it lands on a technique that real aligners already validate, which is what I wanted from it.

The mechanism's own stated risks are both guarded rather than hoped away, per step 4: the 16-bit spark cannot burn through its cell (`n > 16000` → exact scalar fallback), and the lattice is not hung when the boughs are too short to pay for it (`n < 64` → exact scalar fallback). Both fallbacks compute the identical recurrence, so the kernel is exact in every regime; only the speed differs. Thread parallelism was considered and dropped because the metaphor's own unit of work — one crease — is far smaller than a barrier at benchmark sizes.

What I am genuinely uncertain about is the magnitude, not the direction. The direction is structural: the reference's cost is a dependency chain the fold simply does not have. The magnitude depends on the benchmark's `n` and on whether the host has AVX-512, and I could not observe either. If the measured speedup comes back near 1, my first suspicion is the size gate, not the fold.