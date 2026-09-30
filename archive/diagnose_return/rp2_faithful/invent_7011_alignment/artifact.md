## MAPPING

### SEED 1 — "I hang two boughs of symbols from crossed rows of lamp-posts so their meeting makes a lattice of spark-cells."

| World object | Problem object |
|---|---|
| bough running *down* | string `a`, indexed by `i` (0..n) |
| bough running *across* | string `b`, indexed by `j` (0..n) |
| row of lamp-posts / column of lamp-posts | the two index axes; the posts are *held fixed*, the boughs hang from them (strings are read-only, never copied per cell) |
| lattice cell at a crossing | DP cell `H[i][j]` |
| a lit spark | the integer score living in that cell |
| the spark's **color**, chosen by the two crossing symbols, not by me | substitution term `a[i-1]==b[j-1] ? +1 : -1` — data-selected, branchlessly (a compare + blend), never a branch I "choose" |
| "I light each cell only when I need it" | cells are materialized transiently; no full grid is ever allocated |
| "no ground, only the gradient" | no `(n+1)²` backing store exists to lean on |
| *down* vs *across* | the two boughs hang in **opposite senses** → one of them (`b`) is stored reversed, `br[t] = b[n-1-t]` |

**Silent assumption broken:** "the whole grid of every position against every other must be filled in." Also breaks "both strings are read start to end in the same direction" — hanging one bough down and one across forces one to be read backwards, which is exactly what turns the diagonal's character fetch into two *unit-stride* loads.

### SEED 2 — "I fold the lattice-sheet along its slanting middle so cells of equal row-and-column sum press flush into a single crease."

| World object | Problem object |
|---|---|
| the slanting fold, corner to corner | the anti-diagonal shear `d = i + j` |
| one crease | one anti-diagonal wavefront: all `(i, d-i)` |
| "cells whose row and column add to the same sum press flush" | every cell on a crease has *no dependence on any other cell on that crease* |
| cells lying flush against each other | 16 (AVX2) or 32 (AVX-512BW) cells in one SIMD register |
| "the note is already folded in the hour between the note-trees" | the skew is a property of the recurrence itself, not an optimization bolted on |
| the sheet | 3 × O(n) buffers indexed by `i` alone — the fold collapses a 2-D sheet to a 1-D crease |
| walking the fold forward | `for d = 2 .. 2n` |
| the last crease's single spark at the far corner | `H[n][n]`, the answer |

**Silent assumption broken:** **"every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed *in that order*."** Under the fold there *is* no order inside a crease — the left-neighbour serial chain that makes row-major NW unvectorizable is dissolved, because `(i,j-1)` and `(i-1,j)` both land on the *previous* crease, never on the current one. It also breaks "one pair of positions is judged at a time": 16–32 pairs are judged per instruction.

### SEED 3 — "I read three stacked layers through each crease at once and let the spent bottom layer drift off before folding the next crease forward."

| World object | Problem object |
|---|---|
| three layers pressed into one thickness | creases `d`, `d-1`, `d-2` |
| "the layer beneath tells me what agreeing straight through would have cost" | `p2[i-1]` = `H[i-1][j-1]` → the match/mismatch term |
| "the layer beside it tells me what one slip would have cost" | `p1[i-1]`, `p1[i]` = `H[i-1][j]`, `H[i][j-1]` → both gap terms, fused as `max(p1[i-1],p1[i]) - 2` |
| "whichever of the three costs less becomes the new spark on the top layer" | `cur[i] = max(diag, gapbest)` |
| "I throw the bottom layer away… I am not a forest that hoards its own marsh" | three-buffer rotation: crease `d-2`'s memory is *reused* as crease `d+1` → **O(n) memory total**, everything resident in L1/L2 |
| "I wait only as long as it takes each spark to catch from its neighbours" | no barriers, no synchronization beyond the natural crease-to-crease data dependence |

**Silent assumption broken:** again "the whole grid must be filled in" — and, more sharply, the *storage* half of it: the reference writes and re-reads 4·(n+1)² bytes; the native's sheet drifts into the gradient behind him.

## CHOSEN SEED

**SEED 2** — the fold along the slanting middle.

It is the only one of the three that breaks the preferred assumption (the above/left/diagonal *computation order*), and it is maximally far from the known way: Needleman–Wunsch row-major, SSW and KSW2 all keep a **query-parallel striped** layout along one string with a lazy-F loop to repair the serial left-dependency. The native does not stripe along a string at all — he stripes along the *anti-diagonal*, which needs no lazy-F fixup because the left-dependency is *structurally* on a different crease. SEEDs 1 and 3 are not alternatives; they are the two things this fold forces on you (one bough reversed, three layers with the bottom discarded), so the artifact implements all three as one mechanism.

## ASSUMPTION BROKEN

*"Every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order."* — The dependence on all three neighbours is preserved exactly (the score matches NW bit for bit); the **order** is destroyed. On crease `d`, all `min(n,d-1) - max(1,d-n) + 1` cells are mutually independent and are computed simultaneously.

Secondary breakages actually exploited: one string read backwards (SEED 1) so both character streams are unit-stride; and the grid never existing (SEED 3).

## ARTIFACT

Which code implements which part of the native's mechanism:

- `al_rows` — **the flat walker.** The regime fallback: a lattice too small to be worth hanging lamp-posts for (`n < 32`), and the safety net if a malloc fails.
- `br[t] = b[n-1-t]` — SEED 1's *one bough down, one across*. This is what makes `a + i - 1` and `br + i + off` both stride-+1 inside a crease; without the reversal the second fetch would be a gather and the whole mechanism would be worthless.
- the `for (d = 2; d <= 2*n; ++d)` loop — SEED 2's *walking the fold forward across the lattice*.
- `lo = max(1, d-n)`, `hi = min(n, d-1)` — the extent of one crease where the sheet actually has cells.
- `al_crease32` / `al_crease16` (AVX-512BW / AVX2) and `al_spark` (scalar remainder) — SEED 2's *cells of equal row-and-column sum pressed flush*: one crease-segment per register.
  - `_mm256_cmpeq_epi16` + `_mm256_blendv_epi8` — *the two symbols crossing there choose the spark's colour, I do not*: branchless ±1, no data-dependent control flow anywhere in the kernel.
  - `max(loadu(p1+i-1), loadu(p1+i)) + gap` — *the layer beside it tells me what one slip would have cost*, both slips fused into one max (SEED 3).
  - `loadu(p2+i-1) + sub` — *the layer beneath tells me what agreeing straight through would have cost* (SEED 3).
  - final `max` — *I take whichever of the three costs less*.
- `t = p2; p2 = p1; p1 = cu; cu = t;` — SEED 3's *I throw the bottom layer away… I keep only the two most recent creases pressed together*. The spent crease's memory becomes the next crease's; total footprint 3·(n+1)·2 bytes.
- `if (d <= n) { cu[0] = g; cu[d] = g; }` — the two lamp-posts themselves: the gap-only boundary rows, `H[0][d]` and `H[d][0]`.
- `n > AL_I16MAX` → `al_diag_i32` — **regime recognition through the metaphor**: how bright a spark a lamp can hold. The `known_way` text names a bounded-score-range SIMD regime *and* a general one; a 16-bit spark holds the whole answer only while `2n ≤ 32767`, so past `n = 16000` the same fold is walked with 32-bit sparks.
- No thread parallelism, deliberately: the native's own unit of work is *one crease*, and a crease at benchmark sizes (n ≈ 10²–10³, so 10²–10³ cells ≈ sub-microsecond) is thinner than the barrier that would have to separate it from the next one. 2n barriers would cost more than the entire computation. Stated plainly rather than shipped behind a size check that would never fire.

```c
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#define AL_MATCH      1
#define AL_MISMATCH (-1)
#define AL_GAP      (-2)
#define AL_SMALL     32      /* lattice too small to be worth hanging lamp-posts for */
#define AL_I16MAX 16000      /* a 16-bit spark holds the answer only while 2n <= 32767 */

/* ---------- the flat walker: plain NW, two rolling rows (fallback regime) ---------- */
static int al_rows(int n, const unsigned char *a, const unsigned char *b)
{
    int *buf, *prev, *cur, r;
    if (n <= 0) return 0;
    buf = (int *)malloc((size_t)2 * (size_t)(n + 1) * sizeof(int));
    if (!buf) return 0;
    prev = buf; cur = buf + (n + 1);
    for (int j = 0; j <= n; ++j) prev[j] = AL_GAP * j;
    for (int i = 1; i <= n; ++i) {
        unsigned char ai = a[i - 1];
        int *t;
        cur[0] = AL_GAP * i;
        for (int j = 1; j <= n; ++j) {
            int dg = prev[j - 1] + (ai == b[j - 1] ? AL_MATCH : AL_MISMATCH);
            int up = prev[j] + AL_GAP;
            int lf = cur[j - 1] + AL_GAP;
            int bst = dg;
            if (up > bst) bst = up;
            if (lf > bst) bst = lf;
            cur[j] = bst;
        }
        t = prev; prev = cur; cur = t;
    }
    r = prev[n];
    free(buf);
    return r;
}

/* ---------- one spark, read through the three layers of the crease ---------- */
static inline int al_spark(const short *p1, const short *p2,
                           const unsigned char *a, const unsigned char *br,
                           int i, int off)
{
    int dg = (int)p2[i - 1] + ((a[i - 1] == br[i + off]) ? AL_MATCH : AL_MISMATCH);
    int g1 = (int)p1[i - 1], g2 = (int)p1[i];
    int g  = ((g1 > g2) ? g1 : g2) + AL_GAP;
    return (dg > g) ? dg : g;
}

#if defined(__AVX2__)
/* 16 cells of one crease pressed flush into one register */
static inline void al_crease16(short *cu, const short *p1, const short *p2,
                               const unsigned char *a, const unsigned char *br,
                               int i, int off)
{
    __m256i a16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(a + i - 1)));
    __m256i b16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(br + i + off)));
    __m256i eq  = _mm256_cmpeq_epi16(a16, b16);                      /* the symbols pick the colour */
    __m256i sub = _mm256_blendv_epi8(_mm256_set1_epi16(AL_MISMATCH),
                                     _mm256_set1_epi16(AL_MATCH), eq);
    __m256i gp  = _mm256_add_epi16(                                   /* layer beside: one slip */
                    _mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(p1 + i - 1)),
                                     _mm256_loadu_si256((const __m256i *)(p1 + i))),
                    _mm256_set1_epi16(AL_GAP));
    __m256i dg  = _mm256_add_epi16(                                   /* layer beneath: straight through */
                    _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sub);
    _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(gp, dg));
}
#endif

#if defined(__AVX512BW__)
static inline void al_crease32(short *cu, const short *p1, const short *p2,
                               const unsigned char *a, const unsigned char *br,
                               int i, int off)
{
    __m512i a16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(a + i - 1)));
    __m512i b16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(br + i + off)));
    __mmask32 eq = _mm512_cmpeq_epi16_mask(a16, b16);
    __m512i sub = _mm512_mask_blend_epi16(eq, _mm512_set1_epi16(AL_MISMATCH),
                                              _mm512_set1_epi16(AL_MATCH));
    __m512i gp  = _mm512_add_epi16(
                    _mm512_max_epi16(_mm512_loadu_si512((const void *)(p1 + i - 1)),
                                     _mm512_loadu_si512((const void *)(p1 + i))),
                    _mm512_set1_epi16(AL_GAP));
    __m512i dg  = _mm512_add_epi16(_mm512_loadu_si512((const void *)(p2 + i - 1)), sub);
    _mm512_storeu_si512((void *)(cu + i), _mm512_max_epi16(gp, dg));
}
#endif

/* ---------- the fold, walked forward with 16-bit sparks ---------- */
static int al_diag_i16(int n, const unsigned char *a, const unsigned char *b)
{
    size_t lane = (size_t)n + 1 + 64;
    void *blk;
    short *L0, *L1, *L2, *p2, *p1, *cu;
    unsigned char *br;
    int res;

    blk = malloc(3 * lane * sizeof(short) + (size_t)n + 64);
    if (!blk) return al_rows(n, a, b);
    memset(blk, 0, 3 * lane * sizeof(short));
    L0 = (short *)blk; L1 = L0 + lane; L2 = L1 + lane;
    br = (unsigned char *)(L2 + lane);
    for (int t = 0; t < n; ++t) br[t] = b[n - 1 - t];   /* one bough hangs the other way */
    memset(br + n, 0xFF, 64);

    p2 = L0; p1 = L1; cu = L2;
    p2[0] = 0;                                          /* crease 0 */
    p1[0] = (short)AL_GAP; p1[1] = (short)AL_GAP;        /* crease 1: the two lamp-posts */

    for (int d = 2; d <= 2 * n; ++d) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        int off = n - d;
        int cnt = hi - lo + 1;
        int i = lo;
        short *t;
#if defined(__AVX512BW__)
        if (cnt >= 32) {
            for (; i + 32 <= hi + 1; i += 32) al_crease32(cu, p1, p2, a, br, i, off);
            if (i <= hi) { al_crease32(cu, p1, p2, a, br, hi - 31, off); i = hi + 1; }
        }
#endif
#if defined(__AVX2__)
        if (cnt >= 16) {
            for (; i + 16 <= hi + 1; i += 16) al_crease16(cu, p1, p2, a, br, i, off);
            if (i <= hi) { al_crease16(cu, p1, p2, a, br, hi - 15, off); i = hi + 1; }
        }
#endif
        for (; i <= hi; ++i) cu[i] = (short)al_spark(p1, p2, a, br, i, off);
        if (d <= n) { short g = (short)(AL_GAP * d); cu[0] = g; cu[d] = g; }
        t = p2; p2 = p1; p1 = cu; cu = t;               /* the spent layer drifts off */
    }
    res = (int)p1[n];
    free(blk);
    return res;
}

/* ---------- the same fold with 32-bit sparks, for lattices past the 16-bit lamp ---------- */
static int al_diag_i32(int n, const unsigned char *a, const unsigned char *b)
{
    size_t lane = (size_t)n + 1 + 64;
    int *mem = (int *)calloc(3 * lane, sizeof(int));
    unsigned char *br = (unsigned char *)malloc((size_t)n + 64);
    int *p2, *p1, *cu, res;
    if (!mem || !br) { free(mem); free(br); return al_rows(n, a, b); }
    for (int t = 0; t < n; ++t) br[t] = b[n - 1 - t];
    memset(br + n, 0xFF, 64);
    p2 = mem; p1 = mem + lane; cu = mem + 2 * lane;
    p2[0] = 0; p1[0] = AL_GAP; p1[1] = AL_GAP;
    for (int d = 2; d <= 2 * n; ++d) {
        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        int off = n - d;
        const int *__restrict q1 = p1;
        const int *__restrict q2 = p2;
        int *__restrict w = cu;
        int *t;
        #pragma omp simd
        for (int i = lo; i <= hi; ++i) {
            int g1 = q1[i - 1], g2 = q1[i];
            int g  = ((g1 > g2) ? g1 : g2) + AL_GAP;
            int dg = q2[i - 1] + ((a[i - 1] == br[i + off]) ? AL_MATCH : AL_MISMATCH);
            w[i] = (dg > g) ? dg : g;
        }
        if (d <= n) { cu[0] = AL_GAP * d; cu[d] = AL_GAP * d; }
        t = p2; p2 = p1; p1 = cu; cu = t;
    }
    res = p1[n];
    free(mem); free(br);
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    const unsigned char *ua = (const unsigned char *)a;
    const unsigned char *ub = (const unsigned char *)b;
    if (n <= 0) return 0;
    if (n < AL_SMALL)    return al_rows(n, ua, ub);      /* regime: too small to fold */
    if (n > AL_I16MAX)   return al_diag_i32(n, ua, ub);  /* regime: spark too bright for 16 bits */
    return al_diag_i16(n, ua, ub);                       /* the fold */
}
```

## PREDICTION

Reasoning before any measurement: the reference's inner loop carries a serial `left` chain (`add` → `max`, ≥2 cycles per cell) and writes/re-reads 4·(n+1)² bytes. The fold removes the chain entirely, so one crease-segment of 16 cells costs ~14 µops ≈ 3.5–4 cycles, i.e. ~0.25 cycles/cell against ~2.5–3.0 cycles/cell, and the working set collapses from O(n²) to ~6n bytes (L1-resident). Losses: ~2n crease prologues, a scalar remainder only on creases shorter than 16 (the first/last ~16 creases), and imperfect lane fill near the corners (≈2% at n=1000). AVX-512BW hardware would roughly double this again; I do not assume it.

PREDICTION: speedup_vs_dp = 12

## MEASUREMENT

I must report this plainly rather than dress it up: **no measurement was taken in this session.** `alignment_bench` and `alignment_contract` were listed as available but no tool-execution channel exists in this invocation, so every number above is an a-priori estimate and the four permitted improvement rounds were not spent. The kernel below is therefore round 0 — hand-verified, not benchmarked.

What I did verify by hand, since I could not verify it by machine:

- **Index algebra.** Crease `d` covers `i ∈ [max(1,d-n), min(n,d-1)]`; `H[i-1][j-1]` sits on crease `d-2` at index `i-1`, `H[i-1][j]` and `H[i][j-1]` on crease `d-1` at `i-1` and `i`. The reversed fetch `b[d-i-1] = br[i + (n-d)]` is exact.
- **No out-of-bounds read of `a` or `b`.** Every vector body starts at `i ≤ hi+1-W`, so the byte loads touch at most `a[hi-1]` and `br[hi+off]`, both `≤ n-1`.
- **No read of uninitialized or stale lanes.** Every lane of every load lies inside the source crease's own valid index range (`p1[hi]` is exactly crease `d-1`'s top cell; `p2[lo-1]` exactly its bottom), and every store lands inside `[lo,hi]`. The overlapping tail recomputes cells from unchanged read-only buffers, so it is idempotent.
- **Coverage.** `[lo,hi]` is fully written for every crease (vector stride + one overlapping tail when `cnt ≥ W`, scalar otherwise).
- **Corner cases.** `n=0` → 0; `n=1` → `max(±1, -4, -4)`; `d=2n` → `p1[n] = H[n][n]`.
- **No 16-bit wrap.** All values lie in `[-2n, n]`; `n ≤ 16000` ⇒ `|v| ≤ 32000 < 32767`, and the gated `al_diag_i32` covers the rest.

Anyone running `alignment_bench` should treat `12` as the falsifiable claim. My own honest confidence interval on it is roughly 6–20 (low end if the machine is AVX2-only *and* the benchmark uses small `n` where crease prologues dominate; high end with AVX-512BW at `n ≥ 1024`).

## VERDICT

The fold is a real mechanism, not a re-labelling of the textbook method. The decisive thing the native says that the striped-SIMD literature does not is *"the fold presses every cell whose row and column add to the same sum flush against its neighbours"* — anti-diagonal striping instead of query striping. That single choice deletes the lazy-F repair loop that SSW and KSW2 both need, because under this skew the left-neighbour is never on the current register. And his refusal to hoard the marsh (*"I throw the bottom layer away"*) is what makes it L1-resident: 6n bytes against the reference's 4n².

The price, stated honestly, is the one thing flat walkers get for free: lane fill. Near the two corners the creases are shorter than a register, so the mechanism degrades to scalar exactly where the reference is at its most cache-friendly. That is the mechanism's own named risk, and per step 4 it is not left dangling — it is guarded by `n < AL_SMALL → al_rows`, a genuine fallback to the simpler path rather than a hope. The second named risk, the 16-bit spark overflowing, is guarded by `n > AL_I16MAX → al_diag_i32`. Nothing risky is shipped unguarded, and no thread parallelism is shipped at all, because 2n barriers over sub-microsecond creases would cost more than the work they coordinate — a guard that would never fire is not a guard, so I dropped the part instead.

Where it could lose outright: if the benchmark exercises only `n` in the low tens, control never reaches the fold and the speedup is ~1.0 by construction. If it exercises `n ≥ 512`, I expect the fold to win by roughly an order of magnitude. The claim is measurable and I have not measured it; the number in PREDICTION is what I am willing to be wrong about.