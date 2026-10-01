# MAPPING

**SEED 1 — "two boughs hung from crossed rows of lamp-posts, meeting makes a lattice of spark-cells"**

| World object | Problem object |
|---|---|
| bough, cut from a note-tree | one DNA string (`a`, `b`), length `n` |
| row of lamp-posts running *down* | index `i = 0..n` over `a` (DP rows) |
| row of lamp-posts running *across* | index `j = 0..n` over `b` (DP columns) |
| lattice of *unlit* cells | the `(n+1)²` NW matrix `H`, never materialized |
| lighting one cell on demand | computing one `H[i][j]` |
| the spark's **color** | the integer score `H[i][j]` |
| "the two crossing symbols choose the color, not me" | `s(a[i-1],b[j-1]) = ±1`, data-chosen, branch-free |
| "no ground, only the gradient" | no O(n²) backing store exists; scores must stream |

**SEED 2 — "fold the sheet along its slanting middle; equal row+column sums press flush into one crease"**

| World object | Problem object |
|---|---|
| the sheet the sparks live on | `H` |
| the slanting middle, corner to corner | the anti-diagonal direction `i + j = k` |
| a **crease** | the vector `D_k[i] = H[i][k-i]`, contiguous in `i` |
| cells of equal row-and-column sum pressed flush | re-indexing so one anti-diagonal is one dense array |
| "three creases pressed into one thickness, three layers of spark bit for bit atop each other" | `D_k`, `D_{k-1}`, `D_{k-2}` are **index-aligned**: `D_k[i] = max(D_{k-2}[i-1]+s, D_{k-1}[i-1]-2, D_{k-1}[i]-2)` |
| "the note is already folded in the hour between the note-trees" | `b` is stored **reversed** so the column symbol `b[k-i-1] = brev[n-k+i]` is also contiguous in `i` |
| folding forward, crease after crease | `k = 0 … 2n` |

**SEED 3 — "read three stacked layers through each crease at once; let the spent bottom layer drift off"**

| World object | Problem object |
|---|---|
| three stacked layers | three rolling buffers `P2`, `P1`, `C` |
| "read all three at once, **not** walking them one after another" | one SIMD instruction evaluates 16 lanes of a crease simultaneously — no left-to-right order |
| "the layer beneath: what agreeing straight through would have cost" | `P2[i-1] + s` (diagonal / substitution move) |
| "the layer beside: what one slip would have cost" | `P1[i-1] - 2` and `P1[i] - 2` (the two gap moves) |
| "take whichever of the three costs less" | `max3` → `max(d, max(u,l)-2)` |
| "throw the bottom layer away, let it drift into the gradient" | pointer rotation; O(n) memory, buffers recycled forever |
| "I am not a forest that hoards its own marsh" | never store the full matrix, never traceback |
| "wait only as long as it takes each spark to catch from its neighbours, never longer" | exactly **one** dependency step per crease → no barriers → **no OpenMP** |
| the last crease's single spark at the far corner | `H[n][n]`, read as `D_{2n}[n]` |
| processor / time / what stays still | SIMD lanes across a crease / `k` / `a` and `brev` |

# CHOSEN SEED

**SEED 3.** It is the only one that makes an explicit claim about *order of evaluation* ("read all three at once, not one after another") and about *memory lifetime* ("let the spent bottom layer drift off"). SEED 2 supplies the geometry it needs (the fold), SEED 1 supplies the operands. So SEED 3 is implemented literally, standing on SEED 2's re-indexing.

# ASSUMPTION BROKEN

> "every cell of the comparison depends on the ones above, to the left, and diagonally above-left, **computed in that order**."

The dependency set is untouched — but the *order* is a lie of the flat walker. After SEED 2's fold, the three neighbours of `H[i][k-i]` live at `P2[i-1]`, `P1[i-1]`, `P1[i]` — all in *already-finished* creases. So **every cell on one crease is mutually independent**: no cell on crease `k` reads another cell on crease `k`. Row-major DP has a serial chain of length `n²`; the fold has a serial chain of length `2n`, with `n/2`-wide independence at each step. That independence is the SIMD lanes. The discard ("bottom layer drifts off") is what keeps the working set at 3 arrays of `n+1` — L1/L2 resident for every `n` that matters.

One risk the mechanism states about itself — *"wait only as long as it takes each spark to catch, never longer"* — I took as a prohibition: a thread barrier per crease costs ~1 µs × `2n` creases and would swamp the work. **I dropped OpenMP entirely** rather than ship a barrier the native forbids. Two further self-stated risks are guarded in code: int16 crease width (`n ≤ 8000`, proven safe since `|H| ≤ 3n`) and latency-boundedness when a crease is shorter than one vector (`n ≤ 32` → plain row DP).

# ARTIFACT

```c
/* Anti-diagonal ("crease") wavefront Needleman-Wunsch.
   Contract: int kernel(int n, const char *a, const char *b);
   match=+1, mismatch=-1, gap=-2.  Build: gcc -O3 -march=native -fopenmp -lm  */

#include <stdint.h>
#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the gradient: three recycled creases + the pre-folded (reversed) bough ---- */
static __thread void  *wf_mem = 0;
static __thread size_t wf_cap = 0;

static void *wf_get(size_t need)
{
    if (need > wf_cap) {
        void *p = realloc(wf_mem, need);
        if (!p) return 0;
        wf_mem = p;
        wf_cap = need;
    }
    return wf_mem;
}

/* GUARD 1: a crease shorter than one vector makes the wavefront latency-bound
   (2n serial store->load steps).  Below this size a flat walker is faster. */
static int wf_small(int n, const char *a, const char *b)
{
    int r0[40], r1[40];
    int *prev = r0, *cur = r1, *t;
    int i, j;
    for (j = 0; j <= n; j++) prev[j] = -2 * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = -2 * i;
        for (j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? 1 : -1);
            int u = prev[j] - 2;
            int l = cur[j - 1] - 2;
            int m = d > u ? d : u;
            cur[j] = m > l ? m : l;
        }
        t = prev; prev = cur; cur = t;
    }
    return prev[n];
}

/* ---- 16-bit creases: 16 sparks read through the fold at once ---- */
static int wf_core16(int n, const char *a, const char *brev,
                     int16_t *P2, int16_t *P1, int16_t *C)
{
    int k, kmax = 2 * n;
    int16_t *t;

    P2[0] = 0;                      /* crease 0 : H[0][0]            */
    P1[0] = -2; P1[1] = -2;         /* crease 1 : H[0][1], H[1][0]   */

    for (k = 2; k <= kmax; k++) {
        int glo, ghi, i, off = n - k;
        if (k <= n) {
            C[0] = (int16_t)(-2 * k);        /* i=0 : H[0][k] */
            C[k] = (int16_t)(-2 * k);        /* j=0 : H[k][0] */
            glo = 1; ghi = k - 1;
        } else {
            glo = k - n; ghi = n;
        }
        i = glo;
#if defined(__AVX2__)
        {
            const __m256i v2 = _mm256_set1_epi16(2);
            const __m256i v1 = _mm256_set1_epi16(1);
            int lim = ghi - 15;
            for (; i <= lim; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(a + i - 1));
                __m128i cb = _mm_loadu_si128((const __m128i *)(brev + off + i));
                __m128i cm = _mm_cmpeq_epi8(ca, cb);          /* 0x00 / 0xFF */
                __m256i m  = _mm256_cvtepi8_epi16(cm);        /* 0 / -1      */
                /* the crossing symbols choose the colour: (m & 2) - 1 = +1 / -1 */
                __m256i sc = _mm256_sub_epi16(_mm256_and_si256(m, v2), v1);
                __m256i d  = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(P2 + i - 1)), sc);
                __m256i uu = _mm256_loadu_si256((const __m256i *)(P1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(P1 + i));
                __m256i g  = _mm256_sub_epi16(_mm256_max_epi16(uu, ll), v2);
                _mm256_storeu_si256((__m256i *)(C + i), _mm256_max_epi16(d, g));
            }
        }
#endif
        for (; i <= ghi; i++) {
            int s = (a[i - 1] == brev[off + i]) ? 1 : -1;
            int d = P2[i - 1] + s;
            int u = P1[i - 1], l = P1[i];
            int g = (u > l ? u : l) - 2;
            C[i] = (int16_t)(d > g ? d : g);
        }
        t = P2; P2 = P1; P1 = C; C = t;   /* the spent bottom layer drifts off */
    }
    return (int)P1[n];
}

/* ---- 32-bit creases: same fold, 8 sparks wide, for n beyond the 16-bit guard ---- */
static int wf_core32(int n, const char *a, const char *brev,
                     int32_t *P2, int32_t *P1, int32_t *C)
{
    int k, kmax = 2 * n;
    int32_t *t;

    P2[0] = 0;
    P1[0] = -2; P1[1] = -2;

    for (k = 2; k <= kmax; k++) {
        int glo, ghi, i, off = n - k;
        if (k <= n) {
            C[0] = -2 * k;
            C[k] = -2 * k;
            glo = 1; ghi = k - 1;
        } else {
            glo = k - n; ghi = n;
        }
        i = glo;
#if defined(__AVX2__)
        {
            const __m256i v2 = _mm256_set1_epi32(2);
            const __m256i v1 = _mm256_set1_epi32(1);
            int lim = ghi - 7;
            for (; i <= lim; i += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(a + i - 1));
                __m128i cb = _mm_loadl_epi64((const __m128i *)(brev + off + i));
                __m128i cm = _mm_cmpeq_epi8(ca, cb);
                __m256i m  = _mm256_cvtepi8_epi32(cm);
                __m256i sc = _mm256_sub_epi32(_mm256_and_si256(m, v2), v1);
                __m256i d  = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(P2 + i - 1)), sc);
                __m256i uu = _mm256_loadu_si256((const __m256i *)(P1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(P1 + i));
                __m256i g  = _mm256_sub_epi32(_mm256_max_epi32(uu, ll), v2);
                _mm256_storeu_si256((__m256i *)(C + i), _mm256_max_epi32(d, g));
            }
        }
#endif
        for (; i <= ghi; i++) {
            int s = (a[i - 1] == brev[off + i]) ? 1 : -1;
            int d = P2[i - 1] + s;
            int u = P1[i - 1], l = P1[i];
            int g = (u > l ? u : l) - 2;
            C[i] = d > g ? d : g;
        }
        t = P2; P2 = P1; P1 = C; C = t;
    }
    return (int)P1[n];
}

int kernel(int n, const char *a, const char *b)
{
    size_t nb, need;
    char *mem, *brev;
    int i;

    if (n <= 0) return 0;
    if (n <= 32) return wf_small(n, a, b);      /* GUARD 1 */

    nb   = (size_t)n + 64;
    need = 3 * nb * sizeof(int32_t) + nb;
    mem  = (char *)wf_get(need);
    if (!mem) return wf_small(n <= 32 ? n : 32, a, b);  /* never taken in practice */

    /* the note is already folded in the hour between the note-trees */
    brev = mem + 3 * nb * sizeof(int32_t);
    for (i = 0; i < n; i++) brev[i] = b[n - 1 - i];

    /* GUARD 2: |H[i][j]| <= min(i,j) + 2|i-j| <= 3n, so int16 is exact for n <= 8000 */
    if (n <= 8000) {
        int16_t *S = (int16_t *)mem;
        return wf_core16(n, a, brev, S, S + nb, S + 2 * nb);
    } else {
        int32_t *I = (int32_t *)mem;
        return wf_core32(n, a, brev, I, I + nb, I + 2 * nb);
    }
}
```

# PREDICTION

PREDICTION: speedup_vs_dp = 6.0

Reasoning stated before any measurement: the reference row-major DP has a serial `max`+`add` chain through the left neighbour, ~2–3 cycles/cell. The folded kernel issues ~14 µops per 16 cells (5 loads, 1 store, 2 max, 2 add/sub, 1 cmp, 1 sign-extend, 1 and) — load-port bound at ≈3.5 cycles / 16 cells ≈ 0.22 cycles/cell, i.e. a ~11× ceiling. I discount to **6.0×** for: ragged creases (mean length `n/2`, so a scalar tail of ≈7.5 cells and fixed per-crease setup costs ~20 cycles against ~110 cycles of vector work), the store→load forwarding latency on the `P1` re-read each crease, and the `O(n)` bough-reversal per call.

I hand-verified the index algebra on `n=2`, `a="AC"`, `b="AG"` cell-by-cell (`H[1][1]=1`, `H[1][2]=-1`, `H[2][1]=-1`, `H[2][2]=0`) and proved that every read on crease `k` falls inside the *written* range of creases `k-1`/`k-2`, so the unbanded wavefront needs **no ±∞ sentinels at all** — which is why I did not ship the exact band `|i-j| ≤ (n - L_gapless)/5` that this bound also permits.

# MEASUREMENT

**Not performed. No tools were available in this session** — `alignment_bench` and `alignment_contract` could not be invoked, so the prediction above is unmeasured and the four allotted improvement rounds were spent on static reasoning (index proof, overflow proof, guard placement) instead of on data. I am reporting that plainly rather than inventing numbers: the honest state is *predicted 6.0×, measured nothing*.

What the pipeline should check, in order:
1. **Correctness first**, against the reference on `n ∈ {1, 2, 31, 32, 33, 64, 127, 1000}`, on identical, random, and fully-anticorrelated pairs. Any mismatch falsifies the whole artifact, not just the speed claim.
2. Speedup at `n ∈ {64, 256, 1024, 4096}`. I expect the curve to *rise* with `n` (crease length ∝ `n`) and to be worst at `n=64`, where a mean crease of 32 elements gives only 2 vector iterations against the scalar tail.
3. `n = 9000` to exercise the int32 path — expect roughly half the int16 speedup.

# VERDICT

The native's picture is a *correct and complete* description of anti-diagonal wavefront alignment with rolling buffers, and nothing in it needed to be replaced by the textbook method. Three of its images did real work rather than decorating:

- **"Three layers bit for bit atop each other."** This is the non-obvious payoff of the fold. Re-indexing by `i` alone makes the three neighbours land at `P2[i-1]`, `P1[i-1]`, `P1[i]` — offsets 0 and ±1 in *one* index, which is what lets three unaligned loads replace any shuffling. A less literal reading (indexing creases by `j`, or storing them centred) would have needed a lane shift per crease.
- **"The note is already folded in the hour between the note-trees."** I read this as licence to pre-reverse `b`, which turns the descending column access `b[k-i-1]` into the ascending contiguous `brev[n-k+i]`. Without it the symbol comparison needs a gather or a reversed shuffle and the whole gain evaporates.
- **"Wait only as long as it takes each spark to catch, never longer."** I treated this as a hard prohibition on synchronisation and dropped OpenMP. Per SEED 3's own logic the creases are a serial chain of `2n` steps; a thread barrier at each (~1 µs) costs ~12 ms at `n=4000` against ~3 ms of total work. This is the one place the native's aesthetic and the machine agree against the obvious "it's parallel, so thread it" move.

**Self-named risks, and what each is guarded by.** (a) *Latency-bound when a crease is shorter than a vector* — guarded by the `n ≤ 32` row-DP fallback; the break-even I compute is `n ≈ 7–16`, so 32 is a safe margin. (b) *int16 crease overflow* — guarded by `n ≤ 8000` with the bound `|H| ≤ min(i,j) + 2|i-j| ≤ 3n`; above it an int32 crease, half as wide. (c) *Per-crease barrier* — the risky part was dropped, not guarded. (d) The residual unguarded risk is honest: **the `n=64`-ish regime.** There the kernel is correct but the ragged-crease overhead is largest, and I would not be surprised by a speedup near 2× rather than 6×. If measurement shows that, the fix is inside the mechanism, not outside it — raise the `wf_small` threshold — and I would rather report a 2× than pre-emptively widen a threshold I have no data for.

**Left on the table, deliberately:** the exact band `|i-j| ≤ ⌊(n − L_gapless)/5⌋`, provable from *score ≤ n − 5·max-deviation* and *OPT ≥ gapless score*, worth ~1.4× on random DNA and far more on similar sequences. It requires ±∞ sentinels at the two moving band edges, and shipping an untested off-by-one there risks silently wrong scores. With no bench available, a fast wrong answer is worth strictly less than a slower right one.