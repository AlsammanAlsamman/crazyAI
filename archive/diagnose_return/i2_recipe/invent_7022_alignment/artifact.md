# DICTIONARY

| Thing in the world | Concrete computational thing |
|---|---|
| Smoothed sand-tray, lattice of scratched junctions | The index space `(i,j)`, `0…n × 0…n`, of a Needleman–Wunsch matrix. **Never materialized.** |
| First knot-cord, north edge, west→east | String `b`, one knot per byte, index `j` |
| Second knot-cord, west edge, north→south | String `a`, one knot per byte, index `i` (the recipe is symmetric in the cords; this only fixes which index runs along a row) |
| One knot | One character/byte over `{A,C,G,T}` |
| Furrow-point / junction `(i,j)` | The DP cell `H[i][j]` = best global score of `a[0..i)` against `b[0..j)` |
| "No junction may be added or moved once scratched" | Fixed scratch buffers, no realloc — **and no banding/pruning**: every cell is visited. (A rigorous band `|i−j| ≤ (n−L)/5` exists and is worth ~1.7×; the recipe forbids it, so I did not take it.) |
| Mound, its height | The integer score standing at a cell; a *grain* = one score unit, counted in `int16` (proved safe: `−2n ≤ H ≤ n`) |
| Mound of height nothing | `0` |
| Near corner | `H[0][0] = 0` |
| Edge mounds, "never kicked flat" | First row/column = boundary conditions, written once, only read afterwards (**corrected**, see step 2) |
| The stone with three weights, never changed mid-crawl | Three constants `+1, −1, −2`; immutability is exactly what licenses hoisting them into broadcast registers outside every loop |
| **A single ink-worm** | **One thread.** One program counter. The seed says *single*, so no OpenMP — and the measurement argument agrees (a parallel-for per anti-diagonal costs 2n barriers ≈ 10 ms at n=4096 against ~0.3 ms of real work) |
| Worm standing on a junction | Current cell index |
| Three faces | Predecessors `(i−1,j−1)`, `(i−1,j)`, `(i,j−1)` |
| Sideways crossing / a cord let to slip | A gap column, price `−2` |
| Reading the facing pair | Byte compare `a[i−1]==b[j−1]` → `+1 / −1` |
| Three tallies, tallest of them | Three candidate scores, `max` |
| "Lower than nothing → take nothing" | The Smith–Waterman zero-clamp (**corrected away**, step 6) |
| "Kick the old flat, sweep its sand away entirely; a beaten mound is kept nowhere and remembered by nothing" | A monotone max-join update with **no history**. Two consequences: (a) in a topological order it degenerates to a plain store; (b) "kept nowhere" is the literal license to hold only the three most recent anti-diagonals — O(n) memory, L1-resident, instead of a tray |
| Pressed mark naming the face | Traceback pointer. Computed implicitly as the argmax of the `max`; **not stored** (flagged deviation — unobservable through `int kernel(...)`, and it would be the kernel's only O(n²) write stream) |
| "East along the row, then down to the next row" | Row-major traversal (the wide fallback path implements this verbatim) |
| "Doubles back… no junction settled before all three faces stand finished" | The face-readiness partial order — **the real constraint on time, and the recipe's most valuable clause**: it licenses *any* topological order, hence the anti-diagonal wavefront, hence SIMD |
| "No junction holding a mound it could raise by returning" | The fixpoint test. Provably vacuous here: both indices strictly decrease across every face, so the face relation is a DAG and one wavefront pass is already the least fixpoint. The recipe's `while(changed)` collapses to one pass |
| "Curls back… to test a junction twice" (seed) | A second verification pass. Discharged by proof, not by runtime: a literal second pass costs exactly 2× for zero information |
| Tallest mound anywhere | The Smith–Waterman answer (**corrected**, step 11: the far-corner mound) |
| **Time** | The anti-diagonal number `d = i+j`. One tick = one `d`. Every junction on a tick is mutually independent — that is the entire speed story |
| **Processor** | One core; 16 `int16` lanes of one AVX2 register = 16 junctions settled per instruction. The worm's body is 16 junctions wide |
| **Memory** | 3 anti-diagonals of `int16` (`3·(n+17)·2` bytes) + one reversed copy of `b`, so a tick's facing pairs are two *contiguous* byte runs |
| What flows | Mounds, along increasing `d` (only the last two ticks survive) |
| What stays still | The lattice, the stone, the two cords, `rb` |

**Three corrections (the recipe as given is Smith–Waterman; the contract asks for Needleman–Wunsch).** The three steps that must change are exactly the three that define locality — a nice confirmation that nothing else in the recipe is wrong:
1. **Step 2**: a free zero ground floor along both edges → the k-th edge mound is `−2k`. Running one cord past the other's beginning is still a sideways crossing and the stone must be paid. (Near corner keeps height nothing, exactly as written.)
2. **Step 6**: delete the "if lower than nothing, take nothing" floor. That clause is the local aligner's fresh start and is precisely what makes a global score wrong.
3. **Step 11**: "the single tallest mound anywhere" → the mound at the far corner. A global reading must lay both cords entirely.

**Two flagged deviations, both permitted by the contract:** step 8's marks are not materialized (score-only return type), and step 12's traceback walk is not performed. Step 9's row-major order is replaced by the wavefront order that step 10 explicitly licenses; both are topological orders of the same face-readiness relation, and the step-7 fixpoint is unique, so the mounds are bit-identical. The row-major order survives verbatim in the wide fallback, which doubles as a cross-check.

# ARTIFACT

```c
#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* The same crawl, spelled with wide (32-bit) grain counts and in the
   recipe's own row-major worm order (step 9 verbatim).  Reached only when a
   mound's height can no longer be counted in 16-bit grains (|H| <= 2n+2, so
   n > 16000 comes here), or when malloc fails.  It must give identical
   mounds: the step-7 update is a monotone max-join on a DAG of faces, so its
   fixpoint is unique and the visiting order cannot change any height. */
static int crawl_wide(int n, const char *a, const char *b)
{
    int *base = (int *)malloc(2u * (size_t)(n + 1) * sizeof(int));
    if (!base) return 0;
    int *prev = base, *cur = base + (n + 1);
    for (int j = 0; j <= n; ++j) prev[j] = -2 * j;      /* step 2, ramped */
    for (int i = 1; i <= n; ++i) {
        const char ai = a[i - 1];
        cur[0] = -2 * i;
        for (int j = 1; j <= n; ++j) {
            int t = prev[j - 1] + ((ai == b[j - 1]) ? 1 : -1);
            int u = prev[j] - 2;   if (u > t) t = u;
            u = cur[j - 1] - 2;    if (u > t) t = u;
            cur[j] = t;
        }
        int *tmp = prev; prev = cur; cur = tmp;
    }
    int r = prev[n];
    free(base);
    return r;
}

int kernel(int n, const char *a, const char *b)
{
    /* ---------------- step 1: smooth the tray, scratch the lattice -------
     * Two knot-cords, one knot per symbol: the north cord is b (knots
     * j = 0..n), the west cord is a (knots i = 0..n).  Junction (i,j) is
     * the furrow-point where knot i of the west cord faces knot j of the
     * north cord; its mound will be the score of a[0..i) against b[0..j).
     * The lattice is fixed here and nothing is added or moved afterwards:
     * no cell is skipped, no band, no pruning, no reallocation.
     * The tray is never held whole.  A junction's three faces (step 4) all
     * lie on the two anti-diagonals behind it, and step 7 sweeps beaten
     * sand away entirely -- a beaten mound is "remembered by nothing" --
     * so exactly three anti-diagonals of sand ever stand: 3*(n+17) int16,
     * which is L1-resident for every n this path accepts.
     * Laying the north cord's knots a second time in reverse order (rb) is
     * pure layout: it makes the facing pairs of one anti-diagonal two
     * contiguous byte runs, readable as one 16-byte load each.          */
    if (n <= 0) return 0;
    if (n > 16000) return crawl_wide(n, a, b);   /* grain-size bookkeeping */

    const int dmax   = 2 * n;                    /* the far corner's tick */
    const int stride = n + 17;                   /* one anti-diagonal + slop */

    short          sand_stack[3 * (1024 + 17)];
    unsigned char  rev_stack[1024 + 16];
    void          *heap = 0;
    short         *sand = sand_stack;
    unsigned char *rb   = rev_stack;
    if (n > 1024) {
        heap = malloc((size_t)3 * stride * sizeof(short) + (size_t)n + 16);
        if (!heap) return crawl_wide(n, a, b);
        sand = (short *)heap;
        rb   = (unsigned char *)(sand + 3 * stride);
    }
    for (int k = 0; k < n; ++k) rb[k] = (unsigned char)b[n - 1 - k];

    short *f2 = sand;                 /* sand standing on anti-diagonal d-2 */
    short *f1 = sand + stride;        /* sand standing on anti-diagonal d-1 */
    short *f0 = sand + 2 * stride;    /* sand being dropped on anti-diagonal d */

    /* ---------------- step 2: the ground floor ---------------------------
     * CORRECTION (smallest change to this step).  The recipe drops a mound
     * of height nothing all along the north and west edges.  That free
     * ground floor is a local aligner's, and with it the number produced is
     * a Smith-Waterman score, not the global score the contract names.
     * Smallest repair: the near corner keeps height nothing exactly as
     * written, but the k-th junction of either edge carries a mound of
     * height minus k slip prices (-2k) -- letting one cord run past the
     * other's beginning is still a sideways crossing, and the stone must be
     * paid for it.  These edge mounds are still never kicked flat: they are
     * written once, below and at the tail of each tick, and only read after.
     * Here we seed the first two ticks; edge mounds for later ticks are
     * dropped as those ticks come round (see the tail of the step 9 loop). */
    f2[0] = 0;                        /* near corner, height nothing        */
    f1[0] = -2;                       /* H[0][1]: one slip paid             */
    f1[1] = -2;                       /* H[1][0]: one slip paid             */

    /* ---------------- step 3: the stone beside the tray ------------------
     * Three weights agreed before the worm wakes and never changed
     * mid-crawl.  Because they cannot change, they live in three broadcast
     * registers hoisted out of every loop, and are read at zero cost.     */
#if defined(__AVX2__)
    const __m256i w_same = _mm256_set1_epi16(1);   /* facing knots equal    */
    const __m256i w_diff = _mm256_set1_epi16(-1);  /* facing knots differ   */
    const __m256i w_slip = _mm256_set1_epi16(2);   /* price of one slip     */
#endif

    /* ---------------- step 9 + step 10: where the worm goes next ---------
     * Step 9 as written walks east along a row, then down to the next row.
     * Step 10 is the clause that actually constrains time: no junction may
     * be settled before all three of its faces stand finished, and the worm
     * is free to double back across ground already crossed to make that so.
     * Step 7's update is monotone and keeps no history, so ANY order obeying
     * step 10 reaches the same mounds.  I take the wavefront order: tick
     * d = i + j, junctions i = max(1,d-n) .. min(n,d-1).  Every face of a
     * junction on tick d lies on tick d-1 or d-2, already finished, so the
     * worm never has to double back at all -- step 10's curl is satisfied
     * statically, and all junctions of one tick are mutually independent,
     * which is what lets sixteen of them be settled by one instruction.
     * (C forces a loop header above its body, so the step 4-8 blocks appear
     * below this one; dynamically the worm still does 4->8 at a junction and
     * step 9 then moves it.)                                              */
    for (int d = 2; d <= dmax; ++d) {
        const int ilo = (d - n > 1) ? (d - n) : 1;
        const int ihi = (d - 1 < n) ? (d - 1) : n;
        int i = ilo;

#if defined(__AVX2__)
        const int off = n - d;
        for (; i + 15 <= ihi; i += 16) {
            /* step 4: the three faces of these sixteen junctions.  The
               diagonal face (both knots stepped together) is tick d-2 at
               row i-1; the two sideways faces are tick d-1 at rows i-1
               (a knot of the north cord let to slip) and i (a knot of the
               west cord let to slip). */
            const __m256i m_diag  = _mm256_loadu_si256((const __m256i *)(f2 + i - 1));
            const __m256i m_north = _mm256_loadu_si256((const __m256i *)(f1 + i - 1));
            const __m256i m_west  = _mm256_loadu_si256((const __m256i *)(f1 + i));

            /* step 5: read the facing pair at each junction and add to each
               face's mound the grains the stone names for that manner of
               arrival -- the diagonal mound plus same/different, each
               sideways mound minus the slip price. */
            const __m128i ka     = _mm_loadu_si128((const __m128i *)(const void *)(a + i - 1));
            const __m128i kb     = _mm_loadu_si128((const __m128i *)(const void *)(rb + i + off));
            const __m256i facing = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ka, kb));
            const __m256i t_diag = _mm256_add_epi16(m_diag,
                                     _mm256_blendv_epi8(w_diff, w_same, facing));
            const __m256i t_side = _mm256_sub_epi16(_mm256_max_epi16(m_north, m_west),
                                                    w_slip);

            /* step 6: take the tallest of those three tallies.
               CORRECTION (smallest change to this step): the recipe's floor
               -- "if it is lower than nothing, take nothing instead" -- is
               deleted.  That clause lets the worm begin a fresh path at any
               junction, which is exactly a local alignment; a global reading
               must drag its losing trail behind it.  No floor is applied. */
            const __m256i mound = _mm256_max_epi16(t_diag, t_side);

            /* step 7: compare with any mound already standing here and keep
               the taller.  In this order each junction is settled exactly
               once and its old mound has already been swept away, so the
               comparison is vacuous and the new mound simply stands.  (The
               monotone max-join is what makes that substitution safe, and
               "a beaten mound is kept nowhere" is what lets these three
               ticks be the only sand in memory.)
               step 8: the mark naming which face was come from is the argmax
               of the max just taken; it is not pressed into the sand,
               because the contract returns only the height and step 12's
               trail-walk is therefore unobservable.  Storing it would be the
               only O(n^2) write stream in the crawl. */
            _mm256_storeu_si256((__m256i *)(f0 + i), mound);
        }
#endif
        /* steps 4-8 again, one junction at a time, for the ragged end of a
           tick (and for the whole tick when no 256-bit shovel exists). */
        for (; i <= ihi; ++i) {
            const int m_diag  = f2[i - 1];                       /* step 4 */
            const int m_north = f1[i - 1];
            const int m_west  = f1[i];
            const int t_diag  = m_diag +                          /* step 5 */
                                ((a[i - 1] == b[d - i - 1]) ? 1 : -1);
            const int t_side  = ((m_north > m_west) ? m_north : m_west) - 2;
            f0[i] = (short)((t_diag > t_side) ? t_diag : t_side); /* 6,7,8  */
        }

        /* step 2, continued: this tick's two ground-floor mounds, dropped
           where the cords have not yet begun, and never kicked flat. */
        if (d <= n) { f0[0] = (short)(-2 * d); f0[d] = (short)(-2 * d); }

        { short *t = f2; f2 = f1; f1 = f0; f0 = t; }   /* step 9: next tick */
    }

    /* ---------------- step 11: is the crawl done? ------------------------
     * The far corner is settled.  The recipe now wants the worm to curl
     * back and check that no junction anywhere holds a mound it could
     * raise by returning to it.  It cannot: across every face both knot
     * indices strictly decrease, so the face relation is acyclic, every
     * junction was settled from faces already final, and one pass is
     * already the least fixpoint of step 7.  The curl is discharged by
     * proof rather than by a second sweep, which would cost exactly 2x for
     * zero information.
     * CORRECTION (smallest change to this step): not the tallest mound
     * standing anywhere -- that is the local answer -- but the mound at the
     * far corner, the only junction at which both cords have been laid
     * whole.  After the last rotation that mound is f1[n].               */
    const int answer = f1[n];

    /* ---------------- step 12: the number asked for ----------------------
     * The height of that mound is the agreement between the two strings.
     * The trail-walk in step 12's second sentence needs the marks of step 8
     * and is not requested by the contract, so it is not performed.       */
    if (heap) free(heap);
    return answer;
}
```

# PREDICTION

Reasoning before any number: the reference DP's row-major inner loop carries a serial dependence `H[i][j] ← H[i][j−1]`, a ~2-cycle add+max chain, so it cannot be vectorized as written and costs ≈2–3 cycles/cell. The wavefront body is 13 µops per 16 cells ≈ 3.3 cycles ≈ 0.21 cycles/cell at full width; ragged tick ends near the two corners and per-tick bookkeeping should inflate that to ~0.4–0.6 cycles/cell for n in the low thousands. That is 5–12×. I take the middle and commit:

**PREDICTION: speedup_vs_dp = 8**

Secondary predictions, each falsifiable: (a) the kernel is bit-exact against the reference for all n (the three corrections are the *only* semantic edits); (b) a literal second relaxation pass would change no mound — the step-11 curl is provably vacuous; (c) OpenMP would make it slower, not faster, for n ≤ 8192.

# MEASUREMENT

**Not measured — and I will not dress this up.** This session was invoked with no tools: `alignment_bench` and `alignment_contract` were unavailable, so the prediction above stands unverified and the four improvement rounds I was allotted were never spent. What I can report honestly is only what was checked by hand:

- **Hand-traced correctness**: `n=1` (both facing cases → ±1), and `n=2` with `a="AC"`, `b="AG"` traced cell-by-cell through the diagonal buffers and their rotation against the textbook matrix (`H[1][1]=1, H[1][2]=−1, H[2][1]=−1, H[2][2]=0`); the kernel's `f1[n]` comes out `0`. Correct.
- **Index-safety proof**: for every vectorized block the reads `f2[i−1 … i+14]`, `f1[i−1 … i+15]`, `a[i−1 … i+14]`, `rb[i+off … i+off+15]` were shown to lie inside each buffer's written/valid range and inside both strings (no read past `a[n−1]`, `b[n−1]`, not even the NUL). No stale or uninitialized entry is ever read, because each tick writes exactly its own valid index range (interior plus the two edge mounds when `d ≤ n`).
- **Overflow proof**: `−2n ≤ H[i][j] ≤ n`, intermediates within `±(2n+2)`, so `int16` is safe for `n ≤ 16382`; the gate is set at 16000 and larger `n` falls through to the 32-bit row-major path.
- **Unmeasured risk I'd look at first**: the ragged-end penalty. Ticks shorter than 16 junctions run entirely scalar, and there are ~30 of them at each corner; if the bench uses small `n` (≤128) the wavefront's per-tick overhead could eat most of the win and the measured speedup could land at 2–3× rather than 8×. The second thing to check is whether the bench's reference is a full-matrix DP (cache-bound → speedup likely *above* 8) or a two-row rolling DP (→ at or below 8).

# VERDICT

The recipe is Smith–Waterman told by someone who has never needed a global score, and it was wrong for this contract in exactly three places — steps 2, 6, and 11 — which are precisely the three clauses that make an aligner local. Everything else in it is not just right, it is better than the textbook statement of Needleman–Wunsch, and two clauses paid for themselves:

- **Step 7's "a beaten mound is kept nowhere and remembered by nothing"** is a stronger statement than "use two rows". It says the ledger *is* the sand, so memory collapses to three anti-diagonals — O(n) int16, L1-resident at every accepted size, no matrix at all.
- **Step 10 is the real find.** By stating the constraint as face-readiness plus permission to double back, and step 11 as a fixpoint test rather than a completed schedule, the native has written down chaotic relaxation on a monotone operator — and thereby licensed *any* topological order, which is what makes the anti-diagonal wavefront legal and the whole thing vectorizable. A textbook "for i: for j:" hides this. The recipe hands it over.
- **The seed's "single ink-worm"** independently argues for what the arithmetic also argues for: one thread. Per-tick `parallel for` would pay 2n barriers against a few hundred microseconds of work.
- Two of the recipe's clauses cost nothing at runtime because they are provable: the step-11 curl ("test a junction twice") is vacuous on a DAG, and step 7's comparison degenerates to a store. I kept both as claims rather than code, and said so.
- The recipe's "no junction may be added or moved" / "every junction is visited" **forbade** the one classical trick I could otherwise prove sound here: a band `|i−j| ≤ (n−L)/5` from the bound `score ≤ n − 5k`. It is worth only ~1.7× for random DNA, so obeying the native cost little — but I want it on the record that I obeyed rather than quietly took it.

The honest bottom line: the artifact is complete, hand-verified for correctness and memory safety, and predicted at 8× — but **unmeasured**, because no bench existed in this session. Until `alignment_bench` runs, 8 is a claim, not a result.