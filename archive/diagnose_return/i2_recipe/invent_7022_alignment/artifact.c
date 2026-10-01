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
