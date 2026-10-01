```c
        int *cw = buf + (s % 6) * width;
        for (int t = 0; t < width; t++) cw[t] = -1;
        int dm = s / PE;
        if (dm > maxd - 1) dm = maxd - 1;
        if (dm > n) dm = n;
        const int *w4 = (s >= PX) ? buf + ((s - PX) % 6) * width : NULL;
        const int *w5 = (s >= PE) ? buf + ((s - PE) % 6) * width : NULL;

        for (int d = -dm; d <= dm; d++) {
            int bv = -1;
            if (w4) {                                   /* red knot: i+1,j+1 */
                int hp = w4[off + d];
                if (hp >= 0 && hp < n && hp - d < n) { int c = hp + 1; if (c > bv) bv = c; }
            }
            if (w5) {                                   /* the horse slips    */
                int hp = w5[off + d - 1];               /* gap in b: i+1      */
                if (hp >= 0 && hp < n) { int c = hp + 1; if (c > bv) bv = c; }
                hp = w5[off + d + 1];                   /* gap in a: i same   */
                if (hp >= 0 && hp - (d + 1) < n) { int c = hp; if (c > bv) bv = c; }
            }
            if (bv < 0) continue;
            int hh = lotus_run(a, b, n, bv, bv - d);
            cw[off + d] = hh;
            if (d == 0 && hh == n) { free(buf); return n - s / 2; }
        }
    }
    free(buf);
    return n - Pub / 2;                                 /* Pub is achievable */
}

/* ---------- all furrows resident, paced together on the flute ----------
   anti-diagonal s, cells indexed by i:
   D_s[i] = max( D_{s-2}[i-1] + petal/knot, D_{s-1}[i-1] + gap,
                 D_{s-1}[i] + gap )
   only offsets 0 and -1 -> contiguous, branch-free, 16 houses per note.  */
static int furrows_lockstep(int n, const char *a, const char *b, int w)
{
    if (w < 1) w = 1;
    if (w > n) w = n;
    const int IO = 4;
    int m = n + 8;
    int16_t *blk = (int16_t *)malloc((size_t)3 * (size_t)m * sizeof(int16_t));
    char *br = (char *)malloc((size_t)n + 32);
    if (!blk || !br) { free(blk); free(br); return INT_MIN; }
    for (int k = 0; k < n; k++) br[k] = b[n - 1 - k];   /* reversed: both reads run forward */
    memset(br + n, 0, 32);
    for (int t = 0; t < 3 * m; t++) blk[t] = NEG16;

    int16_t *D2 = blk, *D1 = blk + m, *Dc = blk + 2 * m;
    D1[IO + 0] = 0;                                     /* D_0[0] = 0 */

    for (int s = 1; s <= 2 * n; s++) {
        int lo = (s > w) ? ((s - w + 1) >> 1) : 0;
        if (lo < s - n) lo = s - n;
        int hi = (s + w) >> 1;
        if (hi > n) hi = n;
        if (hi > s) hi = s;

        if (lo == 0) Dc[IO + 0] = (int16_t)(D1[IO + 0] + SGAP);
        if (hi == s) Dc[IO + s] = (int16_t)(D1[IO + s - 1] + SGAP);

        int i0 = lo > 1 ? lo : 1;
        int i1 = hi < s - 1 ? hi : s - 1;
        int cnt = i1 - i0 + 1;
        if (cnt > 0) {
            const char *pa = a + (i0 - 1);
            const char *pb = br + (n - s + i0);
            int16_t *cp = Dc + IO + i0;
            const int16_t *q2 = D2 + IO + i0 - 1;
            const int16_t *qa = D1 + IO + i0 - 1;
            const int16_t *qb = D1 + IO + i0;
            int k = 0;
#if defined(__AVX2__)
            const __m256i vg  = _mm256_set1_epi16(SGAP);
            const __m256i vm1 = _mm256_set1_epi16(-1);
            for (; k + 16 <= cnt; k += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(pa + k));
                __m128i cb = _mm_loadu_si128((const __m128i *)(pb + k));
                __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                __m256i sb = _mm256_sub_epi16(vm1, _mm256_add_epi16(eq, eq));
                __m256i dg = _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(q2 + k)), sb);
                __m256i up = _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(qa + k)), vg);
                __m256i lf = _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(qb + k)), vg);
                _mm256_storeu_si256((__m256i *)(cp + k),
                                    _mm256_max_epi16(dg, _mm256_max_epi16(up, lf)));
            }
#endif
            for (; k < cnt; k++) {
                int i = i0 + k;
                int sb = (a[i - 1] == br[n - s + i]) ? SMATCH : SMISMATCH;
                int dg = D2[IO + i - 1] + sb;
                int up = D1[IO + i - 1] + SGAP;
                int lf = D1[IO + i] + SGAP;
                int bst = dg; if (up > bst) bst = up; if (lf > bst) bst = lf;
                Dc[IO + i] = (int16_t)bst;
            }
        }
        int glo = lo - 2; if (glo < -IO) glo = -IO;
        for (int t = glo; t < lo; t++) Dc[IO + t] = NEG16;
        int ghi = hi + 2; if (ghi > n + 3) ghi = n + 3;
        for (int t = hi + 1; t <= ghi; t++) Dc[IO + t] = NEG16;

        int16_t *tmp = D2; D2 = D1; D1 = Dc; Dc = tmp;
    }
    int r = D1[IO + n];
    free(blk); free(br);
    return r;
}

/* ---------- plain two-row NW: tiny n, and the very-large-n safety net --- */
static int nw_two_row(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = j * SGAP;
    for (int i = 1; i <= n; i++) {
        cur[0] = i * SGAP;
        char ai = a[i - 1];
        for (int j = 1; j <= n; j++) {
            int d = prev[j - 1] + ((ai == b[j - 1]) ? SMATCH : SMISMATCH);
            int u = prev[j] + SGAP;
            int l = cur[j - 1] + SGAP;
            int bst = d; if (u > bst) bst = u; if (l > bst) bst = l;
            cur[j] = bst;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ====================== the workroom ====================== */
int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n <= 64) return nw_two_row(n, a, b);

    /* --- weigh the two cheap fists, O(n) --- */
    int H  = knots_flat(n, a, b);
    int P0 = 4 * H;                                   /* no slip            */
    int m1 = knots_one_slip(n, a, b);
    int m2 = knots_one_slip(n, b, a);
    if (m2 < m1) m1 = m2;
    int P1 = 10 + 4 * m1;                             /* exactly one slip   */
    int Pub = P0 < P1 ? P0 : P1;

    /* --- the fist in hand is already the answer: 10g <= Pub <= 19 => g<=1 */
    if (Pub <= 19) return n - Pub / 2;

    /* --- which regime? --- */
    int w = Pub / 10;                                 /* |i-j| <= g <= Pub/10 */
    if (w < 1) w = 1;
    if (w > n) w = n;
    double wfa_cost  = 0.45 * (double)Pub * (double)Pub + 3.0 * (double)n;
    double band_cost = (double)(2 * w + 1) * (double)n / 12.0;

    if (wfa_cost < band_cost) {                       /* light fists: walk the horse */
        int r = fists_by_weight(n, a, b, Pub);
        if (r != INT_MIN) return r;
    }
    if (n <= I16_MAX_N) {                             /* heavy fists: all furrows at once */
        int r = furrows_lockstep(n, a, b, w);
        if (r != INT_MIN) return r;
    }
    return nw_two_row(n, a, b);                       /* int16 range guard   */
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 12**

(Stated before any measurement. Reasoning: the reference pays ~2 cycles/cell over `n²` cells with a serial `cur[j-1]` dependence. The heavy-fist path pays ~4 cycles per 16 cells with no loop-carried dependence over `≈0.8n²` cells at `P_ub ≈ 3n` for unrelated DNA → ~10–15×. If the benchmark uses related sequences, the `Pub ≤ 19` `O(n)` exit or the wavefront path should give 10²–10³×, which I am *not* predicting because I do not know the generator.)

# MEASUREMENT

Not run in this session — no tools were available to me here (`alignment_bench` / `alignment_contract` could not be invoked). I am reporting that plainly rather than inventing numbers. What the pipeline should find, per regime, and what would falsify me:

| regime (detected at runtime) | path taken | expected vs reference |
|---|---|---|
| `P_ub ≤ 19` (≤4 knots, ≤1 slip) | `O(n)` fist in hand, no DP | 10²–10³× |
| `0.45P² + 3n < (2w+1)n/12` (≲1% divergence) | wavefront `fists_by_weight` | 20–200× |
| otherwise (random/unrelated DNA) | `furrows_lockstep`, band `w = P_ub/10 ≈ 0.3n` | 8–15× |
| `n ≤ 64` or `n > 12000` | plain two-row NW | ~1–2× |

If the measured figure lands near 8–15× the benchmark is in the dissimilar regime and my prediction of 12 is roughly right; if it lands above 50× the benchmark uses related sequences and my prediction was badly *low*; if it lands below 5× then either `-mavx2` was not active or the band is not tightening (`P_ub ≈ 4n`), and the honest conclusion is that the metaphor bought nothing beyond SIMD.

# VERDICT

**What the native's mechanism actually is, computationally:** indexing the computation by *fist weight* rather than by grid position, and letting the fish take every dominated state. That is the WaveFront/greedy-diagonal family, and it is the core here — not an accessory to a DP table. The grid is never allocated on that path; only six thin fronts exist at any moment. The two supporting seeds are literal too: lotus petals are genuinely free (match runs are skipped 8 bytes per instruction via XOR+ctz, which is *only* sound because "cool and easy" means zero cost), and the flute's one-note-per-house is what licenses 16 houses per instruction.

**What survived contact with correctness.** The native's own family — one slip, every doorway — is not complete for a global alignment, because the road pins both ends: a slip that opens must close. Completing it to pairs of doorways `p ≤ q` keeps it `O(n)` (running minimum) and makes it *exact* for the whole `g ≤ 1` world. That is not a patch; it is what the metaphor already implied and I had been reading too loosely. It buys a genuine `O(n)` exact exit whenever `P_ub ≤ 19`, with a proof (`10g ≤ P_opt ≤ P_ub`), and it supplies the weight that sizes everything downstream.

**Stated risk, and how it is guarded.** The wavefront is *worse* than a banded sweep when the fists are heavy: its work grows as `P²` while the sweep's grows as `nP/5`, so for unrelated DNA (`P ≈ 3n`) it would be several times slower than the reference, not faster. I do not ship that. The regime test is a cost comparison on a quantity I have already computed and which is a *proven* upper bound, so both branches carry provable work bounds rather than hopes: the wavefront can never run past `s = P_ub`, and the band `w = ⌊P_ub/10⌋` provably contains an optimal path. The `int16` lane width carries its own range risk (`|value| ≤ 2n`), guarded by `n ≤ 12000` with a two-row `int32` fallback. No thread parallelism: at these sizes the metaphor's units of work (one fist, one anti-diagonal) are `O(n)` and too small to pay for a barrier per flute note, so I stayed with vectorization as instructed.

**Where I am least confident.** The extension cost inside the wavefront is the one term I cannot bound tightly a priori (worst case `0.4·P·n`); I chose a conservative crossover (`P_ub ≲ n/27`) so that even a pessimal extension keeps the wavefront in the same absolute time class as the sweep it displaced. And if the benchmark generates independent random sequences, the wavefront path will never execute and the measured number will be almost entirely the banded lockstep sweep — in which case the native's contribution reduces to the band width and the `O(n)` exit, and I would say so rather than claim the wavefront earned it.