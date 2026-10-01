/* Dijkstra by the Nightingale recipe: linear-scan selection over the
   candidate slots, AVX2-hardened. No heap anywhere. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* --- machine-level helpers for step 3 (the scan for the quietest carving) --- */

static double lat_min(const double *fd, int len)
{
    double best = INFINITY;
    int i = 0;
#if defined(__AVX2__)
    if (len >= 16) {
        __m256d b0 = _mm256_set1_pd(INFINITY), b1 = b0, b2 = b0, b3 = b0;
        for (; i + 16 <= len; i += 16) {
            b0 = _mm256_min_pd(b0, _mm256_loadu_pd(fd + i));
            b1 = _mm256_min_pd(b1, _mm256_loadu_pd(fd + i + 4));
            b2 = _mm256_min_pd(b2, _mm256_loadu_pd(fd + i + 8));
            b3 = _mm256_min_pd(b3, _mm256_loadu_pd(fd + i + 12));
        }
        b0 = _mm256_min_pd(b0, b1);
        b2 = _mm256_min_pd(b2, b3);
        b0 = _mm256_min_pd(b0, b2);
        __m128d lo = _mm256_castpd256_pd128(b0);
        __m128d hi = _mm256_extractf128_pd(b0, 1);
        lo = _mm_min_pd(lo, hi);
        lo = _mm_min_sd(lo, _mm_unpackhi_pd(lo, lo));
        best = _mm_cvtsd_f64(lo);
    }
#endif
    for (; i < len; i++) if (fd[i] < best) best = fd[i];
    return best;
}

static double lat_min_par(const double *fd, int len)
{
#ifdef _OPENMP
    if (len >= (1 << 16)) {            /* only when the scan is genuinely long */
        double g = INFINITY;
        #pragma omp parallel
        {
            int nt = omp_get_num_threads(), id = omp_get_thread_num();
            long long a = (long long)len * id / nt;
            long long b = (long long)len * (id + 1) / nt;
            double lm = lat_min(fd + a, (int)(b - a));
            #pragma omp critical
            { if (lm < g) g = lm; }
        }
        return g;
    }
#endif
    return lat_min(fd, len);
}

static int lat_find(const double *fd, int len, double best)
{
    int p = 0;
#if defined(__AVX2__)
    {
        __m256d bb = _mm256_set1_pd(best);
        for (; p + 4 <= len; p += 4) {
            __m256d v = _mm256_loadu_pd(fd + p);
            int mk = _mm256_movemask_pd(_mm256_cmp_pd(v, bb, _CMP_EQ_OQ));
            if (mk) return p + (int)__builtin_ctz((unsigned)mk);
        }
    }
#endif
    for (; p < len; p++) if (fd[p] == best) return p;
    return 0;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    /* ---------------- step 1 ----------------
       Lay out the rigid lattice of old memory: one fixed slot for every
       cross-marked board, no more and no fewer, each cut blank (= INFINITY).
       Beside it the second, smaller lattice: same slots, one plain bead each,
       unsettled (= 0).
       Layout only (not a step of its own): the land is also laid out as CSR
       out-adjacency so step 5 can take "one nightingale for each road leaving"
       a board, and a compacted view `fidx/fdist/pos` holds exactly the slot set
       step 3 looks over (unsettled and not blank). */
    double *dist = dist_out;                                  /* the lattice   */
    unsigned char *settled = (unsigned char *)malloc((size_t)n);
    int    *pos   = (int *)malloc((size_t)n * sizeof(int));
    int    *fidx  = (int *)malloc((size_t)n * sizeof(int));
    double *fdist = (double *)malloc((size_t)n * sizeof(double));
    int    *head  = (int *)malloc((size_t)(n + 2) * sizeof(int));
    int    *adj_v = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *adj_w = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    if (!settled || !pos || !fidx || !fdist || !head || !adj_v || !adj_w) {
        for (int i = 0; i < n; i++) dist[i] = INFINITY;
        free(settled); free(pos); free(fidx); free(fdist);
        free(head); free(adj_v); free(adj_w);
        return;
    }

    #pragma omp parallel for schedule(static) if(n > 65536)
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; settled[i] = 0; pos[i] = -1; }

    memset(head, 0, (size_t)(n + 2) * sizeof(int));
    for (int i = 0; i < m; i++) head[src[i] + 2]++;
    for (int u = 2; u <= n + 1; u++) head[u] += head[u - 1];
    for (int i = 0; i < m; i++) {                 /* head[u+1] used as cursor  */
        int q = head[src[i] + 1]++;
        adj_v[q] = dst[i];
        adj_w[q] = weight[i];
    }                    /* now head[u]..head[u+1) are the roads leaving u     */
    int fsize = 0;

    /* ---------------- step 2 ----------------
       Into the slot for the traveler's own board carve the silence: the note of
       no strides at all, the faintest sound there is. Every other slot blank.
       (It is now unsettled-and-not-blank, so it enters the step-3 view.) */
    if (source >= 0 && source < n) {
        dist[source] = 0.0;
        fidx[0] = source; fdist[0] = 0.0; pos[source] = 0; fsize = 1;
    }

    for (;;) {
        /* ---------------- step 3 ----------------
           Look over all slots whose bead is still unsettled and whose carving
           is not blank; take the one whose carving is quietest. If no such slot
           exists, go to step 11. (Ties: the first such slot found.) */
        if (fsize == 0) break;                           /* -> step 11 */
        double best = lat_min_par(fdist, fsize);
        int p = lat_find(fdist, fsize, best);
        int u = fidx[p];
        double du = best;

        /* ---------------- step 4 ----------------
           Turn that slot's bead to settled. A settled bead is never turned
           back and its board is never returned to: so it leaves the view of
           step 3 forever (last candidate swapped into its place). */
        settled[u] = 1;
        {
            int last = fsize - 1;
            int mv = fidx[last];
            fidx[p] = mv; fdist[p] = fdist[last]; pos[mv] = p;
            pos[u] = -1; fsize = last;
        }

        /* ---------------- step 5 ----------------
           Stand at that settled board. Take one nightingale for each road
           leaving it and loose them all at once; each sings itself to death at
           the first board its road reaches. Send no bird down a road already
           marked dead (step 9: a road into an already-settled board), and never
           a second bird down a road already flown (automatic: u is settled once,
           so each of its roads is flown exactly once). */
        for (int e = head[u], e1 = head[u + 1]; e < e1; e++) {
            int v = adj_v[e];
            if (settled[v]) continue;                    /* dead road */

            /* ---------------- step 6 ----------------
               At the board where the bird falls, open a cow's head: the woman
               draws the settled board's own carved note with that dead bird's
               hoarseness added onto it. */
            double nd = du + adj_w[e];

            /* ---------------- step 7 ----------------
               Compare her milk against what is carved in that fallen board's
               slot. Blank, or thinner than the carving: scrape clean and cut
               her milk in its place. Same or fatter: spill it in the desert. */
            if (nd < dist[v]) {
                dist[v] = nd;
                int q = pos[v];
                if (q < 0) { q = fsize++; fidx[q] = v; pos[v] = q; }
                fdist[q] = nd;
            }

            /* ---------------- step 8 ----------------
               Where two dying notes reach the same board, the comparing above
               has already kept the fainter and let the louder rot. Keep nothing
               else: no predecessor, no second-best, no record of the discarded.
               What's thrown away stays thrown away. (Zero instructions.) */

            /* ---------------- step 9 ----------------
               Mark dead any road whose bird brings nothing thinner and whose
               far board is already settled, and send no bird down it again.
               Since a settled carving is final and weights are >= 0, that is
               exactly "every road into a settled board" -- carried in the
               `settled` beads and consulted by step 5's guard above. No bird
               ever revisits a road anyway. */
        }
        /* ---------------- step 10 ----------------
           Return to step 3. */
    }

    /* ---------------- step 11 ----------------
       Finished: no unsettled slot holds a carving -- no nightingale is left to
       sing and no cow yields a thinner milk. Every reachable board bears its
       one permanent carving, the shortest way there. Every slot still blank is
       a place past reach, blank as the desert (INFINITY). */

    /* ---------------- step 12 ----------------
       Read the traveler the whole lattice in order, slot by slot: each board,
       its carved note, and the desert for the blank ones. The lattice was
       carved in `dist_out` itself, already in slot order, so it is read out as
       it stands. */
    free(settled); free(pos); free(fidx); free(fdist);
    free(head); free(adj_v); free(adj_w);
}
