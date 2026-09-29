# APPROACH

**Mapping of Pip's baskets onto the real problem.**

| Story element | Implementation |
|---|---|
| "a row of baskets, one per minute-mark" | A circular array `bhead[B]` of bucket heads, bucket index `⌊d/Δ⌋ mod B`. `B` is a power of two so the modulo is a mask. Circular is safe because every live tentative key lies in `[d_cur, d_cur + w_max]`, so `B ≥ ⌊w_max/Δ⌋+2` buckets never alias. |
| "drop a tag with the tree's name into the basket matching the guess" | Insert node `v` at the head of its bucket's intrusive doubly-linked list (`bnext`,`bprev`,`inb`). One tag per node — no lazy duplicates, no stored keys (the key is `dist_out[v]`). |
| "pull the tag out and drop it into an earlier basket" | Decrease-key = O(1) unlink + relink. This is literally the story's move, and it is why I used doubly-linked lists instead of a lazy multi-push pool. |
| "walk along the baskets from where she left off; first non-empty one is her tree" | A monotone cursor `curv` (a *virtual*, unwrapped index) that never moves backwards. The walk is accelerated by a **3-level hierarchical bitmap** (`l0`: 1 bit/bucket, `l1`: 1 bit per `l0` word, `l2`: 1 bit per `l1` word) so "skip empty baskets" costs ~3 `ctz` instead of a linear crawl. This matters: naive basket-walking is the one way this mechanism can blow up (a fine Δ plus a long path ⇒ billions of empty baskets), and the bitmap removes that failure mode completely rather than replacing the mechanism. |
| "she never compares every tree at once" | No O(n) scan per step (the assumption the standard method silently makes), and no O(log n) sift per relaxation either. |

**Why popping an arbitrary tag from the first non-empty basket is exact.** If Δ ≤ every positive weight, then relaxing out of bucket `b` produces a key `≥ bΔ + Δ`, i.e. bucket `≥ b+1`. So no node in bucket `b` can ever be improved by another node in bucket `b` — internal order is irrelevant and every pop is a true global minimum. That is the fast path (`orderfree`), chosen whenever `w_max/w_min + 2` fits in the bucket budget and no zero-weight edges exist. It makes equal/integer/narrow-range weights run at BFS speed with a 3-bucket array living in L1.

**When the weight range is too wide** (continuous random weights: `w_min ~ 1e-7`), Δ = `w_max/(B-2)` with `B ≈ next_pow2(2n)`, so the active window is split into ~2n buckets and mean occupancy is < 1. Exactness is then restored by extracting the true minimum *of the current basket only* — a tiny lazy binary heap seeded from that one basket. All other baskets are untouched, so this is still the basket mechanism, not heap-Dijkstra; it degrades gracefully to heap-Dijkstra only in the adversarial case where everything lands in one basket (e.g. weights `{0, 1}`), which is exactly the safety property I wanted.

Zero-weight edges force the ordered path (they relax *within* the current basket, which would otherwise break arbitrary-order popping); an all-zero graph is handled as plain reachability. Keys are clamped to `≥ curv` so no floating-point boundary fuzz can strand a tag in a "past" basket.

Not exploited: early termination for a single target (the contract demands all `n` distances).

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct { double d; int v; } HItem;

static void hpush(HItem **hp, int *hcap, int *hs, double d, int v)
{
    if (*hs == *hcap) {
        int nc = (*hcap) * 2;
        HItem *t = (HItem *)realloc(*hp, (size_t)nc * sizeof(HItem));
        *hp = t; *hcap = nc;
    }
    {
        HItem *h = *hp;
        int i = (*hs)++;
        while (i > 0) {
            int p = (i - 1) >> 1;
            if (h[p].d <= d) break;
            h[i] = h[p]; i = p;
        }
        h[i].d = d; h[i].v = v;
    }
}

static HItem hpop(HItem *h, int *hs)
{
    HItem top = h[0];
    int sz = --(*hs);
    if (sz > 0) {
        HItem last = h[sz];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1, r;
            if (l >= sz) break;
            r = l + 1;
            if (r < sz && h[r].d < h[l].d) l = r;
            if (h[l].d >= last.d) break;
            h[i] = h[l]; i = l;
        }
        h[i] = last;
    }
    return top;
}

/* smallest index q in [p,B) whose bit is set, else -1  (B = nw0*64) */
static int nextset(const uint64_t *l0, const uint64_t *l1, const uint64_t *l2,
                   int nw0, int nw1, int nw2, int p)
{
    int w0 = p >> 6;
    uint64_t m = l0[w0] & (~(uint64_t)0 << (p & 63));
    if (m) return (w0 << 6) + __builtin_ctzll(m);
    {
        int i = w0 + 1;
        if (i < nw0) {
            int w1 = i >> 6;
            uint64_t m1 = l1[w1] & (~(uint64_t)0 << (i & 63));
            if (m1) {
                int wi = (w1 << 6) + __builtin_ctzll(m1);
                return (wi << 6) + __builtin_ctzll(l0[wi]);
            }
            {
                int j = w1 + 1;
                if (j < nw1) {
                    int w2 = j >> 6;
                    uint64_t m2 = l2[w2] & (~(uint64_t)0 << (j & 63));
                    while (!m2) { if (++w2 >= nw2) return -1; m2 = l2[w2]; }
                    {
                        int wj = (w2 << 6) + __builtin_ctzll(m2);
                        int wi = (wj << 6) + __builtin_ctzll(l1[wj]);
                        return (wi << 6) + __builtin_ctzll(l0[wi]);
                    }
                }
            }
        }
    }
    return -1;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    int i;
    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---------- CSR (counting sort by src), weight stats folded in ---------- */
    {
    int *off = (int *)calloc((size_t)n + 1, sizeof(int));
    int *pos, *eto;
    double *ew;
    double wmax = 0.0, wmin = INFINITY;
    int haszero = 0;

    for (i = 0; i < m; i++) off[src[i] + 1]++;
    for (i = 0; i < n; i++) off[i + 1] += off[i];
    pos = (int *)malloc((size_t)n * sizeof(int));
    memcpy(pos, off, (size_t)n * sizeof(int));
    eto = (int *)malloc((size_t)m * sizeof(int));
    ew  = (double *)malloc((size_t)m * sizeof(double));
    for (i = 0; i < m; i++) {
        int u = src[i];
        int p = pos[u]++;
        double w = weight[i];
        eto[p] = dst[i];
        ew[p]  = w;
        if (w > wmax) wmax = w;
        if (w > 0.0) { if (w < wmin) wmin = w; } else haszero = 1;
    }
    free(pos);

    /* ---------- degenerate: every bridge is instantaneous ---------- */
    if (!(wmax > 0.0)) {
        int *stk = (int *)malloc((size_t)n * sizeof(int));
        unsigned char *vis = (unsigned char *)calloc((size_t)n, 1);
        int sp = 0;
        stk[sp++] = source; vis[source] = 1;
        while (sp > 0) {
            int u = stk[--sp], e;
            for (e = off[u]; e < off[u + 1]; e++) {
                int y = eto[e];
                if (!vis[y]) { vis[y] = 1; dist_out[y] = 0.0; stk[sp++] = y; }
            }
        }
        free(stk); free(vis); free(off); free(eto); free(ew);
        return;
    }

    /* ---------- basket geometry ---------- */
    {
    long long tgt = 2LL * (long long)n + 64;
    int B0 = 64, B = 64, nw0, nw1, nw2, orderfree;
    double dall, delta, inv;
    long long span, curv = 0, qsize = 0, maskl;
    int *bhead, *bnext, *bprev, *inb, *cb;
    unsigned char *done;
    uint64_t *l0, *l1, *l2;
    HItem *heap; int hcap = 1024, hs = 0;

    if (tgt < 1024) tgt = 1024;
    if (tgt > (1 << 21)) tgt = (1 << 21);
    while ((long long)B0 < tgt) B0 <<= 1;

    dall = wmax / (double)(B0 - 2);
    if (wmin >= dall) { delta = wmin; orderfree = haszero ? 0 : 1; }
    else              { delta = dall; orderfree = 0; }
    inv = 1.0 / delta;

    span = (long long)(wmax * inv) + 2;
    while ((long long)B < span) B <<= 1;
    if (B > B0) B = B0;
    maskl = (long long)(B - 1);

    nw0 = B >> 6;
    nw1 = (nw0 + 63) >> 6;
    nw2 = (nw1 + 63) >> 6;

    bhead = (int *)malloc((size_t)B * sizeof(int));
    memset(bhead, 0xFF, (size_t)B * sizeof(int));
    bnext = (int *)malloc((size_t)n * sizeof(int));
    bprev = (int *)malloc((size_t)n * sizeof(int));
    inb   = (int *)malloc((size_t)n * sizeof(int));
    for (i = 0; i < n; i++) inb[i] = -1;
    done  = (unsigned char *)calloc((size_t)n, 1);
    cb    = (int *)malloc((size_t)n * sizeof(int));
    l0 = (uint64_t *)calloc((size_t)nw0, 8);
    l1 = (uint64_t *)calloc((size_t)nw1, 8);
    l2 = (uint64_t *)calloc((size_t)nw2, 8);
    heap = (HItem *)malloc((size_t)hcap * sizeof(HItem));

#define BSET(ix) do { int i_ = (ix); \
        l0[i_ >> 6]  |= (uint64_t)1 << (i_ & 63); \
        l1[i_ >> 12] |= (uint64_t)1 << ((i_ >> 6) & 63); \
        l2[i_ >> 18] |= (uint64_t)1 << ((i_ >> 12) & 63); } while (0)

#define BCLR(ix) do { int i_ = (ix), w_ = i_ >> 6; \
        l0[w_] &= ~((uint64_t)1 << (i_ & 63)); \
        if (!l0[w_]) { int x_ = w_ >> 6; \
            l1[x_] &= ~((uint64_t)1 << (w_ & 63)); \
            if (!l1[x_]) l2[x_ >> 6] &= ~((uint64_t)1 << (x_ & 63)); } } while (0)

    /* move node y's tag to the basket matching key nd (already stored in dist) */
#define PLACE(y, nd) do { \
        long long vi_ = (long long)((nd) * inv); \
        int nb_, ob_; \
        if (vi_ < curv) vi_ = curv; \
        nb_ = (int)(vi_ & maskl); \
        ob_ = inb[y]; \
        if (!(ob_ == nb_ && !(!orderfree && vi_ == curv))) { \
            if (ob_ >= 0) { \
                int pp_ = bprev[y], qq_ = bnext[y]; \
                if (pp_ >= 0) bnext[pp_] = qq_; \
                else { bhead[ob_] = qq_; if (qq_ < 0) BCLR(ob_); } \
                if (qq_ >= 0) bprev[qq_] = pp_; \
                inb[y] = -1; qsize--; \
            } \
            if (!orderfree && vi_ == curv) { \
                hpush(&heap, &hcap, &hs, (nd), (y)); \
            } else { \
                int hh_ = bhead[nb_]; \
                bnext[y] = hh_; bprev[y] = -1; \
                if (hh_ >= 0) bprev[hh_] = (y); else BSET(nb_); \
                bhead[nb_] = (y); inb[y] = nb_; qsize++; \
            } \
        } } while (0)

    bhead[0] = source; bnext[source] = -1; bprev[source] = -1; inb[source] = 0;
    qsize = 1; BSET(0);

    while (qsize > 0) {
        int b = (int)(curv & maskl);
        int k = 0, v;
        if (bhead[b] < 0) {                       /* walk to the next full basket */
            int q = nextset(l0, l1, l2, nw0, nw1, nw2, b);
            if (q < 0) q = nextset(l0, l1, l2, nw0, nw1, nw2, 0);
            if (q < 0) break;
            curv += (long long)((unsigned)(q - b) & (unsigned)maskl);
            b = q;
        }
        for (v = bhead[b]; v != -1; v = bnext[v]) { cb[k++] = v; inb[v] = -1; }
        bhead[b] = -1; BCLR(b); qsize -= k;

        if (orderfree) {
            int t;
            for (t = 0; t < k; t++) {
                int u = cb[t], e, ee;
                double du;
                if (done[u] || inb[u] >= 0) continue;
                done[u] = 1;
                du = dist_out[u];
                e = off[u]; ee = off[u + 1];
                for (; e < ee; e++) {
                    int y = eto[e];
                    double nd;
                    if (e + 2 < ee) __builtin_prefetch(&dist_out[eto[e + 2]], 1, 1);
                    nd = du + ew[e];
                    if (nd < dist_out[y] && !done[y]) { dist_out[y] = nd; PLACE(y, nd); }
                }
            }
        } else {
            int t;
            hs = 0;
            for (t = 0; t < k; t++) { int v2 = cb[t]; hpush(&heap, &hcap, &hs, dist_out[v2], v2); }
            while (hs > 0) {
                HItem top = hpop(heap, &hs);
                int u = top.v, e, ee;
                double du = top.d;
                if (done[u] || du != dist_out[u]) continue;
                done[u] = 1;
                e = off[u]; ee = off[u + 1];
                for (; e < ee; e++) {
                    int y = eto[e];
                    double nd;
                    if (e + 2 < ee) __builtin_prefetch(&dist_out[eto[e + 2]], 1, 1);
                    nd = du + ew[e];
                    if (nd < dist_out[y] && !done[y]) { dist_out[y] = nd; PLACE(y, nd); }
                }
            }
        }
    }

#undef PLACE
#undef BSET
#undef BCLR
    free(bhead); free(bnext); free(bprev); free(inb); free(done); free(cb);
    free(l0); free(l1); free(l2); free(heap);
    }
    free(off); free(eto); free(ew);
    }
}
```

# PREDICTION

PREDICTION: 2.5

(Interpretation: ~2.5× faster than the given binary-heap reference on the benchmark's graphs. Reasoning behind the number, written before any measurement: the shared, irreducible costs — CSR counting sort, the `m` random reads of `dist_out`, the `n`-length init — are perhaps 35–45% of the reference's runtime and are unchanged here; the priority-queue half, which in the reference is `O(m)` sifts of depth ~15–20 with branchy pointer-chasing, collapses to `O(1)` unlink/relink plus a bitmap `ctz`. Amdahl on those shares gives ≈2–3×. Narrow-range or integer weights should land nearer 3–4× because the whole basket array fits in L1 and no ordering work happens at all; continuous wide-range weights nearer 2×.)

# MEASUREMENT

**Not performed — and I am not going to dress that up.** The task listed `dijkstra_bench` and `dijkstra_contract`, but this session was explicitly invoked with no tools available, so I compiled nothing, ran nothing, and validated nothing against the reference. The prediction above is an unverified estimate, and the "at most four improvement rounds" budget was spent at zero rounds. Everything in ARTIFACT is desk-checked only.

What I did verify by hand, in lieu of measurement:

- **Correctness argument, fast path:** Δ = min positive weight ⇒ any relaxation out of basket `b` lands in basket ≥ `b+1` ⇒ intra-basket order is provably irrelevant ⇒ every pop is a true global minimum. Zero-weight edges break that premise, so they force `orderfree = 0`.
- **Correctness argument, wide-range path:** the exact minimum is taken from the current basket only; all other baskets hold strictly larger keys, so that minimum is the global minimum. Same-basket insertions (zero edges, or Δ > w_min) go into that same small heap, never into an already-scanned list.
- **No-stranding:** keys are clamped to `≥ curv`, so no floating-point rounding at a basket boundary can push a tag into a basket the cursor has already passed. Worst residual effect of boundary rounding is a settle-order swap between two keys within 1 ulp of each other — bounded by tolerance, not a structural error.
- **Circular window:** live keys span ≤ `w_max`, and `B ≥ ⌊w_max/Δ⌋+2`, so wrap-around never aliases a live basket.
- **Termination:** each outer iteration settles ≥ 1 node (drained nodes are never already `done`), and `qsize` accounting is symmetric across drain / unlink / relink.

**Things a real measurement would most likely expose, in rough order of my worry:**
1. Wall-clock on `B ≈ 2n` — `memset(bhead, 0xFF, 4B)` is ~8 MB per call at n = 1e6 and could eat a chunk of the win if the harness re-invokes the kernel per iteration. If so, the fix is shrinking `tgt` to `n/2` and re-checking occupancy, not changing mechanism.
2. Whether the `__builtin_prefetch` helps or hurts (it adds a predictable branch per edge).
3. A weight distribution like `{0.0, 1.0}` or "all 1.0 plus one 1e-9" that clusters everything into one basket and degrades to heap-Dijkstra. Designed-for, but unmeasured.

# VERDICT

The mechanism translates cleanly and, I believe, faithfully: a monotone array of baskets, one tag per tree, decrease-key by moving the tag to an earlier basket, and a cursor that only ever walks forward. The two places where the story's arithmetic (whole minutes) collides with the contract's reals — basket width versus the smallest weight, and zero-weight bridges — are handled inside the mechanism (choose Δ ≤ w_min when the range allows; otherwise take the exact minimum of the single current basket) rather than by swapping in a different algorithm, and the one way basket-walking can genuinely blow up (oceans of empty baskets) is fixed with a hierarchical bitmap rather than abandoned.

But the honest bottom line is that this is an argued artifact, not a measured one. **PREDICTION: 2.5 stands untested.** Until `dijkstra_contract` confirms it matches the reference within tolerance and `dijkstra_bench` puts a number on it, treat the speedup claim as a hypothesis, and treat the correctness claim as a proof sketch that has never met a compiler.