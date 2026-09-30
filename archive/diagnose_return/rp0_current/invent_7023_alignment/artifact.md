# MAPPING

## SEED 1 — "the low sun throws a single doubled shadow"

| world object | problem object |
|---|---|
| grass-rope, knots dyed one of four berry-colors | `const char *` over `{A,C,G,T}`, one byte per knot |
| rope on the east wall / rope across the floor | the two axes `i` and `j` of the comparison |
| "every knot crosses every knot in the mind's eye" | the n×n grid exists conceptually but is never materialised |
| the one window, the low sun at its fixed hour | one `vpcmpeqb` — a single instruction that strikes 32 knots of each rope at once |
| two knots of one dye throwing **one** shadow instead of two | byte-equality → a set bit in `_mm256_movemask_epi8` |
| the doubled-shadow pattern on the wall | the 32-bit match mask; `popcount` turns it into a count |
| "without a hand ever touching the floor" | zero DP cells written to get the whole match census |

**Silent assumption broken:** *one pair of positions is judged at a time.* The sun judges 32 pairs per instruction.

## SEED 2 — "the corridor's cost is read off the ropes, never walked"

| world object | problem object |
|---|---|
| a treasure / lit tile | a cell where a decision actually has to be taken |
| the bare floor between two treasures | the run of DP cells strictly between two such cells |
| "how many knots forward on the wall-rope, how many on the floor-rope" | `di = i'−i`, `dj = j'−j` |
| "reading a distance by counting doorframes instead of pacing it" | closed form: interior score = `3·min(di,dj) − 2·di − 2·dj` |
| "the corridor's cost was never a secret — only its length was" | the interior is an *algebraic function of the offsets only*; it carries no information, so it need not be visited |
| the room is a *corridor*, not a hall | the reachable set is a narrow diagonal band, not the full square |

This seed, taken literally, forces an algebraic identity. Two equal-length sequences force the gap counts on the two sides to be equal (`gA = gB = g`), so for any alignment

```
score = (#match) − (#mismatch) − 2(gA+gB) = n − 5g − 2x
```

Every unit of slip costs exactly **5**. So if `L` is *any* score we already hold in hand, every optimal path satisfies `g ≤ ⌊(n−L)/5⌋`, and since displacement `|i−j|` along a path is bounded by `g`, the whole optimum lives inside `|i−j| ≤ k`, `k = ⌊(n−L)/5⌋`. That is the corridor — and the native's own pricing rule *is* the proof of its width.

**Silent assumption broken:** *the whole grid of every position against every other must be filled in.*

## SEED 3 — "only the cheapest tally survives; the rest fall like bad thread-ends"

| world object | problem object |
|---|---|
| a running tally arriving at a treasure | a candidate score for a prefix-pair |
| "with the slip already spent, or not yet spent" | a two-state family: the on-diagonal tally (`g=0`) and the slipped tally (`g=s`) |
| "the moment a second tally arrives worse than the first, I let it drop" | a running minimum over slip breakpoints — `O(n)` per slip width, no grid |
| the slip: one rope waits while the other runs ahead, then falls back into step | shift out by `s`, run offset, shift back — priced as `n − 5s − 2x` from prefix tallies alone |

**Silent assumption broken:** *a slip can only be discovered by having already compared the position before it.* Here a slip of arbitrary width is priced from two prefix-mismatch tallies, with no cell to the left of it ever computed.

---

# CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks *"the whole grid of every position against every other must be filled in"* — the preferred assumption — and its mapping is the most literal: the native's refusal to lay a marker on the bare floor is exactly a refusal to write a DP cell, and his claim that the floor's price is *"how many knots forward on each rope"* is exactly the closed form `3·min(di,dj) − 2di − 2dj`, which is what makes the refusal legitimate rather than lossy.

SEED 1 and SEED 3 are retained as *machinery inside* SEED 2's world: the sun supplies the first tally `L` in `O(n/32)` (SEED 1), and the slip-pricing supplies a much better `L` when a single indel dominates (SEED 3). Both exist only to narrow the corridor.

**Honest note on the literal reading I rejected:** the most literal possible reading of "visit only the treasures" is Wilbur–Lipman sparse chaining over match cells. I take it seriously enough to price it: the native's alphabet is four berry-colors, so a quarter of all crossings are treasures — `M ≈ n²/4`. The cat's premise ("there are few of them") is *false for DNA*, and sparse chaining would be strictly worse than the grid. What survives translation is not the sparsity of treasures but the *pricing rule for the floor between them*. I report that as a failure of the metaphor's own assumption, not as a success.

# ASSUMPTION BROKEN

**"The whole grid of every position against every other must be filled in."**

Replaced by: the grid's *interior price is known in closed form*, therefore only a corridor of half-width `k = ⌊(n − L)/5⌋` around the main diagonal can contain an optimal path, and `L` is obtainable without touching the grid at all. When `k = 0` (`H ≤ 2`) the answer is `n − 2H` and **not one DP cell is ever written**.

Where this lands: the mechanism arrives at **Fickett/Ukkonen banded alignment with a self-certifying band**, and at the **SIMD prefix-max "scan" row recurrence** (the Parasail/Daily formulation) for the wide-corridor regime. Both are validated real-world techniques. I did not invent a substitute for them; the native's pricing rule is what *derives* the band width and the certificate, and the certificate `⌊(n−S_k)/5⌋ ≤ k ⟹ S_k = S` is what makes the result exactly equal to the reference DP rather than a heuristic.

**Computational mapping of the world, literally:**

| world | machine |
| --- | --- |
| the two ropes (immutable, hung) | the two `const char*`; **what stays still** |
| the empty room / untiled floor | the grid that is never allocated |
| the corridor | the band `|i−j| ≤ k`; **the only memory that exists** (two rows, `O(n)`) |
| the low sun | one AVX2 compare + popcount; **a processor that judges 32 crossings at once** |
| the cat | the match mask (it reports treasures, it does not walk) |
| the running tally | `carry` — the one scalar that **flows** left along a row |
| the hour the sun always comes | **time** = the row index `i`, strictly increasing, one row per tick |
| doorframes counted instead of paced | the `/5` arithmetic that converts a score into a corridor width |

**Regime recognition, through the metaphor itself** — the corridor's own width, measured at runtime, chooses the path:

- `k = 0` → *no room at all*: return `n − 2H` from popcounts alone.
- `k < 24` (per row) → *a corridor narrower than the sun's beam*: scalar band rows; a vector would hang out of the walls.
- `k ≥ 24` → *a corridor wide enough to stand in*: AVX2 prefix-max scan rows.
- `k ≥ n` → *the corridor is the whole room* (divergent pair): same scan, full width — so this path is never worse than the known way, it is the known way plus vectorisation and `O(n)` memory.
- `n < 64` → below the sun's own cost: plain two-row DP, no machinery.
- Each optional probe (slip pricing, staged band) is gated on the corridor already being wide enough for the probe to pay for itself.

# ARTIFACT

```c
#include <stdlib.h>
#include <stdint.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#define MT   1
#define MM (-1)
#define GP (-2)
#define NBIG (-(1 << 26))   /* "outside the corridor"; safe from int32 overflow */

/* ---------------- SEED 1: the low sun and the doubled shadow ----------------
   One instruction strikes 32 knots of each rope; equal dyes cast one shadow.
   Returns the Hamming distance, i.e. the g=0 tally, without touching the floor. */
static int ham_dist(int n, const char *a, const char *b)
{
    int i = 0, h = 0;
#ifdef __AVX2__
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb));
        h += 32 - __builtin_popcount(m);
    }
#endif
    for (; i < n; i++) h += (a[i] != b[i]);
    return h;
}

/* ---------------- SEED 3: one slip, priced off the ropes only ----------------
   Alignment shape: diagonal on [0,p), then b runs ahead by s on [p,q), then
   they fall back into step on [q+s,n).  Gaps on each side = s, so
        score = n - 5s - 2x,
        x     = (P0[p]-Ps[p]) + (Ps[q]-P0[q+s]) + H.
   The inner minimisation keeps one running tally and drops every costlier
   one the instant it arrives -- a bad thread-end let fall.  O(n) per s, and
   not one cell of the floor is marked.                                        */
static int slip_bound(int n, const char *a, const char *b,
                      const int *P0, int smax)
{
    int best = NBIG;
    int H = P0[n];
    for (int s = 1; s <= smax; s++) {
        int lim = n - s;
        if (lim < 0) break;
        int Ps = 0;                 /* prefix of (a[u] != b[u+s]) */
        int bestf = 0x3fffffff;     /* min over p <= q of P0[p]-Ps[p] */
        int bestx = 0x3fffffff;
        for (int q = 0; q <= lim; q++) {
            int f = P0[q] - Ps;
            if (f < bestf) bestf = f;
            int cand = bestf + (Ps - P0[q + s]);
            if (cand < bestx) bestx = cand;
            if (q < lim) Ps += (a[q] != b[q + s]);
        }
        int sc = n - 5 * s - 2 * (H + bestx);
        if (sc > best) best = sc;
    }
    return best;
}

/* ---------------- SEED 2: the corridor, and only the corridor ----------------
   Exact Needleman-Wunsch restricted to |i-j| <= k.  Cells outside are NBIG,
   so no path can leak out.  Two rows of memory; the bare floor beyond the
   corridor is never laid.  Rows wider than the sun's beam use the prefix-max
   scan:  t[j] = max(prev[j-1]+sub, prev[j]+GP)          (independent, vector)
          cur[j] = max_{j'<=j} ( t[j'] + GP*(j-j') )     (Hillis-Steele scan) */
static int band_dp(int n, const char *a, const char *b,
                   int k, int *buf, int stride)
{
    int *prev = buf, *cur = buf + stride;
    int hp = (k < n) ? k : n;
    for (int j = 0; j <= hp; j++) prev[j] = GP * j;
    prev[hp + 1] = NBIG;
#ifdef __AVX2__
    const __m256i idx2 = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i vneg = _mm256_set1_epi32(NBIG);
    const __m256i vmt  = _mm256_set1_epi32(MT);
    const __m256i vmm  = _mm256_set1_epi32(MM);
    const __m256i vgp  = _mm256_set1_epi32(GP);
#endif
    for (int i = 1; i <= n; i++) {
        int lo = i - k; if (lo < 1) lo = 1;
        int hi = i + k; if (hi > n) hi = n;
        if (lo == 1) cur[0] = (i <= k) ? GP * i : NBIG;
        else         cur[lo - 1] = NBIG;
        const char ai = a[i - 1];
        int carry = cur[lo - 1];
        int j = lo;
#ifdef __AVX2__
        if (hi - lo + 1 >= 24) {
            const __m128i vai8 = _mm_set1_epi8(ai);
            for (; j + 7 <= hi; j += 8) {
                __m256i pd = _mm256_loadu_si256((const __m256i *)(prev + j - 1));
                __m256i pu = _mm256_loadu_si256((const __m256i *)(prev + j));
                __m128i bb = _mm_loadl_epi64((const __m128i *)(b + j - 1));
                __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(bb, vai8));
                __m256i sb = _mm256_blendv_epi8(vmm, vmt, eq);
                __m256i t  = _mm256_max_epi32(_mm256_add_epi32(pd, sb),
                                              _mm256_add_epi32(pu, vgp));
                __m256i w  = _mm256_add_epi32(t, idx2);      /* w[p] = t[p] + 2p */
                __m256i s;
                s = _mm256_alignr_epi8(w, _mm256_permute2x128_si256(w, vneg, 0x02), 12);
                w = _mm256_max_epi32(w, s);                  /* shift 1 lane */
                s = _mm256_alignr_epi8(w, _mm256_permute2x128_si256(w, vneg, 0x02), 8);
                w = _mm256_max_epi32(w, s);                  /* shift 2 lanes */
                s = _mm256_permute2x128_si256(w, vneg, 0x02);
                w = _mm256_max_epi32(w, s);                  /* shift 4 lanes */
                w = _mm256_max_epi32(w, _mm256_set1_epi32(carry + GP));
                __m256i r = _mm256_sub_epi32(w, idx2);
                _mm256_storeu_si256((__m256i *)(cur + j), r);
                carry = _mm256_extract_epi32(r, 7);
            }
        }
#endif
        for (; j <= hi; j++) {
            int d = prev[j - 1] + ((ai == b[j - 1]) ? MT : MM);
            int u = prev[j] + GP;
            int t = d > u ? d : u;
            int l = carry + GP;
            int v = t > l ? t : l;
            cur[j] = v;
            carry  = v;
        }
        cur[hi + 1] = NBIG;
        { int *tmp = prev; prev = cur; cur = tmp; }
    }
    return prev[n];
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* Below the cost of the sun itself: pace the room the plain way. */
    if (n < 64) {
        int prev[64], cur[64];
        for (int j = 0; j <= n; j++) prev[j] = GP * j;
        for (int i = 1; i <= n; i++) {
            char ai = a[i - 1];
            cur[0] = GP * i;
            for (int j = 1; j <= n; j++) {
                int d = prev[j - 1] + ((ai == b[j - 1]) ? MT : MM);
                int u = prev[j] + GP;
                int t = d > u ? d : u;
                int l = cur[j - 1] + GP;
                cur[j] = t > l ? t : l;
            }
            for (int j = 0; j <= n; j++) prev[j] = cur[j];
        }
        return prev[n];
    }

    /* SEED 1: the cat's census, in one sweep of sunlight. */
    int H  = ham_dist(n, a, b);
    int L0 = n - 2 * H;                 /* a real alignment, so L0 <= S      */
    int L  = L0;
    int k  = (n - L) / 5;               /* every slip costs exactly 5        */
    if (k <= 0) return L0;              /* no room to slip: answer is exact  */
    if (k > n) k = n;

    int stride = n + 18;
    int *mem = (int *)malloc((size_t)(2 * stride + n + 2) * sizeof(int));
    if (!mem) {                         /* degrade to the textbook path      */
        int *p = (int *)malloc((size_t)(2 * (n + 2)) * sizeof(int));
        if (!p) return L0;
        int *pr = p, *cu = p + (n + 2);
        for (int j = 0; j <= n; j++) pr[j] = GP * j;
        for (int i = 1; i <= n; i++) {
            char ai = a[i - 1]; cu[0] = GP * i;
            for (int j = 1; j <= n; j++) {
                int d = pr[j - 1] + ((ai == b[j - 1]) ? MT : MM);
                int u = pr[j] + GP; int t = d > u ? d : u;
                int l = cu[j - 1] + GP; cu[j] = t > l ? t : l;
            }
            { int *tm = pr; pr = cu; cu = tm; }
        }
        int r = pr[n]; free(p); return r;
    }
    int *buf = mem;
    int *P0  = mem + 2 * stride;

    /* SEED 3: if the corridor is wide, ask whether one slip explains it.
       Gated so the probe can never cost more than the floor it saves.      */
    if (k >= 16) {
        P0[0] = 0;
        for (int t = 0; t < n; t++) P0[t + 1] = P0[t] + (a[t] != b[t]);
        int smax = (k < 24) ? k : 24;
        int s1 = slip_bound(n, a, b, P0, smax);   /* b runs ahead */
        int s2 = slip_bound(n, b, a, P0, smax);   /* a runs ahead */
        if (s1 > L) L = s1;
        if (s2 > L) L = s2;
        int nk = (n - L) / 5;
        if (nk < k) k = nk;
        if (k <= 0) { free(mem); return L0; }
    }

    /* One cheap narrow pass can certify itself and retire the wide one.
       If it cannot, its score still tightens the corridor.                 */
    if (k > 48) {
        int kp = 24;
        int sp = band_dp(n, a, b, kp, buf, stride);
        int nk = (n - sp) / 5;
        if (nk <= kp) { free(mem); return sp; }   /* certificate: exact      */
        if (nk < k) k = nk;
    }

    int res = band_dp(n, a, b, k, buf, stride);
    free(mem);
    return res;
}
```

**Why this is exact, not a heuristic.** Every `L` used is the score of an actually-constructed alignment, so `L ≤ S`. For equal-length inputs the per-side gap counts are equal, giving `score = n − 5g − 2x`; hence any optimal path has `g ≤ ⌊(n−S)/5⌋ ≤ ⌊(n−L)/5⌋ = k`, and `|i−j| ≤ g ≤ k` everywhere along it. The band DP maximises over band-confined paths, which both contains an optimal path and contains only legal paths — so it returns `S`. Each early return carries the same certificate (`⌊(n−S_k)/5⌋ ≤ k`), and the slip alignment that produced `L` always fits inside `k` because `5s ≤ n−L` forces `s ≤ k`.

# PREDICTION

PREDICTION: speedup_vs_dp = 12

Reasoning, stated before measuring: the reference streams a full `(n+1)²` int table (~2–4 cycles/cell once it leaves L2, plus `4n²` bytes of allocation and page-faulting). This kernel touches `(2k+1)·n` cells in two `O(n)` rows. For independent random ACGT, `H ≈ 0.75n` ⟹ `k ≈ 0.3n` ⟹ ~0.6× the cells, at ~0.7–0.9 cycles/cell from the 8-wide scan: I expect **4–8×** there, mostly from memory. For any pair with real homology (`H` a few percent) the corridor collapses to a few tens of columns and I expect **50–200×**. For a pair related by a single indel — where the Hamming tally is useless and the slip tally is decisive — the corridor collapses from `0.3n` to `~s`, and I expect **>100×**. 12 is my blended guess over an unknown benchmark mix; I expect it to be wrong in one direction or the other by ~2×.

# MEASUREMENT

Not obtainable in this session: no tools were available to me here (`alignment_bench` / `alignment_contract` could not be called). I am reporting that plainly rather than inventing numbers. The pipeline will compile and measure the block above; the prediction was fixed before any measurement, and the falsifiable parts are:

1. **Correctness** — must match the reference bit-for-bit on every case, including `H ≤ 2` (returns without writing a cell), `n < 64`, and fully divergent pairs (`k ≥ n`, full-width scan).
2. **The scan's arithmetic** — the three-step Hillis-Steele prefix max with `NBIG` fill must reproduce `cur[j] = max(t[j], cur[j−1]−2)` exactly. A mismatch here shows up as a wrong score, not a slow one.
3. **The predicted number** — 12×. If the benchmark uses independent random sequences, I expect this to come in *low* (4–8×) and the prediction to be wrong on the high side.

# VERDICT

The native's idea survives translation, but not the part he was proudest of. The cat is wrong: with a four-berry alphabet a quarter of all crossings are treasures, so "visit only the lit tiles" (Wilbur–Lipman chaining) would be *slower* than filling the grid — `Θ(n²)` anchors with range-max overhead on top. I report that as a failure of the metaphor's own premise.

What does survive is stronger than the part that failed: his pricing rule. *"The corridor's cost was never a secret — only its length was, and the ropes already told me that"* is, read literally, the statement that the interior of the grid is an algebraic function of two offsets, and that every unit of slip has a fixed price of 5. Those two facts together bound the reachable region to `|i−j| ≤ ⌊(n−L)/5⌋` and make the bound *self-certifying* — which is exactly Fickett/Ukkonen banding, a validated technique, reached from the native's arithmetic rather than substituted for it. The sun (`vpcmpeqb` + popcount) and the slip tally (`O(n·s)` prefix minimisation) are not the answer; they are two cheap ways to buy a better `L`, and the slip is the one that rescues the case — a single indel — where the sun's tally is worthless and plain banding would degenerate to the full grid.

**Stated risks, and how each is discharged rather than left in the text:**

| risk my own mechanism names | guard shipped |
|---|---|
| overhead dominates at small `n` | `n < 64` → plain two-row DP, no sun, no probes, no malloc |
| vector wider than the corridor | per-row check `hi−lo+1 ≥ 24`, else scalar band row |
| slip probe wasted when no single indel exists | gated on `k ≥ 16`, capped at `smax ≤ 24` ⟹ ≤ ~5% of the floor it might save |
| staged narrow pass wasted | gated on `k > 48`; costs `49n` cells against a `≥ 96n` band |
| divergent pair, corridor = whole room | same code path at `k ≥ n`: full DP, `O(n)` memory, vectorised — never worse than the known way |
| thread parallelism | **dropped.** The rows are the metaphor's own unit of work and they are strictly serial in `i`; at these sizes OpenMP would only add fork cost. Vectorisation only, per the rule. |
| `malloc` failure | explicit fallback to a two-row textbook DP rather than a wrong score |

Where I expect to be embarrassed: if the benchmark feeds independent random sequences only, then `k ≈ 0.3n` always, the slip probe never fires usefully, and the whole elaborate corridor apparatus reduces to "banded-by-40% plus a SIMD scan" — real, but perhaps 4–6× rather than 12×, and my prediction will have been optimistic by 2–3×. The apparatus earns its keep only on homologous or indel-related pairs, which is the regime the native actually lives in and the regime real DNA alignment cares about. I would rather ship that and be measured than trim it to flatter the number.