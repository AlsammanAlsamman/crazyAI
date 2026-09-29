# APPROACH

**How the disguised solution maps onto the real problem**

| Granny's field | Hash kernel |
|---|---|
| The one jar on her belt, stirred in place at every flower | The single FNV-1a accumulator `h`, serially updated once per byte — a latency-bound dependency chain |
| "Every flower must be worked in before she steps" | The assumption that byte *i* must be folded into state before byte *i+1* is read — dropped |
| Granny **plus three helpers** walking together | 4 independent 64-bit lanes = one AVX2 `__m256i` (scalar fallback: 4 independent word-pairs) |
| Each dips at every other flower; **pairs of neighbouring flowers share one little cup**, stirred right there | Leaf level: two 32-byte chunks are combined lane-wise by one nonlinear node `vnode` (each lane pairs neighbouring 8-byte words inside that lane's sub-row) |
| "A row of small cups instead of individual flowers" | An implicit level of intermediate 64-bit node values — never materialised to memory, held in registers |
| **Pair up those cups two at a time, pour into a bigger cup, twist-stir again; repeat, fewer each round, until one cup is left** | Binary tree reduction. Inside a 256-byte block it is fully unrolled (8 vectors → 4 → 2 → 1). Across blocks it is a streaming Merkle/binary-counter reduction: a small stack of pending subtree values, merged pairwise whenever two subtrees of equal rank meet — exact log-depth pairing with only ~1 extra node per block and O(log n) registers, no O(n) scratch |
| That last cup is the field blend | Final vector's 4 lanes reduced pairwise (2 nodes then 1), length folded in, strong 128-bit-multiply + `fmix64` finalizer |
| "Twisting three or four times per flower must mix better" (nobody's sure) | The "more rounds is always better" assumption — rejected. One node per pair suffices because *depth* (log n levels of nonlinear mixing) supplies the avalanche, not repetition at one site |

**Why this is faster, mechanically.** FNV-1a pays one multiply *per byte* on a serial critical path (~4–5 cycles/byte). The tree pays `N-1` nodes for `N` leaf vectors — ~0.25 vector-ops/byte — and every node at a given level is independent, so the machine runs 4 lanes × several nodes in parallel. The node uses `vpmuludq` (32×32→64, the only 64-bit-producing multiply AVX2 has), adds back `a+b` swapped so no input bit is ever discarded, and keeps 4 leaf nodes in flight per block for ILP.

**Avalanche.** A flipped input bit perturbs one lane of one leaf node, then passes through log₂ further nonlinear nodes, the cross-lane reduction (two 128-bit-fold multiplies), and a `mum` + murmur3 `fmix64` finalizer — so one bit reaches ~half the output bits even for the shortest paths. Length is folded in so zero-extensions differ. No out-of-bounds reads: tails are covered by *overlapping* final-block / final-chunk reads (legal because the relevant length floor guarantees the bytes exist), never by reading past `len`.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- keys ---- */
static const uint64_t KS[16] = {
    0xa0761d6478bd642fULL, 0xe7037ed1a0b428dbULL, 0x8ebc6af09c88c6e3ULL,
    0x589965cc75374cc3ULL, 0x1d8e4e27c47d124fULL, 0xff51afd7ed558ccdULL,
    0xc4ceb9fe1a85ec53ULL, 0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL,
    0x94d049bb133111ebULL, 0x2545f4914f6cdd1dULL, 0xd6e8feb86659fd93ULL,
    0xca62c1d6b1a9e38dULL, 0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
    0xc0ac29b7c97c50ddULL
};

static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t rd32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }
static inline uint64_t rotl64(uint64_t x, int r){ return (x<<r)|(x>>(64-r)); }

/* 128-bit product folded to 64 bits */
static inline uint64_t mum(uint64_t a, uint64_t b){
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t ha=a>>32, la=(uint32_t)a, hb=b>>32, lb=(uint32_t)b;
    uint64_t rh=ha*hb, m0=ha*lb, m1=hb*la, rl=la*lb;
    uint64_t t=rl+(m0<<32); uint64_t c=(t<rl);
    uint64_t lo=t+(m1<<32); c+=(lo<t);
    uint64_t hi=rh+(m0>>32)+(m1>>32)+c;
    return lo^hi;
#endif
}

/* scalar tree node: combine two cups into one, twist-stirred */
static inline uint64_t snode(uint64_t x, uint64_t y, uint64_t k){
    uint64_t a = x ^ k;
    uint64_t b = y ^ rotl64(k, 32);
    return mum(a, b) ^ (a + rotl64(b, 32));
}

static inline uint64_t finish(uint64_t h, uint64_t len){
    h ^= len * 0x9e3779b97f4a7c15ULL;
    h ^= mum(h ^ 0xa0761d6478bd642fULL, h + 0xe7037ed1a0b428dbULL);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 29; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 32;
    return h;
}

/* ---- tiny inputs (len < 16) ---- */
static uint64_t hash_small(const unsigned char *p, size_t len){
    uint64_t a, b;
    if (len >= 8)      { a = rd64(p);            b = rd64(p + len - 8); }
    else if (len >= 4) { a = rd32(p);            b = rd32(p + len - 4); }
    else if (len)      { a = ((uint64_t)p[0]<<16) | ((uint64_t)p[len>>1]<<8) | (uint64_t)p[len-1]; b = 0x9e3779b9ULL; }
    else               { a = 0; b = 0; }
    return finish(snode(a, b, KS[0]), (uint64_t)len);
}

/* ---- medium inputs: explicit pairwise tree over 16-byte chunks ---- */
static uint64_t hash_medium(const unsigned char *p, size_t len){
    uint64_t buf[20];
    size_t n = (len + 15) >> 4;            /* 1..16 chunks for len < 256 */
    for (size_t i = 0; i < n; i++) {
        size_t off = (i + 1 < n) ? (i << 4) : (len - 16);   /* last chunk overlaps */
        buf[i] = snode(rd64(p + off), rd64(p + off + 8), KS[i & 15]);
    }
    size_t m = n, r = 0;
    while (m > 1) {                        /* pair up the cups, round after round */
        size_t h2 = m >> 1;
        for (size_t j = 0; j < h2; j++)
            buf[j] = snode(buf[2*j], buf[2*j+1], KS[(j + 5*r) & 15]);
        if (m & 1) { buf[h2] = buf[m-1]; m = h2 + 1; } else m = h2;
        r++;
    }
    return finish(buf[0], (uint64_t)len);
}

#if defined(__AVX2__)
/* ---- 4-lane vector node: Granny plus three helpers, one twist-stir ---- */
static const uint64_t VKS[8][4] = {
    { 0xa0761d6478bd642fULL, 0xe7037ed1a0b428dbULL, 0x8ebc6af09c88c6e3ULL, 0x589965cc75374cc3ULL },
    { 0x1d8e4e27c47d124fULL, 0xff51afd7ed558ccdULL, 0xc4ceb9fe1a85ec53ULL, 0x9e3779b97f4a7c15ULL },
    { 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL, 0x2545f4914f6cdd1dULL, 0xd6e8feb86659fd93ULL },
    { 0xca62c1d6b1a9e38dULL, 0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL, 0xc0ac29b7c97c50ddULL },
    { 0x9ae16a3b2f90404fULL, 0xc3a5c85c97cb3127ULL, 0xb492b66fbe98f273ULL, 0x9ddfea08eb382d69ULL },
    { 0x85ebca6bc2b2ae63ULL, 0xcc9e2d51b873593dULL, 0x1b873593cc9e2d51ULL, 0xe9846af9b1a615d3ULL },
    { 0x2b7e151628aed2a7ULL, 0x3243f6a8885a308dULL, 0x13198a2e03707345ULL, 0xa4093822299f31d0ULL },
    { 0x082efa98ec4e6c89ULL, 0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL, 0xc0ac29b7c97c50ddULL }
};
#define VK(i) _mm256_loadu_si256((const __m256i *)VKS[i])

static inline __m256i vnode(__m256i x, __m256i y, __m256i k){
    __m256i a = _mm256_xor_si256(x, k);
    __m256i b = _mm256_add_epi64(y, k);
    __m256i d = _mm256_xor_si256(a, _mm256_shuffle_epi32(b, 0xB1));
    __m256i p = _mm256_mul_epu32(d, _mm256_srli_epi64(d, 32)); /* lo32*hi32 */
    __m256i s = _mm256_add_epi64(a, b);                        /* keep all bits */
    return _mm256_xor_si256(p, _mm256_shuffle_epi32(s, 0xB1));
}

/* 256-byte block -> one cup: 8 vectors -> 4 -> 2 -> 1 */
static inline __m256i vblock(const unsigned char *q){
    __m256i v0 = _mm256_loadu_si256((const __m256i *)(q +   0));
    __m256i v1 = _mm256_loadu_si256((const __m256i *)(q +  32));
    __m256i v2 = _mm256_loadu_si256((const __m256i *)(q +  64));
    __m256i v3 = _mm256_loadu_si256((const __m256i *)(q +  96));
    __m256i v4 = _mm256_loadu_si256((const __m256i *)(q + 128));
    __m256i v5 = _mm256_loadu_si256((const __m256i *)(q + 160));
    __m256i v6 = _mm256_loadu_si256((const __m256i *)(q + 192));
    __m256i v7 = _mm256_loadu_si256((const __m256i *)(q + 224));
    __m256i a0 = vnode(v0, v1, VK(0));
    __m256i a1 = vnode(v2, v3, VK(1));
    __m256i a2 = vnode(v4, v5, VK(2));
    __m256i a3 = vnode(v6, v7, VK(3));
    __m256i b0 = vnode(a0, a1, VK(4));
    __m256i b1 = vnode(a2, a3, VK(5));
    return vnode(b0, b1, VK(6));
}

static uint64_t hash_big(const unsigned char *p, size_t len){
    size_t nb = len >> 8, rem = len & 255;
    __m256i st[64]; unsigned dp[64]; int top = 0;
    const unsigned char *q = p;
    for (size_t i = 0; i < nb; i++, q += 256) {
        __m256i v = vblock(q);
        unsigned d = 0;                            /* streaming pairwise merge */
        while (top > 0 && dp[top-1] == d) { top--; v = vnode(st[top], v, VK(7 - (d & 1))); d++; }
        st[top] = v; dp[top] = d; top++;
    }
    if (rem) {                                     /* overlapping final block */
        __m256i v = vblock(p + len - 256);
        unsigned d = 0;
        while (top > 0 && dp[top-1] == d) { top--; v = vnode(st[top], v, VK(7 - (d & 1))); d++; }
        st[top] = v; dp[top] = d; top++;
    }
    __m256i acc = st[--top];
    while (top > 0) { top--; acc = vnode(st[top], acc, VK(6)); }
    uint64_t l[4];
    _mm256_storeu_si256((__m256i *)l, acc);
    uint64_t x = snode(l[0], l[1], KS[2]);
    uint64_t y = snode(l[2], l[3], KS[5]);
    return finish(snode(x, y, KS[9]), (uint64_t)len);
}
#else
/* scalar 4-wide fallback: 64-byte block -> one cup */
static inline uint64_t sblock(const unsigned char *q){
    uint64_t w0 = snode(rd64(q +  0), rd64(q +  8), KS[0]);
    uint64_t w1 = snode(rd64(q + 16), rd64(q + 24), KS[1]);
    uint64_t w2 = snode(rd64(q + 32), rd64(q + 40), KS[2]);
    uint64_t w3 = snode(rd64(q + 48), rd64(q + 56), KS[3]);
    return snode(snode(w0, w1, KS[4]), snode(w2, w3, KS[5]), KS[6]);
}
static uint64_t hash_big(const unsigned char *p, size_t len){
    size_t nb = len >> 6, rem = len & 63;
    uint64_t st[64]; unsigned dp[64]; int top = 0;
    const unsigned char *q = p;
    for (size_t i = 0; i < nb; i++, q += 64) {
        uint64_t v = sblock(q); unsigned d = 0;
        while (top > 0 && dp[top-1] == d) { top--; v = snode(st[top], v, KS[8 + (d & 3)]); d++; }
        st[top] = v; dp[top] = d; top++;
    }
    if (rem) {
        uint64_t v = sblock(p + len - 64); unsigned d = 0;
        while (top > 0 && dp[top-1] == d) { top--; v = snode(st[top], v, KS[8 + (d & 3)]); d++; }
        st[top] = v; dp[top] = d; top++;
    }
    uint64_t acc = st[--top];
    while (top > 0) { top--; acc = snode(st[top], acc, KS[12]); }
    return finish(acc, (uint64_t)len);
}
#endif

uint64_t kernel(const unsigned char *data, size_t len){
    if (len < 16) return hash_small(data, len);
#if defined(__AVX2__)
    if (len >= 256) return hash_big(data, len);
#else
    if (len >= 64)  return hash_big(data, len);
#endif
    return hash_medium(data, len);
}
```

# PREDICTION

PREDICTION: 20

(Predicted speedup factor over the FNV-1a baseline on a multi-KB buffer: ~0.08–0.25 cycles/byte for the tree versus ~4–5 cycles/byte for FNV-1a's serial multiply chain, discounted for cache/RAM bandwidth and for short-input cases where the gap narrows to ~1.5–5×.)

# MEASUREMENT

**Not measured — reported plainly as such.** This session was invoked with no tools available: `hash_bench`, `hash_contract`, and the rest were listed in the prompt but are not callable here (the environment exposes only Gmail/Drive/Slack/Docs connectors, two of which are unauthenticated). I therefore have zero of the four permitted improvement iterations backed by data, and the number above is an unverified prediction, not a result.

What I verified by reasoning only, and what the bench should check:

- **Contract**: exact signature; pure function of `(data, len)`; no globals mutated; `len == 0` returns without dereferencing `data` (safe for `NULL`).
- **Bounds**: every read is inside `[data, data+len)`. Overlapping tail reads require `len ≥ 16` (medium), `len ≥ 256` (AVX2 big), `len ≥ 64` (scalar big) — each guaranteed by the dispatch in `kernel`.
- **Cost model**: leaf+internal nodes = `N-1` for `N` 32-byte leaf vectors ⇒ ~0.25 vector-ops/byte ≈ 7 ops per 256 bytes at the leaf, 4-way lane parallelism and 4 independent leaf nodes per block for ILP; the cross-block binary-counter merge amortises to ~1 extra node per 256 bytes.
- **Residual risks I could not rule out without the bench**: (a) `vnode` degenerates to a purely linear step when `d.lo32` or `d.hi32` is zero (~2⁻³¹ per node) — statistically harmless for avalanche, but a strict bit-independence test is the thing that would catch it if I'm wrong; (b) the 256-byte threshold may be mistuned if the harness weights 64–255-byte inputs heavily; (c) `-march=native` with AVX-512 present is untested (only AVX2 intrinsics are used, so it should be correct, just not maximal).

# VERDICT

The mechanism translates cleanly and, on paper, decisively: replacing the one-jar accumulator with a log-depth pairwise combining tree removes the per-byte serial multiply chain — the actual bottleneck in FNV-1a — and replaces it with ~¼ of a vector op per byte spread over four parallel lanes, while keeping avalanche through tree *depth* plus a strong finalizer rather than through more stirring at one site. I expect a large win on multi-KB inputs (order 20×, bandwidth-capped) and a modest one on tiny inputs.

But the honest verdict is **unproven**. I was asked to measure and improve up to four times; with no tool access I did neither. The artifact is complete and ready to compile with `gcc -O3 -march=native -fopenmp -lm`; if `hash_bench` shows it below prediction, the first two things to tune are the 256-byte block threshold and the medium-path chunk size — not the mechanism, which stays a pairwise combining tree.