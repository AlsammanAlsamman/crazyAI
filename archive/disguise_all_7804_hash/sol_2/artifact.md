# APPROACH

**Mapping the bakery onto the hash.**

| Bakery | Hash |
|---|---|
| One bowl, walk the shelf jar‑by‑jar, twist‑stir after every scoop | FNV‑1a: one accumulator, one byte at a time, one multiply per byte — a fully serial dependency chain (~4–5 cycles *per byte*) |
| "Split the jars into a few clusters, not necessarily in shelf order" | Partition the buffer into **independent lanes**: 16 lanes of 64 bits held in 4 AVX2 accumulators (128 B consumed per iteration), each lane owning a fixed stride of the buffer |
| "Each helper gets their own small bowl and twist‑stirs *their* cluster, all at the same time, nobody waiting on anybody else" | Each accumulator runs its **own** bijective multiply‑xorshift round (`t = acc+w; t ^= t>>32; acc = t*K`) — 16 independent dependency chains, so out‑of‑order execution and SIMD overlap them completely |
| "Cellar / cart by the door / kitchen shelf" — physically separate regions | For very large buffers (≥ 2 MB) an extra, coarser split: 8 fixed partitions hashed by **OpenMP** helpers. Partition boundaries depend only on `len`, never on thread count, so the result stays deterministic |
| "Pours all the mini‑swirls together and gives *that* a couple of hard twist‑stirs" | Tree reduction of the lanes (3 vector rounds, 16→4 lanes), then `mum` (64×64→128 multiply, fold hi^lo) pairs, then a MurmurHash3‑grade finalizer. This is where the avalanche is *guaranteed*, cheaply, once — instead of paying for it per byte |
| "Changing any jar still turns the swirl unrecognizable" | Every per‑lane round is a **bijection** (add, xor‑shift, multiply by an odd constant), so no input difference can be destroyed; the final strong mixer then expands any surviving difference to ~half the output bits. `len` is folded in too, so length extension of a lane is distinguished |

The assumptions being broken: mixing does **not** have to happen before the next byte is read, the state does **not** have to be one accumulator, and the buffer does **not** have to be walked strictly in order. The assumption kept (correctly): mixing needs a multiply — but only ~1 multiply per 8 bytes in the bulk, plus a constant number of strong ones at the end, rather than one per byte.

Small inputs get a short path (overlapping head/tail reads, one `mum`, one finalizer) so the fixed "final bowl" cost is not paid twice. All reads stay strictly inside `[data, data+len)` — no over‑read, no scratch allocation.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define HP0 0x9E3779B185EBCA87ULL
#define HP1 0xC2B2AE3D27D4EB4FULL
#define HP2 0x165667B19E3779F9ULL
#define HP3 0x85EBCA77C2B2AE63ULL
#define HP4 0x27D4EB2F165667C5ULL
#define HP5 0xD6E8FEB86659FD93ULL
#define K32 0x9E3779B1u            /* odd -> multiply is a bijection mod 2^64 */

static inline uint64_t rd8(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t rd4(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }

/* 64x64 -> 128 multiply, folded: strong 2-input mixer, ~1 multiply */
static inline uint64_t mum(uint64_t a, uint64_t b)
{
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t al=a&0xffffffffULL, ah=a>>32, bl=b&0xffffffffULL, bh=b>>32;
    uint64_t ll=al*bl, lh=al*bh, hl=ah*bl, hh=ah*bh;
    uint64_t mid=(ll>>32)+(lh&0xffffffffULL)+(hl&0xffffffffULL);
    uint64_t lo=(ll&0xffffffffULL)|(mid<<32);
    uint64_t hi=hh+(lh>>32)+(hl>>32)+(mid>>32);
    return lo ^ hi;
#endif
}

/* final "hard twist-stir": MurmurHash3 fmix64 -- full 64-bit avalanche */
static inline uint64_t fin64(uint64_t h)
{
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33; return h;
}

#if defined(__AVX2__)
/* one helper's twist-stir on 4 lanes at once: acc = ((acc+v) ^ ((acc+v)>>32)) * K  */
static inline __m256i vround(__m256i acc, __m256i v, __m256i k)
{
    __m256i t  = _mm256_add_epi64(acc, v);
    t          = _mm256_xor_si256(t, _mm256_srli_epi64(t, 32));
    __m256i lo = _mm256_mul_epu32(t, k);                          /* lo32(t)*K */
    __m256i hi = _mm256_mul_epu32(_mm256_srli_epi64(t, 32), k);   /* hi32(t)*K */
    return _mm256_add_epi64(lo, _mm256_slli_epi64(hi, 32));       /* == t*K mod 2^64 */
}
#endif

static uint64_t hcore(const unsigned char *p, size_t len, uint64_t seed)
{
    uint64_t h;

    if (len <= 16) {                                  /* one bowl is enough */
        uint64_t a, b;
        if (len >= 8)      { a = rd8(p);              b = rd8(p + len - 8); }
        else if (len >= 4) { a = rd4(p);              b = rd4(p + len - 4); }
        else if (len)      { a = ((uint64_t)p[0]<<16) | ((uint64_t)p[len>>1]<<8)
                                 | (uint64_t)p[len-1]; b = 0; }
        else               { a = 0;                   b = 0; }
        h = mum(a ^ HP2 ^ seed, b ^ HP3 ^ (uint64_t)len);

    } else if (len < 256) {                           /* two helpers, scalar */
        const unsigned char *q = p; size_t n = len;
        uint64_t h0 = HP0 ^ seed, h1 = HP1 ^ (uint64_t)len;
        while (n >= 32) {
            h0 = mum(h0 ^ rd8(q),      HP2 ^ rd8(q + 8));
            h1 = mum(h1 ^ rd8(q + 16), HP3 ^ rd8(q + 24));
            q += 32; n -= 32;
        }
        if (n > 16) h0 = mum(h0 ^ rd8(q), HP2 ^ rd8(q + 8));
        h1 = mum(h1 ^ rd8(p + len - 16), HP3 ^ rd8(p + len - 8)); /* overlapping tail */
        h  = h0 ^ h1;

    } else {
#if defined(__AVX2__)
        const unsigned char *q = p; size_t n = len;
        const __m256i k  = _mm256_set1_epi64x((long long)(uint64_t)K32);
        __m256i a0 = _mm256_set_epi64x((long long)(HP0 ^ seed), (long long)HP1,
                                       (long long)HP2,          (long long)HP3);
        __m256i a1 = _mm256_set_epi64x((long long)HP4, (long long)HP5,
                                       (long long)(HP0 + (uint64_t)len),
                                       (long long)(HP1 ^ (uint64_t)len));
        __m256i a2 = _mm256_set_epi64x((long long)(HP2 ^ seed), (long long)HP3,
                                       (long long)HP4,          (long long)HP5);
        __m256i a3 = _mm256_set_epi64x((long long)HP1, (long long)HP0,
                                       (long long)(HP5 ^ (uint64_t)len),
                                       (long long)(HP4 + (uint64_t)len));
        while (n >= 128) {                            /* 16 lanes stirring at once */
            a0 = vround(a0, _mm256_loadu_si256((const __m256i *)(q      )), k);
            a1 = vround(a1, _mm256_loadu_si256((const __m256i *)(q +  32)), k);
            a2 = vround(a2, _mm256_loadu_si256((const __m256i *)(q +  64)), k);
            a3 = vround(a3, _mm256_loadu_si256((const __m256i *)(q +  96)), k);
            q += 128; n -= 128;
        }
        {                                             /* branch-free overlapping tail */
            const unsigned char *e = p + len - 128;
            a0 = vround(a0, _mm256_loadu_si256((const __m256i *)(e      )), k);
            a1 = vround(a1, _mm256_loadu_si256((const __m256i *)(e +  32)), k);
            a2 = vround(a2, _mm256_loadu_si256((const __m256i *)(e +  64)), k);
            a3 = vround(a3, _mm256_loadu_si256((const __m256i *)(e +  96)), k);
        }
        a0 = vround(a0, _mm256_permute4x64_epi64(a2, 0x4E), k);   /* pour together */
        a1 = vround(a1, _mm256_permute4x64_epi64(a3, 0x1B), k);
        a0 = vround(a0, _mm256_permute4x64_epi64(a1, 0xB1), k);
        {
            uint64_t l[4];
            _mm256_storeu_si256((__m256i *)l, a0);
            h = mum(l[0] ^ HP0, l[1] ^ HP1) ^ mum(l[2] ^ HP2, l[3] ^ HP3);
        }
#else
        const unsigned char *q = p; size_t n = len;
        uint64_t h0 = HP0 ^ seed, h1 = HP1, h2 = HP2, h3 = HP3 ^ (uint64_t)len;
        while (n >= 64) {
            h0 = mum(h0 ^ rd8(q     ), HP4 ^ rd8(q +  8));
            h1 = mum(h1 ^ rd8(q + 16), HP5 ^ rd8(q + 24));
            h2 = mum(h2 ^ rd8(q + 32), HP0 ^ rd8(q + 40));
            h3 = mum(h3 ^ rd8(q + 48), HP1 ^ rd8(q + 56));
            q += 64; n -= 64;
        }
        {
            const unsigned char *e = p + len - 64;
            h0 = mum(h0 ^ rd8(e     ), HP4 ^ rd8(e +  8));
            h1 = mum(h1 ^ rd8(e + 16), HP5 ^ rd8(e + 24));
            h2 = mum(h2 ^ rd8(e + 32), HP0 ^ rd8(e + 40));
            h3 = mum(h3 ^ rd8(e + 48), HP1 ^ rd8(e + 56));
        }
        h = mum(h0 ^ h1 ^ HP2, h2 ^ h3 ^ HP3);
#endif
    }
    return fin64(mum(h ^ (uint64_t)len ^ HP4, HP5 ^ seed));
}

uint64_t kernel(const unsigned char *data, size_t len)
{
#if defined(_OPENMP)
    if (len >= (size_t)(1u << 21)) {                  /* extra helpers for the cellar */
        enum { NP = 8 };
        size_t blk = (len / NP) & ~(size_t)127;       /* depends only on len -> deterministic */
        if (blk >= 4096) {
            uint64_t part[NP];
            int i;
            #pragma omp parallel for schedule(static)
            for (i = 0; i < NP; i++) {
                size_t off = (size_t)i * blk;
                size_t l   = (i == NP - 1) ? (len - off) : blk;
                part[i] = hcore(data + off, l, HP0 * (uint64_t)(i + 1));
            }
            {
                uint64_t h = HP1;
                for (i = 0; i < NP; i++)
                    h = mum(h ^ part[i] ^ HP2, HP3 + (uint64_t)i);
                return fin64(mum(h ^ (uint64_t)len, HP5));
            }
        }
    }
#endif
    return hcore(data, len, 0);
}
```

# PREDICTION

PREDICTION: 25

(Read as: ~25× the throughput of the FNV‑1a reference on the benchmark's aggregate mix. Reasoning: FNV‑1a is latency‑bound at ~0.2 B/cycle; the bulk loop here issues 9 µops per 32 B across 16 independent chains, port‑0‑limited at ~14–16 B/cycle for cache‑resident data, i.e. ~60–70× on large buffers, dropping to ~3–5× on ≤32‑byte inputs where the fixed finalizer dominates. A mixed‑length harness lands in between; a large‑buffer‑only harness would exceed the prediction but would then be memory‑bandwidth‑capped.)

# MEASUREMENT

**Not measured — reported as failure of the measurement step, not as success.** This session was invoked with no tools available (`hash_bench` / `hash_contract` were listed but are not callable here), and there is no compiler run in this environment, so I did zero of the four allowed measure‑and‑improve iterations. Everything above is static reasoning about µop counts, port pressure and dependency‑chain latency, not observed numbers. The prediction stands as written, unverified.

What I *did* verify by hand, since it costs nothing and is the part most likely to fail silently:

- **Bounds.** Every load is inside `[data, data+len)`: short path guarded per length class; medium path's `n > 16` case has `q+16 ≤ p+len`; both tails are `p+len-16` / `p+len-128` under branch guards `len ≥ 17` and `len ≥ 256`. `len == 0` performs no load.
- **Coverage.** Every byte reaches some accumulator (checked at the boundaries `len` = 0,1,3,4,7,8,16,17,32,255,256,383). Tail overlap re‑reads bytes but never skips any.
- **Non‑destructive rounds.** `acc → ((acc+v) ^ ((acc+v)>>32)) * K` with `K` odd is a composition of three bijections, so no single‑bit input difference can be annihilated inside a lane; `fin64` (Murmur3 fmix64) then supplies the ~0.5 flip probability per output bit.
- **Determinism under OpenMP.** Partition count and boundaries are functions of `len` alone; each thread writes a distinct `part[i]`; the combine is a fixed serial order. Result is independent of `OMP_NUM_THREADS`, including 1.

Things I'd expect the real benchmark to challenge, in order: (1) whether the ≥256 B AVX2 threshold is too high or too low for the harness's length mix; (2) whether the 2 MB OpenMP threshold ever trips, and whether it helps or hurts if it does — if it hurts, raise it or delete that block, which changes nothing else; (3) whether `fin64`'s two serial multiplies are overkill for tiny inputs (one `mum` would save ~8 cycles at some avalanche risk).

# VERDICT

The mechanism translates cleanly: helpers → independent SIMD lanes (and, for huge buffers, OpenMP partitions); per‑helper twist‑stirring → a bijective multiply‑xorshift round per lane; pouring together plus a couple of hard stirs → a tree reduction into `mum` pairs and a Murmur3 finalizer. The serial byte‑at‑a‑time chain that makes FNV‑1a slow is gone entirely, while avalanche is preserved by making every lane step information‑preserving and paying for real diffusion once, at the end, instead of once per byte.

I believe this is correct and substantially faster, but **I have not demonstrated either claim** — no compile, no contract check, no timing was possible in this session. Treat the 25× as a hypothesis to be falsified by the pipeline's first run, and the four improvement rounds as unspent.