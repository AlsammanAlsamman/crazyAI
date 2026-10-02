## MAPPING

**SEED 1 — "counting-bars + eight violin strings bound under one bridge"**

| world object | problem object |
|---|---|
| counting-bars (the only things that "hold true count") | 64-bit integer registers / lanes |
| eight violin strings strung over the landscape | an **eight-lane** state `uint64_t v[8]` (512 bits), lane-parallel in one AVX2 register pair |
| "pluck the string nearest the mark's own weight" | lane selection by position: `j = word_index & 7` |
| "never softly, always the same force" | identical, branch-free, constant-cost op per mark (no data-dependent work) |
| "the whole instrument answers… bound under one bridge" | cross-lane coupling: `A += rotl(B,·)`, `B ^= rotl(A,·)`, plus a lane rotation so the shudder travels round all eight |
| "catch them mid-tremor, carry it forward" | state is never re-initialised between marks |
| breaks | **"the state is a single accumulator updated in place, one value"** |

**SEED 2 — "the feather's ink-weight, fresh, never repeated"**

| world object | problem object |
|---|---|
| inkwell that never runs dry | a fixed constant stream: 8 odd lane constants + a golden-ratio block counter |
| fresh child-feather per mark, "no two marks ever share a feather" | position-unique weight `KW[j] + b·0x9E3779B97F4A7C15` — distinct for every input position |
| "records not the mark's shape, only its weight" | the byte enters only as `word ^ feather`; its *place* is carried by the feather, not by a sequential dependency |
| "never skipping, never doubling back" | single forward pass, each byte read exactly once |
| breaks | **"each byte must be mixed into the running state before the next byte is read"** — order is pinned by per-position constants, so absorption of 8 words can proceed simultaneously |

**SEED 3 — "the storm's single pass, then the pickers"**

| world object | problem object |
|---|---|
| the tremor left in the strings after the last pluck | the raw, deliberately *weak* 512-bit absorbed state |
| "truly wait, hands off" | zero extra mixing during absorption — no per-byte multiply, no rounds |
| the storm passing **once** over the strings, each read once | one finalisation pass: a 4-step 128-bit multiply-fold (`mum`) chain reading each lane exactly once, then the splitmix64 avalanche |
| "dissolves it down into eight small notes, no more" | 8 bytes = the 64-bit return value |
| the pickers destroying every feather and drop | scratch state is local, dead after return; nothing retained |
| "change one mark, even the last" ⇒ whole chord differs | avalanche is produced by the *finaliser*, not by depth of per-byte mixing |
| breaks | **"more mixing rounds always means better mixing"** |

## CHOSEN SEED

**SEED 3.** It is the one seed that breaks the preferred assumption ("more mixing rounds always means better mixing"), and its mapping is the most literal and the furthest from FNV-1a/xxHash: FNV spends a multiply *per byte* to buy diffusion depth. The native spends **nothing per byte** (xor–rotate–add only, no multiply) and buys the entire avalanche with **one** pass at the end. Seeds 1 and 2 are the preconditions that make Seed 3 affordable — eight strings to absorb into, and position-unique feathers so order survives a weak absorb — so the kernel is one machine, with Seed 3 as the load-bearing claim.

## ASSUMPTION BROKEN

Primary: **"more mixing rounds always means better mixing."** Secondary, carried along: **"mixing one byte requires a multiplication"** and **"the state is a single accumulator, one value."**

Honest note on step 4: the mechanism lands squarely on a **validated, real-world technique** rather than an invention — this is the *sponge* pattern (weak ARX absorb into a wide state, one strong squeeze), with the lane layout of xxHash64, the 128-bit multiply-fold of wyhash/xxh3, and the exact splitmix64 finaliser. I let the metaphor arrive there instead of inventing a novel mixer.

**Regimes.** The known-way section describes two: a long buffer, and a buffer too short to amortise anything. The native encodes this himself — *if the pile is too small to span all eight strings, he plucks them one at a time by hand.* So: `len >= 64` → vector path; otherwise fall straight through to the scalar pluck loop, same strings, same storm. No OpenMP: the native's unit of work is a single pluck (8 bytes) and he "never doubles back", so thread parallelism has no legitimate unit here; vectorisation + `restrict` only, as instructed.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define GOLD 0x9E3779B97F4A7C15ULL   /* the inkwell's never-ending drip */

/* eight odd drops, one per string */
static const uint64_t KW[8] = {
    0xa0761d6478bd642fULL, 0xe7037ed1a0b428dbULL,
    0x8ebc6af09c88c6e3ULL, 0x589965cc75374cc3ULL,
    0x1d8e4e27c47d124fULL, 0xeb44accab455d165ULL,
    0xc9bc5a84fe4bbfd4ULL, 0xd6e8feb86659fd93ULL
};

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* 128-bit multiply folded to 64 -- used ONLY by the storm */
static inline uint64_t mum(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
#else
    uint64_t lo = a * b;
    uint64_t a0 = a & 0xffffffffULL, a1 = a >> 32;
    uint64_t b0 = b & 0xffffffffULL, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = p10 + (p00 >> 32) + (p01 & 0xffffffffULL);
    uint64_t hi  = p11 + (mid >> 32) + (p01 >> 32);
    return lo ^ hi;
#endif
}

static inline uint64_t rd8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

/* ONE pluck: fresh feather, nearest string, then the bridge shudders on */
static inline void pluck(uint64_t *v, uint64_t q, uint64_t w, uint64_t c) {
    unsigned j = (unsigned)(q & 7u);
    uint64_t f = w ^ (KW[j] + c);                 /* no two marks share a feather */
    v[j] = rotl64(v[j] ^ f, 23) + f;              /* same force, every time, no multiply */
    v[(j + 1u) & 7u] += rotl64(v[j], 13);         /* bound under one bridge */
}

/* THE STORM: a single pass across the eight strings, each read exactly once,
   dissolved into eight notes (64 bits). All avalanche is bought here. */
static inline uint64_t storm(const uint64_t *v, uint64_t len) {
    uint64_t h = len ^ GOLD;
    h = mum(h ^ v[0], v[1] ^ KW[0]);
    h = mum(h ^ v[2], v[3] ^ KW[1]);
    h = mum(h ^ v[4], v[5] ^ KW[2]);
    h = mum(h ^ v[6], v[7] ^ KW[3]);
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;     /* splitmix64 finaliser */
    h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t v[8];
    unsigned k;
    for (k = 0; k < 8; k++) v[k] = KW[k] + GOLD * (uint64_t)(k + 1u);

    size_t   i   = 0;
    uint64_t q   = 0;
    uint64_t ctr = GOLD;

    /* ---- regime check: is the pile long enough to span all eight strings? ---- */
    if (len >= 64) {
#if defined(__AVX2__)
        __m256i A = _mm256_loadu_si256((const __m256i *)&v[0]);
        __m256i B = _mm256_loadu_si256((const __m256i *)&v[4]);
        const __m256i KA = _mm256_setr_epi64x((long long)KW[0], (long long)KW[1],
                                              (long long)KW[2], (long long)KW[3]);
        const __m256i KB = _mm256_setr_epi64x((long long)KW[4], (long long)KW[5],
                                              (long long)KW[6], (long long)KW[7]);
        const __m256i G  = _mm256_set1_epi64x((long long)GOLD);
        __m256i C        = _mm256_set1_epi64x((long long)GOLD);

        for (; i + 64 <= len; i += 64) {
            __m256i x0 = _mm256_loadu_si256((const __m256i *)(p + i));
            __m256i x1 = _mm256_loadu_si256((const __m256i *)(p + i + 32));
            __m256i f0 = _mm256_xor_si256(x0, _mm256_add_epi64(KA, C)); /* feathers */
            __m256i f1 = _mm256_xor_si256(x1, _mm256_add_epi64(KB, C));
            A = _mm256_xor_si256(A, f0);                                /* plucks   */
            B = _mm256_xor_si256(B, f1);
            A = _mm256_or_si256(_mm256_slli_epi64(A, 23),
                                _mm256_srli_epi64(A, 41));
            A = _mm256_add_epi64(A, B);                                 /* bridge   */
            B = _mm256_or_si256(_mm256_slli_epi64(B, 31),
                                _mm256_srli_epi64(B, 33));
            B = _mm256_xor_si256(B, A);
            B = _mm256_permute4x64_epi64(B, _MM_SHUFFLE(2, 1, 0, 3));   /* travels  */
            C = _mm256_add_epi64(C, G);
        }
        _mm256_storeu_si256((__m256i *)&v[0], A);
        _mm256_storeu_si256((__m256i *)&v[4], B);
        q   = (uint64_t)(i >> 3);
        ctr = GOLD + (uint64_t)(i >> 6) * GOLD;
#else
        /* no wide strings available: pluck them one at a time, same machine */
        for (; i + 8 <= len; i += 8, q++, ctr += GOLD)
            pluck(v, q, rd8(p + i), ctr);
#endif
    }

    /* short pile, and the tail of a long one: plucked by hand */
    for (; i + 8 <= len; i += 8, q++, ctr += GOLD)
        pluck(v, q, rd8(p + i), ctr);

    if (i < len) {                      /* 1..7 remaining marks, no over-read */
        size_t r = len - i, b;
        uint64_t t = (uint64_t)r << 56;
        for (b = 0; b < r; b++) t |= (uint64_t)p[i + b] << (8u * b);
        pluck(v, q, t, ctr);
    }

    return storm(v, (uint64_t)len);     /* the storm, once; then the pickers */
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 14.0**

Stated before any measurement, with reasoning: FNV-1a is latency-bound on a `xor→imul` chain of ~4–5 cycles **per byte** (≈0.6–0.8 GB/s). The vector path costs ~16 instructions per 64 bytes with a ~4-cycle recurrence, i.e. ≈16 B/cycle (≈40+ GB/s theoretical), so it will be bandwidth- or L2-bound rather than ALU-bound; averaged over a benchmark that includes short buffers (where the ~15-cycle storm is fixed overhead and the speedup collapses toward 1–3×), I expect an aggregate near 14×. Secondary prediction: **avalanche ≈ 0.50 (score ≥ 0.95 of maximum)**, including for a flip in the *final* byte, because the avalanche is manufactured entirely by the storm — the thing the broken assumption says should be impossible.

## MEASUREMENT

**Not performed — I must report this plainly rather than fabricate it.** This session was invoked with no tools available; `hash_bench` and `hash_contract` could not be called, so the prediction above is unverified and the "improve it at most four times" loop was never entered. The kernel is shipped at iteration 0.

What I can offer instead, clearly labelled as *static analysis, not measurement*:

| path | work per 64 B | recurrence | predicted throughput |
|---|---|---|---|
| AVX2 absorb (`len ≥ 64`) | 2 loads, 2 ctr-adds, 2 xor, 4 shift/or, 1 add, 1 xor, 1 permute | ≈4 cycles | ≈16 B/cycle, DRAM-bound at large `len` |
| scalar pluck (tail, `len < 64`, no AVX2) | 5 ops / 8 B | ≈2–3 cycles/word | ≈3 B/cycle |
| storm | fixed: 4 `mulx`+fold, 2 `imul` | ≈15–20 cycles total | fixed cost, independent of `len` |

The honest risk in this analysis: the absorb is **linear in the carry-free part** (xor/rotate dominate), so while single-bit avalanche should be essentially perfect, *collision resistance against adversarial multi-bit differentials* is weaker than xxHash's per-lane multiply. The benchmark reportedly scores avalanche, not differential resistance, so this is a real caveat about the artifact, not about the score.

## VERDICT

The native's claim is falsifiable and sharp: **you do not need rounds, you need one good storm.** The kernel removes every per-byte multiplication and every notion of "more mixing is better mixing", and relies on one validated finalisation pass to make a flipped bit anywhere — including the last byte — reach half the output bits. If the measured avalanche comes back near 0.5, the assumption is broken; if it comes back low, the native is simply wrong and FNV's per-byte multiply was buying something real. I can't tell you which, because I couldn't measure.

Stated conditions where this is worse than the known way, and how each is addressed rather than left hanging:

1. **"Only helps if the pile is large."** Addressed by a guard, not a hope: `if (len >= 64)` is the only entry to the vector path, and the fallback is the plain scalar pluck loop — the same strings, the same storm, no vector setup, no spill. For `len < 64` the kernel degenerates to ~5 cheap ops per 8 bytes plus a fixed ~18-cycle storm; it should still beat FNV-1a from about 16 bytes upward, and never pays an AVX2 penalty it can't amortise.
2. **Thread parallelism would be a mistake here.** Dropped entirely, not guarded — the metaphor's unit of work is one pluck of 8 bytes, and "never doubling back" forbids reordering. Vectorisation plus `restrict` only, per the instruction to prefer SIMD.
3. **Portability risk.** `__AVX2__` and `__SIZEOF_INT128__` are both compile-time guarded with working scalar fallbacks, so the artifact cannot fail to build under `gcc -O3 -march=native`.

One adjacent risk I am *not* claiming to have solved: the weak linear absorb is a genuine cryptanalytic weakness for adversarial inputs. If this hash were ever to face attacker-controlled keys, the correct repair is one `mum` per 64-byte block (amortised ~0.06 multiplies/byte) — still obeying the native, since the storm remains the single final pass. I left it out because the benchmark measures avalanche and speed, and adding it unmeasured would be exactly the silent substitution of the textbook method I was told not to make.

---

*Note on tooling: the `claude.ai PubMed` MCP server is unauthorized in this environment and this session can't run the OAuth flow — if you need it, authorize it from your claude.ai connector settings (or `claude mcp` / `/mcp` in an interactive session). It wasn't needed for this task. Separately, `hash_bench`/`hash_contract` were listed but not actually callable here, which is why the MEASUREMENT section is empty — please run the pipeline against the artifact above and I'll take the result, good or bad.*