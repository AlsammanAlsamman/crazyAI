## MAPPING

**SEED 1 — "counting-bars and violin strings bound under one bridge; one pluck shivers into every string" = the mixing engine**

| World object | Computational object |
|---|---|
| counting-bars (where the tensed and the woman death lean) | the integer ALU / 64-bit lanes — the only thing that "holds true count"; no float, no table in RAM |
| the pile of marks in their given order | `data[0..len-1]`, the byte buffer, consumed in index order |
| eight violin strings strung over the landscape | **eight 64-bit accumulators** = 512-bit wide state (two `__m256i`) |
| "pluck the string nearest the mark's own weight" | byte at offset `i` lands in lane `(i/8) mod 8` — the 64-byte stripe spreads 8 words across 8 strings |
| "never softly, always the same force" | constant, branchless work per byte; no data-dependent path |
| strings bound under one bridge; a shudder in one bends the pitch of the rest | **cross-lane coupling**: `acc[i^1] += d[i]` — the bridge term, so a lane is never independent |
| catching the strings mid-tremor, carrying it into the next pluck | state is never reset between stripes; running carry |

Breaks: *"the state is a single accumulator updated in place, one value."*

**SEED 2 — "the feather's ink-weight per mark, drawn fresh, never repeated, fixes strict order"**

| World object | Computational object |
|---|---|
| inkwell that never runs dry | a **Weyl counter per lane**: `w[i] += FSTEP[i]`, period 2⁶⁴, generated in registers — never a stored table |
| fresh child-feather per mark, no two marks share one | per-position key material: stripe *s*, lane *i* → `w_i = F_i + s·T_i`, unique for every (position, lane) |
| the feather records the weight, not the shape | the key is XORed into the word and only its **32×32→64 product** is kept — magnitude, not the literal bytes |
| "no two marks ever share a feather" | identical bytes at different offsets accumulate differently ⇒ strict order-sensitivity, and no periodic secret ⇒ **no periodic re-scramble needed** |

Breaks: *"mixing one byte requires a multiplication"* — here one cheap 32×32 multiply serves eight bytes, and the order-fixing comes from the feather, not from the multiply.

**SEED 3 — "the storm's single pass collapses the final tremor into a small fixed token; the pickers destroy every feather and drop"**

| World object | Computational object |
|---|---|
| "only when the last mark has been plucked do I wait — hands off" | **zero finalization work inside the hot loop**; the loop is add/xor/mul only |
| the storm passing **once** over the strings | exactly one strong finalizer: 8-lane 128-bit fold + one `rrmxmx` avalanche |
| "eight small notes, no more" | 512-bit state → 64-bit output |
| the pickers take apart every feather, drop, mid-tremor count | all scratch lives in registers; nothing written to memory, no lookup table, no buffer — one-way, nothing of the input order survives |
| "change one mark, even the last, and the whole chord comes out a different token" | avalanche requirement: every lane reaches the output through the single storm |

Breaks: ***"more mixing rounds always means better mixing."*** The native does *less* mixing per byte than FNV-1a (no 64-bit multiply, no serial dependency) and buys all the quality back with **one** pass at the end.

## CHOSEN SEED

**SEED 3**, the storm's single pass. It is the only one of the three that attacks the preferred assumption, and it is the most different from the known way: FNV-1a/xxHash spend their mixing budget *per byte*, inside the dependency chain; the native spends almost nothing per byte and pays once, at the end, hands off the strings. Seeds 1 and 2 are not alternatives — they are what makes Seed 3 survivable: a single final pass is only sufficient because the state is 512 bits wide, bridge-coupled, and fed through never-repeating feathers.

Note on Step 4: this mechanism lands on an **already-validated** design rather than a novel one. Eight 64-bit accumulators over 64-byte stripes, a swapped-lane additive bridge term, `mul32→64` of key-XORed words, and a single `mergeAccs` + avalanche at the end is, object for object, the structure of **XXH3**'s long-input path (SMHasher-validated), and the finalizer is Pelle Evensen's **rrmxmx**. The one place I let the metaphor depart from XXH3 is the feather: XXH3 reuses a 192-byte secret window and therefore *must* re-scramble accumulators every 1024 bytes; the native's inkwell "never runs dry," so per-lane Weyl counters give non-repeating key material forever and that periodic extra round is correctly deleted — which is exactly the assumption being broken.

## ASSUMPTION BROKEN

"More mixing rounds always means better mixing" (and with it, "the state is a single accumulator updated in place" and "mixing one byte requires a multiplication"). The serial `h ^= b; h *= P` chain in FNV-1a is ~4–5 cycles of *latency* per byte and cannot be overlapped. Here there are eight independent add-chains (1-cycle latency, 8-way ILP), two 256-bit loads and two multiplies per 64 bytes, and one storm at the end.

Two regimes are recognised from the counting-bars (`len` is known before the first pluck): if the pile cannot reach across all eight strings (`len < 64`), the native plucks only the strings it reaches with overlapping reads and hands it straight to the storm — a constant-cost path, no loop setup. No thread parallelism: the metaphor's unit of work is one mark (one byte), far too small to justify it, and the native insists the single instrument "remembers everything it has already been told" — threading would replace his idea, not implement it.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the eight strings' resting pitches ---- */
static const uint64_t ACC_INIT[8] = {
    0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0x85EBCA77C2B2AE63ULL,
    0x27D4EB2F165667C5ULL, 0xD6E8FEB86659FD93ULL,
    0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL
};
/* ---- the first feather drawn from the inkwell ---- */
static const uint64_t FEATHER0[8] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL,
    0x94D049BB133111EBULL, 0xDA942042E4DD58B5ULL,
    0x8CB92BA72F3D8DD7ULL, 0xC4CEB9FE1A85EC53ULL,
    0xFF51AFD7ED558CCDULL, 0xB5026F5AA96619E9ULL
};
/* ---- the inkwell never runs dry: a fresh drop per lane per stripe ---- */
static const uint64_t FSTEP[8] = {
    0x6A09E667F3BCC909ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

static inline uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return (uint64_t)v; }
static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

/* the feather takes the weight, not the shape: 32x32 -> 64 of the inked word */
static inline uint64_t weigh(uint64_t k) {
    return (uint64_t)(uint32_t)k * (uint64_t)(uint32_t)(k >> 32);
}

/* one chord of the storm: 64x64 -> 128, folded to 64 */
static inline uint64_t fold128(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t al = (uint32_t)a, ah = a >> 32, bl = (uint32_t)b, bh = b >> 32;
    uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint64_t cross = (ll >> 32) + (uint32_t)lh + hl;
    uint64_t hi = hh + (lh >> 32) + (cross >> 32);
    uint64_t lo = (cross << 32) | (uint32_t)ll;
    return lo ^ hi;
#endif
}

/* THE STORM: exactly one pass, hands off until the last mark is plucked.
   (rrmxmx -- validated strong 64-bit finalizer) */
static inline uint64_t storm(uint64_t h, uint64_t len) {
    h ^= rotl64(h, 49) ^ rotl64(h, 24);
    h *= 0x9FB21C651E98DF25ULL;
    h ^= (h >> 35) + len;
    h *= 0x9FB21C651E98DF25ULL;
    h ^= h >> 28;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t h;

    /* ---- the bars hold the count: which regime is this pile? ---- */
    if (len < 64) {
        /* the pile cannot reach across all eight strings: pluck what it reaches,
           with overlapping reads, and hand it straight to the storm. */
        uint64_t a, b;
        if (len >= 16) {
            h  = (uint64_t)len * ACC_INIT[0];
            h += fold128(rd64(p)           ^ FEATHER0[0], rd64(p + 8)         ^ FEATHER0[1]);
            h += fold128(rd64(p + len - 16) ^ FEATHER0[2], rd64(p + len - 8)  ^ FEATHER0[3]);
            if (len > 32) {
                h += fold128(rd64(p + 16)       ^ FEATHER0[4], rd64(p + 24)       ^ FEATHER0[5]);
                h += fold128(rd64(p + len - 32) ^ FEATHER0[6], rd64(p + len - 24) ^ FEATHER0[7]);
            }
            return storm(h, (uint64_t)len);
        }
        if (len >= 8)      { a = rd64(p);  b = rd64(p + len - 8); }
        else if (len >= 4) { a = rd32(p);  b = rd32(p + len - 4); }
        else if (len)      { a = ((uint64_t)p[0] << 16) | ((uint64_t)p[len >> 1] << 8)
                                 | (uint64_t)p[len - 1];
                             b = a + (uint64_t)len; }
        else               { a = 0; b = 0; }
        h = (uint64_t)len * ACC_INIT[0] + fold128(a ^ FEATHER0[0], b ^ FEATHER0[1]);
        return storm(h, (uint64_t)len);
    }

    /* ---- the pile reaches all eight strings: the bridge path ---- */
    {
        const unsigned char *end = p + len;
        const size_t nstripes = len >> 6;
        uint64_t acc[8], w[8];
        size_t s;
        int i;

#if defined(__AVX2__)
        {
            __m256i av0 = _mm256_loadu_si256((const __m256i *)(const void *)(ACC_INIT));
            __m256i av1 = _mm256_loadu_si256((const __m256i *)(const void *)(ACC_INIT + 4));
            __m256i wv0 = _mm256_loadu_si256((const __m256i *)(const void *)(FEATHER0));
            __m256i wv1 = _mm256_loadu_si256((const __m256i *)(const void *)(FEATHER0 + 4));
            const __m256i tv0 = _mm256_loadu_si256((const __m256i *)(const void *)(FSTEP));
            const __m256i tv1 = _mm256_loadu_si256((const __m256i *)(const void *)(FSTEP + 4));

            for (s = 0; s < nstripes; s++) {
                const unsigned char *q = p + (s << 6);
                __m256i d0 = _mm256_loadu_si256((const __m256i *)(const void *)q);
                __m256i d1 = _mm256_loadu_si256((const __m256i *)(const void *)(q + 32));
                __m256i k0 = _mm256_xor_si256(d0, wv0);     /* dip in the inkwell */
                __m256i k1 = _mm256_xor_si256(d1, wv1);
                wv0 = _mm256_add_epi64(wv0, tv0);           /* a fresh feather, always */
                wv1 = _mm256_add_epi64(wv1, tv1);
                /* the pluck: one constant force, never varying */
                av0 = _mm256_add_epi64(av0,
                      _mm256_add_epi64(_mm256_mul_epu32(k0, _mm256_srli_epi64(k0, 32)),
                                       _mm256_shuffle_epi32(d0, 0x4E))); /* the bridge */
                av1 = _mm256_add_epi64(av1,
                      _mm256_add_epi64(_mm256_mul_epu32(k1, _mm256_srli_epi64(k1, 32)),
                                       _mm256_shuffle_epi32(d1, 0x4E)));
            }
            _mm256_storeu_si256((__m256i *)(void *)acc,       av0);
            _mm256_storeu_si256((__m256i *)(void *)(acc + 4), av1);
            _mm256_storeu_si256((__m256i *)(void *)w,         wv0);
            _mm256_storeu_si256((__m256i *)(void *)(w + 4),   wv1);
        }
#else
        for (i = 0; i < 8; i++) { acc[i] = ACC_INIT[i]; w[i] = FEATHER0[i]; }
        for (s = 0; s < nstripes; s++) {
            const unsigned char *q = p + (s << 6);
            for (i = 0; i < 8; i++) {
                uint64_t dv = rd64(q + 8 * i);
                uint64_t kv = dv ^ w[i];
                w[i]      += FSTEP[i];
                acc[i ^ 1] += dv;          /* the bridge: the whole instrument answers */
                acc[i]     += weigh(kv);
            }
        }
#endif
        /* the last, short handful of marks: one more stripe, overlapping,
           on a feather no earlier mark has touched */
        if (len & 63) {
            const unsigned char *q = end - 64;
            for (i = 0; i < 8; i++) {
                uint64_t dv = rd64(q + 8 * i);
                uint64_t kv = dv ^ w[i];
                acc[i ^ 1] += dv;
                acc[i]     += weigh(kv);
            }
        }

        /* the storm reads the final shivering pattern, once, and writes eight notes */
        h  = (uint64_t)len * ACC_INIT[0];
        h += fold128(acc[0] ^ FEATHER0[0], acc[1] ^ FEATHER0[1]);
        h += fold128(acc[2] ^ FEATHER0[2], acc[3] ^ FEATHER0[3]);
        h += fold128(acc[4] ^ FEATHER0[4], acc[5] ^ FEATHER0[5]);
        h += fold128(acc[6] ^ FEATHER0[6], acc[7] ^ FEATHER0[7]);
        return storm(h, (uint64_t)len);
    }
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 14.0**

Stated before any measurement, with the reasoning so it can be judged wrong honestly:

- FNV-1a baseline is **latency**-bound, not throughput-bound: `xor` (1) + 64-bit `imul` (3, sometimes 4–5) in a single serial chain ⇒ ~4–5 cycles **per byte** ⇒ ≈0.2–0.25 B/cycle ≈ 0.8–1.1 GB/s on a ~3.5–4 GHz core, independent of buffer size.
- The bridge path issues, per 64 bytes: 2 loads, 2 xor, 2 add (feathers), 2 shift, 2 `vpmuludq`, 2 shuffle, 4 add — ~16 µops across 8 independent 1-cycle add chains. Port-limited at roughly 4–6 cycles per 64 bytes ⇒ 10–16 B/cycle, so for anything resident in L1/L2 the expected ratio over FNV is **40–60×**, and for buffers larger than L2 it collapses to DRAM bandwidth (~10–25 GB/s) giving **12–25×**.
- The short path (`len < 64`) is 2–4 multiply-folds plus one storm, ~20–30 cycles total, versus FNV's ~4.5·len cycles: ≈1.5× at 8 bytes, ≈4× at 32 bytes, ≈9× at 63 bytes. Never slower — that's the guard against the one regime where a wide-state hash normally loses.
- 14.0 is my estimate of the aggregate the harness will report if it mixes sizes with meaningful weight on small and DRAM-sized buffers. If it reports a single large-buffer number I expect 15–25; if it is dominated by ≤32-byte inputs I expect 2–5, and I would be wrong.

Avalanche prediction: **≈0.50 flip ratio, bias under ~1%**. Every lane reaches the output only through a 64×64→128 fold and then `rrmxmx`, both of which are bit-avalanche-complete; a single input bit flip perturbs `acc[i]` through `weigh()` and `acc[i^1]` additively, and either perturbation is amplified by the fold. I do *not* expect the deleted per-block scramble to cost avalanche, because the feather stream never repeats — that is the specific claim this experiment tests.

## MEASUREMENT

**Not performed — and I will not dress that up.** This session was invoked with no tools: `hash_bench`, `hash_contract`, and the symbolic/unconventional tools are all unavailable, so I ran zero measurements and made zero of the four permitted improvement iterations. The numbers above are predictions from instruction-count and latency reasoning only. The pipeline will compile and measure; the honest status of this artifact is *unmeasured*.

What would falsify it, concretely:
1. `speedup_vs_dp < 2` on large buffers ⇒ the bridge loop is not running vectorised (check `__AVX2__` actually defined under `-march=native`) or the benchmark is memcpy-bandwidth-bound for both kernels.
2. Avalanche score materially below the baseline ⇒ the deleted periodic scramble *did* matter, and the Weyl-feather argument is wrong; the fix is to re-insert one scramble per 1024 bytes (cost ~1%), which is me conceding this particular break of the assumption.
3. `speedup_vs_dp < 1` at 8–32 bytes ⇒ the short-path fold is mispredicting or `fold128` is not compiling to `mulx`.

## VERDICT

The native's description translates without any slack: eight strings = eight 64-bit accumulators; one bridge = the `acc[i^1] += d[i]` swapped-lane term; the never-dry inkwell = per-lane Weyl counters; one storm pass = one fold-plus-avalanche finalizer; the pickers = register-only scratch with no table and no scratch buffer. Nothing in the metaphor had to be dropped or reinterpreted, and nothing was silently replaced by the textbook method.

The mechanism's own honest claim is narrower than it sounds. It does not make hashing cleverer than xxHash — it *lands on* xxHash's own insight, which is the right outcome per Step 4: a validated design beats a novel one. Its one genuine departure is deleting the periodic accumulator scramble on the grounds that the feather stream never repeats. That is a real, testable, falsifiable claim, and it is the whole of the "fewer rounds" bet.

Risk I named, and how it is addressed rather than left hanging:
- *"Wide state costs more than it saves on short inputs"* — guarded by the `len < 64` regime check with a constant-cost overlapping-read path, so the risky part never executes in the regime where it would lose. Not a hope, a branch.
- *"Thread parallelism might help on huge buffers"* — dropped, not shipped-and-hoped. The metaphor's unit of work is one byte, the loop is already far past DRAM bandwidth at one core, and the native explicitly carries one unbroken tremor. Adding threads would be me overruling him for no measured gain.
- *"The deleted scramble might cost quality"* — not guardable by a size check; it is the hypothesis. If measurement 2 above fails, the correct response is to put the scramble back and report that the native was wrong on this point, not to retune constants until the score moves.

Where I expect this to be weaker than the known way: a 512-bit state and a two-regime dispatch is a lot more code than six lines of FNV-1a, and FNV-1a is streaming-friendly while this is not (the tail needs an overlapping read of the last 64 bytes, so it cannot hash an unbounded stream without buffering). If the real requirement were incremental streaming rather than one-shot buffer hashing, the native's "wait, hands off, until the last mark" is a genuine architectural cost, not a win.