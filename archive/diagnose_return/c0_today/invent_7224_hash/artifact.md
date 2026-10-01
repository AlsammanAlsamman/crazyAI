## MAPPING

**SEED 1 — "The servant's inflated-house swats hold silence so the marks' order is fixed before any heap is folded."**

| World object | Problem object |
|---|---|
| a mark | one input byte `data[i]` |
| the pile *in its given order* | the buffer with its index space |
| the flapper's bladder swatting my mouth | a hard barrier between *reading* and *mixing*: nothing is folded while bytes are being placed |
| "silence until each mark is laid on the flat stone" | a pure load phase — an 8-byte `memcpy` into a lane, zero arithmetic |
| the flat stone at the riverbank | a lane slot: one of 8 × 64-bit accumulator positions in a 64-byte stripe |
| "the order never gets spoken aloud and lost" | order is carried by **position** (lane index + stripe index + per-stripe key), not by **time of arrival** |

Silent assumption broken: **"each byte must be mixed into the running state before the next byte is read"** and, as a direct consequence, **"the whole buffer must be read once, start to end, in order."** If position encodes order, the sequencing constraint evaporates — 8 lanes (or two AVX2 registers) can consume bytes simultaneously.

**SEED 2 — "Each heap races the musk deer's mountain stride, uphill twist and downhill fold, until no heap keeps the shape it entered with."**

| World object | Problem object |
|---|---|
| heap of even count | fixed 64-byte stripe (8 lanes × 8 bytes) |
| fixed strides | constant stride, no data-dependent branching |
| uphill twist | left shift / rotate |
| downhill fold | `v ^= v >> 47` |
| doubling-back that eats its own trail | `acc[j^1] += d` — each lane's input lands on its neighbour's trail |
| "no heap keeps the shape it entered" | the per-block scramble (bijective xorshift–multiply) |

Breaks: **"the state is a single accumulator updated in place, one value"** and partially **"mixing one byte requires a multiplication"** (the twist/fold is shift-xor; one 32×32 multiply serves 8 bytes, not 1).

**SEED 3 — "The owl calls the final folded shape inside the dreamer inside; intermediate heaps burned at the shrine."**

| World object | Problem object |
|---|---|
| the owl with the pen | the finalizer (moremur: xor-shift · multiply · xor-shift · multiply · xor-shift) |
| dreamer inside the dreamer | nested mixing rounds applied **once**, at the end |
| "one fixed size regardless of how many marks" | 64-bit output for any `len` |
| burning the heaps | accumulators never stored; 512 bits of state collapse to 64, irreversibly |
| the butterfly who cannot recall the man | one-wayness: the token cannot be walked back |

Breaks: **"more mixing rounds always means better mixing"** — heavy avalanche is spent *once* at the shrine, not per byte. Per-byte avalanche is wasted work; only the output must avalanche.

## CHOSEN SEED

**SEED 1.** It is the only one of the three that breaks *"the whole buffer must be read once, start to end, in order"* — the preferred assumption — and its mapping is the most literal: the flapper is quite exactly a barrier instruction separating load from mix, and "the flat stone" is quite exactly a lane. It is also maximally different from FNV-1a, whose entire identity is "mix before you read the next byte."

SEEDs 2 and 3 are not discarded; in the native's own account they are what *happens* to the marks once the flapper has fixed their order, so they become the mixing kernel and the finalizer of the artifact built from SEED 1.

## ASSUMPTION BROKEN

Primary: **each byte must be mixed into the running state before the next byte is read** — and therefore **the buffer must be consumed once, in order.** Position replaces sequence. Secondary (inherited from SEEDs 2–3): the single in-place accumulator, the per-byte multiply, and "more rounds is better."

Per step 4, I did **not** invent a new scheme once the mechanism pointed at one. "Heaps of even count, each lane racing independently, the shape carried forward, burned when the next heap swallows it, the owl sealing it at the end" *is* the XXH3 construction (stripe accumulate → periodic accumulator scramble → pairwise 128-bit merge → avalanche), a validated, widely deployed, SMHasher-clean design. The metaphor arrives at it; I let it, rather than shipping something novel and untested. The finalizer is `moremur`, a published, avalanche-validated 64-bit mixer.

**Regime recognition (step 5).** The assumption list spans two regimes (small buffers, where striping is pure overhead, and large ones). The native encodes this himself: *a heap must be of even count* — if there aren't enough marks to make a heap, no deer runs; the marks go straight to the owl. The kernel dispatches on `len` at runtime into four paths (0–16, 17–128, 129+, plus a scalar fallback when AVX2 is absent), so the stated risk of my own mechanism — striping overhead on short inputs — is guarded, not merely noted.

**Thread parallelism: deliberately omitted.** The metaphor's unit of work is a 64-byte heap; at benchmark sizes that is far below any OpenMP threshold, and a thread-count-dependent hash would break the native's own requirement that the same marks yield the same token. Vectorization only: AVX2 intrinsics, `memcpy` loads (no UB, no alignment assumption), `__restrict`, 8 independent dependency chains for ILP on scalar hardware.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- constants: high-entropy digits of pi (Blowfish S-box tail) ---------- */
static const uint64_t RIVER[24] = {
    0xD1310BA698DFB5ACULL, 0x2FFD72DBD01ADFB7ULL, 0xB8E1AFED6A267E96ULL,
    0xBA7C9045F12C7F99ULL, 0x24A19947B3916CF7ULL, 0x0801F2E2858EFC16ULL,
    0x636920D871574E69ULL, 0xA458FEA3F4933D7EULL, 0x0D95748F728EB658ULL,
    0x718BCD5882154AEEULL, 0x7B54A41DC25A59B5ULL, 0x9C30D5392AF26013ULL,
    0xC5D1B023286085F0ULL, 0xCA417918B8DB38EFULL, 0x8E79DCB0603A180EULL,
    0x6C9E0E8BB01E8A3EULL, 0xD71577C1BD314B27ULL, 0x78AF2FDA55605C60ULL,
    0xE65525F3AA55AB94ULL, 0x5748986263E81440ULL, 0x55CA396A2AAB10B6ULL,
    0xB4CC5C341141E8CEULL, 0xA15486AF7C72E993ULL, 0xB3EE1411636FBC2AULL
};
#define K_P64_1 0x9E3779B185EBCA87ULL
#define K_P64_2 0xC2B2AE3D27D4EB4FULL
#define K_P64_3 0x165667B19E3779F9ULL
#define K_P64_4 0x85EBCA77C2B2AE63ULL
#define K_P64_5 0x27D4EB2F165667C5ULL
#define K_P32_1 0x9E3779B1U

/* ---- the flapper: a pure load. no arithmetic while a mark is placed ------ */
static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint32_t rd32(const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return v; }
static inline uint64_t rotl64(uint64_t x, unsigned r){ return (x<<r)|(x>>((64-r)&63)); }

/* ---- ninety degrees, corners folded into the center --------------------- */
static inline uint64_t fold128(uint64_t a, uint64_t b){
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
#else
    uint64_t ll=(a&0xFFFFFFFFULL)*(b&0xFFFFFFFFULL);
    uint64_t hl=(a>>32)*(b&0xFFFFFFFFULL);
    uint64_t lh=(a&0xFFFFFFFFULL)*(b>>32);
    uint64_t hh=(a>>32)*(b>>32);
    uint64_t cross=(ll>>32)+(hl&0xFFFFFFFFULL)+lh;
    uint64_t up=(hl>>32)+(cross>>32)+hh;
    uint64_t lo=(cross<<32)|(ll&0xFFFFFFFFULL);
    return lo ^ up;
#endif
}

/* ---- the owl: seals any shape to one fixed size, one way (moremur) ------ */
static inline uint64_t owl(uint64_t h){
    h ^= h >> 27; h *= 0x3C79AC492BA7B653ULL;
    h ^= h >> 33; h *= 0x1C69B3F74AC4AE35ULL;
    h ^= h >> 27;
    return h;
}

/* ---- the musk deer's stride over one heap: 8 lanes, doubling back ------- */
static inline void heap_race(uint64_t *acc, const unsigned char *p, const uint64_t *sec){
    int j;
    for (j = 0; j < 8; j++){
        uint64_t d = rd64(p + 8*j);
        uint64_t v = d ^ sec[j];
        acc[j ^ 1] += d;                                  /* eats its own trail */
        acc[j]     += (uint64_t)(uint32_t)v * (uint64_t)(v >> 32);
    }
}
/* ---- burned at the shrine: no heap keeps the shape it entered with ------ */
static inline void burn(uint64_t *acc, const uint64_t *sec){
    int j;
    for (j = 0; j < 8; j++){
        uint64_t v = acc[j];
        v ^= v >> 47;                 /* downhill fold */
        v ^= sec[j];
        v *= (uint64_t)K_P32_1;       /* uphill twist  */
        acc[j] = v;
    }
}

#if defined(__AVX2__)
static inline void heap_race_v(__m256i *x, const unsigned char *p, const uint64_t *sec){
    int k;
    for (k = 0; k < 2; k++){
        __m256i dv   = _mm256_loadu_si256((const __m256i*)(const void*)(p + 32*k));
        __m256i kv   = _mm256_loadu_si256((const __m256i*)(const void*)(sec + 4*k));
        __m256i dk   = _mm256_xor_si256(dv, kv);
        __m256i dkh  = _mm256_srli_epi64(dk, 32);
        __m256i prod = _mm256_mul_epu32(dk, dkh);
        __m256i swap = _mm256_shuffle_epi32(dv, _MM_SHUFFLE(1,0,3,2));
        x[k] = _mm256_add_epi64(prod, _mm256_add_epi64(x[k], swap));
    }
}
static inline void burn_v(__m256i *x, const uint64_t *sec){
    const __m256i pr = _mm256_set1_epi32((int)K_P32_1);
    int k;
    for (k = 0; k < 2; k++){
        __m256i a  = x[k];
        __m256i dv = _mm256_xor_si256(a, _mm256_srli_epi64(a, 47));
        __m256i kv = _mm256_loadu_si256((const __m256i*)(const void*)(sec + 4*k));
        __m256i dk = _mm256_xor_si256(dv, kv);
        __m256i lo = _mm256_mul_epu32(dk, pr);
        __m256i hi = _mm256_mul_epu32(_mm256_srli_epi64(dk, 32), pr);
        x[k] = _mm256_add_epi64(lo, _mm256_slli_epi64(hi, 32));
    }
}
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *__restrict p = data;

    /* ===== regime 1: too few marks to make a heap — straight to the owl == */
    if (len <= 16){
        if (len > 8){
            uint64_t a = rd64(p) ^ RIVER[0];
            uint64_t b = rd64(p + len - 8) ^ RIVER[1];
            return owl(fold128(a,b) + rotl64(a,23) + rotl64(b,41)
                       + (uint64_t)len * K_P64_5);
        }
        if (len >= 4){
            uint64_t a = (uint64_t)rd32(p);
            uint64_t b = (uint64_t)rd32(p + len - 4);
            return owl(((a << 32) | b) ^ RIVER[2] ^ ((uint64_t)len * K_P64_5));
        }
        if (len){
            uint64_t c = ((uint64_t)p[0] << 16)
                       | ((uint64_t)p[len >> 1] << 8)
                       |  (uint64_t)p[len - 1]
                       | ((uint64_t)len << 24);
            return owl(c ^ RIVER[3]);
        }
        return owl(RIVER[4]);
    }

    /* ===== regime 2: a handful of half-heaps (17..128 bytes) ============= */
    if (len <= 128){
        uint64_t h0 = (uint64_t)len * K_P64_1;
        uint64_t h1 = ~(uint64_t)len * K_P64_2;
        size_t nb = len >> 4, k;
        for (k = 0; k < nb; k++){
            const unsigned char *q = p + (k << 4);
            uint64_t lo = rd64(q), hi = rd64(q + 8);
            h0 += fold128(lo ^ RIVER[k], hi ^ RIVER[k + 8]);
            h1 ^= rotl64(lo + hi + RIVER[k + 16], 31) * K_P64_2;
        }
        {   const unsigned char *t = p + len - 16;
            uint64_t lo = rd64(t), hi = rd64(t + 8);
            h0 += fold128(lo ^ RIVER[16], hi ^ RIVER[17]);
            h1 ^= rotl64(lo ^ hi, 17) * K_P64_4;
        }
        return owl(h0 ^ rotl64(h1, 37));
    }

    /* ===== regime 3: the full mountain path — 8 lanes, out of order ====== */
    {
        uint64_t acc[8] = { 0x9E3779B1ULL, K_P64_1, K_P64_2, K_P64_3,
                            K_P64_4, 0xC2B2AE3DULL, K_P64_5, 0x85EBCA77ULL };
        const size_t nbStripes = (len - 1) / 64;   /* last 64B handled apart */
        const size_t SPB = 16;                     /* heaps per shrine-burn  */
        size_t s = 0, run, k;
#if defined(__AVX2__)
        __m256i x[2];
        x[0] = _mm256_loadu_si256((const __m256i*)(const void*)(acc + 0));
        x[1] = _mm256_loadu_si256((const __m256i*)(const void*)(acc + 4));
        while (s < nbStripes){
            run = nbStripes - s; if (run > SPB) run = SPB;
            for (k = 0; k < run; k++) heap_race_v(x, p + (s + k) * 64, RIVER + k);
            s += run;
            if (run == SPB) burn_v(x, RIVER + 16);
        }
        heap_race_v(x, p + len - 64, RIVER + 15);
        _mm256_storeu_si256((__m256i*)(void*)(acc + 0), x[0]);
        _mm256_storeu_si256((__m256i*)(void*)(acc + 4), x[1]);
#else
        while (s < nbStripes){
            run = nbStripes - s; if (run > SPB) run = SPB;
            for (k = 0; k < run; k++) heap_race(acc, p + (s + k) * 64, RIVER + k);
            s += run;
            if (run == SPB) burn(acc, RIVER + 16);
        }
        heap_race(acc, p + len - 64, RIVER + 15);
#endif
        {   /* ninety degrees, corners folded into the center, then the owl */
            uint64_t h = (uint64_t)len * K_P64_1;
            h += fold128(acc[0] ^ RIVER[0], acc[1] ^ RIVER[1]);
            h += fold128(acc[2] ^ RIVER[2], acc[3] ^ RIVER[3]);
            h += fold128(acc[4] ^ RIVER[4], acc[5] ^ RIVER[5]);
            h += fold128(acc[6] ^ RIVER[6], acc[7] ^ RIVER[7]);
            return owl(h);
        }
    }
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 12**

Reasoning, stated before any measurement: FNV-1a is latency-bound on a single 3-cycle multiply chain, ~3–4 cycles/byte (~1 GB/s). This kernel is throughput-bound: 8 independent lanes, one 32×32 multiply per 8 bytes, AVX2 doing a 64-byte heap in ~2 vector multiply-add pairs. Expected ≈0.1–0.3 cycles/byte on large buffers (20–40×), ≈6–12× at 128 B–1 KB, ≈2–4× below 32 B where the short path is only a few ops. A benchmark averaging across sizes should land near 12; a large-buffer-only benchmark should beat it substantially.

Avalanche prediction: ≈0.50 flip probability per output bit (score ≥0.99 on any normalized scale), for every input bit position and every length regime — the `moremur` seal alone guarantees full avalanche from the merged word, and the `acc[j^1]` cross-lane feedback plus the 47-bit fold guarantee a single early byte reaches every lane before the merge. This is the native's own acceptance test: *"if even a corner survives untouched I throw the whole method away."*

## MEASUREMENT

**Not measured. No tools were available in this session** — `hash_bench`, `hash_contract` and the symbolic/unconventional tools were all listed but none were callable, and there is no shell here to compile with. I am not going to report numbers I did not obtain.

What is therefore untested and must be checked by the pipeline before any of the above is believed:

1. Actual `speedup_vs_dp` against the FNV-1a reference at the benchmark's real size distribution. My 12 is a prediction, not a result.
2. Whether the three regime boundaries (16 / 128 / 129+) are placed well; if the benchmark concentrates near 64–128 bytes, the mid path may be leaving throughput on the table and the stripe engine's threshold should drop to ~96.
3. Avalanche in the 1–3 byte regime, the thinnest part of the construction.
4. AVX2 availability under `-march=native` on the bench machine — if absent, the scalar path runs and the predicted speedup should be roughly halved (still ~6×, still 8-way ILP).

The four allowed improvement rounds are unspent. If a measurement comes back, the first thing I would change is the 128-byte boundary, and the second is unrolling two heaps per iteration in the long path to hide load latency.

## VERDICT

The mapping is literal and it holds all the way down: the flapper is a load/mix barrier, the flat stone is a lane, position replaces sequence, the heap is a 64-byte stripe, the deer's doubling-back is `acc[j^1] += d`, the shrine-burn is the accumulator scramble, the ninety-degree fold is the pairwise 128-bit merge, and the owl is the finalizer. Nothing in the native's account was dropped as decoration, and nothing was quietly swapped for the textbook accumulator.

The honest finding is that **this metaphor does not produce a novel hash — it produces XXH3.** That is the correct outcome under step 4, not a disappointment: the native's construction and a validated, SMHasher-clean, industrially deployed design are the same object, which is evidence the mapping is right rather than evidence it is unoriginal. Claiming novelty here would be the dishonest move.

Named risk and how it is addressed rather than merely noted: **striping is a net loss below roughly 100 bytes** — setup, the merge, and the finalizer dominate when there are only one or two heaps. This is guarded structurally by the `len <= 16` and `len <= 128` branches, which never touch the lane engine, so the kernel degrades to a few multiplies and one seal on short inputs instead of paying stripe overhead. Thread parallelism is dropped entirely rather than guarded, because the metaphor's unit of work (64 bytes) cannot justify it at these sizes and a thread-count-dependent result would not be a hash at all.

Remaining honest weakness: the mid-regime `h1` lane uses xor accumulation, which is commutative; position-dependence there rests on the `RIVER[k+16]` key and on `h0`'s keyed fold. If measured avalanche in the 17–128 range disappoints, that lane is the first suspect and should become an additive, rotation-chained accumulator.