# MAPPING

## SEED 1 — the chained prisoners sharpening the blank pillar

| World object | Problem object |
|---|---|
| a mark in the pile | one input byte (`data[i]`) |
| the pile's "given order" | buffer order, index 0 → `len-1` |
| the blank pillar | the mixing constant/secret strip, never reset by data |
| a prisoner who *does not look at the mark* | a stage that never stores the byte, only the derived value |
| "feels how it changes the angle" | a cheap shift/xor delta, **no multiply by the byte** |
| "passes that angle, not the mark" | the stage forwards state, not data — a pipeline |
| "a mark dropped at the very start still trembles in the last hand" | long-range dependency: early bytes reach the final state |
| "the pillar keeps sharpening onward as if nothing passed" | secret is read-only, advanced by position, never data-dependent |

**Assumption broken:** *"mixing one byte requires a multiplication"* — the prisoner transforms an angle (rotate/xor/add), and the byte is consumed without its own multiply.

## SEED 2 — the flood bending anchored wires

| World object | Problem object |
|---|---|
| the river rising on each pass | the accumulator set, monotonically absorbing |
| "does not recede until every mark has gone through" | no output is readable mid-stream — absorb phase is total |
| "a token pulled while the water is still up is worthless" | reading the accumulators without finalization = bad avalanche |
| the tiny **wire shapes**, **anchored** in the riverbed | a fixed-size array of accumulator lanes `acc[8]`, each seeded with a *different* anchor constant |
| "they do not move themselves; the flood moves them" | lanes are updated only by incoming data, never by each other's logic |
| "bending each wire by exactly the angle ground on that pass" | same update rule applied to all lanes; distinct anchors ⇒ distinct states |
| "past where any traveler has ever fused its source" | irreversibility; also: *wires are below the waterline* → all 8 lanes are in flight at once |

**Assumption broken:** *"the state is a single accumulator updated in place, one value"* and, with it, *"each byte must be mixed into the running state before the next byte is read"* — a pass bends 8 wires from 64 bytes simultaneously, so 8 independent dependency chains run concurrently.

## SEED 3 — the silhouette against the sun, read once

| World object | Problem object |
|---|---|
| the flood draining | end of the absorb loop |
| "the shadows stop their chitter / the walls still settling" | the one scrambling/settling step at block boundaries; nothing is read while unsettled |
| climbing to the pit's rim | leaving the hot loop; the finalization |
| the **silhouette** the wires throw | a single projection of all 8 lanes into 64 bits |
| "which lean, which stand straight, **which cross another**" | the fold is *pairwise*: lanes crossed two at a time by a 128-bit multiply folded to 64 bits |
| "small and fixed" | output is 64 bits regardless of `len`; `len` is folded in |
| "drawn only **once**" | exactly **one** strong avalanche finalizer, no matter how many bytes |
| "I keep nothing but that shadow-shape" | no per-byte state retained; the cheap per-byte work is allowed to be weak |
| "a thin pile never raises the water over the wires" | runtime regime test: short buffers never reach the lanes |

**Assumption broken:** *"more mixing rounds always means better mixing."* The native does **fewer** mixing rounds — the per-byte work is deliberately too weak to avalanche anything, and *all* avalanche is bought in one final silhouette. FNV-1a pays a full strong round per byte and still avalanches *worse* on its low bits.

# CHOSEN SEED

**SEED 3.** It is the one seed that breaks the preferred assumption ("more mixing rounds always means better mixing"), and its mapping is the most literal: *anchored wires → lanes*, *crossing wires → pairwise 128-bit fold*, *silhouette read once → single finalizer*, *shallow pit → size-regime branch*. SEED 2 is its absorb half and I use it as written (the wires are named in SEED 3 too); SEED 1 supplies the multiply-free-per-byte prisoner.

# ASSUMPTION BROKEN

**"More mixing rounds always means better mixing."** Replaced by: *one* strong round (the silhouette), applied once at the end, over a wide weakly-filled state.

Per step 4, I let this mechanism land on the **validated** technique rather than inventing one: "cheap wide multi-lane absorb + cross-lane swap-add + one pairwise-fold-and-avalanche finalizer, with a block settling step" **is** the XXH3/NH-family construction (SMHasher-validated). The wires are its 8 accumulator lanes, "which cross another" is its `mul128_fold64` merge, "the shadows stop chittering" is its accumulator scramble. I wrote it from the metaphor with my own pillar constants rather than inventing a novel untested mixer.

**Regime handling (step 5).** The known-way section names two regimes implicitly (whole-buffer sequential vs. the fact that per-byte cost dominates only when there are many bytes). The native's own test is the water level: *"the flood rises on each pass"* — a thin pile never floods the pit, and a token pulled from unflooded wires is worthless. So `len < 64` takes the **shallow-pit path**: the few marks are crossed directly and read, lanes never touched. This is also the guard demanded by my own VERDICT risk (lane setup + merge is pure overhead on tiny inputs).

**No thread parallelism.** The metaphor has *one* river and *one* flood; and 64-byte stripes are vector-sized work, not thread-sized. Vectorization only (AVX2 intrinsics + `restrict` + aligned pillar), with a scalar fallback — per the step-4 default.

# ARTIFACT

```c
/* "The Silhouette" — anchored wires bent by one flood, read once against the sun.
 *
 * SEED 3: per-byte work is deliberately weak (no avalanche attempted); the
 * entire avalanche is bought in a single final projection of the wires.
 * Regimes: len < 64  -> shallow pit, the few marks are crossed directly.
 *          len >= 64 -> the flood; 8 anchored wires, one silhouette at the end.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  include <immintrin.h>
#endif

#if defined(__GNUC__)
#  define ALIGN64 __attribute__((aligned(64)))
#else
#  define ALIGN64
#endif

#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

/* ---------------- the blank pillar: 256 bytes of angle, read-only ---------- */
static const uint64_t PILLAR[32] ALIGN64 = {
    0xbe4ba423396cfeb8ULL, 0x1cad21f72c81017cULL, 0xdb979083e96dd4deULL,
    0x1f67b3b7a4a44072ULL, 0x78e5c0cc4ee679cbULL, 0x2172ffcc7dd05a82ULL,
    0x8e2443f7744c2607ULL, 0x4d7fab50d2e6d5a6ULL, 0x8af3c5f6e8c3b7a1ULL,
    0x3bd9e6c1a5f20d47ULL, 0xc7e4f81d6b3a9052ULL, 0x59a0d7b2e4c86f13ULL,
    0xf2615c8d90ab37e4ULL, 0x6d38b4e7c1052fa9ULL, 0xa94c2f6038d7e1b5ULL,
    0x074e9ab3fd5c6281ULL, 0xe5b17d40c2983f6aULL, 0x2c80fa5e91b4d3c7ULL,
    0xb6439e2af7501c8dULL, 0x81fd5c7b34ea9026ULL, 0x4a2e8b069dc5f731ULL,
    0xd3706af1584be2c9ULL, 0x9fc8325d6e170ab4ULL, 0x50ba13ce8f24d976ULL,
    0x3e91c7a2b05d4f68ULL, 0xc5278de4139a6b02ULL, 0x7b04f9a6ce83152dULL,
    0xa6ed40835c9b27f1ULL, 0x1203e7bd9a46c85fULL, 0xef8a5416b7d20c93ULL,
    0x6459bc02da71e38aULL, 0x97d6e1f4085a3b2eULL
};

/* ---------------- the anchors: where each wire is fixed in the riverbed ---- */
#define W0 0x9E3779B185EBCA87ULL
#define W1 0xC2B2AE3D27D4EB4FULL
#define W2 0x165667B19E3779F9ULL
#define W3 0x85EBCA77C2B2AE63ULL
#define W4 0x27D4EB2F165667C5ULL
#define W5 0x9E3779B97F4A7C15ULL
#define W6 0xFF51AFD7ED558CCDULL
#define W7 0xC4CEB9FE1A85EC53ULL

static inline uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* two wires crossing: full 128-bit product folded back to one silhouette word */
static inline uint64_t cross(uint64_t a, uint64_t b)
{
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)p ^ (uint64_t)(p >> 64);
#else
    uint64_t al = a & 0xFFFFFFFFULL, ah = a >> 32;
    uint64_t bl = b & 0xFFFFFFFFULL, bh = b >> 32;
    uint64_t ll = al * bl, hl = ah * bl, lh = al * bh, hh = ah * bh;
    uint64_t mid = (ll >> 32) + (hl & 0xFFFFFFFFULL) + lh;
    uint64_t up  = (hl >> 32) + (mid >> 32) + hh;
    uint64_t lw  = (mid << 32) | (ll & 0xFFFFFFFFULL);
    return lw ^ up;
#endif
}

/* the shadow-shape: the one and only strong round, drawn once */
static inline uint64_t shadow(uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 29;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 32;
    return h;
}

/* ---------------- scalar flood: one pass bends all eight wires ------------- */
static inline void absorb_s(uint64_t *acc, const unsigned char *in, const uint64_t *sec)
{
    uint64_t w[8];
    int i;
    memcpy(w, in, 64);
    for (i = 0; i < 8; i++) {
        uint64_t v = w[i] ^ sec[i];
        /* no multiply by the byte itself (SEED 1); one 32x32 angle + a cross-add */
        acc[i] += w[i ^ 1] + (uint64_t)(uint32_t)v * (uint64_t)(uint32_t)(v >> 32);
    }
}

/* the walls settling: once per 1024 bytes, not once per byte */
static inline void settle_s(uint64_t *acc, const uint64_t *sec)
{
    int i;
    for (i = 0; i < 8; i++) {
        acc[i] ^= acc[i] >> 47;
        acc[i] ^= sec[i];
        acc[i] *= 0x9E3779B1ULL;
    }
}

#if defined(__AVX2__)
#define ABSORB_V(xa, in, sec)                                                  \
    do {                                                                       \
        __m256i d_ = _mm256_loadu_si256((const __m256i *)(const void *)(in));  \
        __m256i k_ = _mm256_loadu_si256((const __m256i *)(const void *)(sec)); \
        __m256i x_ = _mm256_xor_si256(d_, k_);                                 \
        __m256i p_ = _mm256_mul_epu32(x_, _mm256_srli_epi64(x_, 32));          \
        __m256i s_ = _mm256_shuffle_epi32(d_, _MM_SHUFFLE(1, 0, 3, 2));        \
        (xa) = _mm256_add_epi64((xa), _mm256_add_epi64(p_, s_));               \
    } while (0)

#define SETTLE_V(xa, sec)                                                      \
    do {                                                                       \
        __m256i k_ = _mm256_loadu_si256((const __m256i *)(const void *)(sec)); \
        __m256i t_ = _mm256_xor_si256((xa), _mm256_srli_epi64((xa), 47));      \
        __m256i c_ = _mm256_set1_epi32((int)0x9E3779B1);                       \
        t_ = _mm256_xor_si256(t_, k_);                                         \
        (xa) = _mm256_add_epi64(                                               \
            _mm256_mul_epu32(t_, c_),                                          \
            _mm256_slli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(t_, 32), c_), 32)); \
    } while (0)
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict d = data;
    uint64_t h;

    /* ================= REGIME A: deep pit — the flood and the wires ======== */
    if (len >= 64) {
        uint64_t acc[8];
        size_t nstripes = (len - 1) / 64;      /* full passes; last pass overlaps */
        size_t nblocks  = nstripes >> 4;
        size_t rem      = nstripes - (nblocks << 4);
        const unsigned char *p = d;
        size_t b, s;

#if defined(__AVX2__)
        __m256i x0 = _mm256_setr_epi64x((long long)W0, (long long)W1,
                                        (long long)W2, (long long)W3);
        __m256i x1 = _mm256_setr_epi64x((long long)W4, (long long)W5,
                                        (long long)W6, (long long)W7);
        for (b = 0; b < nblocks; b++) {
            for (s = 0; s < 16; s++) {
                ABSORB_V(x0, p + s * 64,      PILLAR + s);
                ABSORB_V(x1, p + s * 64 + 32, PILLAR + s + 4);
            }
            p += 1024;
            SETTLE_V(x0, PILLAR + 24);
            SETTLE_V(x1, PILLAR + 28);
        }
        for (s = 0; s < rem; s++) {
            ABSORB_V(x0, p + s * 64,      PILLAR + s);
            ABSORB_V(x1, p + s * 64 + 32, PILLAR + s + 4);
        }
        ABSORB_V(x0, d + len - 64,      PILLAR + 23);
        ABSORB_V(x1, d + len - 64 + 32, PILLAR + 27);
        _mm256_storeu_si256((__m256i *)(void *)acc,       x0);
        _mm256_storeu_si256((__m256i *)(void *)(acc + 4), x1);
#else
        acc[0] = W0; acc[1] = W1; acc[2] = W2; acc[3] = W3;
        acc[4] = W4; acc[5] = W5; acc[6] = W6; acc[7] = W7;
        for (b = 0; b < nblocks; b++) {
            for (s = 0; s < 16; s++) absorb_s(acc, p + s * 64, PILLAR + s);
            p += 1024;
            settle_s(acc, PILLAR + 24);
        }
        for (s = 0; s < rem; s++) absorb_s(acc, p + s * 64, PILLAR + s);
        absorb_s(acc, d + len - 64, PILLAR + 23);
#endif
        /* the silhouette: which lean, which stand straight, which cross another */
        h  = (uint64_t)len * W0;
        h += cross(acc[0] ^ PILLAR[16], acc[1] ^ PILLAR[17]);
        h += cross(acc[2] ^ PILLAR[18], acc[3] ^ PILLAR[19]);
        h += cross(acc[4] ^ PILLAR[20], acc[5] ^ PILLAR[21]);
        h += cross(acc[6] ^ PILLAR[22], acc[7] ^ PILLAR[23]);
        return shadow(h);
    }

    /* ======= REGIME B: shallow pit — the water never covers the wires ====== */
    h = PILLAR[0] ^ ((uint64_t)len * 0x9E3779B97F4A7C15ULL);

    if (len >= 16) {
        size_t i = 0;
        while (i + 16 <= len) {
            const uint64_t *k = PILLAR + 4 + ((i >> 4) & 3) * 2;
            h += cross(rd64(d + i) ^ k[0], rd64(d + i + 8) ^ k[1]);
            h  = ROTL64(h, 37);
            i += 16;
        }
        if (i < len)                           /* overlapping tail, full coverage */
            h += cross(rd64(d + len - 16) ^ PILLAR[12],
                       rd64(d + len -  8) ^ PILLAR[13]);
    } else if (len >= 8) {
        uint64_t lo = rd64(d)           ^ PILLAR[1];
        uint64_t hi = rd64(d + len - 8) ^ PILLAR[2];
        h += ROTL64(lo, 32) + hi + cross(lo, hi);
    } else if (len >= 4) {
        uint64_t x = ((uint64_t)rd32(d + len - 4) << 32) | (uint64_t)rd32(d);
        h += cross(x ^ PILLAR[1], ROTL64(x, 32) ^ PILLAR[2]);
    } else if (len > 0) {
        uint64_t x = ((uint64_t)d[0] << 16) | ((uint64_t)d[len >> 1] << 8)
                   |  (uint64_t)d[len - 1]  | ((uint64_t)len << 24);
        h += cross(x ^ PILLAR[1], ROTL64(x, 29) ^ PILLAR[2]);
    }
    return shadow(h);                          /* read once, kept alone */
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 20**

Stated before any measurement (no tools were available in this session, so nothing was measured first).

Reasoning behind the number: FNV-1a is one serially-dependent 64-bit multiply per **byte** — ~4–5 cycles of latency per byte, so ≈0.2 B/cycle (~0.6–0.8 GB/s). The flood absorbs 64 bytes per pass in 2 AVX2 instruction groups with 8 independent dependency chains, so it is throughput- rather than latency-bound: ~10–20 B/cycle on large buffers (~30–50 GB/s), i.e. 40–60× on multi-KB inputs. If the harness averages over a size sweep that includes sub-64-byte buffers (where the shallow-pit path wins only ~2–4×), the reported aggregate should land well below the large-buffer peak — hence 20 rather than 50.

Quality prediction: avalanche ≈ 0.50 bit-flip probability (score ≈ 1.0 on a normalized scale), **better** than the baseline despite doing far less per-byte work — FNV-1a's final multiply leaves its lowest output bits poorly diffused (the low bit of the output is a pure XOR of input low bits), whereas every output bit here passes through `fmix64`.

# MEASUREMENT

**Not performed — honest statement of failure to measure.** `hash_bench` and `hash_contract` were listed as available but no tools are actually callable in this session, so I have no throughput number, no avalanche score, and I made zero of my allotted four improvement rounds. Everything above the PREDICTION line is design, not evidence. The pipeline will compile and measure; if the measured speedup is materially below 20, the prediction was wrong and the likely cause is a size mix dominated by short buffers (shallow-pit path), not a defect in the flood loop.

Specific things a real measurement must check, which I cannot assert:
1. Avalanche at `len` = 1, 3, 7, 15, 32, 63 — the shallow-pit path is the weakest link and the least validated part of this kernel.
2. Whether `-march=native` actually enables `__AVX2__` on the bench machine; if not, the scalar path runs and I'd expect ~6–10× instead of ~20×.
3. That the 16 hand-written pillar words I authored (indices 8–31) are not accidentally degenerate.

# VERDICT

The metaphor translated with unusual fidelity and, exactly as step 4 requires, it **converged on a validated real technique instead of a novel one**: anchored wires = XXH3's 8-lane accumulator; "the flood moves them" = the swap-add + 32×32 absorb; "the shadows stop chittering" = the per-block scramble; "which cross another" = `mul128_fold64` pairwise merge; "the silhouette, drawn only once" = a single `fmix64`. I did not invent a mixer.

The broken assumption is the substantive one. FNV-1a spends a strong round per byte and gets *worse* diffusion; this kernel spends an avalanche-free round per 8 bytes per lane and gets better diffusion, because diffusion is a property of the **final projection**, not of the number of rounds. More rounds bought nothing; wider state plus one good round bought everything.

**Stated risks and how each is discharged (step 4 compliance — no unaddressed risk shipped):**

| Risk named by me | Guard in the shipped code |
|---|---|
| Lane init + 4 crosses + merge is pure overhead on small inputs | `if (len >= 64)` regime branch; buffers below one flood-pass never touch the wires and take a short, cheap, fully-covering path |
| AVX2 may be absent | `#if defined(__AVX2__)` with a complete scalar `absorb_s`/`settle_s` fallback producing the same hash on little-endian |
| `__int128` may be absent | portable 32×32 decomposition inside `cross` |
| Thread parallelism could be a net loss at bench sizes | **dropped entirely** rather than shipped unguarded — the metaphor has one river, and 64-byte stripes are vector work, not thread work |
| Tail bytes could be skipped | last pass is an overlapping read of `d + len - 64`; short path's tail is an overlapping last-16 — full coverage proven for `len = 64, 65, 128, 129, 16, 63` |

**Where this could still be worse than the known way.** At `len` of 1–8 bytes the shallow path does a 128-bit multiply plus `fmix64` where FNV-1a does one or two cheap multiplies; the per-call overhead may make it *slower* than the baseline on single-digit inputs. I chose to keep it there rather than branch further, because it buys genuine avalanche on short keys, which FNV-1a does not have — but if the harness weights 1–8 byte inputs heavily, the aggregate speedup will disappoint and that is the reason, not a bug.

**What the metaphor does not justify.** The `settle` step (per-block scramble) is the one place I added mixing rather than removed it, and the metaphor's support for it is thin ("the walls are still settling"). It is cheap — 8 multiplies per 1024 bytes — and it is in the validated construction for good reason at long lengths, so I kept it; but a strict reading of SEED 3 would omit it, and a measurement comparing with and without it would be the honest first improvement round I never got to run.