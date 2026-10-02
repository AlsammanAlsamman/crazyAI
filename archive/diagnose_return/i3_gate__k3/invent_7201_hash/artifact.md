## MAPPING

### SEED 1 — the grey wax trough

| World object | Problem object |
|---|---|
| the pile of marks | the input byte buffer `data[0..len)` |
| one mark read into the trough | 8 bytes read as a `uint64_t` word |
| the **trough** (a wide vessel, not a cup) | a **wide state surface**: 16 × `uint64_t` cells (1024 bits) in registers |
| wax "kept always at blood heat" | the state is never allowed to settle: every cell stays plastic and keeps flowing into its neighbours |
| a mark *presses its shape* into wax | `cell += word` — displacement, carries push material upward; never `^` (an xor "walks off" if pressed twice) |
| "before the last shape has cooled flat" | the per-press diffusion is **deliberately incomplete** — one cheap spread step per 128-byte press, not full mixing |
| wax flows sideways while warm | `cell[i] += rotl(cell[j], k)` — lateral shear between cells, rotation = the shape spreading inside a cell |
| "never a clean picture of any single mark, only the sum of every collision" | no byte is recoverable from any one cell; each cell is the churned sum of all presses + all lateral flow |
| scrape clean & remelt | the 128-byte pad scratch and the state, re-initialised each call |
| change one mark → *whole* flower changes | avalanche requirement |

**Breaks:** "the state is a single accumulator updated in place, one value" (the trough is a *surface* of 16 coupled cells), and secondarily "each byte must be mixed into the running state before the next byte is read" (128 bytes are pressed before anything is mixed) and "more mixing rounds always means better mixing" (absorb is under-mixed on purpose).

### SEED 2 — seven swarms, forward-only slate wheel

| World object | Problem object |
|---|---|
| seven bell-jars, each a different hunger | 7 parallel keyed lanes with distinct constants |
| bites counted per jar | per-lane accumulate of the same input |
| wheel that *only turns forward*, no bite walked off | **non-cancelling** accumulation: add/rotate only, never xor |
| wheel's resting notch | merged lane state |

**Breaks:** the single-accumulator assumption too — but as 7 independent lanes merged at the end, which *is* the xxHash/XXH3 lane structure, i.e. the known way. (Its one genuinely non-textbook grain — forward-only, nothing cancellable — I keep and use.)

### SEED 3 — brine basin, six frost-flowers

| World object | Problem object |
|---|---|
| single brine plunge at the end | one-time finalisation pass, cost not per byte |
| "wait until the frost-lace stops spreading" | run the permutation until full diffusion; a token squeezed early "is soft and lies" |
| lattice of six frost-flowers, none alike | six state cells read out with six distinct rotations, folded |
| sketched on tin, wax discarded | only the 64-bit return value survives |

**Breaks:** "more mixing rounds always means better mixing" — rounds are worth paying *once*, at the end, not per byte. On its own it is only a finaliser, not a hash core.

## CHOSEN SEED

**Seed 1, the grey wax trough.** All three touch the single-accumulator assumption, so I apply the stated preference and pick the most literal and most different from the known way. Seed 2's literal reading collapses into XXH3's parallel lanes (exactly what the previous attempt was correctly rejected for); seed 3 is only a finaliser. Seed 1 is a *wide, laterally-coupled, never-settled surface* — structurally unlike one multiply-folded accumulator. Seeds 2 and 3 are kept as subordinate details of the same story (forward-only = add-only accumulation; the brine plunge = the final diffusion + six-flower squeeze).

Where this lands, honestly: a wide state, absorbed into by addition, stirred by an under-powered permutation per block, then fully diffused once and squeezed — **is the sponge construction with a reduced-round ARX permutation** (Keccak/Gimli/Xoodoo/SipHash family). That is a validated, real-world technique, so per step 4 I let the wax trough *arrive* at it rather than inventing a private design, and I use Murmur3's `fmix64` verbatim as the crystallisation step. The non-textbook part I keep from the metaphor: the permutation is **pure add+rotate, no xor anywhere** in absorb or plunge (the forward-only wheel — nothing can be walked off, so repeated/duplicate blocks cannot cancel), and absorb is intentionally 1 round where SHA-3 would use 24.

## ASSUMPTION BROKEN

> *the state is a single accumulator updated in place, one value*

The state is a 1024-bit trough of 16 cells. A byte is never mixed into "the" accumulator; it is pressed into one place on the surface and reaches the rest only by lateral flow. Consequently also broken: each byte is *not* mixed before the next is read (128 bytes press first), and per-press mixing is deliberately insufficient — all the mixing debt is paid once in the brine.

**Regime recognition (step 5).** The known way spans small and large inputs. The native weighs the pile against the trough before starting: a pile that cannot even fill the trough would spend all its work warming wax, so it goes into a **narrow channel** instead (4 cells, 32-/8-byte presses, 4 plunge rounds) — guarded by `len < 128`, with the same mechanism in miniature. This is the required guard for the one condition where the wide trough would lose: its fixed ~60-cycle plunge over a 16-cell surface is dead weight on an 8-byte pile. No thread parallelism: the trough is one vessel and the plunge is one event; splitting it would be two troughs, not this mechanism, and the benchmark sizes are not known to be large enough to pay for a fork/join.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- the Weighing Hall -------------------------------------------------
   A grey wax trough of 16 cells, kept at blood heat.  Marks are PRESSED in
   (add: material displaces, carries ride upward; never xor -- the slate
   wheel turns forward only, no bite can be walked off).  After each press
   the warm wax flows sideways ONE step -- not enough to cool flat.  When
   the pile ends: one brine plunge, held until the lace stops spreading,
   then six frost-flowers are read off and sketched to tin.               */

static inline uint64_t rotl64(uint64_t x, int n){ return (x << n) | (x >> (64 - n)); }
static inline uint64_t rd64(const unsigned char *p){ uint64_t v; memcpy(&v, p, 8); return v; }

/* crystallisation: Murmur3 fmix64 (validated finaliser, paid once) */
static inline uint64_t crystallize(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32; return x;
}

/* ---- narrow channel: a pile too small to warm the whole trough -------- */
#define NSPREAD do {               \
        c1 += rotl64(c0, 13);      \
        c3 += rotl64(c2, 29);      \
        c0 += rotl64(c3, 41);      \
        c2 += rotl64(c1, 53);      \
    } while (0)

static uint64_t narrow_channel(const unsigned char *data, size_t len){
    uint64_t c0 = 0x243f6a8885a308d3ULL, c1 = 0x13198a2e03707344ULL;
    uint64_t c2 = 0xa4093822299f31d0ULL, c3 = 0x082efa98ec4e6c89ULL;
    const unsigned char * restrict p = data;
    size_t rem = len;

    c0 += (uint64_t)len;                       /* the pile is weighed first */

    while (rem >= 32){
        c0 += rd64(p);      c1 += rd64(p + 8);
        c2 += rd64(p + 16); c3 += rd64(p + 24);
        NSPREAD;
        p += 32; rem -= 32;
    }
    while (rem >= 8){ c0 += rd64(p); NSPREAD; p += 8; rem -= 8; }
    {   /* scrape the last marks in, with a pad stop so 0-bytes still count */
        uint64_t t = 0;
        if (rem) memcpy(&t, p, rem);
        t += (uint64_t)1 << (8 * rem);         /* rem is 0..7 */
        c1 += t; NSPREAD;
    }
    { int i; for (i = 0; i < 4; i++) NSPREAD; } /* short plunge */

    {   /* six frost-flowers, no two alike */
        uint64_t f = c0
                   + rotl64(c1, 17) + rotl64(c2, 31) + rotl64(c3, 47)
                   + rotl64(c0,  5) + rotl64(c2, 53);
        return crystallize(f);
    }
}

#if !defined(__AVX2__)
/* ---- the full trough, scalar ----------------------------------------- */
static const uint64_t WAX[16] = {
    0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL, 0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
    0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL, 0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
    0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL, 0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
    0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL, 0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
};
/* four shears: two independent, two coupling; each writes a group it does
   not read, so every step is invertible and the loops vectorise.          */
static inline void spread16(uint64_t * restrict s){
    static const int RA[4] = {13,29,41,53}, RB[4] = { 7,19,37,47};
    static const int RC[4] = {11,23,43,59}, RD[4] = {17,31, 5,61};
    int i;
    for (i = 0; i < 4; i++) s[ 4 + i] += rotl64(s[ 0 + ((i + 1) & 3)], RA[i]);
    for (i = 0; i < 4; i++) s[12 + i] += rotl64(s[ 8 + ((i + 2) & 3)], RB[i]);
    for (i = 0; i < 4; i++) s[ 0 + i] += rotl64(s[12 + ((i + 3) & 3)], RC[i]);
    for (i = 0; i < 4; i++) s[ 8 + i] += rotl64(s[ 4 + ((i + 1) & 3)], RD[i]);
}
#endif

uint64_t kernel(const unsigned char *data, size_t len){
    /* regime: can this pile fill the trough at all? */
    if (len < 128) return narrow_channel(data, len);

#if defined(__AVX2__)
    {
        const __m256i L1 = _mm256_setr_epi64x(13,29,41,53), Q1 = _mm256_setr_epi64x(51,35,23,11);
        const __m256i L2 = _mm256_setr_epi64x( 7,19,37,47), Q2 = _mm256_setr_epi64x(57,45,27,17);
        const __m256i L3 = _mm256_setr_epi64x(11,23,43,59), Q3 = _mm256_setr_epi64x(53,41,21, 5);
        const __m256i L4 = _mm256_setr_epi64x(17,31, 5,61), Q4 = _mm256_setr_epi64x(47,33,59, 3);
#define ROLV(x,l,q) _mm256_or_si256(_mm256_sllv_epi64((x),(l)), _mm256_srlv_epi64((x),(q)))
/* one step of warm wax flowing sideways: four invertible shears */
#define SPREAD do {                                                                        \
        B = _mm256_add_epi64(B, _mm256_permute4x64_epi64(ROLV(A,L1,Q1), 0x39));            \
        D = _mm256_add_epi64(D, _mm256_permute4x64_epi64(ROLV(C,L2,Q2), 0x4E));            \
        A = _mm256_add_epi64(A, _mm256_permute4x64_epi64(ROLV(D,L3,Q3), 0x93));            \
        C = _mm256_add_epi64(C, _mm256_permute4x64_epi64(ROLV(B,L4,Q4), 0x39));            \
    } while (0)

        __m256i A = _mm256_setr_epi64x((long long)0x243f6a8885a308d3ULL,(long long)0x13198a2e03707344ULL,
                                       (long long)0xa4093822299f31d0ULL,(long long)0x082efa98ec4e6c89ULL);
        __m256i B = _mm256_setr_epi64x((long long)0x452821e638d01377ULL,(long long)0xbe5466cf34e90c6cULL,
                                       (long long)0xc0ac29b7c97c50ddULL,(long long)0x3f84d5b5b5470917ULL);
        __m256i C = _mm256_setr_epi64x((long long)0x9216d5d98979fb1bULL,(long long)0xd1310ba698dfb5acULL,
                                       (long long)0x2ffd72dbd01adfb7ULL,(long long)0xb8e1afed6a267e96ULL);
        __m256i D = _mm256_setr_epi64x((long long)0xba7c9045f12c7f99ULL,(long long)0x24a19947b3916cf7ULL,
                                       (long long)0x0801f2e2858efc16ULL,(long long)0x636920d871574e69ULL);
        const unsigned char * restrict p = data;
        size_t n = len >> 7, r = len & 127, i;

        A = _mm256_add_epi64(A, _mm256_set1_epi64x((long long)len));   /* weigh the pile */

        for (i = 0; i < n; i++, p += 128){
            A = _mm256_add_epi64(A, _mm256_loadu_si256((const __m256i *)(const void *)(p      )));
            B = _mm256_add_epi64(B, _mm256_loadu_si256((const __m256i *)(const void *)(p +  32)));
            C = _mm256_add_epi64(C, _mm256_loadu_si256((const __m256i *)(const void *)(p +  64)));
            D = _mm256_add_epi64(D, _mm256_loadu_si256((const __m256i *)(const void *)(p +  96)));
            SPREAD;                       /* ONE step: never cools flat */
        }
        {   unsigned char pad[128];
            memset(pad, 0, sizeof pad);
            if (r) memcpy(pad, p, r);
            pad[r] = 0x01;                /* r < 128 */
            A = _mm256_add_epi64(A, _mm256_loadu_si256((const __m256i *)(const void *)(pad      )));
            B = _mm256_add_epi64(B, _mm256_loadu_si256((const __m256i *)(const void *)(pad +  32)));
            C = _mm256_add_epi64(C, _mm256_loadu_si256((const __m256i *)(const void *)(pad +  64)));
            D = _mm256_add_epi64(D, _mm256_loadu_si256((const __m256i *)(const void *)(pad +  96)));
            SPREAD;
        }
        for (i = 0; i < 12; i++) SPREAD;  /* the brine plunge, held */

        {   uint64_t o[16]; uint64_t f;
            _mm256_storeu_si256((__m256i *)(void *)(o     ), A);
            _mm256_storeu_si256((__m256i *)(void *)(o +  4), B);
            _mm256_storeu_si256((__m256i *)(void *)(o +  8), C);
            _mm256_storeu_si256((__m256i *)(void *)(o + 12), D);
            f = o[0]
              + rotl64(o[ 5], 19) + rotl64(o[10], 37) + rotl64(o[15], 52)
              + rotl64(o[ 3], 29) + rotl64(o[ 9],  7);      /* six flowers */
            return crystallize(f);
        }
#undef SPREAD
#undef ROLV
    }
#else
    {
        uint64_t s[16];
        const unsigned char * restrict p = data;
        size_t n = len >> 7, r = len & 127, i; int j;

        memcpy(s, WAX, sizeof s);
        s[0] += (uint64_t)len;

        for (i = 0; i < n; i++, p += 128){
            for (j = 0; j < 16; j++) s[j] += rd64(p + 8 * j);
            spread16(s);
        }
        {   unsigned char pad[128];
            memset(pad, 0, sizeof pad);
            if (r) memcpy(pad, p, r);
            pad[r] = 0x01;
            for (j = 0; j < 16; j++) s[j] += rd64(pad + 8 * j);
            spread16(s);
        }
        for (i = 0; i < 12; i++) spread16(s);
        {   uint64_t f = s[0]
                       + rotl64(s[ 5], 19) + rotl64(s[10], 37) + rotl64(s[15], 52)
                       + rotl64(s[ 3], 29) + rotl64(s[ 9],  7);
            return crystallize(f);
        }
    }
#endif
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 18

Stated before any measurement, with the reasoning so it can be checked: FNV-1a is a serial `xor`→`imul` dependency chain, ~4 cycles per **byte** (≈0.25 B/cycle). The trough presses 128 bytes per iteration with 4 loads + 4 adds + one 4-shear spread (~20 vector µops), and the critical recurrence closes only every two iterations, so the loop should be issue/load-limited near ~7 cycles per 128 B (≈18 B/cycle). That is a theoretical ~70×; I discount hard for loop overhead, the ~60-cycle fixed plunge, and whatever buffer sizes the harness actually uses. Secondary predictions: avalanche ≈ 0.49–0.50 (any single-bit difference reaches the six-flower sum through ≥13 full diffusion steps, then `fmix64`); at 8–32 bytes the narrow channel should still be ~3–5× FNV, not slower.

## MEASUREMENT

**Not measured.** This session was invoked with no tools available — `hash_bench`, `hash_contract` and the symbolic/unconventional tools could not be called, so I ran zero benchmark iterations and performed none of the up-to-four improvement rounds. (Two MCP connectors, claude.ai PubMed, are also unauthorised and would need authorising from claude.ai connector settings or an interactive `claude mcp` session; irrelevant here, but it is part of why nothing could be executed.) I am reporting that plainly rather than inventing numbers: everything in PREDICTION is an estimate, and the pipeline's measurement supersedes it.

What would falsify me, concretely: measured speedup below ~6× (then the spread step is dependency-bound, not issue-bound, and the fix is a second independent shear pair, not more rounds); avalanche below 0.48 (then the plunge is too short — the native's own warning that a token pulled early "is soft and lies" — raise 12 rounds to 20, which costs nothing per byte); or the small-size path losing to FNV (then lower the narrow plunge from 4 rounds to 2 and raise the `len < 128` threshold).

## VERDICT

The core of this kernel **is** the wax trough: a wide 16-cell surface, pressed into by addition only (the forward-only wheel — nothing cancellable, so duplicated blocks can never annihilate), stirred by exactly one deliberately-insufficient lateral-flow step per 128-byte press, then plunged once and read as six frost-flowers. There is no accumulator, no per-byte multiply, and no per-byte full mix anywhere in it. It is not the previous attempt's XXH3 lane structure: the 16 cells are not independent lanes merged at the end — they exchange material every single press, which is what makes "only the churned sum of every collision" literally true of every cell.

Honest accounting of what is standard: this lands on the **sponge construction with a reduced-round ARX permutation**, which is exactly where step 4 says it should land rather than on something novel and untested, and the finaliser is Murmur3's `fmix64` unmodified. The genuinely metaphor-driven, non-textbook choices are the xor-free permutation and the 1-round absorb / 13-round finish split.

Risks I named and therefore had to address, not just flag: the wide trough's fixed plunge cost is pure loss on small piles, so `len < 128` routes to the narrow channel — the same mechanism with 4 cells and a 4-round plunge, a real fallback, not a comment. Thread parallelism is omitted on purpose: one trough, one plunge, and no evidence the benchmark sizes would repay a fork/join. The remaining unaddressed risk is honest ignorance: add-and-rotate diffusion has a known weak direction (carries travel up only), and I am relying on the rotations to recirculate high bits down and on `fmix64` as the safety net. Unmeasured, that is a reasoned expectation, not a result.