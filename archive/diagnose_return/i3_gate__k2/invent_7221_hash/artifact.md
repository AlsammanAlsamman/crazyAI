Before building: the reviewer is right that my previous artifact was xxHash with a costume on. Let me redo the translation from the top, and this time let the mechanism *be* the kernel.

---

## MAPPING

### SEED 1 — "one crease per mark, angle set by the mark **and the crease before it**"

| World object | Problem object |
|---|---|
| huge sheet, laid flat | the hash state living in registers |
| the pile of marks, read in order, never skipping, never looking ahead | `data[0..len)` consumed strictly sequentially |
| one fold per mark | one state transition per input unit |
| fold angle = f(mark, previous crease) | `state ← F(state, input)` — an unbroken serial dependency chain |
| "the paper carries forward everything it has already been told" | full state carry: no byte's effect is ever dropped |
| a fold is a **rotation/reflection** of the sheet, never a stretch | mixing by rotate + add + xor — **no multiplication** |

**Silent assumption broken:** *"mixing one byte requires a multiplication."* A fold has an angle; angles rotate, they don't scale. This seed is ARX mixing.
It explicitly **re-affirms** "each byte must be mixed before the next is read."

### SEED 2 — "test every crease against **all four** wire birds' beaks at once, refold until they agree"

| World object | Problem object |
|---|---|
| the four wire birds | **four** 64-bit state words `v0,v1,v2,v3` — the state is a 256-bit vector, not an accumulator |
| pressing *the same* crease against *each* bird's beak | every input word is injected into the state and then diffused through **all four** words |
| "only when all four beaks agree does the crease count as set" | the per-input mixing step does not terminate until all four words have been updated as a function of all four — a full round, not a lane update |
| "if they disagree I refold tighter" | *c* repetitions of that round per input word |
| spiral markings vs. snail-shell markings "line up wrong on purpose, scrambled" | the four distinct initialisation constants and the mutually non-aligning rotation amounts (13,16,32,17,21,32) |
| "because a clean line-up would mean two different piles could end up looking the same" | collision resistance is the *design goal*, stated outright |

**Silent assumption broken:** *"the state is a single accumulator updated in place, one value."*
Critically — and this is where my last attempt cheated — the four birds are **not** four lanes each reading its own quarter of the bytes (that is xxHash). Every crease is pressed against **all four beaks at once**. Four state words, one input stream.

### SEED 3 — "thin the whole wad at the water's edge to one dense corner, throw away every scrap"

| World object | Problem object |
|---|---|
| carrying the *whole* folded wad to the water's edge | a finalisation phase applied after the last input |
| "the way a body thins going under **with one small suitcase**" | the message **length** is carried into that last fold — in one byte-sized slot |
| holding it down until it is "small, small, small" | extra mixing rounds with no new input, *d* of them |
| one hard dense corner, no bigger than a coin | the 64-bit return value = reduction of all four words |
| every sheet, every scrap, every failed bird-reading thrown in the mud | no intermediate state escapes; only the 64-bit digest is returned |

**Silent assumption broken:** *"more mixing rounds always means better mixing."* The thinning happens **once, at the end**, not everywhere — rounds are spent where they buy diffusion.

---

## CHOSEN SEED

**Seed 2 — the four-bird per-crease consensus.**

First, plainly, as instructed: **none of the three seeds breaks "each byte must be mixed into the running state before the next byte is read."** Seed 1 goes out of its way to *insist* on it ("never skipping, never looking ahead"), seed 2 is per-crease and therefore also serial, and seed 3 is a post-pass over an already-folded wad. So I fall back to the most literal mapping that is also most distant from the known way, which is seed 2: the known way has one accumulator and a multiply; seed 2 has a four-word state in which every input word must pass through all four words before the next word is read, using folds (rotations), not stretches.

## ASSUMPTION BROKEN

**"The state is a single accumulator updated in place, one value."** The state is four words wide and every input word is consulted against all four.

### Step 4: the mechanism lands on a validated technique, so I use that technique

A four-word 64-bit state; each message word XORed in, then a round that mixes all four words; *c* such rounds per word; a final block whose top byte holds the length ("one small suitcase"); *d* extra rounds with no input ("thinned, small, small, small"); output `v0^v1^v2^v3` ("one coin"); only add-rotate-xor, no multiply; rotation constants chosen so nothing aligns ("scrambled on purpose"); and the stated design goal being that two distinct inputs cannot look the same.

That is not something I need to invent. **That is SipHash** (Aumasson–Bernstein 2012) — four-word ARX state, `v3 ^= m; c×SipRound; v0 ^= m`, `b = len<<56` padding, `v2 ^= 0xff`, `d×SipRound`, XOR-fold to one word. It is the default hash-table hasher in Python, Rust, Perl, Haskell, Redis and the Linux kernel, precisely *because* of the native's stated reason. The native's bird-gauge is a SipRound; "refold tighter until all four agree" is *c*; the suitcase is the length byte; the water's edge is the *d*-round finalisation; the coin is the XOR-fold. I let the metaphor arrive here rather than building a novel four-word mixer.

### Step 5: two regimes, recognised in-metaphor

The native's own first sentence contains the regime test: *"only that it's large enough"* — and later he speaks of **sheets**, plural: *"every sheet I used along the way."* So:

- **Small pile → one sheet.** `len < 256`: strictly serial canonical **SipHash-2-4** over the whole buffer. No setup, no merge, no SIMD overhead.
- **Large pile → four sheets, dealt.** `len ≥ 256`: the marks are dealt round-robin onto four sheets (each sheet still reads *its own* marks in order, never skipping, never looking ahead), each sheet carrying its own four birds with differently-scrambled markings. Because the four birds are four *registers*, four sheets fit exactly in one AVX2 vector per bird: **V0,V1,V2,V3 are the four birds; the vector lanes are the sheets.** Per-sheet compression is **SipHash-1-3** (the validated parameterisation shipped as Rust's default hasher), then each sheet is thinned to its own coin, and the four coins — plus the untouched tail, plus the true total length as the suitcase — are carried to the water's edge and folded one final time with full SipHash-2-4.

The 256-byte threshold is the required guard: the four-sheet path's fixed cost (4 sheet-thinnings + one merge fold ≈ 300 cycles) exactly breaks even against the one-sheet path at 256 bytes, so the parallel path can never be the slower choice. Vectorisation only — no OpenMP threads; the sheets are 64-byte-granular units of work, far too small to pay for a thread.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ===================================================================
   THE FOUR WIRE BIRDS.  v0,v1,v2,v3 are the four beaks.  Every crease
   (one message word) is pressed against ALL FOUR at once: the word is
   xored in, then one round carries it through all four words before the
   next mark may be read.  Folds only -- add, rotate, xor.  No stretch,
   no multiply.  The rotation amounts are the spiral and snail markings
   deliberately scrambled so that no two piles can line up.
   This round is the SipRound (Aumasson-Bernstein).
   =================================================================== */
#define ROTL(x, b) ((uint64_t)(((x) << (b)) | ((x) >> (64 - (b)))))

#define BIRDS(v0, v1, v2, v3)                                              \
    do {                                                                   \
        v0 += v1; v1 = ROTL(v1, 13); v1 ^= v0; v0 = ROTL(v0, 32);          \
        v2 += v3; v3 = ROTL(v3, 16); v3 ^= v2;                             \
        v0 += v3; v3 = ROTL(v3, 21); v3 ^= v0;                             \
        v2 += v1; v1 = ROTL(v1, 17); v1 ^= v2; v2 = ROTL(v2, 32);          \
    } while (0)

#define IV0 0x736f6d6570736575ULL   /* "somepseu" */
#define IV1 0x646f72616e646f6dULL   /* "dorandom" */
#define IV2 0x6c7967656e657261ULL   /* "lygenera" */
#define IV3 0x7465646279746573ULL   /* "tedbytes" */

#define K0  0x0706050403020100ULL
#define K1  0x0f0e0d0c0b0a0908ULL

/* "only that it's large enough" -- the regime test. Below this, one
   sheet; at or above it, four dealt sheets. 256 is the exact analytic
   break-even, so the four-sheet path is never the slower choice. */
#define SHEET_MIN ((size_t)256)

/* -------------------------------------------------------------------
   ONE SHEET, STRICTLY SERIAL: canonical SipHash-2-4.
   One crease per mark, each crease's angle set by the mark and by the
   state the previous crease left; two refolds per crease until the four
   beaks agree; then the water's edge -- the length rides in as the one
   small suitcase (top byte of the last fold), four thinning rounds with
   no new marks, and the four birds collapse to one coin.
   ------------------------------------------------------------------- */
static uint64_t fold_one_sheet(const unsigned char *in, size_t inlen,
                               uint64_t k0, uint64_t k1)
{
    uint64_t v0 = IV0 ^ k0;
    uint64_t v1 = IV1 ^ k1;
    uint64_t v2 = IV2 ^ k0;
    uint64_t v3 = IV3 ^ k1;
    uint64_t m, b = ((uint64_t)inlen) << 56;     /* the small suitcase */
    size_t   nw = inlen >> 3, left = inlen & 7u, i;

    for (i = 0; i < nw; i++) {                   /* never skip, never look ahead */
        memcpy(&m, in + (i << 3), 8);
        v3 ^= m;
        BIRDS(v0, v1, v2, v3);
        BIRDS(v0, v1, v2, v3);
        v0 ^= m;
    }
    in += (nw << 3);
    switch (left) {                              /* every trimmed scrap still folded in */
    case 7: b |= (uint64_t)in[6] << 48; /* fall through */
    case 6: b |= (uint64_t)in[5] << 40; /* fall through */
    case 5: b |= (uint64_t)in[4] << 32; /* fall through */
    case 4: b |= (uint64_t)in[3] << 24; /* fall through */
    case 3: b |= (uint64_t)in[2] << 16; /* fall through */
    case 2: b |= (uint64_t)in[1] <<  8; /* fall through */
    case 1: b |= (uint64_t)in[0];       /* fall through */
    default: break;
    }
    v3 ^= b;
    BIRDS(v0, v1, v2, v3);
    BIRDS(v0, v1, v2, v3);
    v0 ^= b;

    v2 ^= 0xffULL;                               /* hold it down at the water's edge */
    BIRDS(v0, v1, v2, v3);
    BIRDS(v0, v1, v2, v3);
    BIRDS(v0, v1, v2, v3);
    BIRDS(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;                    /* one hard dense corner */
}

#if defined(__AVX2__)
/* The four birds, each now holding its gauge reading for all four
   sheets at once: one AVX2 register per bird, one lane per sheet. */
#define VROTL(x, b) _mm256_or_si256(_mm256_slli_epi64((x), (b)),            \
                                    _mm256_srli_epi64((x), 64 - (b)))
#define VR32(x)     _mm256_shuffle_epi32((x), 0xB1)
#define VR16(x)     _mm256_shuffle_epi8((x), r16)
#define VBIRDS(V0, V1, V2, V3)                                             \
    do {                                                                   \
        V0 = _mm256_add_epi64(V0, V1); V1 = VROTL(V1, 13);                 \
        V1 = _mm256_xor_si256(V1, V0); V0 = VR32(V0);                      \
        V2 = _mm256_add_epi64(V2, V3); V3 = VR16(V3);                      \
        V3 = _mm256_xor_si256(V3, V2);                                     \
        V0 = _mm256_add_epi64(V0, V3); V3 = VROTL(V3, 21);                 \
        V3 = _mm256_xor_si256(V3, V0);                                     \
        V2 = _mm256_add_epi64(V2, V1); V1 = VROTL(V1, 17);                 \
        V1 = _mm256_xor_si256(V1, V2); V2 = VR32(V2);                      \
    } while (0)
#endif

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t v0[4], v1[4], v2[4], v3[4], coin[4];
    unsigned char wad[64];
    size_t nblk, bulk, tail, i;
    int j;

    /* ---- regime test: is the sheet large enough to be worth dealing? */
    if (len < SHEET_MIN)
        return fold_one_sheet(p, len, K0, K1);

    /* ---- four sheets, each with its own differently-scrambled markings */
    for (j = 0; j < 4; j++) {
        uint64_t s = 0x9E3779B97F4A7C15ULL * (uint64_t)(j + 1);
        uint64_t t = 0xBF58476D1CE4E5B9ULL * (uint64_t)(j + 1);
        v0[j] = IV0 ^ (K0 ^ s);
        v1[j] = IV1 ^ (K1 ^ t);
        v2[j] = IV2 ^ (K0 ^ s);
        v3[j] = IV3 ^ (K1 ^ t);
    }

    nblk = len >> 5;            /* 32 bytes dealt per step: one word per sheet */
    bulk = nblk << 5;

#if defined(__AVX2__)
    {
        const __m256i r16 = _mm256_setr_epi8(
            6, 7, 0, 1, 2, 3, 4, 5, 14, 15, 8, 9, 10, 11, 12, 13,
            6, 7, 0, 1, 2, 3, 4, 5, 14, 15, 8, 9, 10, 11, 12, 13);
        __m256i V0 = _mm256_loadu_si256((const __m256i *)v0);
        __m256i V1 = _mm256_loadu_si256((const __m256i *)v1);
        __m256i V2 = _mm256_loadu_si256((const __m256i *)v2);
        __m256i V3 = _mm256_loadu_si256((const __m256i *)v3);
        __m256i M;

        for (i = 0; i + 2 <= nblk; i += 2) {
            M  = _mm256_loadu_si256((const __m256i *)(p + (i << 5)));
            V3 = _mm256_xor_si256(V3, M);
            VBIRDS(V0, V1, V2, V3);
            V0 = _mm256_xor_si256(V0, M);
            M  = _mm256_loadu_si256((const __m256i *)(p + ((i + 1) << 5)));
            V3 = _mm256_xor_si256(V3, M);
            VBIRDS(V0, V1, V2, V3);
            V0 = _mm256_xor_si256(V0, M);
        }
        for (; i < nblk; i++) {
            M  = _mm256_loadu_si256((const __m256i *)(p + (i << 5)));
            V3 = _mm256_xor_si256(V3, M);
            VBIRDS(V0, V1, V2, V3);
            V0 = _mm256_xor_si256(V0, M);
        }
        _mm256_storeu_si256((__m256i *)v0, V0);
        _mm256_storeu_si256((__m256i *)v1, V1);
        _mm256_storeu_si256((__m256i *)v2, V2);
        _mm256_storeu_si256((__m256i *)v3, V3);
    }
#else
    for (i = 0; i < nblk; i++) {
        uint64_t m[4];
        memcpy(m, p + (i << 5), 32);
        for (j = 0; j < 4; j++) v3[j] ^= m[j];
        for (j = 0; j < 4; j++) BIRDS(v0[j], v1[j], v2[j], v3[j]);
        for (j = 0; j < 4; j++) v0[j] ^= m[j];
    }
#endif

    /* ---- each sheet thinned to its own coin (SipHash-1-3 finalisation) */
    {
        uint64_t lb = ((uint64_t)(nblk << 3)) << 56;   /* that sheet's suitcase */
        for (j = 0; j < 4; j++) {
            uint64_t a = v0[j], b = v1[j], c = v2[j], d = v3[j];
            d ^= lb;
            BIRDS(a, b, c, d);
            a ^= lb;
            c ^= 0xffULL;
            BIRDS(a, b, c, d);
            BIRDS(a, b, c, d);
            BIRDS(a, b, c, d);
            coin[j] = a ^ b ^ c ^ d;
        }
    }

    /* ---- the water's edge: the whole reduced wad -- four coins and the
           scraps the dealing could not cover -- folded one last time,
           with the true total length as the suitcase.  Nothing else
           survives; every sheet and every intermediate reading is gone. */
    memcpy(wad, coin, 32);
    tail = len - bulk;                              /* < 32 */
    if (tail) memcpy(wad + 32, p + bulk, tail);
    return fold_one_sheet(wad, 32 + tail,
                          K0 ^ (uint64_t)len, K1 ^ (uint64_t)nblk);
}
```

## PREDICTION

Stated before any measurement, from the analytic cost model below.

**PREDICTION: speedup_vs_dp = 8.0**

Reasoning:
- Baseline FNV-1a: critical path `xor(1) + imul(3)` per byte, unvectorisable → **≈4.0 cycles/byte**.
- One-sheet path (`len < 256`): 2 SipRounds per 8 bytes, SipRound critical path ≈6 cycles scalar → **≈1.5 c/B** → ~2.7×.
- Four-sheet AVX2 path: per 32-byte dealt block = 1 `vmovdqu` + 2 `vpxor` + one vector SipRound (4 `vpaddq`, 4 `vpxor`, 2 `vpshufd`, 1 `vpshufb`, 3×(2 shifts + or)) ≈ 27 µops. Port-throughput bound ≈9 cycles; latency bound ≈9 cycles. Both agree at **≈0.28 c/B** → ~14× ideal. Derating for imperfect scheduling, the ≈300-cycle merge, and any sub-KB sizes in the benchmark mix: **~8×**.
- Avalanche: I predict **0.49–0.51** (essentially ideal) on both paths. Every output bit is a SipHash PRF output; the shortest diffusion path for any input bit is ≥4 SipRounds plus a full SipHash-2-4 merge, and 4 SipRounds already achieve full diffusion.

## MEASUREMENT

I have no tools in this session — `hash_bench` and `hash_contract` are unavailable, so **I did not measure, and I will not pretend otherwise.** What the pipeline should find, stated as falsifiable predictions:

| Quantity | Predicted |
|---|---|
| throughput, large buffers (≥4 KB, AVX2) | 10–15 GB/s; speedup 7–13× |
| throughput, 256 B–1 KB | 3–6× (merge overhead amortising) |
| throughput, <256 B | 2.5–3× (one-sheet path) |
| avalanche score | 0.49–0.51 |
| any size where it loses to FNV-1a | none — the 256-byte guard is at analytic break-even |

Ways this prediction can fail, named in advance: if `-march=native` yields no AVX2, the `#else` path gets only SSE2 (2-wide) and the large-path speedup falls to ~4×, not 8×. If the benchmark's avalanche probe is dominated by very short inputs, the measured avalanche should be *unchanged* (SipHash-2-4 is a PRF at every length) — if it is not ≈0.5, my mapping of "scrambled on purpose" onto the SipRound constants is wrong, not merely slow.

## VERDICT

The kernel's core is now literally mechanism 2: **a four-word state in which every single input word is pressed against all four words and refolded until all four have moved, before the next word may be read** — plus mechanism 3 as the finalisation (the length as the one small suitcase, the thinning rounds, the XOR-fold to one coin, nothing else surviving) and mechanism 1's serial crease chain preserved inside each sheet. There is no multiplication anywhere in the mixing path; folds are rotations. The one place the mapping coarsens, stated honestly: a "mark" became an 8-byte word rather than a single byte, because a sheet's fold absorbs one machine word of marks at a time — every byte still enters its crease, no byte is skipped, and crease *n* still depends on crease *n−1*.

Taken seriously, the metaphor did not produce a novel mixer; it produced **SipHash** — which is the right outcome under step 4, since SipHash is a validated, widely deployed construction built for exactly the reason the native gives ("a clean line-up would mean two different piles could end up looking the same"). What I contributed is the fast realisation of it: the four birds become four AVX2 registers, so the metaphor's own "sheets, plural, large enough" regime test turns into a 4-way-dealt SIMD path.

The named risk — the four-sheet path costs ~300 cycles of setup and merge, so it loses on small inputs — is addressed, not shipped: the `SHEET_MIN = 256` check routes small piles to the plain one-sheet serial fold, at the exact analytic break-even. No thread parallelism was added; the sheets are 64-byte units, orders of magnitude too small to pay for a thread at these sizes.

Where this is honestly worse than the known way: xxHash/FNV spend far fewer operations per byte in absolute terms, and a pure-throughput kernel with no collision-resistance requirement could beat this. If `hash_bench` scores throughput alone and ignores avalanche quality, a 4-lane multiply hash would likely win. I am betting the avalanche half of the score on a construction the native actually described, instead of silently returning to the multiply-shift accumulator I was told not to.