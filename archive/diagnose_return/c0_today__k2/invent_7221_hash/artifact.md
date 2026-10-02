# MAPPING

**SEED 1 — "one crease per mark, angle set by the mark *and* the crease before it"**

| World object | Problem object |
|---|---|
| mark in the pile, read in order | input byte `data[i]`, read at stride 1 |
| sheet laid flat | the running state |
| one fold per mark | one state update per byte |
| angle = f(mark, previous crease) | `h = f(h, data[i])` — strict serial recurrence |
| never looking ahead | no lookahead, no reordering |

**Assumption broken: none.** This seed *is* the textbook method. "Angle set by the mark and the crease before it" is literally `h ^= b; h *= prime;` — it asserts that each byte must be mixed before the next is read, and asserts a single in-place accumulator. It confirms both silent assumptions rather than breaking any.

**SEED 2 — "test every crease against all four wire birds' beaks at once, refold until they agree"**

| World object | Problem object |
|---|---|
| the one sheet | one 256-bit accumulator register |
| four wire birds | four independent 64-bit accumulator lanes |
| each bird's own beak | a lane-distinct 64-bit key constant |
| a fold's four corners, one per beak | a 32-byte stripe split into 4×8-byte words, word *i* → lane *i* |
| "pressed against all four **at once**" | one SIMD instruction touches all four lanes in the same cycle |
| "only when all four agree is the crease set" | the final convergence step must consume all four lanes before an output exists |
| "spiral and snail markings made to line up **wrong on purpose, scrambled**, because a clean line-up would mean two piles look the same" | a deliberate cross-lane permutation of the raw data added into the accumulator — present *precisely* to kill the collisions that a symmetric, cleanly-aligned lane layout would create |
| "refold tighter until they agree" | the beak key advances every fold, so no two folds are keyed alike → block order and block position matter |
| "pink or orange, doesn't matter, only that it's **large enough**" | runtime regime test on `len`; a pile too small for a huge sheet is folded on a scrap instead |

**Assumptions broken: "the state is a single accumulator updated in place, one value"** *and* **"each byte must be mixed into the running state before the next byte is read"** — four corners of the same fold are in the beaks simultaneously; byte 8 is being pressed while byte 0 is still being pressed. Also breaks "the whole buffer must be read once, start to end, in order" in the weak sense that reading is 32-byte-granular, not byte-granular and not serially dependent.

**SEED 3 — "thin the wad at the water's edge to one dense corner; throw away every scrap and misreading"**

| World object | Problem object |
|---|---|
| carrying the reduced wad to the water | finalization, after the loop |
| thinning "the way a body thins going under" | a strong, irreversible-looking bit-avalanche finalizer (`fmix64`) |
| one hard dense corner no bigger than a coin | the returned `uint64_t` |
| every used sheet, trimmed scrap, failed reading thrown in the mud | no state survives the call; nothing but 64 bits is retained |
| "if even one mark had been different, the whole spiral would have folded to a different, unrecognizable corner" | the avalanche requirement itself |

**Assumption broken: "more mixing rounds always means better mixing"** and **"mixing one byte requires a multiplication."** The native does *cheap* folding per mark and concentrates all the hard work in one thinning at the end — mixing effort is relocated, not multiplied per byte.

# CHOSEN SEED

**SEED 2 — the four wire birds.** It is the only seed that breaks the preferred assumption ("each byte must be mixed into the running state before the next byte is read"), it is the furthest from the stated known way (which folds "the whole buffer through *one* accumulator"), and its mapping is mechanical rather than interpretive: four birds → four lanes, four corners → four 8-byte words, "at once" → one vector instruction, "scrambled on purpose" → a cross-lane permute.

SEED 3 is not discarded — the native's procedure is one continuous act, and the water's edge is where this kernel's finalizer comes from. SEED 1 survives only as the scrap path for piles too small for a sheet (where it is exactly the baseline, so that path can never be slower than the baseline).

**Letting the mechanism land on validated technique (step 4).** Followed honestly, the four birds do not produce an invention. `acc += permute(data)` plus `acc += lo32(data^key) * hi32(data^key)`, summed over stripes with a per-stripe key, *is* the XXH3 SIMD accumulate step, which is itself the NH/VMAC universal-hash form (a sum of products of key-offset word pairs). The "all four must agree" convergence is xxHash64's `mergeRound`. The water's-edge thinning is MurmurHash3's `fmix64`. I let the metaphor arrive at these three published, validated primitives rather than hand-rolling substitutes for them.

# ASSUMPTION BROKEN

- **Primary:** *each byte must be mixed into the running state before the next byte is read.* Four 8-byte words are absorbed in the same instruction; the accumulator's critical path per 32 bytes is two 1-cycle vector adds, not 32 serial multiply-xors. The products and the permute are computed **off** the critical path.
- **Secondary:** *the state is a single accumulator, one value* → four lanes in one register.
- **Secondary:** *mixing one byte requires a multiplication* → per stripe there is one 32×32 vector multiply shared by four lanes, and the 64-bit multiplies appear only four times total, in the finalization.

**Regime handling (step 5).** The known-way section describes two regimes (per-byte serial mixing; "the whole buffer read start to end"), which split at the point where vector setup + a 4-lane convergence is cheaper than a serial byte chain. The native already encodes the test — the sheet must be "large enough," else there is no sheet, only a scrap. So: `len < 32` → **scrap path** (one bird, SEED-1 carry-forward, i.e. the baseline recurrence over 8-byte words plus the thinning). `len >= 32` → **four-bird path**. This is also the guard demanded by step 4 for my own stated risk (fixed-cost convergence hurting tiny inputs): the risky path is never entered for tiny inputs, and the fallback is the known-good method itself.

**Threads: deliberately not used.** The metaphor has one folder, one sheet, and four gauges — the four birds are lanes, not folders; there is no second person in the world. At 16 B/cycle the single-thread kernel is already DRAM-bandwidth bound, so OpenMP would be pure spawn overhead at any plausible benchmark size. (Note: because the absorb step is a pure sum with an index-derived key, a reduction would be *bit-identical* if ever needed — but shipping an unmeasured threshold is exactly the risk step 4 tells me to drop, so it is dropped.)

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ----- the four birds' beaks, and the constants of the water's edge ----- */
#define BEAK1        0x9E3779B185EBCA87ULL
#define BEAK2        0xC2B2AE3D27D4EB4FULL
#define BEAK3        0x165667B19E3779F9ULL
#define BEAK4        0x85EBCA77C2B2AE63ULL
#define COUNT_PRIME  0x27D4EB2F165667C5ULL

/* each bird refolds tighter after every fold: its beak advances, so no two
   folds are gauged alike -> stripe position and stripe order both matter. */
#define STEP0 0x9E3779B97F4A7C15ULL
#define STEP1 0xBF58476D1CE4E5B9ULL
#define STEP2 0x94D049BB133111EBULL
#define STEP3 0x2545F4914F6CDD1DULL

#define KEY0  0xDA942042E4DD58B5ULL
#define KEY1  0x9FB21C651E98DF25ULL
#define KEY2  0xEB44ACCAB455D165ULL
#define KEY3  0xA0761D6478BD642FULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* SEED 3: the water's edge. The whole wad thins down to one dense corner.
   MurmurHash3 fmix64 -- a validated, bijective 64-bit avalanche. */
static inline uint64_t thin_at_the_water(uint64_t h) {
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return h;
}

/* "only when all four beaks agree does the crease count as set":
   no output exists until every lane has been folded in.
   This is xxHash64's mergeRound -- validated. */
static inline uint64_t beak_agree(uint64_t h, uint64_t lane) {
    lane *= BEAK2;
    lane  = rotl64(lane, 31);
    lane *= BEAK1;
    h ^= lane;
    h  = h * BEAK1 + BEAK4;
    return h;
}

/* ----- REGIME 2: the pile is too small for a huge sheet, so fold on a
   scrap with a single bird. This is SEED 1 / the known way verbatim
   (one in-place accumulator, carry-forward crease), widened to 8-byte
   words, then thinned. It can never be slower than the baseline. ----- */
static uint64_t fold_on_scrap(const unsigned char *d, size_t len) {
    uint64_t h = 1469598103934665603ULL ^ ((uint64_t)len * COUNT_PRIME);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t w;
        memcpy(&w, d + i, 8);
        h ^= w;
        h *= 1099511628211ULL;
        h  = rotl64(h, 29);              /* fold high bits back down */
    }
    for (; i < len; i++) {
        h ^= (uint64_t)d[i];
        h *= 1099511628211ULL;
    }
    return thin_at_the_water(h);
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* "pink or orange, doesn't matter, only that it's large enough":
       the runtime regime test. No sheet -> no birds. */
    if (len < 32u) return fold_on_scrap(data, len);

    const unsigned char *restrict p = data;
    const size_t nfold = len >> 5;       /* one fold per 32 bytes */
    const size_t rem   = len & 31u;      /* the ragged trimmed scrap */

    /* the ragged tail is padded onto its own scrap, with the count of
       marks pressed into the last corner so that padding cannot lie */
    unsigned char tail[32];
    if (rem) {
        memset(tail, 0, sizeof tail);
        memcpy(tail, p + (nfold << 5), rem);
        tail[31] = (unsigned char)rem;
    }

#if defined(__AVX2__)
    /* THE FOUR WIRE BIRDS: four 64-bit lanes of one register, pressed at once. */
    __m256i acc  = _mm256_set_epi64x((int64_t)(BEAK4 ^ (uint64_t)len),
                                     (int64_t)BEAK3,
                                     (int64_t)BEAK2,
                                     (int64_t)BEAK1);
    __m256i beak = _mm256_set_epi64x((int64_t)KEY3, (int64_t)KEY2,
                                     (int64_t)KEY1, (int64_t)KEY0);
    const __m256i step = _mm256_set_epi64x((int64_t)STEP3, (int64_t)STEP2,
                                           (int64_t)STEP1, (int64_t)STEP0);

#if defined(__GNUC__)
#pragma GCC unroll 2
#endif
    for (size_t i = 0; i < nfold; i++, p += 32) {
        __m256i d  = _mm256_loadu_si256((const __m256i *)p);   /* one fold, 4 corners */
        __m256i dk = _mm256_xor_si256(d, beak);                /* corner meets beak   */
        __m256i sw = _mm256_shuffle_epi32(dk, 0xB1);           /* hi32 <-> lo32       */
        __m256i pr = _mm256_mul_epu32(dk, sw);                 /* lo32 * hi32, 4 lanes*/
        /* "lined up wrong on purpose, scrambled, because a clean line-up
            would mean two different piles could look the same" */
        __m256i ds = _mm256_permute4x64_epi64(d, 0x4E);
        acc  = _mm256_add_epi64(acc, ds);                      /* critical path: 1 cy */
        acc  = _mm256_add_epi64(acc, pr);                      /* critical path: 1 cy */
        beak = _mm256_add_epi64(beak, step);                   /* refold tighter      */
    }
    if (rem) {
        __m256i d  = _mm256_loadu_si256((const __m256i *)tail);
        __m256i dk = _mm256_xor_si256(d, beak);
        __m256i sw = _mm256_shuffle_epi32(dk, 0xB1);
        __m256i pr = _mm256_mul_epu32(dk, sw);
        __m256i ds = _mm256_permute4x64_epi64(d, 0x4E);
        acc = _mm256_add_epi64(acc, ds);
        acc = _mm256_add_epi64(acc, pr);
    }

    uint64_t l[4];
    _mm256_storeu_si256((__m256i *)l, acc);
#else
    /* Same arithmetic, four birds held in four scalar registers.
       Produces a bit-identical hash to the AVX2 path. */
    uint64_t l[4];
    uint64_t a0 = BEAK1, a1 = BEAK2, a2 = BEAK3, a3 = BEAK4 ^ (uint64_t)len;
    uint64_t k0 = KEY0,  k1 = KEY1,  k2 = KEY2,  k3 = KEY3;
    for (size_t i = 0; i <= nfold; i++) {
        const unsigned char *q;
        if (i < nfold)      q = p + (i << 5);
        else if (rem)       q = tail;
        else                break;
        uint64_t w0, w1, w2, w3;
        memcpy(&w0, q,      8); memcpy(&w1, q +  8, 8);
        memcpy(&w2, q + 16, 8); memcpy(&w3, q + 24, 8);
        uint64_t d0 = w0 ^ k0, d1 = w1 ^ k1, d2 = w2 ^ k2, d3 = w3 ^ k3;
        a0 += w2; a1 += w3; a2 += w0; a3 += w1;          /* the scramble */
        a0 += (uint64_t)(uint32_t)d0 * (uint64_t)(uint32_t)(d0 >> 32);
        a1 += (uint64_t)(uint32_t)d1 * (uint64_t)(uint32_t)(d1 >> 32);
        a2 += (uint64_t)(uint32_t)d2 * (uint64_t)(uint32_t)(d2 >> 32);
        a3 += (uint64_t)(uint32_t)d3 * (uint64_t)(uint32_t)(d3 >> 32);
        if (i < nfold) { k0 += STEP0; k1 += STEP1; k2 += STEP2; k3 += STEP3; }
    }
    l[0] = a0; l[1] = a1; l[2] = a2; l[3] = a3;
#endif

    /* all four beaks must agree before the crease counts as set */
    uint64_t h = (uint64_t)len * COUNT_PRIME
               + rotl64(l[0],  1) + rotl64(l[1],  7)
               + rotl64(l[2], 12) + rotl64(l[3], 18);
    h = beak_agree(h, l[0]);
    h = beak_agree(h, l[1]);
    h = beak_agree(h, l[2]);
    h = beak_agree(h, l[3]);

    /* carry the wad to the water's edge; keep only the coin, throw the
       sheets, the trimmings and the failed readings into the mud */
    return thin_at_the_water(h);
}
```

Four design revisions were made *on paper* before freezing this (I could not measure between them — see MEASUREMENT, and I am not going to pretend these were empirical):
1. **Rejected the most naive literal reading** — "press *each* crease against *all four* beaks" = every byte into all four lanes. Four independent chains, but still one byte per multiply-latency → ~0 speedup at 4× the work. Re-read "each fold's **corner** against a different bird's beak": a fold has four corners, one per bird. That is partition, not replication, and it is the more literal reading of the sentence.
2. **Removed the multiply from the critical path.** `acc = rotl(acc + d, c)` gave a ~3-cycle chain per 32 B (≈10 B/cy). Switching to two independent `acc +=` terms (the NH/XXH3 form) drops the chain to 2 cycles (≈16 B/cy) and pushes the multiply and the permute off-path entirely.
3. **Fixed an order-insensitivity bug in my own design.** A pure sum with a *constant* beak vector is commutative across stripes — permuting 32-byte blocks would collide. Fixed by advancing the beak every fold (`beak += step`), which the native already states ("refold tighter until they agree"). Cost: one off-path vector add.
4. **Added the small-input regime and the length injection.** Without the `len < 32` scrap path, tiny buffers pay the full 4-lane convergence; without `len` and the `tail[31] = rem` marker, zero-padding makes `"ab"` and `"ab\0"` collide.

# PREDICTION

Fixed before any measurement, and no measurement has been taken at the time of writing.

**PREDICTION: speedup_vs_dp = 22**

Reasoning: the FNV-1a baseline is latency-bound at xor(1) + imul(3) ≈ 4 cycles/byte ≈ 0.25 B/cycle. This kernel's accumulator chain is two 1-cycle vector adds per 32-byte stripe ≈ 16 B/cycle, with ~8 uops/stripe so issue width is not the binder either. Ratio ≈ 64×, but large buffers will clip at L3/DRAM bandwidth (~20–40 GB/s vs. the baseline's ~0.9 GB/s), so the realized figure should land in the 15–35× band; 22× is my point estimate for a typical multi-hundred-KB benchmark buffer. For `len < 32` the scrap path should measure ≈1.0× (by construction — it *is* the baseline), so a benchmark dominated by tiny inputs would legitimately report ≈1.

Avalanche: **≈0.50 flipped-bit fraction (near-ideal)**. Every single-bit input flip changes `acc` by ±2^j in some lane via the raw-data `acc += ds` term, *independently* of whether the 32×32 product degenerates (the degenerate case — one 32-bit half being zero — is exactly why that term exists, and is exactly the native's "scrambled on purpose"). That lane then passes through `beak_agree` (×, rotl 31, ×) and `fmix64`, both full-width avalanchers.

Falsifiers I would accept: measured throughput below ~5 GB/s would mean the two-add chain is not what I think it is (likely a spilled `acc` or a non-unrolled `beak` chain); an avalanche score below ~0.45 would mean the single-lane difference is not surviving the convergence, and the fix would be a second `beak_agree` pass, not more per-byte work.

# MEASUREMENT

**Not performed. No tools were available in this session** — `hash_bench`, `hash_contract` and the symbolic/unconventional tools were listed but are not callable here, and there is no shell in this environment either. I am reporting that plainly rather than inventing numbers: everything in the PREDICTION section is analysis of instruction latency and diffusion structure, not data. The pipeline will compile and measure the artifact; the prediction above (22×, avalanche ≈0.50) is on the record and unhedged, and the falsifiers are stated so a miss is a miss.

Static checks I *can* vouch for by inspection:
- Signature matches the contract exactly: `uint64_t kernel(const unsigned char *data, size_t len)`.
- No reads past `data + len`: the stripe loop covers `32 * (len >> 5)` bytes, the remainder is `memcpy`'d into a 32-byte stack buffer. No overlapping "read the last 32 bytes" trick is used, so `len == 32` and unmapped-page-adjacent buffers are safe.
- `len == 0` → `fold_on_scrap(data, 0)` → no dereference, returns `thin(basis)`.
- All intrinsics used are AVX2: `vpxor`, `vpshufd`, `vpmuludq`, `vpermq`, `vpaddq`. No AVX-512, no `_mm256_mullo_epi64` (which does not exist in AVX2).
- `rotl64(x, r)` is only ever called with `r` in 1..31, so no UB from a 64-bit shift.
- AVX2 and scalar paths compute identical values on little-endian x86 (lane *i* ↔ bytes 8*i..8*i+7, and `0x4E` ↔ `{d2,d3,d0,d1}`).

# VERDICT

The native's four wire birds translate, with no slack in the mapping, into **four accumulator lanes in one SIMD register, each with its own key, with a deliberate cross-lane scramble of the raw data added alongside a keyed 32×32 product** — and when you follow that literally and refuse to optimize the metaphor away, it lands on NH/VMAC-style universal hashing as realized by XXH3's SIMD accumulate, with xxHash64's merge and Murmur3's `fmix64` for the water's edge. That is the outcome I wanted: three validated primitives reached *by* the metaphor, not substituted for it. The seed's own asymmetry clause ("lined up wrong on purpose, because a clean line-up would mean two different piles could look the same") is not decoration — it is the term that guarantees avalanche when the product degenerates, and I would not have included it from the textbook description alone.

What the metaphor does **not** justify is a second folder, so there are no threads; the four birds are lanes, and the vectorization is where the speed comes from, as step 4 prefers.

**Named risks and how each is addressed, not deferred:**
- *Fixed convergence cost hurts small inputs.* Guarded: `len < 32` takes the scrap path, which is the baseline recurrence itself plus a finalizer — it cannot be materially slower than the known way, and it is a genuine second runtime regime, recognized by the native's own "only that it's large enough."
- *Pure summation would make stripes permutable.* Addressed in the artifact by the advancing beak, not left as a caveat.
- *Zero-padding ambiguity on the ragged tail.* Addressed by `tail[31] = rem` and by folding `len` into both the initial lane 3 and the pre-merge `h`.
- *Unaddressed by design, stated openly:* this is a non-cryptographic hash built on additive absorption. Single-bit avalanche should be excellent, but an adversary who can choose inputs can construct multi-word differences that cancel in the sum (the standard NH caveat — real VMAC answers this with a secret key, which a fixed-signature `kernel` has nowhere to put). If the benchmark's avalanche score is adversarial rather than random-single-bit, I expect this kernel to score worse than a serial multiply chain, and that would be a real defeat of the mechanism, not a tuning problem.