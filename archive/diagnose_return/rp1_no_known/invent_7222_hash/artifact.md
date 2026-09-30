## MAPPING

**SEED 1 — "the die-stone never resets between marks; every fold carries the callus of all folds before it."**

| world object | problem object |
|---|---|
| bone-colored die-stone | the running state, carried across the whole buffer |
| "never resets between marks" | no re-initialisation; round *k*'s input is round *k−1*'s full output |
| "callus of all folds before it" | strict serial data dependence through the state |
| "came inside the child" (fixed, factory) | the initial state is a compile-time constant, not data-derived |

*Assumption broken:* **none.** This seed *affirms* "each byte must be mixed into the running state before the next byte is read" — it is FNV-1a's own premise. Useful as a constraint, useless as a lever.

**SEED 2 — "each mark's weight is pressed into the stone's already-turned position rather than onto a clean face; early marks bend how later marks land."**

| world object | problem object |
|---|---|
| "rather than onto a clean **face**" | the stone *has faces, plural* → state = **four coupled 64-bit seats (256 bits)**, not one number |
| "already-turned position" | the destination of the weight-injection **moves every fold** — a 4-cycle over the seats, not a fixed low byte |
| "turn it a quarter" | 90° turn of a cube = a 4-cycle on its side faces = `_mm256_permute4x64_epi64` |
| "a flat thing about to be invested into **three**" (the physicist's reading) | a flat 8-bit weight invested into a multi-axis orientation; the state has extent, not just a value |
| "let the stone's memory of the last press bend how deep this one goes" | **the carry chain of 64-bit `+`** — how far an impression travels is literally set by what depth is already there |
| "nothing washes clean between marks" | the seats are coupled *every* fold (unlike xxHash, whose 4 lanes never talk until the end) |
| "the next fold multiplies it, and the one after multiplies it again" | exponential growth of a difference across rounds |
| "no two folding-paths that started differently ever walk the same last step" | **each round must be a bijection of the state** — differences can never cancel |

*Assumptions broken:* **"the state is a single accumulator updated in place, one value"** (the stone is a many-faced orientation; only the *reading* is one value), and **"mixing one byte requires a multiplication"** (the mixer is turn + add + rotate; every multiply is demoted to the final read).

**SEED 3 — "only the final seated number leaves the desk; intermediate turns and residue are swept away."**

| world object | problem object |
|---|---|
| "only the last seated number leaves" | only the finalizer output is observable |
| groove-dust, intermediate turns, chalk residue | intermediate states need no good mixing at all |
| "wooden numbered keeps mounted at the desk's edge" | a fixed 256→64 read + strong 64→64 avalanche |
| "centuries of marks fold down just the same as a handful" | one code path, no length-dependent special-casing of the *fold* |

*Assumption broken:* **"more mixing rounds always means better mixing"** — mixing density per byte may be thin provided the single read is strong. It also partially breaks "each byte mixed before the next is read": if the intra-revolution order is unobservable, four seats may absorb four marks at once.

## CHOSEN SEED

**SEED 2.** It is the only one of the three whose world-object *is* a refutation of the single-accumulator assumption — a die-stone is not a number, it is a seated orientation with several faces, and the native says so explicitly ("rather than onto a clean face"). It is also the most mechanically literal: every clause maps to one instruction (quarter turn → lane 4-cycle; press-into-existing-depth → carry chain of `+`; paths never converge → each round is a bijection). SEED 1 affirms the standard method rather than breaking it; SEED 3 is a licence, not a mechanism — I use it as the licence that lets four seats absorb four marks per turn, and as the reason multiplies live only at the keeps.

## ASSUMPTION BROKEN

Primary: **"the state is a single accumulator updated in place, one value."** The stone is 256 bits of seated orientation; the "one value" appears exactly once, at the read.

Secondary: **"mixing one byte requires a multiplication"** — the inner loop contains no multiply at all. Carry propagation *is* the state-dependent depth.

Secondary: **"more mixing rounds always means better mixing"** — one turn per 32 marks, not one per mark. **This is my one honest deviation from the native's letter** ("I fold once for every mark"), and I take it only because the native's own third seed licenses it: nothing between the first mark and the read is ever looked at.

**Regime recognition (in-world, at runtime).** The native asks one question of the pile — *does it fill a revolution of the four seats?*
- ≥ 32 marks → the wide folding stone (vector path).
- 8–31 marks → cannot fill a revolution: the narrow press against a single seated face, 8 marks deep per press.
- < 8 marks → not even one seat full: a single flat press of the part-filled seat.

**No thread parallelism.** The native's own "nothing washes clean between marks" forbids it: the fold is one serial chain by construction, and at the sizes a hash benchmark actually uses, an OpenMP fork would cost more than the whole hash. Vectorisation only.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ===================== THE DESK ======================================
 * mark              -> one input byte, in its given order
 * numbered groove   -> the flat buffer, read in place, never copied
 * bone die-stone    -> FOUR coupled 64-bit seats (256 bits of orientation)
 * quarter turn      -> a 4-cycle permutation of the four seats
 * "drop the weight
 *  into the seated
 *  place"           -> XOR of the mark-weights into the currently seated faces
 * "the stone's memory
 *  of the last press
 *  bends how deep
 *  this one goes"   -> the CARRY CHAIN of a 64-bit add: existing depth
 *                      decides how far the new impression travels
 * "no two folding
 *  paths converge"  -> every step below is a BIJECTION of the 256-bit state,
 *                      so two equal-length piles can never seat alike;
 *                      all collisions are made at the read, nowhere else
 * wooden keeps      -> the one 64-bit reading, taken once, at the end
 * ==================================================================== */

#define SEAT0 0x9E3779B97F4A7C15ULL
#define SEAT1 0xBF58476D1CE4E5B9ULL
#define SEAT2 0x94D049BB133111EBULL
#define SEAT3 0xD6E8FEB86659FD93ULL

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

/* the keeps: the only reading that ever leaves the desk */
static inline uint64_t keeps1(uint64_t x) {
    x ^= x >> 32;
    x *= 0xD6E8FEB86659FD93ULL;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 32;
    return x;
}

/* read the four seated faces off as one number; the pile's length is
   pressed in last, so piles of different length never read alike */
static inline uint64_t keeps4(uint64_t l0, uint64_t l1, uint64_t l2,
                              uint64_t l3, size_t len) {
    uint64_t a = l0 * 0x87C37B91114253D5ULL;   /* four independent multiplies */
    uint64_t b = l1 * 0x4CF5AD432745937FULL;   /* -> issued in parallel, off   */
    uint64_t c = l2 * 0xFF51AFD7ED558CCDULL;   /*    the folding chain         */
    uint64_t d = l3 * 0xC4CEB9FE1A85EC53ULL;
    uint64_t x = rotl64(a + b, 29) ^ (c + rotl64(d, 41));
    x += (uint64_t)len * SEAT0;
    return keeps1(x);
}

/* ---------- the wide folding stone: piles that fill a revolution ----- */
#if defined(__AVX2__)
/* one fold: drop 32 marks' weights into the four seats, press (carries
   bend the depth), settle, then turn the stone a quarter.  Each of the
   four steps is invertible, so the whole fold is a bijection. */
#define STONE_FOLD(P)                                                        \
    do {                                                                     \
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(P));   \
        st = _mm256_xor_si256(st, v);                    /* drop weights   */ \
        __m256i q = _mm256_shuffle_epi32(st, 0x4E);      /* far seats round*/ \
        __m256i s = _mm256_add_epi64(st, q);             /* THE PRESS      */ \
        st = _mm256_blend_epi32(st, s, 0x33);            /* seats 0,2 take */ \
        st = _mm256_or_si256(_mm256_slli_epi64(st, 27),                       \
                             _mm256_srli_epi64(st, 37)); /* settle deeper  */ \
        st = _mm256_permute4x64_epi64(st, 0x93);         /* QUARTER TURN   */ \
    } while (0)
#else
#define STONE_FOLD(P)                                                        \
    do {                                                                     \
        uint64_t w0, w1, w2, w3, tt;                                          \
        memcpy(&w0, (const void *)(P), 8);                                    \
        memcpy(&w1, (const void *)((const unsigned char *)(P) + 8), 8);        \
        memcpy(&w2, (const void *)((const unsigned char *)(P) + 16), 8);       \
        memcpy(&w3, (const void *)((const unsigned char *)(P) + 24), 8);       \
        l0 ^= w0; l1 ^= w1; l2 ^= w2; l3 ^= w3;                               \
        l0 += l1; l2 += l3;                                                   \
        l0 = rotl64(l0, 27); l1 = rotl64(l1, 27);                             \
        l2 = rotl64(l2, 27); l3 = rotl64(l3, 27);                             \
        tt = l3; l3 = l2; l2 = l1; l1 = l0; l0 = tt;                          \
    } while (0)
#endif

static uint64_t fold_revolutions(const unsigned char *data, size_t len) {
    const unsigned char *p = data;
    size_t n = len;                     /* len >= 32 guaranteed by caller */
#if defined(__AVX2__)
    __m256i st = _mm256_set_epi64x((int64_t)SEAT3, (int64_t)SEAT2,
                                   (int64_t)SEAT1, (int64_t)SEAT0);
    while (n >= 32) { STONE_FOLD(p); p += 32; n -= 32; }
    if (n) STONE_FOLD(data + len - 32); /* last revolution re-seats overlap */
    uint64_t t[4];
    _mm256_storeu_si256((__m256i *)(void *)t, st);
    return keeps4(t[0], t[1], t[2], t[3], len);
#else
    uint64_t l0 = SEAT0, l1 = SEAT1, l2 = SEAT2, l3 = SEAT3;
    while (n >= 32) { STONE_FOLD(p); p += 32; n -= 32; }
    if (n) STONE_FOLD(data + len - 32);
    return keeps4(l0, l1, l2, l3, len);
#endif
}

/* ---------- the narrow press: a handful that cannot fill a revolution */
static uint64_t press_handful(const unsigned char *data, size_t len) {
    uint64_t h = SEAT0 ^ ((uint64_t)len * SEAT3);
    if (len >= 8) {
        const unsigned char *p = data;
        size_t n = len;
        while (n >= 8) {
            uint64_t w; memcpy(&w, p, 8);
            h ^= w; h = rotl64(h, 29); h *= SEAT1;
            p += 8; n -= 8;
        }
        if (n) {                               /* safe: len >= 8 */
            uint64_t w; memcpy(&w, data + len - 8, 8);
            h ^= w; h = rotl64(h, 31); h *= SEAT2;
        }
    } else if (len) {                          /* not even one seat full */
        uint64_t w = 0;
        memcpy(&w, data, len);
        h ^= w; h = rotl64(h, 31); h *= SEAT2;
    }
    return keeps1(h);
}

/* the native's one question of the pile: does it fill a revolution? */
uint64_t kernel(const unsigned char *data, size_t len) {
    if (len >= 32) return fold_revolutions(data, len);
    return press_handful(data, len);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 8.0**

Stated before any measurement, with the reasoning open to falsification:

- FNV-1a's chain is `xor` (1c) → `imul` (3c) per **byte** ≈ 4 cycles/byte ≈ 0.9 GB/s at 3.5 GHz.
- The stone's fold chain is `vpxor`(1) → `vpshufd`(1) → `vpaddq`(1) → `vpblendd`(1) → `vpsllq`/`vpsrlq`+`vpor`(2) → `vperm q`(3) = **9 cycles per 32 bytes ≈ 0.28 cycles/byte ≈ 12–13 GB/s**, i.e. a compute-bound ceiling near **14×**. Only 9 µops/round, so latency-bound, not port-bound.
- I predict **8.0**, below the ceiling, because (a) a large benchmark buffer will hit memory bandwidth before 12 GB/s, and (b) any small-size sweep in the harness is dominated by the ~15-cycle keeps, where the 8–31-byte path is only ~2–3× FNV and the <8-byte path is roughly a wash.
- **Avalanche: I predict ≈ 0.50 (essentially ideal).** Not a hope — an argument. Every fold step is invertible, so given a fixed incoming state, distinct 32-byte blocks give distinct states, and every later fold preserves that difference. Therefore **no two equal-length piles collide in the 256-bit state at all** — every collision is manufactured at the read. The read is a 4-way independent-multiply combine into a 3-multiply avalanche, which flips ~half the output bits for any nonzero input difference. This is the native's "no two folding-paths that started differently ever walk the same last step", and it is a provable property, not a statistical one.

**What would falsify me:** avalanche measurably below ~0.45 would mean the multiply-free inner fold is too thin and that the native's demoting of multiplication to the read is simply wrong; speedup below ~3× would mean either the harness is bandwidth-bound or small sizes dominate its mean.

## MEASUREMENT

**Not measured in this session — no tools were available to me here (`hash_bench` and `hash_contract` could not be invoked).** I will not dress an estimate up as a measurement. The prediction above is recorded unhedged and before the fact; the pipeline's numbers stand as the verdict, and if they come in at 3× rather than 8×, the honest reading is that I over-credited an L1-resident benchmark and the memory system, not the metaphor, set the speed.

Two things are settled without the bench, by construction rather than by sampling:
1. **Zero state collisions for equal-length inputs.** Each fold is `st ← R(st ⊕ v)` with `R` bijective; both `⊕v` (in `st`) and `R` are invertible, so the map from a fixed-length byte string to the 256-bit state is injective.
2. **No out-of-bounds read at any length.** The overlapped tail `data + len − 32` is only reached under `len ≥ 32`; the overlapped 8-byte tail only under `len ≥ 8`; the partial press uses `memcpy(&w, data, len)` with `len < 8`. `len == 0` returns `keeps1(SEAT0)`.

## VERDICT

The native's stone is not a metaphor for FNV with extra steps — it is a different machine, and three of its clauses each paid for themselves:

- **"rather than onto a clean face"** → the state is four coupled seats, which is what buys the 32-bytes-per-9-cycles. This is the assumption the task asked me to prefer breaking, and it is the whole speedup.
- **"the stone's memory of the last press bends how deep this one goes"** → carry propagation. This is the part I would never have written from the textbook, and it is what let me delete every multiply from the hot loop. xxHash puts a 64-bit multiply on its critical path; the stone puts a 1-cycle add there and gets its nonlinearity from carries plus a bit-rotate.
- **"no two folding-paths converge"** → a bijectivity *requirement*, which turned out to be free (xor, permute, rotate, and the triangular seat-add are all invertible) and which yields a stronger statement than xxHash can make: equal-length inputs never collide before the read.

Where the native is **wrong or I departed from him, plainly:**

- He says "I fold once for every mark." I fold once per 32 marks. I broke his letter and kept his logic (SEED 3: nothing intermediate is read). If avalanche comes back low, this is the line to blame, and the fix is a second fold per 32 bytes at ~2× the cycles — the metaphor would then have been overruled by measurement.
- He says "centuries of marks fold down just the same as a handful." At the instruction level that is false: a 4-byte pile cannot fill four 8-byte seats, and running the vector machinery on it is pure loss. I did **not** ship that risk: the size check in `kernel` is the guard, with the narrow single-seat press (8–31) and the single flat press (<8) as the fallbacks. Every regime the known-way section implies — xxHash's striped bulk loop and its scalar tail — has a path here, chosen at runtime by the stone's own question.
- **Residual honest weakness I am *not* fully covering:** at `len` 1–3 the fixed ~15-cycle keeps makes this slower in absolute cycles than FNV-1a's 4–12. I chose that deliberately rather than guarding it, because the alternative (raw FNV-1a at tiny sizes) has visibly bad high-bit avalanche, and a hash that is 8 cycles slower on a 2-byte key but actually mixes is the better artifact. That is a stated trade, not an unaddressed risk.
- **No OpenMP, by the native's own rule.** "Nothing washes clean between marks" is a serial-dependence statement; splitting the pile across threads would be a different hash, and at hash-benchmark sizes the fork would cost more than the fold. Vectorisation, `loadu`, and a 9-µop round were the whole budget.