## DICTIONARY

| Recipe object | Computational object |
|---|---|
| a **mark** | one input byte `data[i]` |
| the **riverbank stone**, "the order on the stone is the only order" | the buffer read strictly in index order, byte 0 first; every load is big‑endian so index order = grid order on every machine |
| **servant swats the inflated house against your mouth** | no input byte ever reaches the output un-mixed; the only exit is the shrine stone |
| the **grid / shape / token** ("look at every square", "every corner or edge") | one `uint64_t` viewed as an **8×8 bit grid**: row *r* = byte *r* (row 0 = most significant byte), column *c* = bit *c* of that byte. 64 squares → "about half the positions must differ" = 32 output bits flip. This is why the token is 64 bits |
| a **heap** | 8 bytes = 64 bits = one 8×8 grid (it must match the running shape, because step 5 adds them "position against position") |
| **black pebble** | the `0x80` byte, always appended |
| **white pebbles** | zero bytes |
| **pebbles spelling how many marks** | `len` as 8 big-endian bytes |
| **even count of heaps** | exact multiple of 8 bytes |
| the **shrine / running shape** | the single live state variable `S` (one register; no history array — step 7 burns everything else) |
| **uphill twist** ("add… carrying nothing across positions") | `S ^ M` — carryless positionwise addition = XOR |
| **downhill fold: turn ninety degrees** | 90° rotation of the 8×8 bit grid = `transpose8x8(bswap64(x))`; on GFNI hardware, one `gf2p8affineqb` |
| **downhill fold: fold its corners into its own center** | fold the outer quarter of the grid inward so a corner lands on a centre row: rows{0,1} reflected are **added with carry** onto rows{3,2}; rows{7,6} reflected onto rows{4,5}. A fold presses material together → integer addition, not XOR |
| **doubling-back** ("add the result into the shape it was before the twist") | feed-forward `S = T + prev` (unqualified "add" ⇒ *with* carry, in deliberate contrast to the twist) |
| a **stride** | one iteration of {twist, fold, double-back} |
| **stride count, no fewer than eight, count them on your fingers** | `STRIDES = 10` (≥8, and a hand's worth) |
| **burn the heap and the old shape** | no block or state is retained; scratch wiped at the end |
| **the token** | the return value |

**Ambiguities, resolved to the most literal reading**

1. *Step 2* — the count needs 8 marks and a heap is 8 marks, so "the final heap's tail" is the whole final heap: the length gets its own last heap. The `0x80` is laid unconditionally (the recipe insists), so the padded pile is always `len` + (1..8) + 8 bytes.
2. *Step 5* — handedness of "ninety degrees" is not named; the scalar and GFNI paths implement the two opposite 90° turns. Both satisfy the step.
3. *Steps 9–11* are a validation ritual on the *method*, not part of producing a token; they are present as a real, compiled-out `deer_selftest()` rather than being run per call.
4. **No step had to be changed to make the answer correct.** One honest caveat I deliberately did *not* patch (patching would add a step the recipe doesn't have): `S = 0` with an all-zero heap is a fixed point. Step 2's mandatory black pebble and length heap make the final heap never zero, so it cannot be reached at the end of a pile.
5. OpenMP is deliberately unused: steps 5 and 8 mandate strictly "the leftmost heap that has not been raced", which forbids parallelism across heaps. Using it would replace the native's idea.

## ARTIFACT

```c
/* =====================================================================
   deer-stride hash  —  a literal transcription of the recipe.
   Contract:  uint64_t kernel(const unsigned char *data, size_t len);
   Build:     gcc -O3 -march=native -fopenmp -lm
   State = token = one 8x8 bit grid = uint64_t (row r = byte r, row 0 = MSB).
   ===================================================================== */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__GFNI__) && defined(__SSSE3__)
  #include <immintrin.h>
  #define DEER_GFNI 1
#endif

#define SHRINE_IV 0x6A09E667F3BCC908ULL /* step 4: not chosen today, never changed */
#define STRIDES   10                    /* step 6: >= 8, countable on ten fingers  */

/* the marks in the order they were given: row r of the grid = the r-th mark */
static inline uint64_t deer_load_be64(const unsigned char *p)
{
    return ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48)
         | ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32)
         | ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16)
         | ((uint64_t)p[6] <<  8) | ((uint64_t)p[7]);
}

/* --------------------------------------------------------------------
   THE DEER'S STRIDE (step 5), repeated STRIDES times (step 6).
   -------------------------------------------------------------------- */
#ifdef DEER_GFNI
typedef __m128i deer_t;
#define DEER_LIFT(u) _mm_cvtsi64_si128((long long)(uint64_t)(u))
#define DEER_DROP(v) ((uint64_t)_mm_cvtsi128_si64(v))

static inline deer_t deer_race(deer_t S, deer_t M)
{
    /* gf2p8affineqb(sel, X, 0) with sel = one set bit per byte yields
       dst[j][i] = X[7-i][j] : the transpose with reversed rows, i.e. a
       ninety-degree turn of the 8x8 grid, in one instruction.          */
    const __m128i sel = _mm_set1_epi64x((long long)0x8040201008040201ULL);
    /* the corner fold: byte lanes 4,5 <- lanes 7,6 and lanes 2,3 <- lanes 1,0,
       i.e. rows{0,1} reflected onto rows{3,2}, rows{7,6} onto rows{4,5};
       all other lanes zeroed, so the add below touches only the centre.  */
    const __m128i fld = _mm_setr_epi8((char)0x80,(char)0x80, 1, 0, 7, 6,
                                      (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                      (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                      (char)0x80,(char)0x80);
    int s;
    for (s = 0; s < STRIDES; ++s) {
        __m128i prev = S;                                   /* the shape before the twist */
        __m128i T = _mm_xor_si128(S, M);                    /* uphill twist: carryless     */
        T = _mm_gf2p8affine_epi64_epi8(sel, T, 0);          /* fold, part 1: ninety degrees*/
        T = _mm_add_epi16(T, _mm_shuffle_epi8(T, fld));     /* fold, part 2: corners->centre*/
        S = _mm_add_epi64(T, prev);                         /* doubling-back               */
    }
    return S;
}
#else
typedef uint64_t deer_t;
#define DEER_LIFT(u) ((uint64_t)(u))
#define DEER_DROP(v) ((uint64_t)(v))

static inline uint64_t deer_transpose8(uint64_t x)
{
    uint64_t t;
    t = (x ^ (x >>  7)) & 0x00AA00AA00AA00AAULL; x ^= t ^ (t <<  7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL; x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL; x ^= t ^ (t << 28);
    return x;
}
/* reverse the row order, then transpose = turn the grid ninety degrees */
static inline uint64_t deer_rot90(uint64_t x)
{
    return deer_transpose8(__builtin_bswap64(x));
}
/* fold the outer quarter of the grid inward, pressed with carry:
   rows 0,1 reflected land on rows 3,2 ; rows 7,6 land on rows 4,5.
   The read halves (a,d) are untouched, so the fold is a bijection.   */
static inline uint64_t deer_fold(uint64_t x)
{
    uint32_t hi = (uint32_t)(x >> 32), lo = (uint32_t)x;
    uint32_t a = hi >> 16, b = hi & 0xFFFFu;   /* a = rows 0,1   b = rows 2,3 */
    uint32_t c = lo >> 16, d = lo & 0xFFFFu;   /* c = rows 4,5   d = rows 6,7 */
    b = (b + (((a << 8) | (a >> 8)) & 0xFFFFu)) & 0xFFFFu;
    c = (c + (((d << 8) | (d >> 8)) & 0xFFFFu)) & 0xFFFFu;
    return ((uint64_t)((a << 16) | b) << 32) | (uint64_t)((c << 16) | d);
}
static inline uint64_t deer_race(uint64_t S, uint64_t M)
{
    int s;
    for (s = 0; s < STRIDES; ++s) {
        uint64_t prev = S;            /* the shape before the twist  */
        uint64_t T = S ^ M;           /* uphill twist: carrying nothing across positions */
        T = deer_rot90(T);            /* downhill fold: ninety degrees */
        T = deer_fold(T);             /* downhill fold: corners into its own centre */
        S = T + prev;                 /* doubling-back: the trail eats itself */
    }
    return S;
}
#endif

/* ===================================================================== */
uint64_t kernel(const unsigned char *data, size_t len)
{
    unsigned char tail[16];
    size_t rem, whole, i;
    deer_t S;
    uint64_t token;

    /* step 1: lay the marks out in the order they were given, left to right.
       Nothing is spoken aloud: no input byte is ever copied toward the
       result except through the shrine.  Index order is the only order. */
    const unsigned char *stone = data;

    /* step 2: count the marks; at the right end one black pebble, then white
       pebbles to reach a whole count of heaps, then, in the final heap's
       tail, the count of marks before any were added.  The black pebble is
       laid even when the count already came out even (else two piles fold
       to one token).  The tail is therefore always exactly two heaps.     */
    rem = len & (size_t)7;
    if (rem) memcpy(tail, stone + (len - rem), rem);
    tail[rem] = 0x80;                                   /* the black pebble */
    memset(tail + rem + 1, 0, (size_t)(7 - rem));       /* white pebbles    */
    for (i = 0; i < 8; ++i)                             /* how many marks   */
        tail[8 + i] = (unsigned char)(((uint64_t)len) >> (56 - 8 * i));

    /* step 3: cut the row into heaps of even count, left to right, every heap
       the same size (8 marks = one grid).  The tail heap is full by step 2. */
    whole = len >> 3;

    /* step 4: set the running shape at the shrine — a fixed pattern not
       chosen today and never changed, the same for every pile, forever.  */
    S = DEER_LIFT(SHRINE_IV);

    /* step 5: take the leftmost unraced heap, set the running shape beside it,
       and race the pair in the deer's stride, and only in this stride.
       step 6: the same fixed number of strides for every heap, counted, never
       eyeballed (see deer_race: exactly STRIDES iterations, always).
       step 7: whatever comes off the path is the new running shape; the heap
       and the old running shape are burned — nothing else is kept alive.
       step 8: return to step 5 with the next heap, until none remain.     */
    for (i = 0; i < whole; ++i)
        S = deer_race(S, DEER_LIFT(deer_load_be64(stone + 8 * i)));
    S = deer_race(S, DEER_LIFT(deer_load_be64(tail)));
    S = deer_race(S, DEER_LIFT(deer_load_be64(tail + 8)));

    /* steps 9,10,11: the validation ritual — change one single mark, carry the
       changed pile through steps 1..8, and demand that about half the squares
       differ with no corner or edge region untouched, over at least twenty
       different single-mark changes.  It tests the method, not this pile, so
       it lives in deer_selftest() below and is not run per token.          */

    /* step 12: the last heap is burned; the final running shape stands alone
       at the shrine.  Read it off and hand it over; leave the riverbank bare
       so nothing remains that could walk the token back to its marks.      */
    token = DEER_DROP(S);
    memset(tail, 0, sizeof tail);
    __asm__ __volatile__("" :: "r"(tail) : "memory");
    return token;
}

#ifdef DEER_SELFTEST
#include <stdio.h>
int deer_selftest(void)
{
    /* the eight regions of the grid that step 10 forbids leaving untouched */
    static const uint64_t region[8] = {
        0xC0C0000000000000ULL, 0x0303000000000000ULL,  /* top-left, top-right corners */
        0x000000000000C0C0ULL, 0x0000000000000303ULL,  /* bottom-left, bottom-right   */
        0xFF00000000000000ULL, 0x00000000000000FFULL,  /* top edge, bottom edge       */
        0x8080808080808080ULL, 0x0101010101010101ULL   /* left edge, right edge       */
    };
    unsigned char pile[137];
    int k, r, ok = 1;
    for (k = 0; k < 137; ++k) pile[k] = (unsigned char)(k * 37 + 11);

    /* step 11: at least twenty different single-mark changes, in different
       places (first, middle, last among them).                            */
    for (k = 0; k < 64; ++k) {
        int bit = (k == 0) ? 0 : (k == 1 ? 137 * 8 - 1 : (k * 149) % (137 * 8));
        uint64_t A, B, D; int q;
        /* step 9: the original pile, then the pile with one mark changed */
        A = kernel(pile, 137);
        pile[bit >> 3] ^= (unsigned char)(1u << (bit & 7));
        B = kernel(pile, 137);
        pile[bit >> 3] ^= (unsigned char)(1u << (bit & 7));
        D = A ^ B;
        /* step 10: about half the squares differ, no region left the same */
        q = __builtin_popcountll(D);
        if (q < 20 || q > 44) { ok = 0; printf("weak spread %d at bit %d\n", q, bit); }
        for (r = 0; r < 8; ++r)
            if ((D & region[r]) == 0) { ok = 0; printf("region %d survived at bit %d\n", r, bit); }
    }
    /* if this fails, step 10's own remedy: raise STRIDES and start again */
    return ok;
}
#endif
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 0.10**

Stated before any measurement, with the reasoning that produced it:

- **Avalanche ≈ 0.50.** Each stride does one full-range bit rotation (every grid cell moves to a different word position) plus three carry-propagating additions, and the heap is re-injected every stride. 10 strides per 8-byte heap is SHA-class work per byte (80 rounds per 64 bytes vs SHA-256's 64). Difference weight should roughly double per stride from weight 1, saturating by stride 6–7. I expect ~0.499 ± 0.01 and no sticky bit.
- **Throughput is the weak axis, and I predict it will be bad.** The recipe forces a serial Merkle–Damgård chain over *64-bit* heaps (step 5's "position against position" pins the heap to the state size) with ≥8 strides each, so the critical path is ~7 cycles/stride on GFNI hardware (pxor 1, gf2p8affineqb 3, pshufb 1, paddw 1, paddq 1) → **≈ 8.8 cycles/byte ≈ 0.4 GB/s**; on the scalar fallback the delta-swap transpose costs ~15 cycles of latency, giving **≈ 29 cycles/byte ≈ 0.12 GB/s**. A dot-product-style streaming baseline runs at well under 1 cycle/byte, hence ~0.1×. I am not hiding this: the recipe *is* a cryptographic compression function, and it should be priced like one.

## MEASUREMENT

**Not measured.** `hash_bench`, `hash_contract` and the symbolic tools were listed but no tools are callable in this session, so I have no throughput number and no avalanche score to report. Reporting the prediction as if it were a result would be the one thing I must not do, so: the artifact above is unmeasured, and the prediction above stands untested.

What I *could* check by hand, and did:

- **Contract**: signature, includes, and compile flags match; no undefined behaviour on `len == 0` (the `memcpy` is guarded), tail buffer is provably exactly 16 bytes for every `len`, no reads past `data + len`.
- **Injectivity of each stride part**: XOR — bijection; 90° turn — bit permutation, bijection; corner fold — the two halves read (`a`, `d`) are never written, so it is a Feistel-style bijection (I rejected the more obvious `x + reverse(x)` fold precisely because `f(x) = f(reverse(x))` makes it 2-to-1 and hands you a one-block collision family for free); doubling-back — bijection in `T` for fixed `prev`.
- **Padding injectivity**: mandatory `0x80` + length heap ⇒ distinct piles never share a padded row, which is exactly the failure step 2 warns about.

The four allowed improvement rounds are unspent. The pre-registered ladder, in order, had measurements been available: (1) if avalanche < 0.49 or any region survives, apply step 10's *own* remedy — raise `STRIDES` to 12, then 16; (2) if avalanche is fine, spend the budget only on machine-level cost (a `pmovmskb`-based turn for SSSE3-only machines, `movbe` loads, 2-way heap prefetch) — never on shortening the stride count or widening the heap, both of which would replace the native's method rather than implement it.

## VERDICT

The recipe is not a folk mnemonic for a fast hash — it is, read literally, **Merkle–Damgård with length-strengthening (steps 2–3), a fixed IV (step 4), a Matyas–Meyer–Oseas-style feed-forward compression function (step 5's doubling-back), a fixed round count (step 6), state-only retention (step 7), and an empirical strict-avalanche acceptance test (steps 9–11)**. Every object in it landed on something concrete: the grid is an 8×8 bit matrix in one word, "carrying nothing across positions" is XOR against "add" which is carry propagation, "ninety degrees" is `gf2p8affineqb`, "corners into its own centre" is a carry-pressed inward quarter-fold, and "about half the positions must differ, no corner or edge untouched" is literally the avalanche criterion the target problem asks for — the native's world already contains the test the grader runs.

Honest accounting of the two axes:

- **Mixing: I expect this to be strong**, possibly stronger than needed. That is the native's doing, not mine.
- **Speed: I expect this to lose, by roughly 10×.** The loss is structural and traceable to one specific line of the recipe — step 5 makes the heap and the running shape the same shape, which pins the block to 64 bits, and step 6 then demands ≥8 strides per 64 bits on a strictly serial chain (steps 5 and 8). Every fast hash breaks exactly one of those. I could have "fixed" it by taking 32-byte heaps and squeezing at the end, and it would have been ~20× faster — and it would no longer have been this recipe. I left it alone and am telling you the price instead.

The one weakness I found and did **not** repair: the all-zero state with an all-zero heap is a fixed point of the stride (`0 ^ 0 → 0 → 0 → 0 + 0`). Repairing it means adding a round constant, which is a step the recipe does not have. Step 2's mandatory black pebble and trailing length heap keep it unreachable at the end of any pile, so it cannot affect a returned token; it remains a real blemish on the compression function itself, and I would rather flag it than quietly patch the native's design.