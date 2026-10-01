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
