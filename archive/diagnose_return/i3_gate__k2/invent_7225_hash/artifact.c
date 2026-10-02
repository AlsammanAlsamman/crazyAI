#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ============ the tavern's sickened wine (SEED 1: the pour) ============
 * The cup is one 64-bit accumulator that is never pure: a pure (zero) cup
 * holds no memory of the quiet marks -- zero bytes would leave it unchanged.
 * Cut wine remembers every drop: each pull folds the mark into the colour
 * already there, and the colour tasted after a pull decides how far the cup
 * is turned on the next one.  No multiplication anywhere.
 */
#define CUT_WINE 0x9E3779B97F4A7C15ULL   /* phi: cut, structureless, nonzero */
#define TAINT    0x2545F4914F6CDD1DULL   /* odd: every pull re-sours the cup  */

static inline uint64_t turn(uint64_t x, unsigned r) {  /* one bend, UB-safe */
    r &= 63u;
    return (x << r) | (x >> ((64u - r) & 63u));
}

static inline uint64_t pull(uint64_t cup, uint64_t mark) {
    /* the colour waiting from the previous pull sets this pull's turn */
    return turn(cup ^ mark, (unsigned)cup) + TAINT;
}

/* ============ the seven organs (SEED 2: the core) ============
 * bent -> half of it falls off the top edge and is gone, the organ
 * compensating by addition -> keep only what refuses to sit still.
 * The discard ladder halves: 32,16,8,4,2,1, circulation closing at 32.
 */
#define ORGAN(x, r, t, s) do {                                    \
        (x)  = turn((x), (r));          /* bent                */ \
        (x) += (x) << (t);              /* half thrown away    */ \
        (x) ^= (x) >> (s);              /* the restless kept   */ \
    } while (0)

static inline uint64_t organs(uint64_t cup, uint64_t dose) {
    uint64_t x = cup ^ dose;            /* the body knows how much it drank */
    ORGAN(x, 17, 19, 32);
    ORGAN(x, 41, 29, 16);
    ORGAN(x, 23, 11,  8);
    ORGAN(x, 53, 37,  4);
    ORGAN(x, 31, 23,  2);
    ORGAN(x, 47, 43,  1);
    ORGAN(x, 29, 13, 32);
    return x;
}

/* ---- the whole pile as a single unbroken pour ---- */
static uint64_t pour(const unsigned char *data, size_t len) {
    const unsigned char * __restrict p = data;
    uint64_t cup = CUT_WINE;
    size_t ncups = len >> 3;            /* regime: cups the pile can fill */
    size_t rest  = len & 7u;            /* regime: too small for a cup    */
    size_t c = 0;

    /* loads run ahead four cups at a time; the pour itself stays serial */
    for (; c + 4 <= ncups; c += 4, p += 32) {
        uint64_t m0, m1, m2, m3;
        memcpy(&m0, p,      8);
        memcpy(&m1, p +  8, 8);
        memcpy(&m2, p + 16, 8);
        memcpy(&m3, p + 24, 8);
        cup = pull(cup, m0);
        cup = pull(cup, m1);
        cup = pull(cup, m2);
        cup = pull(cup, m3);
    }
    for (; c < ncups; c++, p += 8) {
        uint64_t m; memcpy(&m, p, 8);
        cup = pull(cup, m);
    }
    if (rest) {                         /* one draught, no mark read twice */
        uint64_t w = 0;
        for (size_t k = 0; k < rest; k++) w |= (uint64_t)p[k] << (8u * k);
        cup = pull(cup, w);
    }
    return organs(cup, (uint64_t)len);
}

/* ---- the method that is thrown away to, if condemned ---- */
static uint64_t known_way(const unsigned char *data, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) { h ^= data[i]; h *= 1099511628211ULL; }
    return h;
}

/* ============ the garden door (SEED 3: the gate) ============
 * Change one mark anywhere in the pile -- even the quietest -- and pour the
 * whole thing again from the first cup.  If the new token still resembles
 * the old one's shape, the whole method is thrown away.
 */
static int g_verdict = -1;              /* -1 untested, 1 accepted, 0 condemned */

static int garden_door(void) {
    unsigned char pile[64];
    uint64_t r = 0x853C49E6748FEA9BULL;
    for (int i = 0; i < 64; i++) {
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        pile[i] = (unsigned char)(r >> 33);
    }
    static const unsigned char lens[] = { 1, 2, 3, 5, 8, 13, 21, 34, 64 };
    for (unsigned li = 0; li < sizeof lens / sizeof lens[0]; li++) {
        size_t n = lens[li];
        uint64_t base = pour(pile, n);
        unsigned long long sum = 0, trials = 0;
        for (size_t b = 0; b < n * 8u; b++) {
            pile[b >> 3] ^= (unsigned char)(1u << (b & 7u));
            uint64_t t = pour(pile, n);
            pile[b >> 3] ^= (unsigned char)(1u << (b & 7u));
            unsigned d = (unsigned)__builtin_popcountll(base ^ t);
            if (d < 8u || d > 56u) return 0;      /* it still resembles */
            sum += d; trials++;
        }
        /* beauty for beauty, exactly: the mean must sit on 32 of 64 */
        if (sum * 100ULL < trials * 64ULL * 45ULL) return 0;
        if (sum * 100ULL > trials * 64ULL * 55ULL) return 0;
    }
    return 1;
}

static void __attribute__((constructor)) stand_at_the_door(void) {
    g_verdict = garden_door();
}

uint64_t kernel(const unsigned char *data, size_t len) {
    int v = g_verdict;
    if (__builtin_expect(v < 0, 0)) { v = garden_door(); g_verdict = v; }
    if (__builtin_expect(v != 0, 1)) return pour(data, len);
    return known_way(data, len);        /* condemned: the method thrown away */
}
