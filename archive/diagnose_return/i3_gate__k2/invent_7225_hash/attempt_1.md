## MAPPING

**SEED 1 — the sickened cup, one unbroken pour**

| world object | problem object |
|---|---|
| pile of marks | the byte buffer `data[0..len-1]` |
| one mark | one input byte (eight marks when eight will fill a cup) |
| the cup that is never pure / sickened wine | the 64-bit accumulator, initialised to a *non-pure* constant (φ's digits), never 0 |
| "pure wine holds no memory" | a zero/clean state forgets quiet marks — zero bytes would not change it |
| "wine that is cut remembers every drop" | the update is a function of **both** the mark and the existing colour (chaining) |
| a pull on the cup | one chained state update |
| "taste the colour, let it be the colour waiting for the next pull" | the state after mark *i* seeds mark *i+1*; **the tasted colour sets the next pull's turn** (data-dependent rotate) |
| no mark judged alone or twice | every byte read exactly once, never in isolation, **no overlapping last-8-byte re-read** |

Breaks: *"mixing one byte requires a multiplication."* A pull is a turn and a taint, not a product. (It *affirms* single-accumulator/in-order/streaming.)

**SEED 2 — seven organs**

| world object | problem object |
|---|---|
| the cup's last colour | the final accumulator entering the finalizer |
| the dose of sickness the body carries | `len`, xor'd in at the first organ |
| seven organs, bent once each | a fixed ladder of exactly 7 finalizer stages |
| "bent" | `rotl` (circular, loses nothing) |
| "throws away half of what came before" | `x += x << t` — half the copy falls off the top and is *gone*; the organ compensates by addition (carries = the only nonlinearity, multiply-free) |
| "keeping only what refuses to sit still" | `x ^= x >> s` — keep only bits that disagree with their shifted selves; the halving ladder s = 32,16,8,4,2,1, circulation closing back at 32 |
| garden door, nightingale's note, beauty traded for beauty exactly | the 50 %-of-64-bits avalanche criterion that fixes the stage count |

Breaks: *"more mixing rounds always means better mixing."* Each organ is *destructive by construction* (half is discarded), the count is fixed at seven, and sufficiency is decided by a measured criterion at the door, not by adding rounds.

**SEED 3 — change one mark, re-pour, condemn**

| world object | problem object |
|---|---|
| change one mark, anywhere, even the quietest | flip one bit of a probe buffer, including the LSB of a zero byte |
| pour the whole thing again from the first cup | recompute the full hash, not an incremental update |
| "still resembles the old one's shape" | low Hamming distance between the two 64-bit tokens |
| "throw the whole method away" | runtime fallback to the known way (FNV-1a) |

Breaks: *"the whole buffer must be read once, start to end, in order"* — the method re-reads a probe pile O(bits) times before it will trust itself.

## CHOSEN SEED

**SEED 2**, the seven organs. It is the only one of the three that breaks the preferred assumption ("more mixing rounds always means better mixing"), so the preference rule settles it. Seeds 1 and 3 are the same native's same pour, so they are implemented as the stages they are: SEED 1 is the pour that feeds the organs, SEED 3 is compiled in as the acceptance gate with the fallback it demands. The reviewer's complaint is answered structurally: **there is not one multiplication in the accepted path** — no FNV prime, no Murmur word mixer, no fmix. The quality engine is the organ ladder; the speed comes from how the pull is implemented, not from replacing it.

## ASSUMPTION BROKEN

"More mixing rounds always means better mixing." Seven organs, each *throwing away* half of what it received — adding rounds adds destruction, so the count is fixed and validated at the garden door instead of raised. Secondarily: "mixing one byte requires a multiplication" — the pull is `rotl(cup ^ mark, cup) + TAINT`, and the organs' only nonlinearity is addition carries.

Two regimes are recognised at runtime, both inside the metaphor: (i) a pile that fills cups is poured eight marks at a time, a quantity too small to fill a cup is tipped in as one draught (`len>>3` / `len&7`) — this also guards the fixed organ cost on tiny piles; (ii) the garden door decides at load time whether the method is fit at all, and condemns it to the known way if not. No OpenMP and no SIMD on the pour: the metaphor forbids a second cup ("a single unbroken pour"), so the chain cannot be split — only the loads are allowed to run ahead of it.

## ARTIFACT

```c
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
```

## PREDICTION

PREDICTION: speedup_vs_dp = 8.0

Reasoning, stated before any measurement: the reference is a serial dependent chain of xor + 64-bit multiply, ≈4 cycles per **byte** (≈0.25 B/cycle). The pour is a serial dependent chain of xor + variable rotate + add, ≈3–4 cycles per **8 bytes** (≈2.0–2.7 B/cycle), i.e. 8–11× on buffers of a few hundred bytes and up, pulled down by the 7-organ fixed cost (~35 cycles) on short piles. Break-even against the known way is ≈9–10 bytes; below that the known way is up to ~1.4× faster (bounded by the one-draught sub-cup path, which is why that size check exists). Avalanche: I predict ≈0.50 with per-trial Hamming distance between the two tokens staying in roughly 18–46 of 64 — the organs are a bijection (odd shift-add multipliers, triangular xorshifts, rotates), nonlinear only through addition carries, which is what makes the differential input-dependent rather than a fixed mask.

## MEASUREMENT

Plainly: **no measurement was possible in this session — no tools were available**, so the prediction above stands untested by me and the pipeline's `hash_bench` numbers are the real verdict. I will not dress this up.

What I did instead is put the native's own measurement inside the artifact, which is the thing the reviewer found missing: `garden_door()` runs at load time, flips every single bit of probe piles at nine lengths spanning both regimes (1,2,3,5,8,13,21,34,64 — 1208 single-bit differentials), re-pours each from the first cup, and requires the mean Hamming distance to be 45–55 % of 64 with no individual trial closer than 8 bits. If my analysis of the organ ladder is wrong, the gate condemns the method and `kernel` silently serves FNV-1a — the throughput will read ≈1.0× instead of ≈8×, and that reading is itself the honest report that the mechanism failed its own test. There is no outcome where a bad mixer ships quietly.

## VERDICT

The core is the native's, translated literally and with nothing textbook substituted in: a chained pour whose turn is set by the cup's own tasted colour, a seven-organ finalizer that *discards* half at every stage on a halving ladder, and a differential self-test as a hard acceptance gate with a real fallback. **Zero multiplications in the accepted path** — that is the checkable difference from FNV-1a, xxHash and Murmur, all of which are multiply mixers.

Where it is validated rather than invented: the pull's data-dependent rotation is RC5/RC6's data-dependent rotation; the organ's `x += x << t` is shift-add multiplication by an odd constant, the standard multiply-free odd-multiplier bijection; alternating an upward carry step with a downward xorshift is the proven xorshift-mix skeleton. I let the mechanism arrive at those rather than invent new primitives.

Honest risks, and what I did about each:
- **Tiny piles.** The 7 organs cost ~35 cycles regardless of length, so below ~9 bytes the known way is up to ~1.4× faster. Guarded by the runtime regime check (`len & 7`): a quantity too small to fill a cup is tipped in as one draught, which is the cheapest path that still clears the garden door. I did **not** fall back to FNV there, because FNV's short-key avalanche would fail the native's own gate and that would trade a 1.4× speed loss for a quality loss.
- **Near-linearity of the pour.** The pull is xor/rotate/add; its nonlinearity comes from the data-dependent turn and from carries. A chained `f(cup ^ mark)` is in principle colliding-by-construction if an attacker can see the state and choose the next word — the data-dependent turn frustrates this but does not make it collision-resistant. This kernel is a fast non-cryptographic hash, and I would not claim otherwise; the benchmark measures avalanche and throughput, not adversarial collisions.
- **No parallelism.** The metaphor forbids a second cup, so no OpenMP and no SIMD on the chain, and no overlapping last-8-byte read (that would judge marks twice). The ceiling is therefore the chain latency, ~2–2.7 B/cycle, not memory bandwidth. A split-cup version would be faster and would not be the native's method; I did not ship one.