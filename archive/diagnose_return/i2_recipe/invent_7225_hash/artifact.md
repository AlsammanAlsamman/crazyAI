## DICTIONARY

| Recipe object | Computational object (literal) |
|---|---|
| a **mark** | one input byte, `data[i]`, taken as a 64-bit weight |
| the **pile**, laid in a row left→right in the order given | the buffer `data[0..len-1]`, traversed in index order; never sorted, blocked, grouped, or reordered |
| the **chalk** at the end of the row | the count `len` itself, held as `chalk` |
| the **tail mark** ("says nothing but that count") | one extra pull whose weight is `(uint64_t)len` — length padding; the pile has `len+1` marks |
| the **cup** | one `uint64_t` register, `cup`. The *only* carried state |
| **pure wine** (rejected) | the constant `0` |
| the **sickened draught / collar colour** ("fixed, known to every taster, same in another town") | a published nothing-up-my-sleeve initial constant, `0x9E3779B97F4A7C15` = ⌊2⁶⁴/φ⌋. Public, not secret, not a seed of my own choosing |
| **pouring** a mark | `cup ^= mark` — the mark's weight enters the standing colour |
| the **single held-breath stir**, "one continuous motion, no pause" | one branch-free multiply by a fixed odd constant, `cup *= 0xD6E8FEB86659FD93`. No branch, no table, no memory touch inside the motion |
| a **pull** (steps 3+4 together) | `cup = rotl64((cup ^ mark) * STIR_P, 29)` |
| **tasting / reading the colour** | a rotation — the only operation that re-reads the *same* 64 bits from a new angle without adding or destroying liquid. It is what feeds well-mixed high bits back down into the low bits that the next mark lands on |
| "do not write it down as a step of its own" | no history array, no per-step output; the register is overwritten in place (O(1) memory) |
| "never two marks together", "never skipping", "one unbroken pour" | a single serial dependency chain, 1 byte per iteration. **No SIMD lanes, no blocking, no tree, no OpenMP.** The chosen seed forbids them, so none are used |
| the **final colour** (step 6, "do not stir again") | `colour = cup` with *no* extra mixing round appended after the tail mark |
| an **organ** | one finalizer round: **bend** = `x ^= x >> 32` (bent against a shifted view of the colour it was handed) then **throw away half** = `x *= C_k` (the 128-bit product's upper half is discarded; the surviving low half is the part that churns under carries — "the half that refuses to sit still") |
| **seven organs, fixed order, none skipped, none twice** | seven straight-line rounds with seven *distinct* odd multipliers, written out in order (no table indirection, no loop) |
| **half** | a shift of 32 in a 64-bit word |
| a **harder bend** (step 11's remedy) | a larger shift / stronger multiplier per organ |
| the **token** | the returned `uint64_t` |
| **knotted to hold one size only** | fixed 64-bit output width, independent of `len` |
| the **garden door** / **nightingale's note** | the avalanche self-test: flip one bit, re-pour the whole pile from the collar colour, compare tokens |
| **threads** of the token | output bits |
| "about half the threads differ" | `popcount(old ^ new) ≈ 32` |
| "no run of threads sits unchanged" | longest zero-run in `old ^ new` is short |
| **two different hands, same collar colour, same token** | determinism: no randomness, no time, no pointer values, no thread count, no uninitialised memory |

**Ambiguities, resolved to the most literal reading that is not wrong:**

1. **Step 1, the tail mark.** A mark is a byte, but the count can exceed 255. Smallest change: the tail mark carries the count as its full weight (`uint64_t`), not a truncated byte. Truncating would make piles of length 5 and 261 end at the same token, breaking step 12's first condition.
2. **Step 7, what an organ is.** The sentence names *two* acts per organ. If "throws away half" were the shift itself and nothing more, each organ is `x ^= x>>32`, which is an **involution** — seven of them collapse to one (or to the identity), which step 12 cannot survive. So the bend is the xor-against-shifted-self and the half-discard is the truncating multiply. Step 11's remedy ("a harder bend at each organ") confirms the bend is a tunable strength, not a bare shift.
3. **Step 10, "about half".** Testing for *exactly* 32 differing bits would reject a perfect hash (a perfect hash gives Binomial(64, ½), sd ≈ 4). Read "about half" as a tolerance band and "no unchanged run" as a bound on the longest zero-run of the difference.
4. **Steps 9–12 are the taster's acts, not the pour.** They are compiled as a real self-test function (`pour_selftest`) that `kernel` never calls — hashing must stay one unbroken pour, and folding the test into `kernel` would double every call's cost for no change in the token.
5. **Rejected speed trick, named for honesty.** `cup = cup*P ^ (mark*Q)` takes the mark off the critical path (4 cycles/byte instead of 5) but it *pre-processes the mark before pouring it*, a step the recipe does not have. Declined.

## ARTIFACT

```c
/* pour.c -- "one unbroken pour": a strictly serial, byte-chained 64-bit hash.
 *
 * The CHOSEN SEED fixes the architecture: every mark is poured into one cup and
 * the cup's changed colour is the seed for the next pour.  That forbids lanes,
 * blocks, trees and SIMD on the pour itself, so none are used; OpenMP is
 * declined for the same reason and because step 12 requires that two different
 * hands reach the same token.  Cost of that fidelity: ~5 cycles per byte.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* the tavern's published constants -------------------------------------- */

/* the collar colour: wine already cut.  Not pure wine (not 0): a
 * nothing-up-my-sleeve constant, floor(2^64/phi), so a stranger in another
 * town provably starts from the same colour and not one of my choosing. */
#define COLLAR   0x9E3779B97F4A7C15ULL

#define STIR_P   0xD6E8FEB86659FD93ULL  /* the single held-breath stir  */
#define TASTE_R  29                     /* the angle the colour is read at */

/* seven organs, fixed order, all seven multipliers distinct (none twice) */
#define ORGAN1   0xFF51AFD7ED558CCDULL
#define ORGAN2   0xC4CEB9FE1A85EC53ULL
#define ORGAN3   0xBF58476D1CE4E5B9ULL
#define ORGAN4   0x94D049BB133111EBULL
#define ORGAN5   0x2545F4914F6CDD1DULL
#define ORGAN6   0x9FB21C651E98DF25ULL
#define ORGAN7   0xA24BAED4963EE407ULL
#define HALF     32                     /* half of the 64 threads received */

/* the taste: the same liquid, re-read from a new angle.  A rotation creates
 * and destroys no bits, which is exactly what "reading the colour" means. */
static inline uint64_t taste(uint64_t cup)
{
    return (cup << TASTE_R) | (cup >> (64 - TASTE_R));
}

/* one pull = step 3 (pour + one continuous stir) followed by step 4 (taste).
 * Kept as one inline act so step 5 can literally "repeat steps 3 and 4". */
static inline uint64_t pull(uint64_t cup, uint64_t mark)
{
    return taste((cup ^ mark) * STIR_P);
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* step 1: lay the pile out in a single row, left to right, in exactly the
     * order given -- no sort, no grouping, nothing set aside.  Count them and
     * chalk the count at the end of the row, then lay one final mark after the
     * chalk that says nothing but that count: the tail mark, now part of the
     * pile.  The row is data[0..len-1]; the pile is that row followed by the
     * tail mark, so the pile holds len+1 marks. */
    const unsigned char *row  = data;
    const size_t         chalk     = len;
    const uint64_t       tail_mark = (uint64_t)chalk;

    /* step 2: set out the cup and fill it with the sickened draught.  This
     * starting colour is the collar colour, written down above in the open. */
    uint64_t cup = COLLAR;

    /* step 3: take the leftmost mark, pour it in, and stir once with the
     * breath held -- one continuous motion, no pause -- so the mark's weight
     * and the cup's standing colour mix into a single new colour.
     * step 4: taste the cup and read its colour; that colour becomes the
     * colour the cup now stands at and the one that receives the next pull.
     * The cup is never emptied and never refilled; nothing but the colour
     * carries forward and the mark is discarded off the bench once poured.
     * (If the row is empty the leftmost mark of the pile is the tail mark.) */
    cup = pull(cup, len ? (uint64_t)row[0] : tail_mark);

    /* step 5: move to the next mark on the right and repeat steps 3 and 4
     * exactly as before -- same cup, same single held-breath stir, same
     * tasting.  Once per mark, marching rightward, never skipping, never
     * going back, never pouring a mark twice, never pouring two marks
     * together.  The march ends on the tail mark, the last mark on the row. */
    for (size_t i = 1; i < len; i++)
        cup = pull(cup, (uint64_t)row[i]);
    if (len)
        cup = pull(cup, tail_mark);

    /* step 6: the tail mark has been poured and tasted, so stop pouring.  Do
     * not stir again -- no extra round is appended here.  The cup stands at
     * its final colour, its whole memory of the pile. */
    const uint64_t colour = cup;

    /* step 7: carry the final colour through seven organs in turn, in fixed
     * order, always the same order.  Each organ bends the colour against the
     * colour it was handed (xor with a shifted view of itself) and throws away
     * half of what it received -- the multiply's upper 64 product bits are
     * discarded, and the surviving low half, the half that refuses to sit
     * still under carries, is handed to the next organ.  None skipped, none
     * visited twice. */
    uint64_t x = colour;
    x ^= x >> HALF;  x *= ORGAN1;   /* organ 1 */
    x ^= x >> HALF;  x *= ORGAN2;   /* organ 2 */
    x ^= x >> HALF;  x *= ORGAN3;   /* organ 3 */
    x ^= x >> HALF;  x *= ORGAN4;   /* organ 4 */
    x ^= x >> HALF;  x *= ORGAN5;   /* organ 5 */
    x ^= x >> HALF;  x *= ORGAN6;   /* organ 6 */
    x ^= x >> HALF;  x *= ORGAN7;   /* organ 7 */

    /* step 8: take what the seventh organ hands back -- the token -- and knot
     * it into a collar that holds one size only: 64 bits for a pile of three
     * marks and 64 bits for a pile of ten thousand. */
    return x;
}

/* ---------------------------------------------------------------------- *
 * Steps 9-12 are the taster's acts at the garden door, not part of the
 * pour: they are compiled here but kernel() never calls them, because
 * hashing must remain one unbroken pour.
 * ---------------------------------------------------------------------- */

static int longest_unchanged_run(uint64_t diff)
{
    int best = 0, cur = 0;
    for (int i = 0; i < 64; i++) {
        if (((diff >> i) & 1ULL) == 0ULL) { cur++; if (cur > best) best = cur; }
        else cur = 0;
    }
    return best;
}

/* step 10: hold the new token beside the old one at the nightingale's note and
 * count how many of their threads differ.  It passes if about half the threads
 * differ and no run of threads sits unchanged in both.  "About half" is read
 * as a tolerance band, not as exactly 32: a flawless token gives
 * Binomial(64,1/2) with sd 4, so demanding exactly half would reject a sound
 * pour.  That is the one place where the fully literal reading is wrong, and
 * this band is the smallest change that fixes it. */
static int step10_passes(uint64_t old_token, uint64_t new_token)
{
    const uint64_t diff = old_token ^ new_token;
    const int threads   = __builtin_popcountll(diff);
    return (threads >= 16 && threads <= 48) && (longest_unchanged_run(diff) < 20);
}

__attribute__((used))
static int pour_selftest(void)
{
    enum { N = 64 };
    unsigned char pile[N], probe[N], copy[N];
    long long total_threads = 0;
    int trials = 0;

    for (int i = 0; i < N; i++) pile[i] = (unsigned char)(i * 7 + 1);

    /* step 9: reach back into the pile as it was and change exactly one mark --
     * the quietest one, the one you'd least expect to matter: the lowest thread
     * of the last mark poured, which receives the fewest stirs.  Then pour the
     * whole pile through from step 2 again, from the collar colour, in full:
     * all marks, all seven organs. */
    const uint64_t old_token = kernel(pile, N);
    memcpy(probe, pile, N);
    probe[N - 1] ^= 1u;
    const uint64_t new_token = kernel(probe, N);

    /* step 10 (applied to the quietest mark) */
    if (!step10_passes(old_token, new_token))
        /* step 11: fewer than half differ, or a shape of the old token
         * survives -- throw the whole method away and return to step 2 with a
         * more sickened wine or a harder bend at each organ.  A knot that
         * remembers its old shape after one thread is pulled is no knot. */
        return 11;

    /* step 12: finished when three things hold at once.
     * (a) any two different piles end at two different tokens */
    {
        uint64_t seen[256]; int n = 0;
        for (int L = 0; L <= 40; L++) {
            unsigned char b[40];
            for (int i = 0; i < L; i++) b[i] = (unsigned char)(i * 31 + L);
            seen[n++] = kernel(b, (size_t)L);
        }
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (seen[i] == seen[j]) return 12;
    }
    /* (b) the same pile poured twice, by two different hands, from the same
     * collar colour, ends at the same token: no randomness, no clock, no
     * pointer value, no uninitialised memory, no ordering left to a thread. */
    memcpy(copy, pile, N);
    if (kernel(pile, N) != old_token)   return 12;
    if (kernel(copy, N) != old_token)   return 12;

    /* (c) every one-mark change tried in step 9 passes the test at step 10 */
    for (int i = 0; i < N; i++) {
        for (int b = 0; b < 8; b++) {
            memcpy(probe, pile, N);
            probe[i] ^= (unsigned char)(1u << b);
            const uint64_t t = kernel(probe, N);
            if (!step10_passes(old_token, t)) return 12;
            total_threads += __builtin_popcountll(old_token ^ t);
            trials++;
        }
    }
    if (trials == 0) return 12;
    {   /* the nightingale's note in aggregate: the mean must sit at half */
        const double mean = (double)total_threads / (double)trials;
        if (mean < 31.0 || mean > 33.0) return 12;
    }

    return 0;   /* the pour is sound, and the token is the pile's true name */
}
```

## PREDICTION

Written before any measurement (and, as it turns out, before any measurement was possible):

**PREDICTION: speedup_vs_dp = 0.75**

Reasoning, so the number is falsifiable rather than decorative:

- The pour is one serial dependency chain of `xor → imul → rol` per byte: 1 + 3 + 1 ≈ **5 cycles/byte**, i.e. ~0.2 bytes/cycle, ~0.8 GB/s at 4 GHz. This is a latency floor, not a throughput one — widening it is exactly what the chosen seed forbids.
- Anchor: if the `dp` reference is a byte-at-a-time xor-multiply hash (FNV-1a class, ~4 cycles/byte), I land at **≈0.8**, minus a little for the fixed ~35-cycle seven-organ finalizer on short buffers → **0.75**.
- Honest band: if the reference is a naive hash carrying a `%` per byte, I would be **≈4×**; if it is a word-at-a-time modern hash (~0.5 cycles/byte), I would be **≈0.12**. I am betting on the FNV-class anchor.
- Avalanche: I predict a **near-perfect score** — mean bit-flip rate 0.50 ± 0.01, worst single-bit case within [0.44, 0.56], and `pour_selftest()` returning **0**. Two independent reasons: the taste-rotation feeds well-mixed high bits back into the low bits each mark receives (removing the classic xor-multiply defect where output bit 0 is a bare XOR of all input bit 0s), and seven bend-and-halve organs is roughly twice the finalizer strength needed for full 64→64 diffusion.

## MEASUREMENT

**No measurement was taken, and I will not dress that up.** This session was invoked with no tools: `hash_bench` and `hash_contract` were unavailable, so throughput, the avalanche score, and even `pour_selftest()`'s return value are **unverified predictions, not results**. The permitted four improvement rounds therefore ran **zero** times — the artifact above is the first draft, hardened by reasoning alone.

What the pipeline should check against the prediction above, in order of how likely I am to be wrong:

1. **Throughput / `speedup_vs_dp`.** Highest-variance claim, because it depends entirely on what `dp` is — my ±one-order-of-magnitude band admits that.
2. **Avalanche.** Lower variance; if it comes back below ~0.49 mean or with a bad worst case, the recipe itself names the fix in step 11 (a more sickened collar colour, or harder bends), and I would apply that rather than substitute a different finalizer.
3. **`pour_selftest()` must return 0.** If it returns 11, step 9's quietest-mark case failed. If it returns 12, either a length collision slipped through (which would mean my step 1 tail-mark reading is still too weak) or a single-bit case failed the band.

Two known, non-negotiated costs I would report even if the bench did not notice them: the 7 extra multiplies penalise very short buffers (~35 cycles fixed), and the 1 byte/iteration pour leaves ~87% of a 64-bit lane idle. A word-at-a-time or 4-lane version would be roughly 8× faster and is the obvious textbook move — **it is precisely what the chosen seed forbids**, so it is not in this kernel.

## VERDICT

The recipe is a **serial byte-chained hash with length padding and a seven-round finalizer** — read literally, object for object, it is a coherent and recognisable design, not nonsense. Three of its clauses are load-bearing and good: the "sickened, not pure" collar colour (a public non-zero init), the tail mark (length padding, which is what makes step 12's distinctness condition achievable at all), and the taste as a re-read of the same liquid (the rotation that repairs low-bit diffusion in the pour).

Two clauses do not survive a fully literal reading, and I flagged both rather than papering over them: **step 7's organ**, where "throw away half" as a bare shift makes each organ an involution and collapses all seven into at most one — so the half-discard must be the truncating multiply; and **step 10's "about half"**, which as an exact test would reject a flawless token — so it becomes a binomial tolerance band.

The one thing the recipe pays dearly for is its own chosen seed. "One unbroken pour, never two marks together" is a hard prohibition on the entire modern hashing playbook, and it costs roughly 8× throughput. I kept it. I expect the bench to report a hash with excellent avalanche that is somewhat *slower* than the baseline (predicted 0.75), and I would rather hand back a faithful translation with an honest sub-unity number than a fast kernel that quietly stopped being this recipe.

---

*Note: the `claude.ai` PubMed connector shows as requiring authorization. This session cannot run the OAuth flow, so that capability is unavailable until you authorize it in your claude.ai connector settings (or via `/mcp` in an interactive session). It was not needed for this task.*