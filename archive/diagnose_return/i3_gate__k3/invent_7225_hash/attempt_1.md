# MAPPING

**SEED 1 — the unbroken pour through the impure cup**

| world object | problem object |
|---|---|
| the pile of marks | the byte buffer `data[0..len)` |
| one mark | one input byte |
| "the cup that is never pure", the tavern's **cut** wine | a mixing medium that is a *mixture*, not a formula: a static table of 64-bit drops (`cup[8][256]`, 16 KB) |
| "pure wine holds no memory" | a **pure** function of the current byte alone carries no state |
| "wine that is cut remembers every drop added to it" | the accumulator is a function of *every* byte poured so far |
| "each mark is a pull on the cup" | one indexed draw: `cup[0][(unsigned char)(s ^ byte)]` — the mark chooses *where* in the mixture you draw from |
| "I taste the cup's color against my tongue" | read out the new 64-bit accumulator |
| "let that color become the color waiting for the next pull" | `s` is the seed of the next draw — a strict serial data dependency |
| "a single unbroken pour" | one accumulator, every byte, in order |
| "no mark ever judged alone or twice" | no independent lanes, no second pass, no per-byte side accumulator |

Assumption broken: **"mixing one byte requires a multiplication"** — the pour contains zero multiplies (shift, xor, one load). And in the fast realization of the *same* chain, **"each byte must be mixed into the running state before the next byte is read"** — eight marks are read and folded in one step while producing the bit-identical value of the byte-at-a-time pour.

**SEED 2 — seven organs, each discarding half**

| world object | problem object |
|---|---|
| the cup's final color | the 64-bit accumulator after the last byte |
| "carry it through my own organs" | a finalizer applied once, outside the pour |
| one organ | one fold |
| "throws away half of what came before" | `x >> k`, k ≈ 32 = half the word, discarding half the bits |
| "keeping only what refuses to sit still" | `^=` — XOR keeps only the bits that *differ* |
| "bent seven times, once per organ" | exactly seven folds |
| "the way a body carries a sickness until every organ compensates" | odd multiplications between folds (the nonlinear compensation) |
| garden door, nightingale's last note, "trade beauty for beauty exactly" | the avalanche target: exactly half the output bits |
| "if the two don't trade exactly, I bend again" | add a fold only until the avalanche check passes |

Assumption broken: this is the *closest* of the three to **"more mixing rounds always means better mixing"** — each bending is explicitly **lossy** (half is thrown away) and the stopping rule is the garden-door check, not "more". But it does not actually deny the premise; it bounds it at seven. **Stated plainly: none of the three seeds cleanly breaks "more mixing rounds always means better mixing."**

**SEED 3 — flip one mark, re-pour, condemn**

| world object | problem object |
|---|---|
| "change one mark, anywhere in the pile, even the quietest one" | flip one input bit, including in the last byte |
| "pour the whole thing through again from the first cup" | recompute the full hash from scratch (no incremental shortcut) |
| "if the new token still resembles the old one's shape" | output Hamming distance far from 32/64 |
| "throw the whole method away" | reject the kernel |
| "a door someone forgot to turn" | an identity-like / weak permutation |

Assumption broken: **none of the listed ones** — this is a test protocol, not a mixing mechanism. It cannot be the kernel's core.

# CHOSEN SEED

**SEED 1.** No seed breaks "more rounds = better", so per the rule I fall back to the most literal seed. SEED 1 is also the only one that *is* the kernel (SEED 3 is a test, SEED 2 is an epilogue). Its mapping is the most literal available: every noun has exactly one computational referent, and the one noun the textbook method has no referent for — **"the cup that is never pure"** — is precisely where it diverges. FNV-1a mixes a byte through a *pure closed form* (a multiply by a prime). The native refuses that: the cup is a **cut mixture**, and a mark does not *transform* the state, it **pulls** on the mixture. That is a table-indexed draw, not an arithmetic one.

Taken literally, the pour is:

```
s ← cup[(s ⊕ byte) & 0xFF] ⊕ (s >> 8)
```

one accumulator, every byte, each result seeding the next pull. This is — arriving at it rather than inventing around it — **table-driven chained CRC-64 / tabulation hashing**, a validated real-world mechanism with published guarantees, not a novelty.

The native's own constraint then forces the speed work into the right place. Done drop by drop this is *slower* than FNV-1a (the chain is load-latency-bound, ~6 cycles/byte). The only legitimate way to make it fast is to implement **the same pour** better. Because the cup is GF(2)-linear, eight consecutive drops compose into a single closed step. Writing `A(s) = (s>>8) ⊕ cup₀[s & 0xFF]` and `w = s ⊕ v` (v = the next 8 bytes, little-endian):

> s₈ = A(⋯A(A(s⊕b₀)⊕b₁)⋯⊕b₇) = A⁸(s) ⊕ ⊕ⱼ A⁸⁻ʲ(bⱼ) = A⁸(s ⊕ v) = ⊕ⱼ cup₇₋ⱼ[byteⱼ(w)]

so one cupful (8 loads, independent; one xor-tree on the chain) yields **bit-for-bit the value the drop-by-drop pour would have produced**. This is Kadatch–Jenkins *slicing-by-8*, again a validated technique, not an invention. The pour stays single and unbroken; only its latency is restructured.

**Regime recognition, in-world:** the cup has a lip at eight marks. *If the pile cannot fill the cup to the lip, the native pours drop by drop* (`len < 8`, and always for the tail). The fallback is not an approximation — both regimes return the identical number, which is the strongest form a fallback can take.

**No thread parallelism, deliberately.** The metaphor's unit of work is one indivisible pour; splitting it across threads would be judging marks in separate groups — exactly what "no mark ever judged alone" forbids. Vectorization-level hints only (restrict, `movzx`-friendly indexing, L1-resident table).

# ASSUMPTION BROKEN

Primary: **"mixing one byte requires a multiplication."** The entire pour over `len` bytes executes no multiply. Secondary, forced by making the *same* pour fast: **"each byte must be mixed into the running state before the next byte is read"** — eight bytes are read before any is mixed, and the chained result is provably unchanged. Kept, not broken: the single in-place accumulator, and the single in-order pass (SEED 1 insists on both).

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ================= the cup that is never pure =================
 * A cut mixture, not a formula: 8 x 256 sixty-four-bit drops.
 * cup[0] is one mark's worth of advance; cup[k] is k+1 marks' worth,
 * so eight drops can be drawn in one pull and give the SAME colour.
 * (reflected CRC-64/Jones lattice -- GF(2)-linear, and invertible
 *  because bit 63 of the cut is set, so no two piles of equal depth
 *  ever leave the cup the same colour.)
 */
#define CUP_CUT     0x95AC9329AC4BC9B5ULL
#define FIRST_COLOUR 0x9E3779B97F4A7C15ULL  /* never empty, never pure */

static uint64_t cup[8][256];
static int cup_is_poured = 0;

static void pour_the_cup(void)
{
    unsigned i; int k;
    for (i = 0; i < 256u; ++i) {
        uint64_t c = (uint64_t)i;
        for (k = 0; k < 8; ++k)
            c = (c >> 1) ^ (CUP_CUT & (uint64_t)(-(int64_t)(c & 1u)));
        cup[0][i] = c;                      /* one mark of advance      */
    }
    for (k = 1; k < 8; ++k)                 /* k+1 marks of advance     */
        for (i = 0; i < 256u; ++i) {
            uint64_t p = cup[k - 1][i];
            cup[k][i] = (p >> 8) ^ cup[0][p & 0xFFu];
        }
    cup_is_poured = 1;
}

#if defined(__GNUC__)
__attribute__((constructor)) static void cup_ctor(void) { pour_the_cup(); }
#endif

/* one drop: the mark PULLS on the cup; the cup's new colour is the
 * colour waiting for the next pull. no multiply anywhere. */
static inline uint64_t one_drop(uint64_t s, unsigned char b)
{
    return cup[0][(unsigned char)(s ^ (uint64_t)b)] ^ (s >> 8);
}

/* eight marks at once, bit-identical to eight one_drop() calls */
static inline uint64_t one_cupful(uint64_t s, const unsigned char *p)
{
    uint64_t v, w;
    memcpy(&v, p, 8);
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    v = __builtin_bswap64(v);
#endif
    w = s ^ v;
    return cup[7][(unsigned char)(w)]       ^ cup[6][(unsigned char)(w >>  8)]
         ^ cup[5][(unsigned char)(w >> 16)] ^ cup[4][(unsigned char)(w >> 24)]
         ^ cup[3][(unsigned char)(w >> 32)] ^ cup[2][(unsigned char)(w >> 40)]
         ^ cup[1][(unsigned char)(w >> 48)] ^ cup[0][(unsigned char)(w >> 56)];
}

/* =============== seven organs, each discarding half ===============
 * every bending throws away half of what came before (>> ~32) and
 * keeps only what refuses to sit still (^). the multiplies are the
 * body compensating between organs. checked at the garden door
 * against the nightingale: exactly half the bits must turn.
 */
static inline uint64_t seven_bendings(uint64_t x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;   /* 1 */
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;   /* 2 */
    x ^= x >> 33;                               /* 3 */
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;   /* 4 */
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;   /* 5 */
    x ^= x >> 31;                               /* 6 */
    x ^= x >> 32;                               /* 7 */
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t s = FIRST_COLOUR;

    if (__builtin_expect(!cup_is_poured, 0)) pour_the_cup();

    /* --- regime: is the pile deep enough to fill the cup to its lip? --- */
    if (len >= 8) {                       /* deep pile: pour by cupfuls   */
        size_t n = len >> 3;
        len &= (size_t)7;
        do {
            s = one_cupful(s, p);
            p += 8;
        } while (--n);
    }
    /* shallow pile, and every tail: drop by drop. same number either way. */
    while (len--) s = one_drop(s, *p++);

    return seven_bendings(s);
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 3.2**

Stated before any measurement, with the reasoning so it can be falsified: the cupful step's critical path is `xor(1) + byte-extract(1) + L1 load(≈5) + xor-tree(3) ≈ 10` cycles per 8 bytes ≈ **0.8 B/cycle**; FNV-1a's is `xor(1) + imul(3) = 4` cycles per byte = **0.25 B/cycle**. Ratio ≈ 3.2×. Avalanche predicted **≈ 0.50** (score ≈ 0.99–1.00): a single flipped input bit changes the pre-finalizer colour by a fixed nonzero high-weight constant `Δ = A^m(e)` (nonzero because the cup is invertible), and the seven bendings are a strong nonlinear bijection, so `F(s)` and `F(s⊕Δ)` differ in ~half their bits. Secondary claim, falsifiable by one line of test code: the cupful path and the drop-by-drop path return **identical** values for every buffer.

Failure modes I expect if I am wrong: (a) the 16 KB cup contends with the data stream in L1 and the measured rate falls to ~0.5 B/cycle (speedup ≈ 2); (b) the benchmark uses short keys, where the fixed ~26-cycle finalizer dominates and the speedup drops below 1.

# MEASUREMENT

**No measurement was possible in this session — `hash_bench` and `hash_contract` were not available to me.** I will not dress an estimate up as a result. The numbers above are analytic cycle counts, and the pipeline's measurement governs. Honest expectation of what it will report: throughput ≈ 2.4–3.6 GB/s (speedup 2.5–3.5× over the FNV-1a reference) on buffers of a few KB and up; avalanche ≈ 0.50. If the harness benchmarks buffers under ~16 bytes, I expect **no speedup or a slowdown**, and that is a real failure of this kernel at that size, not a measurement artifact.

# VERDICT

The core is now the native's mechanism and nothing else: **one impure cup, one unbroken pour, every mark a pull, each tasted colour the seed of the next.** The previous attempt violated SEED 1 outright — eight parallel lanes is precisely "marks judged alone", and it had made the *finalizer* (SEED 2) the whole idea. Here the finalizer is what the native says it is: an epilogue carried through the organs after the pile ends. The pour itself contains no multiplication, which is the assumption broken.

What the metaphor bought, concretely: because the cup is a *mixture* rather than a formula, it is GF(2)-linear, and linearity is exactly what lets eight drops collapse into one pull with the identical result. The textbook method cannot do this — a multiply chain does not decompose — so FNV-1a is stuck at one byte per multiply latency forever. The native's "impure cup" is the reason the single serial chain can be made ~3× faster *without* abandoning its serialness. That the destination is slicing-by-8 CRC + a strong finalizer is a feature, not a retreat: per the rule, a validated technique beats an invented one, and the mechanism arrived there on its own.

Named risks, and how each is addressed rather than admitted:

- *Cupfuls need a full cup.* Guarded by the explicit `len >= 8` check, falling back to the simplest possible path (drop by drop), which returns the identical value. This is the runtime regime recognition the metaphor itself supplies (the cup's lip), and it covers the tail of every buffer too.
- *Threads could split the pour.* Dropped entirely, not guarded — the metaphor forbids it and the units of work are single bytes, far too small to be worth it at any benchmark size.
- *Table construction cost.* Moved off every call path into a constructor, with one correctly-predicted branch as a fallback; the fallback is idempotent, so a race would be harmless.
- *The 16 KB cup competes for L1.* Not fully addressable; this is the honest residual. Slicing-by-4 (8 KB) would halve it at ~25% of the throughput. If the measurement comes back near 0.5 B/cycle, that is the cause and that is the fix.
- *Short inputs.* Below roughly 16 bytes the seven bendings dominate and this kernel is no faster than FNV-1a, possibly slower. I keep them anyway, and say why rather than hiding it: the core is linear, so without a strong nonlinear finalizer the avalanche score collapses to a fixed per-bit pattern. The trade is deliberate and visible, not accidental.

One thing I will not pretend about, since SEED 2 is where the native is nearest to a wrong belief: **folds 6 and 7 almost certainly buy nothing.** Murmur3's three bendings plus splitmix64's three already saturate avalanche; the seventh organ is there because the native fixed the count at seven, not because the garden-door check demanded it. That is the native's own "more rounds" instinct showing, and it costs a few cycles per call. By the native's stated stopping rule — bend again only if the nightingale is not matched exactly — five would have sufficed. I kept seven and am naming the cost rather than silently trimming the metaphor to flatter the benchmark.

Finally, SEED 3 is the condition under which I would discard all of this: if the pipeline reports an avalanche score materially below ~0.95, the knot remembers its old shape after one thread is pulled, and the method should be thrown away rather than tuned.