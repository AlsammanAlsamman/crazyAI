## MAPPING

| World object | Computational object |
|---|---|
| pile of marks, in their given order | `data[0..len-1]`, order-significant |
| desk's numbered groove | the buffer's index space; a groove-**cell** = one 8-byte seat (the width of the stone's seated place) |
| "read them as weights, not words" | the cell enters as a **magnitude that adds** (`+`), not a bit-pattern that xors (`^`) |
| "a flat thing about to be invested into three" | the flat byte-row of a cell is lifted into a solid weight before it drops: `w = v ^ (v >> 47)` (invertible, so no two rows invest alike) |
| bone-colored die-stone | one `uint64_t h`. One. Never reset, never spilled |
| "turn it a quarter" | multiply by the odd turning constant `P`; four turns = one revolution = `Q = P^4` |
| "drop the weight into the same seated place" | `h + w` — always the same additive seat |
| "the stone's memory bends how deep this press goes" | the turn acts on `(h + w)`, so the prior callus scales the new weight: `h ← (h + w) * P` |
| callus / nothing washes clean | `P` odd ⇒ every fold is a bijection; no prior press is ever erasable |
| one fold per mark, centuries like a handful | constant work per cell, no length-dependent rounds |
| wooden numbered keeps at the desk's edge | a fixed ladder of constants read once: the Moremur finalizer |
| groove-dust, intermediate turns, chalk residue swept away | no intermediate value is ever emitted or checkpointed |

**Which assumption each SEED breaks**

| SEED | Assumption broken |
|---|---|
| 1 — stone never resets, every fold carries the callus of all prior folds | *"the whole buffer must be read once, start to end, in order"* — inverted: it **insists** on order, and makes order the source of strength, not a constraint to be relaxed. Breaks nothing else. |
| 2 — the weight is pressed into the already-turned seat; weights **add** | *"mixing one byte requires a multiplication"* → partly, and decisively *"each byte must be mixed into the running state before the next byte is read"*: because seat-then-turn is `(h+w)·P`, a **ring** operation, the sum distributes and a whole revolution's marks can be seated in one motion with the result bit-identical to pressing them one at a time. FNV's `^` has no such law. |
| 3 — only the final seated position leaves the desk | *"more mixing rounds always means better mixing"*. If every intermediate is discarded, intermediate mixing quality is worth nothing; the loop need only be **injective**, and all avalanche belongs in the read-off. |

## CHOSEN SEED

**Seed 2.**

Plainly, as instructed: **none of the three seeds breaks "the state is a single accumulator updated in place, one value."** All three *assert* it — one stone, one desk, never lifted. So I fall back to the most literal seed, and Seed 2 is both the most literal (it is the fold itself, step by step) and the most different from the known way: FNV/xxHash **xor** a word into the state; the native **adds a weight** into a seat. That one substitution is the whole engine — addition is what makes the fold distribute over the turn.

Seed 3 is retained as the *discipline* on the read-off, and Seed 1/5 as the correctness condition (`P` odd ⇒ each fold bijective ⇒ differing prefixes never converge).

## ASSUMPTION BROKEN

> *"each byte must be mixed into the running state before the next byte is read"*

Not by reading out of order, and not by splitting the stone. The fold `h ← (h + w)·P` is the native's, unchanged and strictly serial **as a definition**; but because weights *add*, its closed form is
`h_N = P^N·h_0 + Σ_k P^{N−k}·w_k`,
so where the stone *ends up* can be computed by seating four strided partial terms of that one sum and recombining them with fixed powers of `P`. Those four values are not four accumulators of a redefined hash — they are four partial terms of the single stone's final seat, and the recombination is **bit-identical** to pressing mark after mark. The stone is still one stone; I am computing its trajectory, not cloning it.

Secondary break (Seed 3): the loop carries **no** avalanche. Every bit of diffusion happens once, at the read-off.

This lands on validated ground rather than invention: it is Carter–Wegman/Rabin–Karp **polynomial hashing mod 2⁶⁴** with a published-good LCG multiplier (`0x5851F42D4C957F2D`, Knuth/PCG), evaluated by the standard **strided Horner** reassociation, finished with **Moremur** (Pelle Evensen), a measured-bias-tested 64-bit finalizer. I did not invent a mixer.

**Two regimes, recognized at runtime.** `known_way` names FNV-1a (handfuls) and xxHash (bulk). In-world: laying the groove out for revolutions only pays once the pile actually runs long. `len < 64` → the plain serial fold, cell by cell, zero setup. `len ≥ 64` → the same fold, revolutions folded in one motion. Both paths return the identical value. This is also the guard demanded by my own verdict's stated risk (the reassociation's combine costs ~5 extra multiplies) — it is size-checked with a fallback, not shipped bare. No threads: the metaphor has one stone and one desk, and the loop is multiplier-port-bound, not bandwidth-starved.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------------
   THE DESK.  One bone-colored die-stone, 64 bits, never reset.
   The fold, once per groove-cell:      h <- (h + w) * P      (P odd)
   The read-off, once, at the very end, against the wooden numbered keeps.
   Nothing between the first press and the last reading ever leaves.
   --------------------------------------------------------------------- */

#define STONE_BASIS 1469598103934665603ULL   /* the stone's first seat      */
#define STONE_P     6364136223846793005ULL   /* the turn; odd => bijective  */
#define INVEST      47                       /* flat row -> solid weight    */

/* "I read them as weights ... the way the physicist reads a flat thing
   about to be invested into three."  v ^ (v>>k) is triangular with unit
   diagonal => invertible => no two rows ever invest to the same weight. */
static inline uint64_t invest(const unsigned char *p)
{
    uint64_t v;
    memcpy(&v, p, sizeof v);
    return v ^ (v >> INVEST);
}

/* the wooden numbered keeps: Moremur (P. Evensen), a validated,
   bias-measured 64-bit finalizer.  This reading is the only output. */
static inline uint64_t keeps(uint64_t x)
{
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;

    const uint64_t P1 = STONE_P;
    const uint64_t P2 = P1 * P1;
    const uint64_t P3 = P2 * P1;
    const uint64_t Q  = P3 * P1;          /* P^4 : one full revolution */

    uint64_t h = STONE_BASIS;
    const size_t n = len >> 3;            /* whole groove-cells */
    size_t i = 0;

    if (len >= 64) {
        /* ---- long pile ------------------------------------------------
           The same fold.  Its own sum, reassociated: four strided partial
           seats plus a count of the stone's turns.  Contribution of cell
           k = 4t+j is Q^(m-1-t) * P^(4-j) = P^(4m-k) -- exactly the serial
           fold's coefficient.  Bit-identical, 1 turn per cell, ~5 multiplies
           per 32 bytes instead of 32.                                   */
        const size_t m = n >> 2;
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, pw = 1;

        for (size_t k = 0; k < m; k++) {
            const unsigned char *q = p + (k << 5);
            s0 = s0 * Q + invest(q      );
            s1 = s1 * Q + invest(q +  8 );
            s2 = s2 * Q + invest(q + 16 );
            s3 = s3 * Q + invest(q + 24 );
            pw = pw * Q;                  /* turns taken so far */
        }
        h = h * pw + s0 * Q + s1 * P3 + s2 * P2 + s3 * P1;
        i = m << 2;
    }

    /* ---- handful (and the long pile's remainder): press one at a time */
    for (; i < n; i++)
        h = (h + invest(p + (i << 3))) * P1;

    /* the short last cell, seated as it stands */
    const size_t r = len & 7;
    if (r) {
        uint64_t v = 0;
        const unsigned char *t = p + (n << 3);
        for (size_t j = 0; j < r; j++)
            v |= (uint64_t)t[j] << (j * 8);
        h = (h + (v ^ (v >> INVEST))) * P1;
    }

    /* the pile's count, pressed last: a handful never seats like a century */
    h = (h + (uint64_t)len) * P1;

    /* lift the stone, read it against the keeps, sweep the rest away */
    return keeps(h);
}
```

## PREDICTION

Stated before any measurement.

Cycle accounting (Skylake-class: `imul r64,r64` = latency 3, **throughput 1/cycle, port 1**):

- Reference FNV-1a: `xor`(1) + `imul`(3) on the critical chain, serial → **≈ 4 cycles/byte**.
- Long path: 5 multiplies per 32 bytes, five independent chains (5 ≥ latency 3, so port-throughput-bound, not latency-bound); the `invest` shift/xor pairs ride ports 0/5/6 and are free → **≈ 0.16 cycles/byte** (~22 GB/s @ 3.5 GHz), likely clipped to ~10–15 GB/s by memory for out-of-cache buffers.
- Handful path (`len < 64`): one multiply per **8** bytes instead of per byte → **≈ 0.5 cycles/byte**, still ~8× the reference.

So the floor of the speedup is the small-input path (~8×), the ceiling is the bandwidth-limited large path (~25×).

Avalanche: every cell enters `h` with an odd coefficient `P^{N−k}`, so any single-bit input flip yields a state difference `2^b · odd ≠ 0`; `invest` additionally splits a high-bit flip into bits `b` and `b−47`, removing the worst case where only bit 63 differs. Moremur then expands any nonzero difference. Predicted avalanche ≈ **0.50**.

PREDICTION: speedup_vs_dp = 15

## MEASUREMENT

**Not measured in this session — no tools were available to me here.** I will not dress up the estimate above as a reading. `hash_bench` has not been run against this kernel by me; the pipeline's numbers are the real ones, and the honest status of the 15× figure is *predicted, unverified*.

What I can state without a tool, because it is arithmetic rather than performance:

- The two paths are **provably the same function**. Contribution of cell `k = 4t+j` on the long path is `Q^{m−1−t}·P^{4−j} = P^{4(m−1−t)+4−j} = P^{4m−(4t+j)} = P^{N−k}`, which is exactly the serial fold's coefficient; the carried `h` picks up `pw = Q^m = P^N`. No regime discontinuity, no second definition of the hash.
- Every fold is a bijection (`P` odd, `invest` unit-triangular), so distinct prefixes of equal length never converge — the native's mechanism 5 holds as stated, not approximately.

Where I expect the prediction to be wrong: if the benchmark's buffers are large and cold, the long path will be DRAM-bound and land nearer 10× than 25×; if the benchmark is dominated by tiny inputs (≤ 16 bytes), the fixed `keeps` + length fold is a constant overhead and the ratio collapses toward 3–5×.

## VERDICT

The core of this kernel **is** the native's fold, not a textbook hash wearing its clothes: `h ← (h + w)·P`, one stone, never reset, the weight seated into the already-turned position so earlier marks bend how later ones land, the count pressed last, and a single reading against a fixed ladder of keeps with everything else swept away. The previous attempt substituted xxHash's four seeded lanes and its rotl31 round; there is none of that here. The speed comes from the metaphor's own algebra — *weights add*, and addition distributes over the turn — which is precisely the property FNV's xor destroys.

Honest limitations, named rather than buried:

1. **The accumulation is linear over ℤ/2⁶⁴ in the investeded weights.** Because `invest` is GF(2)-linear while the accumulation is ℤ-linear, the composed map from input bits is nonlinear, which defeats naive difference-cancellation — but an adversary who knows `P` can still mount a polynomial-hash attack. This is a non-cryptographic hash and makes the standard Carter–Wegman trade knowingly. The native's "essentially never seat the same way twice" is true for arbitrary piles, not for piles chosen by someone holding the stone.
2. **The stated risk is guarded.** The reassociation's combine costs ~5 multiplies plus a `pw` chain; below 64 bytes that overhead can exceed the work, so it is size-checked with a fallback to the plain serial fold, which returns the identical value. No unguarded risky path ships.
3. **Not hand-vectorized.** AVX2 has no 64×64→64 multiply, so a SIMD port would cost three 32-bit multiplies per lane and lose to scalar `imul` at 1/cycle. Vectorization effort went instead into `restrict`, 8-byte cell loads, and a 32-byte stride that walks half a cache line per iteration. No OpenMP: the metaphor has one stone, and the loop is port-bound rather than starved.
4. **Endianness.** The cell load is native-endian, so the value differs between big- and little-endian machines. Acceptable for a same-machine hash; it would need a byteswap to be portable.

If the measurement comes back near 1× rather than 15×, the failure will be in the multiplier port being the real ceiling on both kernels — and I would report that as plainly as the success.