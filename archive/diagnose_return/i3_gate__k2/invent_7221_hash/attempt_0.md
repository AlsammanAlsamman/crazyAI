# MAPPING

**SEED 1 — "one crease per mark, angle set by the mark and the crease before it"**

| World object | Problem object |
|---|---|
| mark in the pile | one input byte `data[i]` |
| sheet, flat | the running state |
| one fold | one state update |
| fold angle | the mixing function's output |
| "the crease before it" | strict serial dependency `h_i = f(h_{i-1}, byte_i)` |
| never skipping, never looking ahead | single in-order pass |

*Assumption broken: **none.*** This seed is FNV‑1a restated in paper. It actively **asserts** "each byte must be mixed into the running state before the next byte is read" and "the state is a single accumulator." Stating that plainly: seed 1 is the known way.

**SEED 2 — "test every crease against all four wire birds' beaks at once, refold until they agree"**

| World object | Problem object |
|---|---|
| four wire birds | four independent 64‑bit accumulator lanes |
| each bird's beak | that lane's own private dependency chain |
| pressing the fold's corner against a *different* bird | byte-group *i* goes to lane *i* — not to one shared state |
| "at once" | four multiply chains in flight simultaneously (ILP), not four sequential ones |
| "only when all four beaks agree does the crease count" | the lanes are only reconciled at the merge, never mid-stream |
| refold tighter | rotate-and-remultiply inside each lane |
| spiral + snail-shell markings "line up wrong on purpose, scrambled" | lanes must be merged with *distinct* rotations/primes so lane order is not symmetric |
| "a clean line-up would mean two piles look the same" | symmetric merge ⇒ trivial collisions on permuted input |
| "huge sheet… only that it's large enough" | **runtime regime check**: four-lane path only when `len` is big enough to fill it |

*Assumptions broken: **#1 ("each byte must be mixed into the running state before the next byte is read")** and **#2 ("the state is a single accumulator updated in place, one value")**.*

**SEED 3 — "thin the whole wad at the water's edge to one dense corner, throw away every scrap and misreading"**

| World object | Problem object |
|---|---|
| the folded wad (bulky, not yet a token) | the wide multi-lane state at end of input |
| the water's edge | the finalizer, run exactly once |
| "thins the way a body thins going under" | an irreversible-looking but bijective avalanche (xor-shift / multiply ladder) |
| one hard dense corner no bigger than a coin | the single returned `uint64_t` |
| the small suitcase carried under | `len` folded in, so length is part of the token |
| scraps and failed bird readings thrown in the mud | intermediate lanes discarded, never exported |
| tide-boundary closes over the board | hard read bound at `data + len` — no byte past the end is touched |

*Assumptions broken: **#3 ("mixing one byte requires a multiplication")** — the per-byte work is cheap and concentrated; and **#5 ("more mixing rounds always means better mixing")** — one strong terminal mix beats many weak per-byte ones.*

# CHOSEN SEED

**SEED 2**, with SEED 3 as its terminal step. Seed 2 is the only seed that breaks the preferred assumption (#1), and it is the furthest from the known way: the known way has *one* state that every byte must touch before the next byte is read; four wire birds mean byte 0 and byte 8 are in flight at the same instant, in different states, and never meet until the merge. Seed 1 is rejected because it *is* FNV‑1a.

# ASSUMPTION BROKEN

> *each byte must be mixed into the running state before the next byte is read* — and with it, *the state is a single accumulator updated in place, one value*.

Four birds = four lanes = four disjoint carry-forward chains. The paper still "carries forward everything it has been told," but along four parallel creases instead of one. Honest note required by step 4: this mechanism lands *exactly* on **XXH64's four-accumulator 32-byte stripe loop with a `fmix64`-class finalizer** — a validated, widely deployed technique. I let the metaphor arrive there rather than invent a novel untested variant. The birds' "scrambled on purpose" markings are why the merge uses four *different* rotations (1, 7, 12, 18) and asymmetric lane seeds.

**Regime recognition, encoded in the metaphor itself:** the native's own qualifier — *"pink or orange, doesn't matter, only that it's **large enough**"* — is the runtime dispatch. A sheet too small cannot take a four-bird spiral fold, so it is folded the single-crease way (one state, seed 1's path) and taken straight to the water's edge. That is the `len >= 32` guard with a single-accumulator fallback, which directly addresses the risk my own verdict names (four-lane setup + 4-way merge is pure overhead on short buffers). No thread parallelism: the native carries **one** wad to **one** water's edge, and at realistic hash buffer sizes thread setup would dominate — so per step 4 it is dropped, not hedged.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* The spiral and snail-shell markings: five mutually scrambled primes. */
#define P1 11400714785074694791ULL
#define P2 14029467366897019727ULL
#define P3  1609587929392839161ULL
#define P4  9650029242287828579ULL
#define P5  2870177450012600261ULL

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
/* The tide-boundary: never read past data+len; unaligned-safe, compiles to one mov. */
static inline uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* One bird pecks one crease: angle set by the mark AND the crease before it,
   then refolded tighter (rotate) and pressed again (multiply). */
static inline uint64_t peck(uint64_t acc, uint64_t word) {
    acc += word * P2;
    acc  = rotl64(acc, 31);
    acc *= P1;
    return acc;
}

/* "Only when all four beaks agree does the crease count as set" -
   reconciliation happens here, once, never mid-stream. */
static inline uint64_t agree(uint64_t h, uint64_t bird) {
    bird = peck(0, bird);
    h   ^= bird;
    h    = h * P1 + P4;
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *p   = data;
    const unsigned char *end = data + len;
    uint64_t h;

    /* "only that it's large enough" - regime check. */
    if (len >= 32) {
        const unsigned char *limit = end - 32;

        /* Four wire birds, deliberately scrambled relative to one another so
           that permuting lanes cannot make two different piles agree. */
        uint64_t b1 = P1 + P2;
        uint64_t b2 = P2;
        uint64_t b3 = 0;
        uint64_t b4 = (uint64_t)0 - P1;

        /* Each bird takes its own mark. Four independent creases in flight:
           byte 0 is NOT mixed in before byte 8 is read. */
        do {
            b1 = peck(b1, rd64(p +  0));
            b2 = peck(b2, rd64(p +  8));
            b3 = peck(b3, rd64(p + 16));
            b4 = peck(b4, rd64(p + 24));
            p += 32;
        } while (p <= limit);

        /* The beaks agree - four distinct angles, so the merge is asymmetric. */
        h = rotl64(b1, 1) + rotl64(b2, 7) + rotl64(b3, 12) + rotl64(b4, 18);
        h = agree(h, b1);
        h = agree(h, b2);
        h = agree(h, b3);
        h = agree(h, b4);
    } else {
        /* Sheet too small for the spiral: the single-crease fallback path. */
        h = P5;
    }

    /* The small suitcase carried under the water with it. */
    h += (uint64_t)len;

    /* Scraps trimmed off the edge of the sheet, folded in one at a time. */
    while (p + 8 <= end) {
        uint64_t k = peck(0, rd64(p));
        h ^= k;
        h  = rotl64(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)rd32(p) * P1;
        h  = rotl64(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * P5;
        h  = rotl64(h, 11) * P1;
        p++;
    }

    /* The water's edge: thin the whole wad down to one dense corner.
       Every lane, every scrap, every failed reading is discarded here -
       only the coin survives. */
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}
```

# PREDICTION

Reasoning stated before any measurement. The baseline FNV‑1a is latency-bound: one 64‑bit multiply (3–5 cycle latency) per **byte**, so ≈ 0.2–0.33 bytes/cycle. The four-bird loop issues 8 multiplies per **32 bytes** across four independent chains, so it is throughput-bound at ~2 mul/cycle ⇒ ≈ 4 cycles / 32 bytes ≈ 8 bytes/cycle. Ratio ≈ 24–40× in theory; real code loses to load ports, loop overhead, and (at large sizes) memory bandwidth, and short buffers pay the merge. Discounting hard for that:

**PREDICTION: speedup_vs_dp = 9.0**

Avalanche: predicted ≈ 0.49–0.50 (near-ideal). The `fmix64`-class finalizer alone gives near-ideal bit diffusion on the merged word, and the asymmetric merge prevents lane-permutation aliasing — so I expect avalanche to be *as good as or better than* FNV‑1a, not a trade.

# MEASUREMENT

Not run in this session — no tools were available, so `hash_bench` / `hash_contract` could not be invoked. The prediction above is therefore unconditioned by any data, which is the honest state of affairs. Reporting what the pipeline should find, and what would count as failure:

- **Throughput, large buffers (≥ 1 KiB):** expect 8–15× the FNV‑1a baseline. Below ~5× would falsify my latency-vs-throughput model and mean the baseline is not actually latency-bound (e.g. the compiler vectorized it, or the bench is bandwidth-limited).
- **Throughput, small buffers (< 32 B):** expect **parity**, not a win — the fallback path is deliberately the same shape as the baseline. If this regresses, the guard is mis-sized.
- **Avalanche:** expect ≈ 0.50. Anything below ~0.45 means the finalizer is being undersold by the merge and the mechanism has failed on its stated quality goal.

# VERDICT

The native's four wire birds are not decoration — they are the exact structural break the problem needed. The known way's bottleneck is not arithmetic volume but the **single crease**: one accumulator serializes one multiply latency per byte. "Press each fold against a *different* bird's beak, and only when all four agree does the crease count" says, literally, *keep four states, reconcile late* — which is the whole of instruction-level parallelism in hashing. Seeds 1 and 3 were real but subordinate: seed 1 is the known way with no break in it, and seed 3 (one terminal thinning instead of per-byte grinding) is the finalizer that makes late reconciliation safe.

Two honesty points.

First, this mechanism converges on XXH64's established structure rather than something new. I take that as the correct outcome under step 4, not a disappointment: the metaphor earned the technique instead of copying it, and a validated construction beats a novel one. The place the metaphor added something the textbook framing hides is the "scrambled on purpose" clause — it is the *reason* the four lanes must be merged with four different rotations and seeded asymmetrically, which is precisely the detail a naive four-lane rewrite gets wrong (symmetric lane merge ⇒ permuted-block collisions).

Second, the risk I named is addressed in the artifact, not just flagged: the four-lane path has real setup and 4-way merge cost, so it is gated behind `len >= 32` with a single-accumulator fallback — the native's own *"only that it's large enough."* Thread parallelism is deliberately absent: the metaphor gives one wad and one water's edge, and at realistic hash input sizes OpenMP setup would exceed the entire hash, so I dropped it rather than ship a guarded mechanism I have no size evidence for.

Remaining uncertainty I cannot resolve without the bench: whether the benchmark's buffer sizes sit in the stripe-loop regime at all. If it hashes mostly short keys, nearly all of my win evaporates by construction — the fallback is the baseline — and the measured speedup will be near 1.0 regardless of how sound the four-bird loop is. That would not be a flaw in the mechanism, but it would make the 9.0 prediction wrong, and I would report it as wrong.