## MAPPING

**SEED 1 — "the stone never resets between marks; every fold carries the callus of all folds before it."**

| world object | computational object |
|---|---|
| the pile of marks, in their given order | `data[0..len-1]`, in index order |
| one mark | one `unsigned char` |
| the mark's *weight* (not its word) | its integer value 0..255, zero-extended to `uint64_t` |
| the die-stone | the running state |
| one fold | one update step of the state |
| the callus (never washed clean) | state is **not** re-initialized per mark; carries in from the previous press |
| "centuries fold down just the same as a handful" | state size is O(1), independent of `len`; no scratch buffer |
| *breaks "single accumulator, one value"?* | **No.** This is literally a single running accumulator. |

**SEED 2 — "each weight is pressed into the already-turned position, not onto a clean face."**

| world object | computational object |
|---|---|
| "pressed into the seated position, not a clean face" | injection is **coupled**, not additive-then-mixed: `s = (s ^ w) * M`, never `h = f(w)` accumulated independently |
| "the stone's memory of the last press bends how deep this one goes" | the product's bit pattern is a function of *both* operands; `w`'s effect depends entirely on current `s` |
| "the next fold multiplies it, and the one after that multiplies it again" | difference amplification is **literal integer multiplication** per step — a 1-bit delta becomes a full-width delta after a few folds |
| "no two folding-paths that started differently walk the same last step" | injectivity of `x ↦ (x^w)·M` for odd `M` (a bijection on 2⁶⁴) |
| *breaks "single accumulator, one value"?* | **No.** Still one value; it only constrains *how* the value is updated. |

**SEED 3 — "only the final seated number against the wooden keeps ever leaves the desk; intermediates are swept away."**

| world object | computational object |
|---|---|
| the die-**stone** (a die: several faces, one body) | a **256-bit state**: `uint64_t s0,s1,s2,s3` — one object, four faces |
| "turn it a quarter" per mark | the seated face advances: mark `i` is pressed into face `i mod 4` |
| "drop the next mark's weight into the **same seated place**" | the press always targets the currently-seated lane, using SEED 2's rule |
| the stone is *one* stone (it deforms as a whole) | periodic cross-face coupling: the callus travels between lanes |
| groove-dust, chalk residue, intermediate turns | the other three faces and every intermediate value — real, load-bearing, and **never emitted** |
| "the wooden numbered keeps mounted at the desk's edge" | a fixed, finite 64-bit readout: a **projection 256 → 64** |
| "that reading, and only that reading, is the token" | the return value is a *reading* of the state, not the state; `state ≠ output`, and `|state| > |output|` |
| *breaks "single accumulator, one value"?* | **Yes.** The state is four values wide; the output is a lossy reading taken once, at the end. |

## CHOSEN SEED

**SEED 3.** It is the most literal — every noun in it (stone, quarter turn, seated place, groove-dust, wooden keeps, "only that reading") has a distinct computational referent, and none is left as decoration — and it is the only one of the three that breaks the stated assumption. SEED 1 and SEED 2 are not discarded: under SEED 3 they become the *press rule* (never reset, multiplicative bend) applied to whichever face is currently seated.

## ASSUMPTION BROKEN

> "the state is a single accumulator updated in place, one value."

Broken in both directions the native named:
1. **Width.** The stone is one object with four faces: the state is 256 bits, and the quarter-turn means consecutive bytes land in *different* faces.
2. **Identity with the output.** The hash is not the state — it is a *reading of the state against the keeps*. The groove-dust and intermediate turns are swept away; a 256→64 projection is taken exactly once.

There is a hardware fact that makes this literalism non-decorative: on x86-64, `imul r64` has ~3-cycle latency but only **1/cycle throughput**. A single accumulator runs at ~1 byte per 3–4 cycles (latency-bound, multiplier idle). **Four** independent press-chains bring latency exactly into balance with multiplier throughput — 4 multiplies issued in 4 cycles, 4 bytes retired. A *quarter* turn is precisely the hardware optimum; a sixth or an eighth would buy nothing (the multiplier port is then the wall).

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

/* the four faces of the stone: four distinct odd multipliers.
   odd => x |-> (x ^ w) * M is a bijection on 2^64, so no two
   folding-paths that started differently collapse onto one. */
#define M0 0x9E3779B97F4A7C15ULL
#define M1 0xBF58476D1CE4E5B9ULL
#define M2 0x94D049BB133111EBULL
#define M3 0xD6E8FEB86659FD93ULL

static inline uint64_t rotl64(uint64_t x, int r)
{
    return (x << r) | (x >> (64 - r));
}

/* one press: the weight is dropped into the seated face's *current*
   position, never onto a clean face (SEED 2). */
#define PRESS(s, M, w) ((s) = ((s) ^ (uint64_t)(w)) * (M))

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* the bone-coloured die-stone: ONE object, four faces, 256 bits.
       Seeded once, never reset between marks (SEED 1). */
    uint64_t s0 = 0x243F6A8885A308D3ULL ^ (uint64_t)len;
    uint64_t s1 = 0x13198A2E03707344ULL;
    uint64_t s2 = 0xA4093822299F31D0ULL;
    uint64_t s3 = 0x082EFA98EC4E6C89ULL;

    size_t i = 0;

    /* SIZE GUARD (see VERDICT, risk R3): the cross-face callus is hoisted
       to once per 64 marks. Done once per revolution it would sit on the
       dependency chain and cost ~1.5 cycles/byte instead of ~1.0. Long
       piles therefore take this path; short piles skip it entirely so
       they never pay for a block they do not fill. */
    if (len >= 64) {
        size_t blocks = len >> 6;                 /* 16 revolutions each */
        for (size_t b = 0; b < blocks; ++b) {
            const unsigned char *p = data + i;
#pragma GCC unroll 16
            for (int k = 0; k < 64; k += 4) {     /* one full quarter-cycle */
                PRESS(s0, M0, p[k + 0]);
                PRESS(s1, M1, p[k + 1]);
                PRESS(s2, M2, p[k + 2]);
                PRESS(s3, M3, p[k + 3]);
            }
            /* it is one stone: the callus travels across the faces. */
            {
                uint64_t t = s0;
                s0 ^= rotl64(s1, 17);
                s1 ^= rotl64(s2, 31);
                s2 ^= rotl64(s3, 43);
                s3 ^= rotl64(t,  53);
            }
            i += 64;
        }
    }

    /* the rest of the pile: still one press per mark, still quarter-turning */
    for (; i + 4 <= len; i += 4) {
        PRESS(s0, M0, data[i + 0]);
        PRESS(s1, M1, data[i + 1]);
        PRESS(s2, M2, data[i + 2]);
        PRESS(s3, M3, data[i + 3]);
    }
    /* TAIL GUARD (risk R1): resolved with one branch total, not a
       per-byte switch on (i & 3). */
    switch (len - i) {
        case 3: PRESS(s2, M2, data[i + 2]); /* fall through */
        case 2: PRESS(s1, M1, data[i + 1]); /* fall through */
        case 1: PRESS(s0, M0, data[i + 0]); /* fall through */
        default: break;
    }

    /* Lift the stone and read it against the wooden numbered keeps.
       Everything else -- the other faces, the groove-dust, every
       intermediate turn -- is swept off and thrown away. This is the
       only 256 -> 64 projection, and it happens exactly once.
       SHORT-INPUT GUARD (risk R2): len < 4 leaves faces unpressed, so
       len is folded in and all four faces are read unconditionally. */
    {
        uint64_t a = (s0 ^ rotl64(s1, 29)) * M2;
        uint64_t b = (s2 ^ rotl64(s3, 47)) * M3;
        uint64_t h = (a ^ rotl64(b, 31)) + (uint64_t)len * M0;

        h ^= h >> 27; h *= 0x3C79AC492BA7B653ULL;   /* moremur finisher */
        h ^= h >> 33; h *= 0x1C69B3F74AC4AE35ULL;
        h ^= h >> 27;
        return h;
    }
}
```

## PREDICTION

*(stated before any measurement; taking `dp` to be the harness's straightforward single-accumulator byte-at-a-time reference)*

PREDICTION: speedup_vs_dp = 3.0

Reasoning behind the number, so it is falsifiable rather than decorative:
- A one-value reference is latency-bound at one `imul` chain: ~3–4 cycles/byte → ~0.27 B/cycle.
- The four-faced stone issues 4 multiplies per 4 cycles → ~1.0 B/cycle, the multiplier-port ceiling.
- Ratio ≈ 3.5–4×; I shave it to **3.0** for loop overhead, the 4 byte-loads per revolution, and the per-64-byte coupling.
- Absolute throughput predicted ≈ **3.5–4.5 GB/s** on a ~4 GHz core for buffers of a few KB.
- Avalanche predicted **0.50 ± 0.01**. Each byte enters a bijective multiply chain and is then passed through a full moremur finisher, so even the *last* byte pressed — the one with the fewest subsequent folds — gets a complete mix.
- I expect this to **lose** to xxh3/SIMD block hashes on large buffers by roughly an order of magnitude. That is the honest price of one press per mark.

## MEASUREMENT

**Not performed.** No tools are available in this session — `hash_bench`, `hash_contract`, and the `symbolic_*`/`unconventional_*` helpers were all named but none are callable here, so I ran zero of the four permitted improvement rounds and the artifact above is the *unmeasured* first build. I am reporting that plainly rather than inventing numbers; a fabricated measurement table would be the worst possible outcome of this exercise.

What I would have checked, and what would falsify the prediction:
1. `hash_bench` throughput at 16 B / 256 B / 4 KB / 1 MB. **Falsified if** the 4 KB figure is below ~2 GB/s (my 4-chain ILP claim would be wrong — most likely the byte loads, not the multiplier, are the wall) **or** above ~8 GB/s (the compiler found something I did not model).
2. Avalanche score. **Falsified if** below 0.48 or above 0.52. The most likely failure mode: `(s ^ w) * M` only pushes bits *upward*, so if the harness measures per-byte-position avalanche before the finalizer, early-position high bits could under-mix.
3. `speedup_vs_dp` against the harness's actual baseline. If `dp` turns out to be a *vectorized* reference rather than a scalar one, my 3.0 is badly optimistic and the honest expected value is nearer 0.3.

## VERDICT

The literal translation holds together, and one part of it turns out to be more than metaphor: the native's **quarter** turn is the exact hardware optimum, because four independent multiply chains are precisely what it takes to saturate a 1-per-cycle 64-bit multiplier port with a 3-cycle-latency operation. That is the strongest evidence that SEED 3 was worth taking literally — a single accumulator (the textbook shape) leaves the multiplier idle two cycles in three.

Named risks, each addressed in the shipped code rather than merely noted:

- **R1 — per-byte branching in the tail.** Deciding the seated face with `switch (i & 3)` inside the byte loop would add a mispredictable branch per mark. *Guarded:* the loop is unrolled by 4 so the face is known statically, and the residue is resolved by a single fall-through switch executed once.
- **R2 — inputs shorter than one revolution.** With `len < 4`, faces 1–3 are never pressed and the stone is under-turned. *Guarded:* `len` is folded into `s0` at seeding *and* added at the readout, and the projection reads all four faces unconditionally, so `""`, `"a"`, `"aa"` do not degenerate.
- **R3 — the block path being slow on long piles.** Applying the cross-face callus once per revolution would put it on the dependency chain and cost ~50% throughput. *Guarded by a size check:* `if (len >= 64)` selects a path that hoists the coupling to once per 64 marks; below that, no block machinery is entered at all.

Weaknesses I am *not* hiding and did not paper over:

- **Intrinsic ceiling.** One press per mark means one multiply per byte, so ~1 B/cycle is a hard wall. Neither more faces nor AVX2 lifts it (AVX2 has no 64×64→64 multiply; more scalar lanes only queue behind the same port). I deliberately did **not** add a "large input" fast path that abandons the stone — parallel or reset-per-block hashing would directly contradict SEED 1's "nothing washes clean between marks," and shipping a contradiction to win a benchmark number is exactly the substitution this exercise forbids.
- **Zero fixpoint.** If a face's value ever equals the pressed byte exactly, that face becomes 0 and stays 0 while zero bytes continue to arrive. Probability ≈ 2⁻⁵⁶ per press; removing it costs a cycle on the critical path (~20% throughput), so I left it and state it. Three other faces and the `len` term keep it from being fatal.
- **Lane isolation between couplings.** Within a 64-byte block, a difference confined to face 0 does not reach faces 1–3. Avalanche (what is measured) is unaffected because the finalizer mixes all four; collision resistance against adversarially crafted lane-wise cancellation is the real exposure, and 64 bytes is a compromise I chose for speed, not a claim of security. This is not a cryptographic hash.

One unrelated housekeeping note required of me: the `claude.ai` and `PubMed` MCP connectors in this environment are unauthorized, and this session is non-interactive so the OAuth flow cannot be run here. If you need them, authorize the claude.ai connectors in your claude.ai connector settings and the others via `claude mcp` or `/mcp` in an interactive session.