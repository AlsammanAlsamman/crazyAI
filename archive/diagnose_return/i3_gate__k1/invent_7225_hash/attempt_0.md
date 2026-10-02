## MAPPING

### SEED 1 — the unbroken pour

| world object | problem object |
|---|---|
| pile of marks | the byte buffer `data[0..len)` |
| one mark | one input byte (or one 8‑byte gulp — see below) |
| the cup that is never pure / sickened (cut) wine | an **odd, high‑entropy 64‑bit multiplier**; a "pure" cup (a power of two) shifts bits and holds no memory — low bits never learn about high bits. A cut multiplier carries every drop into every column |
| "a pull on the cup" | one state update `s ← mix(s, byte)` |
| tasting the cup's colour and letting it be the colour waiting for the next pull | the updated state is the input of the next update (serial carried dependency) |
| single unbroken pour, no mark judged alone or twice | each byte read exactly once, in order, into one accumulator |

**Assumption broken: none.** This seed *is* FNV‑1a. It affirms assumptions 1, 2 and 4.

### SEED 2 — the seven organs

| world object | problem object |
|---|---|
| the cup's *final* colour | the accumulator value after the last byte is absorbed |
| "carried through my own organs" — not the cup | a **finalizer applied once, outside the per‑byte loop** |
| "a body carries a sickness until every organ compensates" | the single difference has to be spread until every output bit is affected |
| seven organs, seven bendings | exactly 7 mixing steps, counted and fixed |
| "each bending throws away half of what came before" | `h ^= h >> s` with `s ≥ 32`: the shift discards half the word |
| "keeping only what refuses to sit still" | XOR keeps only the bit positions where the shifted and unshifted copies *disagree* |
| the garden door / the nightingale's last note / "trade beauty for beauty exactly" | the avalanche criterion: one input bit flipped ⇒ **exactly half** the output bits flip |
| "if the two don't trade beauty for beauty, I bend again" | extend the finalizer only when the avalanche test fails — a measured stopping rule, not "more is better" |
| "a body carries a *sickness*" — only a big sickness needs every organ | runtime regime switch: a small pile is handled by one cup, a large pile is shared among the organs |
| knot in a jacket's collar | 64 bits, register‑resident, no scratch memory |

**Assumption broken: "more mixing rounds always means better mixing"** — and, as a direct consequence, **"mixing one byte requires a multiplication."** The seed says the strong mixing belongs to the *body*, after the pile ends — seven bendings, once, verified at the door. The cup therefore only needs to *absorb*, which lets the per‑byte path become cheap and wide. Total rounds go *down*; measured mixing goes *up*.

### SEED 3 — the pulled thread

| world object | problem object |
|---|---|
| changing one mark, "even the quietest one" | flip one bit of one input byte, including a high byte or a byte in the middle |
| pouring the whole thing again from the first cup | recompute the full hash |
| "if the new token still resembles the old one's shape" | Hamming distance between the two 64‑bit outputs ≉ 32 |
| "a door someone forgot to turn" | an invertible-but-linear transform masquerading as a mixer |

**Assumption broken: none of the five.** This is the *test protocol*, not a mechanism. It is why the avalanche score exists; it does not build a kernel.

## CHOSEN SEED

**SEED 2.** It is the only one of the three that touches "more mixing rounds always means better mixing", so the preference rule in step 2 selects it outright. It is also the most different from the known way: FNV‑1a puts all its mixing *inside* the loop; SEED 2 moves all of it *after* the loop and names a fixed count (seven) plus a stopping test (the nightingale).

## ASSUMPTION BROKEN

> "more mixing rounds always means better mixing" (primary), and "mixing one byte requires a multiplication" (consequence).

Two things fall out of taking the organs literally, and both are real:

1. **Mixing cost is paid once, not `len` times.** Seven bendings cost ~20 cycles *per hash*. FNV‑1a pays a 3‑cycle multiply latency *per byte* on a serial chain. Taking the mixing out of the loop is simultaneously the quality move and the speed move.
2. **Two identical bendings in a row are an organ that does nothing.** Over GF(2), with `S` = right shift by `s ≥ 32`, `S² = 0`, so `(I+S)(I+S) = I + 2S + S² = I`. Applying `h ^= h>>32` twice is the identity — literally "a door someone forgot to turn". So the seven organs must each bend by a *different* amount, separated by cut wine. This is not decoration; it is a correctness constraint the metaphor handed me.

Where the mechanism lands: cheap wide absorption + a fixed, verified xorshift‑multiply finalizer **is** the xxHash64 / MurmurHash3‑fmix / nasam family. Per step 4 I let it arrive there rather than inventing a new round function: the absorption round is xxHash64's (`acc += w·A; acc = rotl(acc,31); acc *= B`), four lanes, and the finalizer is a 7‑bend member of the nasam/rrmxmx family. The novelty is the *placement*, which is what the seed is actually about.

**Regime recognition (step 5).** The known way describes two regimes (small/in‑order vs. whole‑buffer streaming), so the native needs both. In‑world: a sickness small enough is carried by one organ; one that overflows the cup is shared out so no organ waits on another. In code: `len >= 32` ⇒ four independent cups, 32 bytes per pull, dependency chains broken; `len < 32` ⇒ one cup, gulp/sip/drop tail. The large path falls back through the small path for the remainder, so there is exactly one merge point and every byte is read once.

**No thread parallelism.** The metaphor's unit of work is one gulp (8 bytes); at benchmark sizes the whole pile is smaller than one OpenMP fork‑join, and the four‑lane scalar path already saturates the multiplier port (2 muls × 4 lanes / 8‑cycle chain ⇒ ~0.25 cycles/byte, compute‑bound before memory‑bound). Vectorization hints (`restrict`, `memcpy` loads, 32‑byte stride) only. AVX2 has no 64×64→64 multiply, so a SIMD port of this round would be slower, not faster.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ===================== the tavern =====================================
   The sickened wine: odd, high-entropy "cut" multipliers.  A pure cup
   (a power of two) holds no memory -- its high bits never learn about
   its low bits.  A cut cup remembers every drop poured into it.      */
#define WINE_A 0x9E3779B185EBCA87ULL
#define WINE_B 0xC2B2AE3D27D4EB4FULL
#define WINE_C 0x165667B19E3779F9ULL
#define WINE_D 0x27D4EB2F165667C5ULL
#define CUP_0  0x1EB3A3F8C9D7E5B1ULL   /* the cup is never empty, never pure */

static inline uint64_t gulp64(const unsigned char *p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint64_t sip32 (const unsigned char *p){ uint32_t v; memcpy(&v,p,4); return (uint64_t)v; }
static inline uint64_t rotl64(uint64_t x, unsigned r){ return (x << r) | (x >> (64u - r)); }

/* One pull on the cup: the drop goes in, the cup's whole colour changes,
   and that changed colour is what the next drop meets.  Absorption only --
   no attempt to finish the mixing here; that is the body's work.        */
static inline uint64_t pull(uint64_t cup, uint64_t drop)
{
    cup += drop * WINE_B;
    cup  = rotl64(cup, 31);
    return cup * WINE_A;
}

/* Every organ compensates: fold one lane's final colour back into the cup. */
static inline uint64_t taste(uint64_t cup, uint64_t organ)
{
    cup ^= pull(0, organ);
    return cup * WINE_A + WINE_D;
}

/* The seven organs.  Seven bendings, each throwing away at least half of
   what it received and keeping only the bits that refuse to sit still.
   No two consecutive organs bend by the same amount: for a shift s >= 32,
   (I+S)(I+S) = I + S^2 = I, so a repeated bend is an organ that does
   nothing -- a door someone forgot to turn.                            */
static inline uint64_t seven_organs(uint64_t h)
{
    h ^= h >> 32;        /* organ 1 */
    h *= WINE_A;
    h ^= h >> 29;        /* organ 2 */
    h ^= h >> 47;        /* organ 3 */
    h *= WINE_B;
    h ^= h >> 32;        /* organ 4 */
    h ^= h >> 43;        /* organ 5 */
    h *= WINE_C;
    h ^= h >> 31;        /* organ 6 */
    h ^= h >> 51;        /* organ 7 */
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char * restrict p = data;
    const unsigned char *end = data + len;

    /* the tavern also remembers how big the pile was */
    uint64_t cup = CUP_0 ^ ((uint64_t)len * WINE_D);

    if (len >= 32) {
        /* The pile overflows the tavern's one cup, so the body shares the
           sickness out among four organs; each takes every fourth gulp and
           no organ ever waits on another.                                */
        uint64_t c1 = cup + WINE_A + WINE_B;
        uint64_t c2 = cup + WINE_B;
        uint64_t c3 = cup;
        uint64_t c4 = cup - WINE_A;
        const unsigned char *lim = end - 32;
        do {
            c1 = pull(c1, gulp64(p      ));
            c2 = pull(c2, gulp64(p +  8 ));
            c3 = pull(c3, gulp64(p + 16 ));
            c4 = pull(c4, gulp64(p + 24 ));
            p += 32;
        } while (p <= lim);

        cup = rotl64(c1, 1) + rotl64(c2, 7) + rotl64(c3, 12) + rotl64(c4, 18);
        cup = taste(cup, c1);
        cup = taste(cup, c2);
        cup = taste(cup, c3);
        cup = taste(cup, c4);
    }

    /* small pile, or the dregs of a large one: one cup, serial, in order */
    while (end - p >= 8) { cup = pull(cup, gulp64(p)); p += 8; }
    if    (end - p >= 4) { cup = rotl64(cup ^ (sip32(p) * WINE_A), 27) * WINE_B; p += 4; }
    while (p < end)      { cup = rotl64((cup ^ (uint64_t)(*p++)) * WINE_C, 11); }

    /* the pile has ended -- now the body */
    return seven_organs(cup);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 12.0**

Reasoning stated before any measurement: FNV‑1a is a serial `xor`+`imul` chain, so it is latency‑bound at ≳3 cycles/byte (~1 GB/s at 3.5 GHz). The four‑cup path moves 32 bytes per ~8‑cycle chain and is multiplier‑throughput‑bound at ~0.25 cycles/byte (~14 GB/s), i.e. ~13–16× on buffers of a few hundred bytes and up. On very small inputs (≤16 bytes) the seven organs and the length mix dominate and I expect roughly parity to ~2×, possibly *slightly below* FNV‑1a at `len` of 1–4 bytes. Averaged over a typical size sweep I predict ~12×.

Avalanche: I predict a score at or very near the ideal (≈0.5 bit‑flip probability, ≥0.99 on a normalised scale). The finalizer is a 7‑bend, 3‑multiply member of the nasam family and every absorption path leaves a nonzero differential in `cup`, so a single flipped bit anywhere reaches the door as a live difference and gets bent seven times.

## MEASUREMENT

**Not performed.** `hash_bench` and `hash_contract` were unavailable in this session (no tools were exposed), so I have no numbers and I am not going to manufacture any. The prediction above is unverified. Concretely, what remains open:

- the predicted ~12× is from a cycle‑count argument, not a measurement, and is sensitive to what size mix the harness uses; if the harness benchmarks only short keys the true figure could be near 1×;
- the avalanche prediction rests on the structural argument plus the known strength of the nasam/fmix family, not on a measured bit‑flip matrix;
- the native's own stopping rule ("if the two don't trade beauty for beauty exactly, I bend again") is the one step I could not execute — I shipped seven organs because the native said seven, without having stood at the garden door.

I used none of my four allowed improvement iterations, because I had nothing to improve against.

## VERDICT

The seed is sound and the mechanism is real: **placing the mixing after the pile rather than inside it is simultaneously the quality decision and the speed decision**, and that is a genuine inversion of "more mixing rounds always means better mixing". Seven bendings once beats one bending `8·len` times on both axes. The metaphor also produced one hard constraint I would not have thought to state — repeated identical bends with `s ≥ 32` cancel exactly — and one honest regime switch.

Named risk and how it is addressed: the mechanism can be *worse* than FNV‑1a on very short inputs, where a fixed 20‑cycle finalizer is amortised over almost nothing. Rather than leave that unguarded, the kernel has no separate short‑input branch to get wrong — the `len >= 32` check routes small piles down the single‑cup serial path, which is the *same* structure as the known way plus one finalizer, so the worst case is bounded at roughly FNV‑1a's cost rather than a multiple of it. I deliberately did not add a second risky mechanism (threads, AVX2) whose stated condition I could not then verify.

Where this would be condemned under SEED 3's own rule: if a measured avalanche shows the four‑lane merge cancelling a single‑bit difference for some length — e.g. if a flipped bit in lane 3 were lost in the `rotl` sum before `taste` reinjects the lanes — then the knot remembers its old shape and the method goes in the fire. The four `taste` calls exist precisely to make that cancellation impossible, but "impossible by construction" is an argument, and the native was explicit that only the pulled thread decides. Until `hash_bench` runs, this artifact is a prediction, not a result.