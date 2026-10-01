## MAPPING

**SEED 1 — the unbroken pour**

| World object | Problem object |
|---|---|
| a mark in the pile | one input byte `data[i]` |
| the pile | the buffer, `len` bytes |
| the cup that is never pure / sickened wine | a 64-bit register state, initialised to a non-zero "impure" constant, and an **odd** multiplier (never even, never a power of two = never pure) |
| "pure wine holds no memory" | an even/zero multiplier or zero seed loses bits — information dies |
| "wine that is cut remembers every drop **added** to it" | the state accumulates by **addition** of each stained drop (carries = memory) |
| a pull on the cup | exactly one state-update step per byte |
| tasting the colour, that colour waiting for the next pull | the updated state is the input to the next byte's step (serial data dependency) |
| single unbroken pour, no mark judged alone or twice | one pass, one round per byte, no lanes, no re-reads, no parallel accumulators |

**SEED 2 — seven organs**

| World object | Problem object |
|---|---|
| the cup's last colour | the absorbed state after the final byte |
| an organ | one finalisation stage |
| seven organs, once each | exactly 7 stages, no more, no fewer |
| bending | multiply by a distinct odd 64-bit constant |
| throwing away half of what came before | right-shift by ≈32 (half the width is discarded) |
| keeping only what refuses to sit still | XOR with that shifted copy — only differing bits survive |
| "small enough to knot into a jacket's collar" | the 64-bit return value |
| nightingale's note, trade beauty for beauty exactly | the 0.5 bit-flip target of the avalanche score |

**SEED 3 — the pulled thread**

| World object | Problem object |
|---|---|
| change one mark, even the quietest one | flip one input bit, including the LSB of the last byte |
| pour the whole thing through again from the first cup | recompute the whole hash, not a shortcut |
| new token resembles the old one's shape | output bit-flip fraction far from 0.5 |
| throw the whole method away | reject the design — this seed is the *test*, not the mechanism |

## CHOSEN SEED

**SEED 1.** It is the only one that fixes the mechanism over the *input* (seeds 2 and 3 are the finaliser and the test harness, and I use them as such — literally: 7 organs, and the bit-flip criterion). It is also the seed that breaks the assumption.

## ASSUMPTION BROKEN

Broken assumption: **"more mixing rounds always means better mixing."**

SEED 1 says *no mark is ever judged alone or twice* — exactly **one** cheap step per byte, the minimum possible for a design that still remembers order. All mixing quality is deferred to a fixed, size-independent finaliser. The bet is: per-byte rounds buy you almost nothing for avalanche (a nonzero state difference is a nonzero state difference), and they cost you linearly in time; rounds spent *after* the pile is exhausted cost O(1). So the correct allocation is 1 round/byte + 7 rounds total, not k rounds/byte.

Second, subtler consequence I took literally: the impure wine must touch **every** drop (`b * WINE + LEES`), but that multiply depends only on the byte, so it sits **off** the serial chain. Only a rotate and an add are on the chain → 2 cycles/byte instead of the 4–5 of the textbook byte-at-a-time multiply chain (FNV-1a). Fidelity here bought speed, not cost.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

/* ---- the cup that is never pure ---- */
#define CUP  0x243F6A8885A308D3ULL  /* the wine is already cut before the first mark */
#define WINE 0x9E3779B97F4A7C15ULL  /* the sickened wine: odd, never pure           */
#define LEES 0x6A09E667F3BCC909ULL  /* dregs: no drop ever enters the cup pure      */
#define TILT 23                     /* the cup tips between pulls (gcd(23,64)==1)   */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64 - r));
}

/* ONE PULL: one mark enters the cup, the cup's colour changes, and that
   colour is exactly what the next pull starts from. The wine stains the
   drop (multiply, off the critical path); the cup remembers every drop
   added to it (add, 1 cycle); the cup tips (rotate, 1 cycle).            */
static inline uint64_t pull(uint64_t s, unsigned char b) {
    return rotl64(s, TILT) + ((uint64_t)b * WINE + LEES);
}

/* SEVEN ORGANS: bend, then throw away half of what came before,
   keeping only what refuses to sit still. Exactly seven, size-independent. */
static inline uint64_t organs(uint64_t x) {
    x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;  /* 1 */
    x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 29;  /* 2 */
    x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 32;  /* 3 */
    x *= 0x94d049bb133111ebULL; x ^= x >> 30;  /* 4 */
    x *= 0xd6e8feb86659fd93ULL; x ^= x >> 31;  /* 5 */
    x *= 0xa0761d6478bd642fULL; x ^= x >> 32;  /* 6 */
    x *= 0x2545f4914f6cdd1dULL; x ^= x >> 28;  /* 7 */
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t s = CUP;
    size_t i = 0;

    /* size guard: the unrolled pour only where the loop overhead it removes
       is worth its setup. Same arithmetic, same single chain, no lanes.    */
    if (len >= 32) {
        size_t n = len & ~(size_t)7;
        for (; i < n; i += 8) {
            s = pull(s, data[i + 0]);
            s = pull(s, data[i + 1]);
            s = pull(s, data[i + 2]);
            s = pull(s, data[i + 3]);
            s = pull(s, data[i + 4]);
            s = pull(s, data[i + 5]);
            s = pull(s, data[i + 6]);
            s = pull(s, data[i + 7]);
        }
    }
    for (; i < len; ++i)          /* the tail of the pile, same single cup */
        s = pull(s, data[i]);

    return organs(s);             /* carry the last colour through the body */
}
```

No OpenMP, no `immintrin.h`, no scratch memory — all three are available and all three were rejected on purpose: every one of them requires splitting the pile into more than one cup, which is precisely the hypothesis under test ("a single unbroken pour"). Using them would have been quietly replacing the native's idea with the textbook one.

## PREDICTION

PREDICTION: speedup_vs_dp = 1.8

(Stated before any measurement; no measurement was possible in this session — see MEASUREMENT.)

Reasoning behind the number, so it can be scored honestly:
- Critical path per byte = `rotl` (1c) + `add` (1c) = **2 cycles/byte** ≈ 2 GB/s at 4 GHz. Per-byte uop count ≈ 5 (load/movzx, imul, add, rot, xor-free add) → ~1.25 c/B issue-limited, so the dependency chain, not issue width, is the wall.
- A direct byte-at-a-time reference (`s = (s ^ b) * K`) has a 4-cycle chain → I expect ≈2× on the absorb loop.
- Fixed finaliser cost ≈ 7 × (imul 3 + shift 1 + xor 1) ≈ **35 cycles**, so the 2× erodes on short inputs: ≈1.0–1.2× at len 16, ≈1.8× at len 256, ≈2.0× at len ≥ 4096.
- Avalanche: I predict **0.495–0.505** (essentially ideal). Any single input bit flip yields a nonzero state difference (cancellation would require an exact carry cascade), and 7 multiply-xorshift stages is roughly twice what is needed to scatter a nonzero 64-bit difference. If the harness reports a 0–1 "score" where 1 is ideal, I predict **> 0.99**.

## MEASUREMENT

**Not measured.** `hash_bench`, `hash_contract` and the symbolic/unconventional tools were listed but no tool is actually callable in this session, so I have run nothing: no throughput number, no avalanche number, no compile. I am reporting that plainly rather than inventing figures, and I made **zero** of the four permitted improvement rounds, because an improvement round without a measurement is not an improvement, it is a guess.

What the pipeline should find if my model of the machine is right, and what would falsify it:
- throughput ≈ 2 cycles/byte at large len → falsified if it comes out at ≈4 c/B (would mean gcc put the `imul` on the chain, or `rotl` + `add` are not fusing as predicted — fix: check the asm for `rol`/`add` back-to-back);
- avalanche ≈ 0.5 → falsified by anything below ≈0.48, which would mean the 7-organ finaliser is not the bottleneck I think it is and the absorb chain is leaking structure;
- `speedup_vs_dp` ≈ 1.8 → falsified downward if the reference is vectorised (then expect ≈0.2–0.3×, and the honest conclusion is that the native's design loses on throughput while winning on avalanche).

## VERDICT

The translation is fully literal: mark = byte, cup = one 64-bit register, impurity = odd constants that touch every drop, pull = exactly one state step per byte, tasting = the serial dependency, seven organs = seven bend-and-discard-half stages, jacket collar = `uint64_t`, pulled thread = the avalanche criterion. Nothing was substituted.

Named risks and what I did about each — no stated risk is left unaddressed:

1. **Serial chain on large buffers.** This is inherent: the pour cannot be split without destroying SEED 1. I addressed it *within* the design rather than around it, by driving the critical path to its floor — the byte's multiply is off-chain, so the chain is 2 cycles, not the 4–5 of the textbook chained-multiply hash. I did **not** add a parallel-lane fast path above some size, and I want that on the record as a choice, not an oversight: such a path would score better and would no longer be a test of the native's claim.
2. **Fixed ~35-cycle finaliser dominating tiny inputs.** Guarded by size where a guard is legitimate (`len >= 32` selects the unrolled pour; short inputs take a minimal path with no setup). I refused the other available guard — dropping organs for small `len` — because that is exactly the risky part the native forbids ("if the two don't trade beauty for beauty exactly, I bend again"), and it would trade the metric I expect to win (avalanche) for the one I expect to lose (throughput on short keys).
3. **Weakness the native's design genuinely has, which the benchmark will not catch.** The absorb phase is close to affine: the state is a rotate-add accumulation of byte-stained words. Carries give it some nonlinearity, but it is not collision-resistant against an adversary choosing multi-byte differences, and low-order bits of the accumulator are weakly coupled across distant positions. The avalanche metric (single-bit flip, strong finaliser downstream) cannot see this. A good score here means "mixes every input bit into the output," not "cryptographic." I would not ship this where collisions are attacker-chosen.

Bottom line: the native's allocation of effort — minimum rounds per byte, seven rounds once at the end — is, as far as I can reason without the bench, the *correct* engineering answer for this contract, and it contradicts the assumption it was chosen to break. But I have not measured it, so the claim stands unverified, and the prediction above is the thing to score me on.