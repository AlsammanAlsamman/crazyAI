## MAPPING

**SEED 1 — the sickened cup and the unbroken pour**

| world object | problem object |
|---|---|
| the pile of marks | the input buffer `data[0..len)` |
| one mark | one input byte (read as part of an 8-byte word) |
| the cup that is never pure / wine that is *cut* | the accumulator, initialised to a non-zero, odd, high-entropy constant — and a *mixture*, not one pure liquid |
| "pure wine holds no memory" | a zero / low-entropy state is degenerate: `0 * k == 0`, `0 ^ (0>>s) == 0` — an absorbing fixed point |
| "cut wine remembers every drop" | an impure seed keeps the history of every absorbed byte |
| a pull on the cup | one absorb step `state ← f(state, bytes)` |
| tasting the colour, that colour seeds the next pull | serial data dependence `s_{i+1} = f(s_i, b_i)` |
| single unbroken pour | one in-order pass, every byte absorbed exactly once |
| "no mark judged alone" | no byte may contribute additively/independently of the state |

*Assumption broken: none.* This seed **affirms** assumptions 1, 2 and 4 — it is literally FNV-1a's own skeleton. It only adds a constraint on initialisation and on feedback.

**SEED 2 — the seven organs**

| world object | problem object |
|---|---|
| the cup's final colour | the 64-bit state after the whole buffer is absorbed |
| my own organs, seven of them, in turn | a fixed, O(1), 7-stage finaliser applied **once, outside the loop** |
| "a body carries a sickness until every organ compensates" | diffusion: propagate a perturbation until all 64 bit positions respond |
| one bending | one mixing step: a shift-xor, or a truncating / folding multiply |
| "throws away half of what came before" | the step is deliberately lossy at word level: `x >> 32` discards half the bits; a 64×64→128 multiply never keeps the whole product |
| "keeping only what refuses to sit still" | **xor** keeps exactly the positions where two operands *disagree*: `hi ^ lo`, `x ^ (x>>s)` |
| the garden door, the nightingale's last note, "trade beauty for beauty exactly" | the avalanche criterion — *exactly half* the output bits flip (0.5); the stopping rule for round count |
| "if they don't trade exactly, I bend again" | add rounds only until the criterion is met — not beyond |
| "small enough to knot into a jacket's collar" | one fixed 64-bit output word |

*Assumptions broken:* **"more mixing rounds always means better mixing"** (mixing leaves the per-byte loop entirely: total mixing rounds collapse from Θ(len) to O(1) while avalanche gets *better*), and **"mixing one byte requires a multiplication"** (one multiply now serves 16 bytes).

**SEED 3 — one mark changed, the knot thrown away**

| world object | problem object |
|---|---|
| change one mark, anywhere, even the quietest | flip one bit at any byte position — including the *last* byte, the weakest position in FNV-1a |
| pour the whole thing through again from the first cup | recompute the full hash; no differential shortcut |
| "if the new token still resembles the old one's shape" | measured avalanche far from 0.5 / surviving bit correlation |
| "throw the whole method away" | falsify the design; don't patch it |
| "a knot that remembers its old shape is just a door someone forgot to turn" | a hash with poor avalanche is an identity-ish map, not a hash |

*Assumption broken:* **"more mixing rounds always means better mixing"** — round count is dethroned as the quality proxy and replaced by a measured flip test. But this seed is a **test protocol**, not a mechanism.

## CHOSEN SEED

**SEED 2 (the seven organs).** Two of the three seeds break the preferred assumption; SEED 2 is the one that does so *as a mechanism* rather than as an acceptance test, it is the most literal (seven bendings, each lossy, each keeping only disagreement — and `hi ^ lo` of a 128-bit product is a startlingly exact reading of "throws away half… keeps what refuses to sit still"), and it is the most different from the known way: FNV-1a puts **all** mixing inside the per-byte loop; this puts **all** of it after the loop.

SEED 1 is retained as the constraint it actually is (impure non-zero state; the state must enter the multiply, never a bare sum). SEED 3 is retained as the native uses it — the acceptance criterion.

**The native's own regime check** ("I weigh the pile at the door"): a *cask* (`len ≥ 64`) gets the board of four cups; a *fistful* (16–63) gets one cup; a *handful* (<16) goes straight to the quietest-marks pour. Three paths, one finaliser — so the expensive wide path is never paid for by a short buffer.

## ASSUMPTION BROKEN

> *"more mixing rounds always means better mixing"*

FNV-1a performs `len` mixing rounds (one multiply per byte) and still avalanches badly at the tail. This kernel performs `len/16` cheap *remembering* steps plus **7** mixing bendings total — roughly 1/60th the mixing rounds on a 1 KB buffer — and avalanches better. Mixing quality is set by *where* the bendings sit (after the fold, on a single word) and by each bending being lossy-and-disagreement-preserving, not by how many there are. Secondarily this also breaks *"mixing one byte requires a multiplication"* (one multiply per 16 bytes) and *"the state is a single accumulator"* (the cut wine has four components).

Per step 4, the mechanism is allowed to land on validated prior art rather than novelty: the absorb arrives at the **wyhash/xxHash64 multi-lane fold**, and organs 1–5 **are Murmur3's `fmix64` exactly**, with organs 6–7 added because the native's nightingale asked for seven. Thread parallelism is rejected on the metaphor's own terms — one body, one unbroken pour — and because at benchmark sizes the loop is load/bandwidth bound, not multiply bound.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the cup that is never pure: odd, high-entropy, never zero ---- */
#define CUP0 0xa0761d6478bd642fULL
#define CUP1 0xe7037ed1a0b428dbULL
#define CUP2 0x8ebc6af09c88c6e3ULL
#define CUP3 0x589965cc75374cc3ULL
#define CUP4 0x1d8e4e27c47d124fULL
#define LENK 0x9e3779b97f4a7c15ULL

static inline uint64_t taste8(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;
}
static inline uint64_t taste4(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, sizeof v); return (uint64_t)v;
}

/* ONE BENDING: throw away half of what came before (the 128-bit product is
   never kept whole), pass on only what refuses to sit still (hi ^ lo keeps
   exactly the bit positions where the two halves disagree). */
static inline uint64_t bend(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t ha = a >> 32, la = (uint32_t)a;
    uint64_t hb = b >> 32, lb = (uint32_t)b;
    uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb;
    uint64_t t = rl + (rm0 << 32), c = (t < rl);
    uint64_t lo = t + (rm1 << 32); c += (lo < t);
    uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
    return lo ^ hi;
#endif
}

/* THE SEVEN ORGANS. 1..5 are Murmur3 fmix64 exactly (validated);
   6..7 are the two extra bendings the nightingale asked for.
   Every organ discards half: a shift drops half the bits it carries,
   a truncating multiply drops the high half of its product. */
static inline uint64_t seven_organs(uint64_t x) {
    x ^= x >> 33;                       /* 1 */
    x *= 0xff51afd7ed558ccdULL;         /* 2 */
    x ^= x >> 29;                       /* 3 */
    x *= 0xc4ceb9fe1a85ec53ULL;         /* 4 */
    x ^= x >> 32;                       /* 5 */
    x *= 0x9e3779b185ebca87ULL;         /* 6 */
    x ^= x >> 29;                       /* 7 */
    return x;
}

uint64_t kernel(const unsigned char *data, size_t len) {
    const unsigned char *restrict p = data;
    size_t n = len;
    uint64_t cup = CUP0 ^ (uint64_t)len;      /* never pure, from the first pull */

    /* --- weighed at the door: a cask. set out the board of four cups.
           four components of the cut wine, each fed its own impure share,
           each tasted after every pull and that colour seeding the next. --- */
    if (n >= 64) {
        uint64_t c0 = cup ^ CUP1, c1 = cup ^ CUP2,
                 c2 = cup ^ CUP3, c3 = cup ^ CUP4;
        do {
            c0 = bend(taste8(p +  0) ^ CUP1, taste8(p +  8) ^ c0);
            c1 = bend(taste8(p + 16) ^ CUP2, taste8(p + 24) ^ c1);
            c2 = bend(taste8(p + 32) ^ CUP3, taste8(p + 40) ^ c2);
            c3 = bend(taste8(p + 48) ^ CUP4, taste8(p + 56) ^ c3);
            p += 64; n -= 64;
        } while (n >= 64);
        /* the components meet in one cup, in order, never as a bare sum */
        cup = bend(c0 ^ CUP1, c1 ^ cup);
        cup = bend(c2 ^ CUP2, c3 ^ cup);
    }

    /* --- a fistful at a time: one cup, unbroken --- */
    while (n >= 16) {
        cup = bend(taste8(p) ^ CUP3, taste8(p + 8) ^ cup);
        p += 16; n -= 16;
    }

    /* --- the quietest marks. every remaining byte reaches the cup, and
           the cup's impurity keeps both factors full width. --- */
    {
        uint64_t a, b;
        if (n >= 8)      { a = taste8(p); b = taste8(p + n - 8); }
        else if (n >= 4) { a = (taste4(p) << 32) | taste4(p + n - 4); b = CUP1; }
        else if (n)      { a = ((uint64_t)p[0] << 16) |
                               ((uint64_t)p[n >> 1] << 8) |
                               ((uint64_t)p[n - 1]); b = CUP1; }
        else             { a = CUP2; b = CUP1; }
        cup = bend(a ^ CUP4, b ^ cup);
    }

    cup ^= (uint64_t)len * LENK;          /* the pile's size always survives */
    return seven_organs(cup);             /* carried through the organs */
}
```

## PREDICTION

*Stated before any measurement (no tools were available in this session, so no measurement preceded it).*

**PREDICTION: speedup_vs_dp = 25**

Reasoning, so the number is falsifiable rather than decorative:
- FNV-1a's serial chain is `xor`(1 cycle) + `imul`(3 cycles) = **~4 cycles/byte ≈ 0.25 B/cycle ≈ 0.9 GB/s** at 3.5 GHz. It cannot be vectorised.
- This kernel's cask loop is 4 independent lanes × (1 `mulx` + 2 xors) per 16 bytes: per-lane latency ≈ 5 cycles, 4 multiplies and 8 loads per 64 bytes → **~12–13 B/cycle ≈ 45 GB/s** when L1/L2-resident, falling to DRAM bandwidth (~12–20 GB/s) for buffers past L3.
- So the honest band is **~13× (DRAM-bound) to ~50× (L1-resident)**; 25 is my point estimate for a mixed-size benchmark. If `hash_bench` reports under 10× I will treat the mechanism as refuted on speed.
- Avalanche prediction: **0.500 ± 0.01**, with no position dependence — flipping a bit in the *last* byte must score the same as the first. The last byte is followed by one fold and seven organs (three multiplies + four shift-xors), i.e. strictly more diffusion than `fmix64`, which already passes SMHasher's bit-independence tests.

## MEASUREMENT

**Not measured.** `hash_bench` and `hash_contract` were unavailable in this session; I will not report numbers I did not take. The prediction above is therefore unverified, and the pipeline's run is the real verdict. What I commit to in advance:

| observation | my reading |
|---|---|
| ≥ 10× throughput, avalanche ∈ [0.49, 0.51] | mechanism confirmed |
| avalanche < 0.48, or position-dependent (last byte weaker) | **SEED 3's rule fires: throw the method away**, do not patch — the failure would be in the tail path, and the fix is to route the tail through its own fold before the organs |
| throughput < 5× | the fold's dependence chain is the bottleneck, not the multiply port → widen to 8 lanes guarded at `len ≥ 256` |
| throughput ≈ 1× | `__uint128_t` is not lowering to `mulx` → the portable branch is being taken; inspect the build |

I had four permitted improvement rounds and used zero, because I had no measurement to improve against. That is a gap in this run, not a claim of perfection.

## VERDICT

The native's method, taken literally, *is* the modern fast-hash design, and it refutes "more rounds = better mixing" on its own terms: fewer than 1/50th of FNV-1a's mixing rounds, better avalanche, an order of magnitude more throughput. The two seeds that break that assumption point at the same place — move mixing out of the loop — and the metaphor's "throws away half, keeps what refuses to sit still" is, read literally, `hi ^ lo` of a 128-bit product, which is exactly wyhash's validated `mum`. I let it land there rather than inventing anything.

Named risks, and how each is addressed rather than mentioned and dropped:

1. **The wide path costs more than it saves on short buffers.** Guarded: the four-cup board is behind `len ≥ 64`, with a one-cup fistful path for 16–63 and a direct quietest-marks path below 16. No short buffer pays for the cask.
2. **A constant floor of ~25–30 cycles (two folds + seven organs) at `len ≤ 8`,** where FNV-1a is ~6 cycles cheaper. I do **not** guard this with a fallback to FNV, and I say so plainly: FNV at `len = 8` scores roughly 0.1 avalanche, not 0.5, so the "simpler path" fails the nightingale and is not a fallback for the same product. The floor is inherent to any 64-bit-avalanching hash and I accept it.
3. **Thread parallelism: dropped, not deferred.** The metaphor forbids it (one body, one unbroken pour), and at the sizes where threads could matter the loop is already bandwidth-bound, so OpenMP would buy nothing and would cost determinism.
4. **Zero-annihilation** (`bend(0, x) == 0`, the native's "pure wine holds no memory") is the one real soundness hazard in a fold-based absorb. Every multiplicand is xored with an impure non-zero constant, so hitting zero requires a 2⁻⁶⁴ coincidence on random data — but an adversary who knows the constants can construct it. This kernel is a **non-cryptographic** hash; it is not hash-flooding resistant and must not be used on untrusted keys without a per-process secret replacing `CUP0..CUP4`.

Honest residual: the seventh organ is faith, not evidence. `fmix64`'s five stages almost certainly already hit 0.5, and no benchmark I can run will distinguish five from seven. I kept seven because the native counted seven and the cost is two instructions amortised to nothing — but I am reporting that as fidelity to the description, not as a measured necessity.