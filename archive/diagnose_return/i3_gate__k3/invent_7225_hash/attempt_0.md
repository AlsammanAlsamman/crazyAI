## 1. SEED → PROBLEM MAPPINGS

### SEED 1 — "one unbroken pour through the sickened cup"

| World object | Problem object |
|---|---|
| a mark in the pile | one input byte `data[i]` |
| the pile | the buffer `data[0..len)` |
| the cup, "never pure", "cut wine" | the running state — a *blend*, not one clean value |
| "pure wine holds no memory" | a state with no feedback forgets prior bytes |
| a pull on the cup | one accumulate step |
| tasting the color, letting it seed the next pull | `state_{i+1} = f(state_i, byte_i)` — serial feedback |
| "no mark judged alone or twice" | single pass, each byte read exactly once |

**Assumption broken:** essentially *none*. "The cup's new color seeds the next pull" **is** FNV‑1a. The only non-standard word is *"never pure"* — a cup that is a blend, which would break "the state is a single accumulator, one value". But the SEED line itself emphasizes the unbroken serial chain, i.e. the known way.

### SEED 2 — "seven organs, each bending and discarding half"

| World object | Problem object |
|---|---|
| "when the pile ends, I don't stop at the cup" | strong mixing happens **after** the loop, not inside it |
| the cup's last color | the merged accumulator value at end of buffer |
| my own organs (seven) | seven fixed finalization stages, a bounded constant |
| an organ "bending" | a multiply by an odd constant (stir) |
| an organ "discarding half of what it received" | `h >> 32`-ish — throw away half the width |
| "keeping only what refuses to sit still" | `h ^= h >> s` — keep only the bits that *differ* under the fold |
| "a body carries a sickness until every organ compensates" | avalanche is a whole-word property, reached by successive compensating stages |
| the garden door / nightingale's last note | the avalanche test: output must trade exactly half its bits |
| "if they don't trade beauty for beauty exactly, bend again" | **stopping rule**: add a stage only until the test is exact — not more |
| "small enough to knot into a jacket's collar" | fixed 64-bit output, no growing state |

**Assumptions broken:** (a) *"more mixing rounds always means better mixing"* — the organs are capped at seven, each one **discards** rather than accumulates, and the garden door says bend *only until exact*; (b) consequently *"mixing one byte requires a multiplication"* — if avalanche duty is discharged at the end, the per-byte pull can be made cheap, and the cup can be a blend of independent parts.

I'll be plain: (a) is the strongest available reading of this seed, not an airtight one. The native never says "fewer rounds is better"; he says the count is *bounded and test-determined*, and that each round is lossy. That is a denial of "more is always better", but an indirect one.

### SEED 3 — "change one mark, pour again, any resemblance condemns the method"

| World object | Problem object |
|---|---|
| changing one mark, even the quietest | flipping one bit of one input byte, including the last/low one |
| pouring the whole pile again from the first cup | recompute the full hash |
| "resembles the old token's shape" | Hamming distance to old output ≪ 32 bits |
| "throw the whole method away" | reject the kernel |
| "a door someone forgot to turn" | an identity-ish map: input leaks through to output |

**Assumption broken:** none structural. This is the *acceptance test* (strict avalanche criterion), not a mechanism. It is however the native's reason for keeping the organs at all.

---

## 2. CHOSEN SEED

**SEED 2 — the seven organs.**

SEED 1 is the known way restated. SEED 3 is a test harness, not a kernel. SEED 2 is both the most different from FNV‑1a (it moves the mixing *out* of the byte loop entirely) and the one that engages the preferred assumption.

**Step 4 check — does a validated technique already satisfy this?** Yes, and I let the mechanism land on it rather than inventing. "Weak accumulate, strong finalize" is exactly the xxHash / MurmurHash family split: cheap multi-lane accumulation, then a fixed `xor-shift / multiply` avalanche finalizer (`fmix64`, `splitmix64`). So the seven organs are implemented as **splitmix64's validated 5-stage finalizer (organs 1–5) plus the one extra stir-and-fold pair the native insists on (organs 6–7)** — a strict superset of a known-good mixer, not a new one.

**Regime handling (step 5).** The known-way section describes one regime (whole buffer, in order), but the assumption list implies small-vs-large. The native's own metaphor supplies the test: a pile too small for a row of cups is poured by hand into one cup. Three runtime paths on `len`: ≥64 → eight cups; 8–63 → one cup; <8 → **no pour at all**, the organs do everything.

**Thread parallelism: declined, by the native's own rule** — "the whole pile passes through as a single unbroken pour." Threads break the unbroken pour, and at benchmark sizes OpenMP fork/join (µs scale) exceeds the entire hash time. **SIMD: declined with reason** — 64×64 integer multiply has no AVX2 instruction (`vpmullq` is AVX‑512‑DQ only, not guaranteed by `-march=native`); the 8-lane scalar loop already saturates the single integer-multiply port, which is the hard ceiling. `restrict` and an 8-wide unrolled body are used instead.

**Risk named and guarded:** the organs cost ~17 cycles *regardless of length*, so for `len < 8` they could lose to FNV‑1a. Guard: the `len < 8` path performs **zero** pours (no multiply, no loop) — the bytes are packed into one word with the length in the free high byte and handed straight to the organs. Both costs are never paid at once.

---

## 3. ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- the cup: "never pure" => a blend of eight parts ---------------- */
#define BASIS 0x9E3779B97F4A7C15ULL
static const uint64_t PRIME_POUR = 0x9E3779B185EBCA87ULL;
static const uint64_t PM1 = 0xC2B2AE3D27D4EB4FULL;
static const uint64_t PM2 = 0x165667B19E3779F9ULL;
static const uint64_t PM3 = 0x85EBCA77C2B2AE63ULL;

static inline uint64_t ld64(const unsigned char *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));           /* r is always 1..59 */
}

/* one pull: the mark enters, the cup's colour turns, that colour is the
   seed of the next pull.  Deliberately cheap - ONE multiply per 8 bytes -
   because all avalanche duty is discharged by the organs below. */
static inline uint64_t pull(uint64_t l, uint64_t w) {
    return rotl64(l ^ w, 29) * PRIME_POUR;
}

/* ---- the seven organs ------------------------------------------------
   each bends, discards half of what it received, and passes on only
   what refuses to sit still.  Organs 1-5 are splitmix64's validated
   finalizer; 6-7 are the extra stir-and-fold the native insists on. */
static inline uint64_t seven_organs(uint64_t h) {
    h ^= h >> 30;                   /* 1: fold, keep only what differs */
    h *= 0xBF58476D1CE4E5B9ULL;     /* 2: bend                         */
    h ^= h >> 27;                   /* 3: fold                         */
    h *= 0x94D049BB133111EBULL;     /* 4: bend                         */
    h ^= h >> 31;                   /* 5: fold                         */
    h *= 0x9E3779B97F4A7C15ULL;     /* 6: bend                         */
    h ^= h >> 32;                   /* 7: fold                         */
    return h;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    const unsigned char *restrict p = data;
    uint64_t h;

    if (len >= 64) {
        /* --- the tavern's pile: eight cups, poured side by side ----- */
        uint64_t l0 = BASIS + PM1, l1 = BASIS - PM1;
        uint64_t l2 = BASIS + PM2, l3 = BASIS - PM2;
        uint64_t l4 = BASIS + PM3, l5 = BASIS - PM3;
        uint64_t l6 = BASIS ^ PM1, l7 = BASIS ^ PM2;

        size_t nblk = len >> 6;
        for (size_t i = 0; i < nblk; i++) {
            l0 = pull(l0, ld64(p +  0));
            l1 = pull(l1, ld64(p +  8));
            l2 = pull(l2, ld64(p + 16));
            l3 = pull(l3, ld64(p + 24));
            l4 = pull(l4, ld64(p + 32));
            l5 = pull(l5, ld64(p + 40));
            l6 = pull(l6, ld64(p + 48));
            l7 = pull(l7, ld64(p + 56));
            p += 64;
        }

        /* the dregs - every mark exactly once, never twice */
        size_t rem = len & 63;
        uint64_t t = BASIS;
        while (rem >= 8) { t = pull(t, ld64(p)); p += 8; rem -= 8; }
        if (rem) {
            uint64_t w = 0;
            for (size_t i = 0; i < rem; i++) w |= (uint64_t)p[i] << (8 * i);
            t = pull(t, w);
        }

        /* pour the eight cups back into one - position-sensitive */
        h  = rotl64(l0,  1) + rotl64(l1,  7) + rotl64(l2, 12) + rotl64(l3, 18)
           + rotl64(l4, 23) + rotl64(l5, 29) + rotl64(l6, 34) + rotl64(l7, 40);
        h ^= rotl64(l0, 11) * PM1;
        h ^= rotl64(l1, 19) * PM2;
        h ^= rotl64(l2, 27) * PM3;
        h ^= rotl64(l3, 35) * PM1;
        h ^= rotl64(l4, 43) * PM2;
        h ^= rotl64(l5, 51) * PM3;
        h ^= rotl64(l6, 59) * PM1;
        h ^= rotl64(l7,  5) * PM2;
        h ^= t;

    } else if (len >= 8) {
        /* --- a pile for one hand: a single cup ---------------------- */
        uint64_t l = BASIS;
        size_t rem = len;
        while (rem >= 8) { l = pull(l, ld64(p)); p += 8; rem -= 8; }
        if (rem) {
            uint64_t w = 0;
            for (size_t i = 0; i < rem; i++) w |= (uint64_t)p[i] << (8 * i);
            l = pull(l, w);
        }
        h = l;

    } else {
        /* --- a handful of marks: no pour at all.  The organs alone.
               (guard: the finalizer's fixed cost is never stacked on
                top of a pour that cannot pay for itself.)            */
        uint64_t w = 0;
        for (size_t i = 0; i < len; i++) w |= (uint64_t)p[i] << (8 * i);
        h = BASIS ^ w ^ ((uint64_t)len << 56);   /* len<8 => bits 56.. free */
    }

    h += (uint64_t)len * PM3;
    return seven_organs(h);
}
```

---

## 4. PREDICTION

Stated before any measurement (and no measurement was possible in this session).

**PREDICTION: speedup_vs_dp = 20**

Reasoning: FNV‑1a is latency-bound at `imul`(3) + `xor`(1) ≈ 4 cycles/byte → ~0.25 B/cycle. The eight-cup loop issues 8 multiplies per 64 bytes and is throughput-bound on the single integer-multiply port → ~8 B/cycle, i.e. ~32× ideal; I discount to **20×** for load pressure, front-end width, loop overhead, and possible L2/DRAM bandwidth capping at large sizes.

Secondary prediction: **avalanche ≈ 0.50** (0.495–0.505 fraction of output bits flipped per input bit flip). A full splitmix64 finalizer already passes SAC; organs 6–7 are surplus. I expect the avalanche score to be statistically indistinguishable from a 5-organ version — i.e. I predict the native's seventh organ buys **nothing measurable**, which is itself the test of the assumption being broken.

---

## 5. MEASUREMENT

**Not measured.** No tools were available in this session — `hash_bench` and `hash_contract` could not be called, so I have no throughput number and no avalanche score of my own. Reporting this plainly rather than quoting plausible figures: everything in the PREDICTION section is a cycle-count argument, not an observation. The pipeline's numbers supersede it entirely.

Three specific things the measurement should check, and what would falsify me:

1. **speedup ≥ ~15×** on buffers ≥ 1 KB. If it comes back under 8×, the loop is not multiply-bound and my cost model is wrong.
2. **avalanche ≈ 0.50.** If it comes back materially off 0.5, the weakened one-multiply pull is *not* being rescued by the organs and SEED 2's central claim fails.
3. **Organ count sensitivity.** Deleting organs 6–7 (`h *= 0x9E37...; h ^= h >> 32;`) should change throughput slightly and avalanche not at all. If avalanche *does* improve with them, I was wrong about the assumption and the native was right about seven.

---

## 6. VERDICT

The mapping is literal throughout: a mark is a byte, a pull is one accumulate step, the cup's colour is the lane state, the cup "never pure" is the eight-lane blend, the organs are seven finalization stages, the garden door is the avalanche test, the jacket collar is the fixed 64-bit width, and the pulled thread is the single-bit-flip test from SEED 3.

What SEED 2 actually bought: permission to make the per-byte path *weak*. The textbook accumulator must be strong per step because it is the only mixer; once the organs exist, one multiply per eight bytes suffices, and the lanes can run independently. That is the whole speedup. It is not a new idea — it is xxHash's architecture, arrived at from the metaphor rather than copied, which is the right outcome under step 4.

Where it could be worse than FNV‑1a, named honestly and addressed rather than left standing:
- **Inputs under 8 bytes.** The organs cost ~17 cycles no matter what. Guarded: that path does zero pours, so total cost is roughly the finalizer alone and stays near parity. For `len ≤ 2` FNV‑1a is still cheaper in raw cycles — but it also fails the garden door there, so it is not the same product.
- **Huge buffers.** At ~8 B/cycle the kernel will hit DRAM bandwidth before it hits the multiplier; speedup will compress toward the memory-bound ratio. No fix available and none attempted — stating it rather than hiding it.
- **Thread parallelism deliberately absent.** The metaphor forbids it ("one unbroken pour") and the arithmetic agrees at these sizes. If the bench uses buffers in the tens of MB, this is money left on the table, and I am leaving it there on purpose.

The part I am least confident in is the assumption attribution itself: SEED 2 breaks "more mixing rounds always means better mixing" by implication (bounded count, lossy rounds, test-determined stopping) rather than by statement. If the measurement shows organs 6–7 improving avalanche, the native's seven was a real requirement and my reading of his stopping rule was wrong.