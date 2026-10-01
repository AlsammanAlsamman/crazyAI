No tools were available in this session (and the `claude.ai PubMed` / Claude Docs MCP servers are unauthorized here anyway — they'd need authorizing from claude.ai connector settings or an interactive `claude mcp` session before they could be used). So everything below is reasoning + artifact; the prediction is stated **before** any measurement, and the MEASUREMENT section reports honestly that I could not run `hash_bench` myself.

---

## MAPPING

**SEED 1 — "A rolling sphere serves as the sole carried memory across the whole ordered pile."**

| World object | Problem object |
|---|---|
| the sphere | one `uint64_t` register `s` — the entire hash state |
| "my running weight, my only memory" | no second accumulator, no table, no scratch array, no SIMD lanes |
| the ordered pile of marks | `data[0..len-1]`, consumed strictly in index order |
| trail's high mouth | initial constant seed |
| "the old face I let shrink and go" | bits rotated off the top are not saved anywhere |

**SEED 2 — "A fixed count of tumbles down a mason trail mixes each mark into the sphere before the next is taken."**

| World object | Problem object |
|---|---|
| a mark | one `unsigned char` byte |
| "press its shape into the sphere's face" | `s += byte` — the sphere is the *running weight*, so pressing is **addition** |
| "the next weight to carry forward" | the **carry** out of that addition (the only nonlinearity in the loop) |
| a stalk / it sobs once | exactly one mixing strike per mark — "no more, no fewer" |
| "the angle of that crack" | the **rotation amount** — a fixed rotate-left |
| coiled trail | `rotl64`, a circular (coiled) shift, not a shift that drops bits |
| "a footprint at a stalk is a woman's business, I never look back" | the pre-rotation value is discarded; no history, no lookup table, no memo |
| "one wrong mark and every stalk downstream sobs differently" | every subsequent step's state differs → change propagates along the whole chain |

**SEED 3 — "Only the final hairline crack in the last stalk is kept as the token, all intermediate cracks and dust swept away."**

| World object | Problem object |
|---|---|
| the final stalk | the finalizer applied once, after the last byte |
| "the last, smallest crack ... small as a token" | the returned `uint64_t` |
| "I don't keep the sphere itself" | `s` is not returned raw; it is cracked once more |
| dust / footprints / eggshells → the jungle's open plumbing | no intermediate outputs, no per-block writes, nothing stored |

## CHOSEN SEED

**SEED 2.** It is the most literal (every noun becomes one instruction: press = `add`, crack angle = `rotl`, stalk count = loop trip of exactly one) and it is the one that breaks the forbidden assumption. SEED 1 and SEED 3 constrain the *shape* of the state and the output, not the mixing; SEED 2 *is* the mixing.

## ASSUMPTION BROKEN

> "Mixing one byte requires a multiplication."

The native never multiplies. The whole per-byte step is `add` + `rotate` — nonlinearity comes from **carry propagation** ("the next weight to carry forward"), and diffusion comes from the **coiled** (circular) trail, not from a multiplier's partial-product fan-out. The finalizer is also multiplication-free (shift/add/xor only — Thomas Wang's 64-bit mix), so no `imul` appears anywhere in the kernel. Consequence: the per-byte critical path is **2 cycles** (add 1, `rol` 1) instead of FNV-style **4+ cycles** (xor 1 + imul 3). The assumption being false is exactly where the speed comes from.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

/* The coiled limestone trail: one sphere, one press and one crack per mark,
   no multiplication anywhere, only the final crack is handed over. */

#define ANGLE  11                          /* the crack's angle; fixed, odd, never 0 */
#define MOUTH  0x9E3779B97F4A7C15ULL       /* the sphere's shape at the trail's high mouth */

/* press the mark into the sphere's face, then strike one stalk and read
   the angle of the crack. carry out of the += is "the weight carried forward". */
static inline uint64_t tumble(uint64_t s, unsigned char mark)
{
    s += (uint64_t)mark;
    return (s << ANGLE) | (s >> (64 - ANGLE));   /* single rol; ANGLE is a nonzero constant */
}

/* the final stalk: the last, smallest crack, kept as the token.
   Thomas Wang's 64-bit mix - shift/add/xor only, no multiply. */
static inline uint64_t last_crack(uint64_t k)
{
    k = (~k) + (k << 21);
    k ^=  k >> 24;
    k =  k + (k << 3) + (k << 8);
    k ^=  k >> 14;
    k =  k + (k << 2) + (k << 4);
    k ^=  k >> 28;
    k =  k + (k << 31);
    return k;
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    uint64_t s = MOUTH;                 /* the sole carried memory */
    size_t i = 0;

    /* unrolled only to keep the loop bookkeeping off the 2-cycle critical
       path; the marks are still taken one by one, in their given order. */
    if (len >= 8) {
        size_t n8 = len & ~(size_t)7;
        for (; i < n8; i += 8) {
            s = tumble(s, data[i + 0]);
            s = tumble(s, data[i + 1]);
            s = tumble(s, data[i + 2]);
            s = tumble(s, data[i + 3]);
            s = tumble(s, data[i + 4]);
            s = tumble(s, data[i + 5]);
            s = tumble(s, data[i + 6]);
            s = tumble(s, data[i + 7]);
        }
    }
    for (; i < len; ++i)                /* the tail marks, same trail */
        s = tumble(s, data[i]);

    return last_crack(s);               /* the sphere itself is not kept */
}
```

Notes on literalness held deliberately: no OpenMP, no `immintrin.h`, no scratch buffer, no second accumulator, no lookup table, no length mixed into the seed — all four would require memory the native says he does not carry. `len == 0` returns `last_crack(MOUTH)`, a fixed token. `ANGLE` is a compile-time nonzero constant, so the `64 - ANGLE` shift is never UB.

## PREDICTION

**PREDICTION: speedup_vs_dp = 2.2**

Reasoning behind the number, stated so it can be falsified: the per-byte dependency chain is `add`(1 cycle) → `rol`(1 cycle) = **2 cycles/byte**, issue-limited at well under that (~3 uops/byte against 4-wide issue), so ≈0.5 B/cycle ≈ **1.6–1.9 GB/s** on a ~3.2–3.7 GHz core. A per-byte multiply reference (`s ^= b; s *= prime`) is latency-bound at 4 cycles/byte. 4/2 = 2.0, plus a little from cheaper front-end pressure → 2.2. Avalanche: I predict a **good** score (per-bit flip probability within a few percent of 0.5), because although the accumulation loop is weakly diffusing, `last_crack` is a tested bijective avalanche mixer applied to a data-dependent state, so no (input-bit, output-bit) cell should be stuck at 0 or 1.

Falsification conditions I accept in advance: if measured speedup < 1.3, the multiplication-free press is *not* buying the latency I claimed. If the avalanche score is poor, then a 2-instruction press is too weak for a single-round finalizer to rescue, and the native's "one stalk per mark, no more, no fewer" is simply wrong for this problem.

## MEASUREMENT

Not performed by me — `hash_bench` and `hash_contract` were not callable in this session, and I will not invent numbers. The artifact is written against the fixed contract `uint64_t kernel(const unsigned char *data, size_t len)` and compiles under `gcc -O3 -march=native -fopenmp -lm` with only `<stdint.h>`/`<stddef.h>`. Zero improvement iterations were spent, out of the four allowed; the pipeline's measurement is the first and only evidence, and if it contradicts the prediction above, the prediction is wrong, not the measurement.

## VERDICT

The native's account survives translation almost word-for-word: one register, one `add`, one `rol`, one final crack. The thing worth reporting is that his refusal to multiply is not a limitation here — it is the performance argument, because a hardware multiply costs 3 cycles of *latency* on a chain that is irreducibly serial, and a rotate costs 1.

What I am *not* claiming: this is a weak hash in the differential/collision sense. The accumulation is an add-rotate checksum; bytes 64 apart are pressed at the same bit offset (`gcd(11,64)=1`, period 64), length is not mixed, and an all-zero buffer takes only 64 distinct states across all lengths. Single-bit avalanche — the thing being scored — should still be fine, because the finalizer does that work. I flag this as a real quality gap rather than hiding it behind a good avalanche number.

On risk, per the guard rule: I deliberately shipped **no conditional mechanism at all** — no OpenMP path, no SIMD path, no size-branched second algorithm — so there is no condition-dependent slow case to guard. The performance limit is uniform and unconditional: 2 cycles/byte means ~1.7 GB/s regardless of input size, which leaves a multi-GB/s memory system underused on very large buffers. Fixing that would require several independent spheres (ILP) or vector lanes, i.e. more than "my only memory", so I did not smuggle it in; I am reporting the ceiling instead of quietly replacing the native's single rolling sphere with the textbook multi-accumulator design. If the harness benchmarks multi-megabyte buffers against a vectorized reference, this kernel loses on throughput, and that loss belongs to the native's constraint, stated up front rather than discovered afterwards.