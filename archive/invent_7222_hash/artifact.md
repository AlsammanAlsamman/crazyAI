# MAPPING

| Seed | World object | Problem object | Assumption it breaks |
|---|---|---|---|
| **A** — "die-stone never resets between marks, so every fold carries the callus of all folds before it" | die-stone | the running hash accumulator `h` | pile of marks | the byte stream | never resetting between marks | never re‑initializing the accumulator between bytes | **None.** This restates "state persists, updated in place, one running value" — it's exactly assumptions 1 and 2 of the known way, not a break of them. |
| **B** — "each mark pressed into the stone's already‑turned position, not a clean face; the stone's memory of the last press bends how deep this one goes" | die-stone's current turned position | current bit-pattern of `h` before this byte | quarter turn through the groove | a fixed bit‑shift applied to `h` | pressing the mark's weight in | combining the byte into `h` via add/xor, not multiply | **Breaks assumption 3** — "mixing one byte requires a multiplication." The whole description (turn, press, let memory bend depth) is built entirely out of *rotation/shift* and *contact/combination*, never out of a scaling/multiplicative operation. |
| **C** — "only the stone's final seated number ever leaves the desk; everything else is swept away" | final seated position | return value `h` at the end | groove-dust, intermediate turns | intermediate loop values | sweeping away | not retaining/exporting intermediate state | **None.** This is just "return one value at the end, discard scratch" — already true of the known way (FNV-1a keeps only `h`, prints only `h`). |

None of the three seeds breaks "the state is a single accumulator updated in place, one value" — all three take the *single* die-stone completely for granted (there is never more than one stone, and it's explicitly never lifted off the desk / never reset). Stating that plainly, per the instructions I fall back to the most literal seed among the remaining candidates that *does* break something: **Seed B**, which is both the most literal (it directly describes the byte-mixing step, the most central mechanical operation) and the most different from the known way (it explicitly removes multiplication from the mixing step).

# CHOSEN SEED

> "Each mark's weight is pressed into the stone's already-turned position rather than onto a clean face, so early marks bend how later marks land."

# ASSUMPTION BROKEN

**"Mixing one byte requires a multiplication."** The native never multiplies anything — the die-stone is *turned* (shifted) and the mark is *pressed* (combined by addition/XOR) into wherever the turning left it. No scaling operation appears anywhere in the description.

# OBJECT MAPPING (literal)

- **byte** = one "mark" pressed against the stone, in pile order.
- **state** = the single die-stone's seated position → one `uint64_t h`, updated in place (consistent with the "single stone" fact established above — no seed licenses a second stone).
- **"quarter turn through the chalk-bank groove"** = a fixed left-shift-and-fold of the current state into itself (`h += h << k`) — a partial rotation of the accumulator's own bits, done *before* the multiply-free mark is dropped in.
- **"the stone's memory of the last press bends how deep this one goes"** = an XOR-fold that pulls high bits back down (`h ^= h >> k`), whose effect depends entirely on whatever bit pattern the previous byte already left behind (state-dependent, not byte-dependent scaling).
- **"read off final seated position against the wooden numbered keeps"** = a fixed 3-step avalanche finisher after the loop, then `return h`.
- **"groove-dust / intermediate turns swept away"** = no auxiliary array, no history kept — just the one stack-resident `h`.

This is a direct 64-bit literal scaling of Bob Jenkins' well-known, validated **"one-at-a-time" hash** (1997) — a real, previously-analyzed technique that satisfies exactly the assumption we're breaking (no multiply, only add/shift/xor), so per the contract's instruction I let the mechanism arrive at that known technique rather than inventing a novel untested one. Its 32-bit shift constants (10, 6 inner; 3, 11, 15 final) are doubled for the 64-bit word (20, 12; 6, 22, 30), preserving the same relative fold proportions.

Sequential dependency (assumption 1) and start-to-end single pass (assumption 4) are *not* broken — the metaphor is explicit that this is one continuous, unbroken fold regardless of pile length ("centuries of marks fold down just the same as a handful"), which rules out chunking the buffer for OpenMP thread parallelism: there is no valid unit of independent work under this metaphor, at any size. So, per the vectorization-before-threading guidance, I add only loop/compiler hints (`restrict`, a modest unroll pragma) and add no thread parallelism at all — dropping it isn't a workaround, it's what the metaphor itself mandates.

# ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

uint64_t kernel(const unsigned char *restrict data, size_t len) {
    /* seat the stone at a starting position that already reflects the pile's length */
    uint64_t h = (uint64_t)len;

    #pragma GCC unroll 8
    for (size_t i = 0; i < len; i++) {
        h += data[i];   /* press this mark's weight into the stone's current seated place */
        h += h << 20;   /* quarter turn through the groove: fold the current position forward */
        h ^= h >> 12;   /* the stone's memory of the last press bends how deep this one sinks */
    }

    /* final seating against the wooden numbered keeps: one settled reading, nothing else leaves the desk */
    h += h << 6;
    h ^= h >> 22;
    h += h << 30;
    return h;
}
```

# PREDICTION

PREDICTION: speedup_vs_dp = 0.75

Reasoning: per byte, FNV-1a executes 2 instructions on the critical dependency chain (`xor`, `imul`), and `imul` throughput on modern x86 is excellent (effectively 1/cycle, latency ~3 cycles). This kernel executes 5 chained instructions per byte (`add`, `shl+add`, `shr+xor`) — individually cheap (1-cycle latency each) but more of them on the same serial dependency chain that assumption 1 forces on both designs. So I expect measured throughput to be somewhat *below* the FNV-1a baseline (~0.7-0.8x), while avalanche quality should be comparable to or better than FNV-1a's, since Jenkins deliberately designed and tested one-at-a-time for near-50% output-bit-flip probability, whereas FNV-1a is known to have some avalanche weakness in its low output bits for short inputs.

# MEASUREMENT

Not executed in this session — no `hash_bench`/`hash_contract` tool is available here (only unrelated Gmail/Slack/Drive/Docs tools are exposed to me this turn). Per the session note, I'm reporting the prediction and the complete artifact for the external pipeline to compile and measure; I have not fabricated numbers in their place.

# VERDICT

This mechanism trades one multiply for several cheap shift/add/xor ops, arriving at a real, previously validated technique (Jenkins' one-at-a-time) rather than an invented one, and it does so without violating the metaphor's own insistence on a single, never-reset, in-place accumulator processed strictly in order. Its known risk — "could be worse than the known way" — is that modern hardware multiplies are cheap enough that trading 2 chained instructions/byte for 5 chained instructions/byte is likely to lose on raw throughput even though it avoids multiplication entirely, exactly per my prediction above. This risk is **not size-conditional** (it isn't "only overhead when the buffer is small" or "only pays off when large") — it's a constant per-byte instruction-count disadvantage that applies uniformly regardless of `len`. Since there is no size threshold to guard, a runtime size-check/fallback would be theater, not a real fix; the honest fix, if the measurement bears out the prediction, is algorithmic (e.g., reintroducing a single multiply into the fold), not a branch. I did not add OpenMP or SIMD-across-bytes: the metaphor forces one unbroken, order-dependent fold with no independent units of work at any pile size, so thread/vector parallelism across bytes would silently discard the seed's own claim rather than implement it — I dropped that option instead of shipping something that only pretends to honor the metaphor.