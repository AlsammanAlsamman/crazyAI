## MAPPING

| World object | Problem object | Silent assumption touched |
|---|---|---|
| **Seed 1** — rolling sphere, "my only memory" | single `uint64_t` accumulator `h`, updated in place | *Confirms* "state is a single accumulator, one value" — doesn't break anything, it's the same assumption the known method already uses |
| pile of marks, given order | bytes `data[0..len)`, read in order | *Confirms* "whole buffer read once, in order" — not broken |
| **Seed 2** — pile of marks pressed one by one; sphere tumbles a *fixed* count of turns down a coiled mason trail; striking a stalk cracks a hairline; the crack's angle is carried forward | for each byte: incorporate it into `h`, then run a **fixed, small, constant number of add/shift/xor "tumble" steps** (no multiply) that perturb the bits of `h`; the resulting bit pattern of `h` is what's carried into the next byte | **Breaks** "mixing one byte requires a multiplication" — mixing here is done by cheap rotate/shift/xor "collisions", never a multiply |
| **Seed 3** — footprint at a stalk is "not mine", swept away; only the last hairline crack is handed over | no per-byte scratch/history kept; only the final `h` (after one finalization tumble) is returned | *Confirms* standard "return only final accumulator" behavior — every hash already does this, nothing new broken |

Only Seed 2 breaks one of the three "prefer if broken" assumptions (multiplication). Seeds 1 and 3 are literal but merely restate assumptions the known method already holds.

## CHOSEN SEED
**Seed 2**: "It tumbles a fixed count of turns down the mason trail, no more, no fewer, and where it strikes a stalk it sobs once, cracking a hairline into itself, and I read the angle of that crack as the next weight to carry forward."

## ASSUMPTION BROKEN
"Mixing one byte requires a multiplication." The tumble/crack mechanism is described entirely in terms of a ball rolling (shift), striking something (xor), leaving a hairline crack (add-derived bit perturbation) — never a product of two numbers. So the mixer must be built from shift + xor + add only.

Step 4 check: a well-known, validated real technique already satisfies exactly this — **Bob Jenkins' "one-at-a-time" hash** (1997; used historically in Perl's hashing, ancestor of `lookup3`/SpookyHash mixing ideas). Its per-byte step is literally "add the byte, tumble a fixed count of shift/xor turns" and its finalizer is "one last, smaller tumble, and that's the only thing you keep" — matching Seed 2 *and* Seed 3 almost verbatim. Per the instructions, I arrive at that technique rather than inventing a novel multiply-free mixer. I only generalize its shift constants from 32-bit to 64-bit, preserving their original ratios-of-width exactly (10/32→20/64, 6/32→12/64, 3/32→6/64, 11/32→22/64, 15/32→30/64), since the contract needs a 64-bit output.

Vectorization/threading: the tumble chain is a genuine read-after-write dependency (`h` feeds itself every step), same as FNV's chain — it cannot be lane-split without introducing a *second* sphere, which Seed 1 explicitly forbids ("the sphere is my only memory"). So, per the guard rule, I drop thread/SIMD parallelism entirely rather than ship an unaddressed risk, and only add the literal, cheap vectorization hint available: `restrict` on the pointer so the compiler doesn't have to guard against aliasing.

## ARTIFACT

```c
#include <stdint.h>
#include <stddef.h>

/* Bob Jenkins' one-at-a-time hash, generalized from 32-bit to 64-bit by
 * scaling its shift constants to preserve the original ratios-of-width
 * (10/32, 6/32 in the loop; 3/32, 11/32, 15/32 in the finalizer).
 * Mixing is done purely by add/shift/xor ("tumbles"), no multiplication -
 * a fixed count of tumbles (3) per byte, one running accumulator (the
 * sphere), only the final tumbled value kept (the last hairline crack).
 */
uint64_t kernel(const unsigned char * restrict data, size_t len) {
    uint64_t h = 0;                     /* the sphere: sole running memory */

    for (size_t i = 0; i < len; i++) {
        h += data[i];                   /* press the mark into the sphere */
        h += (h << 20);                 /* tumble 1 */
        h ^= (h >> 12);                 /* tumble 2 (strike -> crack)     */
    }

    /* the last, smallest crack: one final fixed tumble, kept as the token */
    h += (h << 6);
    h ^= (h >> 22);
    h += (h << 30);

    return h;
}
```

PREDICTION: speedup_vs_dp = 1.15

(Reasoning for the number: the FNV-1a baseline's per-byte critical path is `xor` (1 cyc) chained into a 64-bit `imul` (~3–4 cyc latency on modern x86) ≈ 4–5 cycles/byte. This kernel's per-byte chain is `add, shift+add, xor+shift` — three cheap ALU ops each ~1 cycle latency ≈ 3 cycles/byte, no multiply unit involvement. I expect a modest but real speedup, not a large one, since both are equally serial/latency-bound over the buffer with no parallel lanes. I also expect avalanche quality in the same ballpark as FNV-1a — decent but not top-tier, since Jenkins OAT is known to have some residual bit-correlation weaknesses versus purpose-built avalanche finalizers like xxHash/Murmur's multiply-based fmix, which is the real cost of refusing multiplication.)

## MEASUREMENT
No tools were available in this session (per the session's explicit instruction to reason and predict only, with the pipeline compiling/measuring separately) — `hash_bench` was not actually invoked here. The prediction above (`speedup_vs_dp = 1.15`) and the avalanche expectation ("comparable to, likely slightly below, a multiply-based mixer") are stated *before* any measurement, as required, so the harness's real numbers can be checked against them.

## VERDICT
This kernel is a literal, honest translation of the "tumble down the mason trail, no multiplication, one sphere, keep only the last crack" description into Jenkins' one-at-a-time hash, width-generalized to 64 bits. Its own stated risk — a fully serial single-accumulator dependency chain means it cannot be helped by SIMD or threads no matter the buffer size — is not something I tried to route around with an untested fix; instead I simply didn't add parallelism, which *is* the guard (no size check needed because there is no parallel fallback being risked in the first place). If the real measurement shows avalanche noticeably worse than FNV-1a (Jenkins OAT's known weakness), the correct next step would be to swap the finalizer alone for a stronger *multiplication-free* diffuser (e.g., an additional rotate-xor round) rather than reintroducing multiplication — since reintroducing it would just silently undo the one assumption this design exists to break.