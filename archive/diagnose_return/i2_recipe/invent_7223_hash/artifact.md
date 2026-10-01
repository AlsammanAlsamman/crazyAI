## DICTIONARY

| Recipe object | Computational object |
|---|---|
| a **mark** | one input byte `data[i]` (8 bits) |
| a **flagstone** | index `i` into the buffer |
| the **coiled limestone trail**, high mouth → down | the buffer traversed `i = 0 … len-1`, ascending = downhill |
| "lay them in the order given, do not sort, do not skip" | strictly sequential single-byte scan; no reordering, no block/tree reduction, every byte consumed exactly once |
| "count them as you lay them" | `stones = len` |
| the **sphere** | one `uint64_t sphere` register — **the entire state**. No array, no lane vector, no second accumulator |
| the sphere's **face** | the current 64-bit value of that register |
| the **priest's opening shape** | a hardcoded init constant `0xcbf29ce484222325` (FNV-1a-64 basis). Fixed ⇒ two makers apart agree; not seeded by time/address |
| "press the mark in **alongside**, held on one face, combined, **not side by side in a row**" | `sphere ^= byte` (combination on the same word). Explicitly *not* `sphere = (sphere<<8)\|byte`, which is the forbidden "row" |
| a **tumble / turn** | one round of the mixing function |
| the **fixed count of turns** | `TURNS`, a compile-time constant, identical for every byte ⇒ zero data-dependent branches |
| **striking a stalk** | multiply by an odd 64-bit constant (`0x9e3779b97f4a7c15`) — the operation that carries low bits up into high |
| **sobbing / cracking a hairline** | the xor-fold of a thin high slice back onto the whole word: `sphere ^ (sphere >> 32)` |
| the **angle of the crack** | a rotation; the rotated crack becomes the new face: `sphere = rotl(crack, 27)` |
| "the old face is **dead**, let it shrink away" | pure overwrite of the single register; no history is retained anywhere |
| **footprint / stone dust / eggshell → open plumbing** | all per-turn temporaries die at end of turn; **zero scratch memory allocated**; O(1) working set |
| **walking back up to the high mouth** | the loop back-edge |
| "**marks × fixed count**" equality (step 8) | `tumbles == stones * TURNS`, a real source-level stopping proof (discharged at compile time) |
| pressing **the number of marks** in (step 9) | `sphere ^= (uint64_t)len`, then `TURNS` more rounds — length finalization |
| the **token** | the return value: the final crack, un-rotated |
| **throwing the sphere into the plumbing** | the final face is never returned or stored; no `static`/global/cache ⇒ the function is pure |
| **changing one mark and comparing tokens** (step 12) | the avalanche test; `TURNS` is the knob it tunes |

**Ambiguities, resolved to the most literal reading:**
- *Step 5, "the angle of the crack ... as the new face"* — the crack is the xor-fold, the angle is a rotation of it. The three verbs (strike / sob-crack / read-angle) map to the three ops multiply / xorshift-fold / rotate, in that order.
- *Step 10, "that last, smallest one only"* — the token is the crack of the **final** stalk only (not a sum of all cracks) and it is the crack, **not** the face, so the returned word differs from the discarded final face by exactly the step-5 rotation. This is what makes steps 10 and 11 non-redundant.
- **One repair.** The hyper-literal reading of "hairline ... smallest" as the bare 32-bit sliver `sphere >> 32` would return a 64-bit value with 32 always-zero bits — a guaranteed avalanche failure under the contract. Smallest fix, taken from the recipe's *own* rule in step 3: things pressed onto one face are **combined**, so the crack is the full-width fold `sphere ^ (sphere >> 32)`. Nothing else is changed.
- **Deliberately unused:** OpenMP, `immintrin.h`, scratch memory. The seed makes the sphere "the sole carried memory" and the order strict; any parallel accumulator or multi-lane state would be a second memory and a second trail. I refuse that substitution and pay for it in throughput below.

## ARTIFACT

```c
#include <stddef.h>
#include <stdint.h>

/* ---- the world's fixed constants ----------------------------------- */
#define TURNS          2u                      /* the fixed count of turns   */
#define OPENING_SHAPE  0xcbf29ce484222325ULL   /* the priest's opening shape */
#define STALK          0x9e3779b97f4a7c15ULL   /* the stalk (odd => bijective) */
#define HAIRLINE       32                      /* width of the crack it sobs */
#define ANGLE          27                      /* the angle read off the crack */

static inline uint64_t rotl64(uint64_t x, unsigned r)
{
    return (x << r) | (x >> (64 - r));
}

uint64_t kernel(const unsigned char *data, size_t len)
{
    /* step 1: lay the pile along the coiled trail in the order given, one
       mark to one flagstone, high mouth downward -- nothing sorted, no
       stone skipped -- and count them as they are laid.  The buffer IS the
       trail: flagstone i carries mark data[i], ascending i = downhill. */
    const unsigned char *trail  = data;
    const size_t         stones = len;

    /* step 2: set a fresh sphere at the high mouth and press the priest's
       opening shape into its face -- the same shape for every pile, so two
       makers working apart begin identically.  From here this one register
       is the ONLY memory: no array, no lanes, no note kept anywhere. */
    uint64_t sphere  = OPENING_SHAPE;
    uint64_t token   = 0;   /* the crack that step 10 will lift out */
    size_t   tumbles = 0;   /* turns spoken aloud, for the step-8 equality */

    for (size_t stone = 0; stone < stones; ++stone) {

        /* step 3: take this flagstone's mark and press its shape into the
           sphere's face ALONGSIDE the shape already there -- carried weight
           and new mark combined on the one face, not set side by side in a
           row (hence xor, never shift-and-append). */
        sphere ^= (uint64_t)trail[stone];

        /* step 4: release it and let it tumble exactly the fixed count of
           turns -- same count for every mark, never fewer for a simple one
           nor more for a stubborn one, so no branch anywhere depends on the
           data.  The count is spoken aloud into `tumbles`. */
        for (unsigned turn = 0; turn < TURNS; ++turn) {
            uint64_t crack;

            /* step 5: it strikes a stalk (full-width odd multiply, which
               carries the WHOLE face forward, low bits up into high), sobs
               once and cracks a hairline (the thin high slice folded back
               onto the face); read the angle of the crack it carries and
               take that angle as the new face.  The face it had before is
               dead the instant it is overwritten -- one register, nowhere
               for it to survive, never looked at again. */
            sphere *= STALK;
            crack   = sphere ^ (sphere >> HAIRLINE);
            sphere  = rotl64(crack, ANGLE);

            /* step 6: sweep the footprint at that stalk, the stone dust and
               any eggshell off into the open plumbing.  Every temporary of
               this turn dies here and no scratch memory is ever allocated;
               nothing is kept but the sphere and its one new face (plus the
               single running candidate for the step-10 token). */
            token = crack;
            ++tumbles;
        }

        /* step 7: walk the sphere back up to the high mouth and begin again
           with the next flagstone's mark, in strict order down the trail --
           press in, tumble, read the crack, discard the old face, sweep. */
    }

    /* step 8: stop neither early on a pleasing face nor late out of
       tiredness: the flagstone just used is the last one laid AND the turns
       made equal marks times the fixed count.  That equality is the proof
       of completion; here it holds by construction, so the check below is
       discharged rather than merely hoped for. */
    if (tumbles != stones * (size_t)TURNS) return 0;   /* unreachable */

    /* step 9: after the last mark, one final tumble of the fixed count with
       the number of marks itself pressed into the face, so a short pile and
       a long pile that happen to run alike still part ways at the end. */
    sphere ^= (uint64_t)stones;
    for (unsigned turn = 0; turn < TURNS; ++turn) {
        uint64_t crack;
        sphere *= STALK;
        crack   = sphere ^ (sphere >> HAIRLINE);
        sphere  = rotl64(crack, ANGLE);
        token   = crack;
    }

    /* step 10: take the hairline crack in the final stalk -- that last one
       only, un-rotated, not the sphere's face and not any sum of the earlier
       cracks -- and lift it out as the token. */
    /* step 11: throw the sphere itself into the plumbing.  No face it wore
       along the way is copied anywhere -- no static, no global, no cache:
       the function is pure, and anyone who wants the token again must lay
       the pile out and roll it from the opening shape. */
    return token;
}
```

*(Step 12 is the recipe's verification, not a kernel step, so by the "do not add a step" rule it is not a block in the code — it is the measurement below, and the sole thing it tunes is `TURNS`.)*

## PREDICTION

PREDICTION: speedup_vs_dp = 0.03

Stated before any measurement (none was possible — see MEASUREMENT). Basis:

- **Latency, not bandwidth, is the cost.** The seed forbids a second memory, so there is exactly one dependency chain: `imul` (3 cyc) → `shr`+`xor` (2 cyc) → `rol` (1 cyc) ≈ **6 cycles per turn**, × `TURNS=2` ≈ **12–13 cycles per byte**. Loads, the index increment and the `token` move are all off that chain and hide under it.
- A dot-product reference compiled `-O3 -march=native` is vectorised and bandwidth-bound: order 0.2–0.5 cycles per byte. Ratio ≈ 0.4/12.5 ≈ **0.03** (I'd accept anything in 0.015–0.06 as confirming).
- **Avalanche should be near-perfect:** every turn is a bijection on the full 64-bit face (odd multiply ∘ xor-fold with shift ≥ 1 ∘ rotate), so no information is ever lost mid-trail — exactly what step 12 demands ("carried the whole face forward and not a sliver of it"). The worst case, a flip in the *last* byte, still gets 2·`TURNS` = 4 bijective rounds plus the length injection. I predict flip fraction 0.50 ± 0.01 and an avalanche score ≥ 0.99.

## MEASUREMENT

**Not measured. `hash_bench` and `hash_contract` were not available in this session** (the environment exposes only Gmail/Drive/Slack/Docs MCP tools; no `hash_bench`, and no shell to compile with). So I have zero measured numbers, and the four permitted improvement rounds were not spent — the artifact above is the un-tuned first roll. Reporting that plainly rather than inventing a table.

What I would have run, and the decision rule fixed in advance:

1. `hash_contract` to confirm the signature, then `hash_bench` on the artifact as written.
2. **If avalanche score < 0.99 or flip fraction outside 0.49–0.51** → step 12's own instruction applies verbatim: `TURNS` is too small. Raise `TURNS` 2 → 3 → 4, re-roll every pile from the opening shape, accept the proportional throughput loss (≈ 18, ≈ 25 cycles/byte). Nothing else changes; the diagnosis is named by the recipe.
3. **If avalanche passes** → the only honest downward probe is `TURNS = 1`, which is still "a fixed count, the same for every mark" and therefore legal under step 4; it would roughly double throughput (≈ 6–7 cycles/byte, predicted speedup ≈ 0.06) while leaving the last byte 2 rounds + the fold. I would adopt it **only** if avalanche stayed ≥ 0.99, and revert on any degradation — step 12 permits raising the count, never lowering it on a hunch.
4. I would **not** have tried the obvious speed fix — several independent lanes over strided slices of the buffer, combined at the end, which would reach ~1 cycle/byte. That is a second carried memory and a re-ordered trail. It is the textbook answer and it is not this recipe.

## VERDICT

The recipe is a **faithful, complete, and correct description of an iterated single-register streaming hash with a per-byte round count and a length-injected finalizer** — recognisably the FNV/xorshift-multiply family, arrived at through masonry. Every clause pulls its weight: the fixed init is reproducibility across implementations; "combined, not in a row" is xor rather than shift-append; the constant turn count is timing-invariance; "the whole face, not a sliver" is the bijectivity requirement on the round function; step 9 is the length-extension/ambiguity fix that a naive byte loop genuinely lacks; step 11 is purity; step 12 is the avalanche test *with the correct tuning knob identified*. One genuine subtlety is encoded too: it insists the output be the crack rather than the sphere — a detail that costs nothing and shows the native was tracking what leaves versus what is destroyed.

Where the recipe is honestly weak is **throughput, and it is weak by construction, not by accident.** "A rolling sphere serves as the *sole* carried memory across the whole *ordered* pile" is precisely the constraint that makes a modern hash fast when you drop it: one serial multiply chain instead of 4–8 independent ones, one byte per step instead of 8. My predicted 0.03 vs a vectorised dot product is the price of the seed, and I paid it rather than smuggling in lanes. A reader who wants ~30× more speed must reject the seed's first sentence — and should say so out loud, not quietly.

Two caveats I cannot discharge: the prediction is unverified, and the single interpretive repair (step 10's "smallest" crack widened from a 32-bit sliver to the full-width fold) is mine, not the native's — without it the kernel would return 32 zero bits and fail the contract outright.

---

*Note: the `claude.ai PubMed` MCP server shown in this environment is unauthorized and cannot be authorized from a non-interactive session — it would need authorizing via your claude.ai connector settings (or `claude mcp` / `/mcp` in an interactive session). It was not needed here; the blocking gap for this task was `hash_bench`/`hash_contract`, which are absent entirely.*