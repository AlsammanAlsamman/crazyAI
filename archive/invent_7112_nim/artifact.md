# MAPPING

**SEED1 — "fold two by two at ever coarser reaches"**

| World object | Problem object |
|---|---|
| heap | one pile, `piles[i]` |
| stick | one unit counted in that pile |
| folding sticks two by two | extracting one bit of `piles[i]` (divide/mod by 2) |
| a fold ("first-hand", "second-hand"...) | one bit position `k` |
| coarsest reach | the highest set bit (MSB) across the row |
| plainest single stick | bit 0 (LSB) |
| unpaired hand at a reach | bit `k` of `piles[i]` equals 1 |

Assumption broken: **none**. This step processes each heap by itself (binary decomposition), which is exactly what the "known way" and even naive minimax already do implicitly when they treat heaps as independent state components. It sets the stage but breaks nothing on its own.

**SEED2 — "the trouble slaps sideways, heap to heap, at that same reach"**

| World object | Problem object |
|---|---|
| unpaired hand's trouble | a 1-bit signal at fixed reach `k` |
| slapping sideways along the row | a sequential pass `i = 0..n-1` at fixed `k`, carrying a running flag |
| "that same reach only" | never mixes bit levels — one `k` at a time |
| heap swallows the slap, goes quiet | flag XOR 1 (already-1 bit) → 0 |
| heap takes the slap, grows loud | flag XOR 0 (already-0 bit) → 1 |
| "no future/earlier turns cross my mind, only the row as it stands" | the computation reads only the current `piles[]` values — no hypothetical successor position, no opponent reply, no recursion |

Assumption broken: **"a winning move can only be found by looking ahead through the game's possible futures."** This is the one seed whose own text names the broken assumption almost verbatim — the trouble-propagation never inspects a future position, only the present row's bits at one reach.

**SEED3 — "remove from whichever heap answers loud at the deepest reach"**

| World object | Problem object |
|---|---|
| deepest, coarsest reach that came back loud | highest set bit of `xor_all` |
| the heap loud at that reach | a pile `i` with that bit set (`piles[i] & bit`) |
| removing sticks until every reach folds quiet | reduce `piles[i]` to `piles[i] ^ xor_all` |
| "the rest I throw on the forest floor" | the discarded difference `piles[i] - target` |
| "touched once, correctly" | exactly one pile is modified — the move |

Assumption broken: **"the value of a position is unknown until every reachable position from it has been examined"** — the corrective heap and amount are read straight off the row's own bit signature, no successor positions are ever constructed or evaluated.

# CHOSEN SEED

SEED2 (the sideways slap). It is the only one of the three whose own wording explicitly rules out looking at other turns ("no future in my hands, only the row as it stands"), which is exactly the assumption the task prioritizes breaking. SEED1 and SEED3 are still needed to build a working kernel, but SEED2 is the centerpiece and gives the literal *shape* of the computation: not "XOR the array" as one opaque operator, but a heap-by-heap ripple pass at each bit level.

# ASSUMPTION BROKEN

"A winning move can only be found by looking ahead through the game's possible futures." The native's process never constructs a hypothetical position, never asks "what if I moved here", never recurses. It only ever reads the *current* bit pattern of the row, reach by reach, heap by heap.

# ARTIFACT

Every object mapped literally: heap = `piles[i]`; a "fold" = one bit position `k`, walked from the coarsest reach present (`maxbit-1`) down to the plainest (bit 0); "trouble slapping sideways" = a `trouble` flag XORed sequentially across the row at fixed `k`; "quiet across the whole row" = `xor_all == 0`; "loud at the deepest reach" = the top set bit of `xor_all`; "touched once" = a single pile is reduced.

```c
#include <string.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* SEED2, literally: for each reach (bit level), from the coarsest
       fold present across the row down to the plainest single stick,
       walk the heaps in order letting an unpaired hand's "trouble"
       slap sideways: a heap already unpaired at that reach (bit=1)
       swallows the slap and goes quiet (1^1=0); a heap already paired
       (bit=0) takes the slap and grows loud (0^1=1). Only the row as
       it currently stands is ever consulted - no future turn, no
       hypothetical successor position is built or examined. */

    /* the coarsest reach: highest bit present anywhere in the row */
    int maxbit = 0;
    for (int i = 0; i < n; i++) {
        int v = piles[i];
        int b = 0;
        while (v > 0) { v >>= 1; b++; }
        if (b > maxbit) maxbit = b;
    }

    int xor_all = 0; /* which reaches came back loud across the row */

    for (int k = maxbit - 1; k >= 0; k--) {
        int trouble = 0; /* the row starts quiet at this reach */
        for (int i = 0; i < n; i++) {
            int hand = (piles[i] >> k) & 1;  /* unpaired(1) / paired(0) here */
            trouble ^= hand;                  /* swallow or grow loud */
        }
        if (trouble) xor_all |= (1 << k);     /* this reach answered loud */
    }

    if (xor_all == 0) {
        /* every reach, coarsest to plainest, came back quiet:
           the row itself must move me next turn, not my will */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* the heap loud at the deepest, coarsest reach: touch it alone,
       removing sticks until every reach in the row folds quiet */
    for (int i = 0; i < n; i++) {
        int target = piles[i] ^ xor_all;
        if (target < piles[i]) {
            *out_pile = i;
            *out_remove = piles[i] - target;
            return;
        }
    }
}
```

PREDICTION: speedup_vs_dp = 0.05

Reasoning for the number: correctness should be 100% — this is still exactly Bouton's move, just computed by an explicit bit-by-bit ripple instead of one `x ^= piles[i]` accumulation. But the literal translation costs an extra factor of roughly `maxbit` (≈20–30 for typical 32-bit pile sizes) in work, since the row is walked once per bit level instead of once total. I expect it to land somewhere around 15–30× slower than the direct-XOR baseline, hence ≈0.05× (20×) the speed.

# MEASUREMENT

No `nim_bench` / `nim_contract` tool call was available in this session (the tool list provided to me for this turn does not include them), so I cannot execute the harness myself. The prediction above is recorded before any measurement, as required; the pipeline should compile and run this exact kernel through `nim_contract`/`nim_bench` to get the actual accuracy and speedup numbers. I have not silently substituted the textbook single-pass XOR for the ripple version — the artifact above is the literal one, even though I predict it costs real time.

# VERDICT

Pending actual measurement by the pipeline. Predicted outcome: **accuracy ≈ 100%** (it is mathematically identical to Bouton's theorem — same `xor_all`, same chosen pile, same amount removed), **speed ≈ 20× slower** than a direct DP/XOR baseline, because staying literal to "fold, then let trouble slap sideways reach by reach" turns one O(n) pass into O(n·bits). If measurement confirms accuracy=100% with a real slowdown, that would be the honest result of taking the native's process seriously rather than quietly collapsing it back into `x ^= piles[i]`.