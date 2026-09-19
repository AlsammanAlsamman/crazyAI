# MAPPING

| World object | Problem object |
|---|---|
| duck | one object/unit inside a pile |
| flock | one pile, `piles[i]` |
| beauty-shelf | one binary place-value / bit position (1, 2, 4, 8, …) |
| "climb the shelves, read forward and backward in one glance" | decompose a pile's count into its bits |
| "a shelf lit by an even number of flocks... let go unremembered" | for bit position *b*, count piles with bit *b* set; if even, that bit of the XOR is 0 |
| "what remains lit and alone, highest such shelf" | the most-significant set bit of `xor_all` |
| "if nothing remains lit anywhere... hunter loses" | `xor_all == 0` → mover is in a losing (P-)position |
| "whichever flock lit it... the leader, chosen by arithmetic" | the pile `i` whose bit at the top surviving shelf is set |
| "trim that flock, duck by duck, until its own shelves go dark exactly where all others still burn" | reduce `piles[i]` to `piles[i] ^ xor_all` |
| "hand the shot back... hunter cannot recover" | the resulting position has XOR 0 for the opponent |

Per-seed breakdown:

**SEED 1** ("beauty-shelves stack each flock... count read at a glance")
- beauty-shelf → bit position; flock → pile; "read at a glance" → a pile's value is fully available as bits with no incremental scan.
- Breaks (weakly, per-heap only): a single heap's "value" (its bits) is known instantly, without inspecting any move from it. Doesn't touch the *game's* win/loss value.

**SEED 2** ("Forgetting cancels every shelf lit by a paired number of flocks, leaving only the one true mark")
- shelf → bit of the running XOR; "lit" → bit set in a given pile; "paired flocks" → even count of piles with that bit set; "forgetting cancels" → XOR/parity; "one true mark" → the position's win/loss verdict, read off directly from current pile shapes.
- Breaks directly: **"the value of a position is unknown until every reachable position from it has been examined."** The position's status (winning iff some shelf still burns) is read off the *current* piles alone — zero reachable positions are examined.

**SEED 3** ("the flock holding the leader is trimmed, duck by duck, until its own shelves go dark exactly where all others still burn")
- flock → chosen pile; leader → that pile's index; trimming → the actual `piles[i] -= remove`.
- Mostly *confirms* "a move only affects the one heap it touches" (only one flock is trimmed) rather than breaking any assumption; it executes the move whose target was already diagnosed by Seed 2's logic, without playing the game out further.

# CHOSEN SEED

**SEED 2** — it is the only one of the three that directly breaks the assumption "the value of a position is unknown until every reachable position from it has been examined," which the task asks us to prefer. Its mapping (bit-position parity counting) is also maximally literal to the native's words: shelves = bit positions, "paired flocks cancel" = even-count bits vanish, "the one true mark" = the surviving high bit — nothing here is a paraphrase or reinterpretation.

# ASSUMPTION BROKEN

"The value of a position is unknown until every reachable position from it has been examined." Bouton's parity-counting diagnosis reads the position's status straight off the piles present *right now*; no future/child position is ever constructed or inspected.

# ARTIFACT

Literal mapping of every world object to a computational one:
- **heap/pile** = `piles[i]`, an integer count of ducks.
- **shelf** = bit position `b` in `0..30`.
- **"a shelf is lit by a flock"** = `(piles[i] >> b) & 1`.
- **"forgetting cancels paired shelves"** = parity (`count & 1`) of how many piles light shelf `b`; this literally re-derives `xor_all` bit-by-bit instead of trusting a single hardware `^=` reduction.
- **"the one true mark"** = highest `b` with odd parity = MSB of `xor_all`.
- **knowing (win/lose)** = `mark < 0` ⇔ `xor_all == 0` ⇔ mover loses.
- **the leader** = the pile index `i` with bit `mark` set.
- **trimming the flock** = `remove = piles[i] - (piles[i] ^ xor_all)`.

```c
#include <stddef.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* "beauty-shelves" = binary place-value rows: 1,2,4,8,... */
    const int SHELVES = 31; /* enough shelves for any 32-bit signed pile size */

    /* For each shelf, count how many flocks (piles) light it (have that
       bit set). A shelf lit by an even number of flocks is let go
       unremembered (cancels, "forgetting"); an odd count leaves it
       burning. */
    int lit[31];
    int xor_all = 0;
    for (int b = 0; b < SHELVES; b++) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            if ((piles[i] >> b) & 1) count++;
        }
        lit[b] = count & 1; /* 1 if this shelf still burns after forgetting pairs */
        if (lit[b]) xor_all |= (1 << b);
    }

    /* The mark I trust: the highest shelf still burning. */
    int mark = -1;
    for (int b = SHELVES - 1; b >= 0; b--) {
        if (lit[b]) { mark = b; break; }
    }

    if (mark < 0) {
        /* No shelf burns anywhere: the hunter (mover) already loses.
           No winning move exists; take anything legal. */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* Go to whichever flock lit the highest surviving shelf: the leader,
       picked by arithmetic (bit presence), not by plumage. */
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> mark) & 1) {
            /* Trim that flock, duck by duck, until its own shelves go
               dark exactly where all others still burn. */
            int target = piles[i] ^ xor_all;
            if (target < piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - target;
                return;
            }
        }
    }

    /* Unreachable given mark's definition, but stay legal. */
    *out_pile = 0;
    *out_remove = 1;
}
```

This is arithmetically identical to Bouton's theorem — `xor_all` is reconstructed bit-by-bit (an explicit enactment of "climbing the shelves and letting paired light cancel") instead of via a single hardware XOR reduction, and the move is the standard one. Complexity is `O(n · 31)` (constant-factor larger than `O(n)`), still no search of reachable positions whatsoever.

# PREDICTION

PREDICTION: speedup_vs_dp = 500

Reasoning: a game-tree/DP baseline for Nim explores reachable positions (exponential in pile sizes or heap count for a naive memoized minimax), whereas this kernel does a fixed `O(31n)` scan with no recursion and no state table. I expect accuracy 100% (mathematically identical to Bouton's rule) and a large constant-factor+ speedup, likely in the hundreds-to-thousands range depending on the DP baseline's pile-size scaling; I'm giving 500 as an order-of-magnitude central estimate, with wide uncertainty since I have not seen the DP baseline's exact size parameters.

# MEASUREMENT

No tool execution was performed in this session (the session explicitly disabled tool calls: "No tools are available in this session... the pipeline will compile and measure it"). I did not run `nim_bench` or `nim_contract` myself, so I cannot report an actually-observed speedup/accuracy number here — only the pre-registered prediction above. Reporting this plainly rather than fabricating a measured result.

# VERDICT

Correctness: the kernel is a literal, bit-by-bit re-derivation of Bouton's XOR rule (parity-of-lit-shelves ≡ XOR), so it should be 100% accurate against `nim_contract` — it is not a heuristic, it is the exact theorem computed by an unconventional (explicit bit-loop) route rather than the hardware `^` operator. It breaks the targeted assumption (no reachable position is ever examined to know the position's value) more explicitly than the minimal baseline example does, since it makes the "no search" property visible in the algorithm's structure (per-shelf parity tally) rather than hiding it inside one CPU instruction. Given the session has no measurement tool, this is presented as a single, un-iterated artifact with an explicit, falsifiable prediction rather than four rounds of tool-guided tuning — consistent with the instruction to state predictions honestly rather than simulate measurements I cannot actually take.