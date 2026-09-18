# MAPPING

| World object | Problem object | Silent assumption broken |
|---|---|---|
| **SEED 1** — heap count written as nested circular glyphs, one ring inside the next, "that script already halves a number" | A pile's size expressed in its own machine representation: bit *b* = whether ring *b* is present, obtained by repeated halving (shift‑right) exactly like the ordinary binary expansion | *"the value of a position is unknown until every reachable position from it has been examined"* — the ring‑pattern already **is** the full value; nothing about it needs to be discovered by exploring future positions. |
| **SEED 2** — laying heaps' glyphs side by side under the twin moon, reading down every column at once "the way a physicist's eye jumps straight to where the beam already lands"; a column with an odd count of filled rings means "the sentence is still open" | For each bit‑position *b*, count how many piles have that bit set; odd count = that bit of XOR‑of‑all‑piles is 1. Reading *all* columns *simultaneously* across *all* heaps = computing `xor_all` in one pass, not heap‑by‑heap and not by search. | Breaks **two** assumptions at once: *"a winning move can only be found by looking ahead through the game's possible futures"* (no lookahead — the open columns are read directly off the current position) **and** *"each heap must be considered on its own before the others"* (columns are read across all heaps together, in parallel). |
| **SEED 3** — standing motionless like the reading skeleton, discarding only what was already removed, and re‑closing whatever column the opponent reopens; "I have already seen the ending" | The mirroring/invariant strategy: once every column is closed (`xor_all = 0`) after your move, you know the outcome already, and only ever react to restore the invariant — you never need to simulate the game to termination. | Breaks *"the game must be played out to know who wins"* — the win is known from the single computed move onward, before any further play happens. |

# CHOSEN SEED

**SEED 2** — reading every glyph‑column at once, physicist's‑eye‑fashion, to find which columns are "open" (odd) and which single heap reaches the highest open one.

It is the most literal (each ring = a bit, each column = a bit‑position, "odd count" = XOR bit) **and** the most different from the textbook framing: the textbook says "compute XOR" as an atomic black‑box operation; the native's ritual insists on building the *column* (per‑bit‑position, cross‑heap) view explicitly, one ring‑layer at a time, before anything is touched — i.e. it demands the parity be read out digit‑column‑by‑digit across the whole row of heaps simultaneously, not folded pile‑by‑pile the way a running XOR accumulator would.

# ASSUMPTION BROKEN

*"A winning move can only be found by looking ahead through the game's possible futures"* and *"each heap must be considered on its own before the others."* No heap is examined in isolation, and no future position is ever visited — the move is read directly off the present ring‑pattern of all heaps at once.

# ARTIFACT

```c
void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* SEED 1: each pile's count is already nested circular glyphs --
       bit b = whether ring b is filled, found by repeated halving. */

    /* SEED 2: lay all heaps' glyphs side by side and read every column
       (bit-position) at once, across every heap simultaneously.
       A column with an odd number of filled rings is "still open". */
    int col_parity[32];
    for (int b = 0; b < 32; b++) {
        int filled = 0;
        for (int i = 0; i < n; i++) filled += (piles[i] >> b) & 1;
        col_parity[b] = filled & 1;               /* odd == open */
    }
    int xor_all = 0;
    for (int b = 0; b < 32; b++) xor_all |= (col_parity[b] << b);

    if (xor_all == 0) {
        /* every column already closes even -- no move reopens what
           isn't already paired; take one straw, contract needs a move */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* highest unclosed column */
    int highest_open = 31;
    while (!((xor_all >> highest_open) & 1)) highest_open--;

    /* SEED 2 cont'd: the one heap whose glyphs reach that column */
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> highest_open) & 1) {
            /* take exactly enough luggage that every column closes
               even again: the heap's new ring-pattern must equal the
               parity of everybody else's rings, i.e. piles[i] ^ xor_all */
            int target = piles[i] ^ xor_all;
            if (target < piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - target;
                return;
            }
        }
    }

    /* unreachable when xor_all != 0, kept only to satisfy the contract */
    *out_pile = 0;
    *out_remove = 1;
}
```

PREDICTION: speedup_vs_dp = 1000

Reasoning for the prediction: the "dp" baseline referred to in the harness is a game‑state search (the standard silent‑assumption approach — memoized lookahead over reachable positions), whose cost scales with the size/product of pile values rather than with `n`. This kernel does two linear‑in‑`n` passes (bounded by 32 bit‑columns), so its cost is O(n) with a small constant, independent of pile magnitude. Against any DP baseline whose state space grows with pile values, the gap should be large (many orders of magnitude for non‑trivial pile sizes); I'm giving a conservative round number (1000×) rather than guessing the exact ratio, since I have no way to run the actual benchmark in this session.

# MEASUREMENT

Not run in this session — no tools (including `nim_bench`/`nim_contract`) were available to me here; per the task's own note, the artifact and prediction above are handed to the external pipeline, which compiles and measures it. I am reporting this plainly rather than fabricating numbers.

# VERDICT

Pending the pipeline's actual run of `nim_contract`/`nim_bench`. Correctness expectation: the kernel is algebraically identical to Bouton's theorem (the column‑parity vector built bit‑by‑bit is exactly `xor_all`, and `piles[i]^xor_all` is exactly the standard target), so it should score 100% on accuracy against any Nim win/lose oracle. Speed expectation: O(n) like the minimal reference solution, with a modest constant‑factor overhead (an explicit 32‑wide column scan instead of a single folded XOR) — so versus the *minimal XOR example* I'd expect near‑parity or a small slowdown, but versus the exponential/lookahead **dp** baseline named in the contract, the predicted ~1000× speedup should hold up once actually measured.