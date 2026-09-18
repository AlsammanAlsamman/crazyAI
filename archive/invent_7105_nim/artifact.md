## MAPPING

**SEED 1 — "lays a strip of embroidery against each cistern's brass gauge... where an odd number of gauges show a crossed thread in the same column, that column is unquiet"**

| World object | Problem object |
|---|---|
| cistern | a pile |
| brass gauge | binary representation of a pile's size |
| girls' embroidered cloth, doubling rank by rank down the hem | the fixed template of bit-place values 1,2,4,8,… |
| crossed thread | a set bit (1) at a given rank |
| bare gap | an unset bit (0) |
| laying the swatch against the gauge, column by column | iterating over bit positions, reading every pile's bit at that position |
| "odd number of gauges cross in that column" | parity (XOR) of that bit across all piles is 1 |
| unquiet column | a bit position where the XOR of all piles is 1 |

Assumption broken: *"the value of a position is unknown until every reachable position from it has been examined"* — here the position's status is read straight off static column parity; nothing about future moves is examined.

**SEED 2 — "opens the valve on the tallest disagreeing cistern and drains it until every column pairs off into silence"**

| World object | Problem object |
|---|---|
| tallest cistern | the pile with the largest value among those with the highest unquiet bit set |
| the valve | the single pile chosen for the move |
| draining until columns pair | reducing that pile to `pile[i] XOR xor_all` |
| "touch nothing else" | only one pile is ever modified |

Assumption broken: *"each heap must be considered on its own before the others"* — the drained amount for the chosen cistern is computed by comparing it against the combined pressure (XOR) of **every** heap at once, not by evaluating that heap in isolation.

**SEED 3 — "the winged lemur goes still the instant the count is true"**

| World object | Problem object |
|---|---|
| winged lemur | a post-move verifier |
| smelling imbalance from valleys off | recomputing parity of the resulting position directly, not by search |
| going still | confirming resulting XOR = 0 |

Assumption broken: *"the game must be played out to know who wins"* — balance is confirmed by an instantaneous re-read of the columns, never by playing out a continuation.

## CHOSEN SEED

SEED 1 (the embroidery/gauge column-reading). It is the most literal and the most structurally different from the known way: Bouton's textbook solution computes `xor_all` with a single pile-major accumulation (`xor_all ^= piles[i]`), treating XOR as an opaque hardware operator. SEED 1 instead demands a bit-major, column-by-column process — for every rank (bit position), sweep across every cistern (pile) and count crossed threads (set bits), only afterward reassembling the "pressure" from the column readings. That is a genuinely different loop nesting (transposed order) even though it is mathematically identical to XOR.

## ASSUMPTION BROKEN

"The value of a position is unknown until every reachable position from it has been examined." The world's process never looks at a *future* position at all — it reads the *current* gauges' bit-columns and their parities, a purely static, one-pass fact about the piles as they stand.

## ARTIFACT

```c
#include <string.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* The embroidered cloth: 32 columns (bit ranks), doubling rank by rank
       down the hem (1,2,4,8,...). For each column we lay the swatch against
       every cistern's brass gauge and count how many gauges show a crossed
       thread (a set bit) in that column. An odd count means the column is
       "unquiet" (it hums with disagreement). */
    int unquiet[32];
    memset(unquiet, 0, sizeof(unquiet));

    for (int b = 0; b < 32; b++) {
        int crossed = 0;
        for (int i = 0; i < n; i++) {
            crossed += (piles[i] >> b) & 1;
        }
        unquiet[b] = crossed & 1;   /* odd crossed threads -> humming column */
    }

    /* Reassemble the total pressure (xor of all piles) purely from the
       column readings -- this is reading the cloth, not looking ahead. */
    int xor_all = 0;
    for (int b = 0; b < 32; b++) {
        if (unquiet[b]) xor_all |= (1 << b);
    }

    if (xor_all == 0) {
        /* Every column already pairs off in silence: no winning valve
           exists. Touch the first cistern minimally, as the contract
           still requires a legal move. */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* Find the highest unquiet column -- the highest rank down the hem
       where the thread still hums. */
    int top = -1;
    for (int b = 31; b >= 0; b--) {
        if (unquiet[b]) { top = b; break; }
    }

    /* Among cisterns whose gauge shows a crossed thread in that column,
       pick the tallest one (largest water level). */
    int chosen = -1;
    int chosen_val = -1;
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> top) & 1) {
            if (piles[i] > chosen_val) {
                chosen_val = piles[i];
                chosen = i;
            }
        }
    }

    /* Open that valve and drain it until every column pairs off into
       silence again: new level = piles[chosen] XOR xor_all. */
    int target = piles[chosen] ^ xor_all;

    /* The winged lemur check: recompute the column parities as if the
       water had already run off, and confirm every column goes still
       (pure verification -- changes nothing when the count is true). */
    int still = 1;
    for (int b = 0; b < 32 && still; b++) {
        int crossed = 0;
        for (int i = 0; i < n; i++) {
            int v = (i == chosen) ? target : piles[i];
            crossed += (v >> b) & 1;
        }
        if (crossed & 1) still = 0;
    }
    if (!still) {
        /* Should never trigger by construction; defensive fallback only. */
        target = piles[chosen] ^ xor_all;
    }

    *out_pile = chosen;
    *out_remove = chosen_val - target;
}
```

PREDICTION: speedup_vs_dp = 0.12

Reasoning for the number: the reference/minimal solution does ~2 passes of O(n) hardware XORs. This artifact does a 32×n column-parity sweep, a 32-entry reassembly, a 2×n pile sweep, and a second 32×n verification sweep — roughly 64×n bit-ops versus ~2×n XORs, i.e. an order of magnitude more elementary work, only partly offset by the ops being cheap/branchless-ish. I expect it noticeably slower than the DP baseline (well under 1×), while producing bit-identical, 100%-correct moves since it is arithmetically equivalent to Bouton's theorem.

## MEASUREMENT

No `nim_bench`/`nim_contract` tools are exposed in this session (the environment note states explicitly that no tools are available here and that the harness will compile and measure the artifact externally). I have not fabricated a measurement; the number above is a stated prediction only, to be checked against the real `nim_bench` run by the pipeline. I did not perform the "improve at most four times" loop for the same reason — there is nothing to measure against yet in this session.

## VERDICT

Correctness: by construction this kernel computes exactly `xor_all` (via bit-column parity instead of hardware XOR) and exactly the Bouton move (`piles[chosen] ^ xor_all`), so it should match the reference kernel's win/loss behavior and move validity 100% of the time — it is not a different algorithm, only a different (and deliberately more literal, more expensive) route to the same number. Performance: expected to be markedly slower than the trivial DP/XOR-loop baseline because the world's "read every column of every gauge" process forces O(32n) work in three separate sweeps where the textbook method needs O(n) in two, so I predict a sub-1× speedup (~0.1×) rather than any gain — the fidelity to the seed's literal mechanism is bought at a real, honestly-reported performance cost, not concealed as a win.