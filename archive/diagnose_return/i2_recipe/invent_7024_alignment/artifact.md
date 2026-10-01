## DICTIONARY

| Recipe object | Computational object |
|---|---|
| **near cord / near row**, coil = one symbol | `const unsigned char *a`, index `i`. Read-only, never copied, never permuted ("do not move this row again"). |
| **shadow cord / shadow row** | `const unsigned char *b`, index-aligned: shadow coil `j` starts opposite near coil `i=j`. |
| **a coil cut to the exact shape of one symbol** | one byte in `{'A','C','G','T'}`; "same shape" = byte equality. |
| **the sun windowing through both** | the predicate `a[i]==b[j]`. |
| **one fire drops to the water** | `+1` added to the current room's accumulator. |
| **a quarrel** (they refuse each other) | `a[i]!=b[i]` on the straight walk — contributes nothing (recipe drops *nothing*, not a penalty). |
| **a crossing / a walk** | one candidate global alignment path. |
| **a room** | one `int32_t` accumulator holding that crossing's fire pile. |
| **the queen's threshold** | the register set of live candidate piles (the running maxima). |
| **the tower below** | the discard sink. A room that sinks is *unread*: encoded as the sentinel `SUNK`, arithmetically unreachable by any real pile. |
| **breaking the door** | inserting one gap‑pair: skip one coil of one row and hold a permanent ±1 offset for the rest of the walk. |
| **the shadow crouches forward** | offset the shadow row: after the break, `a[i]` faces `b[i+1]` (off‑diagonal +1). |
| **the near crouches forward** | offset the near row: after the break, `a[i+1]` faces `b[i]` (off‑diagonal −1). |
| **a place still owed** | a diagonal mismatch position `p` — the seed of exactly two new crossings (steps 7–9 and step 10). |
| **height of a pile** | the alignment score of that room (see the one repair below). |
| **time** | the single left‑to‑right index sweep. Nothing flows backwards. |
| **what stays still** | both cords (steps 1–2 forbid moving them). All motion is in the *offset*, not the data. |
| **what flows** | three running prefix counters `P`, `Q1`, `Q2`. |
| **the processor** | the scalar integer pipeline walking the threshold; the three totals are the vectorizable part. |

**Ambiguity resolved (step 7).** "dropping fires … *until you reach* that marked place. There, break the door" — I take the exclusive reading: coils `i < p` are walked straight, the door breaks *at* `p`, so `a[p]` faces `b[p+1]`. (Inclusive vs. exclusive cannot change the fire count, because `p` is a quarrel and drops no fire either way; it only fixes the geometry at `n−1` paired columns.)

**The one repair (step 12).** Followed exactly, step 12 hands back *a count of fires*. That is a match count, not a Needleman–Wunsch score, and the two families of rooms are not even commensurable: a straight room has `n` paired columns and no gaps (score `2F − n`), a broken room has `n−1` paired columns and two gap columns (score `2F − n − 3`). So comparing raw piles picks the wrong room. **Smallest change: in step 12 only, "compare the piles by height" means compare each room's pile converted to its own score‑height**, and hand back that height. Steps 1–11 are untouched.

**The discovery.** Taking "each crossing gets its own room" completely literally forces the question *what do two rooms share?* — and the answer is: every room's pile is `prefix of the straight walk` + `suffix of one of two fixed off‑diagonals`. So the recipe's `1 + 2q` walks of length `n` (naively `O(n²)`, exactly DP's cost) collapse to **`O(n)`: three prefix counters and one sweep of the threshold.** The native's insistence on one room per crossing is a separability statement about the rooms. That is the whole speedup.

## ARTIFACT

```c
/* "Rooms at the threshold, tower below, highest pile wins."
   Literal translation of the recipe. Steps 1-11 unmodified;
   step 12 compares piles by score-height (see DICTIONARY). */
#include <stdint.h>

int kernel(int n, const char *a, const char *b)
{
    /* step 1: lay the first cord flat along the near edge, in full sun,
       coil by coil left to right, one coil per symbol, and never move
       this row again for the whole work. -> `a` in place, no copy. */
    const unsigned char *near_row = (const unsigned char *)a;

    /* step 2: lay the second cord behind it along the shadow-line, the
       same way, its first coil standing opposite the first coil of the
       near row. -> `b` in place, index-aligned with the near row. */
    const unsigned char *shadow_row = (const unsigned char *)b;

    if (n <= 0) return 0;

    /* step 3: empty the tower below of every old room, and hang one
       empty room at the threshold for the crossing about to be walked.
       This first crossing is the straight one: no door may break in it.
       SUNK is the tower: any room holding it is unread by step 12. */
    const int32_t SUNK = -4 * (int32_t)n - 16;
    int32_t room_straight = 0;

    /* step 4: walk the crossing. Step once along both rows together,
       coil against coil. Where the near coil and the shadow coil are the
       same shape, one fire drops into this crossing's room. Where they
       refuse each other, drop nothing and step on anyway.
       step 5: keep stepping until one row runs out of coils, then close
       the room and let it hang at the threshold with its fires. */
    {
        int32_t f = 0;
        for (int i = 0; i < n; i++)
            f += (int32_t)(near_row[i] == shadow_row[i]);
        room_straight = f;                 /* room 0, closed, hanging */
    }

    /* step 6: count the quarrels seen on the straight walk and mark each
       place where one happened; each such place is one attempt owed.
       The mark is the predicate near_row[p] != shadow_row[p]; it is
       re-evaluated in place rather than materialised as a list. */
    const int32_t owed = (int32_t)n - room_straight;

    /* step 7: for each place still owed, hang a new empty room and walk
       again from the very first coil, dropping fires exactly as in step
       4 until that marked place; there break the door once - the shadow
       row's coil crouches forward past the quarrel, is skipped, and
       every shadow coil after it is held one step out of step.
       step 8: finish that walk to the end of the shorter row, still
       dropping a fire wherever the two facing coils agree; refuse any
       second break - the door breaks once per crossing only.
       step 9: do 7 and 8 again, once for every place still owed, each
       with its own fresh room, each breaking at its own single place.
       Every such room's pile is (straight prefix before p) + (shadow-
       forward suffix from p), so all the owed rooms are walked in one
       sweep of the threshold without re-walking any shared coil. */
    int32_t best_shadow_fires = SUNK;
    if (owed > 0) {
        int32_t q1tot = 0;                       /* whole +1 off-diagonal */
        for (int i = 0; i + 1 < n; i++)
            q1tot += (int32_t)(near_row[i] == shadow_row[i + 1]);

        int32_t P = 0, Q = 0, best = SUNK;
        for (int p = 0; p + 1 < n; p++) {
            int32_t agree = (int32_t)(near_row[p] == shadow_row[p]);
            int32_t d     = P - Q;               /* this room minus q1tot */
            int32_t m     = -agree;              /* no quarrel -> no room */
            int32_t cand  = (d & ~m) | (SUNK & m);
            best = cand > best ? cand : best;
            P += agree;
            Q += (int32_t)(near_row[p] == shadow_row[p + 1]);
        }
        {   /* the last place on the row */
            int32_t agree = (int32_t)(near_row[n-1] == shadow_row[n-1]);
            int32_t d     = P - Q;
            int32_t m     = -agree;
            int32_t cand  = (d & ~m) | (SUNK & m);
            best = cand > best ? cand : best;
        }
        best_shadow_fires = best + q1tot;
    }

    /* step 10: do the same the other way round - for each place still
       owed, hang another room and walk again, but let the NEAR row's
       coil crouch forward past the quarrel, skipping the near coil
       instead of the shadow one, holding that offset to the end. */
    int32_t best_near_fires = SUNK;
    if (owed > 0) {
        int32_t q2tot = 0;                       /* whole -1 off-diagonal */
        for (int i = 0; i + 1 < n; i++)
            q2tot += (int32_t)(near_row[i + 1] == shadow_row[i]);

        int32_t P = 0, Q = 0, best = SUNK;
        for (int p = 0; p + 1 < n; p++) {
            int32_t agree = (int32_t)(near_row[p] == shadow_row[p]);
            int32_t d     = P - Q;
            int32_t m     = -agree;
            int32_t cand  = (d & ~m) | (SUNK & m);
            best = cand > best ? cand : best;
            P += agree;
            Q += (int32_t)(near_row[p + 1] == shadow_row[p]);
        }
        {
            int32_t agree = (int32_t)(near_row[n-1] == shadow_row[n-1]);
            int32_t d     = P - Q;
            int32_t m     = -agree;
            int32_t cand  = (d & ~m) | (SUNK & m);
            best = cand > best ? cand : best;
        }
        best_near_fires = best + q2tot;
    }

    /* step 11: any walk spoiled - a door broken twice, a row run past
       its end, a fire dropped where the shapes did not truly agree - is
       cut loose and sinks into the tower, unread. By construction the
       door can break at most once per room (the offset is set once and
       held), no index leaves [0,n), and a fire is added only under byte
       equality; so the only rooms that ever sink are the two whole
       families when nothing is owed, which already hold SUNK. */

    /* step 12: go along the threshold and compare the piles by height,
       bring up the single room whose pile burned highest, and hand back
       that count and nothing else. REPAIR: height is the room's own
       score-height, not its raw fire count - the straight room pairs n
       coils with no gap (2F - n), a broken room pairs n-1 coils and
       leaves two coils unpaired (2F - (n-1) - 4). */
    int32_t score = 2 * room_straight - (int32_t)n;
    if (owed > 0) {
        int32_t s1 = 2 * best_shadow_fires - (int32_t)n - 3;
        int32_t s2 = 2 * best_near_fires   - (int32_t)n - 3;
        if (s1 > score) score = s1;
        if (s2 > score) score = s2;
    }
    return (int)score;
}
```

## PREDICTION

Cost model: 5 linear passes over `n` bytes (three cheap vectorizable reduction passes, two latency‑bound sweeps at ~3 cycles/element). At `n=1024`: ~8 k cycles ≈ 2.7 µs. A scalar reference NW is `n²` cells at ~1.2 ns/cell ≈ 1.26 ms.

**PREDICTION: speedup_vs_dp = 400**

Secondary prediction, stated before any measurement: **correctness will FAIL** on random or weakly‑related sequences for `n` beyond roughly 30–60, and the returned value will be a strict *under*estimate, never an overestimate.

## MEASUREMENT

**Not performed.** `alignment_bench` and `alignment_contract` were not reachable in this session (no tools were exposed), so I have no honest numbers to report and I will not invent any. The prediction above is stated cold, before measurement, exactly as required; the pipeline's numbers stand as the verdict on it.

What I *can* establish without the bench, by hand:

- **Lower‑bound property (proved, not measured).** Every room corresponds to a legal global alignment: the straight room is the diagonal path; a shadow‑break room at `p` is `a[0..p-1]/b[0..p-1]`, then `−/b[p]`, then `a[p..n-2]/b[p+1..n-1]`, then `a[n-1]/−`. Its NW score is exactly `2F − n − 3`. Hence `kernel(n,a,b) ≤ DP(n,a,b)` always. The failure mode is one‑sided.
- **Hand checks.** `n=1`, any pair → matches DP. `"AC"/"CC"` → 0 = DP. `"ACGT"/"TACG"` → `−1` = DP (gap, 3 matches, gap).
- **Where it must break.** Steps 8 and 11 forbid a second door. The DP optimum for two random 4‑letter sequences uses `Θ(n)` gap‑pairs; each break the recipe cannot afford costs it real score. So the gap between `kernel` and DP grows linearly in `n` on random input and is zero on input that is a pure substitution series or a single indel.

## VERDICT

The recipe is **fast and wrong**, and both halves of that are worth stating plainly.

*Fast:* taking "each crossing gets its own room" at face value — rooms as separate, closed, non‑interacting accumulators — exposes that the rooms are **separable**: every one of the `1 + 2q` piles is a prefix of the diagonal plus a suffix of one of exactly two off‑diagonals. The native's `O(n²)` walking collapses to three prefix counters and a single sweep, `O(n)` total, no scratch memory, no dependence chain longer than an integer add. That is a genuinely different shape from the DP, and it fell out of being literal rather than clever.

*Wrong:* the seed itself is the defect. "The door breaks once per crossing only" (steps 8, 11) restricts the search to alignments with at most one gap‑pair, anchored at a diagonal mismatch. That is a band of width 1 around the main diagonal with a single allowed transition — a heuristic, and a weak one. It reproduces the reference DP exactly when the optimum is a pure‑substitution diagonal or a single indel, and it under‑reports by a linearly growing margin otherwise. I did not patch this: repairing it means allowing unbounded door‑breaking, which is the Needleman–Wunsch recurrence itself, i.e. quietly throwing the native's idea away and calling the textbook method a translation of it. The honest report is that the recipe computes a cheap, provable lower bound on the global alignment score in linear time, and does not compute the global alignment score.

The correct next move, if this were to continue, is not to fix the kernel but to fix the question asked of it: this is a `O(n)` admissible bound, which is exactly the shape of a pruning filter in front of a real DP — not a replacement for one.

---

*Note on tooling:* the `claude.ai` and `PubMed` MCP connectors in this environment are unauthorized, and this session is non‑interactive so the OAuth flow cannot be run here. They will stay unavailable until you authorize them (claude.ai connectors via your claude.ai connector settings; others via `claude mcp` or `/mcp` in an interactive session). This did not affect the task above — `alignment_bench`/`alignment_contract` were simply not present.