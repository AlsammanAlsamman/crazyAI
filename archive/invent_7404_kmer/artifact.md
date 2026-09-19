## MAPPING

**SEED 1** — "A gut-string loop pinched over exactly k beads reads the current window as a single color-word."

| World object | Problem object |
|---|---|
| strand of beads | `seq[0..n-1]` |
| gut-string loop pinched at both ends over exactly k beads | the current k-mer window `[i, i+k)` |
| the four colors inside the loop | the 4 possible base values (A/C/G/T) held in that window |
| reading the loop's contents off as a single word | packing the window into one integer `code` (2 bits/base, high base first) |

Breaks: *"each k-mer's code must be built fully from scratch, one base at a time, before it can be counted"* — the loop is presented as a persistent object that already *reads* as a word, not as something reassembled letter‑by‑letter each time.

**SEED 2** — "I keep a cairn... I walk to the cairn whose order matches... lay one more pebble... if no cairn yet answers... begin one, using the reflection to confirm."

| World object | Problem object |
|---|---|
| one cairn per pattern-word | `counts[code]` bucket, one per possible k-mer code |
| laying a pebble | `counts[code]++` |
| "order the doors along the garden wall open, sideways" | the index space of `counts[]` is the same 2‑bit‑per‑base packing used elsewhere, not an arbitrary row |
| checking the reflection before starting a new cairn | avoiding a duplicate/colliding bucket (relevant for hashing, irrelevant here since indexing is direct) |

Breaks: *"the count table can only be updated one position's result at a time"* — described as visiting a stack that already exists (direct-address indexing), not a sequential array walk.

**SEED 3** — "the necessary throwing-away: the bead at the trailing end... I drop through the nearest sideways door... the loop slides forward exactly one bead... catching the next color."

| World object | Problem object |
|---|---|
| trailing bead at the far end of the loop | the base leaving the window, i.e. the top 2 bits about to be shifted past the window width |
| sideways door | the bit-mask `& ((1<<2k)-1)` that discards those bits |
| loop sliding forward one bead | `code = (code<<2 \| new_base) & mask` |
| cuff catching the next color | reading exactly one new base per step |

Breaks: *"each k-mer's code must be built fully from scratch, one base at a time, before it can be counted"* **and** *"one window is examined, then discarded, before the next begins"* — the window is never discarded, it **persists** and is updated in O(1), and each base is read exactly once total instead of up to k times.

## CHOSEN SEED

SEED 3. It is the most literal (the "door" is exactly the mask operation; the "trailing bead" is exactly the base that falls out of the shifted register) and it is the one that most directly attacks the naive algorithm's actual cost: re-scanning k bases per window.

## ASSUMPTION BROKEN

"Each k-mer's code must be built fully from scratch, one base at a time, before it can be counted" — and its consequence, "one window is examined, then discarded, before the next begins." Instead the window (the loop) is built once and then updated incrementally: each base is converted to a 2‑bit code exactly once for the whole sequence, not once per window it participates in.

## ARTIFACT

```c
#include <stdint.h>

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    /* how a bead's color is read off: A/C/G/T -> 2-bit code */
    unsigned char code_of[256] = {0};
    code_of[(unsigned char)'A'] = 0;
    code_of[(unsigned char)'C'] = 1;
    code_of[(unsigned char)'G'] = 2;
    code_of[(unsigned char)'T'] = 3;

    uint64_t mask = (k >= 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);

    /* the gut-string loop is pinched over the first k beads and
       assembled once, base by base -- this full build happens exactly
       once for the whole strand, never again */
    uint64_t code = 0;
    for (int j = 0; j < k; j++) {
        code = (code << 2) | (uint64_t)code_of[(unsigned char)seq[j]];
    }
    counts[code]++;

    /* the loop then slides forward one bead at a time: the cuff
       catches the next color at the front, and the sideways door
       (the mask) drops the trailing bead's bits for free -- no bead
       is ever re-read once it has left the loop */
    for (int i = k; i < n; i++) {
        code = ((code << 2) | (uint64_t)code_of[(unsigned char)seq[i]]) & mask;
        counts[code]++;
    }
}
```

PREDICTION: speedup_vs_naive = 8

Reasoning behind the number: naive is O(n·k) because every base is re-read and re-shifted into a fresh code up to k times (once per window it belongs to); the rolling version is O(n) — every base is converted and shifted exactly once, ever. The asymptotic gain is ~k, but both versions share the same O(n) floor (one pass over `seq`, one increment per position), so for a plausible bench k in the ~10–16 range I expect the measured wall-clock gain to land below the raw k-fold instruction-count reduction — hence 8 rather than ~12–16.

## MEASUREMENT

Not run. No tool access (`kmer_bench`/`kmer_contract`) was available in this session — the harness note explicitly says the reasoning and artifact are to be produced here and the pipeline will compile and measure it afterward. I am not fabricating a number for this section; the prediction above stands unverified until that run happens.

## VERDICT

Pending actual measurement. What I can state with confidence from the algorithm alone: this kernel strictly dominates the naive one in asymptotic work (O(n) vs O(n·k) base-conversions), it is a correct, literal translation of SEED 3 (mask = door, shift+or = loop sliding, single first-window build = loop first pinched into place), and it should not regress for any k, n — the only risk is that for very small k (k≈1–2) the constant-factor overhead of the extra masking makes the gain negligible rather than large. No further "at most four" improvement iterations were attempted since there is no measurement feedback available in this session to act on.