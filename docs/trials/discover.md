# Discover mode: baseline ablation and more seeds

*Runs 2026-09-20 · pipelines `crazyai discover`, `crazyai discover-baseline`
· archives `archive/discover_8001..8005`, `archive/discover_8101..8115`, `archive/discover_baseline_9001..9005`*

[← back to trials index](README.md)

## What discover does

No target, no contract. Three personas in sequence:

1. **Maker** - invents a mechanism/rule/pattern purely because it is
   interesting in its own blended world (no "solve" vocabulary).
2. **Observer** - states its structure as generally as possible (allowed to
   say "this doesn't reduce cleanly").
3. **Scientist** - given only the structure, proposes 0-3 real problems in
   any field it might solve, with a concrete check for each. "Nothing
   applies" is a legitimate answer.

## Original batch (8001-8005)

All 5 seeds produced technically precise, correctly cited, honestly hedged
proposals:

| Seed | Proposed applications |
|---|---|
| 8001 | Side-channel crypto leakage (Kocher; Coron & Prouff), structural health monitoring, the 737 MAX MCAS hazard pattern |
| 8002 | Wright-Fisher population genetics, clock-domain-crossing synchronizer MTBF, LCFS-PR batch-arrival queueing |
| 8003 | Bifurcate Killing horizons / Hawking radiation (Steinhauer's BEC experiments 2016, 2019), Euler buckling; rejected persistence diagrams as forced |
| 8004 | CDC metastability (Chaney & Molnar 1973), Chord DHT under churn; rejected Kleppmann's Redlock critique |
| 8005 | Muddy-children epistemic logic, "cache becomes source of truth", shortcut learning (Geirhos et al. 2020) |

**Citation fact-check:** every named citation checked against real sources -
all real, no fabrications, one loose attribution (the Chord churn-threshold
claim belongs more precisely to a related paper). The *application* claims
themselves are not verified.

## Baseline ablation (9001-9005)

`crazyai discover-baseline`: one plain "creative scientist" persona, one
direct call, no world-blend, no persona chain - with the **identical**
honesty instruction.

| Seed | Invented | Proposals |
|---|---|---|
| 9001 | Rotor-router walk with a decaying charge | Discharging arguments; rotor-router cover time (Holroyd-Propp, spot-checked real); honest null on scheduling |
| 9002 | Chip-firing with sign-inverting reflective boundaries | Sandpile order-independence breakdown; lightweight PRNG |
| 9003 | "Ordinal diffusion" by rank order | Comparison-only sorting on miscalibrated sensors; rank-only DeGroot alternative |
| 9004 | "Holonomy ledger" on a trace monoid | Trotter-Suzuki error bounds (self-flagged as likely known); CRDT/OT merge cost |
| 9005 | Mixed-radix counter with upward carries + downward reflection pulses | Avalanche quality as a checksum core; a termination combinatorics question |

**Result: the direct prompt matched the narrative batch on every axis
judged.** Narrative citations carry slightly more bibliographic detail
(authors, years); baseline more often names a theory without a full citation.
Honesty discipline is equal in both, because the instruction producing it is
identical by design.

### One proposal tested for real (seed 9005)

`crazyai-trials/scripts/verify_reflected_odometer_avalanche.py` (pure simulation, no LLM):

| Moduli / ticks | Plain odometer | Reflected | Welch t |
|---|---|---|---|
| `[3,5,7,4,6,9,5,7]`, T=40, n=2000 | 0.0166 | 0.0227 | 5.31 |
| `[2,3,2,3,2,3,2,3]`, T=80, n=1000 | 0.0147 | 0.0470 | 12.51 |

Directionally **confirmed** and robust, but both sit at 1.5-4.7% differing
bits vs the ~50% a real mixing function needs - far too weak for the
proposed "checksum core" use.

## More seeds (8101-8115)

15 more seeds (depth 1-3 mix) on top of the original 5. Quality across all 20
is consistently high and **capped at the same tier**: correct pattern-matching
to published theory (Armitage-Doll carcinogenesis, TCP AIMD, cuckoo hashing
with a stash, the Will Rogers phenomenon, silent synapses, tension-zone hybrid
theory). Nothing qualitatively beyond the first 5. Depth scaled word count,
not idea quality.

## Conclusion

- The discover proposals are genuinely good, but the world-blend + persona
  chain is **not shown to be why** - a direct prompt does as well.
- More seeds and more depth buy more samples of the same quality
  distribution. The bottleneck is the model's reasoning ceiling.

## Operational notes

- ~33% of the first 15 extra seeds hit the transient `claude -p` failure.
  A nohup'd background retry loop survived `kill <pid>` on Windows/git-bash
  and kept spawning children - verify with an OS-level process list before
  assuming a background loop stopped.
- `claude -p` subprocesses inherit the operator's global `CLAUDE.md`
  (seed 8001 mentioned `brainny`). No corrupted artifact found, but it is a
  real context leak into every persona; a `--no-user-config`-style fix is
  still open.
