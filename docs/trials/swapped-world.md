# Swapped-world test: is the narrative's content causal or decorative?

*Run 2026-09-20 · no new pipeline code · archives `archive/invent_7541..7544_alignment/`*

[← back to trials index](README.md)

## Hypothesis

Maybe the blended world never *selects* the technique; the model already has
the known-good answer in reach and the narrative is packaging. Motivating
evidence: alignment's narrative run and its direct-prompt baseline converged
on the identical technique (antidiagonal wavefront DP).

## Design

`Invent.step_world` reuses an existing `world.json` instead of drawing a new
blend. 4 new alignment seeds (7541-7544, assumptions 1-4, matching
protocol-v2's 7022-7025) were pre-seeded with a `world.json` copied verbatim
from an **unrelated** target's run - abstract poetry (matmul), nautical decay
(hash), a chessboard/family scene (dijkstra), a noir elegy (quantum). Same
assumption, same current `bend_prompt`; only the world text differs.

## Results

| Assumption | Protocol v2 (matched world) | Swapped (foreign world) | Technique |
|---|---|---|---|
| 1 | 7022: **4.85x** | 7541: **2.184x** | same - antidiagonal wavefront |
| 2 | 7023: **4.701x** | 7542: **0.598x** (regression) | different - banded/radius-limited, didn't pay off |
| 3 | 7024: **4.549x** | 7543: **4.488x** | same - antidiagonal reordering |
| 4 | 7025: **3.633x** | 7544: **0.118x** (severe regression) | same family (antidiagonal + SIMD), execution failed |

Mean: **4.43x (4/4 wins) -> 1.85x (2/4 wins)**. All kernels exact.

## Conclusion

- **Technique selection is robust to the story** - 3 of 4 seeds still reached
  for antidiagonal reordering from totally foreign material.
- **Execution quality is not** - performance dropped sharply, with two real
  regressions.

The narrative's contribution looks less like *choosing the idea* and more like
*scaffolding for executing it cleanly* - a narrower claim than either
"decorative" or "the narrative selects the idea".
