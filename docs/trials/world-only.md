# World-only: does "material over persona" transfer to real targets?

*Run 2026-09-21 · pipeline `crazyai invent-world-only` · 3 targets x 5 seeds = 15 real trials*

[← back to trials index](README.md)

## Background

The imagination-effect tuning campaign (a divergent-thinking task, no target)
found that the **corpus-blended material** carries the measured effect, not the
alien-persona swap: `world_only` (blended world, no persona) was statistically
indistinguishable from the full persona + world structure.

This tests whether that transfers to `invent`'s real job, on the 3 targets
where the full persona + material pipeline lost decisively to Baseline.

## Design

A clean one-variable ablation against Baseline: Baseline's exact persona,
task and instructions (no assumption-pinning), **plus** a freshly blended
world offered as optional inspiration. No persona swap, no requirement to
interpret it literally.

## Results

| Target | Baseline mean (wins/5) | World-only mean (wins/5) | Delta |
|---|---|---|---|
| alignment | 2.345x (3/5) | 2.189x (2/5) | -0.156 |
| hash | 14.503x (5/5) | **15.473x (5/5)** | **+0.970** |
| dijkstra | 1.284x (5/5) | 1.029x (2/5) | -0.255 |

- **hash improved** and got more consistent: world-only's worst seed was
  15.16x vs baseline's 9.77x outlier.
- **alignment** was a wash, trending slightly worse.
- **dijkstra** got clearly worse: 5/5 -> 2/5 wins, 3 of 5 seeds below 1.0x.
  Expected, since its gap is a specific missing capability (see
  [protocol v3](protocol-v3.md)), which extra material doesn't address.

## Conclusion

Mixed, not uniform. "Material over persona" is real on the creativity task
and transfers **partially** (one real win on hash), but does not generalize
into "simplify the pipeline and expect better results everywhere".

47 -> 49 tests passing (mock end-to-end, prompt-refactor parity check).
`direct_prompt` now wraps the shared `_direct_task_block` helper, behaviour
unchanged.
