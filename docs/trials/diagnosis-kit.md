# Diagnosis kit: the return-path probe (where the idea is lost, and can a prompt stop it?)

*Run 2026-09-29 · `crazyai diagnose-return` (`crazyai/pipeline/return_path.py`) · report
`examples/12_diagnose_report.py` · archive `archive/diagnose_return/` · Claude Code 2.1.274*

[← back to trials index](README.md)

## The question

The [retroactive immersion check](retro-immersion.md) showed the native stays in its world and
located the loss at the translate ("bend") step. This probe asks two things:

1. **How often does the engineer fall back** to the textbook method instead of building the
   native's mechanism?
2. **Can the translate prompt stop it?**

## Design

- **Only the translate step varies.** 15 archived native texts (`ideas.md`) are held fixed:
  alignment 7011 and 7022–7025, hash 7221–7225, dijkstra 7521–7525, with the same pinned
  assumption as the original run.
- **Three prompt versions, each adding one change, all run the same day:**

| Version | Change |
|---|---|
| `rp0_current` | today's `bend_prompt`, unchanged |
| `rp1_no_known` | rp0 minus step 4's "let your mechanism arrive at that [known] technique" sentence |
| `rp2_faithful` | rp1 plus "build THAT seed's mechanism; don't switch; improve the implementation rather than replace it" |

- `orig` is each source's archived kernel, from an earlier date and an older prompt version.
- **Blind judge.** A separate Claude call sees only the native's text and the code. It never
  sees the prompt version, the engineer's explanation or the speed. It rates survival
  (full / partial / none) and fallback (yes / no). Two verdicts were checked by hand and hold up.
- **Fair timing.** All 60 kernels were re-timed 5 times each, interleaved, with the machine idle.
- **Extra check.** Every alignment kernel was also checked on 48 fresh random inputs, because
  the harness uses one fixed input per length. Every kernel the benchmark called exact was also
  exact on the fresh inputs.

## Results

Fresh mean speedup (exact / wins / n):

| Version | alignment | dijkstra | hash |
|---|---|---|---|
| orig (archived) | 5.05x (5/5/5) | 1.01x (5/1/5) | 6.84x (5/2/5) |
| rp0_current | 11.81x (5/5/5) | 0.85x (4/2/5)\* | 11.06x (5/5/5) |
| rp1_no_known | 7.45x (4/4/5) | **1.70x** (5/4/5) | 12.03x (5/5/5) |
| rp2_faithful | **13.21x** (5/5/5) | 1.12x (5/3/5) | **14.95x** (5/5/5) |

\*One rp0 dijkstra binary was deleted by the machine's endpoint security (CrowdStrike Falcon)
and scored 0.

Idea survival and fallback (blind judge):

| Version | full | partial | none | fell back |
|---|---|---|---|---|
| orig | 6 | 5 | 4 | 12/15 |
| rp0_current | 8 | 6 | 1 | 10/15 |
| rp1_no_known | 7 | 7 | 1 | 10/15 |
| rp2_faithful | 9 | 4 | 2 | 8/15 |

Paired tests over the same 15 sources:

| Step | Speedup | Survival | Fallback |
|---|---|---|---|
| rp0 → rp1 (remove "arrive at known technique") | 7.91 → 7.06x, p = 0.21 | p = 0.74 | 10 → 10 |
| rp1 → rp2 (add "build it faithfully") | 7.06 → 9.76x, p = 0.49 | p = 0.71 | 10 → 8, p = 0.63 |
| rp0 → rp2 (both) | 7.91 → 9.76x, p = 0.30 | p = 1.00 | 10 → 8, p = 0.63 |

**Does keeping the idea make the kernel faster?** No. The within-target correlation of
survival with speed is rho = −0.03 (permutation p = 0.82). Not falling back trends slightly
positive, rho = +0.18, but not significantly (p = 0.17).

## What it means

1. **Fallback is the norm, not the exception.** 8–12 of 15 kernels fall back under every prompt
   version. The engineer usually keeps some of the native's structure (the "partial" verdicts)
   and puts it on top of a known technique such as AVX2 banded alignment, xxHash-style lanes,
   or bucketed Dijkstra.
2. **The prompt changes tested don't measurably stop it.**
   - Removing the "arrive at the known technique" sentence changed nothing (10 → 10).
   - Adding "build it faithfully" moved fallback from 10 to 8, well within chance.
   - The speed differences between prompt versions aren't significant, and they point different
     ways by target: rp2 is best on alignment and hash, rp1 on dijkstra. That's the same "which
     third depends on the target" pattern as every earlier condition.
3. **The fallback doesn't cost speed.** Surviving ideas are no faster than fallbacks. The
   engineer falls back to strong known techniques, and those win. So losing the narrative hurts
   *novelty*, not *performance*. A kit aimed at speed shouldn't spend effort preventing fallback.
4. **The biggest effect is the date, not the prompt.** With the same native texts and a nearly
   identical prompt, today's run roughly doubles the archived kernels on alignment (5.05 →
   11.81x) and hash (6.84 → 11.06x). It isn't a clean model control: the archived runs also
   predate step 5, protocol v3's regime rule, and for 7011 also step 4. But it's strong evidence
   that the model changed between 09-19 and 09-29.
   - **What this means for [disguise-all](disguise-all.md):** its alignment result (12.72x) is
     about what today's model does with the *ordinary* prompt (rp0 11.81x, rp2 13.21x), so the
     alignment gain is mostly model drift. Its hash (29.8x quality-gated) and dijkstra (2.00x)
     results are above anything this probe produced, so those gains are more likely to be real
     effects of building all three and letting the benchmark pick.

## AImirror, first test: a local model as the judge

`examples/14_mirror_judge.py` sent the same 60 blind judge prompts to a local `gemma4:26b`,
run through Ollama on the RTX 4000 Ada. It needs `think: false`: with thinking on, it spent its
whole token budget reasoning and never answered. It takes about 97 seconds per call.

| | Agreement | Chance | Cohen's kappa |
|---|---|---|---|
| survival | 38% | 28% | 0.14 |
| fallback | 70% | 64% | 0.16 |

The local model is far stricter. It called 56 of 60 kernels fallback and gave only 7 "full"
verdicts where Claude gave 30. Sometimes that's defensible: on 7011 it argued the native's
"folding along the diagonal" simply *is* the textbook diagonal sweep. Either way, agreement is
only slight (kappa below 0.2), so **this model can't yet stand in for Claude as the judge.**
Next options: a Qwen model, a few worked examples in the prompt, or checking both judges
against human verdicts on a sample.

## Conclusion

The narrative is lost at the translate step, and it's lost most of the time: about two in three
kernels fall back to a known technique. Simple prompt instructions don't stop this measurably at
n = 15. More importantly, it doesn't matter for speed, because the fallbacks are the fast
kernels. The largest effect seen here is the model itself changing between runs a week apart.
For speed, the productive levers are selection (letting the benchmark choose) and the model. For
*novelty*, preventing fallback matters, and that needs a different target to optimise than
speedup.

## Possible next steps

- Record the exact model id in every run (the `claudecode` provider doesn't yet).
- If novelty is the goal, score it directly: judge-rated distance from the known way, rewarded
  alongside speed.
- A clean model control for disguise-all on hash and dijkstra: today's single-pick translate on
  the same disguises.
- AImirror: try a Qwen model and few-shot examples, and validate both judges against a small set
  of human verdicts.

---

## Part 2 (2026-09-30 to 10-01): four ways to stop the fallback

Same 15 native texts, all run on a pinned model (`claude-opus-5`, effort high) against a
same-day control, then re-timed 5 times interleaved (`report_antifallback.txt`).

| Arm | Idea |
|---|---|
| `c0_today` | today's normal translate prompt (control) |
| `i1_hidden` | 1. hide the textbook: no known way, no assumption list, no example kernel |
| `i2_recipe` | 2. the native writes a numbered in-world recipe; the engineer codes it step by step |
| `i3_gate` | 3. the blind judge checks each kernel; on a fallback, feed the verdict back and retry (up to 2 times) |
| offline | 4. reward novelty: per story, the fastest kernel that did *not* fall back |

`i3_gate` completed 11 of 15: its last 4 dijkstra stories hit the usage limit. Its numbers
below are on those 11, paired with the control on the same 11.

| Arm | Mean speedup vs control (paired) | Fell back | Notes |
|---|---|---|---|
| control | 13.69x (18.07x on the gate's 11) | 12/15 (9/11) | |
| 1. hidden | 7.07x, p = 0.04 | 9/15, n.s. | speed roughly halves on all targets |
| 2. recipe | 2.84x, p = 0.002 | 11/15, n.s. | most "full" survival (10/15), but hash collapses to 0.51x |
| 3. gate | 8.79x vs 18.07x, p = 0.014 | **5/11** (4 removed, 0 added, p = 0.125) | a retry rescued 4 of 9 fallbacks; 2 of 11 lost correctness |

**Idea 4 (novelty-gated selection)** over the four arms' kernels per story: the fastest kernel
averages 14.60x; the fastest *non-fallback* kernel averages 6.51x. Only 8 of 15 stories produced
any exact non-fallback kernel. By target:
- **Alignment:** only 1 of 5 stories did (18.5x).
- **Dijkstra:** 3 of 5 did, at up to 2.09x.
- **Hash:** 4 of 5 did. Two of them are fast *and* pass the SMHasher-style quality test:
  - gate 7224, 29.2x: a wide 1024-bit ARX state with Speck-style rounds and BLAKE2b-style mixing;
  - control 7225, 31.0x: chained hardware-CRC lanes with an avalanche finalizer.
  
  Two slower non-fallback hashes fail the quality test (i1 7223, gate 7223). One binary was
  blocked by endpoint security.

**Day-to-day noise is large.** The same prompt on the same texts gave hash 11.06x on 09-29 and
24.40x on 09-30. Differences between arms smaller than about 2x shouldn't be trusted at n = 15.

### What it means

1. **Only the gate reliably cuts fallback**, and it costs about half the speed and some
   correctness. Hiding the textbook and the recipe don't measurably reduce fallback, and both
   cost speed; the recipe badly so.
2. **Whether keeping the idea is viable depends on the target's design space.**
   - **Alignment:** every fast exact kernel is a known technique, so forcing the native's idea
     only makes it slower.
   - **Hash:** the design space is wide. The native's mechanism can be kept and still be fast,
     and the two good kernels show it. Even so, they are assembled from known primitives (ARX,
     CRC, BLAKE2-style rounds), not a new hash family.
3. **A path worth following:** gate plus selection, on wide-design-space targets. The gate keeps
   the idea; generating several gated candidates and letting the benchmark pick recovers speed.
   It needs a strong correctness/quality gate, since the gate arm produced two broken hashes.
