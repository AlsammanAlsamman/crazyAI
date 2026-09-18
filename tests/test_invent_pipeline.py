import json
import shutil

import pytest

from crazyai import imagination as im
from crazyai.pipeline.invent import Invent, load_invent_index
from crazyai.pipeline.invent_bias import arm_weights
from crazyai.providers import get_provider
from crazyai.toolkit.invent.blend import ALL_MODELS, MODELS
from crazyai.toolkit.measure.imagination import score_text
from crazyai.toolkit.registry import build_toolkit


def test_corpus_loads_all_worlds():
    kinds = im.by_kind(im.corpus(include_harvest=False))
    assert all(len(kinds[k]) >= 20 for k in im.KINDS)
    assert im.is_content("wardrobe") and not im.is_content("the")
    assert im.slot_class("the") == "N" and im.slot_class("is") == "V" and im.slot_class("wardrobe") == "O"


def test_imagination_scale_orders_texts():
    c = im.corpus(include_harvest=False)
    verbatim = score_text(c[0].text)
    salad = score_text("watch river lion wardrobe melt ant staircase castle boulder pheromone scythe mirror queen "
                       "crocodile lamp ledger cloud orchard tollbooth turtle balloon librarian hexagon.")
    assert verbatim["originality"] == 0 and verbatim["score"] < 0.01
    assert salad["imagination"] > 0.7 and salad["coherence"] == 0 and salad["score"] < 0.35
    repeated = score_text("watch river lion wardrobe melt ant staircase " * 8 + ".")
    assert repeated["originality"] < 0.2                          # repetition is not imagination


def test_blend_models_are_seeded_and_scored():
    a = build_toolkit(11).call("blend_compare", {})
    b = build_toolkit(11).call("blend_compare", {})
    assert a["texts"] == b["texts"]                              # same seed, same worlds
    assert set(r["model"] for r in a["ranking"]) == set(MODELS)
    best = a["ranking"][0]
    assert best["score"] > 0.5 and best["coherence"] > 0.5      # pushed far, still readable
    assert a["texts"][best["model"]] != build_toolkit(12).call("blend_compare", {})["texts"][best["model"]]


def test_kernel_bench_checks_and_times():
    tk = build_toolkit(1)
    ex = tk.call("kernel_contract", {})["example"]
    r = tk.call("kernel_bench", {"source": ex, "sizes": [64, 128], "budget": 0.05})
    if "error" in r and "gcc not found" in r["error"]:
        pytest.skip("no C compiler")
    assert r["status"] == "exact" and r["speedup_vs_naive"] > 1
    assert tk.call("kernel_bench", {"source": ex.replace("a * B", "a + B"), "sizes": [64], "budget": 0.02})["status"] == "WRONG"
    assert tk.call("kernel_bench", {"source": "int x = ;", "sizes": [64]})["status"] == "COMPILE_ERROR"


def test_invent_pipeline_with_mock(tmp_path):
    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    run = Invent(seed=42, target="matmul", blend="graft", harvest=2, archive_dir=tmp_path)
    s = run.execute(get_provider("mock"))
    for f in ("seed.json", "harvest.yaml", "world.md", "world.json", "ideas.md", "artifact.md", "artifact.c", "measure.json", "run.json"):
        assert (run.dir / f).exists(), f
    assert (tmp_path / "imagination" / "harvest_42.yaml").exists()
    assert s["status"] == "exact" and s["value"] > 1 and s["prediction"] == 1.5 and 0 < s["calibration"] <= 1
    m = json.loads((run.dir / "measure.json").read_text())
    assert m["measurement"]["largest_n"] == 512 and m["discovery"] > 0
    # resumable and indexed
    again = Invent(seed=42, target="matmul", blend="graft", harvest=2, archive_dir=tmp_path).execute(get_provider("mock"))
    assert again["value"] == s["value"]
    assert len(load_invent_index(tmp_path)) == 1
    # an unmeasured target still runs end to end
    p = Invent(seed=43, target="physics", blend="nest", archive_dir=tmp_path).execute(get_provider("mock"))
    assert p["status"] == "UNMEASURED" and p["value"] is None


def test_bend_prompt_carries_the_kernel_contract():
    # the engineer must never have to guess the signature (a real run once crashed on argument order)
    from crazyai.pipeline.invent_prompts import bend_prompt
    from crazyai.targets import get_target
    p = bend_prompt("SEED: x", get_target("matmul"), [])
    assert "void kernel(int n, const double *A, const double *B, double *C)" in p and "do not guess" in p
    assert "kernel(" not in bend_prompt("SEED: x", get_target("physics"), [])


_BYTE_ASSUMPTION = "numbers are IEEE doubles and multiply is the primitive"


def test_assumption_steering_is_exact_and_opt_in():
    from crazyai.pipeline.invent_prompts import bend_prompt, immerse_prompt
    from crazyai.targets import get_target
    tgt = get_target("matmul")
    hint = tgt.assumption_hints[_BYTE_ASSUMPTION]

    unhinted = immerse_prompt("world text", tgt, 2)
    hinted = immerse_prompt("world text", tgt, 2, hint)
    assert hint not in unhinted and hint in hinted

    unfocused = bend_prompt("SEED: x", tgt, [])
    focused = bend_prompt("SEED: x", tgt, [], _BYTE_ASSUMPTION)
    assert "preferring one that breaks this assumption" not in unfocused
    assert "preferring one that breaks this assumption" in focused and _BYTE_ASSUMPTION in focused


def test_assumption_pin_resolves_and_rejects(tmp_path):
    exact = Invent(seed=1, target="matmul", assumption=_BYTE_ASSUMPTION, archive_dir=tmp_path, force=True, log=None)
    by_index = Invent(seed=2, target="matmul", assumption="4", archive_dir=tmp_path, force=True, log=None)
    unset = Invent(seed=3, target="matmul", archive_dir=tmp_path, force=True, log=None)
    assert exact._pinned_assumption == _BYTE_ASSUMPTION
    assert by_index._pinned_assumption == _BYTE_ASSUMPTION
    assert unset._pinned_assumption == ""
    with pytest.raises(ValueError):
        Invent(seed=4, target="matmul", assumption="not one of the eight", archive_dir=tmp_path, force=True, log=None)


def test_assumption_pin_reaches_seed_json_deterministically(tmp_path):
    for seed in (10, 11, 12):
        s = Invent(seed=seed, target="matmul", assumption=4, archive_dir=tmp_path, force=True, log=None).step_seed()
        assert s["assumption_focus"] == _BYTE_ASSUMPTION


def _row(seed, blend_model, discovery, target="matmul"):
    return {"seed": seed, "target": target, "blend_model": blend_model, "depth": 2,
           "assumption_focus": "n^3 multiplications are needed", "discovery": discovery}


def test_arm_weights_needs_a_minimum_sample_and_then_favours_the_winner():
    rows = [_row(1, "anneal", 9.0), _row(2, "anneal", 11.0), _row(3, "cutup", 0.0)]
    assert arm_weights(rows, "blend_model", MODELS, min_samples=10) is None          # too little history: unbiased
    w = arm_weights(rows, "blend_model", MODELS, min_samples=3)
    weights = dict(zip(MODELS, w))
    assert weights["anneal"] > weights["cutup"]                                     # the actual winner is favoured
    assert weights["markov"] > 0                                                    # an unseen arm stays reachable


def test_bias_from_history_is_off_by_default_and_deterministic_when_on(tmp_path):
    (tmp_path / "invent_index.jsonl").write_text(
        "\n".join(json.dumps(_row(i, "anneal", 10.0)) for i in range(6)) + "\n", encoding="utf-8")
    unbiased = Invent(seed=1, target="matmul", archive_dir=tmp_path, force=True, log=None).step_seed()
    assert unbiased["bias"] == {"enabled": False, "min_samples": 20, "samples_seen": 0, "active": False}
    a = Invent(seed=7, target="matmul", archive_dir=tmp_path, force=True, log=None,
              bias_from_history=True, bias_min_samples=5).step_seed()
    b = Invent(seed=7, target="matmul", archive_dir=tmp_path, force=True, log=None,
              bias_from_history=True, bias_min_samples=5).step_seed()
    assert a["bias"]["active"] is True and a["bias"]["samples_seen"] == 6
    assert a == b                                                                    # same seed + frozen archive


def test_evolve_corpus_promotes_only_new_bests_and_they_become_drawable(tmp_path):
    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    provider = get_provider("mock")
    outcomes = []
    for seed in (1, 2, 3):
        run = Invent(seed=seed, target="matmul", archive_dir=tmp_path, force=True, log=None, evolve_corpus=True)
        outcomes.append(run.execute(provider))
    assert "promoted" in outcomes[0] and outcomes[0]["promoted"] is True             # first run always beats 0.0
    discoveries = [o["discovery"] for o in outcomes]
    for i in range(1, 3):
        assert outcomes[i]["promoted"] == (discoveries[i] > max(discoveries[:i]))
    prom = im.promoted(tmp_path)
    assert len(prom) == sum(o["promoted"] for o in outcomes)
    if prom:
        seen_ids = set()
        for i in range(20):
            probe = Invent(seed=2000 + i, target="matmul", archive_dir=tmp_path, force=True, log=None,
                           evolve_corpus=True)
            world = probe.step_world(probe.step_seed())
            seen_ids |= {f["id"] for f in world["fragments"]}
        assert seen_ids & {f.id for f in prom}, "a promoted fragment should be drawable once evolve_corpus is on"


def test_remix_is_selectable_but_not_in_the_default_pool():
    assert "remix" in ALL_MODELS and "remix" not in MODELS
    r = build_toolkit(1).call("blend_remix", {"k": 4, "steps": 5})
    assert r["model"] == "remix" and r["text"]
