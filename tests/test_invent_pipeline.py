import json
import shutil

import pytest

from crazyai import imagination as im
from crazyai.pipeline.invent import Invent, load_invent_index
from crazyai.providers import get_provider
from crazyai.toolkit.invent.blend import MODELS
from crazyai.toolkit.measure.imagination import score_text
from crazyai.toolkit.registry import build_toolkit


def test_corpus_loads_three_worlds():
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
