import json

from crazyai.pipeline.report import compare, load_index, markdown_report, radar_svg
from crazyai.pipeline.run import Run, compute_score
from crazyai.providers import get_provider


def test_full_pipeline_with_mock(tmp_path):
    run = Run(seed=5, generator="statmodel", runs=2, archive_dir=tmp_path)
    summary = run.execute(get_provider("mock"))
    for f in ("seed.json", "mutation.json", "artifact.md", "formal.md", "key.json", "verdicts.json", "score.json", "run.json"):
        assert (run.dir / f).exists(), f
    assert summary["metrics"]["n"] == 2 and summary["metrics"]["acceptance_rate"] == 1.0
    key = json.loads((run.dir / "key.json").read_text())
    assert key["flaw_count"] == 1 and key["rule_id"] == summary["rule_id"]
    # resumable: second execute reuses files and yields the same mutation
    again = Run(seed=5, generator="statmodel", runs=2, archive_dir=tmp_path).execute(get_provider("mock"))
    assert again["rule_id"] == summary["rule_id"] and again["operator"] == summary["operator"]

    det = Run(seed=6, generator="debate", runs=2, archive_dir=tmp_path).execute(get_provider("mock", detect=True))
    assert det["metrics"]["detection_rate"] == 1.0 and det["metrics"]["acceptance_rate"] == 0.0

    rows = load_index(tmp_path)
    assert len(rows) == 2
    md = markdown_report(rows)
    assert "## By generator" in md and "statmodel" in md
    assert radar_svg(rows[0]["metrics"]).startswith("<svg")
    d = compare(tmp_path / "run_5_statmodel", tmp_path / "run_6_debate")
    assert d["metrics"]["detection_rate"]["delta"] == 1.0


def test_compute_score_flags_discovery():
    key = {"rigor": 9, "novelty": 8, "cost_of_possibility_band": "low", "flaw_count": 1}
    judged = [{"detection_level": "none", "accepts_conclusion": True, "confidence": 0.9, "false_flaws": 0, "hedged": False},
              {"detection_level": "exact", "accepts_conclusion": False, "confidence": 0.8, "false_flaws": 1, "hedged": False}]
    s = compute_score(key, judged)
    assert s["detection_rate"] == 0.5 and s["acceptance_rate"] == 0.5 and s["false_flaw_rate"] == 0.5
    assert s["confidence_when_wrong"] == 0.9 and s["flagged_for_review"]
    assert compute_score({**key, "flaw_count": 2}, judged)["discovery_value"] == 0.0
