"""Example 5 - the whole eight-step pipeline, offline, with the mock provider.

Produces a complete run folder (seed, mutation, artifact, formalisation, key,
verdicts, score, run summary) without calling any API. The artifact is a
template - this example is about the machinery, not the content.
"""

import json
import tempfile
from pathlib import Path

from crazyai.pipeline.report import load_index, markdown_report, radar_svg
from crazyai.pipeline.run import Run
from crazyai.providers import get_provider


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp)
        for gen, detect in (("formula", False), ("debate", True)):
            provider = get_provider("mock", detect=detect)
            run = Run(seed=42, generator=gen, runs=3, archive_dir=archive)
            summary = run.execute(provider)
            print(f"\n{gen}: files -> {sorted(p.name for p in run.dir.iterdir())}")
            print("metrics:", json.dumps(summary["metrics"]))

        rows = load_index(archive)
        print("\n" + markdown_report(rows))
        svg = radar_svg(rows[0]["metrics"], title="run 42 formula")
        print(f"radar svg: {len(svg)} bytes")


if __name__ == "__main__":
    main()
