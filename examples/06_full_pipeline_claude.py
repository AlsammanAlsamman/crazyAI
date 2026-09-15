"""Example 6 - a real run with Claude (needs credentials).

    export ANTHROPIC_API_KEY=...      # or `ant auth login`
    .venv/bin/python examples/06_full_pipeline_claude.py [seed] [generator]

Runs one seed through all eight steps with claude-opus-5, writes the run
folder under archive/, and prints the score. Server-side refusal fallbacks
are on by default (see crazyai/providers/anthropic.py).
"""

import json
import sys

from crazyai.pipeline.run import Run
from crazyai.providers import get_provider


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    generator = sys.argv[2] if len(sys.argv) > 2 else "formula"
    provider = get_provider("anthropic")  # model=claude-opus-5, effort=high, adaptive thinking, streaming
    run = Run(seed=seed, generator=generator, runs=3)
    summary = run.execute(provider)
    print(json.dumps(summary["metrics"], indent=2))
    print(f"\nartifact: {run.dir / 'artifact.md'}\nkey:      {run.dir / 'key.json'}")


if __name__ == "__main__":
    main()
