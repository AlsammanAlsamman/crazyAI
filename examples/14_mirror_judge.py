"""AImirror, first test: can a small local model stand in for Claude as the idea-survival judge?

Sends the exact blind judge prompt `crazyai diagnose-return` gave Claude to a local Ollama
model, for every judged kernel, and reports agreement with Claude's verdicts (survival and
fallback). High agreement means the local model can pre-screen prompt variants for free;
low agreement means it can't.

    python examples/14_mirror_judge.py [model] [archive_dir]        (default model: gemma4:26b)
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from crazyai.pipeline.return_path import JUDGE_SCHEMA, JUDGE_SYSTEM, judge_prompt
from crazyai.targets import get_target

ORDER = ["orig", "rp0_current", "rp1_no_known", "rp2_faithful"]


def ask(model: str, system: str, user: str) -> dict:
    body = {"model": model, "stream": False, "think": False, "format": JUDGE_SCHEMA, "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 2048},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user + "\n\nAnswer with a single JSON object matching the schema."}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        text = json.loads(r.read())["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:26b"
    archive = Path(sys.argv[2] if len(sys.argv) > 2 else "archive")
    root = archive / "diagnose_return"
    out_path = root / f"mirror_{model.replace(':', '_').replace('/', '_')}.jsonl"
    done = {(r["variant"], r["source"]) for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines())} \
        if out_path.exists() else set()
    rows = []
    for v in ORDER:
        for d in sorted((root / v).glob("invent_*")):
            code_p = (archive / d.name / "artifact.c") if v == "orig" else (d / "artifact.c")
            if not (d / "judge.json").exists() or not code_p.exists():
                continue
            claude = json.loads((d / "judge.json").read_text(encoding="utf-8"))
            if (v, d.name) not in done:
                ideas = (archive / d.name / "ideas.md").read_text(encoding="utf-8")
                tgt = get_target(d.name.rsplit("_", 1)[1])
                try:
                    local = ask(model, JUDGE_SYSTEM, judge_prompt(ideas, tgt, code_p.read_text(encoding="utf-8")))
                except Exception as exc:  # noqa: BLE001
                    local = {"error": str(exc)[-300:]}
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"variant": v, "source": d.name, "local": local}) + "\n")
                print(v, d.name, local.get("survival"), local.get("fallback"), flush=True)
            rows.append((v, d.name, claude))
    local = {(r["variant"], r["source"]): r["local"] for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines())}
    pairs = [(c, local[(v, s)]) for v, s, c in rows if (v, s) in local and "error" not in local[(v, s)]]
    n = len(pairs)
    surv = sum(c.get("survival") == l.get("survival") for c, l in pairs)
    fb = sum(c.get("fallback") == l.get("fallback") for c, l in pairs)
    order = {"none": 0, "partial": 1, "full": 2}
    near = sum(abs(order.get(c.get("survival"), 0) - order.get(l.get("survival"), 0)) <= 1 for c, l in pairs)
    print(f"\nmodel {model}: n={n}")
    print(f"survival exact agreement {surv}/{n} ({surv / max(n, 1):.0%}), within one level {near}/{n}")
    print(f"fallback agreement {fb}/{n} ({fb / max(n, 1):.0%})")
    print("confusion (claude -> local):", dict(Counter((c.get("survival"), l.get("survival")) for c, l in pairs)))
    # chance baseline: agreement expected if the local model guessed from Claude's own label frequencies
    cs = Counter(c.get("survival") for c, _ in pairs)
    ls = Counter(l.get("survival") for _, l in pairs)
    chance = sum(cs[k] * ls[k] for k in cs) / max(n * n, 1)
    print(f"chance-level survival agreement {chance:.0%}")


if __name__ == "__main__":
    main()
