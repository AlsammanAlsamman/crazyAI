"""07 - `crazyai invent`, offline.

Blend the imagination corpus with every model and compare them on the
imagination scale; then run the whole invent loop with the mock provider on the
matrix-multiplication target: world -> the native's paragraph -> the engineer's
C kernel -> measured against a cache-blocked loop.
"""

from pathlib import Path
import shutil

from crazyai.pipeline.invent import Invent
from crazyai.providers import get_provider
from crazyai.toolkit.registry import build_toolkit

tk = build_toolkit(42)
cmp = tk.call("blend_compare", {})
print("blend models on seed 42, ranked on the imagination scale:")
for r in cmp["ranking"]:
    print(f"  {r['model']:8s} score={r['score']:.3f} (imagination {r['imagination']:.2f}, readable {r['readable']:.2f})")
print(f"\nbest world ({cmp['best']}):\n{cmp['texts'][cmp['best']]}\n")

arch = Path("archive/example07")
shutil.rmtree(arch, ignore_errors=True)
run = Invent(seed=42, target="matmul", blend="anneal", harvest=3, archive_dir=arch)
s = run.execute(get_provider("mock"))
print("\nthe native said:\n" + (run.dir / "ideas.md").read_text()[:600] + "...\n")
print(f"measured: status={s['status']} value={s['value']} (speedup over a blocked loop x exactness), "
      f"predicted {s['prediction']}, calibration {s['calibration']}, discovery {s['discovery']}")
print(f"run folder: {run.dir}")
