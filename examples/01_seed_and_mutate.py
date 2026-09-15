"""Example 1 - controlled chaos, offline.

Draw a seed, apply every operator to it, and show that the same seed gives the
same draws every time. No API key needed.
"""

from crazyai.config import OPERATORS
from crazyai.toolkit.registry import build_toolkit


def main() -> None:
    tk = build_toolkit(seed=7)
    seed = tk.call("chaos_draw_seed")
    rule = seed["rule"]
    print(f"seed 7 -> {seed['domain']} / {seed['concept']} / {rule['id']}")
    print(f"  original: {rule['statement']}\n")

    for op in OPERATORS:
        args = {"rule_id": rule["id"], "operator": op}
        if op == "TRANSPOSE":
            args["target_domain"] = tk.call("chaos_draw_analogy_pair")["to_domain"]
        if op == "COMPOSE":
            args["other_rule_id"] = "phys.thermo.second"
        out = tk.call("mutate_apply_operator", args)
        print(f"  {op:11s} {out['mutated']['statement'][:110]}")

    depth = tk.call("chaos_draw_depth")["depth"]
    print(f"\n  depth drawn: {depth}")
    print(f"  what_if affects {tk.call('unconventional_what_if', {'rule_id': rule['id']})['count']} downstream rules")

    # reproducibility: a fresh toolkit with the same seed makes the same draws
    again = build_toolkit(seed=7).call("chaos_draw_seed")
    assert again["rule"]["id"] == rule["id"], "same seed must give the same draw"
    print("\n  reproducible: same seed -> same draw  (rng log entries:", len(tk.rng.log), ")")


if __name__ == "__main__":
    main()
