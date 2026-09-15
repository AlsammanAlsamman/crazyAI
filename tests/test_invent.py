import pytest

from crazyai.config import OPERATORS
from crazyai.domains import load_domains


def test_draw_seed_is_reproducible_and_respects_domain(tk):
    from crazyai.toolkit.registry import build_toolkit

    a = tk.call("chaos_draw_seed", {"domain": "physics"})
    b = build_toolkit(123).call("chaos_draw_seed", {"domain": "physics"})
    assert a["rule"]["id"] == b["rule"]["id"] and a["domain"] == "physics"
    assert "error" in tk.call("chaos_draw_seed", {"domain": "astrology"})


@pytest.mark.parametrize("op", OPERATORS)
def test_every_operator_on_every_rule(tk, op):
    for d in load_domains().values():
        for r in d.rules():
            args = {"rule_id": r.id, "operator": op}
            if op == "TRANSPOSE":
                args["target_domain"] = "narrative" if d.name != "narrative" else "physics"
            if op == "COMPOSE":
                args["other_rule_id"] = "phys.thermo.second"
            out = tk.call("mutate_apply_operator", args)
            assert "error" not in out, (r.id, op, out)
            assert out["mutated"]["statement"] and out["mutated"]["statement"] != r.statement or op == "SUBSTITUTE"


def test_transform_round_trip_exact(tk):
    structure = {"nodes": [{"id": str(i), "x": i, "y": (i * 7) % 5, "weight": 1 + i / 10, "group": i % 3} for i in range(1, 13)]}
    img = tk.call("transform_structure_to_image_description", {"structure": structure})
    score = tk.call("transform_image_description_to_music", {"image": img})
    back = tk.call("transform_image_description_to_structure",
                   {"image": tk.call("transform_music_to_image_description", {"score": score})})
    cmp = tk.call("symbolic_compare_structures", {"a": structure, "b": back, "tolerance": 1e-3})
    assert cmp["identical"], cmp
    rev = tk.call("transform_reverse", {"score": score})
    assert [n["id"] for n in rev["notes"]] == [n["id"] for n in score["notes"]][::-1]


def test_disguise_and_unconventional(tk):
    tells = tk.call("disguise_formalise_tone", {"text": "I think this is obviously impossible! Isn't it?"})
    assert {"first_person", "intensifier", "exclamation", "flaw_words"} <= set(tells["tells"])
    assert tk.call("disguise_bury", {"statement": "x", "depth": 3})["depth"] == 3
    assert "not" in tk.call("unconventional_invert", {"assumption": "the system is isolated"})["inverted"] or \
        "exchanges" in tk.call("unconventional_invert", {"assumption": "the system is isolated"})["inverted"]
    lim = tk.call("unconventional_extreme_case", {"expression": "1/(1+x)", "parameter": "x"})["limits"]
    assert lim["to_plus_inf"] == "0" and lim["to_zero"] == "1"
