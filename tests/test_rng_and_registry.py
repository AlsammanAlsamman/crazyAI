from crazyai.rng import RunRNG
from crazyai.toolkit.registry import build_toolkit


def test_same_seed_same_draws():
    a, b = RunRNG(9), RunRNG(9)
    for _ in range(20):
        assert a.integer("i", 0, 1000) == b.integer("i", 0, 1000)
    assert a.shuffle("s", list(range(10))) == b.shuffle("s", list(range(10)))
    assert a.log == b.log


def test_registry_has_both_kinds_and_schemas(tk):
    specs = tk.all_specs()
    kinds = {s.kind for s in specs.values()}
    assert kinds == {"invent", "measure"}
    for s in specs.values():
        assert s.input_schema["type"] == "object"
        assert "rng" not in s.input_schema["properties"]
        assert "ctx" not in s.input_schema["properties"]
        assert s.description
    api = tk.to_api_tools(tk.names(families=["chaos"]))
    assert all({"name", "description", "input_schema"} <= set(t) for t in api)


def test_unknown_tool_and_bad_args_do_not_raise(tk):
    assert "error" in tk.call("nope_tool", {})
    assert "error" in tk.call("symbolic_take_limit", {"expression": "x", "variable": "x"})  # missing target
    assert len(tk.calls) == 2 and not tk.calls[0]["ok"]
