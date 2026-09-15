import numpy as np

from crazyai.toolkit import native


def test_dimensions(tk):
    ok = tk.call("symbolic_check_dimensions", {"equation": "E = m*c**2", "units": {"E": "kg*m**2/s**2", "m": "kg", "c": "m/s"}})
    bad = tk.call("symbolic_check_dimensions", {"equation": "E = m*c", "units": {"E": "kg*m**2/s**2", "m": "kg", "c": "m/s"}})
    mixed = tk.call("symbolic_check_dimensions", {"equation": "W = m*v**2 - m*v", "units": {"W": "kg*m**2/s**2", "m": "kg", "v": "m/s"}})
    assert ok["consistent"] and not bad["consistent"] and not mixed["consistent"]
    assert "exp" in tk.call("symbolic_check_dimensions", {"equation": "x = exp(t)", "units": {"x": "1", "t": "s"}})["flaw"]


def test_symbolic_basics(tk):
    assert tk.call("symbolic_derive", {"expression": "m*v**2/2", "rule": "differentiate", "argument": "v"})["output"] == "m*v"
    assert tk.call("symbolic_verify_identity", {"lhs": "(x+1)**2", "rhs": "x**2+2*x+1"})["identical"]
    assert tk.call("symbolic_take_limit", {"expression": "sin(x)/x", "variable": "x", "target": "0"})["limit"] == "1"
    assert tk.call("symbolic_solve", {"equation": "x**2 = 4", "variable": "x"})["solutions"] == ["-2", "2"]
    pred = tk.call("symbolic_define_predicate", {"name": "P", "expression": "isprime(n) & (Mod(n,4)==1)", "test_values": [5, 7, 13]})
    assert pred["truth_table"] == {"5": True, "7": False, "13": True}


def test_native_matches_python():
    n = 2000
    ps = native.sieve(n)
    assert int(ps.sum()) == 303 and ps[2] and ps[1997] and not ps[1998]
    ln, pk = native.collatz(30)
    assert int(ln[27]) == 111 and int(pk[27]) == 9232
    cnt = native.even_plus_prime_counts(20, ps)
    # 9 = 7+2 = 5+4 = 3+6 -> 3 ways
    assert int(cnt[9]) == 3
    m, v = native.mc_linear_gaussian([1.0, 2.0], [0.0, 1.0], [1.0, 1.0], 100000, 1)
    assert abs(m - 2.0) < 0.05 and abs(v - 5.0) < 0.15


def test_stats_confounding_shows_up(tk):
    spec = {"n": 2000, "seed": 3, "vars": [
        {"name": "u", "dist": "normal", "params": [0, 1]},
        {"name": "x", "formula": "0.8*u + normal(0, 1)"},
        {"name": "y", "formula": "2*x + 1.5*u + normal(0, 1)"}]}
    biased = tk.call("stats_fit", {"spec": spec, "outcome": "y", "regressors": ["x"]})["coefficients"]["x"]["coef"]
    adjusted = tk.call("stats_fit", {"spec": spec, "outcome": "y", "regressors": ["x", "u"]})["coefficients"]["x"]["coef"]
    assert biased > 2.4 and abs(adjusted - 2.0) < 0.1
    inj = tk.call("stats_inject_confounder", {"spec": spec, "treatment": "x", "outcome": "y", "strength": 2.0})
    assert inj["effect_after"] - 2.0 > 0.5  # true effect is 2; the hidden variable pushes the estimate up
    assert tk.call("stats_check_identifiability", {"spec": spec, "regressors": ["x", "u"]})["identifiable"]
    pw = tk.call("stats_power_analysis", {"effect_size": 0.5, "n_per_group": 64})["power"]
    assert 0.75 < pw < 0.85
    bs = tk.call("stats_bootstrap", {"values": list(np.arange(50.0)), "statistic": "mean"})
    assert bs["ci95"][0] < 24.5 < bs["ci95"][1]


def test_logic(tk):
    assert not tk.call("logic_check_consistency", {"formulas": ["A -> B", "A", "~B"]})["consistent"]
    assert tk.call("logic_check_consistency", {"formulas": ["A -> B", "~A"]})["consistent"]
    assert tk.call("logic_entails", {"premises": ["p -> q", "q -> r", "p"], "conclusion": "r"})["entails"]
    assert not tk.call("logic_entails", {"premises": ["p -> q", "q"], "conclusion": "p"})["entails"]
    chain = [{"id": "1", "claim": "a", "from": []}, {"id": "2", "claim": "b", "from": ["1"]},
             {"id": "3", "claim": "c", "from": ["2"]}, {"id": "4", "claim": "d", "from": ["3"]}]
    tr = tk.call("logic_trace_argument", {"chain": chain})
    assert tr["grounded"] and tr["load_bearing"] == ["1", "2", "3"]
    assert "error" in tk.call("logic_check_consistency", {"formulas": ["A -> B", "A -->> B"]})


def test_narrative(tk):
    events = [{"id": "a", "t": 1, "reveals": [{"fact": "f", "to": ["x"]}]},
              {"id": "b", "t": 2, "requires": [{"fact": "f", "by": "y"}], "causes": ["a"]}]
    out = tk.call("narrative_check_timeline", {"events": events})
    assert out["flaw_count"] == 2
    ws = tk.call("narrative_word_stats", {"text": "One. Two words. \"Three words here.\""})
    assert ws["text"]["sentences"] == 3 and ws["reference"]["sentences"] > 10
    assert tk.call("ground_flaw_count", {"measurements": [out, {"consistent": True}]})["flaw_count"] == 2
