"""Example 2 - a world where primes are not quite prime, offline.

The mutated rule: primality is partial; an integer is prime *to the degree*
that it (a) is an even number plus a prime, (b) is a "fake even", (c) is a
variation of pi. Question: in such a world, how would you predict primality?

Everything numeric here is computed - the sieve and the even+prime counter run
in C++ when the kernel is built - and a real predictor is fitted with real
accuracy numbers. Then the measure tools say what breaks: unique factorisation.
"""

import numpy as np

from crazyai.toolkit import native
from crazyai.toolkit.registry import build_toolkit
from crazyai.worlds.partial_primes import features, label_world


def main() -> None:
    tk = build_toolkit(seed=11)
    limit = 20000
    w = label_world(limit, pi_window=5000)
    n = w["n"][2:]
    print(f"backend: {native.backend()}   integers labelled: {len(n)}")
    print("partial-primality histogram (0..3):", np.bincount(w["partial_primality"][2:], minlength=4).tolist())
    print("ordinary primes:", int(w["is_prime"][2:].sum()))

    # 1. the three conditions as formal predicates, checked on small values
    for name, expr in [("P_fake_even", "(Mod(n,2)==1) & (Mod(n*n,8)==1)"), ("P_prime", "isprime(n)")]:
        out = tk.call("symbolic_define_predicate", {"name": name, "expression": expr, "test_values": list(range(2, 14))})
        print(f"{name}: {out['true_count']}/{out['tested']} true on 2..13")

    # 2. a real predictor of 'fully partial-prime' (score == 3) from cheap features
    X = features(n)
    y = (w["partial_primality"][2:] == 3).astype(float)
    cols = {**{k: v.tolist() for k, v in X.items()}, "full": y.tolist()}
    fit = tk.call("stats_fit_table", {"columns": cols, "outcome": "full",
                                      "regressors": ["mod2", "mod4_is1", "digit_sum", "log_n"], "model": "logit"})
    print(f"\nlogit predictor of full partial-primality: accuracy={fit['accuracy']}  base rate={fit['base_rate']}")
    for k, c in fit["coefficients"].items():
        print(f"   {k:10s} coef={c['coef']:+.3f}  p={c['p']:.4f}")

    # 3. the same predictor on ORDINARY primality, for contrast
    cols["prime"] = w["is_prime"][2:].astype(float).tolist()
    fit2 = tk.call("stats_fit_table", {"columns": cols, "outcome": "prime",
                                       "regressors": ["mod2", "mod4_is1", "digit_sum", "log_n"], "model": "logit"})
    print(f"same features on ordinary primality: accuracy={fit2['accuracy']}  base rate={fit2['base_rate']}")

    # 4. density in the limit: ordinary primes thin out; 'partial primes' do not
    lim = tk.call("symbolic_take_limit", {"expression": "(x/log(x))/x", "variable": "x", "target": "oo"})
    print(f"\ndensity of ordinary primes as x -> oo: {lim['limit']}")
    frac = [float((w['partial_primality'][2:k] == 3).mean()) for k in (1000, 5000, 20000)]
    print("fraction of fully partial-prime integers up to 1k/5k/20k:", [round(f, 3) for f in frac])

    # 5. ground: what has to give
    g = tk.call("ground_what_must_be_true", {"rule_id": "nt.prime.def", "operator": "SUBSTITUTE"})
    cost = tk.call("ground_cost_of_possibility", {"rule_ids": ["nt.prime.def"]})
    print("\nwhat must be true:", [c["rule_id"] for c in g["minimal_changes"]])
    print("collateral:", [c["rule_id"] for c in g["collateral"]])
    print(f"cost of possibility: {cost['total']} ({cost['band']}) - {cost['reading']}")


if __name__ == "__main__":
    main()
