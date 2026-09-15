"""stats.* - statistical modelling that is actually fitted (NumPy/SciPy).

A data-generating process (DGP) is a small spec:

    {"n": 500, "seed": 1,
     "vars": [{"name": "u", "dist": "normal", "params": [0, 1]},
              {"name": "x", "formula": "0.8*u + normal(0, 1)"},
              {"name": "y", "formula": "2*x + 1.5*u + normal(0, 1)"}]}

Distributions: normal(mu, sigma), uniform(a, b), bernoulli(p), poisson(lam),
exponential(scale). Formulas are Python expressions over earlier variables
and the distribution functions.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
from scipy import stats as sps

from crazyai.toolkit import native
from crazyai.toolkit.registry import tool

_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod,
                  ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq, ast.IfExp,
                  ast.BoolOp, ast.And, ast.Or)


def _safe_eval(expr: str, env: dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax in formula: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in env:
            raise ValueError(f"unknown name in formula: {node.id}")
    return eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, env)


def _dist_env(gen: np.random.Generator, n: int) -> dict[str, Any]:
    return {
        "normal": lambda mu=0.0, sigma=1.0: gen.normal(mu, sigma, n),
        "uniform": lambda a=0.0, b=1.0: gen.uniform(a, b, n),
        "bernoulli": lambda p=0.5: gen.binomial(1, p, n).astype(float),
        "poisson": lambda lam=1.0: gen.poisson(lam, n).astype(float),
        "exponential": lambda scale=1.0: gen.exponential(scale, n),
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "sin": np.sin, "cos": np.cos,
        "where": np.where, "floor": np.floor, "round": np.round, "pi": np.pi, "e": np.e,
    }


def _simulate(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    n = int(spec.get("n", 200))
    gen = np.random.default_rng(int(spec.get("seed", 0)))
    env = _dist_env(gen, n)
    data: dict[str, np.ndarray] = {}
    for v in spec.get("vars", []):
        name = v["name"]
        if "formula" in v:
            arr = _safe_eval(v["formula"], {**env, **data})
            arr = np.broadcast_to(np.asarray(arr, dtype=float), (n,)).copy()
        else:
            arr = env[v["dist"]](*v.get("params", []))
        data[name] = arr
    return data


@tool("stats", "measure")
def simulate_dgp(spec: dict) -> dict:
    """Simulate data from a data-generating-process spec (see module doc). Returns summary statistics and a preview.

    Args:
        spec: {"n": int, "seed": int, "vars": [{"name", "dist", "params"} | {"name", "formula"}]}.
    """
    try:
        data = _simulate(spec)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    summary = {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                   "min": float(v.min()), "max": float(v.max())} for k, v in data.items()}
    names = list(data)
    corr = np.corrcoef(np.vstack([data[k] for k in names])) if len(names) > 1 else np.ones((1, 1))
    return {"n": int(next(iter(data.values())).size), "variables": names, "summary": summary,
            "correlation": {a: {b: round(float(corr[i, j]), 4) for j, b in enumerate(names)} for i, a in enumerate(names)},
            "preview": {k: [round(float(x), 4) for x in v[:5]] for k, v in data.items()}}


def _ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / np.where(se > 0, se, np.nan)
    p = 2 * (1 - sps.t.cdf(np.abs(t), dof))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return {"beta": beta, "se": se, "t": t, "p": p, "r2": r2, "n": n, "k": k, "sigma": float(np.sqrt(sigma2))}


@tool("stats", "measure")
def fit(spec: dict, outcome: str, regressors: list, model: str = "ols") -> dict:
    """Simulate the DGP and fit a model of outcome on regressors (with intercept). Returns coefficients, SEs, p-values, R².

    Args:
        spec: DGP spec as for stats_simulate_dgp.
        outcome: Name of the outcome variable.
        regressors: Names of regressor variables.
        model: 'ols' (default) or 'logit' (outcome must be 0/1).
    """
    try:
        data = _simulate(spec)
        y = data[outcome]
        X = np.column_stack([np.ones_like(y)] + [data[r] for r in regressors])
        names = ["intercept"] + list(regressors)
        if model == "ols":
            r = _ols(y, X)
        elif model == "logit":
            from scipy.optimize import minimize

            def nll(b):
                z = X @ b
                return float(np.sum(np.logaddexp(0, z) - y * z))
            res = minimize(nll, np.zeros(X.shape[1]), method="BFGS")
            b = res.x
            pr = 1 / (1 + np.exp(-(X @ b)))
            W = pr * (1 - pr)
            cov = np.linalg.pinv((X * W[:, None]).T @ X)
            se = np.sqrt(np.diag(cov))
            z = b / se
            r = {"beta": b, "se": se, "t": z, "p": 2 * (1 - sps.norm.cdf(np.abs(z))), "r2": None,
                 "n": X.shape[0], "k": X.shape[1], "sigma": None}
        else:
            return {"error": f"unknown model {model!r}"}
        table = {nm: {"coef": round(float(r["beta"][i]), 5), "se": round(float(r["se"][i]), 5),
                      "t": round(float(r["t"][i]), 3), "p": round(float(r["p"][i]), 5)} for i, nm in enumerate(names)}
        return {"model": model, "outcome": outcome, "n": r["n"], "coefficients": table,
                "r2": None if r["r2"] is None else round(r["r2"], 4)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@tool("stats", "measure")
def inject_confounder(spec: dict, treatment: str, outcome: str, strength: float = 1.0, name: str = "u_hidden") -> dict:
    """Return a copy of the DGP with a hidden variable added that drives both treatment and outcome, and the resulting bias.

    Args:
        spec: DGP spec.
        treatment: Treatment variable name (must have a formula).
        outcome: Outcome variable name (must have a formula).
        strength: Coefficient of the confounder in both equations.
        name: Name of the injected variable.
    """
    try:
        new_vars = [{"name": name, "dist": "normal", "params": [0, 1]}]
        for v in spec.get("vars", []):
            v = dict(v)
            if v["name"] in (treatment, outcome) and "formula" in v:
                v["formula"] = f"({v['formula']}) + {strength}*{name}"
            new_vars.append(v)
        new_spec = {**spec, "vars": new_vars}
        regs = [treatment]
        before = fit(spec, outcome, regs)
        after = fit(new_spec, outcome, regs)
        b0 = before["coefficients"][treatment]["coef"]
        b1 = after["coefficients"][treatment]["coef"]
        return {"spec_with_confounder": new_spec, "effect_before": b0, "effect_after": b1,
                "bias": round(b1 - b0, 5), "flaw": f"unmeasured {name} biases the {treatment} effect"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@tool("stats", "measure")
def bootstrap(values: list, statistic: str = "mean", draws: int = 2000, seed: int = 0) -> dict:
    """Bootstrap confidence interval for a statistic of a sample.

    Args:
        values: The sample.
        statistic: 'mean', 'median' or 'sd'.
        draws: Number of resamples.
        seed: RNG seed.
    """
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return {"error": "need at least 2 values"}
    fn = {"mean": np.mean, "median": np.median, "sd": lambda a: np.std(a, ddof=1)}.get(statistic)
    if fn is None:
        return {"error": f"unknown statistic {statistic!r}"}
    gen = np.random.default_rng(seed)
    boots = np.array([fn(gen.choice(x, x.size, replace=True)) for _ in range(int(draws))])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"statistic": statistic, "estimate": float(fn(x)), "ci95": [float(lo), float(hi)], "draws": int(draws)}


@tool("stats", "measure")
def power_analysis(effect_size: float, n_per_group: int, alpha: float = 0.05) -> dict:
    """Power of a two-sample t-test for a standardised effect size (Cohen's d).

    Args:
        effect_size: Cohen's d.
        n_per_group: Sample size per group.
        alpha: Significance level.
    """
    n = int(n_per_group)
    dof = 2 * n - 2
    nc = effect_size * np.sqrt(n / 2)
    crit = sps.t.ppf(1 - alpha / 2, dof)
    power = 1 - sps.nct.cdf(crit, dof, nc) + sps.nct.cdf(-crit, dof, nc)
    return {"effect_size": effect_size, "n_per_group": n, "alpha": alpha, "power": float(power)}


@tool("stats", "measure")
def monte_carlo(coefficients: list, means: list, sds: list, draws: int = 200000, seed: int = 0) -> dict:
    """Monte Carlo mean and variance of a linear combination of independent normals (native C++ when available).

    Args:
        coefficients: Weights c_i.
        means: Means mu_i.
        sds: Standard deviations sigma_i.
        draws: Number of samples.
        seed: RNG seed.
    """
    if not (len(coefficients) == len(means) == len(sds)):
        return {"error": "coefficients, means and sds must have the same length"}
    m, v = native.mc_linear_gaussian(coefficients, means, sds, int(draws), int(seed))
    c, mu, sd = map(np.asarray, (coefficients, means, sds))
    return {"mc_mean": m, "mc_variance": v, "exact_mean": float(c @ mu), "exact_variance": float((c**2) @ (sd**2)),
            "draws": int(draws), "backend": native.backend()}


@tool("stats", "measure")
def check_identifiability(spec: dict, regressors: list) -> dict:
    """Check whether the regressors are linearly identifiable (full-rank design) in data simulated from the spec.

    Args:
        spec: DGP spec.
        regressors: Regressor names.
    """
    try:
        data = _simulate(spec)
        X = np.column_stack([np.ones(int(spec.get("n", 200)))] + [data[r] for r in regressors])
        rank = int(np.linalg.matrix_rank(X))
        cond = float(np.linalg.cond(X))
        return {"rank": rank, "columns": X.shape[1], "identifiable": rank == X.shape[1],
                "condition_number": cond, "flaw": None if rank == X.shape[1] else "collinear regressors"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@tool("stats", "measure")
def describe(values: list) -> dict:
    """Mean, sd, min, max, quartiles of a list of numbers.

    Args:
        values: Numbers.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"error": "empty"}
    q = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "min": float(x.min()), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]), "max": float(x.max())}


@tool("stats", "measure")
def fit_table(columns: dict, outcome: str, regressors: list, model: str = "ols") -> dict:
    """Fit a model on raw data given as columns {name: [values]} (with intercept). Same output shape as stats_fit.

    Args:
        columns: Mapping variable name -> list of numbers, all the same length.
        outcome: Outcome column.
        regressors: Regressor columns.
        model: 'ols' or 'logit'.
    """
    try:
        data = {k: np.asarray(v, dtype=float) for k, v in columns.items()}
        n = len(data[outcome])
        if any(len(v) != n for v in data.values()):
            return {"error": "columns have different lengths"}
        y = data[outcome]
        X = np.column_stack([np.ones(n)] + [data[r] for r in regressors])
        names = ["intercept"] + list(regressors)
        if model == "ols":
            r = _ols(y, X)
            table = {nm: {"coef": round(float(r["beta"][i]), 5), "se": round(float(r["se"][i]), 5),
                          "t": round(float(r["t"][i]), 3), "p": round(float(r["p"][i]), 5)} for i, nm in enumerate(names)}
            return {"model": model, "outcome": outcome, "n": n, "coefficients": table, "r2": round(r["r2"], 4)}
        if model == "logit":
            from scipy.optimize import minimize

            def nll(b):
                z = X @ b
                return float(np.sum(np.logaddexp(0, z) - y * z))
            res = minimize(nll, np.zeros(X.shape[1]), method="BFGS")
            b = res.x
            pr = 1 / (1 + np.exp(-(X @ b)))
            W = pr * (1 - pr)
            cov = np.linalg.pinv((X * W[:, None]).T @ X)
            se = np.sqrt(np.diag(cov))
            z = b / np.where(se > 0, se, np.nan)
            p = 2 * (1 - sps.norm.cdf(np.abs(z)))
            table = {nm: {"coef": round(float(b[i]), 5), "se": round(float(se[i]), 5),
                          "z": round(float(z[i]), 3), "p": round(float(p[i]), 5)} for i, nm in enumerate(names)}
            acc = float(np.mean((pr >= 0.5) == (y >= 0.5)))
            return {"model": model, "outcome": outcome, "n": n, "coefficients": table, "accuracy": round(acc, 4),
                    "base_rate": round(float(y.mean()), 4)}
        return {"error": f"unknown model {model!r}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
