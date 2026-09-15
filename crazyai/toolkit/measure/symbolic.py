"""symbolic.* - mathematics that is actually checked (SymPy)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import sympy as sp

from crazyai.toolkit import native
from crazyai.toolkit.registry import tool

_BASE_UNITS = {u: sp.Symbol(u, positive=True) for u in ("kg", "m", "s", "A", "K", "mol", "cd")}


class isprime(sp.logic.boolalg.BooleanFunction):  # noqa: N801 - displayed name
    """Deferred primality test so 'isprime(n)' can be written before n is known."""

    @classmethod
    def eval(cls, n):
        if getattr(n, "is_Integer", False):
            return sp.true if sp.isprime(int(n)) else sp.false
        return None


_KNOWN = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi, "oo": sp.oo, "Mod": sp.Mod, "isprime": isprime,
    "floor": sp.floor, "ceiling": sp.ceiling, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "gamma": sp.gamma, "factorial": sp.factorial,
}
_IDENT = re.compile(r"[A-Za-z_]\w*")


def _parse(expr: str) -> sp.Expr:
    """sympify where every unknown identifier is a plain Symbol (so E, I, S, N, Q are not sympy constants)."""
    local = dict(_KNOWN)
    for ident in set(_IDENT.findall(expr)):
        if ident not in local:
            local[ident] = sp.Symbol(ident)
    return sp.sympify(expr, locals=local)


def _unit_expr(unit: str) -> sp.Expr:
    if unit.strip() in ("1", "", "dimensionless"):
        return sp.Integer(1)
    return sp.sympify(unit, locals=_BASE_UNITS)


@tool("symbolic", "measure")
def derive(expression: str, rule: str, argument: str = "") -> dict:
    """Apply one named algebraic rule to an expression and return the result, so each derivation step is checked.

    Args:
        expression: SymPy-parsable expression, e.g. 'm*v**2/2'.
        rule: One of simplify, expand, factor, differentiate, integrate, solve, substitute, collect, cancel.
        argument: For differentiate/integrate/solve/collect: the variable; for substitute: 'sym=value' pairs separated by commas.
    """
    try:
        e = _parse(expression)
        rule = rule.lower()
        if rule == "simplify":
            r = sp.simplify(e)
        elif rule == "expand":
            r = sp.expand(e)
        elif rule == "factor":
            r = sp.factor(e)
        elif rule == "cancel":
            r = sp.cancel(e)
        elif rule == "differentiate":
            r = sp.diff(e, sp.Symbol(argument))
        elif rule == "integrate":
            r = sp.integrate(e, sp.Symbol(argument))
        elif rule == "collect":
            r = sp.collect(e, sp.Symbol(argument))
        elif rule == "solve":
            sols = sp.solve(e, sp.Symbol(argument))
            return {"input": str(e), "rule": rule, "solutions": [str(s) for s in sols]}
        elif rule == "substitute":
            subs = {}
            for pair in argument.split(","):
                k, _, v = pair.partition("=")
                subs[sp.Symbol(k.strip())] = _parse(v.strip())
            r = e.subs(subs)
        else:
            return {"error": f"unknown rule {rule!r}"}
        return {"input": str(e), "rule": rule, "argument": argument, "output": str(r)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


class _DimError(Exception):
    pass


def _units_of(e: sp.Expr, sub: dict[sp.Symbol, sp.Expr]) -> sp.Expr:
    """Recursive unit evaluation: additive terms must agree, function arguments must be dimensionless."""
    if e.is_Number:
        return sp.Integer(1)
    if e.is_Symbol:
        if e not in sub:
            raise _DimError(f"no units for {e}")
        return sub[e]
    if e.is_Add:
        units = [sp.simplify(_units_of(a, sub)) for a in e.args]
        for u in units[1:]:
            if sp.simplify(u / units[0]) != 1:
                raise _DimError(f"adding terms with different units: {units[0]} and {u}")
        return units[0]
    if e.is_Mul:
        out = sp.Integer(1)
        for a in e.args:
            out *= _units_of(a, sub)
        return sp.simplify(out)
    if e.is_Pow:
        base, ex = e.args
        if ex.free_symbols:
            if sp.simplify(_units_of(base, sub)) != 1:
                raise _DimError("dimensional base raised to a symbolic power")
            return sp.Integer(1)
        return sp.simplify(_units_of(base, sub) ** ex)
    if isinstance(e, sp.Function):
        for a in e.args:
            if sp.simplify(_units_of(a, sub)) != 1:
                raise _DimError(f"argument of {e.func.__name__} is not dimensionless")
        return sp.Integer(1)
    return sp.Integer(1)


@tool("symbolic", "measure")
def check_dimensions(equation: str, units: dict) -> dict:
    """Dimensional analysis: check that both sides of 'lhs = rhs' reduce to the same SI units (term by term).

    Args:
        equation: An equation string 'lhs = rhs', e.g. 'E = m*c**2'.
        units: Mapping symbol -> unit expression in kg, m, s, A, K, mol, cd ('1' for dimensionless), e.g. {"E": "kg*m**2/s**2", "m": "kg", "c": "m/s"}.
    """
    try:
        if "=" not in equation:
            return {"error": "equation must contain '='"}
        lhs_s, rhs_s = equation.split("=", 1)
        lhs, rhs = _parse(lhs_s), _parse(rhs_s)
        sub = {sp.Symbol(k): _unit_expr(v) for k, v in units.items()}
        missing = sorted({str(s) for s in (lhs.free_symbols | rhs.free_symbols)} - set(units))
        if missing:
            return {"error": f"no units given for symbols: {missing}"}
        try:
            ul, ur = sp.simplify(_units_of(lhs, sub)), sp.simplify(_units_of(rhs, sub))
        except _DimError as exc:
            return {"equation": equation, "consistent": False, "flaw": str(exc)}
        consistent = sp.simplify(ul / ur) == 1
        return {"equation": equation, "lhs_units": str(ul), "rhs_units": str(ur),
                "consistent": bool(consistent), "flaw": None if consistent else "dimensions do not match"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@tool("symbolic", "measure")
def take_limit(expression: str, variable: str, target: str) -> dict:
    """Take the limit of an expression as a variable goes to a target (a number, 'oo' or '-oo').

    Args:
        expression: SymPy-parsable expression.
        variable: The symbol.
        target: The limit point, e.g. '0', 'oo', '-oo', 'c'.
    """
    try:
        e = _parse(expression)
        r = sp.limit(e, sp.Symbol(variable), _parse(target))
        return {"expression": str(e), "variable": variable, "target": target, "limit": str(r)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@tool("symbolic", "measure")
def series_expand(expression: str, variable: str, around: str = "0", order: int = 4) -> dict:
    """Series expansion of an expression around a point.

    Args:
        expression: SymPy-parsable expression.
        variable: The symbol to expand in.
        around: Expansion point.
        order: Number of terms.
    """
    try:
        e = _parse(expression)
        r = sp.series(e, sp.Symbol(variable), _parse(around), int(order))
        return {"series": str(r)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@tool("symbolic", "measure")
def verify_identity(lhs: str, rhs: str) -> dict:
    """Check whether two expressions are identically equal.

    Args:
        lhs: Left expression.
        rhs: Right expression.
    """
    try:
        d = sp.simplify(_parse(lhs) - _parse(rhs))
        ok = d == 0
        return {"lhs": lhs, "rhs": rhs, "identical": bool(ok), "difference": str(d)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@tool("symbolic", "measure")
def solve(equation: str, variable: str) -> dict:
    """Solve 'lhs = rhs' for a variable.

    Args:
        equation: Equation string containing '='.
        variable: Symbol to solve for.
    """
    try:
        l, r = equation.split("=", 1)
        sols = sp.solve(sp.Eq(_parse(l), _parse(r)), sp.Symbol(variable))
        return {"solutions": [str(s) for s in sols]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@tool("symbolic", "measure")
def compare_structures(a: dict, b: dict, tolerance: float = 1e-6) -> dict:
    """Compare two structures {nodes:[...]} node by node and report exactly what differs (used after a transform round trip).

    Args:
        a: Original structure.
        b: Round-tripped structure.
        tolerance: Numeric tolerance for x, y, weight.
    """
    na = {str(n.get("id")): n for n in a.get("nodes", [])}
    nb = {str(n.get("id")): n for n in b.get("nodes", [])}
    only_a, only_b = sorted(set(na) - set(nb)), sorted(set(nb) - set(na))
    diffs = []
    for k in sorted(set(na) & set(nb)):
        for f in ("x", "y", "weight"):
            va, vb = float(na[k].get(f, 0)), float(nb[k].get(f, 0))
            if abs(va - vb) > tolerance:
                diffs.append({"id": k, "field": f, "a": va, "b": vb})
        if int(na[k].get("group", 0)) != int(nb[k].get("group", 0)):
            diffs.append({"id": k, "field": "group", "a": na[k].get("group"), "b": nb[k].get("group")})
    order_a = [str(n.get("id")) for n in a.get("nodes", [])]
    order_b = [str(n.get("id")) for n in b.get("nodes", [])]
    return {"identical": not (only_a or only_b or diffs) and order_a == order_b,
            "only_in_a": only_a, "only_in_b": only_b, "field_differences": diffs,
            "order_preserved": order_a == order_b, "order_reversed": order_a == order_b[::-1]}


@tool("symbolic", "measure")
def primes_up_to(n: int) -> dict:
    """Sieve the primes up to n (native C++ when available). Returns count, the first 50, and the last 10.

    Args:
        n: Upper bound (<= 50,000,000).
    """
    n = int(n)
    if n < 2 or n > 50_000_000:
        return {"error": "n must be in [2, 50000000]"}
    mask = native.sieve(n)
    ps = np.flatnonzero(mask)
    return {"n": n, "count": int(ps.size), "first": ps[:50].tolist(), "last": ps[-10:].tolist(),
            "backend": native.backend()}


@tool("symbolic", "measure")
def collatz_orbits(n: int) -> dict:
    """Collatz stopping time and orbit peak for 1..n (native C++ when available), plus summary statistics.

    Args:
        n: Upper bound (<= 5,000,000).
    """
    n = int(n)
    if n < 1 or n > 5_000_000:
        return {"error": "n must be in [1, 5000000]"}
    ln, pk = native.collatz(n)
    ln, pk = ln[1:], pk[1:]
    longest = int(np.argmax(ln)) + 1
    return {"n": n, "max_stopping_time": int(ln.max()), "argmax": longest,
            "mean_stopping_time": float(ln.mean()), "max_peak": int(pk.max()),
            "first_20": [{"k": k, "steps": int(ln[k - 1]), "peak": int(pk[k - 1])} for k in range(1, min(n, 20) + 1)],
            "backend": native.backend()}


@tool("symbolic", "measure")
def define_predicate(name: str, expression: str, variable: str = "n", test_values: list | None = None) -> dict:
    """Define a boolean predicate from a SymPy expression and evaluate it on test values.

    Args:
        name: Predicate name, e.g. 'P_even'.
        expression: A SymPy boolean/relational expression in the variable, e.g. 'Mod(n, 2) == 0' or 'n > 3'.
        variable: The free variable.
        test_values: Integers to evaluate on (default 1..20).
    """
    try:
        v = sp.Symbol(variable, integer=True)
        local = dict(_KNOWN)
        for ident in set(_IDENT.findall(expression)):
            if ident not in local:
                local[ident] = sp.Symbol(ident, integer=True)
        local[variable] = v
        local["Eq"] = sp.Eq
        from sympy.parsing.sympy_parser import convert_equals_signs, parse_expr, standard_transformations

        e = parse_expr(expression.replace("==", "="), local_dict=local,
                       transformations=standard_transformations + (convert_equals_signs,))
        vals = test_values or list(range(1, 21))
        table = {}
        for t in vals:
            r = sp.simplify(e.subs(v, int(t)))
            if r in (sp.true, True):
                table[str(t)] = True
            elif r in (sp.false, False):
                table[str(t)] = False
            else:
                return {"error": f"predicate did not reduce to a boolean at {t}: {r}"}
        return {"name": name, "expression": str(e), "truth_table": table,
                "true_count": sum(table.values()), "tested": len(table)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _to_dict(x: Any) -> Any:
    return x
