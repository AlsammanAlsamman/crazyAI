"""Tool registry.

A tool is a plain Python function decorated with @tool(family, kind). The JSON
schema Claude sees is generated from the signature and docstring, so adding a
tool means adding a function. Every call goes through Toolkit.call, which
serialises the result and appends it to the call log.
"""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, get_args, get_origin, get_type_hints

from crazyai.rng import RunRNG

Kind = typing.Literal["invent", "measure"]

_REGISTRY: dict[str, "ToolSpec"] = {}


@dataclass
class ToolSpec:
    name: str
    family: str
    kind: str
    fn: Callable[..., Any]
    description: str
    input_schema: dict[str, Any]
    needs_rng: bool
    needs_context: bool


def _json_type(tp: Any) -> dict[str, Any]:
    origin = get_origin(tp)
    if tp in (str,):
        return {"type": "string"}
    if tp in (int,):
        return {"type": "integer"}
    if tp in (float,):
        return {"type": "number"}
    if tp in (bool,):
        return {"type": "boolean"}
    if tp is Any or tp is object:
        return {}
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _json_type(args[0])
        return {"anyOf": [_json_type(a) for a in args]}
    if origin in (list, typing.List):
        (inner,) = get_args(tp) or (Any,)
        return {"type": "array", "items": _json_type(inner)}
    if origin in (dict, typing.Dict) or tp is dict:
        return {"type": "object"}
    if tp is list:
        return {"type": "array"}
    if origin is typing.Literal:
        return {"type": "string", "enum": list(get_args(tp))}
    return {}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {arg: description}) from a Google-style docstring."""
    if not doc:
        return "", {}
    lines = [l.rstrip() for l in inspect.cleandoc(doc).splitlines()]
    summary_lines: list[str] = []
    args: dict[str, str] = {}
    in_args = False
    current = None
    for line in lines:
        if line.strip().lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if line.strip().lower().startswith(("returns:", "raises:", "example")):
                in_args = False
                continue
            if line and not line.startswith(" ") and ":" not in line:
                in_args = False
                continue
            stripped = line.strip()
            if ":" in stripped and not line.startswith("        "):
                name, _, desc = stripped.partition(":")
                current = name.strip()
                args[current] = desc.strip()
            elif current and stripped:
                args[current] += " " + stripped
        else:
            summary_lines.append(line)
    summary = " ".join(l.strip() for l in summary_lines if l.strip())
    return summary, args


def tool(family: str, kind: Kind) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as a tool.

    Functions may declare a leading parameter named `rng` (RunRNG) and/or `ctx`
    (dict) - those are injected by the toolkit and hidden from Claude.
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        summary, arg_docs = _parse_docstring(fn.__doc__)
        props: dict[str, Any] = {}
        required: list[str] = []
        needs_rng = needs_context = False
        for pname, param in sig.parameters.items():
            if pname == "rng":
                needs_rng = True
                continue
            if pname == "ctx":
                needs_context = True
                continue
            schema = _json_type(hints.get(pname, Any))
            if pname in arg_docs:
                schema = {**schema, "description": arg_docs[pname]}
            props[pname] = schema
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        name = f"{family}_{fn.__name__}"
        spec = ToolSpec(
            name=name,
            family=family,
            kind=kind,
            fn=fn,
            description=summary or fn.__name__,
            input_schema={"type": "object", "properties": props, "required": required},
            needs_rng=needs_rng,
            needs_context=needs_context,
        )
        _REGISTRY[name] = spec
        fn.__tool__ = spec  # type: ignore[attr-defined]
        return fn

    return deco


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class Toolkit:
    """A run-scoped view over the registry: knows the run's RNG and context, logs calls."""

    rng: RunRNG
    ctx: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def all_specs() -> dict[str, ToolSpec]:
        return dict(_REGISTRY)

    def names(self, families: list[str] | None = None, kind: str | None = None) -> list[str]:
        out = []
        for name, spec in _REGISTRY.items():
            if families and spec.family not in families:
                continue
            if kind and spec.kind != kind:
                continue
            out.append(name)
        return sorted(out)

    def to_api_tools(self, names: list[str]) -> list[dict[str, Any]]:
        """Tool definitions in the shape the Messages API expects."""
        defs = []
        for n in names:
            spec = _REGISTRY[n]
            defs.append({"name": spec.name, "description": spec.description, "input_schema": spec.input_schema})
        return defs

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = _REGISTRY.get(name)
        if spec is None:
            result: dict[str, Any] = {"error": f"unknown tool {name!r}"}
            self.calls.append({"tool": name, "input": arguments, "output": result, "ok": False})
            return result
        kwargs = dict(arguments or {})
        if spec.needs_rng:
            kwargs["rng"] = self.rng
        if spec.needs_context:
            kwargs["ctx"] = self.ctx
        try:
            raw = spec.fn(**kwargs)
            result = _jsonable(raw)
            if not isinstance(result, dict):
                result = {"result": result}
            ok = "error" not in result
        except Exception as exc:  # tools must never crash the loop
            result = {"error": f"{type(exc).__name__}: {exc}"}
            ok = False
        self.calls.append({"tool": name, "input": arguments, "output": result, "ok": ok})
        return result

    def call_json(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        return json.dumps(self.call(name, arguments), ensure_ascii=False, default=str)


def build_toolkit(seed: int = 0, ctx: dict[str, Any] | None = None) -> Toolkit:
    """Import every tool module (so they register) and return a run-scoped Toolkit."""
    from crazyai.toolkit import invent, measure  # noqa: F401  (import for side effects)

    return Toolkit(rng=RunRNG(seed), ctx=ctx or {})
