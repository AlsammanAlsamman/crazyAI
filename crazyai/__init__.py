"""crazyAI - the impossible, disguised as possible and true.

Public entry points:

    from crazyai import Run, build_toolkit
    run = Run(seed=42, generator="formula")
    run.execute(provider)
"""

from crazyai.rng import RunRNG
from crazyai.toolkit.registry import Toolkit, build_toolkit

__all__ = ["RunRNG", "Toolkit", "build_toolkit", "Run"]
__version__ = "0.1.0"


def __getattr__(name):  # lazy: the pipeline pulls in providers, which are heavier
    if name == "Run":
        from crazyai.pipeline.run import Run
        return Run
    raise AttributeError(name)
