"""Muscle activation surrogate."""

from surrogate.model import MLPModel

__all__ = ["MLPModel", "main"]


def __getattr__(name: str):
    if name == "main":
        from surrogate.train import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
