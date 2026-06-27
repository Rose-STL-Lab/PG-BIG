"""Nimblephysics tooling for PG-BIG."""

__all__ = [
    "retarget_athletes",
]


def __getattr__(name: str):
    if name == "retarget_athletes":
        from nimble.retarget import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
