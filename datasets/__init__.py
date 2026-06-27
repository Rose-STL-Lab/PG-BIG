"""Dataset loaders for PG-BIG."""

__all__ = [
    "Retargeted183Dataset",
    "retargeted183_data_loader",
    "cycle",
]


def __getattr__(name: str):
    if name in __all__:
        from datasets import athletes_retarget
        return getattr(athletes_retarget, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
