"""Motion-text evaluation utilities."""

__all__ = ["EvaluatorModelWrapper"]


def __getattr__(name: str):
    if name == "EvaluatorModelWrapper":
        from eval.wrapper import EvaluatorModelWrapper
        return EvaluatorModelWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
