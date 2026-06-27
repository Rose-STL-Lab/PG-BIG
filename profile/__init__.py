"""Personalized profile encoder and subject prior."""

from profile.model import ProfileActionToMotionTransformer, ProfileEncoder

__all__ = [
    "ProfileActionToMotionTransformer",
    "ProfileEncoder",
    "main_encoder",
    "main_prior",
]


def __getattr__(name: str):
    if name == "main_encoder":
        from profile.train_encoder import main
        return main
    if name == "main_prior":
        from profile.train_prior import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
