"""VQ-VAE motion representation."""

from vqvae.model import HumanVQVAE, VQVAE_251

__all__ = ["HumanVQVAE", "VQVAE_251", "main"]


def __getattr__(name: str):
    if name == "main":
        from vqvae.train import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
