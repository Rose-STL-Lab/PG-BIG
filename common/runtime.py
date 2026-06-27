"""Runtime helpers: device resolution and seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["resolve_torch_device", "set_seed"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_device(local_rank: int | None = None) -> torch.device:
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")
