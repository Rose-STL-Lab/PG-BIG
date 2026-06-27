"""Muscle activation surrogate model."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MLPModel"]


class MLPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc_main = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        main_out = self.fc_main(x)
        x = main_out.contiguous().view(batch_size, -1, 80)
        return torch.tanh(x) / 2 + 0.5
