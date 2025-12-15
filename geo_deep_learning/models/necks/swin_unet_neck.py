from __future__ import annotations

import torch
from torch import nn


class IdentityNeck(nn.Module):
    """A minimal neck that keeps the interface explicit."""

    def __init__(self, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.norm = norm if norm is not None else nn.Identity()

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.norm(bottleneck), skips