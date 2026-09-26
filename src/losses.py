"""Pérdidas adversariales controladas para los tres experimentos."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AdversarialLoss:
    """Interfaz común para BCE no saturante y hinge loss."""

    VALID_NAMES = {"bce_non_saturating", "hinge"}

    def __init__(self, name: str) -> None:
        if name not in self.VALID_NAMES:
            raise ValueError(f"Pérdida desconocida: {name}. Opciones: {sorted(self.VALID_NAMES)}")
        self.name = name
        self._bce = nn.BCEWithLogitsLoss()

    def discriminator(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.name == "bce_non_saturating":
            real_targets = torch.ones_like(real_logits)
            fake_targets = torch.zeros_like(fake_logits)
            return self._bce(real_logits, real_targets) + self._bce(fake_logits, fake_targets)
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

    def generator(self, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.name == "bce_non_saturating":
            return self._bce(fake_logits, torch.ones_like(fake_logits))
        return -fake_logits.mean()
