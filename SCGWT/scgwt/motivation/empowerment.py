from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch
from torch import nn


@dataclass
class EmpowermentConfig:
    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    queue_capacity: int = 128
    temperature: float = 0.1


class InfoNCEEmpowermentEstimator(nn.Module):
    """
    Lightweight InfoNCE-style empowerment estimator with a replay queue of latent states.
    """

    def __init__(self, config: EmpowermentConfig) -> None:
        super().__init__()
        self.config = config
        self.action_proj = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self._queue: Deque[torch.Tensor] = deque(maxlen=config.queue_capacity)

    def forward(self, action: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute empowerment reward using the current latent as a positive example and
        previously observed latents as negatives.
        """
        embedded_action = self.action_proj(action)
        embedded_latent = self.latent_proj(latent)

        negatives = self._collect_negatives(latent)
        all_latents = torch.cat([embedded_latent.unsqueeze(1), negatives], dim=1)

        logits = torch.einsum("bd,bnd->bn", embedded_action, all_latents) / self.temperature.clamp(min=1e-3)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits, labels, reduction="none")

        self._enqueue_latents(latent.detach())
        return -loss  # Higher reward for lower InfoNCE loss.

    def _collect_negatives(self, latent: torch.Tensor) -> torch.Tensor:
        batch, dim = latent.shape
        if not self._queue:
            return self.latent_proj(latent.detach()).unsqueeze(1)
        negatives = []
        queue_tensor = torch.stack(list(self._queue)).to(latent.device)
        idx = torch.randint(0, queue_tensor.size(0), (batch,), device=latent.device)
        sampled = queue_tensor[idx].detach()
        embedded = self.latent_proj(sampled)
        negatives.append(embedded.unsqueeze(1))
        return torch.cat(negatives, dim=1)

    def _enqueue_latents(self, latent: torch.Tensor) -> None:
        for row in latent:
            self._queue.append(row.cpu())
