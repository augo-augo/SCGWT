from __future__ import annotations

from dataclasses import dataclass

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
        self.register_buffer("_queue", torch.zeros(config.queue_capacity, config.latent_dim))
        self.register_buffer("_queue_step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_queue_count", torch.zeros(1, dtype=torch.long))

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
        batch, _ = latent.shape
        available = int(self._queue_count.item())
        if available == 0:
            return self.latent_proj(latent.detach()).unsqueeze(1)
        capacity = self._queue.size(0)
        limit = min(available, capacity)
        queue_tensor = self._queue[:limit]
        if queue_tensor.device != latent.device:
            queue_tensor = queue_tensor.to(latent.device, non_blocking=True)
        idx = torch.randint(0, limit, (batch,), device=latent.device)
        sampled = queue_tensor.index_select(0, idx).detach()
        embedded = self.latent_proj(sampled)
        return embedded.unsqueeze(1)

    def _enqueue_latents(self, latent: torch.Tensor) -> None:
        if latent.numel() == 0:
            return
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        device = self._queue.device
        data = latent.detach().to(device=device, dtype=self._queue.dtype, non_blocking=True)
        capacity = self._queue.size(0)
        start = int(self._queue_step.item())
        positions = (torch.arange(data.size(0), device=device, dtype=torch.long) + start) % capacity
        self._queue.index_copy_(0, positions, data.detach())
        self._queue_step.add_(data.size(0))
        self._queue_count.add_(data.size(0))
        self._queue_count.clamp_(max=capacity)
