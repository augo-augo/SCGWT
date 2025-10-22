from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class WorkspaceConfig:
    broadcast_slots: int
    self_bias: float
    novelty_weight: float
    progress_weight: float
    cost_weight: float
    progress_momentum: float = 0.1
    action_cost_scale: float = 1.0
    ucb_weight: float = 0.2
    ucb_beta: float = 1.0


class WorkspaceRouter:
    """
    Implements a lightweight attention bottleneck that selects a subset of slots for the GW.
    """

    def __init__(self, config: WorkspaceConfig) -> None:
        self.config = config

    def score_slots(
        self,
        novelty: torch.Tensor,
        progress: torch.Tensor,
        ucb: torch.Tensor,
        cost: torch.Tensor,
        self_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attraction scores combining novelty, progress, and usage cost. The tensors are
        expected to have shape [batch, num_slots].
        """
        score = (
            self.config.novelty_weight * novelty
            + self.config.progress_weight * progress
            + self.config.ucb_weight * ucb
            - self.config.cost_weight * cost
        )
        score = score + self.config.self_bias * self_mask
        return score

    def broadcast(self, slots: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Select top-k slots according to provided scores.

        Args:
            slots: [batch, num_slots, slot_dim]
            scores: [batch, num_slots]
        """
        k = self.config.broadcast_slots
        topk = torch.topk(scores, k=k, dim=1).indices
        return torch.gather(
            slots, dim=1, index=topk.unsqueeze(-1).expand(-1, -1, slots.shape[-1])
        )
