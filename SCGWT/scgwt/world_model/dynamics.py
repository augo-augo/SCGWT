from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DynamicsConfig:
    """Hyperparameters for a single dynamics model in the ensemble."""

    latent_dim: int
    action_dim: int
    hidden_dim: int = 256


class DynamicsModel(nn.Module):
    """Simple GRU-based dynamics model stub."""

    def __init__(self, config: DynamicsConfig) -> None:
        super().__init__()
        self.config = config
        self.input_layer = nn.Linear(config.latent_dim + config.action_dim, config.hidden_dim)
        self.transition = nn.GRUCell(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        latent_state: torch.Tensor,
        action: torch.Tensor,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next latent state given the previous latent state and action."""
        if latent_state.shape[0] != action.shape[0]:
            raise ValueError("latent_state and action batch dimensions must match")
        joint = torch.cat([latent_state, action], dim=-1)
        hidden = torch.relu(self.input_layer(joint))
        next_latent = self.transition(hidden, latent_state)

        if output_buffer is None:
            return next_latent

        output_buffer.copy_(next_latent)
        return output_buffer
