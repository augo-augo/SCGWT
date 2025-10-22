from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class ActorConfig:
    latent_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


@dataclass
class CriticConfig:
    latent_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(num_layers):
        layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """Actor producing Gaussian action distributions conditioned on GW broadcast and memory context."""

    def __init__(self, config: ActorConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _build_mlp(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))

    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        hidden = self.backbone(features)
        mean = self.mean_head(hidden)
        std = torch.exp(self.log_std).clamp(min=1e-4, max=10.0)
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std),
            1,
        )


class CriticNetwork(nn.Module):
    """Critic estimating state value from aggregated latent features."""

    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _build_mlp(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(features)).squeeze(-1)
