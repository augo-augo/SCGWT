from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _resolve_activation(name: str) -> nn.Module:
    name = name.lower()
    activations = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "identity": nn.Identity,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }
    if name in activations:
        return activations[name]()
    raise ValueError(f"Unsupported activation '{name}'")


@dataclass
class DecoderConfig:
    """Configuration for the shared decoder."""

    observation_shape: tuple[int, int, int]
    slot_dim: int
    hidden_channels: Sequence[int] = field(default_factory=lambda: (256, 128, 64))
    initial_spatial: tuple[int, int] = (4, 4)
    activation: str = "relu"
    output_activation: str = "sigmoid"
    init_log_std: float = -0.5
    learn_log_std: bool = True

    def __post_init__(self) -> None:
        if len(self.hidden_channels) < 1:
            raise ValueError("hidden_channels must include at least one entry")
        h, w = self.initial_spatial
        if h <= 0 or w <= 0:
            raise ValueError("initial_spatial must provide positive dimensions")


class SharedDecoder(nn.Module):
    """Shared decoder that projects latent slots back into observation space distributions."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.activation = _resolve_activation(config.activation)
        self.output_activation = _resolve_activation(config.output_activation)
        self._build_network()

    def _build_network(self) -> None:
        c, _, _ = self.config.observation_shape
        init_h, init_w = self.config.initial_spatial
        hidden_channels = list(self.config.hidden_channels)
        self.fc = nn.Linear(
            self.config.slot_dim, hidden_channels[0] * init_h * init_w
        )
        deconv_layers: list[nn.Module] = []
        in_channels = hidden_channels[0]
        for next_channels in hidden_channels[1:]:
            deconv_layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        next_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    self.activation,
                ]
            )
            in_channels = next_channels
        deconv_layers.append(
            nn.ConvTranspose2d(
                in_channels, c, kernel_size=3, stride=1, padding=1
            )
        )
        self.deconv = nn.Sequential(*deconv_layers)
        log_std = torch.full(self.config.observation_shape, self.config.init_log_std)
        if self.config.learn_log_std:
            self.log_std = nn.Parameter(log_std)
        else:
            self.register_buffer("log_std", log_std)

    def forward(self, latent_slots: torch.Tensor) -> torch.distributions.Distribution:
        """
        Decode latent slots into a factorized Normal observation distribution.

        Callers should provide aggregated slot representations with shape [batch, slot_dim].
        """
        if latent_slots.ndim != 2:
            raise ValueError("latent_slots must have shape [batch, slot_dim]")
        batch, slot_dim = latent_slots.shape
        if slot_dim != self.config.slot_dim:
            raise ValueError("slot_dim mismatch with decoder configuration")
        init_h, init_w = self.config.initial_spatial
        hidden = self.activation(self.fc(latent_slots))
        hidden = hidden.view(batch, self.config.hidden_channels[0], init_h, init_w)
        mean = self.deconv(hidden)
        # Resize to the target spatial resolution if necessary.
        _, target_h, target_w = self.config.observation_shape
        if mean.shape[-2:] != (target_h, target_w):
            mean = F.interpolate(mean, size=(target_h, target_w), mode="bilinear", align_corners=False)
        mean = self.output_activation(mean)
        std = torch.exp(self.log_std)
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std),
            len(self.config.observation_shape),
        )
