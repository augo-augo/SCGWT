from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EncoderConfig:
    """Configuration container for the shared encoder."""

    observation_shape: tuple[int, int, int]
    slot_dim: int
    num_slots: int
    cnn_channels: Sequence[int] = field(default_factory=lambda: (32, 64, 128))
    kernel_size: int = 5
    slot_iterations: int = 3
    mlp_hidden_size: int = 128
    epsilon: float = 1e-6


class _ConvBackbone(nn.Module):
    """Simple CNN feature extractor with positional embeddings."""

    def __init__(self, in_channels: int, channels: Sequence[int], kernel_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for hidden in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        current,
                        hidden,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                ]
            )
            current = hidden
        layers.append(nn.Conv2d(current, current, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _PositionalEmbedding(nn.Module):
    """Adds sine-cosine positional encoding to flattened features."""

    def __init__(self, num_channels: int, height: int, width: int) -> None:
        super().__init__()
        self.register_buffer(
            "pos", self._build_embedding(num_channels, height, width), persistent=False
        )

    @staticmethod
    def _build_embedding(num_channels: int, height: int, width: int) -> torch.Tensor:
        if num_channels % 4 != 0:
            raise ValueError("Position embedding channels must be divisible by 4")
        y_range = torch.linspace(-1.0, 1.0, steps=height)
        x_range = torch.linspace(-1.0, 1.0, steps=width)
        yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")
        dim_t = torch.arange(num_channels // 4, dtype=torch.float32)
        dim_t = 1.0 / (10000 ** (dim_t / (num_channels // 4)))

        pos_y = yy[..., None] * dim_t
        pos_x = xx[..., None] * dim_t
        pe = torch.stack(
            [torch.sin(pos_y), torch.cos(pos_y), torch.sin(pos_x), torch.cos(pos_x)],
            dim=-1,
        )
        return pe.view(height * width, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.pos.shape[-1]:
            raise ValueError("Mismatched positional embedding dimension")
        if x.shape[1] != self.pos.shape[0]:
            raise ValueError("Mismatched positional embedding spatial size")
        return x + self.pos


class _SlotAttention(nn.Module):
    """Slot Attention implementation following Locatello et al., 2020."""

    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int,
        mlp_hidden_size: int,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        self.slot_mu = nn.Parameter(torch.zeros(1, 1, dim))
        self.slot_sigma = nn.Parameter(torch.ones(1, 1, dim))

        self.project_q = nn.Linear(dim, dim, bias=False)
        self.project_k = nn.Linear(dim, dim, bias=False)
        self.project_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, n, d = inputs.shape
        inputs = self.norm_inputs(inputs)
        mu = self.slot_mu.expand(b, self.num_slots, -1)
        sigma = F.softplus(self.slot_sigma)
        slots = mu + sigma * torch.randn_like(mu)

        k = self.project_k(inputs)
        v = self.project_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            dots = torch.matmul(k, q.transpose(1, 2)) / (d**0.5)
            attn = dots.softmax(dim=-1) + self.epsilon
            attn = attn / attn.sum(dim=-2, keepdim=True)

            updates = torch.matmul(attn.transpose(1, 2), v)
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d),
            )
            slots = slots.view(b, self.num_slots, d)
        slots = slots.clone() + self.mlp(self.norm_mlp(slots))
        return slots


class SlotAttentionEncoder(nn.Module):
    """Slot Attention encoder producing object-centric slots and a dedicated self-state."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        c, h, w = config.observation_shape
        self.backbone = _ConvBackbone(c, config.cnn_channels, config.kernel_size)
        feature_dim = config.cnn_channels[-1]
        self.positional = _PositionalEmbedding(feature_dim, h, w)
        self.pre_slots = nn.Linear(feature_dim, config.slot_dim)
        self.slot_attention = _SlotAttention(
            num_slots=config.num_slots,
            dim=config.slot_dim,
            iters=config.slot_iterations,
            mlp_hidden_size=config.mlp_hidden_size,
            epsilon=config.epsilon,
        )
        self.self_state = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, config.slot_dim),
        )

    def forward(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Encode a batch of observations into latent slots.

        Returns:
            Mapping containing the self slot (`z_self`) and object slots (`slots`).
        """
        if observation.ndim != 4:
            raise ValueError("observation must be [batch, channels, height, width]")
        features = self.backbone(observation)
        batch, channels, height, width = features.shape
        flat = features.view(batch, channels, height * width).permute(0, 2, 1)
        flat = self.positional(flat)
        flat = self.pre_slots(flat)
        slots = self.slot_attention(flat)
        # ``torch.compile`` with CUDA graphs reuses the same output storage across
        # invocations. Downstream consumers persist ``z_self`` beyond a single
        # step, so the buffer must be materialised into fresh storage to avoid
        # accidental overwrites between iterations.
        z_self = self.self_state(features).clone()
        return {"z_self": z_self, "slots": slots}
