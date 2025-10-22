from __future__ import annotations

import math

import torch


def estimate_observation_entropy(observation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Estimate sensory entropy using a Gaussian approximation over flattened pixels.

    Args:
        observation: Tensor of shape [batch, channels, height, width].
        eps: Numerical stability constant.
    """
    if observation.ndim != 4:
        raise ValueError("observation must be [batch, channels, height, width]")
    flat = observation.reshape(observation.size(0), -1)
    variance = flat.var(dim=1, unbiased=False).clamp_min(eps)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * variance)
    return entropy
