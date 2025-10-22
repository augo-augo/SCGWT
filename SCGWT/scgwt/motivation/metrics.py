from __future__ import annotations

import math
from typing import Iterable

import torch


def jensen_shannon_divergence(distributions: Iterable[torch.distributions.Distribution]) -> torch.Tensor:
    """
    Approximate the Jensen-Shannon divergence over discrete samples.

    The current implementation uses Monte Carlo sampling for a lightweight placeholder.
    A production system should integrate an analytic solution when available.
    """
    dists = list(distributions)
    if not dists:
        raise ValueError("At least one distribution is required")
    samples = [dist.rsample((1,)) for dist in dists]
    stacked = torch.stack(samples)  # [num_models, 1, ...]
    log_probs = torch.stack(
        [dist.log_prob(sample.squeeze(0)) for dist, sample in zip(dists, samples)]
    )
    mean_log_prob = torch.logsumexp(log_probs, dim=0) - math.log(len(dists))
    kl_terms = log_probs - mean_log_prob
    return kl_terms.mean()
