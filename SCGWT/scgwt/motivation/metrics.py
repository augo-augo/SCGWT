from __future__ import annotations

import math
from typing import Iterable

import torch


def jensen_shannon_divergence(distributions: Iterable[torch.distributions.Distribution]) -> torch.Tensor:
    """
    Monte Carlo approximation of the Jensen-Shannon divergence between ensemble predictions.

    Each distribution contributes a single sample that is evaluated under every ensemble member to
    form a mixture. The resulting KL terms are averaged and clamped to ensure numerical stability.
    """
    dists = list(distributions)
    if not dists:
        raise ValueError("At least one distribution is required")

    num_models = len(dists)
    kl_terms: list[torch.Tensor] = []
    for idx, dist in enumerate(dists):
        sample = dist.rsample((1,)).squeeze(0)
        log_probs = torch.stack([other.log_prob(sample) for other in dists], dim=0)
        mixture_log_prob = torch.logsumexp(log_probs, dim=0) - math.log(num_models)
        kl = log_probs[idx] - mixture_log_prob
        kl_terms.append(kl)

    js = torch.stack(kl_terms).mean()
    return js.clamp(min=0.0)
