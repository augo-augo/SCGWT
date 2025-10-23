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
        logsumexp = None
        log_prob_idx = None
        for j, other in enumerate(dists):
            log_prob = other.log_prob(sample).float()
            if logsumexp is None:
                logsumexp = log_prob
            else:
                logsumexp = torch.logaddexp(logsumexp, log_prob)
            if j == idx:
                log_prob_idx = log_prob
        if log_prob_idx is None or logsumexp is None:
            raise RuntimeError("Failed to compute log probabilities for Jensen-Shannon divergence")
        mixture_log_prob = logsumexp - math.log(num_models)
        kl = log_prob_idx - mixture_log_prob
        kl_terms.append(kl)

    js = torch.stack(kl_terms).mean()
    return js.to(sample.dtype).clamp(min=0.0)
