from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


JSDMetric = Callable[[list[torch.distributions.Distribution]], torch.Tensor]


class _RewardScaler:
    """Running mean/variance normalizer applied to individual intrinsic components."""

    def __init__(self, clamp_value: float = 5.0, eps: float = 1e-6) -> None:
        self.clamp_value = clamp_value
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.var: torch.Tensor | None = None
        self.count: torch.Tensor | None = None

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if value.numel() == 0:
            return value
        stats_value = value.detach().to(dtype=torch.float32)
        if self.mean is None or self.mean.device != stats_value.device:
            device = stats_value.device
            self.mean = torch.zeros(1, device=device)
            self.var = torch.ones(1, device=device)
            self.count = torch.tensor(self.eps, device=device)
        flat = stats_value.view(-1)
        batch_mean = flat.mean()
        batch_var = flat.var(unbiased=False)
        batch_count = flat.numel()
        with torch.no_grad():
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean = self.mean + delta * (batch_count / total)
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
            self.var = torch.clamp(m2 / total, min=self.eps)
            self.count = total
        mean = self.mean.to(dtype=value.dtype)
        var = self.var.to(dtype=value.dtype)
        normalized = (value - mean) / torch.sqrt(var + self.eps)
        return normalized.clamp(-self.clamp_value, self.clamp_value)


@dataclass
class IntrinsicRewardConfig:
    alpha_fast: float
    novelty_high: float
    anxiety_penalty: float
    safety_entropy_floor: float
    lambda_comp: float
    lambda_emp: float
    lambda_safety: float
    lambda_explore: float = 0.0
    component_clip: float = 5.0


class IntrinsicRewardGenerator:
    """
    Bundle of competence, empowerment, safety, and exploration signals for the intrinsic reward.
    """

    def __init__(
        self,
        config: IntrinsicRewardConfig,
        empowerment_estimator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        novelty_metric: JSDMetric,
    ) -> None:
        self.config = config
        self.empowerment_estimator = empowerment_estimator
        self.novelty_metric = novelty_metric
        self.ema_fast = torch.tensor(0.0)
        self._scalers = {
            "competence": _RewardScaler(clamp_value=config.component_clip),
            "empowerment": _RewardScaler(clamp_value=config.component_clip),
            "safety": _RewardScaler(clamp_value=config.component_clip),
            "explore": _RewardScaler(clamp_value=config.component_clip),
        }

    def get_novelty(self, predicted_observations: list[torch.distributions.Distribution]) -> torch.Tensor:
        """Compute epistemic novelty from an ensemble of predicted observation distributions."""
        return self.novelty_metric(predicted_observations)

    def _update_fast_ema(self, novelty: torch.Tensor) -> torch.Tensor:
        alpha = self.config.alpha_fast
        if self.ema_fast.device != novelty.device:
            self.ema_fast = self.ema_fast.to(novelty.device)
        self.ema_fast = (1 - alpha) * self.ema_fast + alpha * novelty.detach()
        return self.ema_fast

    def get_competence(self, novelty: torch.Tensor) -> torch.Tensor:
        ema_prev = self.ema_fast.clone()
        ema_current = self._update_fast_ema(novelty)
        progress = ema_prev - ema_current
        penalty = self.config.anxiety_penalty * torch.relu(novelty - self.config.novelty_high)
        return progress - penalty

    def get_safety(self, observation_entropy: torch.Tensor) -> torch.Tensor:
        deficit = torch.relu(self.config.safety_entropy_floor - observation_entropy)
        return -deficit

    def get_intrinsic_reward(
        self,
        novelty: torch.Tensor,
        observation_entropy: torch.Tensor,
        action: torch.Tensor,
        latent: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        r_comp = self.get_competence(novelty)
        r_emp = self.empowerment_estimator(action, latent)
        r_safe = self.get_safety(observation_entropy)
        r_explore = torch.clamp(novelty.detach(), min=0.0)
        batch = action.shape[0]
        if r_comp.ndim == 0:
            r_comp = r_comp.expand(batch)
        if r_safe.ndim == 0:
            r_safe = r_safe.expand(batch)
        if r_explore.ndim == 0:
            r_explore = r_explore.expand(batch)
        normalized = {
            "competence": self._scalers["competence"](r_comp),
            "empowerment": self._scalers["empowerment"](r_emp),
            "safety": self._scalers["safety"](r_safe),
            "explore": self._scalers["explore"](r_explore),
        }
        intrinsic = (
            self.config.lambda_comp * normalized["competence"]
            + self.config.lambda_emp * normalized["empowerment"]
            + self.config.lambda_safety * normalized["safety"]
            + self.config.lambda_explore * normalized["explore"]
        )
        if return_components:
            raw = {
                "competence": r_comp.detach(),
                "empowerment": r_emp.detach(),
                "safety": r_safe.detach(),
                "explore": r_explore.detach(),
            }
            return intrinsic, normalized, raw
        return intrinsic
