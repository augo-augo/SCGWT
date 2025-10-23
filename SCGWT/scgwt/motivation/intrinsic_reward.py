from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


JSDMetric = Callable[[list[torch.distributions.Distribution]], torch.Tensor]


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


class IntrinsicRewardGenerator:
    """
    Bundle of competence, empowerment, and safety signals for the intrinsic reward.
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        intrinsic = (
            self.config.lambda_comp * r_comp
            + self.config.lambda_emp * r_emp
            + self.config.lambda_safety * r_safe
            + self.config.lambda_explore * r_explore
        )
        if return_components:
            return intrinsic, r_comp, r_emp, r_safe, r_explore
        return intrinsic
