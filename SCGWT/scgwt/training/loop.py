from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Callable

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from scgwt.agents import (
    ActorConfig,
    ActorNetwork,
    CriticConfig,
    CriticNetwork,
)
from scgwt.cognition import WorkspaceConfig, WorkspaceRouter
from scgwt.memory import EpisodicBuffer, EpisodicBufferConfig
from scgwt.motivation import (
    EmpowermentConfig,
    IntrinsicRewardConfig,
    IntrinsicRewardGenerator,
    InfoNCEEmpowermentEstimator,
    estimate_observation_entropy,
    jensen_shannon_divergence,
)
from scgwt.world_model import (
    DecoderConfig,
    DynamicsConfig,
    EncoderConfig,
    WorldModelConfig,
    WorldModelEnsemble,
)
from .buffer import RolloutBuffer


def _resolve_compile() -> Callable[[nn.Module], nn.Module]:
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return lambda module: module
    try:
        import torch._dynamo as _dynamo  # type: ignore[attr-defined]

        _dynamo.config.suppress_errors = True
    except Exception:
        pass
    def _compiler(module: nn.Module) -> nn.Module:
        try:
            return compile_fn(module)  # type: ignore[misc]
        except Exception:
            return module
    return _compiler


_maybe_compile = _resolve_compile()


class RunningMeanStd:
    """Track running mean and variance for streaming tensors."""

    def __init__(self, device: torch.device, epsilon: float = 1e-4) -> None:
        self.device = device
        self.epsilon = epsilon
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = torch.tensor(epsilon, device=device)

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        values = x.detach().to(self.device, dtype=torch.float32).reshape(-1, 1)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = values.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = torch.clamp(new_var, min=self.epsilon)
        self.count = total_count


class RewardNormalizer:
    """Keeps intrinsic rewards within a bounded scale using running statistics."""

    def __init__(self, device: torch.device, clamp_value: float = 5.0, eps: float = 1e-6) -> None:
        self.stats = RunningMeanStd(device=device)
        self.clamp_value = clamp_value
        self.eps = eps
        self.device = device

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        reward_fp32 = reward.to(dtype=torch.float32)
        self.stats.update(reward_fp32)
        denom = torch.sqrt(self.stats.var + self.eps)
        normalized = (reward_fp32 - self.stats.mean) / denom
        normalized = torch.clamp(normalized, -self.clamp_value, self.clamp_value)
        return normalized.to(dtype=reward.dtype)


@dataclass
class StepResult:
    action: torch.Tensor
    intrinsic_reward: torch.Tensor
    novelty: torch.Tensor
    observation_entropy: torch.Tensor
    slot_scores: torch.Tensor
    reward_components: dict[str, torch.Tensor] | None = None
    raw_reward_components: dict[str, torch.Tensor] | None = None
    training_loss: float | None = None
    training_metrics: dict[str, float] | None = None


@dataclass
class TrainingConfig:
    encoder: EncoderConfig
    decoder: DecoderConfig
    dynamics: DynamicsConfig
    world_model_ensemble: int
    workspace: WorkspaceConfig
    reward: IntrinsicRewardConfig
    empowerment: EmpowermentConfig
    episodic_memory: EpisodicBufferConfig
    rollout_capacity: int = 1024
    batch_size: int = 32
    optimizer_lr: float = 1e-3
    optimizer_empowerment_weight: float = 0.1
    actor: ActorConfig = field(
        default_factory=lambda: ActorConfig(latent_dim=0, action_dim=0)
    )
    critic: CriticConfig = field(default_factory=lambda: CriticConfig(latent_dim=0))
    dream_horizon: int = 5
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    critic_coef: float = 0.5
    world_model_coef: float = 1.0
    self_state_dim: int = 0
    device: str = "cpu"


class TrainingLoop:
    """High-level container wiring the major subsystems together."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.progress_momentum = config.workspace.progress_momentum
        self.action_cost_scale = config.workspace.action_cost_scale
        wm_config = WorldModelConfig(
            encoder=config.encoder,
            decoder=config.decoder,
            dynamics=config.dynamics,
            ensemble_size=config.world_model_ensemble,
        )
        world_model = WorldModelEnsemble(wm_config).to(self.device)
        if self.device.type == "cuda":
            self.world_model = _maybe_compile(world_model)
        else:
            self.world_model = world_model
        self.workspace = WorkspaceRouter(config.workspace)
        self.memory = EpisodicBuffer(config.episodic_memory)
        empowerment = InfoNCEEmpowermentEstimator(config.empowerment).to(self.device)
        if self.device.type == "cuda":
            self.empowerment = _maybe_compile(empowerment)
        else:
            self.empowerment = empowerment
        self.reward = IntrinsicRewardGenerator(
            config.reward,
            empowerment_estimator=self.empowerment,
            novelty_metric=jensen_shannon_divergence,
        )
        self.reward_normalizer = RewardNormalizer(device=self.device)
        # Policy dimensions derived from encoder/workspace layout.
        slot_dim = config.encoder.slot_dim
        policy_feature_dim = (
            slot_dim
            + slot_dim * config.workspace.broadcast_slots
            + config.episodic_memory.key_dim
        )
        actor_net = ActorNetwork(
            ActorConfig(
                latent_dim=policy_feature_dim,
                action_dim=config.dynamics.action_dim,
                hidden_dim=config.actor.hidden_dim,
                num_layers=config.actor.num_layers,
                dropout=config.actor.dropout,
            )
        ).to(self.device)
        critic_net = CriticNetwork(
            CriticConfig(
                latent_dim=policy_feature_dim,
                hidden_dim=config.critic.hidden_dim,
                num_layers=config.critic.num_layers,
                dropout=config.critic.dropout,
            )
        ).to(self.device)
        if self.device.type == "cuda":
            self.actor = _maybe_compile(actor_net)
            self.critic = _maybe_compile(critic_net)
        else:
            self.actor = actor_net
            self.critic = critic_net

        self.self_state_dim = config.self_state_dim
        if self.self_state_dim > 0:
            self.self_state_encoder = nn.Linear(
                self.self_state_dim, slot_dim, bias=False
            ).to(self.device)
            self.self_state_predictor = nn.Linear(
                slot_dim, self.self_state_dim
            ).to(self.device)
        else:
            self.self_state_encoder = None
            self.self_state_predictor = None

        self._slot_baseline: torch.Tensor | None = None
        self._ucb_mean: torch.Tensor | None = None
        self._ucb_counts: torch.Tensor | None = None
        self._step_count: int = 0
        self._novelty_trace: torch.Tensor | None = None
        self._latest_self_state: torch.Tensor | None = None

        self.rollout_buffer = RolloutBuffer(capacity=config.rollout_capacity)
        self.batch_size = config.batch_size
        params: list[torch.nn.Parameter] = []
        params.extend(self.world_model.parameters())
        params.extend(self.empowerment.parameters())
        params.extend(self.actor.parameters())
        params.extend(self.critic.parameters())
        if self.self_state_encoder is not None:
            params.extend(self.self_state_encoder.parameters())
        if self.self_state_predictor is not None:
            params.extend(self.self_state_predictor.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.optimizer_lr)
        self.optimizer_empowerment_weight = config.optimizer_empowerment_weight
        self.autocast_enabled = self.device.type == "cuda"
        self.grad_scaler = GradScaler(enabled=self.autocast_enabled)

    def _autocast_ctx(self):
        if self.autocast_enabled:
            return autocast()
        return nullcontext()

    def step(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        next_observation: torch.Tensor | None = None,
        self_state: torch.Tensor | None = None,
        train: bool = False,
    ) -> StepResult:
        """
        Encode an observation, optionally sample an action, compute intrinsic reward,
        and record the transition. When `train=True` and `next_observation` is supplied,
        the optimizer performs a Stable Dreaming update once enough rollouts are stored.
        """
        observation = observation.to(self.device, non_blocking=True)
        batch = observation.size(0)
        state_tensor: torch.Tensor | None
        if self.self_state_dim > 0:
            if self_state is None:
                state_tensor = torch.zeros(
                    batch,
                    self.self_state_dim,
                    device=self.device,
                    dtype=observation.dtype,
                )
            else:
                state_tensor = self_state.to(self.device, non_blocking=True)
                if state_tensor.ndim == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                if state_tensor.size(0) != batch:
                    if state_tensor.size(0) == 1:
                        state_tensor = state_tensor.expand(batch, -1)
                    else:
                        raise ValueError("self_state batch dimension mismatch")
        else:
            state_tensor = None

        if state_tensor is not None:
            self._latest_self_state = state_tensor.detach()

        with torch.no_grad():
            with self._autocast_ctx():
                latents = self.world_model(observation)
                memory_context = self._get_memory_context(latents["z_self"])
                if action is not None:
                    action_for_routing = action.to(self.device, non_blocking=True)
                else:
                    action_for_routing = torch.zeros(
                        batch, self.config.dynamics.action_dim, device=self.device
                    )
                (
                    broadcast,
                    scores,
                    slot_novelty,
                    slot_progress,
                    slot_cost,
                ) = self._route_slots(
                    latents["slots"],
                    latents["z_self"],
                    action_for_routing,
                    state_tensor,
                    update_stats=True,
                )
                features = self._assemble_features(latents["z_self"], broadcast, memory_context)
                if action is None:
                    action_dist = self.actor(features)
                    action = action_dist.rsample()
                    (
                        broadcast,
                        scores,
                        slot_novelty,
                        slot_progress,
                        slot_cost,
                    ) = self._route_slots(
                        latents["slots"],
                        latents["z_self"],
                        action,
                        state_tensor,
                        update_stats=False,
                    )
                    features = self._assemble_features(latents["z_self"], broadcast, memory_context)
                else:
                    action = action.to(self.device, non_blocking=True)

                latent_state = broadcast.mean(dim=1)
                predictions = self.world_model.predict_next_latents(latent_state, action)
                decoded = self.world_model.decode_predictions(predictions)
                novelty = self.reward.get_novelty(decoded).to(self.device)
                observation_entropy = estimate_observation_entropy(observation)
                intrinsic_raw, norm_components, raw_components = self.reward.get_intrinsic_reward(
                    novelty, observation_entropy, action, latent_state, return_components=True
                )

        intrinsic = self.reward_normalizer(intrinsic_raw)
        reward_components = {key: value.detach() for key, value in norm_components.items()}
        raw_reward_components = {key: value.detach() for key, value in raw_components.items()}
        self._write_memory(latents["z_self"], broadcast)

        if train and next_observation is not None:
            self.store_transition(
                observation=observation,
                action=action,
                next_observation=next_observation,
                self_state=state_tensor,
            )
        train_loss: float | None = None
        training_metrics: dict[str, float] | None = None

        return StepResult(
            action=action.detach(),
            intrinsic_reward=intrinsic.detach(),
            novelty=slot_novelty.detach(),
            observation_entropy=observation_entropy.detach(),
            slot_scores=scores.detach(),
            reward_components=reward_components,
            raw_reward_components=raw_reward_components,
            training_loss=train_loss,
            training_metrics=training_metrics,
        )

    def store_transition(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        self_state: torch.Tensor | None = None,
    ) -> None:
        obs_cpu = observation.detach().to("cpu", non_blocking=True).contiguous()
        act_cpu = action.detach().to("cpu", non_blocking=True).contiguous()
        next_cpu = next_observation.detach().to("cpu", non_blocking=True).contiguous()
        state_cpu = (
            self_state.detach().to("cpu", non_blocking=True).contiguous()
            if self_state is not None
            else None
        )
        if torch.cuda.is_available():
            obs_cpu = obs_cpu.pin_memory()
            act_cpu = act_cpu.pin_memory()
            next_cpu = next_cpu.pin_memory()
            if state_cpu is not None:
                state_cpu = state_cpu.pin_memory()

        batch_items = obs_cpu.shape[0]
        for idx in range(batch_items):
            self.rollout_buffer.push(
                obs_cpu[idx],
                act_cpu[idx],
                next_cpu[idx],
                state_cpu[idx] if state_cpu is not None else None,
            )

    def _route_slots(
        self,
        slot_values: torch.Tensor,
        z_self: torch.Tensor,
        action: torch.Tensor,
        self_state: torch.Tensor | None,
        update_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slot_novelty = slot_values.var(dim=-1, unbiased=False)
        if self._slot_baseline is None:
            self._slot_baseline = slot_values.mean(dim=0).detach().cpu()

        if self._novelty_trace is None:
            slot_progress = torch.zeros_like(slot_novelty)
            if update_stats:
                self._novelty_trace = slot_novelty.detach().cpu()
        else:
            prev_trace = self._novelty_trace.to(self.device)
            slot_progress = prev_trace - slot_novelty
            if update_stats:
                updated_trace = (
                    (1 - self.progress_momentum) * self._novelty_trace
                    + self.progress_momentum * slot_novelty.detach().cpu()
                )
                self._novelty_trace = updated_trace

        if update_stats:
            baseline_update = slot_values.mean(dim=0).detach().cpu()
            self._slot_baseline = (
                (1 - self.progress_momentum) * self._slot_baseline
                + self.progress_momentum * baseline_update
            )

        action_cost = torch.norm(action, dim=-1, keepdim=True) * self.action_cost_scale
        slot_cost = action_cost.expand(-1, slot_values.size(1))

        slot_norm = torch.nn.functional.normalize(slot_values, dim=-1)
        z_self_norm = torch.nn.functional.normalize(z_self, dim=-1)
        self_similarity = (
            slot_norm * z_self_norm.unsqueeze(1)
        ).sum(dim=-1).clamp(min=0.0)

        state_similarity = torch.zeros_like(self_similarity)
        if (
            self_state is not None
            and self.self_state_encoder is not None
            and self.self_state_dim > 0
        ):
            projected_state = self.self_state_encoder(
                self_state.to(self.device, non_blocking=True)
            )
            projected_state = torch.nn.functional.normalize(projected_state, dim=-1)
            state_similarity = (
                slot_norm * projected_state.unsqueeze(1)
            ).sum(dim=-1).clamp(min=0.0)

        self_mask = torch.clamp(self_similarity + state_similarity, min=0.0)

        batch_mean = slot_novelty.mean(dim=0).detach().cpu()
        if self._ucb_mean is None:
            self._ucb_mean = batch_mean.clone()
            self._ucb_counts = torch.ones_like(batch_mean)
        else:
            assert self._ucb_counts is not None
            self._ucb_counts += 1
            self._ucb_mean += (batch_mean - self._ucb_mean) / self._ucb_counts
        assert self._ucb_mean is not None and self._ucb_counts is not None
        self._step_count += 1
        ucb_bonus = (
            self._ucb_mean.to(self.device)
            + self.config.workspace.ucb_beta
            * torch.sqrt(
                torch.log1p(torch.tensor(float(self._step_count), device=self.device))
                / self._ucb_counts.to(self.device)
            )
        )
        ucb = ucb_bonus.unsqueeze(0).expand(slot_values.size(0), -1)

        scores = self.workspace.score_slots(
            novelty=slot_novelty,
            progress=slot_progress,
            ucb=ucb,
            cost=slot_cost,
            self_mask=self_mask,
        )
        broadcast = self.workspace.broadcast(slot_values, scores=scores)
        return broadcast, scores, slot_novelty, slot_progress, slot_cost

    def _get_memory_context(self, keys: torch.Tensor) -> torch.Tensor:
        batch = keys.shape[0]
        if len(self.memory) == 0:
            return torch.zeros(batch, self.memory.config.key_dim, device=self.device)
        _, values = self.memory.read(keys)
        context = values[:, 0, :].to(self.device)
        return context

    def _write_memory(self, z_self: torch.Tensor, slots: torch.Tensor) -> None:
        key = z_self.detach().cpu()
        value = slots.mean(dim=1).detach().cpu()
        self.memory.write(key, value)

    def _assemble_features(
        self,
        z_self: torch.Tensor,
        broadcast: torch.Tensor,
        memory_context: torch.Tensor,
    ) -> torch.Tensor:
        broadcast_flat = broadcast.flatten(start_dim=1)
        return torch.cat([z_self, broadcast_flat, memory_context], dim=-1)

    def _optimize(self) -> dict[str, float] | None:
        if len(self.rollout_buffer) < self.batch_size:
            return None
        observations, actions, next_observations, self_states = self.rollout_buffer.sample(
            self.batch_size
        )
        observations = observations.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        next_observations = next_observations.to(self.device, non_blocking=True)
        if self_states is not None:
            self_states = self_states.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast_ctx():
            latents = self.world_model(observations)
            memory_context = self._get_memory_context(latents["z_self"])
            broadcast, _, _, _, _ = self._route_slots(
                latents["slots"],
                latents["z_self"],
                actions,
                self_states,
                update_stats=False,
            )
            latent_state = broadcast.mean(dim=1)
            features = self._assemble_features(latents["z_self"], broadcast, memory_context)

            predictions = self.world_model.predict_next_latents(latent_state, actions)
            decoded = self.world_model.decode_predictions(predictions, use_frozen=False)
            log_likelihoods = torch.stack(
                [dist.log_prob(next_observations).mean() for dist in decoded]
            )
            world_model_loss = -log_likelihoods.mean()

            encoded_next = self.world_model(next_observations)
            predicted_latent = torch.stack(predictions).mean(dim=0)
            target_latent = encoded_next["slots"].mean(dim=1)
            latent_alignment = torch.nn.functional.mse_loss(predicted_latent, target_latent)
            world_model_loss = (
                world_model_loss + 0.1 * latent_alignment
            ) * self.config.world_model_coef

            self_state_loss = torch.tensor(0.0, device=self.device)
            if (
                self_states is not None
                and self.self_state_dim > 0
                and self.self_state_predictor is not None
            ):
                predicted_state = self.self_state_predictor(latents["z_self"])
                self_state_loss = torch.nn.functional.mse_loss(predicted_state, self_states)
                self_state_loss = self.config.workspace.self_bias * self_state_loss

            dream_loss, actor_loss, critic_loss, dream_metrics = self._stable_dreaming(latents)
            total_loss = world_model_loss + actor_loss + critic_loss + dream_loss + self_state_loss

        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        metrics: dict[str, float] = {
            "train/total_loss": float(total_loss.detach().cpu().item()),
            "train/world_model_loss": float(world_model_loss.detach().cpu().item()),
            "train/actor_loss": float(actor_loss.detach().cpu().item()),
            "train/critic_loss": float(critic_loss.detach().cpu().item()),
            "train/dream_loss_empowerment": float(dream_loss.detach().cpu().item()),
            "train/self_state_loss": float(self_state_loss.detach().cpu().item()),
        }
        for key, value in dream_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = float(value.detach().cpu().item())
            else:
                metrics[key] = float(value)
        return metrics

    def _stable_dreaming(
        self, latents: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        current_latents = latents
        memory_context = self._get_memory_context(current_latents["z_self"])
        actor_losses = []
        critic_losses = []
        entropies = []
        rewards = []
        values = []
        log_probs = []
        competence_terms = []
        empowerment_terms = []
        safety_terms = []
        intrinsic_terms = []
        explore_terms = []
        raw_explore_terms = []

        dream_actions = []
        for step in range(self.config.dream_horizon):
            broadcast, _, _, _, _ = self._route_slots(
                current_latents["slots"],
                current_latents["z_self"],
                torch.zeros(
                    current_latents["slots"].size(0),
                    self.config.dynamics.action_dim,
                    device=self.device,
                ),
                None,
                update_stats=False,
            )
            features = self._assemble_features(
                current_latents["z_self"], broadcast, memory_context
            )
            action_dist = self.actor(features)
            dream_action = action_dist.rsample()
            dream_log_prob = action_dist.log_prob(dream_action)
            dream_entropy = action_dist.entropy()
            dream_actions.append(dream_action)

            latent_state = broadcast.mean(dim=1)
            predictions = self.world_model.predict_next_latents(latent_state, dream_action)
            decoded = self.world_model.decode_predictions(predictions, use_frozen=False)
            novelty = self.reward.get_novelty(decoded)
            predicted_obs = decoded[0].rsample()
            observation_entropy = estimate_observation_entropy(predicted_obs)
            dream_reward, norm_components, raw_components = self.reward.get_intrinsic_reward(
                novelty,
                observation_entropy,
                dream_action,
                latent_state,
                return_components=True,
            )
            dream_comp = norm_components["competence"]
            dream_emp = norm_components["empowerment"]
            dream_safe = norm_components["safety"]
            dream_explore = norm_components["explore"]
            raw_explore = raw_components["explore"]
            competence_terms.append(dream_comp.detach().mean())
            empowerment_terms.append(dream_emp.detach().mean())
            safety_terms.append(dream_safe.detach().mean())
            intrinsic_terms.append(dream_reward.detach().mean())
            explore_terms.append(dream_explore.detach().mean())
            raw_explore_terms.append(raw_explore.detach().mean())

            normalized_reward = self.reward_normalizer(dream_reward)

            critic_value = self.critic(features)
            values.append(critic_value)
            rewards.append(normalized_reward)
            log_probs.append(dream_log_prob)
            entropies.append(dream_entropy)

            current_latents = self.world_model(predicted_obs)
            memory_context = self._get_memory_context(current_latents["z_self"])

        final_broadcast, _, _, _, _ = self._route_slots(
            current_latents["slots"],
            current_latents["z_self"],
            torch.zeros(
                current_latents["slots"].size(0),
                self.config.dynamics.action_dim,
                device=self.device,
            ),
            None,
            update_stats=False,
        )
        final_features = self._assemble_features(
            current_latents["z_self"], final_broadcast, memory_context
        )
        next_value = self.critic(final_features)

        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        advantages, returns = self._compute_gae(
            rewards_tensor, values_tensor, next_value
        )
        actor_loss = -(
            (advantages.detach() * log_probs_tensor).mean()
            + self.config.entropy_coef * entropies_tensor.mean()
        )
        critic_loss = (
            self.config.critic_coef * 0.5 * (returns.detach() - values_tensor).pow(2).mean()
        )

        empowerment_term = self.empowerment(
            dream_actions[-1].detach(), final_broadcast.mean(dim=1).detach()
        ).mean()
        dream_loss = -self.optimizer_empowerment_weight * empowerment_term

        intrinsic_stack = torch.stack(intrinsic_terms)
        competence_stack = torch.stack(competence_terms)
        empowerment_stack = torch.stack(empowerment_terms)
        safety_stack = torch.stack(safety_terms)
        explore_stack = torch.stack(explore_terms)
        raw_explore_stack = torch.stack(raw_explore_terms)

        dreaming_metrics = {
            "dream/intrinsic_reward": intrinsic_stack.mean().detach(),
            "dream/competence": competence_stack.mean().detach(),
            "dream/empowerment": empowerment_stack.mean().detach(),
            "dream/safety": safety_stack.mean().detach(),
            "dream/policy_entropy": entropies_tensor.mean().detach(),
            "dream/explore": explore_stack.mean().detach(),
            "dream/explore_min": explore_stack.min().detach(),
            "dream/explore_max": explore_stack.max().detach(),
            "dream/explore_raw": raw_explore_stack.mean().detach(),
            "dream/explore_raw_min": raw_explore_stack.min().detach(),
            "dream/explore_raw_max": raw_explore_stack.max().detach(),
        }

        return dream_loss, actor_loss, critic_loss, dreaming_metrics

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon, batch = rewards.shape
        values_ext = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(batch, device=self.device)
        for t in reversed(range(horizon)):
            delta = (
                rewards[t]
                + self.config.discount_gamma * values_ext[t + 1]
                - values_ext[t]
            )
            last_advantage = delta + (
                self.config.discount_gamma
                * self.config.gae_lambda
                * last_advantage
            )
            advantages[t] = last_advantage
        returns = advantages + values
        return advantages, returns





