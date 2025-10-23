from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import nullcontext
import warnings
from typing import Dict, List, Optional, Tuple, cast # Added Optional, Tuple, cast

import torch
from torch import nn
# Use the newer torch.amp namespace
from torch.amp import GradScaler, autocast as torch_autocast # Renamed to avoid conflict

try:
    from torch._dynamo.eval_frame import OptimizedModule as _OptimizedModuleType
except (ImportError, AttributeError):
    _OptimizedModuleType = ()

try:
    from torch.compiler import cudagraph_mark_step_begin
except (ImportError, AttributeError):
    try:
        from torch._inductor.utils import cudagraph_mark_step_begin
    except (ImportError, AttributeError):
        try:
            from torch._dynamo import mark_step_begin as cudagraph_mark_step_begin
        except (ImportError, AttributeError):
            warnings.warn(
                "CUDAGraph safety helpers are unavailable; compiled runs will not "
                "be protected by cudagraph_mark_step_begin.",
                RuntimeWarning,
            )
            cudagraph_mark_step_begin = None
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
        values = x.detach().to(device=self.device, dtype=torch.float32).reshape(-1, 1)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = values.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @staticmethod
    def _update_running_moments(
            mean: torch.Tensor, var: torch.Tensor, count: torch.Tensor,
            batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float,
            epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = batch_mean - mean
        total_count = count + batch_count
        new_mean = mean + delta * (batch_count / total_count)
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * count * batch_count / total_count
        new_var = m2 / total_count
        new_var = torch.clamp(new_var, min=epsilon)
        return new_mean, new_var, total_count

    def _update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        # Use the static method for the update logic
        new_mean, new_var, new_count = self._update_running_moments(
            self.mean, self.var, self.count,
            batch_mean, batch_var, batch_count,
            self.epsilon
        )
        self.mean = new_mean
        self.var = new_var
        self.count = new_count


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
        mean = self.stats.mean
        var = self.stats.var
        denom = torch.sqrt(var + self.eps)
        normalized = (reward_fp32 - mean) / denom
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
    precision: str = "float32"
    log_every_steps: int = 50
    log_images: bool = True
    channels_last: bool = False
    compile_model: bool = False


def _is_compiled_artifact(original: nn.Module, candidate: nn.Module) -> bool:
    """Best-effort detection that ``torch.compile`` wrapped the module."""
    if _OptimizedModuleType and isinstance(candidate, _OptimizedModuleType):
        return True
    if candidate is original:
        return False
    # Check common attributes added by compile in various versions
    return hasattr(candidate, "_orig_mod") or \
           hasattr(candidate, "__compiled_fn__") or \
           hasattr(candidate, "graph_module") or \
           hasattr(candidate, "original_module")


class TrainingLoop:
    """High-level container wiring the major subsystems together."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        precision_key = config.precision.lower()

        # Determine autocast dtype
        if precision_key in {"bf16", "bfloat16"}:
            self.autocast_dtype = torch.bfloat16
        elif precision_key in {"fp16", "float16", "half"}:
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.float32

        # Check if autocast is enabled (CUDA with non-float32 or CPU with bfloat16)
        self.autocast_enabled = (self.device.type == 'cuda' and self.autocast_dtype != torch.float32) or \
                                (self.device.type == 'cpu' and self.autocast_dtype == torch.bfloat16)

        self.progress_momentum = config.workspace.progress_momentum
        self.action_cost_scale = config.workspace.action_cost_scale

        wm_config = WorldModelConfig(
            encoder=config.encoder,
            decoder=config.decoder,
            dynamics=config.dynamics,
            ensemble_size=config.world_model_ensemble,
        )
        self.world_model = WorldModelEnsemble(wm_config).to(self.device)

        self._slot_dim = config.encoder.slot_dim
        self._num_slots = config.encoder.num_slots
        self._latent_dim = config.dynamics.latent_dim
        self._ensemble_size = config.world_model_ensemble

        first_param = next(self.world_model.parameters(), None)
        self._world_model_param_dtype = (
            first_param.dtype if first_param is not None else torch.float32
        )

        self._latent_buffers: list[dict[str, torch.Tensor]] = []
        self._latent_buffer_index: int = 0
        self._prediction_buffer: list[torch.Tensor] | None = None

        self.workspace = WorkspaceRouter(config.workspace)
        self.memory = EpisodicBuffer(config.episodic_memory)
        self.empowerment = InfoNCEEmpowermentEstimator(config.empowerment).to(self.device)
        self.reward = IntrinsicRewardGenerator(
            config.reward,
            empowerment_estimator=self.empowerment,
            novelty_metric=jensen_shannon_divergence,
        )
        self.reward_normalizer = RewardNormalizer(device=self.device)

        policy_feature_dim = (
            self._slot_dim
            + self._slot_dim * config.workspace.broadcast_slots
            + config.episodic_memory.key_dim
        )
        self.actor = ActorNetwork(
            ActorConfig(
                latent_dim=policy_feature_dim,
                action_dim=config.dynamics.action_dim,
                hidden_dim=config.actor.hidden_dim,
                num_layers=config.actor.num_layers,
                dropout=config.actor.dropout,
            )
        ).to(self.device)
        self.critic = CriticNetwork(
            CriticConfig(
                latent_dim=policy_feature_dim,
                hidden_dim=config.critic.hidden_dim,
                num_layers=config.critic.num_layers,
                dropout=config.critic.dropout,
            )
        ).to(self.device)

        self.self_state_dim = config.self_state_dim
        if self.self_state_dim > 0:
            self.self_state_encoder = nn.Linear(
                self.self_state_dim, self._slot_dim, bias=False
            ).to(self.device)
            self.self_state_predictor = nn.Linear(
                self._slot_dim, self.self_state_dim
            ).to(self.device)
        else:
            self.self_state_encoder = None
            self.self_state_predictor = None

        if config.channels_last and self.device.type == "cuda":
            modules_to_format = [self.world_model, self.actor, self.critic]
            if self.self_state_encoder: modules_to_format.append(self.self_state_encoder)
            if self.self_state_predictor: modules_to_format.append(self.self_state_predictor)
            for module in modules_to_format:
                try:
                    module.to(memory_format=torch.channels_last)
                    for parameter in module.parameters():
                        if parameter.ndim >= 4:
                            parameter.data = parameter.data.contiguous(memory_format=torch.channels_last)
                except Exception as e:
                    print(f"Warning: Failed to apply channels_last to {module.__class__.__name__}: {e}")


        compiled_world_model = False
        compiled_any_module = False
        if config.compile_model:
            modules_to_compile = {"world_model": self.world_model,
                                  "actor": self.actor,
                                  "critic": self.critic}
            if self.self_state_predictor:
                 modules_to_compile["self_state_predictor"] = self.self_state_predictor

            for name, original_module in modules_to_compile.items():
                try:
                    compiled_module = torch.compile(original_module, mode="max-autotune", fullgraph=False)
                    was_compiled = _is_compiled_artifact(original_module, compiled_module)
                    print(f"Compiling {name}... Success: {was_compiled}")
                    setattr(self, name, compiled_module)
                    if was_compiled:
                        compiled_any_module = True
                        if name == "world_model":
                            compiled_world_model = True
                except Exception as e:
                    print(f"Failed to compile {name}: {e}")
                    setattr(self, name, original_module)

        self._compiled_runtime = compiled_any_module
        # Use buffers ONLY if the world model compiled successfully
        self._use_output_buffers = compiled_world_model

        self._slot_baseline: torch.Tensor | None = None
        self._ucb_mean: torch.Tensor | None = None
        self._ucb_counts: torch.Tensor | None = None
        self._step_count: int = 0
        self._novelty_trace: torch.Tensor | None = None
        self._latest_self_state: torch.Tensor | None = None

        self.rollout_buffer = RolloutBuffer(capacity=config.rollout_capacity)
        self.batch_size = config.batch_size

        params = [p for p in self.world_model.parameters() if p.requires_grad] + \
                 [p for p in self.empowerment.parameters() if p.requires_grad] + \
                 [p for p in self.actor.parameters() if p.requires_grad] + \
                 [p for p in self.critic.parameters() if p.requires_grad]
        if self.self_state_encoder:
            params.extend(p for p in self.self_state_encoder.parameters() if p.requires_grad)
        if self.self_state_predictor:
            params.extend(p for p in self.self_state_predictor.parameters() if p.requires_grad)

        try:
            self.optimizer = torch.optim.AdamW(params, lr=config.optimizer_lr, fused=True)
            print("Using Fused AdamW optimizer.")
        except (TypeError, ValueError, RuntimeError):
            print("Fused AdamW not available, falling back to standard AdamW.")
            self.optimizer = torch.optim.AdamW(params, lr=config.optimizer_lr)

        self.optimizer_empowerment_weight = config.optimizer_empowerment_weight

        # Updated GradScaler initialization (handles device type internally)
        self.grad_scaler = GradScaler(
            enabled=self.autocast_enabled and self.autocast_dtype == torch.float16
        )
        print(f"GradScaler enabled: {self.grad_scaler.is_enabled()}")

    def _autocast_ctx(self):
        """Returns the appropriate autocast context manager."""
        if self.autocast_enabled:
            # Use torch.amp.autocast, providing device type and dtype
            return torch_autocast(device_type=self.device.type, dtype=self.autocast_dtype)
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
        Encode observation, select action, compute reward, store transition, and optionally train.
        """
        observation = observation.to(self.device, non_blocking=True)
        batch = observation.size(0)

        state_tensor: Optional[torch.Tensor] = None
        if self.self_state_dim > 0:
            if self_state is None:
                state_tensor = torch.zeros(
                    batch, self.self_state_dim, device=self.device, dtype=observation.dtype
                )
            else:
                state_tensor = self_state.to(self.device, non_blocking=True)
                if state_tensor.ndim == 1: state_tensor = state_tensor.unsqueeze(0)
                if state_tensor.size(0) == 1 and batch > 1:
                     state_tensor = state_tensor.expand(batch, -1)
                elif state_tensor.size(0) != batch:
                     raise ValueError(f"self_state batch dim mismatch: got {state_tensor.size(0)}, expected {batch}")
            self._latest_self_state = state_tensor.detach()

        with torch.no_grad():
            with self._autocast_ctx():
                self._graph_mark()

                latent_buffer = self._next_latent_buffer(batch) # Returns None if not using buffers
                latents = self.world_model(observation, output_buffer=latent_buffer)
                # No cloning needed, use latents directly

                memory_context = self._get_memory_context(latents["z_self"])

                action_for_routing: torch.Tensor
                if action is not None:
                    action_for_routing = action.to(self.device, non_blocking=True)
                else:
                    action_for_routing = torch.zeros(
                        batch, self.config.dynamics.action_dim, device=self.device, dtype=latents["z_self"].dtype
                    )

                (
                    broadcast, scores, slot_novelty, slot_progress, slot_cost,
                ) = self._route_slots(
                    latents["slots"], latents["z_self"], action_for_routing,
                    state_tensor, update_stats=True
                )

                features = self._assemble_features(latents["z_self"], broadcast, memory_context)

                if action is None:
                    self._graph_mark()
                    action_dist = self.actor(features)
                    action = action_dist.rsample()
                    (
                         broadcast, scores, slot_novelty, _, _,
                    ) = self._route_slots(
                         latents["slots"], latents["z_self"], action,
                         state_tensor, update_stats=False
                    )
                    features = self._assemble_features(latents["z_self"], broadcast, memory_context)
                else:
                     action = action.to(self.device, non_blocking=True)

                latent_state = broadcast.mean(dim=1)
                self._graph_mark()
                prediction_buffer = self._prepare_prediction_buffer(batch, latent_state.dtype) # Returns None if not using buffers
                predictions = self.world_model.predict_next_latents(
                    latent_state, action, output_buffer=prediction_buffer
                )
                # No cloning needed

                self._graph_mark()
                decoded = self.world_model.decode_predictions(predictions, use_frozen=True)
                novelty = self.reward.get_novelty(decoded).to(self.device)
                observation_entropy = estimate_observation_entropy(observation)
                intrinsic_raw, norm_components, raw_components = self.reward.get_intrinsic_reward(
                    novelty, observation_entropy, action, latent_state, return_components=True
                )
                del decoded
                del predictions

        intrinsic = self.reward_normalizer(intrinsic_raw)
        reward_components_detached = {k: v.detach() for k, v in norm_components.items()} if norm_components else None
        raw_components_detached = {k: v.detach() for k, v in raw_components.items()} if raw_components else None

        self._write_memory(latents["z_self"], broadcast)

        train_loss: Optional[float] = None
        training_metrics: Optional[Dict[str, float]] = None
        if train and next_observation is not None:
            obs_cpu = observation.detach().cpu().contiguous()
            act_cpu = action.detach().cpu().contiguous()
            next_cpu = next_observation.to(self.device, non_blocking=True).detach().cpu().contiguous() # Ensure next_obs on CPU too
            state_cpu = state_tensor.detach().cpu().contiguous() if state_tensor is not None else None

            if torch.cuda.is_available():
                obs_cpu, act_cpu, next_cpu = obs_cpu.pin_memory(), act_cpu.pin_memory(), next_cpu.pin_memory()
                if state_cpu is not None: state_cpu = state_cpu.pin_memory()

            batch_items = obs_cpu.shape[0]
            for idx in range(batch_items):
                 current_state_cpu = state_cpu[idx] if state_cpu is not None else None
                 self.rollout_buffer.push(obs_cpu[idx], act_cpu[idx], next_cpu[idx], current_state_cpu)

            training_metrics = self._optimize()
            if training_metrics is not None:
                train_loss = training_metrics.get("train/total_loss")

        return StepResult(
            action=action.detach(),
            intrinsic_reward=intrinsic.detach(),
            novelty=slot_novelty.detach(),
            observation_entropy=observation_entropy.detach(),
            slot_scores=scores.detach(),
            reward_components=reward_components_detached,
            raw_reward_components=raw_components_detached,
            training_loss=train_loss,
            training_metrics=training_metrics,
        )

    # --- Rest of the methods (_route_slots, _get_memory_context, etc.) remain largely the same ---
    # ... (Keep the existing implementations for the rest of the class,
    #      including _optimize, _stable_dreaming, _compute_gae,
    #      _latent_output_dtype, _create_latent_buffer, _ensure_latent_buffers,
    #      _next_latent_buffer, _prepare_prediction_buffer, _graph_mark) ...

    # Make sure these buffer methods correctly use self._use_output_buffers
    def _ensure_latent_buffers(self, batch: int, dtype: torch.dtype) -> None:
        if not self._use_output_buffers: # Check flag
            return
        needs_rebuild = len(self._latent_buffers) != 2
        if not needs_rebuild:
            buffer = self._latent_buffers[0]
            slots = buffer["slots"]
            z_self = buffer["z_self"]
            if (slots.shape[0] != batch or slots.dtype != dtype or slots.device != self.device or
                z_self.shape[0] != batch or z_self.dtype != dtype or z_self.device != self.device):
                needs_rebuild = True
        if needs_rebuild:
            # print(f"Rebuilding latent buffers for batch={batch}, dtype={dtype}, device={self.device}") # Optional: for debugging
            self._latent_buffers = [
                self._create_latent_buffer(batch, dtype),
                self._create_latent_buffer(batch, dtype),
            ]
            self._latent_buffer_index = 0

    def _next_latent_buffer(self, batch: int) -> Optional[Dict[str, torch.Tensor]]:
        if not self._use_output_buffers: # Check flag
            return None
        dtype = self._latent_output_dtype()
        self._ensure_latent_buffers(batch, dtype)
        buffer = self._latent_buffers[self._latent_buffer_index]
        self._latent_buffer_index = (self._latent_buffer_index + 1) % len(self._latent_buffers)
        return buffer

    def _prepare_prediction_buffer(self, batch: int, dtype: torch.dtype) -> Optional[List[torch.Tensor]]:
        if not self._use_output_buffers or self._ensemble_size == 0: # Check flag
             return None

        needs_rebuild = self._prediction_buffer is None or len(self._prediction_buffer) != self._ensemble_size
        if not needs_rebuild:
             sample = self._prediction_buffer[0]
             if (sample.shape[0] != batch or sample.dtype != dtype or sample.device != self.device):
                 needs_rebuild = True

        if needs_rebuild:
            # print(f"Rebuilding prediction buffer for batch={batch}, dtype={dtype}, device={self.device}") # Optional: for debugging
            self._prediction_buffer = [
                torch.empty((batch, self._latent_dim), device=self.device, dtype=dtype)
                for _ in range(self._ensemble_size)
            ]
        # Ensure correct return type hint compliance
        return cast(List[torch.Tensor], self._prediction_buffer)


    # --- Keep the _route_slots, _get_memory_context, _write_memory, _assemble_features methods ---
    def _route_slots(
        self,
        slot_values: torch.Tensor,
        z_self: torch.Tensor,
        action: torch.Tensor,
        self_state: torch.Tensor | None,
        update_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates slot scores and performs broadcast."""
        slot_novelty = slot_values.var(dim=-1, unbiased=False)

        if self._slot_baseline is None:
            self._slot_baseline = slot_values.mean(dim=(0,)).detach().cpu() # Mean over batch

        slot_progress: torch.Tensor
        if self._novelty_trace is None:
            slot_progress = torch.zeros_like(slot_novelty)
            if update_stats:
                self._novelty_trace = slot_novelty.detach().mean(dim=0).cpu() # Mean over batch
        else:
            prev_trace_device = self._novelty_trace.to(device=slot_novelty.device, non_blocking=True)
            current_novelty_mean = slot_novelty.mean(dim=0) # Mean over batch for update
            slot_progress = prev_trace_device.unsqueeze(0) - slot_novelty # Broadcast trace
            if update_stats:
                self._novelty_trace = (
                    (1 - self.progress_momentum) * self._novelty_trace
                    + self.progress_momentum * current_novelty_mean.detach().cpu()
                )

        if update_stats:
            baseline_update = slot_values.mean(dim=0).detach().cpu() # Mean over batch
            self._slot_baseline = (
                (1 - self.progress_momentum) * self._slot_baseline
                + self.progress_momentum * baseline_update
            )

        action_cost = torch.norm(action, p=2, dim=-1, keepdim=True) * self.action_cost_scale
        slot_cost = action_cost.expand(-1, slot_values.size(1))

        slot_norm = torch.nn.functional.normalize(slot_values, p=2, dim=-1)
        z_self_norm = torch.nn.functional.normalize(z_self, p=2, dim=-1)
        self_similarity = (slot_norm * z_self_norm.unsqueeze(1)).sum(dim=-1).clamp_(min=0.0)

        state_similarity = torch.zeros_like(self_similarity)
        if (
            self_state is not None
            and self.self_state_encoder is not None
        ):
             projected_state = self.self_state_encoder(self_state)
             projected_state_norm = torch.nn.functional.normalize(projected_state, p=2, dim=-1)
             state_similarity = (slot_norm * projected_state_norm.unsqueeze(1)).sum(dim=-1).clamp_(min=0.0)

        self_mask = (self_similarity + state_similarity).clamp_(min=0.0)

        batch_novelty_mean = slot_novelty.mean(dim=0).detach().cpu() # Mean over batch
        if self._ucb_mean is None or self._ucb_counts is None:
            self._ucb_mean = batch_novelty_mean
            self._ucb_counts = torch.ones_like(batch_novelty_mean)
        elif update_stats:
            self._ucb_counts += 1
            delta = batch_novelty_mean - self._ucb_mean
            self._ucb_mean += delta / self._ucb_counts

        if update_stats: self._step_count += 1

        # Ensure UCB stats are on the correct device for calculation
        ucb_mean_dev = self._ucb_mean.to(device=slot_novelty.device, non_blocking=True)
        ucb_counts_dev = self._ucb_counts.to(device=slot_novelty.device, non_blocking=True)
        ucb_exploration_term = self.config.workspace.ucb_beta * torch.sqrt(
            torch.log1p(torch.tensor(float(self._step_count), device=self.device))
             / ucb_counts_dev.clamp_(min=1e-6)
        )
        ucb_bonus_per_slot = ucb_mean_dev + ucb_exploration_term
        ucb = ucb_bonus_per_slot.unsqueeze(0).expand_as(slot_novelty) # Expand to batch size


        scores = (
            self.config.workspace.novelty_weight * slot_novelty
            + self.config.workspace.progress_weight * slot_progress
            + self.config.workspace.ucb_weight * ucb
            - self.config.workspace.cost_weight * slot_cost
            + self.config.workspace.self_bias * self_mask
        )

        broadcast = self.workspace.broadcast(slot_values, scores=scores)

        return broadcast, scores, slot_novelty, slot_progress, slot_cost


    def _get_memory_context(self, keys: torch.Tensor) -> torch.Tensor:
        batch = keys.shape[0]
        if len(self.memory) == 0:
            return torch.zeros(
                batch, self.memory.config.key_dim, device=self.device, dtype=keys.dtype
            )
        _, values = self.memory.read(keys, k=1) # Read nearest neighbor
        context = values[:, 0, :] # Squeeze k dimension
        return context.to(device=self.device, dtype=keys.dtype)


    def _write_memory(self, z_self: torch.Tensor, broadcast: torch.Tensor) -> None:
        value = broadcast.mean(dim=1) # Aggregate broadcasted slots
        key_cpu = z_self.detach().to(dtype=torch.float32, device="cpu")
        value_cpu = value.detach().to(dtype=torch.float32, device="cpu")
        self.memory.write(key_cpu, value_cpu)


    def _assemble_features(
        self,
        z_self: torch.Tensor,
        broadcast: torch.Tensor,
        memory_context: torch.Tensor,
    ) -> torch.Tensor:
        broadcast_flat = broadcast.flatten(start_dim=1)
        return torch.cat([z_self, broadcast_flat, memory_context], dim=-1)

    # --- Keep the _optimize method ---
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

        batch_size = observations.size(0)

        with self._autocast_ctx():
            self._graph_mark()
            latent_buffer = self._next_latent_buffer(batch_size) # Get buffer if needed
            latents = self.world_model(observations, output_buffer=latent_buffer)
            # No clone needed here

            memory_context = self._get_memory_context(latents["z_self"])
            broadcast, _, _, _, _ = self._route_slots(
                latents["slots"], latents["z_self"], actions, self_states, update_stats=False
            )
            latent_state = broadcast.mean(dim=1)
            features = self._assemble_features(latents["z_self"], broadcast, memory_context)

            self._graph_mark()
            prediction_buffer = self._prepare_prediction_buffer(batch_size, latent_state.dtype) # Get buffer if needed
            predictions = self.world_model.predict_next_latents(
                latent_state, actions, output_buffer=prediction_buffer
            )
            # No clone needed here

            self._graph_mark()
            decoded = self.world_model.decode_predictions(predictions, use_frozen=False) # Use trainable decoder
            log_likelihoods = []
            for dist in decoded:
                log_prob = dist.log_prob(next_observations)
                if log_prob.ndim > 1:
                    # Reduce over non-batch dims only when they exist
                    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
                log_likelihoods.append(log_prob)
            log_likelihoods = torch.stack(log_likelihoods)
            world_model_recon_loss = -log_likelihoods.mean(dim=0).mean() # Avg over ensemble and batch
            del decoded

            self._graph_mark()
            next_latent_buffer = self._next_latent_buffer(batch_size) # Get buffer if needed
            encoded_next = self.world_model(next_observations, output_buffer=next_latent_buffer)
            # No clone needed here

            predicted_latent_mean = torch.stack(predictions).mean(dim=0)
            target_latent = encoded_next["slots"].mean(dim=1)
            # Detach target to prevent gradients flowing back into next_obs encoder
            latent_alignment_loss = torch.nn.functional.mse_loss(predicted_latent_mean, target_latent.detach())
            del predictions
            del encoded_next

            world_model_loss = (
                world_model_recon_loss + 0.1 * latent_alignment_loss
            ) * self.config.world_model_coef

            self_state_loss = torch.tensor(0.0, device=self.device, dtype=world_model_loss.dtype)
            if self_states is not None and self.self_state_predictor is not None:
                self._graph_mark()
                predicted_self_state = self.self_state_predictor(latents["z_self"].detach()) # Detach input? Might prevent grad flow to encoder
                self_state_pred_loss = torch.nn.functional.mse_loss(predicted_self_state, self_states)
                self_state_loss = self.config.workspace.self_bias * self_state_pred_loss

            dream_loss, actor_loss, critic_loss, dream_metrics = self._stable_dreaming(latents)
            total_loss = world_model_loss + actor_loss + critic_loss + dream_loss + self_state_loss

        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)

        params_to_clip = list(self.world_model.parameters()) + \
                         list(self.actor.parameters()) + \
                         list(self.critic.parameters())
        if self.self_state_predictor: params_to_clip.extend(p for p in self.self_state_predictor.parameters() if p.requires_grad)
        params_to_clip.extend(p for p in self.empowerment.parameters() if p.requires_grad)
        # Filter parameters that actually have gradients
        params_with_grad = [p for p in params_to_clip if p.grad is not None]
        if params_with_grad:
            torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=5.0)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        metrics: dict[str, float] = {
            "train/total_loss": total_loss.item(),
            "train/world_model_loss": world_model_loss.item(),
            "train/wm_recon_loss": world_model_recon_loss.item(),
            "train/wm_align_loss": latent_alignment_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/critic_loss": critic_loss.item(),
            "train/dream_loss_empowerment": dream_loss.item(),
            "train/self_state_loss": self_state_loss.item(),
        }
        metrics.update({k: v.item() if isinstance(v, torch.Tensor) else float(v)
                        for k, v in dream_metrics.items()})

        return metrics

    # --- Keep the _stable_dreaming method ---
    def _stable_dreaming(
        self, latents: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:

        # Detach initial latents for dreaming to prevent gradients flowing back from dream into encoder
        current_latents = {k: v.detach() for k, v in latents.items()}
        batch_size = current_latents["z_self"].size(0)
        dtype = current_latents["z_self"].dtype

        memory_context = self._get_memory_context(current_latents["z_self"]) # Context depends on initial state

        rewards, values, log_probs, entropies = [], [], [], []
        competence_terms, empowerment_terms, safety_terms = [], [], []
        intrinsic_terms, explore_terms, raw_explore_terms = [], [], []
        last_dream_action: Optional[torch.Tensor] = None

        for step in range(self.config.dream_horizon):
            self._graph_mark()
            broadcast, _, _, _, _ = self._route_slots(
                current_latents["slots"], current_latents["z_self"],
                torch.zeros(batch_size, self.config.dynamics.action_dim, device=self.device, dtype=dtype),
                None, update_stats=False
            )
            features = self._assemble_features(current_latents["z_self"], broadcast, memory_context)

            self._graph_mark()
            action_dist = self.actor(features)
            dream_action = action_dist.rsample()
            dream_log_prob = action_dist.log_prob(dream_action)
            dream_entropy = action_dist.entropy()
            last_dream_action = dream_action # Keep track, overwrite each step

            self._graph_mark()
            latent_state = broadcast.mean(dim=1)
            prediction_buffer = self._prepare_prediction_buffer(batch_size, latent_state.dtype) # Get buffer if needed
            predictions = self.world_model.predict_next_latents(
                latent_state, dream_action, output_buffer=prediction_buffer
            )
            # No clone needed

            self._graph_mark()
            decoded = self.world_model.decode_predictions(predictions, use_frozen=False) # Use trainable decoder
            novelty = self.reward.get_novelty(decoded)
            predicted_obs_sample = decoded[0].rsample()
            observation_entropy = estimate_observation_entropy(predicted_obs_sample)

            dream_reward, norm_components, raw_components = self.reward.get_intrinsic_reward(
                novelty, observation_entropy, dream_action, latent_state, return_components=True
            )
            del decoded
            del predictions

            # --- Store metrics (detached) ---
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
            # --- End Store metrics ---

            normalized_reward = self.reward_normalizer(dream_reward)

            self._graph_mark()
            critic_value = self.critic(features)

            values.append(critic_value)
            rewards.append(normalized_reward)
            log_probs.append(dream_log_prob)
            entropies.append(dream_entropy)

            self._graph_mark()
            next_latent_buffer = self._next_latent_buffer(batch_size) # Get buffer if needed
            # IMPORTANT: Detach predicted_obs_sample before feeding back into encoder
            # We don't want dream gradients flowing back through the observation encoder in the next step
            current_latents = self.world_model(
                 predicted_obs_sample.detach(), output_buffer=next_latent_buffer
            )
             # No clone needed for current_latents itself due to buffer

            del predicted_obs_sample
            memory_context = self._get_memory_context(current_latents["z_self"]) # Update context based on new state


        self._graph_mark()
        final_broadcast, _, _, _, _ = self._route_slots(
            current_latents["slots"], current_latents["z_self"],
            torch.zeros(batch_size, self.config.dynamics.action_dim, device=self.device, dtype=dtype),
            None, update_stats=False
        )
        final_features = self._assemble_features(
            current_latents["z_self"], final_broadcast, memory_context
        )
        # Detach final value prediction, it's a target for GAE
        self._graph_mark()
        next_value = self.critic(final_features).detach()

        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        advantages, returns = self._compute_gae(rewards_tensor, values_tensor, next_value)

        # Actor loss needs gradients through log_probs and entropies
        actor_loss = -(
             (advantages.detach() * log_probs_tensor).mean() # Detach advantages
             + self.config.entropy_coef * entropies_tensor.mean()
        )

        # Critic loss needs gradients through values_tensor
        critic_loss = (
             self.config.critic_coef * 0.5 * (returns.detach() - values_tensor).pow(2).mean() # Detach returns
        )

        assert last_dream_action is not None
        # Empowerment loss only affects empowerment estimator, detach inputs
        empowerment_term = self.empowerment(
             last_dream_action.detach(), final_broadcast.mean(dim=1).detach()
        ).mean()
        dream_loss = -self.optimizer_empowerment_weight * empowerment_term

        # --- Collect Metrics ---
        metrics_to_stack = {
            "dream/intrinsic_reward": intrinsic_terms, "dream/competence": competence_terms,
            "dream/empowerment": empowerment_terms, "dream/safety": safety_terms,
            "dream/explore": explore_terms, "dream/explore_raw": raw_explore_terms,
        }
        dreaming_metrics: Dict[str, torch.Tensor | float] = {
            "dream/policy_entropy": entropies_tensor.mean().detach() # Detach entropy metric
        }
        for name, term_list in metrics_to_stack.items():
            if term_list:
                 stacked_terms = torch.stack(term_list)
                 dreaming_metrics[name] = stacked_terms.mean().detach() # Detach metric
                 if "explore" in name:
                     dreaming_metrics[f"{name}_min"] = stacked_terms.min().detach() # Detach metric
                     dreaming_metrics[f"{name}_max"] = stacked_terms.max().detach() # Detach metric
            else: # Handle empty list case
                 dreaming_metrics[name] = 0.0
                 if "explore" in name:
                     dreaming_metrics[f"{name}_min"] = 0.0
                     dreaming_metrics[f"{name}_max"] = 0.0

        return dream_loss, actor_loss, critic_loss, dreaming_metrics


    # --- Keep the _compute_gae method ---
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon, batch = rewards.shape
        values_ext = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(batch, device=self.device, dtype=rewards.dtype)
        for t in reversed(range(horizon)):
            delta = rewards[t] + self.config.discount_gamma * values_ext[t + 1].detach() - values_ext[t].detach() # Detach values used in delta
            # Advantage calculation should not propagate gradients through future advantages or values
            last_advantage = delta + (
                self.config.discount_gamma * self.config.gae_lambda * last_advantage.detach()
            )
            advantages[t] = last_advantage # Store the calculated advantage

        # Returns = Advantages + Values (detach values here as returns are targets for critic)
        returns = advantages + values.detach()
        return advantages, returns # Return non-detached advantages for actor loss


    # --- Keep the buffer helper methods ---
    def _latent_output_dtype(self) -> torch.dtype:
        return self.autocast_dtype if self.autocast_enabled else self._world_model_param_dtype

    def _create_latent_buffer(self, batch: int, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            "z_self": torch.empty((batch, self._slot_dim), device=self.device, dtype=dtype),
            "slots": torch.empty(
                (batch, self._num_slots, self._slot_dim), device=self.device, dtype=dtype
            ),
        }

    def _ensure_latent_buffers(self, batch: int, dtype: torch.dtype) -> None:
        if not self._use_output_buffers:
            return
        needs_rebuild = len(self._latent_buffers) != 2
        if not needs_rebuild:
            buffer = self._latent_buffers[0]
            slots = buffer["slots"]
            z_self = buffer["z_self"]
            if (slots.shape[0] != batch or slots.dtype != dtype or slots.device != self.device or
                z_self.shape[0] != batch or z_self.dtype != dtype or z_self.device != self.device):
                needs_rebuild = True
        if needs_rebuild:
            self._latent_buffers = [
                self._create_latent_buffer(batch, dtype),
                self._create_latent_buffer(batch, dtype),
            ]
            self._latent_buffer_index = 0

    def _next_latent_buffer(self, batch: int) -> Optional[Dict[str, torch.Tensor]]:
        if not self._use_output_buffers:
            return None
        dtype = self._latent_output_dtype()
        self._ensure_latent_buffers(batch, dtype)
        buffer = self._latent_buffers[self._latent_buffer_index]
        self._latent_buffer_index = (self._latent_buffer_index + 1) % len(self._latent_buffers)
        return buffer

    def _prepare_prediction_buffer(self, batch: int, dtype: torch.dtype) -> Optional[List[torch.Tensor]]:
        if not self._use_output_buffers or self._ensemble_size == 0:
             return None
        needs_rebuild = self._prediction_buffer is None or len(self._prediction_buffer) != self._ensemble_size
        if not needs_rebuild:
             sample = self._prediction_buffer[0]
             if (sample.shape[0] != batch or sample.dtype != dtype or sample.device != self.device):
                 needs_rebuild = True
        if needs_rebuild:
             self._prediction_buffer = [
                 torch.empty((batch, self._latent_dim), device=self.device, dtype=dtype)
                 for _ in range(self._ensemble_size)
             ]
        return cast(List[torch.Tensor], self._prediction_buffer)

    # --- Keep the _graph_mark method ---
    def _graph_mark(self) -> None:
        if self._compiled_runtime and cudagraph_mark_step_begin is not None:
            cudagraph_mark_step_begin()
