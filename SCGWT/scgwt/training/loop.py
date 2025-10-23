from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import nullcontext
import warnings
from typing import Dict, List, Optional, Tuple, cast

import torch
from torch import nn
# Corrected import location for GradScaler and autocast
from torch.cuda.amp import GradScaler, autocast

try:
    from torch._dynamo.eval_frame import OptimizedModule as _OptimizedModuleType
except (ImportError, AttributeError):
    # Define as empty tuple if import fails, for isinstance checks
    _OptimizedModuleType = ()

try:
    from torch.compiler import cudagraph_mark_step_begin
except (ImportError, AttributeError):
    try:
        from torch._inductor.utils import cudagraph_mark_step_begin
    except (ImportError, AttributeError):
        try:
            # Fallback for older torch versions
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
        # Ensure calculations happen on the correct device and dtype
        values = x.detach().to(device=self.device, dtype=torch.float32).reshape(-1, 1)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = values.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    # Make static as it doesn't depend on instance state besides inputs
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
        # Combine variances using parallel algorithm
        m2 = m_a + m_b + delta.pow(2) * count * batch_count / total_count
        new_var = m2 / total_count
        # Clamp variance for numerical stability
        new_var = torch.clamp(new_var, min=epsilon)
        return new_mean, new_var, total_count


    def _update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        # Prevent in-place modification issues with graph capture if needed
        # Though detached inputs should make this safe, explicit copy is safer
        current_mean = self.mean.clone()
        current_var = self.var.clone()
        current_count = self.count.clone()

        new_mean, new_var, new_count = self._update_running_moments(
            current_mean, current_var, current_count,
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
        self.device = device # Store device for potential use

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        # Convert to float32 for stable statistics update
        reward_fp32 = reward.to(dtype=torch.float32)
        self.stats.update(reward_fp32)
        # Use updated stats for normalization
        mean = self.stats.mean
        var = self.stats.var
        denom = torch.sqrt(var + self.eps)
        # Normalize: (X - mu) / sigma
        normalized = (reward_fp32 - mean) / denom
        # Clamp the normalized reward
        normalized = torch.clamp(normalized, -self.clamp_value, self.clamp_value)
        # Convert back to original dtype
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

# Helper to check if a module was likely wrapped by torch.compile
def _is_compiled_artifact(original: nn.Module, candidate: nn.Module) -> bool:
    """Best-effort detection that ``torch.compile`` wrapped the module."""
    # Check exact type if available
    if _OptimizedModuleType and isinstance(candidate, _OptimizedModuleType):
        return True
    # If they are the same object, it wasn't wrapped
    if candidate is original:
        return False
    # Check for common attributes added by compile
    return hasattr(candidate, "_orig_mod") or hasattr(candidate, "__compiled_fn__")


class TrainingLoop:
    """High-level container wiring the major subsystems together."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        precision_key = config.precision.lower()
        if precision_key in {"bf16", "bfloat16"}:
            self.autocast_dtype = torch.bfloat16
        elif precision_key in {"fp16", "float16", "half"}:
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.float32
        # Enable autocast only for CUDA and if dtype is not float32
        self.autocast_enabled = self.device.type == "cuda" and self.autocast_dtype != torch.float32

        self.progress_momentum = config.workspace.progress_momentum
        self.action_cost_scale = config.workspace.action_cost_scale

        wm_config = WorldModelConfig(
            encoder=config.encoder,
            decoder=config.decoder,
            dynamics=config.dynamics,
            ensemble_size=config.world_model_ensemble,
        )
        self.world_model = WorldModelEnsemble(wm_config).to(self.device)

        # Store dimensions needed for buffer creation
        self._slot_dim = config.encoder.slot_dim
        self._num_slots = config.encoder.num_slots
        self._latent_dim = config.dynamics.latent_dim # Assuming dynamics output dim is the target
        self._ensemble_size = config.world_model_ensemble

        # Determine default dtype from model parameters
        first_param = next(self.world_model.parameters(), None)
        self._world_model_param_dtype = (
            first_param.dtype if first_param is not None else torch.float32
        )

        # Initialize buffer management attributes
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

        # Policy dimensions derived from encoder/workspace layout.
        policy_feature_dim = (
            self._slot_dim # Use stored dim
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

        # Apply channels_last memory format if configured and on CUDA
        if config.channels_last and self.device.type == "cuda":
            # Apply to relevant modules that benefit (Conv layers primarily)
            modules_to_format = [self.world_model, self.actor, self.critic]
            # Include sub-modules if they exist and are relevant (e.g., self-state)
            if self.self_state_encoder: modules_to_format.append(self.self_state_encoder)
            if self.self_state_predictor: modules_to_format.append(self.self_state_predictor)

            for module in modules_to_format:
                module.to(memory_format=torch.channels_last)
                # Ensure parameters are also contiguous in the desired format
                # This might be needed for some optimizers or operations
                for parameter in module.parameters():
                     if parameter.ndim >= 4: # Typically weights of Conv layers
                         # Use data to modify in-place
                         parameter.data = parameter.data.contiguous(memory_format=torch.channels_last)


        # Compile models if configured
        compiled_world_model = False
        if config.compile_model:
            # List modules to potentially compile
            modules_to_compile = {"world_model": self.world_model,
                                  "actor": self.actor,
                                  "critic": self.critic}
            # Add optional modules if they exist
            if self.self_state_predictor:
                 modules_to_compile["self_state_predictor"] = self.self_state_predictor
            # Note: self_state_encoder is usually simple Linear, less benefit from compile

            for name, original_module in modules_to_compile.items():
                try:
                    # Attempt compilation
                    compiled_module = torch.compile(original_module, mode="max-autotune", fullgraph=False)
                    # Check if compilation actually happened (useful for debugging)
                    was_compiled = _is_compiled_artifact(original_module, compiled_module)
                    print(f"Compiling {name}... Success: {was_compiled}")
                    # Update the attribute on self
                    setattr(self, name, compiled_module)
                    # Track if the world_model specifically was compiled
                    if name == "world_model" and was_compiled:
                        compiled_world_model = True
                except Exception as e:
                    # Fallback to original module if compilation fails
                    print(f"Failed to compile {name}: {e}")
                    setattr(self, name, original_module) # Ensure attribute remains

        # Flag indicating if we are running in a compiled context (for graph marking)
        self._compiled_runtime = compiled_world_model
        # Use output buffers ONLY if compiling, otherwise it adds overhead
        self._use_output_buffers = config.compile_model

        # Initialize state for routing and statistics
        self._slot_baseline: torch.Tensor | None = None
        self._ucb_mean: torch.Tensor | None = None
        self._ucb_counts: torch.Tensor | None = None
        self._step_count: int = 0
        self._novelty_trace: torch.Tensor | None = None
        self._latest_self_state: torch.Tensor | None = None # For potential stateful logic

        # Initialize replay buffer
        self.rollout_buffer = RolloutBuffer(capacity=config.rollout_capacity)
        self.batch_size = config.batch_size

        # Gather parameters for the optimizer
        params: list[torch.nn.Parameter] = []
        # Use list comprehension for cleaner parameter gathering
        params.extend(p for p in self.world_model.parameters() if p.requires_grad)
        params.extend(p for p in self.empowerment.parameters() if p.requires_grad)
        params.extend(p for p in self.actor.parameters() if p.requires_grad)
        params.extend(p for p in self.critic.parameters() if p.requires_grad)
        if self.self_state_encoder is not None:
            params.extend(p for p in self.self_state_encoder.parameters() if p.requires_grad)
        if self.self_state_predictor is not None:
            params.extend(p for p in self.self_state_predictor.parameters() if p.requires_grad)

        # Initialize optimizer, trying fused version first
        try:
             # Fused AdamW is often faster on CUDA
            self.optimizer = torch.optim.AdamW(params, lr=config.optimizer_lr, fused=True)
            print("Using Fused AdamW optimizer.")
        except (TypeError, ValueError, RuntimeError): # Catch potential errors if fused is unavailable/incompatible
            print("Fused AdamW not available, falling back to standard AdamW.")
            self.optimizer = torch.optim.AdamW(params, lr=config.optimizer_lr)

        self.optimizer_empowerment_weight = config.optimizer_empowerment_weight

        # Initialize GradScaler for mixed precision, explicitly setting device_type
        # scaler_device_type = self.device.type # 'cuda' or 'cpu'
        self.grad_scaler = GradScaler(
            # device=self.device, # Not a valid argument
            enabled=self.autocast_enabled and self.autocast_dtype == torch.float16, # Enable only for fp16 on CUDA
            # init_scale=..., growth_factor=..., backoff_factor=..., # Can tune these
        )
        print(f"GradScaler enabled: {self.grad_scaler.is_enabled()}")


    def _autocast_ctx(self):
        """Returns the appropriate autocast context manager."""
        if self.autocast_enabled:
            # Explicitly set device_type for CUDA autocast
            return autocast(device_type=self.device.type, dtype=self.autocast_dtype)
        # Return a null context manager if autocast is disabled
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

        # Process self_state if provided and configured
        state_tensor: Optional[torch.Tensor] = None
        if self.self_state_dim > 0:
            if self_state is None:
                # Default to zeros if no state provided
                state_tensor = torch.zeros(
                    batch, self.self_state_dim, device=self.device, dtype=observation.dtype
                )
            else:
                state_tensor = self_state.to(self.device, non_blocking=True)
                # Handle single instance vs batch
                if state_tensor.ndim == 1: state_tensor = state_tensor.unsqueeze(0)
                # Expand if batch dim mismatch (e.g., single state for batch obs)
                if state_tensor.size(0) == 1 and batch > 1:
                     state_tensor = state_tensor.expand(batch, -1)
                elif state_tensor.size(0) != batch:
                     raise ValueError(f"self_state batch dim mismatch: got {state_tensor.size(0)}, expected {batch}")
            # Keep a detached copy of the latest state if needed elsewhere
            self._latest_self_state = state_tensor.detach()

        # --- Inference Phase (No Gradients) ---
        with torch.no_grad():
            with self._autocast_ctx(): # Apply mixed precision if enabled
                self._graph_mark() # Mark beginning for CUDAGraphs if compiled

                # Encode observation, using buffer if compiling
                latent_buffer = self._next_latent_buffer(batch) if self._use_output_buffers else None
                latents = self.world_model(observation, output_buffer=latent_buffer)
                # NOTE: No cloning needed here, buffer swapping handles persistence

                memory_context = self._get_memory_context(latents["z_self"])

                # Determine action: either provided or sampled from policy
                action_for_routing: torch.Tensor
                if action is not None:
                    action_for_routing = action.to(self.device, non_blocking=True)
                else:
                    # Use zeros if no action provided yet (for initial routing)
                    action_for_routing = torch.zeros(
                        batch, self.config.dynamics.action_dim, device=self.device, dtype=latents["z_self"].dtype
                    )

                # Route slots based on current state and tentative action
                (
                    broadcast, scores, slot_novelty, slot_progress, slot_cost,
                ) = self._route_slots(
                    latents["slots"], latents["z_self"], action_for_routing,
                    state_tensor, update_stats=True # Update routing stats
                )

                # Assemble features for policy/critic input
                features = self._assemble_features(latents["z_self"], broadcast, memory_context)

                # If action wasn't provided, sample it now using updated features
                if action is None:
                    self._graph_mark()
                    action_dist = self.actor(features)
                    action = action_dist.rsample() # Sample action
                    # Re-route slots based on the *actual* sampled action (no stat update)
                    (
                         broadcast, scores, slot_novelty, _, _, # Don't need progress/cost again
                    ) = self._route_slots(
                         latents["slots"], latents["z_self"], action,
                         state_tensor, update_stats=False # Don't update stats twice
                    )
                    # Re-assemble features based on final broadcast for consistency
                    features = self._assemble_features(latents["z_self"], broadcast, memory_context)
                else:
                     # Ensure provided action is on the correct device
                     action = action.to(self.device, non_blocking=True)


                # Predict next state using dynamics model ensemble
                latent_state = broadcast.mean(dim=1) # Aggregate broadcasted slots
                self._graph_mark()
                prediction_buffer = self._prepare_prediction_buffer(
                    batch, latent_state.dtype # Use consistent dtype
                ) if self._use_output_buffers else None
                predictions = self.world_model.predict_next_latents(
                    latent_state, action, output_buffer=prediction_buffer
                )
                # NOTE: No cloning needed here

                # Decode predictions and calculate intrinsic reward components
                self._graph_mark()
                decoded = self.world_model.decode_predictions(predictions, use_frozen=True) # Use frozen decoder
                novelty = self.reward.get_novelty(decoded).to(self.device) # Ensure novelty is on device
                observation_entropy = estimate_observation_entropy(observation)
                intrinsic_raw, norm_components, raw_components = self.reward.get_intrinsic_reward(
                    novelty, observation_entropy, action, latent_state, return_components=True
                )
                # Clean up distributions and predictions immediately
                del decoded
                del predictions

        # --- Post-Inference Processing ---
        # Normalize the final intrinsic reward
        intrinsic = self.reward_normalizer(intrinsic_raw)
        # Detach reward components for logging/storage
        reward_components_detached = {k: v.detach() for k, v in norm_components.items()} if norm_components else None
        raw_components_detached = {k: v.detach() for k, v in raw_components.items()} if raw_components else None

        # Write current experience to episodic memory
        self._write_memory(latents["z_self"], broadcast)

        # --- Training Phase (Optional) ---
        train_loss: Optional[float] = None
        training_metrics: Optional[Dict[str, float]] = None
        if train and next_observation is not None:
            # Prepare tensors for storage (move to CPU, make contiguous)
            obs_cpu = observation.detach().cpu().contiguous()
            act_cpu = action.detach().cpu().contiguous()
            next_cpu = next_observation.detach().cpu().contiguous()
            state_cpu = state_tensor.detach().cpu().contiguous() if state_tensor is not None else None

            # Pin memory if using CUDA for faster transfer later
            if torch.cuda.is_available():
                obs_cpu, act_cpu, next_cpu = obs_cpu.pin_memory(), act_cpu.pin_memory(), next_cpu.pin_memory()
                if state_cpu is not None: state_cpu = state_cpu.pin_memory()

            # Add transition(s) to the rollout buffer
            # Handle potential batch dimension > 1 from observation
            batch_items = obs_cpu.shape[0]
            for idx in range(batch_items):
                 current_state_cpu = state_cpu[idx] if state_cpu is not None else None
                 self.rollout_buffer.push(obs_cpu[idx], act_cpu[idx], next_cpu[idx], current_state_cpu)

            # Perform optimization step if buffer has enough samples
            training_metrics = self._optimize() # Returns metrics dict or None
            if training_metrics is not None:
                # Extract total loss for convenience, if available
                train_loss = training_metrics.get("train/total_loss")


        # Return results of the step
        return StepResult(
            action=action.detach(), # Detach action before returning
            intrinsic_reward=intrinsic.detach(),
            novelty=slot_novelty.detach(),
            observation_entropy=observation_entropy.detach(),
            slot_scores=scores.detach(),
            reward_components=reward_components_detached,
            raw_reward_components=raw_components_detached,
            training_loss=train_loss,
            training_metrics=training_metrics,
        )

    def _route_slots(
        self,
        slot_values: torch.Tensor,
        z_self: torch.Tensor,
        action: torch.Tensor,
        self_state: torch.Tensor | None,
        update_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates slot scores and performs broadcast."""
        # Calculate novelty as variance across slot dimension
        slot_novelty = slot_values.var(dim=-1, unbiased=False)

        # Initialize baseline if first time
        if self._slot_baseline is None:
            # Ensure baseline is on CPU to avoid unnecessary GPU memory use if static
            self._slot_baseline = slot_values.mean(dim=(0,)).detach().cpu() # Mean over batch

        # Calculate learning progress based on novelty trace (EMA)
        slot_progress: torch.Tensor
        if self._novelty_trace is None:
            slot_progress = torch.zeros_like(slot_novelty)
            if update_stats:
                # Initialize trace on CPU
                self._novelty_trace = slot_novelty.detach().mean(dim=0).cpu() # Mean over batch
        else:
             # Ensure trace is on the correct device for calculation
            prev_trace_device = self._novelty_trace.to(device=slot_novelty.device, non_blocking=True)
            current_novelty_mean = slot_novelty.mean(dim=0) # Mean over batch for update
            # Progress is change from previous trace value (broadcasted)
            slot_progress = prev_trace_device.unsqueeze(0) - slot_novelty
            if update_stats:
                # Update trace with momentum on CPU
                self._novelty_trace = (
                    (1 - self.progress_momentum) * self._novelty_trace
                    + self.progress_momentum * current_novelty_mean.detach().cpu()
                )


        # Update slot baseline if required
        if update_stats:
             # Update baseline with momentum on CPU
            baseline_update = slot_values.mean(dim=0).detach().cpu()
            self._slot_baseline = (
                (1 - self.progress_momentum) * self._slot_baseline
                + self.progress_momentum * baseline_update
            )


        # Calculate action cost (L2 norm scaled)
        action_cost = torch.norm(action, p=2, dim=-1, keepdim=True) * self.action_cost_scale
        # Expand cost to match number of slots
        slot_cost = action_cost.expand(-1, slot_values.size(1))

        # Calculate self-similarity features
        slot_norm = torch.nn.functional.normalize(slot_values, p=2, dim=-1)
        z_self_norm = torch.nn.functional.normalize(z_self, p=2, dim=-1)
        # Cosine similarity between each slot and z_self
        self_similarity = (slot_norm * z_self_norm.unsqueeze(1)).sum(dim=-1).clamp_(min=0.0)

        # Calculate state similarity if self-state is available
        state_similarity = torch.zeros_like(self_similarity)
        if (
            self_state is not None
            and self.self_state_encoder is not None
        ):
             # Project self-state and normalize
             projected_state = self.self_state_encoder(self_state) # Assuming state_tensor is already on device
             projected_state_norm = torch.nn.functional.normalize(projected_state, p=2, dim=-1)
             # Cosine similarity between each slot and projected state
             state_similarity = (slot_norm * projected_state_norm.unsqueeze(1)).sum(dim=-1).clamp_(min=0.0)

        # Combine similarities for self-mask (add -> clamp ensures non-negative)
        self_mask = (self_similarity + state_similarity).clamp_(min=0.0)

        # Update UCB (Upper Confidence Bound) statistics for exploration bonus
        # Calculate mean novelty over batch dimension
        batch_novelty_mean = slot_novelty.mean(dim=0).detach().cpu()
        if self._ucb_mean is None or self._ucb_counts is None:
             # Initialize UCB stats on CPU
            self._ucb_mean = batch_novelty_mean
            self._ucb_counts = torch.ones_like(batch_novelty_mean)
        elif update_stats: # Only update stats if requested
             # Incremental update of mean and counts
            self._ucb_counts += 1
            # Welford's algorithm for stable mean update
            delta = batch_novelty_mean - self._ucb_mean
            self._ucb_mean += delta / self._ucb_counts

        # Increment global step counter only when updating stats
        if update_stats: self._step_count += 1

        # Calculate UCB bonus (ensure stats are on the correct device)
        ucb_mean_dev = self._ucb_mean.to(device=slot_novelty.device, non_blocking=True)
        ucb_counts_dev = self._ucb_counts.to(device=slot_novelty.device, non_blocking=True)
        # UCB formula: mean + beta * sqrt(log(total_steps + 1) / count_for_arm)
        ucb_exploration_term = self.config.workspace.ucb_beta * torch.sqrt(
            torch.log1p(torch.tensor(float(self._step_count), device=self.device))
             / ucb_counts_dev.clamp_(min=1e-6) # Avoid division by zero
        )
        ucb_bonus_per_slot = ucb_mean_dev + ucb_exploration_term
        # Expand UCB bonus to match batch size
        ucb = ucb_bonus_per_slot.unsqueeze(0).expand_as(slot_novelty)


        # Combine components into final slot scores
        scores = (
            self.config.workspace.novelty_weight * slot_novelty
            + self.config.workspace.progress_weight * slot_progress # Use calculated progress
            + self.config.workspace.ucb_weight * ucb # Use calculated UCB
            - self.config.workspace.cost_weight * slot_cost # Penalize action cost
            + self.config.workspace.self_bias * self_mask # Add self-centric bias
        )

        # Perform broadcast using top-k selection based on scores
        broadcast = self.workspace.broadcast(slot_values, scores=scores)

        return broadcast, scores, slot_novelty, slot_progress, slot_cost


    def _get_memory_context(self, keys: torch.Tensor) -> torch.Tensor:
        """Retrieves context from episodic memory based on query keys."""
        batch = keys.shape[0]
        # Return zeros if memory is empty
        if len(self.memory) == 0:
            return torch.zeros(
                batch, self.memory.config.key_dim, device=self.device, dtype=keys.dtype
            )
        # Perform k-NN search (assuming k=1 for context)
        # Note: memory.read handles device transfer
        _, values = self.memory.read(keys, k=1) # Read nearest neighbor
        # Get the value of the nearest neighbor [batch, k, dim] -> [batch, dim]
        context = values[:, 0, :] # Squeeze k dimension
        return context.to(device=self.device, dtype=keys.dtype) # Ensure correct device/dtype


    def _write_memory(self, z_self: torch.Tensor, broadcast: torch.Tensor) -> None:
        """Writes current experience to episodic memory."""
        # Use aggregated broadcast slots as the value
        value = broadcast.mean(dim=1) # [batch, num_broadcast, dim] -> [batch, dim]
        # FAISS requires float32 CPU tensors
        key_cpu = z_self.detach().to(dtype=torch.float32, device="cpu")
        value_cpu = value.detach().to(dtype=torch.float32, device="cpu")
        # Write key-value pairs
        self.memory.write(key_cpu, value_cpu)


    def _assemble_features(
        self,
        z_self: torch.Tensor,
        broadcast: torch.Tensor,
        memory_context: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenates features for policy/critic input."""
        # Flatten broadcast slots: [batch, num_broadcast, dim] -> [batch, num_broadcast * dim]
        broadcast_flat = broadcast.flatten(start_dim=1)
        # Concatenate self-state, flattened broadcast, and memory context
        return torch.cat([z_self, broadcast_flat, memory_context], dim=-1)


    def _optimize(self) -> dict[str, float] | None:
        """Performs a single optimization step using Stable Dreaming."""
        # Check if buffer has enough samples
        if len(self.rollout_buffer) < self.batch_size:
            return None

        # Sample a batch from the replay buffer
        observations, actions, next_observations, self_states = self.rollout_buffer.sample(
            self.batch_size
        )
        # Move sampled data to the training device
        observations = observations.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        next_observations = next_observations.to(self.device, non_blocking=True)
        if self_states is not None:
            self_states = self_states.to(self.device, non_blocking=True)

        # Reset gradients before backward pass
        self.optimizer.zero_grad(set_to_none=True) # More efficient than setting to zero

        batch_size = observations.size(0) # Get actual batch size

        # Perform forward and backward pass within autocast context if enabled
        with self._autocast_ctx():
            # --- World Model Forward Pass ---
            self._graph_mark()
            latent_buffer = self._next_latent_buffer(batch_size) if self._use_output_buffers else None
            latents = self.world_model(observations, output_buffer=latent_buffer)
            # NOTE: No clone needed if using buffers

            # --- Calculate Losses Based on Real Data ---
            memory_context = self._get_memory_context(latents["z_self"])
            broadcast, _, _, _, _ = self._route_slots(
                latents["slots"], latents["z_self"], actions, self_states, update_stats=False
            )
            latent_state = broadcast.mean(dim=1)

            # Dynamics Prediction Loss (Reconstruction/Likelihood)
            self._graph_mark()
            prediction_buffer = self._prepare_prediction_buffer(
                batch_size, latent_state.dtype
            ) if self._use_output_buffers else None
            predictions = self.world_model.predict_next_latents(
                latent_state, actions, output_buffer=prediction_buffer
            )
            # NOTE: No clone needed

            self._graph_mark()
            # Decode using the *trainable* decoder for WM loss
            decoded = self.world_model.decode_predictions(predictions, use_frozen=False)
            # Calculate negative log-likelihood loss (reconstruction)
            # Sum over ensemble members, then mean over batch
            log_likelihoods = torch.stack(
                 [dist.log_prob(next_observations).mean(dim=list(range(1, next_observations.ndim))) # Mean over spatial/channel dims
                  for dist in decoded]
            ) # Shape [ensemble_size, batch_size]
            # Average likelihood across ensemble, then mean over batch -> negative for loss
            world_model_recon_loss = -log_likelihoods.mean(dim=0).mean()
            del decoded # Free memory

            # Latent Alignment Loss (Consistency between predicted and encoded next state)
            self._graph_mark()
            next_latent_buffer = self._next_latent_buffer(batch_size) if self._use_output_buffers else None
            encoded_next = self.world_model(next_observations, output_buffer=next_latent_buffer)
             # NOTE: No clone needed

            # Average predictions across ensemble for alignment target
            predicted_latent_mean = torch.stack(predictions).mean(dim=0) # [batch, latent_dim]
            # Use mean of encoded next slots as target
            target_latent = encoded_next["slots"].mean(dim=1) # [batch, latent_dim]
            # MSE loss for latent alignment
            latent_alignment_loss = torch.nn.functional.mse_loss(predicted_latent_mean, target_latent.detach()) # Detach target
            del predictions # Free memory
            del encoded_next

            # Combine WM losses with coefficient
            world_model_loss = (
                world_model_recon_loss + 0.1 * latent_alignment_loss # Weight alignment loss
            ) * self.config.world_model_coef

            # Self-State Prediction Loss (if applicable)
            self_state_loss = torch.tensor(0.0, device=self.device, dtype=world_model_loss.dtype)
            if (
                self_states is not None and self.self_state_predictor is not None
            ):
                 # Predict self-state from current latent self-state
                predicted_self_state = self.self_state_predictor(latents["z_self"])
                 # MSE loss against actual self-state
                self_state_pred_loss = torch.nn.functional.mse_loss(predicted_self_state, self_states)
                 # Apply weighting coefficient
                self_state_loss = self.config.workspace.self_bias * self_state_pred_loss # Reusing self_bias, consider separate config?


            # --- Stable Dreaming for Actor-Critic Update ---
            dream_loss, actor_loss, critic_loss, dream_metrics = self._stable_dreaming(latents)

            # --- Total Loss ---
            total_loss = world_model_loss + actor_loss + critic_loss + dream_loss + self_state_loss


        # --- Backward Pass and Optimization Step ---
        # Scale the loss for mixed precision
        self.grad_scaler.scale(total_loss).backward()

        # Unscale gradients before clipping and optimizer step
        self.grad_scaler.unscale_(self.optimizer)

        # Clip gradients to prevent explosion
        # Gather all parameters needing grad clipping
        params_to_clip = list(self.world_model.parameters()) + \
                         list(self.actor.parameters()) + \
                         list(self.critic.parameters())
        if self.self_state_predictor: params_to_clip.extend(self.self_state_predictor.parameters())
        # Add empowerment model parameters if they exist and require grad
        params_to_clip.extend(p for p in self.empowerment.parameters() if p.requires_grad)

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=5.0) # Common max_norm value


        # Perform optimizer step using scaled gradients
        self.grad_scaler.step(self.optimizer)

        # Update the scaler for the next iteration
        self.grad_scaler.update()

        # --- Logging Metrics ---
        # Collect metrics, detaching tensors and moving to CPU
        metrics: dict[str, float] = {
            "train/total_loss": total_loss.item(),
            "train/world_model_loss": world_model_loss.item(),
            "train/wm_recon_loss": world_model_recon_loss.item(), # Log sub-component
            "train/wm_align_loss": latent_alignment_loss.item(), # Log sub-component
            "train/actor_loss": actor_loss.item(),
            "train/critic_loss": critic_loss.item(),
            "train/dream_loss_empowerment": dream_loss.item(), # Renamed for clarity
            "train/self_state_loss": self_state_loss.item(),
        }
        # Add metrics from the dreaming phase
        metrics.update({k: v.item() if isinstance(v, torch.Tensor) else float(v)
                        for k, v in dream_metrics.items()})

        return metrics


    def _stable_dreaming(
        self, latents: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
        """Performs imagined rollouts (dreaming) to train actor and critic."""

        # Initialize current latent state for dreaming (NO clone needed if latents is from buffer)
        current_latents = latents
        batch_size = current_latents["z_self"].size(0)
        dtype = current_latents["z_self"].dtype

        # Get initial memory context
        memory_context = self._get_memory_context(current_latents["z_self"])

        # Lists to store trajectory data during dreaming
        rewards, values, log_probs, entropies = [], [], [], []
        # Metrics storage
        competence_terms, empowerment_terms, safety_terms = [], [], []
        intrinsic_terms, explore_terms, raw_explore_terms = [], [], []

        last_dream_action: Optional[torch.Tensor] = None # To store the final action for emp loss

        # Dream for a fixed horizon
        for step in range(self.config.dream_horizon):
            # --- Policy Action Selection ---
            self._graph_mark()
            # Route slots based on current dream state (no stat updates needed)
            # Use zeros as placeholder action for routing, actual action sampled next
            broadcast, _, _, _, _ = self._route_slots(
                current_latents["slots"], current_latents["z_self"],
                torch.zeros(batch_size, self.config.dynamics.action_dim, device=self.device, dtype=dtype),
                None, # No self_state in dream
                update_stats=False
            )
            features = self._assemble_features(current_latents["z_self"], broadcast, memory_context)

            # Sample action from actor
            action_dist = self.actor(features)
            dream_action = action_dist.rsample() # Use rsample for reparameterization trick
            dream_log_prob = action_dist.log_prob(dream_action)
            dream_entropy = action_dist.entropy()
            last_dream_action = dream_action # Keep track of the last action

            # --- World Model Prediction ---
            self._graph_mark()
            latent_state = broadcast.mean(dim=1) # Aggregate state
            prediction_buffer = self._prepare_prediction_buffer(
                batch_size, latent_state.dtype
            ) if self._use_output_buffers else None
            predictions = self.world_model.predict_next_latents(
                latent_state, dream_action, output_buffer=prediction_buffer
            )
            # NOTE: No clone needed

            # --- Intrinsic Reward Calculation ---
            self._graph_mark()
            # Decode predictions using *trainable* decoder (important for gradients)
            decoded = self.world_model.decode_predictions(predictions, use_frozen=False)
            novelty = self.reward.get_novelty(decoded)
            # Sample an observation from the *first* ensemble member's prediction
            # Using rsample allows gradients to flow back through the decoder/dynamics
            predicted_obs_sample = decoded[0].rsample()
            observation_entropy = estimate_observation_entropy(predicted_obs_sample)

            # Calculate intrinsic reward for the dreamed step
            dream_reward, norm_components, raw_components = self.reward.get_intrinsic_reward(
                novelty, observation_entropy, dream_action, latent_state, return_components=True
            )
            # Clean up distributions immediately
            del decoded
            del predictions

            # Store reward components for metrics (detached)
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


            # Normalize reward for Actor-Critic update
            normalized_reward = self.reward_normalizer(dream_reward)

            # --- Critic Value Estimation ---
            self._graph_mark()
            critic_value = self.critic(features)

            # Store trajectory data (needs clone as value will be overwritten in next step)
            # These clones are necessary because the tensors are used across loop iterations
            # and might be modified by subsequent calculations (like GAE). Buffer swapping
            # only helps avoid clones *between* graph invocations (step/optimize).
            values.append(critic_value.clone())
            rewards.append(normalized_reward.clone())
            log_probs.append(dream_log_prob.clone())
            entropies.append(dream_entropy.clone())


            # --- Prepare for Next Dream Step ---
            self._graph_mark()
            # Encode the sampled next observation to get the next latent state
            # Use buffer if compiling
            next_latent_buffer = self._next_latent_buffer(batch_size) if self._use_output_buffers else None
            current_latents = self.world_model(
                 predicted_obs_sample, output_buffer=next_latent_buffer
            )
             # NOTE: No clone needed if using buffers

            # Clean up sampled observation
            del predicted_obs_sample
            # Update memory context for the next step based on the new latent state
            memory_context = self._get_memory_context(current_latents["z_self"])


        # --- After Dream Horizon ---
        # Get value estimate for the final state
        self._graph_mark()
        # Route slots for the final state
        final_broadcast, _, _, _, _ = self._route_slots(
            current_latents["slots"], current_latents["z_self"],
            torch.zeros(batch_size, self.config.dynamics.action_dim, device=self.device, dtype=dtype),
            None, update_stats=False
        )
        final_features = self._assemble_features(
            current_latents["z_self"], final_broadcast, memory_context
        )
        # Final value estimate (detach as it's a target)
        next_value = self.critic(final_features).detach()

        # Convert trajectory lists to tensors
        rewards_tensor = torch.stack(rewards) # [horizon, batch]
        values_tensor = torch.stack(values)   # [horizon, batch]
        log_probs_tensor = torch.stack(log_probs) # [horizon, batch]
        entropies_tensor = torch.stack(entropies) # [horizon, batch]

        # --- Calculate Actor & Critic Losses using GAE ---
        advantages, returns = self._compute_gae(rewards_tensor, values_tensor, next_value)

        # Actor Loss (Policy Gradient)
        # Maximize advantage * log_prob + entropy bonus
        actor_loss = -(
             (advantages.detach() * log_probs_tensor).mean() # Detach advantages - treat as fixed targets
             + self.config.entropy_coef * entropies_tensor.mean() # Encourage exploration
        )

        # Critic Loss (Value Regression)
        # Minimize MSE between predicted value and GAE return
        critic_loss = (
             self.config.critic_coef * 0.5 * (returns - values_tensor).pow(2).mean() # Use returns as target
        )


        # --- Empowerment Loss Component (Dream Loss) ---
        # Calculated based on the *final* state and *last* action of the dream
        assert last_dream_action is not None, "Last dream action should have been stored"
        # Detach inputs to empowerment estimator - its loss affects only its own parameters
        empowerment_term = self.empowerment(
            last_dream_action.detach(), final_broadcast.mean(dim=1).detach()
        ).mean()
        # Maximize empowerment -> Minimize negative empowerment term
        dream_loss = -self.optimizer_empowerment_weight * empowerment_term


        # --- Collect Metrics ---
        # Stack metrics collected during the loop
        metrics_to_stack = {
            "dream/intrinsic_reward": intrinsic_terms,
            "dream/competence": competence_terms,
            "dream/empowerment": empowerment_terms,
            "dream/safety": safety_terms,
            "dream/explore": explore_terms,
            "dream/explore_raw": raw_explore_terms,
        }
        dreaming_metrics: Dict[str, torch.Tensor | float] = {
            # Average policy entropy over horizon
            "dream/policy_entropy": entropies_tensor.mean().detach()
        }
        # Calculate mean, min, max for stacked metrics
        for name, term_list in metrics_to_stack.items():
            if term_list: # Check if list is not empty
                 stacked_terms = torch.stack(term_list)
                 dreaming_metrics[name] = stacked_terms.mean().detach()
                 # Only add min/max for exploration terms
                 if "explore" in name:
                     dreaming_metrics[f"{name}_min"] = stacked_terms.min().detach()
                     dreaming_metrics[f"{name}_max"] = stacked_terms.max().detach()
            else:
                 # Handle case where horizon might be 0 or list is empty
                 dreaming_metrics[name] = 0.0
                 if "explore" in name:
                     dreaming_metrics[f"{name}_min"] = 0.0
                     dreaming_metrics[f"{name}_max"] = 0.0


        return dream_loss, actor_loss, critic_loss, dreaming_metrics


    def _compute_gae(
        self,
        rewards: torch.Tensor,     # [horizon, batch]
        values: torch.Tensor,      # [horizon, batch]
        next_value: torch.Tensor,  # [batch] - Value estimate of state after horizon
    ) -> tuple[torch.Tensor, torch.Tensor]: # (advantages, returns)
        """Computes Generalized Advantage Estimation (GAE)."""
        horizon, batch = rewards.shape
        # Append next_value to values for easier calculation: [horizon+1, batch]
        values_ext = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        advantages = torch.zeros_like(rewards) # Initialize advantages: [horizon, batch]
        last_advantage = torch.zeros(batch, device=self.device, dtype=rewards.dtype) # Accumulator

        # Iterate backwards through the horizon
        for t in reversed(range(horizon)):
            # Calculate TD error (delta): r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.config.discount_gamma * values_ext[t + 1] - values_ext[t]
            # Update GAE advantage: delta_t + gamma * lambda * A_{t+1}
            last_advantage = delta + (
                self.config.discount_gamma * self.config.gae_lambda * last_advantage
            )
            # Store advantage for this timestep
            advantages[t] = last_advantage

        # Calculate returns: A_t + V(s_t)
        returns = advantages + values
        return advantages, returns


    def _latent_output_dtype(self) -> torch.dtype:
        """Determines the appropriate dtype for model outputs based on autocast."""
        return self.autocast_dtype if self.autocast_enabled else self._world_model_param_dtype


    def _create_latent_buffer(self, batch: int, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Creates a dictionary structure for latent state buffers."""
        # Allocate empty tensors with correct shape, device, and dtype
        return {
            "z_self": torch.empty((batch, self._slot_dim), device=self.device, dtype=dtype),
            "slots": torch.empty(
                (batch, self._num_slots, self._slot_dim), device=self.device, dtype=dtype
            ),
        }


    def _ensure_latent_buffers(self, batch: int, dtype: torch.dtype) -> None:
        """Ensures latent double buffers exist and match required batch size/dtype."""
        # Only manage buffers if _use_output_buffers is True
        if not self._use_output_buffers:
            return

        # Check if buffers need rebuilding (size mismatch, dtype mismatch, or not initialized)
        needs_rebuild = len(self._latent_buffers) != 2
        if not needs_rebuild:
            # Check properties of the first buffer (assuming consistency)
            buffer = self._latent_buffers[0]
            slots = buffer["slots"]
            z_self = buffer["z_self"]
            if (slots.shape[0] != batch or slots.dtype != dtype or slots.device != self.device or
                z_self.shape[0] != batch or z_self.dtype != dtype or z_self.device != self.device):
                needs_rebuild = True

        # Rebuild if necessary
        if needs_rebuild:
            print(f"Rebuilding latent buffers for batch={batch}, dtype={dtype}, device={self.device}")
            self._latent_buffers = [
                self._create_latent_buffer(batch, dtype),
                self._create_latent_buffer(batch, dtype),
            ]
            self._latent_buffer_index = 0 # Reset index after rebuilding


    def _next_latent_buffer(self, batch: int) -> Optional[Dict[str, torch.Tensor]]:
        """Gets the next available latent buffer for writing, cycling through the pair."""
        # Return None if not using output buffers
        if not self._use_output_buffers:
            return None

        dtype = self._latent_output_dtype() # Determine target dtype
        self._ensure_latent_buffers(batch, dtype) # Ensure buffers match requirements

        # Get the buffer at the current index
        buffer = self._latent_buffers[self._latent_buffer_index]
        # Advance the index for the next call (wraps around using modulo)
        self._latent_buffer_index = (self._latent_buffer_index + 1) % len(self._latent_buffers)
        return buffer


    def _prepare_prediction_buffer(self, batch: int, dtype: torch.dtype) -> Optional[List[torch.Tensor]]:
        """Ensures prediction buffer list exists and matches requirements."""
        # Return None if not using output buffers or ensemble size is 0
        if not self._use_output_buffers or self._ensemble_size == 0:
             return None


        # Check if buffer needs rebuilding
        needs_rebuild = self._prediction_buffer is None or len(self._prediction_buffer) != self._ensemble_size
        if not needs_rebuild:
             # Check properties of the first tensor (assuming consistency)
             sample = self._prediction_buffer[0]
             if (sample.shape[0] != batch or sample.dtype != dtype or sample.device != self.device):
                 needs_rebuild = True


        # Rebuild if necessary
        if needs_rebuild:
             print(f"Rebuilding prediction buffer for batch={batch}, dtype={dtype}, device={self.device}")
             self._prediction_buffer = [
                 torch.empty((batch, self._latent_dim), device=self.device, dtype=dtype)
                 for _ in range(self._ensemble_size)
             ]

        # Cast is safe here because we ensure it's not None if needs_rebuild was False
        return cast(List[torch.Tensor], self._prediction_buffer)


    def _graph_mark(self) -> None:
        """Marks a step boundary for CUDAGraphs if running in compiled mode."""
        # Check if running compiled and the helper function is available
        if self._compiled_runtime and cudagraph_mark_step_begin is not None:
            cudagraph_mark_step_begin()
