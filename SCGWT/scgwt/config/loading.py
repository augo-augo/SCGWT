from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from omegaconf import OmegaConf

from scgwt.training import TrainingConfig
from scgwt.agents import ActorConfig, CriticConfig
from scgwt.world_model import DecoderConfig, DynamicsConfig, EncoderConfig
from scgwt.cognition import WorkspaceConfig
from scgwt.motivation import EmpowermentConfig, IntrinsicRewardConfig
from scgwt.memory import EpisodicBufferConfig


def load_training_config(path: str | Path, overrides: Iterable[str] | None = None) -> TrainingConfig:
    """Load a TrainingConfig dataclass from a YAML file with optional overrides."""
    cfg = OmegaConf.load(Path(path))
    if overrides:
        override_conf = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_conf)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Configuration root must be a mapping")

    encoder = EncoderConfig(**_extract_section(resolved, "encoder"))
    decoder = DecoderConfig(**_extract_section(resolved, "decoder"))
    dynamics = DynamicsConfig(**_extract_section(resolved, "dynamics"))
    workspace = WorkspaceConfig(**_extract_section(resolved, "workspace"))
    reward = IntrinsicRewardConfig(**_extract_section(resolved, "reward"))
    empowerment = EmpowermentConfig(**_extract_section(resolved, "empowerment"))
    episodic = EpisodicBufferConfig(**_extract_section(resolved, "episodic_memory"))
    actor_mapping = resolved.get("actor")
    if isinstance(actor_mapping, Mapping):
        actor_section = actor_mapping
    else:
        actor_section = {}
    critic_mapping = resolved.get("critic")
    if isinstance(critic_mapping, Mapping):
        critic_section = critic_mapping
    else:
        critic_section = {}
    actor_cfg = ActorConfig(
        latent_dim=0,
        action_dim=0,
        hidden_dim=actor_section.get("hidden_dim", 256),
        num_layers=actor_section.get("num_layers", 2),
        dropout=actor_section.get("dropout", 0.0),
    )
    critic_cfg = CriticConfig(
        latent_dim=0,
        hidden_dim=critic_section.get("hidden_dim", 256),
        num_layers=critic_section.get("num_layers", 2),
        dropout=critic_section.get("dropout", 0.0),
    )

    world_model_ensemble = resolved.get("world_model_ensemble")
    if world_model_ensemble is None:
        raise KeyError("world_model_ensemble is required in the configuration")
    rollout_capacity = resolved.get("rollout_capacity", 1024)
    batch_size = resolved.get("batch_size", 32)
    optimizer_lr = resolved.get("optimizer_lr", 1e-3)
    optimizer_empowerment_weight = resolved.get("optimizer_empowerment_weight", 0.1)
    dream_horizon = resolved.get("dream_horizon", 5)
    discount_gamma = resolved.get("discount_gamma", 0.99)
    gae_lambda = resolved.get("gae_lambda", 0.95)
    entropy_coef = resolved.get("entropy_coef", 0.01)
    critic_coef = resolved.get("critic_coef", 0.5)
    world_model_coef = resolved.get("world_model_coef", 1.0)
    device = resolved.get("device", "cpu")
    self_state_dim = resolved.get("self_state_dim", 0)

    return TrainingConfig(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        world_model_ensemble=world_model_ensemble,
        workspace=workspace,
        reward=reward,
        empowerment=empowerment,
        episodic_memory=episodic,
        rollout_capacity=rollout_capacity,
        batch_size=batch_size,
        optimizer_lr=optimizer_lr,
        optimizer_empowerment_weight=optimizer_empowerment_weight,
        actor=actor_cfg,
        critic=critic_cfg,
        dream_horizon=dream_horizon,
        discount_gamma=discount_gamma,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef,
        world_model_coef=world_model_coef,
        self_state_dim=self_state_dim,
        device=device,
    )


def _extract_section(resolved: Mapping[str, object], key: str) -> Mapping[str, object]:
    section = resolved.get(key)
    if not isinstance(section, Mapping):
        raise KeyError(f"Configuration missing required mapping: {key}")
    return section
