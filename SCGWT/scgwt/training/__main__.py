from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import crafter
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

from scgwt.config import load_training_config
from scgwt.training import TrainingLoop


def _scalarize(value: Any) -> float | int | bool:
    """Convert tensors and numpy scalars to plain Python types for logging."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Only scalar tensors can be logged to Weights & Biases.")
        value = value.detach().cpu().item()
    elif isinstance(value, np.generic):
        value = value.item()
    return value


def _sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, float | int | bool]:
    sanitized: Dict[str, float | int | bool] = {}
    for key, value in metrics.items():
        scalar = _scalarize(value)
        if isinstance(scalar, (bool, int, float)):
            sanitized[key] = scalar
        else:
            sanitized[key] = float(scalar)
    return sanitized


def _frame_to_chw(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], repeats=3, axis=2)
    if array.shape[-1] == 1:
        array = np.repeat(array, repeats=3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return array.transpose(2, 0, 1)


def _preprocess_frame(
    frame: np.ndarray, target_shape: Tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.expand_dims(array, -1)
    tensor = torch.from_numpy(array)
    if tensor.ndim != 3:
        raise ValueError(f"Observation must be [H, W, C] or [C, H, W], got {tensor.shape}")
    if tensor.shape[-1] == target_shape[0]:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.shape[0] != target_shape[0]:
        raise ValueError(f"Incompatible observation shape {tensor.shape} for expected {target_shape}")
    tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last)
    spatial_size = (target_shape[1], target_shape[2])
    if tensor.shape[-2:] != spatial_size:
        tensor = F.interpolate(tensor, size=spatial_size, mode="bilinear", align_corners=False)
    if tensor.shape[1] != target_shape[0]:
        if tensor.shape[1] == 1 and target_shape[0] == 3:
            tensor = tensor.repeat(1, target_shape[0], 1, 1)
        else:
            raise ValueError("Unable to match channel count for observation tensor")
    return tensor.clamp(0.0, 1.0)


def _compute_self_state(
    info: dict | None, step_count: int, horizon: int, state_dim: int
) -> torch.Tensor:
    """Derive self-centric signals from Crafter status fields."""
    if state_dim <= 0:
        return torch.empty(0, dtype=torch.float32)

    if info is None:
        info = {}

    health = float(info.get("health", 9.0))
    food = float(info.get("food", 9.0))
    health_norm = np.clip(health / 9.0, 0.0, 1.0)
    food_norm = np.clip(food / 9.0, 0.0, 1.0)

    denom = max(1, horizon)
    energy = max(0.0, 1.0 - step_count / denom)
    is_sleeping = float(info.get("is_sleeping", 0.0))

    features: List[float] = [health_norm, food_norm, energy, is_sleeping]
    if state_dim <= len(features):
        selected = features[:state_dim]
    else:
        selected = features + [0.0] * (state_dim - len(features))
    return torch.tensor(selected, dtype=torch.float32)


def _select_env_action(action_tensor: torch.Tensor, action_space_n: int) -> int:
    if action_tensor.ndim != 2:
        raise ValueError("Expected batched action tensor from TrainingLoop.step")
    usable = min(action_tensor.shape[-1], action_space_n)
    slice_tensor = action_tensor[0, :usable]
    index = int(torch.argmax(slice_tensor).item())
    return index % action_space_n


def main() -> None:
    parser = argparse.ArgumentParser(description="SC-GWT training harness (Crafter integration)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override configuration values (OmegaConf dotlist syntax).",
    )
    parser.add_argument("--device", default=None, help="Runtime device override.")
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed.")
    parser.add_argument(
        "--max-steps", type=int, default=5000, help="Total environment steps to execute."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="How frequently to print intrinsic reward diagnostics.",
    )
    args = parser.parse_args()

    raw_cfg = OmegaConf.load(args.config)
    if args.override:
        raw_cfg = OmegaConf.merge(raw_cfg, OmegaConf.from_dotlist(list(args.override)))

    config = load_training_config(args.config, overrides=args.override)
    if args.device:
        config.device = args.device
        raw_cfg.device = args.device

    runtime_device = torch.device(config.device)

    wandb.init(
        project="scgwt-crafter",
        config=OmegaConf.to_container(raw_cfg, resolve=True),
        name=f"crafter_seed{args.seed}",
    )

    loop = TrainingLoop(config)
    env = crafter.Env()
    total_steps = 0
    episode = 0
    episode_steps = 0
    episode_horizon = args.max_steps

    observation = env.reset()
    frame = observation
    observation_tensor = _preprocess_frame(frame, config.encoder.observation_shape, runtime_device)
    self_state_vec = _compute_self_state(
        info=None, step_count=episode_steps, horizon=episode_horizon, state_dim=config.self_state_dim
    ).unsqueeze(0).to(runtime_device)
    log_videos = bool(getattr(config, "log_videos", getattr(config, "log_images", True)))
    video_log_freq = max(1, int(getattr(config, "video_log_freq", 1)))
    capture_videos = log_videos and getattr(config, "log_images", True)

    episode_frames = [_frame_to_chw(frame)] if capture_videos else []

    log_interval = args.log_interval if args.log_interval is not None else config.log_every_steps
    if not log_interval or log_interval <= 0:
        log_interval = 1

    step_metric_accumulator: Dict[str, List[float]] = defaultdict(list)
    passthrough_step_keys = {"step/total_steps", "step/episode", "step/episode_steps"}

    def flush_step_metrics(step_idx: int) -> None:
        if not step_metric_accumulator:
            return
        aggregated: Dict[str, float] = {}
        for key, values in step_metric_accumulator.items():
            if not values:
                continue
            if key in passthrough_step_keys:
                aggregated[key] = float(values[-1])
            else:
                aggregated[f"{key}_mean"] = float(np.mean(values))
        step_metric_accumulator.clear()
        if aggregated:
            wandb.log(_sanitize_metrics(aggregated), step=step_idx)

    try:
        while total_steps < args.max_steps:
            with torch.no_grad():
                policy_result = loop.step(
                    observation_tensor,
                    self_state=self_state_vec,
                    train=False,
                )
            env_action = _select_env_action(policy_result.action, env.action_space.n)
            next_observation, env_reward, terminated, info = env.step(env_action)
            truncated = False
            next_tensor = _preprocess_frame(next_observation, config.encoder.observation_shape, runtime_device)
            training_result = loop.step(
                observation_tensor,
                action=policy_result.action,
                next_observation=next_tensor,
                self_state=self_state_vec,
                train=True,
            )
            if capture_videos:
                episode_frames.append(_frame_to_chw(next_observation))

            next_total_steps = total_steps + 1
            next_episode_steps = episode_steps + 1
            next_self_state_vec = _compute_self_state(
                info,
                next_episode_steps,
                episode_horizon,
                config.self_state_dim,
            ).unsqueeze(0).to(runtime_device)

            step_metrics = {
                "step/total_steps": next_total_steps,
                "step/episode": episode,
                "step/episode_steps": next_episode_steps,
                "step/intrinsic_reward": float(policy_result.intrinsic_reward.mean().item()),
                "step/observation_entropy": float(policy_result.observation_entropy.mean().item()),
                "step/avg_slot_novelty": float(policy_result.novelty.mean().item()),
                "step/env_reward": float(env_reward),
            }
            if isinstance(info, dict):
                player_stats = ["health", "food", "drink", "energy"]
                for stat in player_stats:
                    if stat in info:
                        step_metrics[f"crafter_stats/{stat}"] = float(info[stat])
                if isinstance(info.get("achievements"), dict):
                    step_metrics["crafter_stats/achievements_unlocked"] = len(info["achievements"])
            if policy_result.reward_components is not None:
                explore_tensor = policy_result.reward_components["explore"]
                explore_value = float(explore_tensor.mean().item())
                raw_components = policy_result.raw_reward_components or {}
                raw_explore_value = (
                    float(raw_components["explore"].mean().item())
                    if "explore" in raw_components
                    else explore_value
                )
                step_metrics.update(
                    {
                        "step/reward_competence": float(
                            policy_result.reward_components["competence"].mean().item()
                        ),
                        "step/reward_empowerment": float(
                            policy_result.reward_components["empowerment"].mean().item()
                        ),
                        "step/reward_safety": float(
                            policy_result.reward_components["safety"].mean().item()
                        ),
                        "step/reward_explore_raw": raw_explore_value,
                        "step/reward_explore": float(max(explore_value, 0.0)),
                    }
                )
            state_names = ["health_norm", "food_norm", "energy_step", "is_sleeping"]
            if next_self_state_vec.numel() > 0:
                for idx in range(next_self_state_vec.shape[1]):
                    name = state_names[idx] if idx < len(state_names) else f"feature_{idx}"
                    step_metrics[f"self_state/{name}"] = float(next_self_state_vec[0, idx].item())

            for key, value in step_metrics.items():
                scalar = _scalarize(value)
                step_metric_accumulator[key].append(float(scalar))

            should_flush_metrics = (
                next_total_steps % log_interval == 0
                or terminated
                or truncated
                or next_total_steps >= args.max_steps
            )
            if should_flush_metrics:
                flush_step_metrics(next_total_steps)

            if training_result.training_metrics is not None:
                wandb.log(
                    _sanitize_metrics(training_result.training_metrics),
                    step=next_total_steps,
                )

            if log_interval and next_total_steps % log_interval == 0:
                intrinsic = policy_result.intrinsic_reward.mean().item()
                novelty = policy_result.novelty.mean().item()
                entropy = policy_result.observation_entropy.mean().item()
                loss_str = (
                    f"{training_result.training_loss:.4f}"
                    if training_result.training_loss is not None
                    else "n/a"
                )
                print(
                    f"[step {next_total_steps:05d}] intrinsic={intrinsic:.4f} "
                    f"novelty={novelty:.4f} entropy={entropy:.4f} loss={loss_str}"
                )

            observation_tensor = next_tensor
            total_steps = next_total_steps
            episode_steps = next_episode_steps
            self_state_vec = next_self_state_vec
            frame = next_observation

            if terminated or truncated:
                if capture_videos and episode_frames and episode % video_log_freq == 0:
                    video_array = np.stack(episode_frames, axis=0)
                    wandb.log(
                        {
                            "episode/video": wandb.Video(
                                video_array,
                                fps=8,
                                caption=f"Episode {episode} (info: {info})",
                            )
                        },
                        step=next_total_steps,
                    )
                episode += 1
                episode_steps = 0
                if isinstance(info, dict) and isinstance(info.get("achievements"), dict):
                    wandb.log(
                        {"episode/final_achievements": len(info["achievements"])},
                        step=next_total_steps,
                    )
                observation = env.reset()
                frame = observation
                observation_tensor = _preprocess_frame(frame, config.encoder.observation_shape, runtime_device)
                episode_horizon = args.max_steps
                self_state_vec = _compute_self_state(
                    info=None, step_count=episode_steps, horizon=episode_horizon, state_dim=config.self_state_dim
                ).unsqueeze(0).to(runtime_device)
                episode_frames = [_frame_to_chw(frame)] if capture_videos else []
                print(f"Episode {episode} reset (info: {info})")

    finally:
        flush_step_metrics(total_steps)
        if (
            capture_videos
            and total_steps >= args.max_steps
            and episode_frames
            and len(episode_frames) > 1
            and episode % video_log_freq == 0
        ):
            video_array = np.stack(episode_frames, axis=0)
            wandb.log(
                {
                    "episode/video_truncated": wandb.Video(
                        video_array,
                        fps=8,
                        caption=f"Episode {episode} (truncated at step limit)",
                    )
                },
                step=total_steps,
            )
        wandb.finish()


if __name__ == "__main__":
    main()



