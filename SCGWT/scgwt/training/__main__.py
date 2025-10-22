from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import minigrid  # registers MiniGrid envs with Gymnasium
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

from scgwt.config import load_training_config
from scgwt.training import TrainingLoop


def _get_visual_frame(env: gym.Env, observation: object) -> np.ndarray:
    frame = env.render()
    if frame is None:
        if isinstance(observation, dict):
            frame = observation.get("rgb") or observation.get("image")
        if frame is None:
            frame = observation
    return np.asarray(frame)


def _frame_to_chw(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], repeats=3, axis=2)
    if array.shape[-1] == 1:
        array = np.repeat(array, repeats=3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return array.transpose(2, 0, 1)


def _preprocess_frame(frame: np.ndarray, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.expand_dims(array, -1)
    if array.shape[-1] == target_shape[0]:
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    elif array.shape[0] == target_shape[0]:
        tensor = torch.from_numpy(array)
    else:
        raise ValueError(f"Incompatible observation shape {array.shape} for expected {target_shape}")
    tensor = tensor.float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0)
    spatial_size = (target_shape[1], target_shape[2])
    if tensor.shape[-2:] != spatial_size:
        tensor = F.interpolate(tensor, size=spatial_size, mode="bilinear", align_corners=False)
    if tensor.shape[1] != target_shape[0]:
        if tensor.shape[1] == 1 and target_shape[0] == 3:
            tensor = tensor.repeat(1, target_shape[0], 1, 1)
        else:
            raise ValueError("Unable to match channel count for observation tensor")
    return tensor.clamp(0.0, 1.0)


def _compute_self_state(env: gym.Env, step_count: int, horizon: int) -> torch.Tensor:
    denom = max(1, horizon)
    energy = max(0.0, 1.0 - step_count / denom)
    goal_progress = 0.0
    agent_pos = getattr(env, "agent_pos", None)
    goal_pos = getattr(env, "goal_pos", None)
    if agent_pos is not None and goal_pos is not None:
        agent = np.asarray(agent_pos, dtype=np.float32)
        goal = np.asarray(goal_pos, dtype=np.float32)
        extent = float(getattr(env, "width", 1) + getattr(env, "height", 1))
        if extent <= 0:
            extent = float(np.abs(goal - agent).sum() + 1.0)
        dist = float(np.abs(agent - goal).sum())
        goal_progress = 1.0 - dist / max(1.0, extent)
        goal_progress = float(np.clip(goal_progress, 0.0, 1.0))
    return torch.tensor([energy, goal_progress], dtype=torch.float32)


def _select_env_action(action_tensor: torch.Tensor, action_space_n: int) -> int:
    if action_tensor.ndim != 2:
        raise ValueError("Expected batched action tensor from TrainingLoop.step")
    usable = min(action_tensor.shape[-1], action_space_n)
    slice_tensor = action_tensor[0, :usable]
    index = int(torch.argmax(slice_tensor).item())
    return index % action_space_n


def main() -> None:
    parser = argparse.ArgumentParser(description="SC-GWT training harness (MiniGrid integration)")
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
    parser.add_argument(
        "--env-id",
        default="MiniGrid-Empty-6x6-v0",
        help="Gymnasium environment identifier to launch.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed.")
    parser.add_argument(
        "--max-steps", type=int, default=5000, help="Total environment steps to execute."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
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

    wandb.init(
        project="scgwt-minigrid",
        config=OmegaConf.to_container(raw_cfg, resolve=True),
        name=f"{args.env_id}_seed{args.seed}",
    )

    loop = TrainingLoop(config)
    env = gym.make(args.env_id, render_mode="rgb_array")
    total_steps = 0
    episode = 0
    episode_steps = 0
    episode_horizon = int(getattr(env, "max_steps", args.max_steps))

    observation, _info = env.reset(seed=args.seed)
    frame = _get_visual_frame(env, observation)
    observation_tensor = _preprocess_frame(frame, config.encoder.observation_shape)
    self_state_vec = _compute_self_state(env, episode_steps, episode_horizon).unsqueeze(0)
    episode_frames = [_frame_to_chw(frame)]

    try:
        while total_steps < args.max_steps:
            with torch.no_grad():
                policy_result = loop.step(
                    observation_tensor,
                    self_state=self_state_vec,
                    train=False,
                )
            env_action = _select_env_action(policy_result.action, env.action_space.n)
            next_observation, _reward, terminated, truncated, info = env.step(env_action)
            next_frame = _get_visual_frame(env, next_observation)
            next_tensor = _preprocess_frame(next_frame, config.encoder.observation_shape)
            training_result = loop.step(
                observation_tensor,
                action=policy_result.action,
                next_observation=next_tensor,
                self_state=self_state_vec,
                train=True,
            )
            episode_frames.append(_frame_to_chw(next_frame))

            next_total_steps = total_steps + 1
            next_episode_steps = episode_steps + 1

            step_metrics = {
                "step/total_steps": next_total_steps,
                "step/episode": episode,
                "step/episode_steps": next_episode_steps,
                "step/intrinsic_reward": float(policy_result.intrinsic_reward.mean().item()),
                "step/observation_entropy": float(policy_result.observation_entropy.mean().item()),
                "step/avg_slot_novelty": float(policy_result.novelty.mean().item()),
            }
            if policy_result.reward_components is not None:
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
                        "step/reward_explore": float(
                            policy_result.reward_components["explore"].mean().item()
                        ),
                    }
                )
            step_metrics.update(
                {
                    "self_state/energy": float(self_state_vec[0, 0].item()),
                    "self_state/goal_progress": float(self_state_vec[0, 1].item()),
                }
            )
            wandb.log(step_metrics, step=next_total_steps)

            if training_result.training_metrics is not None:
                wandb.log(training_result.training_metrics, step=next_total_steps)

            if args.log_interval and next_total_steps % args.log_interval == 0:
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
            self_state_vec = _compute_self_state(env, episode_steps, episode_horizon).unsqueeze(0)
            frame = next_frame

            if terminated or truncated:
                if episode_frames:
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
                observation, _info = env.reset()
                frame = _get_visual_frame(env, observation)
                observation_tensor = _preprocess_frame(frame, config.encoder.observation_shape)
                episode_horizon = int(getattr(env, "max_steps", args.max_steps))
                self_state_vec = _compute_self_state(env, episode_steps, episode_horizon).unsqueeze(0)
                episode_frames = [_frame_to_chw(frame)]
                print(f"Episode {episode} reset (info: {info})")

    finally:
        if total_steps >= args.max_steps and episode_frames and len(episode_frames) > 1:
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
        env.close()
        wandb.finish()


if __name__ == "__main__":
    main()
