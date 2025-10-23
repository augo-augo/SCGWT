from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Tuple

import crafter
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

from scgwt.config import load_training_config
from scgwt.training import TrainingLoop


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


def _actor_loop(
    worker_id: int,
    loop: TrainingLoop,
    config,
    runtime_device: torch.device,
    max_steps: int,
    log_interval: int,
    steps_lock: threading.Lock,
    policy_lock: threading.Lock,
    shared_state: Dict[str, int],
    stop_event: threading.Event,
    metrics_queue: Queue,
    seed: int,
) -> None:
    env = crafter.Env()
    try:
        if hasattr(env, "seed"):
            env.seed(seed)
        episode_frames: List[np.ndarray] = []
        episode_steps = 0
        observation = env.reset()
        frame = observation
        observation_tensor = _preprocess_frame(frame, config.encoder.observation_shape, runtime_device)
        with steps_lock:
            shared_state["episodes"] += 1
            episode_id = shared_state["episodes"]
        self_state_vec = _compute_self_state(
            info=None,
            step_count=episode_steps,
            horizon=max_steps,
            state_dim=config.self_state_dim,
        ).unsqueeze(0).to(runtime_device)
        episode_frames = [_frame_to_chw(frame)]
        while not stop_event.is_set():
            with torch.no_grad():
                with policy_lock:
                    policy_result = loop.step(
                        observation_tensor,
                        self_state=self_state_vec if self_state_vec.numel() > 0 else None,
                        train=False,
                    )
            env_action = _select_env_action(policy_result.action, env.action_space.n)
            next_observation, env_reward, terminated, info = env.step(env_action)
            truncated = False
            next_tensor = _preprocess_frame(
                next_observation, config.encoder.observation_shape, runtime_device
            )
            loop.store_transition(
                observation_tensor,
                policy_result.action,
                next_tensor,
                self_state_vec if self_state_vec.numel() > 0 else None,
            )
            next_episode_steps = episode_steps + 1
            next_self_state_vec = _compute_self_state(
                info,
                next_episode_steps,
                max_steps,
                config.self_state_dim,
            ).unsqueeze(0).to(runtime_device)
            episode_frames.append(_frame_to_chw(next_observation))
            with steps_lock:
                if shared_state["steps"] >= max_steps:
                    stop_event.set()
                    step_index = shared_state["steps"]
                    reached_limit = True
                else:
                    shared_state["steps"] += 1
                    step_index = shared_state["steps"]
                    reached_limit = shared_state["steps"] >= max_steps
                    if reached_limit:
                        stop_event.set()
            info_dict = info if isinstance(info, dict) else {}
            reward_components = {}
            if policy_result.reward_components is not None:
                reward_components = {
                    name: float(value.mean().item())
                    for name, value in policy_result.reward_components.items()
                }
            raw_components = {}
            if policy_result.raw_reward_components is not None:
                raw_components = {
                    name: float(value.mean().item())
                    for name, value in policy_result.raw_reward_components.items()
                }
            self_state_list: List[float] = []
            if next_self_state_vec.numel() > 0:
                self_state_list = [float(x) for x in next_self_state_vec.squeeze(0).tolist()]
            should_log = log_interval > 0 and step_index % log_interval == 0
            achievements = info_dict.get("achievements") if isinstance(info_dict, dict) else None
            achievements_count = len(achievements) if isinstance(achievements, dict) else 0
            metrics_queue.put(
                {
                    "kind": "step",
                    "worker": worker_id,
                    "step": step_index,
                    "episode": episode_id,
                    "episode_steps": next_episode_steps,
                    "intrinsic": float(policy_result.intrinsic_reward.mean().item()),
                    "novelty": float(policy_result.novelty.mean().item()),
                    "entropy": float(policy_result.observation_entropy.mean().item()),
                    "env_reward": float(env_reward),
                    "reward_components": reward_components,
                    "raw_reward_components": raw_components,
                    "self_state": self_state_list,
                    "info": info_dict,
                    "log": should_log,
                    "done": terminated or truncated or reached_limit,
                    "achievements_count": achievements_count,
                }
            )
            done = terminated or truncated or reached_limit
            if done and episode_frames and len(episode_frames) > 1:
                try:
                    video_array = np.stack(episode_frames, axis=0)
                except ValueError:
                    video_array = None
                if video_array is not None:
                    metrics_queue.put(
                        {
                            "kind": "video",
                            "worker": worker_id,
                            "step": step_index,
                            "episode": episode_id,
                            "frames": video_array,
                            "info": info_dict,
                            "truncated": reached_limit and not terminated and not truncated,
                        }
                    )
            if done:
                if shared_state["steps"] >= max_steps:
                    break
                observation = env.reset()
                frame = observation
                observation_tensor = _preprocess_frame(
                    frame, config.encoder.observation_shape, runtime_device
                )
                episode_steps = 0
                with steps_lock:
                    shared_state["episodes"] += 1
                    episode_id = shared_state["episodes"]
                episode_frames = [_frame_to_chw(frame)]
                self_state_vec = _compute_self_state(
                    info=None,
                    step_count=episode_steps,
                    horizon=max_steps,
                    state_dim=config.self_state_dim,
                ).unsqueeze(0).to(runtime_device)
                continue
            observation_tensor = next_tensor
            self_state_vec = next_self_state_vec
            episode_steps = next_episode_steps
    finally:
        if hasattr(env, "close"):
            env.close()


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
        default=50,
        help="How frequently to print intrinsic reward diagnostics.",
    )
    parser.add_argument(
        "--actor-workers",
        type=int,
        default=2,
        help="Number of parallel actor threads to use for experience collection.",
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
    wandb.define_metric("step/total_steps", summary="max")
    wandb.define_metric("step/*", step_metric="step/total_steps")
    wandb.define_metric("train/*", step_metric="step/total_steps")
    wandb.define_metric("dream/*", step_metric="step/total_steps")

    loop = TrainingLoop(config)
    num_workers = max(1, args.actor_workers)
    stop_event = threading.Event()
    steps_lock = threading.Lock()
    policy_lock = threading.Lock()
    shared_state: Dict[str, int] = {"steps": 0, "episodes": 0}
    metrics_queue: Queue = Queue()

    actor_threads = []
    for worker_id in range(num_workers):
        worker_seed = args.seed + worker_id
        thread = threading.Thread(
            target=_actor_loop,
            args=(
                worker_id,
                loop,
                config,
                runtime_device,
                args.max_steps,
                args.log_interval,
                steps_lock,
                policy_lock,
                shared_state,
                stop_event,
                metrics_queue,
                worker_seed,
            ),
            daemon=True,
        )
        thread.start()
        actor_threads.append(thread)

    latest_training_loss: float | None = None
    try:
        while True:
            processed = False
            while True:
                try:
                    message = metrics_queue.get_nowait()
                except Empty:
                    break
                processed = True
                if message["kind"] == "step":
                    step_value = int(message["step"])
                    step_metrics = {
                        "step/total_steps": step_value,
                        "step/episode": int(message["episode"]),
                        "step/episode_steps": int(message["episode_steps"]),
                        "step/intrinsic_reward": float(message["intrinsic"]),
                        "step/observation_entropy": float(message["entropy"]),
                        "step/avg_slot_novelty": float(message["novelty"]),
                        "step/env_reward": float(message["env_reward"]),
                    }
                    reward_components: Dict[str, float] = message.get("reward_components", {})  # type: ignore[arg-type]
                    raw_components: Dict[str, float] = message.get("raw_reward_components", {})  # type: ignore[arg-type]
                    if reward_components:
                        step_metrics.update(
                            {
                                "step/reward_competence": reward_components.get("competence", 0.0),
                                "step/reward_empowerment": reward_components.get("empowerment", 0.0),
                                "step/reward_safety": reward_components.get("safety", 0.0),
                            }
                        )
                        explore_value = reward_components.get("explore", 0.0)
                        raw_explore_value = raw_components.get("explore", explore_value)
                        step_metrics["step/reward_explore_raw"] = raw_explore_value
                        step_metrics["step/reward_explore"] = max(explore_value, 0.0)
                    info_dict: Dict[str, object] = message.get("info", {})  # type: ignore[assignment]
                    player_stats = ["health", "food", "drink", "energy"]
                    for stat in player_stats:
                        value = info_dict.get(stat)
                        if isinstance(value, (int, float)):
                            step_metrics[f"crafter_stats/{stat}"] = float(value)
                    achievements = info_dict.get("achievements")
                    if isinstance(achievements, dict):
                        step_metrics["crafter_stats/achievements_unlocked"] = len(achievements)
                    self_state_values: List[float] = message.get("self_state", [])  # type: ignore[assignment]
                    state_names = ["health_norm", "food_norm", "energy_step", "is_sleeping"]
                    for idx, value in enumerate(self_state_values):
                        name = state_names[idx] if idx < len(state_names) else f"feature_{idx}"
                        step_metrics[f"self_state/{name}"] = float(value)
                    wandb.log(step_metrics, step=step_value)
                    if message.get("done") and message.get("achievements_count", 0):
                        wandb.log(
                            {"episode/final_achievements": int(message["achievements_count"])},
                            step=step_value,
                        )
                    if message.get("log"):
                        loss_str = (
                            f"{latest_training_loss:.4f}" if latest_training_loss is not None else "n/a"
                        )
                        print(
                            f"[worker {message['worker']} step {step_value:05d}] "
                            f"intrinsic={step_metrics['step/intrinsic_reward']:.4f} "
                            f"novelty={step_metrics['step/avg_slot_novelty']:.4f} "
                            f"entropy={step_metrics['step/observation_entropy']:.4f} "
                            f"loss={loss_str}"
                        )
                elif message["kind"] == "video":
                    frames = message.get("frames")
                    if isinstance(frames, np.ndarray) and frames.shape[0] > 1:
                        label = "episode/video_truncated" if message.get("truncated") else "episode/video"
                        caption = (
                            f"Worker {message['worker']} Episode {message['episode']} (info: {message.get('info')})"
                        )
                        wandb.log(
                            {label: wandb.Video(frames, fps=8, format="gif", caption=caption)},
                            step=int(message.get("step", 0)),
                        )
            metrics = loop._optimize()
            if metrics:
                with steps_lock:
                    current_step = shared_state["steps"]
                wandb.log(metrics, step=current_step)
                latest_training_loss = metrics.get("train/total_loss")
                processed = True
            if (
                stop_event.is_set()
                and all(not thread.is_alive() for thread in actor_threads)
                and metrics_queue.empty()
            ):
                break
            if not processed:
                time.sleep(0.001)
    finally:
        stop_event.set()
        for thread in actor_threads:
            thread.join()
        wandb.finish()


if __name__ == "__main__":
    main()



