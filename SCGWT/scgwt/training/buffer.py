from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import torch


class RolloutBuffer:
    """Simple FIFO buffer for on-policy rollouts."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._storage: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        self_state: torch.Tensor | None = None,
    ) -> None:
        self._storage.append((observation, action, next_observation, self_state))

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._storage) < batch_size:
            raise ValueError("Not enough samples in buffer for requested batch size")
        batch = random.sample(self._storage, batch_size)
        observations, actions, next_observations, self_states = zip(*batch)
        state_tensor: torch.Tensor | None
        if self_states[0] is None:
            if any(state is not None for state in self_states[1:]):
                raise ValueError("Inconsistent self_state entries in rollout buffer")
            state_tensor = None
        else:
            if any(state is None for state in self_states):
                raise ValueError("Inconsistent self_state entries in rollout buffer")
            state_tensor = torch.stack(self_states)
        return (
            torch.stack(observations),
            torch.stack(actions),
            torch.stack(next_observations),
            state_tensor,
        )

    def __len__(self) -> int:
        return len(self._storage)
