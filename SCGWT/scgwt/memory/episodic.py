from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import faiss


@dataclass
class EpisodicBufferConfig:
    capacity: int
    key_dim: int


class EpisodicBuffer:
    """Thin wrapper around a FAISS index for approximate episodic recall."""

    def __init__(self, config: EpisodicBufferConfig) -> None:
        self.config = config
        self.index = faiss.IndexFlatL2(config.key_dim)
        self.values: Dict[int, torch.Tensor] = {}
        self.next_id = 0

    def __len__(self) -> int:
        return len(self.values)

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.ndim != 2:
            raise ValueError("key must be shape [batch, key_dim]")
        if key.shape[1] != self.config.key_dim:
            raise ValueError("key dimension mismatch")
        if len(self) >= self.config.capacity:
            self._evict_oldest()
        batch = key.shape[0]
        self.index.add(key.detach().cpu().numpy())
        for row in value.detach():
            self.values[self.next_id] = row
            self.next_id += 1

    def read(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 2:
            raise ValueError("query must be shape [batch, key_dim]")
        distances, indices = self.index.search(query.detach().cpu().numpy(), k)
        retrieved = [self.values.get(idx, torch.zeros_like(query[0])) for idx in indices.flatten()]
        values = torch.stack(retrieved).view(query.shape[0], k, -1)
        return torch.from_numpy(distances), values

    def _evict_oldest(self) -> None:
        oldest_idx = min(self.values)
        del self.values[oldest_idx]
        # FAISS does not support removing individual entries from IndexFlatL2; this will be
        # revisited when swapping to an IVF index or a rolling rebuild strategy.
