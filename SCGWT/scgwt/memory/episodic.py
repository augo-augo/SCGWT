from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
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
        self._cpu_index = faiss.IndexFlatL2(config.key_dim)
        self._gpu_resources: faiss.StandardGpuResources | None = None
        self._pending_keys: List[np.ndarray] = []
        self._trained = False
        self._using_gpu = False
        self._nlist = max(1, min(4096, max(1, config.capacity // 8)))
        self._train_threshold = min(config.capacity, max(self._nlist * 2, 32))
        self.index = self._cpu_index
        self._init_index()
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
        key_cpu = key.detach().to(dtype=torch.float32, device="cpu").contiguous()
        key_np = np.ascontiguousarray(key_cpu.numpy())
        value_cpu = value.detach().to(device="cpu").contiguous()
        self._cpu_index.add(key_np)
        if self._using_gpu:
            self._add_gpu_keys(key_np)
        for idx in range(batch):
            self.values[self.next_id] = value_cpu[idx]
            self.next_id += 1
        if self._using_gpu and not self._trained:
            total_pending = sum(arr.shape[0] for arr in self._pending_keys)
            if total_pending >= self._train_threshold:
                self._finalize_gpu_index()

    def read(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 2:
            raise ValueError("query must be shape [batch, key_dim]")
        target_device = query.device
        backend = self.index if (self._using_gpu and self._trained) else self._cpu_index
        distances, indices = backend.search(query.detach().cpu().numpy(), k)
        fallback = torch.zeros_like(query[0]).cpu()
        retrieved_cpu = [self.values.get(idx, fallback) for idx in indices.flatten()]
        stacked_cpu = torch.stack(retrieved_cpu).view(query.shape[0], k, -1)
        values = stacked_cpu.to(target_device)
        distances_tensor = torch.from_numpy(distances).to(target_device)
        return distances_tensor, values

    def _init_index(self) -> None:
        try:
            self._gpu_resources = faiss.StandardGpuResources()
            gpu_index = faiss.GpuIndexIVFFlat(
                self._gpu_resources,
                self.config.key_dim,
                self._nlist,
                faiss.METRIC_L2,
            )
            gpu_index.setNumProbes(min(32, self._nlist))
            self.index = gpu_index
            self._using_gpu = True
            self._trained = False
            self._train_threshold = min(self.config.capacity, max(self._nlist * 2, 32))
        except Exception:
            self.index = self._cpu_index
            self._using_gpu = False
            self._trained = True
            self._pending_keys.clear()

    def _add_gpu_keys(self, key_array: np.ndarray) -> None:
        key_np = np.ascontiguousarray(key_array)
        if not self._trained:
            self._pending_keys.append(key_np.copy())
        else:
            self.index.add(key_np)

    def _finalize_gpu_index(self) -> None:
        if not self._using_gpu or self._trained:
            return
        if not self._pending_keys:
            return
        train_data = np.concatenate(self._pending_keys, axis=0)
        self.index.train(train_data)
        self.index.add(train_data)
        self._pending_keys.clear()
        self._trained = True

    def _evict_oldest(self) -> None:
        oldest_idx = min(self.values)
        del self.values[oldest_idx]
        # FAISS does not support removing individual entries from IndexFlatL2; this will be
        # revisited when swapping to an IVF index or a rolling rebuild strategy.
