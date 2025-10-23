from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, List, MutableMapping, MutableSequence

import torch
from torch import nn

from .decoder import DecoderConfig, SharedDecoder
from .dynamics import DynamicsConfig, DynamicsModel
from .encoder import EncoderConfig, SlotAttentionEncoder


@dataclass
class WorldModelConfig:
    encoder: EncoderConfig
    decoder: DecoderConfig
    dynamics: DynamicsConfig
    ensemble_size: int = 5


class WorldModelEnsemble(nn.Module):
    """
    Container module bundling the shared encoder/decoder with an ensemble of dynamics models.
    """

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SlotAttentionEncoder(config.encoder)
        self.decoder = SharedDecoder(config.decoder)
        self.frozen_decoder = copy.deepcopy(self.decoder)
        self.dynamics_models = nn.ModuleList(
            [DynamicsModel(config.dynamics) for _ in range(config.ensemble_size)]
        )

    @torch.no_grad()
    def refresh_frozen_decoder(self) -> None:
        """Synchronize the frozen observer head with the trainable decoder."""
        self.frozen_decoder.load_state_dict(self.decoder.state_dict())

    def forward(
        self,
        observation: torch.Tensor,
        output_buffer: MutableMapping[str, torch.Tensor] | None = None,
    ) -> MutableMapping[str, torch.Tensor]:
        """Encode an observation into a dict of latent slots."""
        return self.encoder(observation, output_buffer=output_buffer)

    def predict_next_latents(
        self,
        latent_state: torch.Tensor,
        action: torch.Tensor,
        output_buffer: MutableSequence[torch.Tensor] | None = None,
    ) -> MutableSequence[torch.Tensor]:
        """Run the ensemble forward to obtain next-state predictions."""
        if output_buffer is None:
            return [model(latent_state, action) for model in self.dynamics_models]

        if len(output_buffer) != len(self.dynamics_models):
            raise ValueError("output_buffer length must match ensemble size")

        for dest, model in zip(output_buffer, self.dynamics_models):
            dest.copy_(model(latent_state, action))
        return output_buffer

    def decode_predictions(
        self, predicted_latents: Iterable[torch.Tensor], use_frozen: bool = True
    ) -> List[torch.distributions.Distribution]:
        """Decode predicted latent states into observation distributions."""
        decoder = self.frozen_decoder if use_frozen else self.decoder
        return [decoder(latent) for latent in predicted_latents]
