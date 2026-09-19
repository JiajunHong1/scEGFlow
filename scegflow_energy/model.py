"""Conditional scalar energy architecture."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _mlp(dimensions: Sequence[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
        layers.append(nn.Linear(input_dim, output_dim))
        if output_dim != dimensions[-1]:
            layers.extend((nn.LayerNorm(output_dim), nn.SiLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ConditionalScalarEnergy(nn.Module):
    """Return one compatibility energy per cellular state and condition."""

    def __init__(
        self,
        state_dim: int,
        drug_feature_dim: int,
        context_feature_dim: int = 0,
        latent_dim: int = 128,
        condition_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.drug_feature_dim = int(drug_feature_dim)
        self.context_feature_dim = int(context_feature_dim)

        self.state_encoder = _mlp((state_dim, hidden_dim, hidden_dim, latent_dim), dropout)
        self.drug_encoder = _mlp((drug_feature_dim, hidden_dim, condition_dim), dropout)
        self.dose_encoder = _mlp((1, max(8, hidden_dim // 4), condition_dim), dropout)
        if context_feature_dim:
            self.context_encoder: nn.Module | None = _mlp(
                (context_feature_dim, max(8, hidden_dim // 4), condition_dim), dropout
            )
        else:
            self.context_encoder = None
        energy_input_dim = latent_dim + 2 * condition_dim
        if self.context_encoder is not None:
            energy_input_dim += condition_dim
        self.energy_head = _mlp((energy_input_dim, hidden_dim, max(8, hidden_dim // 2), 1), dropout)

        mean = torch.zeros(state_dim) if input_mean is None else input_mean.detach().float()
        std = torch.ones(state_dim) if input_std is None else input_std.detach().float()
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std.clamp_min(1e-6))

    def forward(
        self,
        state: torch.Tensor,
        drug_features: torch.Tensor,
        doses: torch.Tensor,
        term_mask: torch.Tensor,
        context_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate ``E(state, condition)`` for a padded condition batch."""

        if drug_features.ndim != 3:
            raise ValueError("drug_features must have shape [batch, terms, features]")
        if doses.shape != term_mask.shape or doses.shape != drug_features.shape[:2]:
            raise ValueError("doses and term_mask must match the first two drug_features dimensions")
        state_embedding = self.state_encoder((state - self.input_mean) / self.input_std)
        term_count = drug_features.shape[1]
        state_terms = state_embedding.unsqueeze(1).expand(-1, term_count, -1)
        drug_embedding = self.drug_encoder(drug_features)
        dose_embedding = self.dose_encoder(torch.log1p(doses.clamp_min(0)).unsqueeze(-1))
        inputs = [state_terms, drug_embedding, dose_embedding]

        if self.context_encoder is not None:
            if context_features is None:
                raise ValueError("context_features are required when context_feature_dim is non-zero")
            context_embedding = self.context_encoder(context_features)
            inputs.append(context_embedding.unsqueeze(1).expand(-1, term_count, -1))
        elif context_features is not None:
            raise ValueError("context_features were provided to a model without a context encoder")

        term_energy = self.energy_head(torch.cat(inputs, dim=-1)).squeeze(-1)
        weights = term_mask.to(dtype=term_energy.dtype)
        if torch.any(weights.sum(dim=1) == 0):
            raise ValueError("each condition must contain at least one valid term")
        return (term_energy * weights).sum(dim=1) / weights.sum(dim=1)

