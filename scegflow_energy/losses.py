"""The two energy-training objectives described in the manuscript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .model import ConditionalScalarEnergy


@dataclass(frozen=True)
class EnergyLoss:
    total: torch.Tensor
    path: torch.Tensor
    condition: torch.Tensor


def _ranking_loss(lower: torch.Tensor, higher: torch.Tensor, margin: float) -> torch.Tensor:
    return torch.nn.functional.softplus(lower - higher + float(margin)).mean()


def path_ranking_loss(
    model: ConditionalScalarEnergy,
    target_state: torch.Tensor,
    flow_state: torch.Tensor,
    drug_features: torch.Tensor,
    doses: torch.Tensor,
    term_mask: torch.Tensor,
    context_features: torch.Tensor | None = None,
    *,
    margin: float = 1.0,
    alphas: Sequence[float] = (0.75, 0.5, 0.25),
) -> torch.Tensor:
    """Order energy from the measured endpoint toward the frozen flow state."""

    descending_alphas = sorted((float(alpha) for alpha in alphas), reverse=True)
    if any(alpha <= 0 or alpha >= 1 for alpha in descending_alphas):
        raise ValueError("alphas must lie strictly between zero and one")
    states = [target_state]
    states.extend((1 - alpha) * flow_state + alpha * target_state for alpha in descending_alphas)
    states.append(flow_state)
    energies = [
        model(state, drug_features, doses, term_mask, context_features)
        for state in states
    ]
    return torch.stack(
        [_ranking_loss(lower, higher, margin) for lower, higher in zip(energies[:-1], energies[1:])]
    ).sum()


def conditional_discrimination_loss(
    model: ConditionalScalarEnergy,
    target_state: torch.Tensor,
    true_drug_features: torch.Tensor,
    true_doses: torch.Tensor,
    true_term_mask: torch.Tensor,
    corrupted_drug_features: torch.Tensor,
    corrupted_doses: torch.Tensor,
    corrupted_term_mask: torch.Tensor,
    true_context_features: torch.Tensor | None = None,
    corrupted_context_features: torch.Tensor | None = None,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Prefer the true condition to a corrupted condition for the same endpoint."""

    true_energy = model(
        target_state,
        true_drug_features,
        true_doses,
        true_term_mask,
        true_context_features,
    )
    corrupted_energy = model(
        target_state,
        corrupted_drug_features,
        corrupted_doses,
        corrupted_term_mask,
        corrupted_context_features,
    )
    return _ranking_loss(true_energy, corrupted_energy, margin)


def energy_training_objective(
    model: ConditionalScalarEnergy,
    target_state: torch.Tensor,
    flow_state: torch.Tensor,
    true_drug_features: torch.Tensor,
    true_doses: torch.Tensor,
    true_term_mask: torch.Tensor,
    corrupted_drug_features: torch.Tensor,
    corrupted_doses: torch.Tensor,
    corrupted_term_mask: torch.Tensor,
    true_context_features: torch.Tensor | None = None,
    corrupted_context_features: torch.Tensor | None = None,
    *,
    path_weight: float = 1.0,
    condition_weight: float = 1.0,
    margin: float = 1.0,
    alphas: Sequence[float] = (0.75, 0.5, 0.25),
) -> EnergyLoss:
    """Compute ``path_weight * L_path + condition_weight * L_cond``."""

    path = path_ranking_loss(
        model,
        target_state,
        flow_state,
        true_drug_features,
        true_doses,
        true_term_mask,
        true_context_features,
        margin=margin,
        alphas=alphas,
    )
    condition = conditional_discrimination_loss(
        model,
        target_state,
        true_drug_features,
        true_doses,
        true_term_mask,
        corrupted_drug_features,
        corrupted_doses,
        corrupted_term_mask,
        true_context_features,
        corrupted_context_features,
        margin=margin,
    )
    total = float(path_weight) * path + float(condition_weight) * condition
    return EnergyLoss(total=total, path=path, condition=condition)

