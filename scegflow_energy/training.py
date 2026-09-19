"""Small, framework-independent training loop for the scalar energy model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .losses import EnergyLoss, energy_training_objective
from .model import ConditionalScalarEnergy


@dataclass
class EnergyBatch:
    target_state: torch.Tensor
    flow_state: torch.Tensor
    true_drug_features: torch.Tensor
    true_doses: torch.Tensor
    true_term_mask: torch.Tensor
    corrupted_drug_features: torch.Tensor
    corrupted_doses: torch.Tensor
    corrupted_term_mask: torch.Tensor
    true_context_features: torch.Tensor | None = None
    corrupted_context_features: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "EnergyBatch":
        values = {
            name: None if value is None else value.to(device)
            for name, value in vars(self).items()
        }
        return EnergyBatch(**values)


def train_step(
    model: ConditionalScalarEnergy,
    optimizer: torch.optim.Optimizer,
    batch: EnergyBatch,
    *,
    path_weight: float = 1.0,
    condition_weight: float = 1.0,
    margin: float = 1.0,
    alphas: Sequence[float] = (0.75, 0.5, 0.25),
    clip_norm: float | None = None,
) -> EnergyLoss:
    model.train()
    loss = energy_training_objective(
        model,
        batch.target_state,
        batch.flow_state,
        batch.true_drug_features,
        batch.true_doses,
        batch.true_term_mask,
        batch.corrupted_drug_features,
        batch.corrupted_doses,
        batch.corrupted_term_mask,
        batch.true_context_features,
        batch.corrupted_context_features,
        path_weight=path_weight,
        condition_weight=condition_weight,
        margin=margin,
        alphas=alphas,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
    optimizer.step()
    return EnergyLoss(
        total=loss.total.detach(),
        path=loss.path.detach(),
        condition=loss.condition.detach(),
    )


def fit_energy_model(
    model: ConditionalScalarEnergy,
    batches: Iterable[EnergyBatch],
    *,
    epochs: int,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    path_weight: float = 1.0,
    condition_weight: float = 1.0,
    margin: float = 1.0,
    device: torch.device | str = "cpu",
) -> list[dict[str, float]]:
    """Fit the energy model and return epoch-level loss history."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    materialized_batches = list(batches)
    if not materialized_batches:
        raise ValueError("batches must not be empty")
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        totals = {"total": 0.0, "path": 0.0, "condition": 0.0}
        for batch in materialized_batches:
            loss = train_step(
                model,
                optimizer,
                batch.to(device),
                path_weight=path_weight,
                condition_weight=condition_weight,
                margin=margin,
            )
            totals["total"] += float(loss.total)
            totals["path"] += float(loss.path)
            totals["condition"] += float(loss.condition)
        count = len(materialized_batches)
        history.append({"epoch": float(epoch + 1), **{key: value / count for key, value in totals.items()}})
    return history

