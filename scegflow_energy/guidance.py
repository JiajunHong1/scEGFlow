"""Inference-time low-energy refinement."""

from __future__ import annotations

import torch

from .model import ConditionalScalarEnergy


def energy_relaxation_step(
    model: ConditionalScalarEnergy,
    state: torch.Tensor,
    drug_features: torch.Tensor,
    doses: torch.Tensor,
    term_mask: torch.Tensor,
    context_features: torch.Tensor | None = None,
    *,
    strength: float,
    max_update_norm: float | None = None,
) -> torch.Tensor:
    """Apply ``state <- state - strength * dE/dstate`` during inference."""

    if strength < 0:
        raise ValueError("strength must be non-negative")
    with torch.enable_grad():
        differentiable_state = state.detach().requires_grad_(True)
        energy = model(
            differentiable_state,
            drug_features,
            doses,
            term_mask,
            context_features,
        )
        gradient = torch.autograd.grad(energy.sum(), differentiable_state)[0]
        update = float(strength) * gradient
        if max_update_norm is not None:
            if max_update_norm <= 0:
                raise ValueError("max_update_norm must be positive")
            norms = update.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = (float(max_update_norm) / norms).clamp(max=1.0)
            update = update * scale
    return (differentiable_state - update).detach()

