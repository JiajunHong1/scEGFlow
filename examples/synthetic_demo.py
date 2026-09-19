"""Run one synthetic energy-training step and one refinement step."""

from __future__ import annotations

import torch

from scegflow_energy import (
    ConditionalScalarEnergy,
    EnergyBatch,
    energy_relaxation_step,
    train_step,
)


def main() -> None:
    torch.manual_seed(7)
    batch_size, state_dim, drug_dim = 16, 32, 12
    model = ConditionalScalarEnergy(
        state_dim=state_dim,
        drug_feature_dim=drug_dim,
        context_feature_dim=3,
        latent_dim=16,
        condition_dim=8,
        hidden_dim=32,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.randn(batch_size, state_dim)
    flow = target + 0.4 * torch.randn_like(target)
    true_drug = torch.randn(batch_size, 1, drug_dim)
    wrong_drug = torch.roll(true_drug, shifts=1, dims=0)
    doses = torch.ones(batch_size, 1)
    mask = torch.ones(batch_size, 1, dtype=torch.bool)
    context = torch.nn.functional.one_hot(torch.arange(batch_size) % 3, num_classes=3).float()
    batch = EnergyBatch(
        target_state=target,
        flow_state=flow,
        true_drug_features=true_drug,
        true_doses=doses,
        true_term_mask=mask,
        corrupted_drug_features=wrong_drug,
        corrupted_doses=doses,
        corrupted_term_mask=mask,
        true_context_features=context,
        corrupted_context_features=torch.roll(context, shifts=1, dims=0),
    )
    loss = train_step(model, optimizer, batch)
    refined = energy_relaxation_step(
        model,
        flow,
        true_drug,
        doses,
        mask,
        context,
        strength=0.05,
    )
    print(
        f"loss={float(loss.total):.4f} "
        f"path={float(loss.path):.4f} "
        f"condition={float(loss.condition):.4f} "
        f"mean_update_norm={float((refined - flow).norm(dim=1).mean()):.6f}"
    )


if __name__ == "__main__":
    main()

