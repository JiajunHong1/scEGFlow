import unittest

import torch

from scegflow_energy import (
    ConditionalScalarEnergy,
    conditional_discrimination_loss,
    energy_relaxation_step,
    path_ranking_loss,
)


def _inputs(batch_size: int = 4):
    state = torch.randn(batch_size, 6)
    drug = torch.randn(batch_size, 2, 5)
    doses = torch.ones(batch_size, 2)
    mask = torch.tensor([[True, True]] * batch_size)
    context = torch.randn(batch_size, 3)
    return state, drug, doses, mask, context


class EnergyTests(unittest.TestCase):
    def test_model_returns_one_energy_per_state(self) -> None:
        model = ConditionalScalarEnergy(6, 5, 3, latent_dim=4, condition_dim=4, hidden_dim=8, dropout=0)
        state, drug, doses, mask, context = _inputs()
        self.assertEqual(model(state, drug, doses, mask, context).shape, (4,))

    def test_two_training_losses_are_finite(self) -> None:
        model = ConditionalScalarEnergy(6, 5, 3, latent_dim=4, condition_dim=4, hidden_dim=8, dropout=0)
        target, drug, doses, mask, context = _inputs()
        flow = target + 0.2 * torch.randn_like(target)
        path = path_ranking_loss(model, target, flow, drug, doses, mask, context)
        condition = conditional_discrimination_loss(
            model,
            target,
            drug,
            doses,
            mask,
            torch.roll(drug, 1, 0),
            doses,
            mask,
            context,
            torch.roll(context, 1, 0),
        )
        self.assertTrue(torch.isfinite(path))
        self.assertTrue(torch.isfinite(condition))

    def test_relaxation_moves_downhill(self) -> None:
        torch.manual_seed(3)
        model = ConditionalScalarEnergy(6, 5, 3, latent_dim=4, condition_dim=4, hidden_dim=8, dropout=0)
        model.eval()
        state, drug, doses, mask, context = _inputs()
        before = model(state, drug, doses, mask, context).mean()
        refined = energy_relaxation_step(
            model,
            state,
            drug,
            doses,
            mask,
            context,
            strength=0.01,
        )
        after = model(refined, drug, doses, mask, context).mean()
        self.assertLessEqual(float(after.detach()), float(before.detach()) + 1e-6)


if __name__ == "__main__":
    unittest.main()
