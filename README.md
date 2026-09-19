# scEGFlow conditional energy module

This anonymous repository contains the conditional scalar energy component used
in the accompanying manuscript. It is intentionally limited to the method
described in the paper:

- a scalar compatibility model conditioned on cellular state, molecular
  features, dose, and optional biological context;
- path-wise energy ranking from a frozen flow prediction toward the measured
  perturbed state;
- conditional discrimination using corrupted perturbation conditions;
- checkpoint utilities and inference-time energy relaxation.

The repository does not include the flow-matching backbone, datasets, trained
weights, evaluation results, or experimental energy variants. In particular,
the training objective contains only the path-ranking and conditional
discrimination terms described in the manuscript.

## Method correspondence

For a cellular state `x` and complete condition `xi`, the model returns a scalar
energy `E(x, xi)`, where lower values indicate stronger compatibility. Training
uses

```text
L_energy = path_weight * L_path + condition_weight * L_cond
```

`L_path` orders states along the path from a frozen flow prediction to the
measured endpoint. `L_cond` assigns lower energy to the true condition than to a
corrupted condition. During inference, `energy_relaxation_step` applies the
paper's low-energy update `x <- x - strength * dE/dx`. This derivative is used
only to refine predictions at inference and is not an additional training loss.

## Environment / Installation

The package is tested with Python 3.9 or newer. It depends only on NumPy and
PyTorch at runtime. The test suite is run with pytest.

Create an isolated environment before installing the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

If PyTorch needs a platform-specific wheel, install the appropriate `torch`
build first, then run `python -m pip install -e .`.

## Data Preparation

No paper datasets, trained weights, or generated result files are included in
this anonymous release. The code expects precomputed tensors from an upstream
single-cell perturbation pipeline:

- `target_state`: measured perturbed cellular states with shape
  `(batch_size, state_dim)`;
- `flow_state`: frozen flow-model predictions with the same shape as
  `target_state`;
- `true_drug_features`: molecular features with shape
  `(batch_size, max_terms, drug_feature_dim)`;
- `true_doses`: dose values with shape `(batch_size, max_terms)`;
- `true_term_mask`: Boolean mask with shape `(batch_size, max_terms)`;
- optional context features with shape `(batch_size, context_feature_dim)`.

For combination perturbations encoded as strings such as
`drugA_1.0+drugB_0.1`, use `parse_condition` and `build_condition_tensors` to
construct padded drug, dose, and mask tensors. Corrupted conditions for the
conditional discrimination loss can be created by shuffling drug features,
doses, masks, or context features across the batch while keeping tensor shapes
unchanged.

Keep raw datasets, processed datasets, generated outputs, logs, wandb runs,
checkpoints, and model weights outside the Git repository. The `.gitignore`
file excludes those paths and file types by default.

## Training

The released training objective contains only the two terms described in the
manuscript:

```text
L_energy = path_weight * L_path + condition_weight * L_cond
```

Use `EnergyBatch` and `train_step` for a minimal training step:

```bash
python examples/synthetic_demo.py
```

For integration into a larger pipeline, instantiate `ConditionalScalarEnergy`,
prepare an optimizer, build `EnergyBatch` objects from precomputed tensors, and
call `train_step` or `fit_energy_model`. The upstream flow model should remain
frozen when producing `flow_state`; this release does not train or distribute
the flow-matching backbone.

## Evaluation

This repository provides the scalar energy module and inference-time relaxation
utility. It does not include benchmark data or paper result tables. To evaluate
on private or external perturbation data:

1. Generate frozen flow predictions for the evaluation conditions.
2. Compute `E(x, xi)` with `ConditionalScalarEnergy`; lower energy means higher
   compatibility between state `x` and condition `xi`.
3. Optionally apply `energy_relaxation_step` to refine flow predictions.
4. Compare the original and refined predictions against measured perturbed
   states with the metrics used by the surrounding benchmark pipeline.

## Reproduction Instructions

Run the synthetic release check from a fresh environment:

```bash
python -m pip install -e .
python examples/synthetic_demo.py
pytest
```

The synthetic demo seeds PyTorch, performs one energy-training step on generated
tensors, and prints the total loss, path loss, condition loss, and mean update
norm from inference-time energy relaxation. The tests cover condition parsing,
condition tensor padding, finite training losses, and downhill relaxation on a
small model.

Because this anonymous artifact intentionally omits data, trained weights, and
the upstream flow backbone, exact paper-level reproduction requires the
manuscript data-processing pipeline and benchmark data described outside this
repository. This repository is sufficient to reproduce and inspect the released
energy objective, checkpoint utilities, condition parsing, and synthetic demo.

## API Example

```python
import torch

from scegflow_energy import ConditionalScalarEnergy, energy_relaxation_step

model = ConditionalScalarEnergy(
    state_dim=2000,
    drug_feature_dim=1024,
    context_feature_dim=3,
)

batch_size = 8
state = torch.randn(batch_size, 2000)
drug_features = torch.randn(batch_size, 1, 1024)
doses = torch.ones(batch_size, 1)
term_mask = torch.ones(batch_size, 1, dtype=torch.bool)
context = torch.nn.functional.one_hot(
    torch.arange(batch_size) % 3,
    num_classes=3,
).float()

energy = model(state, drug_features, doses, term_mask, context)
refined = energy_relaxation_step(
    model,
    state,
    drug_features,
    doses,
    term_mask,
    context,
    strength=0.05,
)
```

See [`examples/synthetic_demo.py`](examples/synthetic_demo.py) for a complete
synthetic training and refinement example. No paper data are distributed in
this repository.

## Tests

```bash
python examples/synthetic_demo.py
pytest
```

## Repository scope

This is an anonymous review artifact. Please use the issue tracker for questions
that do not require disclosure of author identity.
