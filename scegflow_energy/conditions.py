"""Condition parsing and tensor construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ConditionParts:
    """Drug names and doses parsed from one condition label."""

    value: str
    drugs: tuple[str, ...]
    doses: tuple[float, ...]


def _split_top_level_plus(value: str) -> list[str]:
    tokens: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        elif character == "+" and depth == 0:
            token = value[start:index].strip()
            if token:
                tokens.append(token)
            start = index + 1
    final_token = value[start:].strip()
    if final_token:
        tokens.append(final_token)
    return tokens


def _split_drug_dose(token: str) -> tuple[str, float | None]:
    token = token.strip()
    if "_" not in token:
        return token, None
    drug, possible_dose = token.rsplit("_", 1)
    try:
        return drug, float(possible_dose)
    except ValueError:
        return token, None


def parse_condition(value: object, default_dose: float = 1.0) -> ConditionParts:
    """Parse labels such as ``DrugA_1.0+DrugB_0.1`` without splitting ``(+)``."""

    condition = str(value)
    tokens = _split_top_level_plus(condition)
    parsed = [_split_drug_dose(token) for token in tokens]
    drugs = tuple(drug for drug, _ in parsed)
    doses = tuple(float(default_dose if dose is None else dose) for _, dose in parsed)
    return ConditionParts(condition, drugs, doses)


def build_condition_tensors(
    conditions: Sequence[ConditionParts | str],
    drug_features: Mapping[str, np.ndarray | torch.Tensor],
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert variable-length drug combinations into padded model tensors."""

    if not drug_features:
        raise ValueError("drug_features must not be empty")
    parsed = [item if isinstance(item, ConditionParts) else parse_condition(item) for item in conditions]
    feature_dim = int(next(iter(drug_features.values())).shape[0])
    max_terms = max(1, max((len(item.drugs) for item in parsed), default=0))
    target_device = torch.device(device or "cpu")
    features = torch.zeros(len(parsed), max_terms, feature_dim, device=target_device)
    doses = torch.ones(len(parsed), max_terms, device=target_device)
    mask = torch.zeros(len(parsed), max_terms, dtype=torch.bool, device=target_device)

    for row, item in enumerate(parsed):
        for column, (drug, dose) in enumerate(zip(item.drugs, item.doses)):
            if drug not in drug_features:
                raise KeyError(f"Missing molecular features for drug '{drug}'")
            features[row, column] = torch.as_tensor(
                drug_features[drug], dtype=torch.float32, device=target_device
            )
            doses[row, column] = float(dose)
            mask[row, column] = True
    return features, doses, mask

