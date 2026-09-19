"""Portable checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .model import ConditionalScalarEnergy


def save_checkpoint(
    path: str | Path,
    model: ConditionalScalarEnergy,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata or {})}, target)


def load_checkpoint(
    path: str | Path,
    model: ConditionalScalarEnergy,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    model.load_state_dict(payload["state_dict"])
    return dict(payload.get("metadata", {}))

