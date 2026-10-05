"""Safe serialization for the fine-tuned MobileNetV4 classifier."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def metadata_path(checkpoint_path: Path) -> Path:
    """Return the JSON sidecar path for a MobileNetV4 weights file."""
    return checkpoint_path.with_suffix(".json")


def _require_safetensors(checkpoint_path: Path) -> None:
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError(f"MobileNetV4 checkpoint must use .safetensors: {checkpoint_path}")


def save_checkpoint(checkpoint: dict, checkpoint_path: Path) -> None:
    """Write model tensors and JSON metadata without using pickle."""
    _require_safetensors(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_state = checkpoint["model_state"]
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in model_state.items()}
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state"}
    save_file(tensors, str(checkpoint_path))
    metadata_path(checkpoint_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load a tensor-only checkpoint and its JSON metadata sidecar."""
    _require_safetensors(checkpoint_path)
    sidecar = metadata_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fine-tuned MobileNetV4 weights missing: {checkpoint_path}")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Fine-tuned MobileNetV4 metadata missing: {sidecar}")
    checkpoint = json.loads(sidecar.read_text(encoding="utf-8"))
    checkpoint["model_state"] = load_file(str(checkpoint_path), device="cpu")
    return checkpoint
