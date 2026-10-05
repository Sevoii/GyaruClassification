"""Explicitly convert a trusted legacy MobileNetV4 .pt checkpoint to safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mobilenetv4_checkpoint import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a trusted legacy MobileNetV4 checkpoint to safetensors.")
    parser.add_argument("source", type=Path, help="Trusted .pt checkpoint to convert")
    parser.add_argument("output", type=Path, help="Destination .safetensors weights path")
    args = parser.parse_args()
    if args.source.suffix != ".pt":
        parser.error("source must be a .pt checkpoint")
    if not args.source.is_file():
        parser.error(f"checkpoint not found: {args.source}")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("output weights or metadata already exists; choose a new destination")

    # This deliberately permits pickle because conversion is an explicit action
    # on a checkpoint the operator has designated as trusted.
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    save_checkpoint(checkpoint, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
