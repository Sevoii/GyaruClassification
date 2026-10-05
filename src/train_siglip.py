"""Train a binary gyaru head from a pretrained, frozen base SigLIP encoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.numpy import save_file
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, SiglipVisionModel

from gyaru_dataset import load_split, prepare_split
from metrics import binary_metrics, format_metrics


DEFAULT_ENCODER = "google/siglip-base-patch16-384"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def embed_rows(rows: list[dict], root: Path, processor, encoder, device: torch.device,
               batch_size: int, workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    class ImageRows(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            with Image.open(root / row["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            return image, int(row["label"])

    def collate(batch):
        images, labels = zip(*batch)
        flipped = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images]
        pixels = processor(images=list(images) + flipped, return_tensors="pt")["pixel_values"]
        return pixels, torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(ImageRows(), batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=collate)
    features, targets = [], []
    encoder.eval()
    with torch.inference_mode():
        for pixels, labels in loader:
            pixels = pixels.to(device=device, dtype=next(encoder.parameters()).dtype)
            output = encoder(pixel_values=pixels).pooler_output.float()
            output = torch.nn.functional.normalize(output, dim=-1)
            original, flipped = output.split(len(labels))
            feature = torch.nn.functional.normalize(original + flipped, dim=-1)
            features.append(feature.cpu())
            targets.append(labels)
    return torch.cat(features), torch.cat(targets)


def score_head(head, features: torch.Tensor, targets: torch.Tensor) -> tuple[dict, list[float]]:
    head.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(head(features), dim=1)[:, 1].tolist()
    labels = targets.tolist()
    return binary_metrics(labels, [int(p >= 0.5) for p in probabilities]), probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary head from a pretrained, frozen base SigLIP encoder.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits_seed_42.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gyaru_siglip_heads.safetensors"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive; workers >= 0")
    if args.output.exists():
        parser.error("Output already exists; choose a new --output to preserve prior results")
    if not args.splits.is_file():
        parser.error(f"Split manifest not found: {args.splits}")

    set_seed(args.seed)
    model_name = DEFAULT_ENCODER
    split = load_split(args.splits)
    if split.get("seed") != args.seed:
        parser.error("Existing manifest seed differs from --seed; choose a matching seed or another --splits path")
    root = args.data_dir.parent
    split = prepare_split(split, root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting from pretrained base encoder {model_name} on {device}; training a new binary head")
    processor = AutoImageProcessor.from_pretrained(model_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    encoder = SiglipVisionModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    extracted = {}
    for partition in ("train", "validation"):
        print(f"Extracting {partition} embeddings ({len(split[partition])} images)")
        extracted[partition] = embed_rows(split[partition], root, processor, encoder, device,
                                           args.batch_size, args.workers)
    del encoder
    train_x, train_y = extracted["train"]
    val_x, val_y = extracted["validation"]
    head = nn.Linear(train_x.shape[1], 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(train_y))
        loss_sum = 0.0
        for indices in order.split(args.batch_size):
            logits = head(train_x[indices])
            loss = loss_fn(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(indices)
        metrics, _ = score_head(head, val_x, val_y)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_y), "validation": metrics})
        print(f"Epoch {epoch}/{args.epochs} loss={loss_sum / len(train_y):.4f}  {format_metrics(metrics)}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({
        "main_coef": head.weight.detach().numpy().astype(np.float32),
        "main_intercept": head.bias.detach().numpy().astype(np.float32),
    }, str(args.output), metadata={
        "main_classes": json.dumps(["not_gyaru", "gyaru"]),
        "embed_model": model_name,
    })
    args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved best validation-F1 head to {args.output}")
    print("The held-out test partition is embedded in the split manifest and was not used for training or selection.")
    print("Set GYARU_SIGLIP_HEADS to this file to use it in the app.")


if __name__ == "__main__":
    main()
