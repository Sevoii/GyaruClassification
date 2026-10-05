"""Fine-tune MobileNetV4 to classify gyaru (1) versus not gyaru (0)."""

from __future__ import annotations

import argparse
import random
import json
import numpy as np
from pathlib import Path

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from gyaru_dataset import GyaruDataset, build_transform, create_balanced_split, load_split, prepare_split
from metrics import binary_metrics, format_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(model, loader, device):
    model.eval()
    targets, probabilities = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1)
            probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
            targets.extend(labels.tolist())
    return targets, probabilities


def evaluate(model, loader, device, threshold: float = 0.5):
    targets, probabilities = predict(model, loader, device)
    return binary_metrics(targets, [int(p >= threshold) for p in probabilities])


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits_seed_42.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mobilenetv4_gyaru.pt"))
    parser.add_argument("--resume", type=Path, help="Continue fine-tuning from a saved checkpoint's model weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model", default="mobilenetv4_hybrid_large.e600_r384_in1k")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.image_size, args.patience) < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, image size, patience and learning rate must be positive; workers >= 0")
    if args.output.exists():
        parser.error("Output checkpoint already exists; choose a new --output to preserve the previous run")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Resume checkpoint does not exist: {args.resume}")

    set_seed(args.seed)
    if args.rebuild_split or not args.splits.exists():
        split = create_balanced_split(args.data_dir, args.splits, args.seed)
        print(f"Created {args.splits}: {len(split['train'])} train, {len(split['test'])} test")
    else:
        split = load_split(args.splits)
        if split["seed"] != args.seed:
            parser.error("Existing manifest seed differs from --seed; choose another --splits path")
        print(f"Using existing reproducible split: {args.splits}")

    root = args.data_dir.parent
    split = prepare_split(split, root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; train={len(split['train'])}, validation={len(split['validation'])}, test={len(split['test'])}")
    model = timm.create_model(args.model, pretrained=args.resume is None, num_classes=1).to(device)
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if resume_checkpoint.get("model_name") != args.model:
            parser.error("Resume checkpoint model does not match --model")
        if resume_checkpoint.get("image_size") != args.image_size:
            parser.error("Resume checkpoint image size does not match --image-size")
        if resume_checkpoint.get("seed") != split["seed"] or resume_checkpoint.get("split") != split:
            parser.error("Resume checkpoint split does not match the selected split")
        model.load_state_dict(resume_checkpoint["model_state"])
        print(f"Resuming model weights from {args.resume} (checkpoint epoch {resume_checkpoint.get('epoch', '?')})")
    config = timm.data.resolve_model_data_config(model)
    preprocessing = {"mean": list(config["mean"]), "std": list(config["std"])}
    train_ds = GyaruDataset(split["train"], root, build_transform(args.image_size, True, **preprocessing))
    val_ds = GyaruDataset(split["validation"], root, build_transform(args.image_size, False, **preprocessing))
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, shuffle=True, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(args.seed), **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)

    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.learning_rate},
                                  {"params": head, "lr": args.learning_rate * 10}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = int(resume_checkpoint.get("epoch", 0)) if resume_checkpoint else 0
    best_f1 = float(resume_checkpoint.get("validation_metrics", {}).get("f1", -1.0)) if resume_checkpoint else -1.0
    stale = 0
    history_path = args.resume.with_suffix(".history.json") if args.resume else None
    history = json.loads(history_path.read_text()) if history_path and history_path.is_file() else []
    if resume_checkpoint:
        # Seed the new output with the best known checkpoint so it remains usable
        # even if none of the continuation epochs improves validation F1.
        torch.save(resume_checkpoint, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images.to(device)).squeeze(1)
                loss = loss_fn(logits, labels.to(device, dtype=torch.float32))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; try --no-amp and a lower learning rate")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["f1"])
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_ds), "validation": metrics})
        args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch}/{start_epoch + args.epochs} loss={total_loss / len(train_ds):.4f}  {format_metrics(metrics)}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            stale = 0
            torch.save({"model_state": model.state_dict(), "model_name": args.model, "image_size": args.image_size,
                        "seed": split["seed"], "split_path": str(args.splits), "split": split,
                        "preprocessing": preprocessing, "threshold": 0.5, "epoch": epoch,
                        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        "validation_metrics": metrics}, args.output)
            print(f"Saved best checkpoint to {args.output}")
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping: validation F1 has not improved")
                break


if __name__ == "__main__":
    main()
