"""Evaluate one or more saved gyaru classifiers on their held-out splits."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pkgutil
from pathlib import Path
import sys

import timm
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

from finetune_mobilenetv4 import predict
from gyaru_dataset import GyaruDataset, build_transform, load_split, prepare_split, sample_path
from metrics import binary_metrics, format_metrics

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def probability_metrics(targets: list[int], probabilities: list[float]) -> dict[str, float | None]:
    """Compute threshold-independent ranking and calibration statistics."""
    positives = sum(targets)
    negatives = len(targets) - positives
    brier = sum((p - y) ** 2 for y, p in zip(targets, probabilities)) / len(targets)
    eps = 1e-15
    log_loss = -sum(y * math.log(max(eps, min(1 - eps, p))) +
                    (1 - y) * math.log(max(eps, min(1 - eps, 1 - p)))
                    for y, p in zip(targets, probabilities)) / len(targets)
    roc_auc = None
    if positives and negatives:
        ordered = sorted(zip(probabilities, targets))
        rank_sum = 0.0
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                j += 1
            average_rank = ((i + 1) + j) / 2
            rank_sum += average_rank * sum(y for _, y in ordered[i:j])
            i = j
        roc_auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    # Average precision is the area under the stepwise precision-recall curve.
    average_precision = None
    if positives:
        ordered_desc = sorted(zip(probabilities, targets), reverse=True)
        found = 0
        precision_sum = 0.0
        for rank, (_, label) in enumerate(ordered_desc, start=1):
            if label:
                found += 1
                precision_sum += found / rank
        average_precision = precision_sum / positives
    return {"roc_auc": roc_auc, "average_precision": average_precision,
            "brier_score": brier, "log_loss": log_loss}


def evaluate_checkpoint(path: Path, args, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "split" not in checkpoint:
        raise ValueError("checkpoint has no embedded split; retrain with the current training script")
    split = checkpoint["split"]
    if args.splits and load_split(args.splits)["test"] != [
        {k: v for k, v in row.items() if k != "sha256"} for row in split["test"]
    ]:
        raise ValueError("provided split differs from checkpoint holdout")
    split = prepare_split(split, args.data_dir.parent)
    dataset = GyaruDataset(split["test"], args.data_dir.parent,
                           build_transform(checkpoint["image_size"], False, **checkpoint["preprocessing"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=torch.cuda.is_available())
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=1).to(device)
    model.load_state_dict(checkpoint["model_state"])
    targets, probabilities = predict(model, loader, device)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)
    predictions = [int(p >= threshold) for p in probabilities]
    metrics = binary_metrics(targets, predictions)
    tn, fp = metrics["true_negative"], metrics["false_positive"]
    tp, fn = metrics["true_positive"], metrics["false_negative"]
    specificity = tn / (tn + fp) if tn + fp else None
    metrics.update({"specificity": specificity,
                    "balanced_accuracy": ((metrics["recall"] + specificity) / 2) if specificity is not None else None,
                    "matthews_correlation": ((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
                    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) else None})
    metrics.update(probability_metrics(targets, probabilities))

    report_dir = args.report_dir / path.stem
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint": str(path), "model_name": checkpoint["model_name"], "images": len(dataset),
              "threshold": threshold, **metrics}
    (report_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    with (report_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "probability_gyaru", "prediction", "error"])
        for row, target, probability, prediction in zip(dataset.samples, targets, probabilities, predictions):
            error = "FP" if prediction == 1 and target == 0 else "FN" if prediction == 0 and target == 1 else ""
            writer.writerow([row["path"], target, probability, prediction, error])
    return result


def discover_model_adapters(models_dir: Path) -> dict[str, type]:
    """Discover runnable classifiers using the same package contract as app.py."""
    discovered = {}
    for item in pkgutil.iter_modules([str(models_dir)]):
        if not item.ispkg or item.name.startswith("_"):
            continue
        module = importlib.import_module(f"models.{item.name}.model")
        class_name = getattr(module, "MODEL_CLASS", None)
        if class_name:
            adapter = getattr(module, class_name)
            discovered[adapter.model_id] = adapter
    return discovered


def evaluate_adapter(model_id: str, adapter_type: type, args, device: torch.device, split: dict) -> dict:
    from metrics import binary_metrics

    split = prepare_split(split, args.data_dir.parent)
    adapter = adapter_type()
    adapter.load(str(device))
    threshold = args.threshold if args.threshold is not None else float(getattr(adapter, "threshold", 0.5))
    targets, probabilities, predictions = [], [], []
    rows = split["test"]
    for row in rows:
        with Image.open(sample_path(args.data_dir.parent, row["path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            result = adapter.predict(image)
        target = int(row["label"])
        probability = float(result["gyaru_p"])
        targets.append(target)
        probabilities.append(probability)
        predictions.append(int(probability >= threshold))
    metrics = binary_metrics(targets, predictions)
    tn, fp = metrics["true_negative"], metrics["false_positive"]
    tp, fn = metrics["true_positive"], metrics["false_negative"]
    specificity = tn / (tn + fp) if tn + fp else None
    metrics.update({"specificity": specificity,
                    "balanced_accuracy": ((metrics["recall"] + specificity) / 2) if specificity is not None else None,
                    "matthews_correlation": ((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tn + fp) * (tp + fn) * (tn + fn)))
                    if (tp + fp) * (tn + fp) * (tp + fn) * (tn + fn) else None})
    metrics.update(probability_metrics(targets, probabilities))
    report_dir = args.report_dir / model_id
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {"model_id": model_id, "model_name": adapter_type.display_name, "images": len(rows),
              "threshold": threshold, **metrics}
    (report_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    with (report_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "probability_gyaru", "prediction", "error"])
        for row, target, probability, prediction in zip(rows, targets, probabilities, predictions):
            error = "FP" if prediction and not target else "FN" if not prediction and target else ""
            writer.writerow([row["path"], target, probability, prediction, error])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved classifier checkpoints on their held-out test sets.")
    parser.add_argument("checkpoints", nargs="*", type=Path,
                        help="Checkpoint paths; defaults to every .pt file in artifacts/")
    parser.add_argument("--checkpoint", dest="checkpoints_option", action="append", type=Path,
                        help="Select a checkpoint (repeatable; alternative to positional paths)")
    parser.add_argument("--splits", type=Path, default=None, help="Optionally verify each checkpoint's embedded split")
    parser.add_argument("--model", dest="models_option", action="append",
                        help="Select a model package from src/models by id (repeatable)")
    parser.add_argument("--models-dir", type=Path, default=HERE / "models",
                        help="Directory containing classifier packages")
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Prediction cutoff (default: each checkpoint's saved threshold, or 0.5)")
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Threshold must be in [0, 1], batch size positive, workers nonnegative")
    selected = args.checkpoints_option or args.checkpoints
    has_selection = bool(selected or args.models_option)
    checkpoints = selected if selected else ([] if has_selection else sorted(Path("artifacts").glob("*.pt")))
    checkpoints = list(dict.fromkeys(checkpoints))
    adapters = discover_model_adapters(args.models_dir)
    selected_models = args.models_option if args.models_option else ([] if has_selection else sorted(adapters))
    unknown = [model for model in selected_models if model not in adapters]
    if unknown:
        parser.error("Unknown model id(s): " + ", ".join(unknown) + "; available: " + ", ".join(sorted(adapters)))
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        parser.error("Checkpoint not found: " + ", ".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; evaluating {len(checkpoints)} checkpoint(s) and {len(selected_models)} packaged model(s)")
    results = []
    for path in checkpoints:
        try:
            result = evaluate_checkpoint(path, args, device)
        except (KeyError, ValueError, RuntimeError) as exc:
            parser.error(f"Could not evaluate {path}: {exc}")
        results.append(result)
        print(f"\n{path} ({result['images']} images, threshold={result['threshold']:.2f})")
        print(format_metrics(result))
        print("specificity={specificity:.4f}  balanced_accuracy={balanced_accuracy:.4f}  "
              "MCC={matthews_correlation:.4f}  ROC-AUC={roc_auc:.4f}  AP={average_precision:.4f}  "
              "Brier={brier_score:.4f}  log_loss={log_loss:.4f}".format(**{
                  k: (v if v is not None else float("nan")) for k, v in result.items()}))
    if selected_models:
        split_path = args.splits or Path("artifacts/splits_seed_42.json")
        if not split_path.is_file():
            parser.error(f"A split manifest is needed to evaluate packaged models: {split_path}")
        split = load_split(split_path)
        for model_id in selected_models:
            try:
                result = evaluate_adapter(model_id, adapters[model_id], args, device, split)
            except (KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
                parser.error(f"Could not evaluate model {model_id}: {exc}")
            results.append(result)
            print(f"\n{model_id} ({result['images']} images, threshold={result['threshold']:.2f})")
            print(format_metrics(result))
            print("specificity={specificity:.4f}  balanced_accuracy={balanced_accuracy:.4f}  "
                  "MCC={matthews_correlation:.4f}  ROC-AUC={roc_auc:.4f}  AP={average_precision:.4f}  "
                  "Brier={brier_score:.4f}  log_loss={log_loss:.4f}".format(**{
                      k: (v if v is not None else float("nan")) for k, v in result.items()}))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nReports saved under {args.report_dir}")


if __name__ == "__main__":
    main()
