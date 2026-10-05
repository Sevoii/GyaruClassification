"""Dependency-free binary classification metrics."""

from __future__ import annotations


def binary_metrics(targets: list[int], predictions: list[int]) -> dict[str, float | int]:
    if not targets or len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have equal, nonzero length")
    if any(v not in (0, 1) for v in targets + predictions):
        raise ValueError("Expected binary labels")
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    total = len(targets)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / total if total else 0.0, "precision": precision, "recall": recall, "f1": f1,
            "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "false_negative_rate": fn / (fn + tp) if fn + tp else None}


def format_metrics(metrics: dict[str, float | int]) -> str:
    return (f"accuracy={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}\n"
            f"TP={metrics['true_positive']}  TN={metrics['true_negative']}  FP={metrics['false_positive']}  FN={metrics['false_negative']}")
