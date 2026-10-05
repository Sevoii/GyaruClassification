"""Reproducible dataset creation and image transforms for binary gyaru classification."""

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _valid_images(directory: Path) -> list[Path]:
    """Return decodable image files, excluding corrupt files before sampling."""
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.load()
            files.append(path)
        except (OSError, ValueError):
            print(f"Skipping unreadable image: {path}")
    return files


def create_balanced_split(data_dir: Path, split_path: Path, seed: int, samples_per_class: int = 1000,
                          test_fraction: float = 0.20) -> dict:
    """Sample gyaru vs every non-gyaru folder and save a deterministic split manifest."""
    positive_dir = data_dir / "gyaru"
    if not positive_dir.is_dir():
        raise FileNotFoundError(f"Expected positive class directory: {positive_dir}")
    positives = _valid_images(positive_dir)
    negatives: list[Path] = []
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and child.name != "gyaru":
            negatives.extend(_valid_images(child))
    if len(positives) < samples_per_class or len(negatives) < samples_per_class:
        raise ValueError(f"Need {samples_per_class} valid images per class; found {len(positives)} gyaru and {len(negatives)} non-gyaru.")

    rng = random.Random(seed)
    selected = [(path, 1) for path in rng.sample(positives, samples_per_class)]
    selected += [(path, 0) for path in rng.sample(negatives, samples_per_class)]
    train, test = [], []
    test_count = round(samples_per_class * test_fraction)
    for label in (0, 1):
        class_samples = [sample for sample in selected if sample[1] == label]
        rng.shuffle(class_samples)
        test.extend(class_samples[:test_count])
        train.extend(class_samples[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)

    root = data_dir.parent
    payload = {"seed": seed, "data_dir": str(data_dir), "samples_per_class": samples_per_class,
               "test_fraction": test_fraction, "labels": {"0": "not_gyaru", "1": "gyaru"},
               "train": [{"path": str(path.relative_to(root)), "label": label} for path, label in train],
               "test": [{"path": str(path.relative_to(root)), "label": label} for path, label in test]}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_split(split_path: Path) -> dict:
    return json.loads(split_path.read_text(encoding="utf-8"))


def prepare_split(split: dict, root: Path) -> dict:
    """Preserve the original holdout; reserve 20% of training for validation.

    Check exact file duplicates and conflicting labels, including old manifests.
    Near duplicates and related subjects require a separate manual/group audit.
    """
    split = json.loads(json.dumps(split))
    if "validation" not in split:
        rng = random.Random(split["seed"])
        train, validation = [], []
        for label in (0, 1):
            rows = [r for r in split["train"] if r["label"] == label]
            rng.shuffle(rows)
            count = round(len(rows) * 0.2)
            validation.extend(rows[:count])
            train.extend(rows[count:])
        rng.shuffle(train)
        split.update(train=train, validation=validation)
    seen = {}
    for partition in ("train", "validation", "test"):
        if not split[partition]:
            raise ValueError(f"Empty {partition} partition")
        for row in split[partition]:
            if row["label"] not in (0, 1):
                raise ValueError(f"Invalid label: {row}")
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f"Path escapes data root: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate image: {row['path']} and {seen[digest]}; resolve duplicates before training")
            if "sha256" in row and row["sha256"] != digest:
                raise ValueError(f"Image changed since training: {path}")
            seen[digest] = row["path"]
            row["sha256"] = digest
    return split


class ResizePad:
    """Aspect-ratio-preserving resize with symmetric padding; never crops an image."""
    def __init__(self, size: int, fill: tuple[int, int, int] = (124, 116, 104)):
        self.size, self.fill = size, fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = self.size / max(width, height)
        resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
        horizontal, vertical = self.size - resized.width, self.size - resized.height
        return ImageOps.expand(resized, border=(horizontal // 2, vertical // 2, horizontal - horizontal // 2,
                               vertical - vertical // 2), fill=self.fill)


def build_transform(image_size: int, training: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Callable:
    fill = tuple(round(v * 255) for v in mean)
    steps: list[Callable] = []
    if training:
        steps.extend([transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(8, expand=True, interpolation=transforms.InterpolationMode.BILINEAR, fill=fill),
                      transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08)])
    steps.extend([ResizePad(image_size, fill), transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


class GyaruDataset(Dataset):
    def __init__(self, samples: Iterable[dict], root: Path, transform: Callable):
        self.samples, self.root, self.transform = list(samples), root, transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(self.root / sample["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), int(sample["label"])
