"""Fine-tuned MobileNetV4 binary classifier using the shared prediction contract."""

import torch
import timm
from PIL import Image, ImageOps

from models.interface import Classifier
from gyaru_dataset import build_transform
from mobilenetv4_checkpoint import load_checkpoint

MODEL_CLASS = "MobileNetV4GyaruClassifier"


class MobileNetV4GyaruClassifier(Classifier):
    model_id = "mobilenetv4_gyaru"
    display_name = "Fine-tuned MobileNetV4"

    def load(self, device):
        from os import environ
        from pathlib import Path

        bundled_checkpoint = Path(__file__).with_name("model_internals") / "checkpoint.safetensors"
        checkpoint_path = Path(environ.get("GYARU_MOBILENET_CHECKPOINT", bundled_checkpoint))
        checkpoint = load_checkpoint(checkpoint_path)
        self.model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=1)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model = self.model.to(device).eval()
        self.device = device
        self.threshold = float(checkpoint.get("threshold", 0.5))
        self.mean, self.std = checkpoint["preprocessing"]["mean"], checkpoint["preprocessing"]["std"]
        self.transform = build_transform(checkpoint["image_size"], training=False, mean=self.mean, std=self.std)

    def predict(self, image: Image.Image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            gyaru_p = float(torch.sigmoid(self.model(x).reshape(-1)[0]).item())
        probs = [{"label": "gyaru", "p": gyaru_p}, {"label": "not_gyaru", "p": 1.0 - gyaru_p}]
        probs.sort(key=lambda item: -item["p"])
        top, confidence = probs[0]["label"], probs[0]["p"]
        return {"top": top, "confidence": confidence, "probs": probs, "gyaru_p": gyaru_p,
                "bw": False, "substyle": {"show": False, "top": None, "confidence": None, "probs": []}}
