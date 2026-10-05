"""SigLIP SO400M vision embeddings with portable NumPy binary heads."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.numpy import load_file
from transformers import AutoImageProcessor, SiglipVisionModel

from models.interface import Classifier

MODEL_CLASS = "Siglip400GyaruClassifier"


class NpHead:
    def __init__(self, coef, intercept):
        self.coef, self.intercept = coef, intercept

    def predict_proba(self, embeddings):
        logits = embeddings @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)


class Siglip400GyaruClassifier(Classifier):
    """Binary gyaru classifier with the larger SigLIP SO400M image encoder."""

    model_id = "gyaru_siglip_400"
    display_name = "SigLIP SO400M Gyaru Classifier"
    heads_path: Path | None = None

    def load(self, device: str) -> None:
        default_bundle = Path(__file__).with_name("model_internals") / "heads.safetensors"
        bundle = self.heads_path or default_bundle
        if bundle.suffix != ".safetensors":
            raise ValueError(f"SigLIP SO400M heads must use .safetensors: {bundle}")
        if not bundle.is_file():
            raise FileNotFoundError(
                f"SigLIP SO400M heads not found: {bundle}. "
                "Run src/train_siglip_400.py and pass the result with --siglip-400-heads."
            )
        data = load_file(str(bundle))
        from safetensors import safe_open
        with safe_open(str(bundle), framework="np") as stream:
            details = stream.metadata() or {}
        self.classes = json.loads(details["main_classes"])
        self.name = details["embed_model"]
        if self.name != "google/siglip-so400m-patch14-384":
            raise ValueError(f"SO400M adapter requires its matching encoder, found: {self.name}")
        self.main = NpHead(data["main_coef"], data["main_intercept"])
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(self.name)
        self.encoder = SiglipVisionModel.from_pretrained(self.name, torch_dtype=self.dtype).to(device).eval()

    def predict(self, image: Image.Image) -> dict:
        image = ImageOps.exif_transpose(image).convert("RGB")
        saturation = float(np.asarray(image.resize((128, 128)).convert("HSV"))[..., 1].mean()) / 255
        with torch.inference_mode():
            views = (image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
            embeddings = []
            for view in views:
                pixels = self.processor(images=[view], return_tensors="pt")["pixel_values"]
                pixels = pixels.to(self.device, dtype=self.dtype)
                embedding = self.encoder(pixel_values=pixels).pooler_output.float()
                embeddings.append(torch.nn.functional.normalize(embedding, dim=-1))
            embedding = torch.nn.functional.normalize(embeddings[0] + embeddings[1], dim=-1).cpu().numpy()
        probabilities = self.main.predict_proba(embedding)[0]
        ranked = sorted(zip(self.classes, probabilities), key=lambda item: -item[1])
        gyaru_p = float(probabilities[self.classes.index("gyaru")])
        return {
            "top": ranked[0][0],
            "confidence": float(ranked[0][1]),
            "probs": [{"label": label, "p": float(probability)} for label, probability in ranked],
            "gyaru_p": gyaru_p,
            "bw": saturation < 0.06,
        }
