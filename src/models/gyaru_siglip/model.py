"""SigLIP vision embeddings with portable NumPy logistic-regression heads."""

import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, SiglipVisionModel

from models.interface import Classifier

MODEL_CLASS = "SiglipGyaruClassifier"


class NpHead:
    def __init__(self, coef, intercept):
        self.coef, self.intercept = coef, intercept

    def predict_proba(self, x):
        z = x @ self.coef.T + self.intercept
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class SiglipGyaruClassifier(Classifier):
    model_id = "gyaru_siglip"
    display_name = "SigLIP Gyaru Classifier"

    def load(self, device):
        bundle = Path(__file__).with_name("model_internals") / "heads.npz"
        data = np.load(bundle, allow_pickle=False)
        self.classes = [str(c) for c in data["main_classes"]]
        self.main = NpHead(data["main_coef"], data["main_intercept"])
        self.sub = NpHead(data["sub_coef"], data["sub_intercept"])
        self.subclasses = [str(c) for c in data["sub_classes"]]
        self.name = str(data["embed_model"])
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(self.name)
        self.encoder = SiglipVisionModel.from_pretrained(self.name, torch_dtype=self.dtype).to(device).eval()

    def predict(self, image: Image.Image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        saturation = float(np.asarray(image.resize((128, 128)).convert("HSV"))[..., 1].mean()) / 255
        with torch.inference_mode():
            embeddings = []
            for view in (image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
                pixels = self.processor(images=[view], return_tensors="pt")["pixel_values"].to(self.device, dtype=self.dtype)
                embeddings.append(torch.nn.functional.normalize(self.encoder(pixel_values=pixels).pooler_output.float(), dim=-1))
            embedding = torch.nn.functional.normalize(embeddings[0] + embeddings[1], dim=-1).cpu().numpy()
        probabilities = self.main.predict_proba(embedding)[0]
        ranked = sorted(zip(self.classes, probabilities), key=lambda item: -item[1])
        gyaru_p = float(probabilities[self.classes.index("gyaru")])
        sub_probs = self.sub.predict_proba(embedding)[0]
        subranked = sorted(zip(self.subclasses, sub_probs), key=lambda item: -item[1])
        top, confidence = ranked[0]
        return {
            "top": top, "confidence": float(confidence),
            "probs": [{"label": label, "p": float(p)} for label, p in ranked],
            "gyaru_p": gyaru_p, "bw": saturation < 0.06,
            "substyle": {"show": bool(saturation >= 0.06 and (top == "gyaru" or gyaru_p >= 0.5)),
                         "top": subranked[0][0], "confidence": float(subranked[0][1]),
                         "probs": [{"label": label, "p": float(p)} for label, p in subranked]},
        }
