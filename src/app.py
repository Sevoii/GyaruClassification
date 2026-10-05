"""Flask UI/API for interchangeable classifiers under src/models/."""
from __future__ import annotations

import argparse
import importlib
import io
import os
import pkgutil
from threading import Lock

import torch
from flask import Flask, abort, jsonify, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from models.interface import Classifier

from pathlib import Path

HERE = Path(__file__).resolve().parent
UI = HERE / "ui.html"
PORT = int(os.environ.get("PORT", "8080"))
MAX_BYTES = 20 * 1024 * 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def discover_models():
    """Load adapters from each package in src/models that declares MODEL_CLASS."""
    discovered = {}
    for item in pkgutil.iter_modules([str(HERE / "models")]):
        if not item.ispkg or item.name.startswith("_"):
            continue
        module = importlib.import_module(f"models.{item.name}.model")
        adapter_type = getattr(module, "MODEL_CLASS", None)
        if adapter_type:
            adapter_type = getattr(module, adapter_type)
            if not issubclass(adapter_type, Classifier):
                raise TypeError(f"{item.name}.model.MODEL_CLASS must implement models.interface.Classifier")
            discovered[adapter_type.model_id] = adapter_type
    return discovered


MODEL_TYPES = discover_models()
_loaded_models = {}
_model_lock = Lock()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


def get_model(model_id):
    if model_id not in MODEL_TYPES:
        abort(404, description=f"Unknown model: {model_id}")
    if model_id not in _loaded_models:
        with _model_lock:
            if model_id not in _loaded_models:
                model = MODEL_TYPES[model_id]()
                model.load(DEVICE)
                _loaded_models[model_id] = model
    return _loaded_models[model_id]


@app.get("/")
@app.get("/index.html")
def index():
    return send_file(UI)


@app.get("/models")
def list_models():
    return jsonify([model_type().metadata() for model_type in MODEL_TYPES.values()])


@app.post("/predict")
def predict():
    model_id = request.args.get("model", next(iter(MODEL_TYPES), ""))
    payload = request.get_data(cache=False)
    if not payload:
        abort(400, description="Image is missing")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            image = ImageOps.exif_transpose(source)
            if image.mode in ("P", "LA", "RGBA"):
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                image = background
            image = image.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        abort(400, description="Request body is not a readable image")
    try:
        result = get_model(model_id).predict(image)
    except FileNotFoundError as exc:
        abort(503, description=str(exc))
    result["model"] = model_id
    return jsonify(result)


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(413)
@app.errorhandler(503)
def api_error(error):
    return jsonify(error=str(getattr(error, "description", error))), error.code


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local gyaru classifier web app.")
    parser.add_argument("--siglip-400-heads", type=Path, default=None,
                        help="Trained .safetensors head bundle for gyaru_siglip_400")
    args = parser.parse_args()
    if args.siglip_400_heads is not None:
        MODEL_TYPES["gyaru_siglip_400"].heads_path = args.siglip_400_heads
    print(f"Available models: {', '.join(MODEL_TYPES)}; device={DEVICE}")
    print(f"Open http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)


if __name__ == "__main__":
    main()
