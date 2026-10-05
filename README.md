# Gyaru classifier

## Classifier web app

Install the web dependency in the project environment, then start Flask from the
repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\app.py
```

Open `http://127.0.0.1:8800`. The model picker reads adapters discovered under
`src/models/`. Each model is a package with a `model.py` exporting a
`MODEL_CLASS`; that class implements `src/models/interface.py` (`load(device)`,
`predict(image)`, and metadata). The shared prediction response has `top`,
`confidence`, and a ranked `probs` list. Model-specific optional fields can be
included when supported. The Flask endpoints are `GET /models` and
`POST /predict?model=<model_id>` with raw image bytes in the request body.

The SigLIP adapter and its portable NumPy heads are in
`src/models/gyaru_siglip/`; downloaded SigLIP encoder weights use the Hugging
Face cache. The fine-tuned MobileNetV4 adapter is in
`src/models/mobilenetv4_gyaru/`, with its current fine-tuned checkpoint in
`model_internals/checkpoint.pt`. Set `GYARU_MOBILENET_CHECKPOINT` to use another
compatible training checkpoint.
No joblib files are needed. Flask's built-in server is suitable for this local
testing UI; use a production WSGI server if exposing it beyond the local machine.

Run commands from the project root using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe src\finetune_mobilenetv4.py --output artifacts\gyaru_v2.pt
.\.venv\Scripts\python.exe src\test_mobilenetv4.py --checkpoint artifacts\gyaru_v2.pt
```

Training always loads pretrained ImageNet weights. All layers are fine-tuned;
the new classifier learns at 10 times the backbone learning rate. CUDA is used
when available, with mixed precision by default (`--no-amp` disables it).
Reduce `--batch-size` if GPU memory runs out.

The original pooled random sample is preserved: 1,000 gyaru and 1,000 negatives
from all other folders. The original 400-image test holdout stays unchanged.
A seeded subdivision of the original training pool produces 1,280 training and
320 validation images. Checkpoint selection, learning-rate reduction, and early
stopping use validation F1 at threshold 0.5. Test images are never evaluated by
the training loop. The checkpoint embeds all three partitions and file hashes.
Keep the JSON manifest to reuse the original sample. A different seed requires
a different `--splits` path (or deliberate `--rebuild-split`). Existing checkpoint
outputs are protected from overwriting.

To continue from a saved checkpoint, pass it with `--resume` and choose a new
output path. The script restores the model weights and prior best checkpoint,
then runs the requested number of additional epochs with a fresh optimizer:

```powershell
.\.venv\Scripts\python.exe src\finetune_mobilenetv4.py --splits artifacts\splits_seed_42.json --resume artifacts\gyaru_v2.pt --epochs 20 --patience 6 --output artifacts\gyaru_v2_continued.pt
```

Train a new binary gyaru head from the pretrained base SigLIP encoder with the
same train and validation partitions. The base encoder is frozen; only the new
head is trained, and the best head is selected by validation F1. The test
partition remains untouched. The output bundle contains the binary head; it
does not depend on or overwrite any existing head bundle. Substyle output is
unavailable for this newly trained binary-only bundle.

```powershell
.\.venv\Scripts\python.exe src\train_siglip.py --output artifacts\gyaru_siglip_heads.safetensors
$env:GYARU_SIGLIP_HEADS = "artifacts\gyaru_siglip_heads.safetensors"
.\.venv\Scripts\python.exe src\app.py
```

The script starts from `google/siglip-base-patch16-384` and uses
`artifacts\splits_seed_42.json` by default. Choose a new output filename for
each training run. It writes a sibling `.history.json` with per-epoch validation
metrics.

Images are EXIF-oriented, converted to RGB, mildly rotated with an expanded
canvas, and resized with aspect ratio preserved and padding. Training also uses
horizontal flips and mild color changes. Evaluation is deterministic. Rotation
precedes resizing to avoid clipped corners. Normalization comes from the model's
pretrained configuration and is saved in the checkpoint.

Evaluation accepts checkpoint paths (positional paths or repeatable
`--checkpoint` options) and packaged model IDs (repeatable `--model`). With no
selection, it evaluates every `.pt` file in `artifacts/` and discovers every
classifier package under `src/models/`, including SigLIP. Packaged models use
the split in `artifacts/splits_seed_42.json` by default; pass `--splits` to use
another manifest. Each model gets its own `metrics.json` and `predictions.csv`,
and `summary.json` collects all results under `artifacts/evaluation` (change
`--report-dir` for separate runs). Reports include threshold-based confusion
metrics plus specificity, balanced accuracy, Matthews correlation, ROC-AUC,
average precision, Brier score, and log loss. Gyaru is positive: FP means a negative
image predicted as gyaru, FN means a gyaru image missed. JSON includes error rates
as well as counts; F1/precision/recall describe the positive class. Undefined
precision/recall/F1 use zero; undefined error rates use null. Do not tune the
threshold using test results. Legacy checkpoints selected on the test set must
be retrained. Training history is saved beside the checkpoint.

Limitations and further experiments:

- Exact file duplicates and data changes are rejected. Re-encoded images, alternate
  crops, repeated characters, and shared artists/sources need a manual or perceptual
  audit and potentially group-based splitting.
- A balanced benchmark does not establish precision at real-world class prevalence.
  Test on independently collected data representative of the intended use.
- Equal negative-folder sampling is not universally better; it changes the target
  distribution. There are only 172 delinquent images, so 250 unique samples from
  each of four folders is impossible. Compare sampling strategies on validation.
- Head-only warmup, threshold tuning, and different augmentation strengths are
  experiments, not guaranteed improvements. The current implementation uses
  different head/backbone learning rates and a fixed 0.5 threshold.
- Seeds stabilize sampling and initialization; bitwise training reproducibility
  across hardware, library versions, and CUDA kernels is not guaranteed.
- If the original test set was already used to select a model or tune settings,
  preserving it does not erase that prior exposure. Collect a fresh final holdout
  for an unbiased final estimate.
