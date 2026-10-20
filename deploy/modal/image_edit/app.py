"""Scale-to-zero Modal wrapper for Qwen-Image-Edit reference renders.

The open-weights replacement for the hosted image APIs: no OpenAI or Gemini key, just the Modal
workspace the other model functions already run in. Weights are Apache-2.0 and ungated, so no
Hugging Face token is needed either. See README.md.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import modal

APP_NAME = "litereality-image-edit"  # included into `litereality` by deploy/modal/app.py
FUNCTION_NAME = "image_edit"
MODEL_VOLUME = "litereality-image-edit-models"
WEIGHTS_ROOT = Path("/models")

# One container until the Volume holds the weights: concurrent cold starts would each pull ~58 GB
# into the same cache. After the first call has populated it, redeploy with a higher value
# (LR_IMAGE_EDIT_MAX_CONTAINERS=4) so a room's references stop queueing behind one GPU.
MAX_CONTAINERS = int(os.environ.get("LR_IMAGE_EDIT_MAX_CONTAINERS", "1"))

# CPU-built from wheels; the default torch wheels bundle the CUDA runtime the H100 needs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.36.0",
        "transformers==4.57.5",
        "accelerate==1.10.1",
        "pillow>=10",
    )
    .env({"HF_HOME": str(WEIGHTS_ROOT / "huggingface"), "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("litereality_agent")
)
weights = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)
_pipelines: dict[str, object] = {}


def _load_pipeline(model_id: str | None):
    from litereality_agent.models.image_edit.inference import DEFAULT_MODEL, load_pipeline

    key = model_id or DEFAULT_MODEL
    if key not in _pipelines:
        _pipelines[key] = load_pipeline(key)
        weights.commit()
    return _pipelines[key]


@app.function(
    name=FUNCTION_NAME,
    image=image,
    gpu="H100",
    volumes={str(WEIGHTS_ROOT): weights},
    timeout=15 * 60,
    scaledown_window=60,
    max_containers=MAX_CONTAINERS,
)
def infer(request: dict) -> dict:
    """Edit the request's images per its prompt and return one PNG render."""
    from litereality_agent.models.image_edit.inference import DEFAULT_STEPS, edit

    png = edit(
        _load_pipeline(request.get("model_id")),
        [base64.b64decode(raw, validate=True) for raw in request["images_b64"]],
        request["prompt"],
        steps=int(request.get("steps") or DEFAULT_STEPS),
        seed=int(request.get("seed", 0)),
    )
    return {"png_b64": base64.b64encode(png).decode("ascii")}
