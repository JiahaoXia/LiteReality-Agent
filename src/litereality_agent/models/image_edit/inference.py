"""Open-weights instruction image editing (Qwen-Image-Edit) for object reference renders.

The open alternative to the hosted gpt-image / Gemini references: the same evidence sheet and
prompt go in, one isolated object render on black comes out. Runs wherever a ~60 GB GPU is
available — the Modal function in deploy/modal/image_edit is the default host.
"""

from __future__ import annotations

import io
import os
from typing import Any

# Apache-2.0 and multi-image aware, so it can be used commercially and accepts the evidence sheet
# as-is. Any checkpoint served by diffusers' QwenImageEditPlusPipeline (e.g. a later 25xx release)
# drops in through $LR_QWEN_IMAGE_MODEL.
DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2509"

# Upstream's recommended sampling for the Plus pipeline. True CFG needs a non-empty negative
# prompt to switch on; a single space is the documented way to enable it without steering.
DEFAULT_STEPS = 40
DEFAULT_TRUE_CFG = 4.0
NEGATIVE_PROMPT = " "
# The pipeline rejects anything longer, and the reference prompts run to several hundred tokens.
MAX_SEQUENCE_LENGTH = 1024


def load_pipeline(model_id: str | None = None) -> Any:
    """Load the edit pipeline onto the GPU in bf16 (~58 GB: 20B transformer + 7B text encoder)."""
    import torch
    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_id or os.environ.get("LR_QWEN_IMAGE_MODEL") or DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def edit(
    pipe: Any,
    images: list[bytes],
    prompt: str,
    *,
    steps: int = DEFAULT_STEPS,
    true_cfg_scale: float = DEFAULT_TRUE_CFG,
    seed: int = 0,
) -> bytes:
    """Run one edit over `images` (encoded bytes) and return the render as PNG bytes."""
    import torch
    from PIL import Image

    inputs = [Image.open(io.BytesIO(raw)).convert("RGB") for raw in images]
    result = pipe(
        image=inputs if len(inputs) > 1 else inputs[0],
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=steps,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    out = io.BytesIO()
    result.images[0].save(out, format="PNG")
    return out.getvalue()
