# Setting up Qwen-Image-Edit for reference images

Qwen-Image-Edit is the open-weights replacement for the OpenAI / Gemini reference-image call
(`LR_IMAGE_PROVIDER=qwen`). It takes each object's evidence sheet plus prompt and returns one
isolated render on black, which TRELLIS then turns into a mesh.

- Model: [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509),
  Apache-2.0, ungated (no Hugging Face token needed).
- Size: ~58 GB in bf16 — a 20B edit transformer (~41 GB) plus a Qwen2.5-VL-7B text encoder
  (~17 GB) and a small VAE.
- Status: **experimental.** The pipeline prompts were tuned on gpt-image; compare results before
  relying on it (see [Evaluating quality](#4-evaluating-quality)).

There are two ways to run it:

| | Where it runs | Used by `litereality run` today? |
|---|---|---|
| [A. Modal](#2-path-a--modal-what-the-pipeline-uses) | one H100 in your Modal workspace | **yes** |
| [B. This server](#3-path-b--local-on-this-server) | the lab's RTX 6000 Ada GPUs | no — standalone testing only |

Path B is the cheapest way to try the model on real evidence sheets before paying for Modal time.
Wiring a local backend into the pipeline is not implemented yet (see [Status](#6-status-and-next-steps)).

---

## 1. This server's hardware (checked 2026-10-20)

| | |
|---|---|
| GPUs | 4 × NVIDIA RTX 6000 Ada, **48 GB each**, compute capability 8.9 |
| Driver / CUDA | 595.71.05 / CUDA 13.2 (runs PyTorch cu12x wheels) |
| System RAM | **16 GB** |
| CPU | 16 cores |
| Disk | ~1.4 TB free on `/` |

At the time of checking, **GPUs 0 and 3 were at 100 % utilisation** (30.7 GB and 2.6 GB in use by
other jobs) and **GPUs 1 and 2 were idle**. Re-check with `nvidia-smi` before you start and pick
idle cards.

What this means for Qwen-Image-Edit:

- **The full bf16 model (~58 GB) does not fit on one 48 GB card.** Either split it across two
  cards (Option B1) or quantize it to fit one (Option B2).
- **16 GB of RAM rules out CPU offload.** `enable_model_cpu_offload()` and
  `enable_sequential_cpu_offload()` park weights in system RAM and will run the machine out of
  memory. Load weights straight onto the GPUs, as below; safetensors are read shard by shard, so
  loading itself stays within RAM.

---

## 2. Path A — Modal (what the pipeline uses)

You need the Modal token pair in `.env` (see the main [README](../README.md#install)).

**1. Select the provider** in `.env`:

```dotenv
LR_IMAGE_PROVIDER=qwen
# Optional — any QwenImageEditPlusPipeline checkpoint:
# LR_QWEN_IMAGE_MODEL=Qwen/Qwen-Image-Edit-2509
```

No `OPENAI_API_KEY` or `GEMINI_API_KEY` is needed.

**2. Deploy.** Deploy the combined app, which is the app name the pipeline looks up (`litereality`):

```bash
uv sync --extra modal
uv run modal deploy --env main deploy/modal/app.py
```

> `litereality setup` currently deploys each function as its own app (`litereality-dino`,
> `litereality-trellis`, `litereality-image-edit`), while the pipeline defaults to looking them up
> in `litereality`. Until that is fixed, use the command above, or set
> `MODAL_IMAGE_EDIT_APP=litereality-image-edit` if you deployed with `setup`.

**3. Warm the cache.** The first call downloads ~58 GB into the `litereality-image-edit-models`
Volume and takes several minutes. The function starts with **one container** so parallel cold
starts cannot race on that download. Run one scan (or one object, below) first.

**4. Raise concurrency** once the cache is populated, so a room's references stop queueing
behind a single GPU:

```bash
LR_IMAGE_EDIT_MAX_CONTAINERS=4 uv run modal deploy --env main deploy/modal/app.py
```

**5. Run as usual:**

```bash
uv run litereality run <scan>
```

Each reference logs as `[qwen-image] <file> $0.0`. The cost shows as zero because it is billed as
Modal H100 time, not per image.

If anything is misconfigured — no Modal credentials, the `modal` extra missing, or the function
not deployed — the run now **stops with an explanation** instead of silently using the raw crop
collage as every object's reference.

---

## 3. Path B — local on this server

### 3.1 Create a separate environment

Keep it apart from the app environment: it pulls a ~3 GB PyTorch install.

```bash
cd /home/jx198/XJH/workspace/LiteReality-Agent
uv venv backends/qwen-env --python 3.11
source backends/qwen-env/bin/activate
uv pip install "torch==2.8.0" "torchvision==0.23.0" \
    "diffusers==0.36.0" "transformers==4.57.5" "accelerate==1.10.1" \
    "bitsandbytes" "pillow>=10"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

These are the same versions the Modal image pins. `bitsandbytes` is only needed for Option B2.

### 3.2 Download the weights once

Put the Hugging Face cache on a disk with ≥ 70 GB free and keep it there for later runs:

```bash
export HF_HOME=/home/jx198/XJH/hf-cache        # or any large disk
hf download Qwen/Qwen-Image-Edit-2509            # ~58 GB, no token needed
```

If `hf` is missing: `uv pip install -U "huggingface_hub[cli]"`.

### 3.3 Smoke-test script

Save this as `backends/qwen_smoke.py`. It runs one edit using the same sampling settings as
`src/litereality_agent/models/image_edit/inference.py`.

```python
"""One Qwen-Image-Edit reference render on this server.

usage: python qwen_smoke.py SHEET PROMPT_TXT OUT_PNG [bf16|nf4]
"""
import sys
import time
from pathlib import Path

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

sheet, prompt_file, out = map(Path, sys.argv[1:4])
mode = sys.argv[4] if len(sys.argv) > 4 else "bf16"
model = "Qwen/Qwen-Image-Edit-2509"

t0 = time.time()
if mode == "bf16":
    # B1: whole components spread over the visible GPUs (transformer on one, text encoder on other).
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="balanced")
else:
    # B2: 4-bit NF4 transformer + text encoder, fits one 48 GB card with room to spare.
    from diffusers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                      "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder"],
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, quantization_config=quant).to("cuda")
print(f"loaded in {time.time() - t0:.0f}s")

# The pipeline appends the background instruction itself; match its black-background path.
prompt = prompt_file.read_text() + " Place it on a SOLID PURE BLACK (#000000) background."
t0 = time.time()
image = pipe(
    image=Image.open(sheet).convert("RGB"),
    prompt=prompt,
    negative_prompt=" ",
    true_cfg_scale=4.0,
    num_inference_steps=40,
    guidance_scale=1.0,
    generator=torch.manual_seed(0),
    max_sequence_length=1024,
).images[0]
image.save(out)
print(f"edited in {time.time() - t0:.0f}s -> {out}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} peak {torch.cuda.max_memory_allocated(i) / 2**30:.1f} GiB")
```

### 3.4 Run it

Use the evidence sheet and prompt that a previous pipeline run already wrote for an object:

```bash
OBJ=<output_root>/object_refs/<scan>/<Object>     # e.g. .../object_refs/lab_scan/Table_0
```

**Option B1 — full bf16 across two idle GPUs (best quality, recommended):**

```bash
CUDA_VISIBLE_DEVICES=1,2 python backends/qwen_smoke.py \
    $OBJ/input2imagegen.jpg $OBJ/prompt.txt $OBJ/qwen_bf16.png bf16
```

`device_map="balanced"` places whole components, so the ~41 GB transformer goes on one card and
the text encoder and VAE on the other. That leaves about 7 GB of headroom on the transformer's
card for activations at 1024². It should fit; if it runs out of memory, use B2.

**Option B2 — 4-bit on one GPU (fits easily, slight quality loss):**

```bash
CUDA_VISIBLE_DEVICES=1 python backends/qwen_smoke.py \
    $OBJ/input2imagegen.jpg $OBJ/prompt.txt $OBJ/qwen_nf4.png nf4
```

This needs about 20 GB of GPU memory, so it can also share a card with another job.

The first load reads ~58 GB from disk, so expect a few minutes. The script prints load time, edit
time and peak memory per GPU — note them, they decide whether local serving is worth wiring in.

---

## 4. Evaluating quality

The open model has to match gpt-image on what the prompt demands: **one object only, exact counts
of repeated parts (drawers, legs, panes), nothing invented, clutter removed.** For 5–10 objects
from one scan, put side by side:

1. `input2imagegen.jpg` — the evidence sheet
2. `clean_obj_reference.png` — the existing OpenAI / Gemini reference
3. `qwen_bf16.png` (and `qwen_nf4.png`)

Then, for the promising ones, run the full pipeline on Modal with `LR_IMAGE_PROVIDER=qwen` and
compare the TRELLIS meshes and the completeness-gate verdicts against the OpenAI run.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` in B1 | Check both visible cards are idle (`nvidia-smi`); otherwise use B2. |
| Process killed while loading | Out of system RAM (16 GB) — something enabled CPU offload or another job is using RAM. Load straight to GPU as in the script. |
| `ImportError: QwenImageEditPlusPipeline` | diffusers is older than 0.36 — reinstall with the pinned version. |
| `max_sequence_length cannot be greater than 1024` | Keep `max_sequence_length=1024`; the reference prompts are long. |
| Pipeline stops with `Qwen image generation runs on Modal` | Path A only: add the Modal token pair to `.env`, `uv sync --extra modal`. |
| Pipeline stops with `image_edit function is not deployed` | Deploy `deploy/modal/app.py` (step 2 of Path A), or set `MODAL_IMAGE_EDIT_APP`. |
| Output has background clutter or extra objects | Expected failure mode for open edit models; record it in the comparison and try a different seed or the 4-bit vs bf16 variant. |

---

## 6. Status and next steps

- **Implemented** (branch `feat/open-image-provider`): the `qwen` provider in
  `pipeline/scene_init/ingest/references/image_gen.py`, the Modal function in
  `deploy/modal/image_edit/`, and tests in `tests/test_image_gen_qwen.py`.
- **Not yet run:** no live Modal deploy or real-scan comparison.
- **Not implemented:** a local backend that lets `litereality run` use this server's GPUs. If
  Path B quality and speed look good, the natural shape is a worker subprocess in its own
  environment, the same pattern as `LR_DINO_PYTHON` / `TRELLIS_PYTHON` (see
  [local-gpu.md](local-gpu.md)), using B1 or B2 loading.
- **Possible improvement:** Qwen-Image-Edit-2509 accepts up to three separate input images.
  Sending the best crops individually instead of one collage may improve fidelity.
