# Qwen-Image-Edit on Modal

The open-weights alternative to the hosted reference-image APIs. Each object's evidence sheet and
prompt go to [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Apache-2.0,
ungated) and one isolated render on black comes back — the same contract as the OpenAI and Gemini
paths, so nothing downstream changes.

Select it in `.env`; no OpenAI or Gemini key is needed:

```dotenv
LR_IMAGE_PROVIDER=qwen
# LR_QWEN_IMAGE_MODEL=Qwen/Qwen-Image-Edit-2509   # any QwenImageEditPlusPipeline checkpoint
```

`litereality setup` deploys it with the other functions. By hand:

```bash
MODAL_PROFILE=<profile> modal deploy --env main deploy/modal/app.py
```

## Cost and throughput

It runs in bf16 on one H100 (~58 GB of weights: the 20B edit transformer plus its Qwen2.5-VL
text encoder). The first call downloads the weights into the `litereality-image-edit-models`
Volume, which takes several minutes once; later cold starts load from the Volume.

The function starts at **one container** so concurrent cold starts cannot each pull the weights
into the same cache. Once the first call has finished, redeploy with
`LR_IMAGE_EDIT_MAX_CONTAINERS=4` set so a room's references stop queueing behind one GPU.

## Status

Experimental. The prompts were tuned against gpt-image, and they ask for exact counts of repeated
parts and no invented features — which open edit models follow less reliably. Compare a scan's
references and TRELLIS meshes against the OpenAI path before relying on it.

For the full setup — including testing the model on a local GPU before deploying — see
[`deploy/qwen-image-edit.md`](../../qwen-image-edit.md).
