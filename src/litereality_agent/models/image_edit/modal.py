"""Qwen-Image-Edit through one scale-to-zero Modal function."""

from __future__ import annotations

import base64
from typing import Any

from litereality_agent.runtimes.modal import ModalClient


class ModalImageEditService:
    """Reference-render adapter for the deployed Modal image-edit function."""

    name = "image-edit-modal"

    def __init__(
        self,
        *,
        app_name: str,
        function_name: str = "image_edit",
        environment_name: str = "main",
        profile: str | None = None,
        credentials: tuple[str, str] | None = None,
        client: Any = None,
    ) -> None:
        self.client = client or ModalClient(
            app_name,
            function_name,
            environment_name=environment_name,
            profile=profile,
            credentials=credentials,
        )

    def edit(
        self,
        images: list[bytes],
        prompt: str,
        *,
        model_id: str | None = None,
        steps: int | None = None,
        seed: int = 0,
    ) -> bytes:
        """Send encoded `images` plus `prompt`; return the rendered PNG bytes."""
        payload = {
            "images_b64": [base64.b64encode(raw).decode("ascii") for raw in images],
            "prompt": prompt,
            "model_id": model_id,
            "steps": steps,
            "seed": seed,
        }
        output = self.client.map([payload])[0]
        if isinstance(output, BaseException):
            raise RuntimeError(f"Modal image edit failed: {output}") from output
        if output.get("error"):
            raise RuntimeError(str(output["error"]))
        return base64.b64decode(output["png_b64"], validate=True)
