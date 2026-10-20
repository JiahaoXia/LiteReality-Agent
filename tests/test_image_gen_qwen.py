"""The open-weights Qwen reference path, and failing loudly when no image backend is usable.

The second half matters because the callers catch per-object errors and write a placeholder: a
missing key used to feed TRELLIS the raw crop collage for every object while reporting success.
"""

from __future__ import annotations

import base64

import pytest

from litereality_agent.models.image_edit.modal import ModalImageEditService
from litereality_agent.pipeline.scene_init.ingest.references import image_gen


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LR_IMAGE_PROVIDER", "LR_OPENAI_IMAGE_MODEL", "LR_QWEN_IMAGE_MODEL",
                "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(image_gen.time, "sleep", lambda _: None)


class _Fn:
    def __init__(self, reply):
        self.reply = reply
        self.payloads: list[dict] = []

    def map(self, inputs):
        self.payloads.extend(inputs)
        return [self.reply]


def _service(reply) -> tuple[ModalImageEditService, _Fn]:
    fn = _Fn(reply)
    return ModalImageEditService(app_name="litereality", client=fn), fn


def _sheet(tmp_path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"sheet-bytes")
    return sheet


@pytest.mark.parametrize(
    "env,model,expected",
    [
        ("qwen", None, "qwen"),
        ("QWEN", "gpt-image-2", "qwen"),
        (None, "Qwen/Qwen-Image-Edit-2509", "qwen"),
        (None, "qwen-image-edit", "qwen"),
        (None, "gpt-image-2", "openai"),
        (None, "gemini-2.5-flash-image", "gemini"),
    ],
)
def test_provider_resolution(monkeypatch, env, model, expected):
    if env:
        monkeypatch.setenv("LR_IMAGE_PROVIDER", env)
    assert image_gen.resolve_provider(model) == expected


def test_qwen_round_trips_through_modal(tmp_path, monkeypatch):
    png = b"\x89PNG-render"
    service, fn = _service({"png_b64": base64.b64encode(png).decode()})
    monkeypatch.setattr(image_gen, "_qwen_service", lambda: service)
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "qwen")
    out = tmp_path / "out.png"

    # The callers' default --image-model is an OpenAI name; it must not reach the Qwen function.
    meta = image_gen.generate_reference(_sheet(tmp_path), "a chair", out, model="gpt-image-2")

    assert out.read_bytes() == png
    assert meta["model"] == "qwen:Qwen/Qwen-Image-Edit-2509"
    assert meta["cost_usd"] == 0.0
    (payload,) = fn.payloads
    assert base64.b64decode(payload["images_b64"][0]) == b"sheet-bytes"
    assert payload["model_id"] == "Qwen/Qwen-Image-Edit-2509"
    assert payload["prompt"].startswith("a chair")
    assert "SOLID PURE BLACK" in payload["prompt"]


def test_qwen_model_override(tmp_path, monkeypatch):
    service, fn = _service({"png_b64": base64.b64encode(b"x").decode()})
    monkeypatch.setattr(image_gen, "_qwen_service", lambda: service)
    image_gen.generate_reference(
        _sheet(tmp_path), "p", tmp_path / "out.png", model="Qwen/Qwen-Image-Edit-2511"
    )
    assert fn.payloads[0]["model_id"] == "Qwen/Qwen-Image-Edit-2511"


def test_qwen_remote_error_is_retried_then_raised(tmp_path, monkeypatch):
    service, fn = _service({"error": "CUDA out of memory"})
    monkeypatch.setattr(image_gen, "_qwen_service", lambda: service)
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "qwen")

    with pytest.raises(RuntimeError, match="CUDA out of memory") as info:
        image_gen.generate_reference(_sheet(tmp_path), "p", tmp_path / "out.png", retries=2)

    assert not isinstance(info.value, image_gen.ImageGenConfigError)
    assert len(fn.payloads) == 2


def test_qwen_without_modal_is_a_config_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "litereality_agent.models.registry.image_edit_from_settings", lambda: None
    )
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "qwen")
    with pytest.raises(image_gen.ImageGenConfigError, match="MODAL_TOKEN_ID"):
        image_gen.generate_reference(_sheet(tmp_path), "p", tmp_path / "out.png")


def test_undeployed_function_fails_on_the_first_attempt(tmp_path, monkeypatch):
    class NotFoundError(Exception):
        pass

    class _Missing:
        calls = 0

        def edit(self, *_, **__):
            _Missing.calls += 1
            raise RuntimeError("lookup failed") from NotFoundError("image_edit")

    monkeypatch.setattr(image_gen, "_qwen_service", lambda: _Missing())
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "qwen")
    with pytest.raises(image_gen.ImageGenConfigError, match="litereality setup"):
        image_gen.generate_reference(_sheet(tmp_path), "p", tmp_path / "out.png", retries=3)
    assert _Missing.calls == 1


def test_missing_openai_key_is_a_config_error(tmp_path):
    with pytest.raises(image_gen.ImageGenConfigError, match="OPENAI_API_KEY"):
        image_gen.generate_reference(_sheet(tmp_path), "p", tmp_path / "out.png")


def test_missing_key_stops_the_scan_instead_of_writing_placeholders(tmp_path, monkeypatch):
    from litereality_agent.pipeline.scene_init.ingest.references import object_references

    obj_dir = tmp_path / "Table_0"
    obj_dir.mkdir()
    monkeypatch.setattr(object_references, "load_scene_objects", lambda scan: {})
    monkeypatch.setattr(object_references, "object_dirs", lambda *a: [obj_dir])
    monkeypatch.setattr(
        object_references, "select_evidence_images", lambda *a: [{"image": "x", "score": 1.0}]
    )
    monkeypatch.setattr(object_references, "make_weighted_input_sheet",
                        lambda scored, out: (out.parent.mkdir(parents=True, exist_ok=True),
                                             out.write_bytes(b"x")))

    with pytest.raises(image_gen.ImageGenConfigError):
        object_references.generate_for_scan("scan", tmp_path / "out")
