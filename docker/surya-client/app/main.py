"""Internal HTTP client for Surya 2 OCR.

Author: Perijn
Summary: Runs Surya's SDK in its compatible Pillow environment and exposes one page-recognition endpoint to ocr-ensemble.
Usage: Uvicorn starts this module on port 8091. It is internal-only; callers use role/ocr through llama-swap instead.

Surya's current SDK requires Pillow <11 while MinerU requires Pillow >=11. This
small service contains only the SDK/client logic and delegates VLM execution to
the internal llama.cpp Surya worker at SURYA_INFERENCE_URL.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_predictor_lock = threading.Lock()
_predictor: Any = None

app = FastAPI(title="strix-halo surya client", version="0.1.0")


class SuryaRequest(BaseModel):
    """One base64-encoded rendered document page."""

    image_b64: str = Field(..., description="Base64-encoded PNG or JPEG page image.")


class SuryaBlock(BaseModel):
    """One Surya OCR block with page-relative geometry.

    Attributes:
        text: Flattened text or HTML-derived table/equation content.
        bbox: Normalized ``[left, top, right, bottom]`` coordinates in ``[0, 1]``.
        confidence: Mean decode confidence reported by Surya, when available.
        type: Canonical Surya layout label.
    """

    text: str
    bbox: list[float] | None
    confidence: float | None
    type: str


class SuryaResponse(BaseModel):
    """Flattened Surya output, mean confidence, and normalized block geometry."""

    text: str
    confidence: float | None
    blocks: list[SuryaBlock]


def _get_predictor() -> Any:
    """Create and cache Surya's remote-inference recognition predictor.

    Returns:
        A ``RecognitionPredictor`` whose inference manager uses
        ``SURYA_INFERENCE_URL`` instead of spawning a local backend.
    """
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor

            _predictor = RecognitionPredictor(SuryaInferenceManager())
        return _predictor


def _decode_image(value: str) -> Any:
    """Decode one base64 or data-URI image into RGB PIL data."""
    from PIL import Image

    if value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="image_b64 is not valid base64") from exc
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001 - client supplied malformed image.
        raise HTTPException(status_code=422, detail=f"cannot decode image: {exc}") from exc
    return image.convert("RGB")


def _html_to_text(value: str) -> str:
    """Flatten Surya's block HTML while preserving block boundaries as spaces."""
    return " ".join(re.sub(r"<[^>]+>", " ", value).split())


def _normalized_bbox(block: Any, page: Any) -> list[float] | None:
    """Normalize one Surya pixel bbox against the page's image bbox."""
    bbox = getattr(block, "bbox", None)
    image_bbox = getattr(page, "image_bbox", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if not isinstance(image_bbox, (list, tuple)) or len(image_bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        page_x0, page_y0, page_x1, page_y1 = (float(value) for value in image_bbox)
    except (TypeError, ValueError):
        return None
    width, height = page_x1 - page_x0, page_y1 - page_y0
    if width <= 0 or height <= 0:
        return None
    return _clamp_bbox(
        [(x0 - page_x0) / width, (y0 - page_y0) / height, (x1 - page_x0) / width, (y1 - page_y0) / height]
    )


def _clamp_bbox(value: list[float]) -> list[float]:
    """Clamp a normalized box and ensure its right/bottom follow left/top."""
    left, top, right, bottom = (min(1.0, max(0.0, coordinate)) for coordinate in value)
    return [min(left, right), min(top, bottom), max(left, right), max(top, bottom)]


@app.get("/health")
def health() -> dict[str, str]:
    """Return liveness without starting a Surya prediction."""
    return {"status": "ok"}


@app.post("/ocr", response_model=SuryaResponse)
def ocr(request: SuryaRequest) -> SuryaResponse:
    """Recognize one document page with the remote Surya VLM worker.

    Args:
        request: Base64-encoded page image.

    Returns:
        Flattened text in Surya reading order plus mean block confidence.
    """
    image = _decode_image(request.image_b64)
    predictor = _get_predictor()
    pages = predictor([image])
    text_parts: list[str] = []
    confidences: list[float] = []
    blocks: list[SuryaBlock] = []
    for page in pages:
        for block in getattr(page, "blocks", []) or []:
            text = _html_to_text(getattr(block, "html", "") or "")
            if text:
                text_parts.append(text)
            confidence = getattr(block, "confidence", None)
            normalized_confidence = float(confidence) if confidence is not None else None
            if normalized_confidence is not None:
                confidences.append(normalized_confidence)
            blocks.append(
                SuryaBlock(
                    text=text,
                    bbox=_normalized_bbox(block, page),
                    confidence=normalized_confidence,
                    type=str(
                        getattr(block, "label", None)
                        or getattr(block, "raw_label", None)
                        or "unknown"
                    ),
                )
            )
    return SuryaResponse(
        text="\n".join(text_parts),
        confidence=(sum(confidences) / len(confidences) if confidences else None),
        blocks=blocks,
    )
