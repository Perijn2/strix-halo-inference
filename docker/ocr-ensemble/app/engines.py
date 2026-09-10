"""OCR engine adapters for the ensemble sidecar.

Author: Perijn
Summary: Wraps MinerU, PP-OCRv5, Surya, and Tesseract behind one recognize() contract with isolated lazy loading.
Usage: Imported by app.main. Each adapter loads lazily and reports its own availability instead of failing the service.

Every adapter is loaded independently. An engine whose import path, model files, or
native library is unavailable is marked unavailable with a precise reason and the
service continues with the remaining engines. A validation service that cannot
start is worth less than one that reports exactly what it could not load.

The MinerU adapter carries both the transformers and vLLM backends so moving from
one to the other is an OCR_MINERU_BACKEND change, not a code change.
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from dataclasses import dataclass, field
from typing import Any

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
OCR_DEVICE = os.environ.get("OCR_DEVICE", "cpu")
MINERU_BACKEND = os.environ.get("OCR_MINERU_BACKEND", "transformers")
MINERU_MODEL_PATH = os.environ.get(
    "OCR_MINERU_MODEL_PATH", f"{MODELS_DIR}/mineru/MinerU2.5-Pro-2605-1.2B"
)
PADDLE_LANGUAGE = os.environ.get("OCR_PADDLE_LANGUAGE", "en")
TESSERACT_LANGUAGE = os.environ.get("OCR_TESSERACT_LANGUAGE", "eng")


@dataclass
class EngineResult:
    """One engine's read of a single page.

    Args:
        engine: Adapter name, matching the ensemble configuration key.
        kind: ``vlm`` for autoregressive models, ``classical`` for non-generative
            engines. The fusion layer treats these classes differently because a
            classical engine cannot hallucinate fluent prose.
        text: Recognized text for the page.
        confidence: Engine-reported confidence in ``[0, 1]``, or ``None`` when the
            engine exposes no usable confidence signal. VLMs fall in the ``None``
            category, which is why the ensemble cannot rely on confidence alone.
        blocks: JSON-serializable per-text blocks. Every geometry-bearing block has
            a normalized ``bbox`` in ``[left, top, right, bottom]`` form, with each
            coordinate in ``[0, 1]`` relative to the rendered page.
        error: Set when the engine ran but failed on this page.
    """

    engine: str
    kind: str
    text: str = ""
    confidence: float | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class EngineUnavailable(RuntimeError):
    """Raised by an adapter's loader when that engine cannot run in this image."""


class BaseEngine:
    """Lazy-loading engine with cached availability state.

    Subclasses implement ``_load`` returning a callable that accepts a PIL image
    and yields ``(text, confidence, blocks)``. Loading happens once, on first use.
    """

    name: str = "base"
    kind: str = "classical"

    def __init__(self) -> None:
        self._invoke: Any = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        """Whether the engine loaded successfully, loading it if never attempted."""
        if self._invoke is None and self._load_error is None:
            try:
                self._invoke = self._load()
            except Exception as exc:  # noqa: BLE001 - isolation is the contract here.
                self._load_error = f"{type(exc).__name__}: {exc}"
        return self._invoke is not None

    @property
    def load_error(self) -> str | None:
        """Reason the engine failed to load, when it did."""
        return self._load_error

    def recognize(self, image: Any) -> EngineResult:
        """Run this engine over one PIL image.

        Args:
            image: A ``PIL.Image.Image`` of a single rendered document page.

        Returns:
            An ``EngineResult``. Failures are reported on the result rather than
            raised, so one bad page cannot take down the ensemble.
        """
        if not self.available:
            return EngineResult(
                engine=self.name, kind=self.kind, error=f"unavailable: {self._load_error}"
            )
        try:
            text, confidence, blocks = self._invoke(image)
        except Exception as exc:  # noqa: BLE001 - per-page failures are data, not crashes.
            return EngineResult(
                engine=self.name, kind=self.kind, error=f"{type(exc).__name__}: {exc}"
            )
        return EngineResult(
            engine=self.name,
            kind=self.kind,
            text=text,
            confidence=confidence,
            blocks=blocks,
        )

    def _load(self) -> Any:  # pragma: no cover - abstract.
        raise NotImplementedError


class MinerUEngine(BaseEngine):
    """MinerU2.5-Pro primary parser, transformers now and vLLM later.

    The backend is selected by ``OCR_MINERU_BACKEND``. Both paths construct a
    ``MinerUClient`` and call ``two_step_extract``, so the adapter body is shared.
    """

    name = "mineru"
    kind = "vlm"

    def _load(self) -> Any:
        from mineru_vl_utils import MinerUClient

        if MINERU_BACKEND == "transformers":
            from mineru_vl_utils.transformers_loading import (
                load_transformers_model,
                load_transformers_processor,
            )

            model = load_transformers_model(MINERU_MODEL_PATH, device_map=OCR_DEVICE)
            processor = load_transformers_processor(MINERU_MODEL_PATH)
            client = MinerUClient(
                backend="transformers", model=model, processor=processor
            )
        elif MINERU_BACKEND == "vllm-engine":
            try:
                from vllm import LLM

                from mineru_vl_utils import MinerULogitsProcessor
            except ImportError as exc:  # pragma: no cover - optional extra.
                raise EngineUnavailable(
                    "OCR_MINERU_BACKEND=vllm-engine needs the vLLM ROCm image. "
                    f"Import failed: {exc}"
                ) from exc

            llm = LLM(
                model=MINERU_MODEL_PATH,
                logits_processors=[MinerULogitsProcessor],
                gpu_memory_utilization=float(
                    os.environ.get("OCR_VLLM_GPU_MEMORY_UTILIZATION", "0.35")
                ),
            )
            client = MinerUClient(backend="vllm-engine", vllm_llm=llm)
        else:
            raise EngineUnavailable(
                f"Unknown OCR_MINERU_BACKEND {MINERU_BACKEND!r}; "
                "expected 'transformers' or 'vllm-engine'."
            )

        def _run(image: Any) -> tuple[str, float | None, list[dict[str, Any]]]:
            extracted = client.two_step_extract(image)
            blocks = [_mineru_block(block) for block in extracted]
            parts = [block["text"] for block in blocks if block["text"]]
            return "\n\n".join(parts), None, blocks

        return _run


class PPOCRv5Engine(BaseEngine):
    """PP-OCRv5 two-stage detection plus recognition.

    Non-autoregressive, so it cannot produce the fluent-hallucination class that
    the VLM engines can. It also reports per-line confidence, which no VLM here
    exposes. Runs on CPU because no PaddlePaddle ROCm build exists for gfx1151.
    """

    name = "ppocrv5"
    kind = "classical"

    def _load(self) -> Any:
        from paddleocr import PaddleOCR

        # The mobile PP-OCRv5 detector and recognizer are the small classical
        # validation pair selected for this stack. Paddle's current 3.x API uses
        # explicit model names and ``predict`` rather than the legacy ``ocr`` call.
        ocr = PaddleOCR(
            text_detection_model_name=os.environ.get(
                "OCR_PADDLE_DETECTION_MODEL", "PP-OCRv5_mobile_det"
            ),
            text_recognition_model_name=os.environ.get(
                "OCR_PADDLE_RECOGNITION_MODEL", "PP-OCRv5_mobile_rec"
            ),
            lang=PADDLE_LANGUAGE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=OCR_DEVICE,
        )

        def _run(image: Any) -> tuple[str, float | None, list[dict[str, Any]]]:
            lines: list[str] = []
            confidences: list[float] = []
            blocks: list[dict[str, Any]] = []
            for text, confidence, bbox in _iter_paddle_pairs(ocr, image):
                lines.append(text)
                if confidence is not None:
                    confidences.append(confidence)
                blocks.append(
                    {
                        "text": text,
                        "bbox": bbox,
                        "confidence": confidence,
                        "type": "text",
                    }
                )
            mean_conf = (
                sum(confidences) / len(confidences) if confidences else None
            )
            return "\n".join(lines), mean_conf, blocks

        return _run


class SuryaEngine(BaseEngine):
    """Surya 2 independent layout, reading-order, and table cross-check.

    Surya's SDK pins Pillow below 11 while MinerU pins it at 11 or above, so it
    runs in the purpose-built ``surya-client`` sidecar. This adapter makes the
    client an ordinary ensemble participant without reintroducing that impossible
    dependency resolution into the MinerU/Paddle image.
    """

    name = "surya"
    kind = "vlm"

    def _load(self) -> Any:
        import httpx

        url = os.environ.get("SURYA_CLIENT_URL", "http://surya-client:8091/ocr")
        timeout = float(os.environ.get("SURYA_CLIENT_TIMEOUT_SECONDS", "600"))
        client = httpx.Client(timeout=timeout)

        def _run(image: Any) -> tuple[str, float | None, list[dict[str, Any]]]:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            response = client.post(
                url,
                json={"image_b64": base64.b64encode(buffer.getvalue()).decode("ascii")},
            )
            response.raise_for_status()
            payload = response.json()
            text = payload.get("text")
            if not isinstance(text, str):
                raise EngineUnavailable("Surya client returned no text field")
            confidence = payload.get("confidence")
            raw_blocks = payload.get("blocks") or []
            if not isinstance(raw_blocks, list):
                raise EngineUnavailable("Surya client returned non-list blocks")
            blocks = [_surya_block(block) for block in raw_blocks]
            return text, (float(confidence) if confidence is not None else None), blocks

        return _run


class TesseractEngine(BaseEngine):
    """Tesseract CLI adapter with per-word confidence from TSV output.

    Maximum lineage distance from the Qwen-derived VLMs. Its confidence is
    directionally useful but not well calibrated, so the fusion layer consumes it
    asymmetrically: only a high-confidence Tesseract disagreement is treated as a
    meaningful signal.
    """

    name = "tesseract"
    kind = "classical"

    def _load(self) -> Any:
        import pytesseract
        from PIL import Image  # noqa: F401 - validates Pillow availability for image_to_data.

        def _run(image: Any) -> tuple[str, float | None, list[dict[str, Any]]]:
            data = pytesseract.image_to_data(
                image, lang=TESSERACT_LANGUAGE, output_type=pytesseract.Output.DICT
            )
            words: list[str] = []
            confidences: list[float] = []
            blocks: list[dict[str, Any]] = []
            width, height = image.size
            for index, word in enumerate(data["text"]):
                text = str(word).strip()
                if not text:
                    continue
                words.append(text)
                confidence = _tesseract_confidence(data["conf"][index])
                if confidence is not None:
                    confidences.append(confidence)
                blocks.append(
                    {
                        "text": text,
                        "bbox": _pixel_bbox_to_normalized(
                            data["left"][index],
                            data["top"][index],
                            _add(data["left"][index], data["width"][index]),
                            _add(data["top"][index], data["height"][index]),
                            width,
                            height,
                        ),
                        "confidence": confidence,
                        "type": "word",
                    }
                )
            mean_conf = (
                sum(confidences) / len(confidences) if confidences else None
            )
            return " ".join(words), mean_conf, blocks

        return _run


def _iter_paddle_pairs(ocr: Any, image: Any) -> Any:
    """Yield text, confidence, and normalized geometry from PaddleOCR 3.x.

    ``PaddleOCR.predict`` returns result objects whose ``json`` payload contains
    aligned ``rec_texts``, ``rec_scores``, and ``rec_polys`` arrays. Recognition
    polygons are reduced to axis-aligned normalized boxes so all engines expose
    one serializable geometry contract.
    """
    width, height = image.size
    for page in ocr.predict(np_array(image)):
        raw = page
        json_payload = getattr(page, "json", None)
        if json_payload is not None:
            raw = json_payload() if callable(json_payload) else json_payload
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise EngineUnavailable(
                f"unexpected PaddleOCR 3.x result type: {type(raw).__name__}"
            )
        texts = raw.get("rec_texts") or []
        scores = raw.get("rec_scores") or []
        polygons = raw.get("rec_polys") or []
        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else None
            polygon = polygons[index] if index < len(polygons) else None
            yield (
                str(text),
                float(score) if score is not None else None,
                _polygon_to_normalized_bbox(polygon, width, height),
            )


def _mineru_block(block: Any) -> dict[str, Any]:
    """Convert one MinerU block to the ensemble's normalized geometry contract."""
    bbox = getattr(block, "bbox", None)
    return {
        "text": str(getattr(block, "content", "") or ""),
        # MinerU documents bbox coordinates as normalized already.
        "bbox": _normalized_bbox(bbox),
        "confidence": None,
        "type": str(getattr(block, "type", "unknown")),
    }


def _surya_block(block: Any) -> dict[str, Any]:
    """Validate one normalized Surya-client block before returning it to fusion."""
    if not isinstance(block, dict):
        raise EngineUnavailable("Surya client returned a non-object block")
    return {
        "text": str(block.get("text") or ""),
        "bbox": _normalized_bbox(block.get("bbox")),
        "confidence": _optional_float(block.get("confidence")),
        "type": str(block.get("type") or "unknown"),
    }


def _polygon_to_normalized_bbox(
    polygon: Any, width: int, height: int
) -> list[float] | None:
    """Reduce a PaddleOCR recognition polygon to a normalized axis-aligned box."""
    if not isinstance(polygon, (list, tuple)) or not polygon:
        return None
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
    except (IndexError, TypeError, ValueError):
        return None
    return _pixel_bbox_to_normalized(
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
        width,
        height,
    )


def _pixel_bbox_to_normalized(
    left: Any, top: Any, right: Any, bottom: Any, width: int, height: int
) -> list[float] | None:
    """Normalize one pixel-space box to ``[left, top, right, bottom]``."""
    if width <= 0 or height <= 0:
        return None
    try:
        return _clamp_bbox(
            [float(left) / width, float(top) / height, float(right) / width, float(bottom) / height]
        )
    except (TypeError, ValueError):
        return None


def _normalized_bbox(value: Any) -> list[float] | None:
    """Validate and clamp an already normalized four-coordinate bbox."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return _clamp_bbox([float(coordinate) for coordinate in value])
    except (TypeError, ValueError):
        return None


def _clamp_bbox(value: list[float]) -> list[float]:
    """Clamp a normalized box and ensure its right/bottom follow left/top."""
    left, top, right, bottom = (min(1.0, max(0.0, coordinate)) for coordinate in value)
    return [min(left, right), min(top, bottom), max(left, right), max(top, bottom)]


def _optional_float(value: Any) -> float | None:
    """Return a finite confidence value when one is supplied."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _add(left: Any, right: Any) -> float:
    """Add OCR TSV numeric fields that may arrive as strings."""
    return float(left) + float(right)


def _tesseract_confidence(value: Any) -> float | None:
    """Convert Tesseract's 0–100 score, excluding its ``-1`` sentinel."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed / 100.0 if parsed >= 0 else None


def np_array(image: Any) -> Any:
    """Convert a PIL image to the BGR ndarray PaddleOCR expects."""
    import numpy as np

    return np.asarray(image.convert("RGB"))


# Numeric tokens are the concrete hallucination surface: a dosage, an amount, or an
# identifier that appears in the parse but nowhere on the page.
_NUMBER_PATTERN = re.compile(r"\d[\d,]*\.?\d*")


def numeric_tokens(text: str) -> list[str]:
    """Return numeric tokens from text, normalized by stripping thousands separators.

    Args:
        text: Any OCR output.

    Returns:
        Normalized numeric strings, deduplicated while preserving order.
    """
    seen: set[str] = set()
    tokens: list[str] = []
    for match in _NUMBER_PATTERN.findall(text):
        normalized = match.replace(",", "")
        if normalized not in seen:
            seen.add(normalized)
            tokens.append(normalized)
    return tokens


def build_engines() -> list[BaseEngine]:
    """Instantiate every configured engine without loading any of them yet."""
    return [MinerUEngine(), PPOCRv5Engine(), SuryaEngine(), TesseractEngine()]
