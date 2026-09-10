"""OCR ensemble service and durable document-review API.

Author: Perijn
Summary: Runs four-engine page OCR, renders uploaded PDFs, persists 90-day audit history, and exposes review artifacts through llama-swap.
Usage: Uvicorn serves this module on port 8090. Use POST /documents for PDF/image review workspaces, GET /history for audits, and POST /ocr for a transient single-page audit.

The OpenAI chat endpoint remains deliberately narrow and stateless. The native
review endpoints retain the source document, rendered pages, complete audit JSON,
and per-engine geometry in the private audit volume plus PostgreSQL ledger.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import threading
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .engines import EngineResult, build_engines
from .fusion import FusionConfig, FusionEngine, FusionResult
from .storage import AuditStore

_run_lock = threading.Lock()
_ENGINES = build_engines()
_STORE = AuditStore()
_CONFIG = FusionConfig(
    agree=float(os.environ.get("OCR_AGREE_THRESHOLD", "0.92")),
    classical_pair_agree=float(os.environ.get("OCR_CLASSICAL_PAIR_AGREE", "0.90")),
    classical_pair_hard=float(os.environ.get("OCR_CLASSICAL_PAIR_HARD", "0.75")),
    entropy_accept=float(os.environ.get("OCR_ENTROPY_ACCEPT_MAX", "0.25")),
    entropy_review_floor=float(os.environ.get("OCR_ENTROPY_REVIEW_FLOOR", "0.60")),
)
_FUSION = FusionEngine(_CONFIG)

app = FastAPI(
    title="strix-halo ocr-ensemble",
    summary="Four-engine OCR with Consensus Entropy, PDF rendering, and durable review audits.",
    version="0.2.0",
)


class OcrRequest(BaseModel):
    """One rendered page submitted for a transient full audit."""

    image_b64: str = Field(
        ..., description="Base64-encoded PNG, JPEG, or WebP page image."
    )
    page_ref: str | None = Field(None, description="Caller correlation id.")
    engines: list[str] | None = Field(
        None,
        description="Optional diagnostic engine subset; subset results always require review.",
    )


class DocumentRequest(BaseModel):
    """One image or PDF source document submitted for durable review."""

    document_b64: str = Field(
        ..., description="Base64-encoded image or PDF source document."
    )
    filename: str = Field(..., min_length=1, max_length=255)
    media_type: str = Field("application/pdf", max_length=127)


class OcrResponse(BaseModel):
    """Fused page result with engine status and geometry audit data."""

    page_ref: str | None = None
    verdict: str
    review: bool
    priority: int
    text: str
    entropy: float | None
    pairwise: dict[str, float] = Field(default_factory=dict)
    ungrounded: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    engines: dict[str, Any] = Field(default_factory=dict)
    engine_outputs: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int = 0


class DocumentResponse(BaseModel):
    """Persisted document identifier and page-level OCR audits."""

    id: UUID
    filename: str
    media_type: str
    page_count: int
    expires_at: str
    pages: list[OcrResponse]


@app.on_event("startup")
def initialize_storage() -> None:
    """Ensure audit storage exists before accepting source documents."""
    _STORE.initialize()


def _decode_base64(payload: str, label: str) -> bytes:
    """Decode a plain base64 or data-URI payload into bytes."""
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{label} is not valid base64") from exc


def _decode_image(payload: str) -> Any:
    """Decode a base64 image payload into an RGB PIL image."""
    from PIL import Image

    data = _decode_base64(payload, "image_b64")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - malformed caller data is a 422.
        raise HTTPException(status_code=422, detail=f"image_b64 is not a decodable image: {exc}") from exc
    return image.convert("RGB")


def _render_document(source: bytes, media_type: str, filename: str) -> list[bytes]:
    """Render a PDF or image source into page PNG bytes at a review-safe DPI."""
    from PIL import Image

    is_pdf = media_type == "application/pdf" or filename.lower().endswith(".pdf")
    if not is_pdf:
        try:
            image = Image.open(io.BytesIO(source)).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - malformed caller data is a 422.
            raise HTTPException(status_code=422, detail=f"source image cannot be decoded: {exc}") from exc
        return [_png_bytes(image)]

    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(source)
        pages = []
        # 192 DPI preserves small print without exploding CPU RAM on long PDFs.
        for page in document:
            pages.append(_png_bytes(page.render(scale=192 / 72).to_pil().convert("RGB")))
    except Exception as exc:  # noqa: BLE001 - external PDF parser boundary.
        raise HTTPException(status_code=422, detail=f"PDF rendering failed: {exc}") from exc
    if not pages:
        raise HTTPException(status_code=422, detail="PDF contains no renderable pages")
    return pages


def _png_bytes(image: Any) -> bytes:
    """Encode one PIL image as deterministic PNG bytes for durable overlay display."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _select_engines(names: list[str] | None) -> list[Any]:
    """Return requested engine adapters or all configured engines."""
    if names is None:
        return list(_ENGINES)
    by_name = {engine.name: engine for engine in _ENGINES}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown engine(s): {unknown}")
    return [by_name[name] for name in names]


def _run_ensemble(
    image: Any, names: list[str] | None
) -> tuple[FusionResult, list[EngineResult]]:
    """Run selected engines serially under the configured memory boundary.

    A caller-selected subset is useful for diagnostic comparisons, but it cannot
    yield a four-engine validated acceptance verdict.
    """
    with _run_lock:
        results = [engine.recognize(image) for engine in _select_engines(names)]
    return _FUSION.fuse(results, allow_auto_accept=names is None), results


def _response_from_results(
    fused: FusionResult,
    results: list[EngineResult],
    page_ref: str | None,
    elapsed_ms: int,
) -> OcrResponse:
    """Convert fusion and engine results into a JSON-safe public audit record."""
    outputs = {
        result.engine: {
            "text": result.text,
            "confidence": result.confidence,
            "blocks": getattr(result, "blocks", []),
            "error": result.error,
        }
        for result in results
    }
    return OcrResponse(
        page_ref=page_ref,
        verdict=fused.verdict,
        review=fused.review,
        priority=fused.priority,
        text=fused.text,
        entropy=fused.entropy,
        pairwise=fused.pairwise,
        ungrounded=fused.ungrounded,
        reasons=fused.reasons,
        engines=fused.engines,
        engine_outputs=outputs,
        elapsed_ms=elapsed_ms,
    )


def _audit_page(
    image: Any, page_ref: str | None = None, names: list[str] | None = None
) -> OcrResponse:
    """Run one page through the requested ensemble and retain its audit in memory."""
    started = time.monotonic()
    fused, results = _run_ensemble(image, names)
    return _response_from_results(
        fused, results, page_ref, int((time.monotonic() - started) * 1000)
    )


@app.get("/health")
def health(response: Response) -> dict[str, Any]:
    """Readiness probe requiring all four validation engines to load."""
    status = {
        engine.name: {
            "kind": engine.kind,
            "available": engine.available,
            "error": engine.load_error,
        }
        for engine in _ENGINES
    }
    live = [name for name, value in status.items() if value["available"]]
    ready = len(live) == len(_ENGINES)
    if not ready:
        response.status_code = 503
    return {
        "status": "ok" if ready else "degraded",
        "live_engines": live,
        "engines": status,
        "thresholds": _CONFIG.__dict__,
    }


@app.post("/ocr", response_model=OcrResponse)
def ocr(request: OcrRequest) -> OcrResponse:
    """Run a transient, single-page audit without retaining source data."""
    return _audit_page(
        _decode_image(request.image_b64), request.page_ref, request.engines
    )


@app.post("/documents", response_model=DocumentResponse)
def submit_document(request: DocumentRequest) -> DocumentResponse:
    """Render, audit, and persist every page of an uploaded image or PDF for 90 days."""
    source = _decode_base64(request.document_b64, "document_b64")
    pages = _render_document(source, request.media_type, request.filename)
    document_id, paths = _STORE.save_document(
        filename=Path(request.filename).name,
        media_type=request.media_type,
        source=source,
        pages=pages,
    )
    audits: list[OcrResponse] = []
    from PIL import Image

    for page_number, path in enumerate(paths, start=1):
        image = Image.open(path).convert("RGB")
        audit = _audit_page(image, f"{document_id}:{page_number}")
        _STORE.save_page_audit(document_id, page_number, audit.model_dump(mode="json"))
        audits.append(audit)
    document = _STORE.get_document(document_id)
    assert document is not None  # The preceding insert is transactional and required.
    return DocumentResponse(
        id=document_id,
        filename=request.filename,
        media_type=request.media_type,
        page_count=len(pages),
        expires_at=document["expires_at"].isoformat(),
        pages=audits,
    )


@app.get("/history")
def history(limit: int = 100) -> list[dict[str, Any]]:
    """List up to 100 recent retained document audits."""
    return jsonable_encoder(_STORE.list_documents(min(max(limit, 1), 100)))


@app.get("/history/{document_id}")
def document_history(document_id: UUID) -> dict[str, Any]:
    """Return a retained document with its complete page audit history."""
    document = _STORE.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found or expired")
    return jsonable_encoder(document)


@app.get("/history/{document_id}/pages/{page_number}/image")
def page_image(document_id: UUID, page_number: int) -> FileResponse:
    """Serve one retained rendered page for client-side bounding-box overlays."""
    path = _STORE.get_page_path(document_id, page_number)
    if path is None:
        raise HTTPException(status_code=404, detail="rendered page not found or expired")
    return FileResponse(path, media_type="image/png")


@app.post("/v1/chat/completions")
def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    """Return fused OCR text through the standard OpenAI image-message surface."""
    encoded = _extract_image_from_messages(payload.get("messages") or [])
    if encoded is None:
        raise HTTPException(status_code=422, detail="no image content part found")
    audit = _audit_page(_decode_image(encoded))
    return {
        "id": f"ocr-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "ocr-ensemble",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": audit.text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(audit.text), "total_tokens": len(audit.text)},
    }


def _extract_image_from_messages(messages: list[Any]) -> str | None:
    """Find the first base64/data-URI image in OpenAI-style message content."""
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            image_url = part.get("image_url")
            if part.get("type") in ("image_url", "input_image"):
                if isinstance(image_url, dict) and image_url.get("url"):
                    return str(image_url["url"])
                if isinstance(image_url, str):
                    return image_url
            if part.get("type") == "image" and part.get("image_b64"):
                return str(part["image_b64"])
    return None
