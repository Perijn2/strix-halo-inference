<!--
Author: Perijn
Summary: Describes safe operation, profiling, rollback, and review handling for the inference and validated OCR stack.

Usage:
Core principle: Preserve Qwen and retrieval availability in orchestration mode; treat OCR disagreement as a review signal, never as a license to silently rewrite source documents.

Setup: Configure `.env`, download verified model artifacts with `scripts/download-models.sh`, run `scripts/validate.sh`, build Compose, and confirm all service health endpoints.

Workflow: Check Caddy, llama-swap, ocr-ensemble, Surya, and PostgreSQL health. Send normal model requests to stable role IDs. Upload a PDF/image to the review workspace or `/upstream/ocr-ensemble/documents`; consume the rendered pages and structured audit only for review, not as a replacement for the original document.

API guide: `/health` verifies server readiness, `/v1/models` lists callable role IDs, `/api/profiles` reports or changes llama-swap routing state, `/upstream/ocr-ensemble/ocr` returns one transient audit, and `/upstream/ocr-ensemble/documents` creates a retained multi-page audit.

Worked example: Start the stack, wait for the OCR sidecar's first model load, upload a difficult PDF to `/ocr-playground/`, and review every page with `review: true` before releasing an extraction.
-->

# Operations

## Memory policy

Halogen uses two 131K-context slots and a 262K shared KV pool. `HALOGEN_HOST_RESERVE_GIB` is intentionally set to 28 GiB, leaving headroom for the persistent embedding and reranker workloads. Do not raise slots, context, or pool size without a fresh load and concurrency benchmark.

Qwen has no inactivity-based unload: llama-swap sets its TTL to zero. Halogen’s startup can take minutes because it loads a large checkpoint. llama-swap waits for the model `/health` endpoint before forwarding the first request.

## Halogen PP/TG telemetry

Halogen reports its measured prefill and decode timings in its per-request `serve_api:` log ledger, not in the OpenAI response. `halogen-telemetry-proxy` is launched automatically between llama-swap and Halogen: it parses that ledger and adds llama.cpp-compatible `timings` to completed chat responses. llama-swap then records prompt-processing (PP) and token-generation (TG) speeds in Activity. The proxy preserves streaming and inserts its timing event immediately before the terminal SSE `[DONE]` event.

The timings are the engine's own prefill and decode-window measurements. Halogen does not expose a request ID in this ledger, so the proxy serializes decorated chat requests and only accepts ledger lines emitted after each request begins; this trades telemetry throughput for correct attribution. A missing PP/TG value means that no matching ledger line was available before the proxy's two-second wait; investigate the llama-swap upstream log stream rather than substituting HTTP wall-clock speed. Confirm the adapter after deployment with a non-streaming `role/general` chat request and `GET /api/metrics/activity?model=qwen3.8-flash-next`.

The OCR budget is deliberately separate and strict: `OCR_ENSEMBLE_MEMORY_LIMIT_GIB=6`, `OCR_SURYA_CLIENT_MEMORY_LIMIT_GIB=2`, and `OCR_SURYA_MEMORY_LIMIT_GIB=4` total 12 GiB. The Surya SDK client is separate because it requires Pillow <11 while MinerU requires Pillow >=11. The ensemble starts CPU-first (`OCR_DEVICE=cpu`) because upstream ROCm and PaddlePaddle do not validate `gfx1151`; Surya uses the existing Vulkan llama.cpp image. Do not change any cap, move MinerU to vLLM, or enable a Python GPU device until you measure all resident Qwen, Ornith, embedding, reranker, MinerU, and Surya memory together.

If you later move MinerU to vLLM, rebuild the OCR image from a tested TheRock-compatible base and set `OCR_VLLM_GPU_MEMORY_UTILIZATION` conservatively. vLLM’s available-memory accounting is not reliable on unified-memory hardware, so it must not be allowed to preallocate around the existing 28 GiB host reserve.

## OCR validation and review

`role/ocr` is a llama-swap role, but the fan-out and fusion happen in the `ocr-ensemble` sidecar. llama-swap starts a supervised local `socat` forwarder and proxies it; it cannot itself vote between models. The sidecar runs MinerU2.5-Pro-2605 as primary, PP-OCRv5 and Tesseract as classical checks, and calls the compatibility-isolated Surya SDK client, which calls the internal Surya worker for independent VLM validation.

The normal OpenAI endpoint returns only fused text. Submit a base64 page image to `POST /upstream/ocr-ensemble/ocr` when your caller needs the audit record. Persist the original page image and the audit response together; do not preserve extracted text without its source page and verdict.

| Verdict | Meaning | Required action |
| --- | --- | --- |
| `accept` | Every live engine agreed above the calibrated threshold and Consensus Entropy was low. | May proceed; retain the audit record. |
| `accept_weighted` | MinerU had at least one classical corroborator, but the page was not unanimous. | Spot-check according to your sampling policy. |
| `review_hallucination` | A mutually corroborating classical pair disagreed with MinerU, or MinerU/selected output emitted an ungrounded numeric value. | Highest-priority human review. |
| `review_hard_page` | PP-OCRv5 and Tesseract disagree strongly. | Review regardless of either VLM result. |
| `review_entropy` | Too little corroboration, high disagreement, or no engine returned text. | Review before release. |

The numeric grounding rule is intentionally conservative: it only treats a primary-model numeric token as strong evidence when the classical pair corroborates each other. A model-generated bounding box is not independent evidence; verify regulated values against a trusted PDF text layer or the original rendered page.

### Threshold calibration

The defaults in `.env.example` are starting points, not benchmark-derived guarantees. Before production:

1. Assemble a held-out corpus containing your actual difficult documents: faint scans, stamps, handwriting over print, skew, near-blank pages, tables, and multi-column layouts.
2. Record the raw per-engine text and the audit verdict for every page.
3. Label extraction faults against the original page or a trusted PDF text layer.
4. Tune only with a held-out split. Never tune against the same pages used to assert an error rate.
5. Choose a review capacity target, then measure fault discovery at that queue depth. Do not turn off a structural or numeric review rule to make the queue shorter.

## Switching modes

The active llama-swap profile is an API routing policy, not a background scheduler. Select `engineering` before implementation work. The matrix permits the configured Qwen, Ornith, retrieval, and OCR forwarder processes concurrently; memory limits—not profile names—remain the practical resource boundary. You can explicitly unload Qwen through `POST /api/models/unload/qwen3.8-flash-next`; the next Qwen role request reloads Halogen. Do not rely on an idle timeout.

## Update policy

Pin and validate any production image digest after a successful soak test. Test llama.cpp fork updates with the PP512, PP2048, TG64, 64K-context, retrieval, OCR image request, review-verdict, and restart measurements before replacing the current image.

The OCR `requirements.txt` intentionally uses constrained version ranges until one complete target-host build resolves. After that first green build, freeze the resolved Python package versions, keep the lockfile or image digest with the corpus results, and rerun the calibration suite after any MinerU, PaddleOCR, Surya, Tesseract, or Consensus Entropy update.

## PostgreSQL and OCR retention

PostgreSQL starts by default. It is the durable OCR audit ledger and may also store RAG vectors/metadata; it does not participate in model routing. Original sources and rendered pages reside in the private `ocr_audit_data` Docker volume, while PostgreSQL stores document metadata and page audit JSON. The default durable endpoint limits sources to 24 MiB, PDFs to 100 pages, rendered pages to 40 million pixels each, and total retained artifacts to 2 GiB (`OCR_MAX_*` and `OCR_AUDIT_MAX_BYTES`). On every document submission, the ensemble removes records and artifact directories whose expiry has passed, then reconciles UUID artifact directories against live ledger IDs to clean any crash-orphaned data. `OCR_AUDIT_RETENTION_DAYS` defaults to 90; change it only with an explicit retention-policy decision.

Back up both PostgreSQL and the `ocr_audit_data` volume together if audit traceability matters. The browser workspace and unauthenticated Caddy listener expose retained source documents to anyone with network access to the stack; restrict that access before uploading sensitive files.