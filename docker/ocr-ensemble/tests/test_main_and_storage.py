"""Regression tests for OCR request routing and audit persistence.

Author: Perijn
Summary: Verifies engine-subset propagation and cleanup of failed document persistence.
Usage: Run from docker/ocr-ensemble with `python -m unittest discover -s tests -v` after installing requirements.txt.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.engines import BaseEngine, EngineResult, _optional_float, numeric_tokens
from app.main import _render_document, _run_ensemble
from app.storage import AuditStore


class _Engine:
    """Minimal OCR adapter used to observe ensemble selection behavior."""

    def __init__(self, name: str) -> None:
        """Store the adapter name returned in the synthetic result."""
        self.name = name

    def recognize(self, image: object) -> EngineResult:
        """Return a successful result without requiring a real OCR runtime."""
        return EngineResult(self.name, "classical", "test text")


class MainRoutingTests(unittest.TestCase):
    """Verify the native OCR endpoint's subset policy plumbing."""

    def test_selected_engines_are_run_without_auto_acceptance(self) -> None:
        """Pass a caller-selected subset into both selection and fusion policy."""
        engine = _Engine("tesseract")
        with (
            patch("app.main._select_engines", return_value=[engine]) as select,
            patch("app.main._FUSION.fuse", return_value="fused") as fuse,
        ):
            fused, results = _run_ensemble(object(), ["tesseract"])

        self.assertEqual("fused", fused)
        self.assertEqual(["tesseract"], [result.engine for result in results])
        select.assert_called_once_with(["tesseract"])
        fuse.assert_called_once_with(results, allow_auto_accept=False)


class DocumentLimitTests(unittest.TestCase):
    """Verify durable document limits reject oversized work before OCR starts."""

    def test_source_byte_limit_rejects_before_decoding(self) -> None:
        """Reject an oversized upload before a parser can allocate for it."""
        from fastapi import HTTPException

        with patch("app.main._MAX_SOURCE_BYTES", 3):
            with self.assertRaises(HTTPException) as raised:
                _render_document(b"four", "image/png", "scan.png")

        self.assertEqual(413, raised.exception.status_code)

    def test_image_pixel_limit_rejects_decoded_image(self) -> None:
        """Reject a valid image whose raster exceeds the configured pixel cap."""
        from fastapi import HTTPException
        from PIL import Image

        image = Image.new("RGB", (3, 2))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        with patch("app.main._MAX_PAGE_PIXELS", 5):
            with self.assertRaises(HTTPException) as raised:
                _render_document(encoded.getvalue(), "image/png", "scan.png")

        self.assertEqual(413, raised.exception.status_code)


class _Cursor:
    """In-memory database cursor that records SQL issued by read-path tests."""

    def __init__(self, row: object | None = None) -> None:
        """Initialize an empty query log and an optional fetch-one result."""
        self.queries: list[str] = []
        self.row = row

    def __enter__(self) -> _Cursor:
        """Enter the synthetic cursor context."""
        return self

    def __exit__(self, *args: object) -> None:
        """Leave the synthetic cursor context without suppressing errors."""
        return None

    def execute(self, query: str, parameters: object = None) -> None:
        """Record the SQL text without contacting PostgreSQL."""
        del parameters
        self.queries.append(query)

    def fetchall(self) -> list[object]:
        """Return no synthetic list rows."""
        return []

    def fetchone(self) -> object | None:
        """Return the configured synthetic row."""
        return self.row


class _Connection:
    """In-memory PostgreSQL connection that returns a fixed cursor."""

    def __init__(self, cursor: _Cursor) -> None:
        """Store the cursor returned for every query."""
        self._cursor = cursor

    def __enter__(self) -> _Connection:
        """Enter the synthetic connection context."""
        return self

    def __exit__(self, *args: object) -> None:
        """Leave the synthetic connection context without suppressing errors."""
        return None

    def cursor(self, **kwargs: object) -> _Cursor:
        """Return the fixed cursor while accepting psycopg keyword arguments."""
        del kwargs
        return self._cursor


class EngineInitializationTests(unittest.TestCase):
    """Verify expensive engine setup cannot race between health and OCR threads."""

    def test_loader_runs_once_under_concurrent_availability_checks(self) -> None:
        """Serialize simultaneous first-use checks through the engine load lock."""
        calls = 0
        calls_lock = threading.Lock()

        class _SlowEngine(BaseEngine):
            """Synthetic loader that records every attempted initialization."""

            def _load(self) -> object:
                nonlocal calls
                with calls_lock:
                    calls += 1
                threading.Event().wait(0.02)
                return lambda image: ("", None, [])

        engine = _SlowEngine()
        threads = [threading.Thread(target=lambda: engine.available) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(1, calls)

    def test_numeric_normalization_preserves_decimal_commas_and_ids(self) -> None:
        """Handle common European and US values without converting commas blindly."""
        self.assertEqual(
            ["1.5", "1234.56", "2024", "01", "03", "001,234", "5.0"],
            numeric_tokens("1,5 1.5 1,234.56 1.234,56 2024-01-03 ID 001,234 dose 5,0"),
        )

    def test_non_finite_confidence_is_rejected(self) -> None:
        """Keep NaN and infinity out of JSON and confidence calculations."""
        self.assertIsNone(_optional_float("nan"))
        self.assertIsNone(_optional_float("inf"))
        self.assertEqual(0.5, _optional_float("0.5"))


class AuditStoreTests(unittest.TestCase):
    """Verify filesystem cleanup and expiry predicates for durable audit storage."""

    def test_failed_database_write_removes_document_artifacts(self) -> None:
        """Remove source and rendered pages when the transaction cannot start."""
        with tempfile.TemporaryDirectory() as directory:
            store = AuditStore()
            store.root = Path(directory)
            store.cleanup_expired = lambda: 0
            store._connect = lambda: (_ for _ in ()).throw(
                RuntimeError("database down")
            )
            document_id = uuid4()

            with self.assertRaisesRegex(RuntimeError, "database down"):
                store.save_document_with_audits(
                    document_id=document_id,
                    filename="scan.png",
                    media_type="image/png",
                    source=b"source",
                    pages=[b"page"],
                    audits=[{"review": True}],
                )

            self.assertFalse((store.root / str(document_id)).exists())

    def test_artifact_quota_rejects_before_writing_files(self) -> None:
        """Reject a document that would exceed the retained-artifact quota."""
        with tempfile.TemporaryDirectory() as directory:
            store = AuditStore()
            store.root = Path(directory)
            store.max_artifact_bytes = 1
            store.cleanup_expired = lambda: 0

            with self.assertRaisesRegex(ValueError, "artifact quota"):
                store.save_document_with_audits(
                    document_id=uuid4(),
                    filename="scan.png",
                    media_type="image/png",
                    source=b"source",
                    pages=[b"page"],
                    audits=[{"review": True}],
                )

            self.assertEqual([], list(Path(directory).iterdir()))

    def test_document_read_queries_filter_expired_records(self) -> None:
        """Include a database expiry predicate after cleanup to close read races."""
        store = AuditStore()
        store.cleanup_expired = lambda: 0

        list_cursor = _Cursor()
        store._connect = lambda: _Connection(list_cursor)
        self.assertEqual([], store.list_documents())
        self.assertIn("expires_at > now()", list_cursor.queries[0])

        document_cursor = _Cursor()
        store._connect = lambda: _Connection(document_cursor)
        self.assertIsNone(store.get_document(uuid4()))
        self.assertIn("expires_at > now()", document_cursor.queries[0])


@unittest.skipUnless(
    os.environ.get("OCR_TEST_POSTGRES_DSN"), "requires OCR_TEST_POSTGRES_DSN"
)
class PostgresAuditIntegrationTests(unittest.TestCase):
    """Exercise psycopg JSONB adaptation against a real PostgreSQL server."""

    def test_audit_is_inserted_as_jsonb(self) -> None:
        """Persist an audit and have PostgreSQL report its actual JSONB column type."""
        import psycopg

        with tempfile.TemporaryDirectory() as directory:
            store = AuditStore()
            store.root = Path(directory)
            store._connect = lambda: psycopg.connect(
                os.environ["OCR_TEST_POSTGRES_DSN"]
            )
            store.initialize()
            document_id = uuid4()
            store.save_document_with_audits(
                document_id=document_id,
                filename="one-page.png",
                media_type="image/png",
                source=b"source",
                pages=[b"page"],
                audits=[{"review": True, "engine_outputs": {"ppocrv5": {}}}],
            )
            with store._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_typeof(audit)::text, audit->>'review' FROM ocr_pages WHERE document_id = %s",
                    (document_id,),
                )
                self.assertEqual(("jsonb", "true"), cursor.fetchone())


if __name__ == "__main__":
    unittest.main()
