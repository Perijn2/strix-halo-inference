"""Regression tests for OCR request routing and audit persistence.

Author: Perijn
Summary: Verifies engine-subset propagation and cleanup of failed document persistence.
Usage: Run from docker/ocr-ensemble with `python -m unittest discover -s tests -v` after installing requirements.txt.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.engines import EngineResult
from app.main import _run_ensemble
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


if __name__ == "__main__":
    unittest.main()
