"""Durable PostgreSQL and filesystem storage for OCR document audits.

Author: Perijn
Summary: Persists original uploads, rendered pages, and audit JSON for the configured retention period.
Usage: Imported by app.main. The PostgreSQL schema is created idempotently at service startup, allowing existing database volumes to adopt OCR history.

Original documents and rendered page PNGs live under OCR_AUDIT_DATA_DIR. PostgreSQL
stores the audit ledger and relative paths only; deleting an expired document removes
both the database row and its corresponding directory.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE TABLE IF NOT EXISTS ocr_documents (
  id UUID PRIMARY KEY, filename TEXT NOT NULL, media_type TEXT NOT NULL,
  source_path TEXT NOT NULL, page_count INTEGER NOT NULL CHECK (page_count > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), expires_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS ocr_pages (
  id UUID PRIMARY KEY, document_id UUID NOT NULL REFERENCES ocr_documents(id) ON DELETE CASCADE,
  page_number INTEGER NOT NULL CHECK (page_number > 0), image_path TEXT NOT NULL,
  audit JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (document_id, page_number)
);
CREATE INDEX IF NOT EXISTS ocr_documents_expires_at_idx ON ocr_documents (expires_at);
CREATE INDEX IF NOT EXISTS ocr_pages_review_idx ON ocr_pages ((audit->>'review'));
"""


class AuditStore:
    """Stores OCR review artifacts using PostgreSQL metadata and a private volume."""

    def __init__(self) -> None:
        """Read storage configuration from the Compose-provided environment."""
        self.root = Path(os.environ.get("OCR_AUDIT_DATA_DIR", "/audit-data"))
        self.retention_days = int(os.environ.get("OCR_AUDIT_RETENTION_DAYS", "90"))
        self.host = os.environ.get("OCR_AUDIT_DB_HOST", "postgres")
        self.database = os.environ.get("OCR_AUDIT_DB_NAME", "inference")
        self.user = os.environ.get("OCR_AUDIT_DB_USER", "inference")
        self.password_file = Path(
            os.environ.get("OCR_AUDIT_DB_PASSWORD_FILE", "/run/secrets/postgres_password")
        )

    def initialize(self) -> None:
        """Create storage directories and ensure the ledger schema exists."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(_SCHEMA)
        self.cleanup_expired()

    def save_document(
        self, *, filename: str, media_type: str, source: bytes, pages: list[bytes]
    ) -> tuple[UUID, list[Path]]:
        """Persist one source document and its rendered page PNGs.

        Args:
            filename: Original caller-provided file name.
            media_type: Declared source media type.
            source: Original document/image bytes.
            pages: Rendered PNG bytes in page order.

        Returns:
            New document UUID and absolute rendered-page paths.
        """
        self.cleanup_expired()
        document_id = uuid4()
        directory = self.root / str(document_id)
        directory.mkdir(parents=True, exist_ok=False)
        source_path = directory / "source"
        source_path.write_bytes(source)
        page_paths = []
        for number, page in enumerate(pages, start=1):
            path = directory / f"page-{number:04d}.png"
            path.write_bytes(page)
            page_paths.append(path)
        expires_at = datetime.now(UTC) + timedelta(days=self.retention_days)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO ocr_documents
                   (id, filename, media_type, source_path, page_count, expires_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (document_id, filename, media_type, str(source_path), len(page_paths), expires_at),
            )
        return document_id, page_paths

    def save_page_audit(self, document_id: UUID, page_number: int, audit: dict[str, Any]) -> None:
        """Persist one page's complete, JSON-serializable audit record."""
        page_path = self.root / str(document_id) / f"page-{page_number:04d}.png"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO ocr_pages (id, document_id, page_number, image_path, audit)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (document_id, page_number)
                   DO UPDATE SET audit = EXCLUDED.audit, image_path = EXCLUDED.image_path""",
                (uuid4(), document_id, page_number, str(page_path), json.dumps(audit)),
            )

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent document summaries in newest-first order."""
        with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT d.id, d.filename, d.media_type, d.page_count, d.created_at, d.expires_at,
                          count(p.id) AS audited_pages,
                          bool_or((p.audit->>'review')::boolean) AS needs_review
                   FROM ocr_documents d LEFT JOIN ocr_pages p ON p.document_id = d.id
                   GROUP BY d.id ORDER BY d.created_at DESC LIMIT %s""",
                (limit,),
            )
            return list(cursor.fetchall())

    def get_document(self, document_id: UUID) -> dict[str, Any] | None:
        """Return one document and ordered page audit records, when retained."""
        with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM ocr_documents WHERE id = %s", (document_id,))
            document = cursor.fetchone()
            if document is None:
                return None
            cursor.execute(
                "SELECT page_number, audit FROM ocr_pages WHERE document_id = %s ORDER BY page_number",
                (document_id,),
            )
            document["pages"] = list(cursor.fetchall())
            return document

    def get_page_path(self, document_id: UUID, page_number: int) -> Path | None:
        """Return a retained rendered-page path only when its document still exists."""
        path = self.root / str(document_id) / f"page-{page_number:04d}.png"
        return path if path.is_file() else None

    def cleanup_expired(self) -> int:
        """Delete expired ledger rows and their private artifact directories."""
        with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("DELETE FROM ocr_documents WHERE expires_at <= now() RETURNING id")
            expired = list(cursor.fetchall())
        for row in expired:
            shutil.rmtree(self.root / str(row["id"]), ignore_errors=True)
        return len(expired)

    def _connect(self) -> psycopg.Connection[Any]:
        """Connect with the Docker secret rather than a password environment variable."""
        password = self.password_file.read_text(encoding="utf-8").strip()
        return psycopg.connect(
            host=self.host, dbname=self.database, user=self.user, password=password
        )
