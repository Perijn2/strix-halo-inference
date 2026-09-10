-- Author: Perijn
-- Summary: Creates durable document and page audit tables for the OCR review workspace.
-- Usage: PostgreSQL executes this for a new inference database; app storage also creates the same schema for existing volumes.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS ocr_documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename TEXT NOT NULL,
  media_type TEXT NOT NULL,
  source_path TEXT NOT NULL,
  page_count INTEGER NOT NULL CHECK (page_count > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS ocr_pages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES ocr_documents(id) ON DELETE CASCADE,
  page_number INTEGER NOT NULL CHECK (page_number > 0),
  image_path TEXT NOT NULL,
  audit JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (document_id, page_number)
);

CREATE INDEX IF NOT EXISTS ocr_documents_expires_at_idx ON ocr_documents (expires_at);
CREATE INDEX IF NOT EXISTS ocr_pages_review_idx ON ocr_pages ((audit->>'review'));
