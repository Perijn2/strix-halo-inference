-- Author: Perijn
-- Summary: Enables the pgvector extension in the inference database.
-- Usage: PostgreSQL executes this once while initializing an empty database volume.
CREATE EXTENSION IF NOT EXISTS vector;
