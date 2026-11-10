-- Documents, chunk sets and chunks. Chunks are list-partitioned by chunk set; each partition
-- gets its own HNSW index when the set is built (see rag_arena/db/repo.py).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id       text PRIMARY KEY,
    path     text NOT NULL,
    title    text NOT NULL,
    url      text NOT NULL,
    text     text NOT NULL,
    sha256   text NOT NULL,
    sections jsonb NOT NULL DEFAULT '[]'::jsonb
);

CREATE TABLE chunk_sets (
    id              text PRIMARY KEY,
    chunker         text NOT NULL,
    params          jsonb NOT NULL,
    embedding_model text NOT NULL,
    embedding_dim   integer NOT NULL CHECK (embedding_dim > 0),
    chunk_count     integer NOT NULL DEFAULT 0,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- The embedding column has no fixed dimension so chunk sets can use different embedding
-- models; each partition's HNSW index casts to its set's dimension.
CREATE TABLE chunks (
    chunk_set    text NOT NULL REFERENCES chunk_sets (id) ON DELETE CASCADE,
    id           text NOT NULL,
    document_id  text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal      integer NOT NULL,
    start_char   integer NOT NULL,
    end_char     integer NOT NULL,
    text         text NOT NULL,
    index_text   text NOT NULL,
    heading_path text[] NOT NULL,
    url          text NOT NULL,
    embedding    vector,
    PRIMARY KEY (chunk_set, id),
    CHECK (end_char > start_char)
) PARTITION BY LIST (chunk_set);

CREATE INDEX chunks_document_idx ON chunks (chunk_set, document_id, ordinal);
