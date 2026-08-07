-- ModelScout v1 schema. Applied automatically on first container start via
-- docker-entrypoint-initdb.d. To re-apply after editing, you must recreate the
-- volume (docker compose down -v) since Postgres only runs init scripts once.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS models (
    model_id        TEXT PRIMARY KEY,              -- HF repo id, e.g. "Qwen/Qwen2-VL-2B-Instruct"
    pipeline_tag    TEXT,
    downloads       BIGINT,
    likes           BIGINT,
    last_modified   TIMESTAMPTZ,
    tags            JSONB,
    card_data       JSONB,                          -- raw HF ModelInfo.card_data
    readme_text     TEXT,
    matched_profile TEXT,                            -- which interest profile ingested this
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_card_chunks (
    id          BIGSERIAL PRIMARY KEY,
    model_id    TEXT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   vector(384) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS model_card_chunks_embedding_idx
    ON model_card_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS triage_results (
    id           BIGSERIAL PRIMARY KEY,
    model_id     TEXT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    profile_name TEXT NOT NULL,
    is_relevant  BOOLEAN NOT NULL,
    confidence   REAL NOT NULL,
    raw_labels   JSONB,                              -- full zero-shot label/score distribution
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS extracted_specs (
    id                     BIGSERIAL PRIMARY KEY,
    model_id               TEXT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    params_billion          REAL,
    license                TEXT,
    architecture_family     TEXT,
    hardware_requirements   TEXT,
    quantization_available  JSONB,                    -- list[str]
    declared_benchmarks     JSONB,                    -- list[{name, metric, score}]
    parse_error            BOOLEAN NOT NULL DEFAULT FALSE,
    parse_error_detail      TEXT,
    raw_model_response      JSONB,                     -- audit trail: tool_use.input or raw text fallback
    extracted_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS digest_runs (
    id                   BIGSERIAL PRIMARY KEY,
    profile_name          TEXT NOT NULL,
    run_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    n_ingested            INT,
    n_triage_pass          INT,
    n_extracted            INT,
    n_parse_errors          INT,
    digest_markdown_path    TEXT
);
