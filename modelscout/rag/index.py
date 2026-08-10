"""Chunk + embed model card READMEs from `models` into `model_card_chunks`."""

from __future__ import annotations

from modelscout.rag.chunking import chunk_readme
from modelscout.rag.embeddings import embed_passages

_UPSERT_CHUNK_SQL = """
INSERT INTO model_card_chunks (model_id, chunk_index, chunk_text, embedding)
VALUES (%s, %s, %s, %s)
ON CONFLICT (model_id, chunk_index) DO UPDATE SET
    chunk_text = EXCLUDED.chunk_text,
    embedding = EXCLUDED.embedding
"""


def index_model_readmes(conn, model_ids: list[str] | None = None) -> int:
    """Chunk + embed READMEs for the given model_ids (or all models if None),
    upserting into model_card_chunks. Returns the number of chunks written.
    """
    with conn.cursor() as cur:
        # model_ids is None -> index every model; model_ids == [] -> index
        # none. `if model_ids:` would treat both the same (both falsy),
        # which silently re-indexes the entire table on the empty-list case
        # -- verified live: pipeline.run() always passes a list here, and a
        # profile run where 0 candidates pass triage/filters hung for
        # minutes re-embedding every existing model's README instead of
        # doing nothing.
        if model_ids is not None:
            cur.execute(
                "SELECT model_id, readme_text FROM models WHERE model_id = ANY(%s)",
                (model_ids,),
            )
        else:
            cur.execute("SELECT model_id, readme_text FROM models")
        rows = cur.fetchall()

    total_chunks = 0
    with conn.cursor() as cur:
        for model_id, readme_text in rows:
            chunks = chunk_readme(readme_text or "")
            if not chunks:
                continue
            vectors = embed_passages(chunks)
            for idx, (text, vector) in enumerate(zip(chunks, vectors, strict=True)):
                cur.execute(_UPSERT_CHUNK_SQL, (model_id, idx, text, vector))
            total_chunks += len(chunks)

    return total_chunks
