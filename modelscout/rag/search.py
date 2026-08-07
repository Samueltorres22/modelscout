"""Semantic search over indexed model card chunks -- the concrete proof RAG
works as a real query interface, not just internal glue.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelscout.rag.embeddings import embed_query

_SEARCH_SQL = """
SELECT
    c.model_id,
    c.chunk_text,
    1 - (c.embedding <=> %(qvec)s::vector) AS score,
    m.pipeline_tag,
    m.downloads
FROM model_card_chunks c
JOIN models m ON m.model_id = c.model_id
ORDER BY c.embedding <=> %(qvec)s::vector
LIMIT %(k)s
"""


@dataclass
class SearchResult:
    model_id: str
    chunk_text: str
    score: float
    pipeline_tag: str | None
    downloads: int | None
    hf_url: str


def semantic_search(conn, query: str, k: int = 5) -> list[SearchResult]:
    qvec = embed_query(query)
    # Pass as pgvector's text literal ("[v1,v2,...]") rather than a bare Python
    # list -- psycopg would otherwise send it as a double precision[] array,
    # which the `<=>` operator can't compare against `vector` without this.
    qvec_literal = "[" + ",".join(repr(float(x)) for x in qvec) + "]"
    with conn.cursor() as cur:
        cur.execute(_SEARCH_SQL, {"qvec": qvec_literal, "k": k})
        rows = cur.fetchall()

    return [
        SearchResult(
            model_id=row[0],
            chunk_text=row[1],
            score=float(row[2]),
            pipeline_tag=row[3],
            downloads=row[4],
            hf_url=f"https://huggingface.co/{row[0]}",
        )
        for row in rows
    ]
