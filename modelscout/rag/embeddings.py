"""Embedding wrapper around BAAI/bge-small-en-v1.5 (384 dims, CPU-fast).

bge models are asymmetric: the query prefix below is required for good
retrieval quality on the QUERY side only -- indexed passages are embedded
plain. Getting this backwards (or applying it to both) silently degrades
search relevance without erroring, so it's centralized here rather than left
to call sites to remember.
"""

from __future__ import annotations

import threading

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    # Loaded once per process, not once per call -- matters even at this
    # project's small scale since model load is the expensive part. Plain
    # double-checked locking rather than @lru_cache: a bare lru_cache isn't
    # safe if this is ever called from multiple threads before the first
    # load completes (see the identical fix in agents/triage_node.py, where
    # this exact race was verified empirically against the classifier model).
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    model = _get_model()
    vector = model.encode(_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()
