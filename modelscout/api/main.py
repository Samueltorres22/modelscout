"""Minimal FastAPI app: a pipeline trigger and a real RAG query interface.

Not a "serving layer" yet (no auth, no async job queue, no dashboard) --
POST /pipeline/run runs synchronously, which is fine for proving the trigger
path works end to end. A production version would move this to
BackgroundTasks + a run-status endpoint.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from modelscout.api import queries
from modelscout.api.schemas import (
    AgentCallSummary,
    DigestRun,
    ModelDetail,
    ModelsResponse,
    ModelSummary,
    ObservabilitySummary,
    RunPipelineRequest,
    RunsResponse,
    SearchResponse,
    SearchResultItem,
)
from modelscout.db import get_connection
from modelscout.pipeline import run as run_pipeline
from modelscout.rag.embeddings import _get_model
from modelscout.rag.search import semantic_search

logging.basicConfig(level=logging.WARNING)
logging.getLogger("modelscout").setLevel(logging.INFO)

app = FastAPI(title="ModelScout API", version="0.1.0")

# The dashboard (dashboard/) runs as its own Vite dev server on a different
# origin/port than this API -- local dev only, no auth on either side yet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        db_ok = True
    except Exception:
        db_ok = False

    try:
        _get_model()
        embedding_model_ok = True
    except Exception:
        embedding_model_ok = False

    return {"db_ok": db_ok, "embedding_model_ok": embedding_model_ok}


@app.post("/pipeline/run")
def pipeline_run(req: RunPipelineRequest) -> dict:
    try:
        return run_pipeline(req.profile_name, limit_per_tag=req.limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/search", response_model=SearchResponse)
def search(q: str, k: int = 5) -> SearchResponse:
    with get_connection() as conn:
        results = semantic_search(conn, q, k=k)

    return SearchResponse(
        query=q,
        results=[
            SearchResultItem(
                model_id=r.model_id,
                chunk_text=r.chunk_text,
                score=r.score,
                pipeline_tag=r.pipeline_tag,
                downloads=r.downloads,
                hf_url=r.hf_url,
            )
            for r in results
        ],
    )


@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    with get_connection() as conn:
        rows = queries.get_models_catalog(conn)
    return ModelsResponse(models=[ModelSummary(**row) for row in rows])


@app.get("/models/{model_id:path}", response_model=ModelDetail)
def get_model(model_id: str) -> ModelDetail:
    with get_connection() as conn:
        row = queries.get_model_detail(conn, model_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    row["hf_url"] = f"https://huggingface.co/{row['model_id']}"
    return ModelDetail(**row)


@app.get("/runs", response_model=RunsResponse)
def list_runs() -> RunsResponse:
    with get_connection() as conn:
        rows = queries.get_digest_runs(conn)
    return RunsResponse(runs=[DigestRun(**row) for row in rows])


@app.get("/observability/summary", response_model=ObservabilitySummary)
def observability_summary() -> ObservabilitySummary:
    with get_connection() as conn:
        rows = queries.get_observability_summary(conn)
    return ObservabilitySummary(agents=[AgentCallSummary(**row) for row in rows])
