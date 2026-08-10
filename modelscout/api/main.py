"""Minimal FastAPI app: a pipeline trigger and a real RAG query interface.

POST /pipeline/run schedules the run via BackgroundTasks and returns a
run_id immediately (202) instead of blocking the request for however long
ingestion + triage + extraction + fact-checking takes -- poll
GET /pipeline/runs/{run_id} for its outcome. See api/run_registry.py for the
(in-memory, single-process) status store backing that.
"""

from __future__ import annotations

import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from modelscout.api import queries, run_registry
from modelscout.api.schemas import (
    AgentCallSummary,
    DigestRun,
    ModelDetail,
    ModelsResponse,
    ModelSummary,
    ObservabilitySummary,
    PipelineRunAccepted,
    PipelineRunStatus,
    RunPipelineRequest,
    RunsResponse,
    SearchResponse,
    SearchResultItem,
)
from modelscout.config import load_interest_profile, settings
from modelscout.db import get_connection

logging.basicConfig(level=logging.WARNING)
logging.getLogger("modelscout").setLevel(logging.INFO)

app = FastAPI(title="ModelScout API", version="0.1.0")

# The dashboard (dashboard/) runs as its own Vite dev server on a different
# origin/port than this API in local dev -- a deployed dashboard's origin is
# added via CORS_EXTRA_ORIGINS so the same image works in both places.
_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_origins += [o.strip() for o in settings.cors_extra_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    try:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        db_ok = True
    except Exception:  # noqa: BLE001 -- a health check must report unhealthy on ANY failure, not just a chosen subset
        db_ok = False

    embedding_model_ok = None
    if settings.enable_ml_features:
        try:
            from modelscout.rag.embeddings import _get_model

            _get_model()
            embedding_model_ok = True
        except Exception:  # noqa: BLE001 -- same as above
            embedding_model_ok = False

    return {
        "db_ok": db_ok,
        "embedding_model_ok": embedding_model_ok,
        "ml_features_enabled": settings.enable_ml_features,
    }


def _execute_pipeline_run(run_id: str, profile_name: str, limit: int) -> None:
    from modelscout.pipeline import run as run_pipeline

    run_registry.mark_running(run_id)
    try:
        result = run_pipeline(profile_name, limit_per_tag=limit)
        run_registry.mark_completed(run_id, result)
    except Exception as exc:  # noqa: BLE001 -- this runs in a BackgroundTasks
        # thread with no caller to propagate to; swallowing here and
        # recording the failure in the registry is what makes it visible
        # at all, instead of vanishing into an unhandled-exception log line
        # while GET /pipeline/runs/{run_id} sits stuck on "running" forever.
        run_registry.mark_failed(run_id, str(exc))


@app.post("/pipeline/run", response_model=PipelineRunAccepted, status_code=202)
def pipeline_run(req: RunPipelineRequest, background_tasks: BackgroundTasks) -> PipelineRunAccepted:
    if not settings.enable_ml_features:
        raise HTTPException(
            status_code=503,
            detail="Pipeline runs are disabled on this deployment (ENABLE_ML_FEATURES=false). Run locally instead.",
        )

    # Validated synchronously so a bad profile_name still 404s immediately
    # instead of only surfacing as a "failed" status after the fact.
    try:
        load_interest_profile(req.profile_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run_id = run_registry.create_run()
    background_tasks.add_task(_execute_pipeline_run, run_id, req.profile_name, req.limit)
    return PipelineRunAccepted(run_id=run_id, status="pending")


@app.get("/pipeline/runs/{run_id}", response_model=PipelineRunStatus)
def pipeline_run_status(run_id: str) -> PipelineRunStatus:
    run = run_registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return PipelineRunStatus(run_id=run["run_id"], status=run["status"], result=run["result"], error=run["error"])


@app.get("/search", response_model=SearchResponse)
def search(q: str, k: int = 5) -> SearchResponse:
    if not settings.enable_ml_features:
        raise HTTPException(
            status_code=503,
            detail="Semantic search is disabled on this deployment (ENABLE_ML_FEATURES=false). Run locally instead.",
        )
    from modelscout.rag.search import semantic_search

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
