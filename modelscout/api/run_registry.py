"""In-memory registry for async pipeline runs. POST /pipeline/run schedules a
FastAPI BackgroundTasks job and returns immediately with a run_id; GET
/pipeline/runs/{run_id} polls this registry for status.

Deliberately in-memory, not a DB table: this project only ever runs as a
single process (locally, or ENABLE_ML_FEATURES=false in the one deployed
environment, where pipeline runs are disabled entirely -- see README's
Deploy section), so there's no multi-worker consistency problem to solve,
and losing in-flight run status on a restart is an acceptable tradeoff at
this scope. Promoting this to a table is real future work if a multi-worker
or crash-resilient deployment ever needs it -- not done preemptively.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from typing import Literal

RunStatus = Literal["pending", "running", "completed", "failed"]

_lock = threading.Lock()
_runs: dict[str, dict] = {}


def create_run() -> str:
    run_id = str(uuid.uuid4())
    with _lock:
        _runs[run_id] = {
            "run_id": run_id,
            "status": "pending",
            "created_at": datetime.now(UTC),
            "result": None,
            "error": None,
        }
    return run_id


def mark_running(run_id: str) -> None:
    with _lock:
        _runs[run_id]["status"] = "running"


def mark_completed(run_id: str, result: dict) -> None:
    with _lock:
        _runs[run_id]["status"] = "completed"
        _runs[run_id]["result"] = result


def mark_failed(run_id: str, error: str) -> None:
    with _lock:
        _runs[run_id]["status"] = "failed"
        _runs[run_id]["error"] = error


def get_run(run_id: str) -> dict | None:
    # Returns a shallow copy so a caller mutating the result can't corrupt
    # the registry's own state.
    with _lock:
        run = _runs.get(run_id)
        return dict(run) if run is not None else None
