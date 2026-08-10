"""Pure unit tests for the in-memory async-run registry (modelscout/api/
run_registry.py) -- no DB, no network, always runs regardless of Docker.
"""

from __future__ import annotations

from modelscout.api import run_registry


def test_create_run_starts_pending_with_no_result_or_error():
    run_id = run_registry.create_run()

    run = run_registry.get_run(run_id)

    assert run["status"] == "pending"
    assert run["result"] is None
    assert run["error"] is None


def test_create_run_returns_unique_ids():
    assert run_registry.create_run() != run_registry.create_run()


def test_mark_running_then_completed_stores_result():
    run_id = run_registry.create_run()

    run_registry.mark_running(run_id)
    assert run_registry.get_run(run_id)["status"] == "running"

    run_registry.mark_completed(run_id, {"profile": "test_profile", "counts": {"ingested": 3}})

    run = run_registry.get_run(run_id)
    assert run["status"] == "completed"
    assert run["result"] == {"profile": "test_profile", "counts": {"ingested": 3}}
    assert run["error"] is None


def test_mark_failed_stores_error_not_result():
    run_id = run_registry.create_run()

    run_registry.mark_failed(run_id, "boom")

    run = run_registry.get_run(run_id)
    assert run["status"] == "failed"
    assert run["error"] == "boom"
    assert run["result"] is None


def test_get_run_returns_none_for_unknown_id():
    assert run_registry.get_run("does-not-exist") is None


def test_get_run_returns_a_defensive_copy():
    run_id = run_registry.create_run()

    run = run_registry.get_run(run_id)
    run["status"] = "mutated"

    assert run_registry.get_run(run_id)["status"] == "pending"
