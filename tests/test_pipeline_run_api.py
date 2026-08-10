"""Tests for the async /pipeline/run + /pipeline/runs/{id} fast-fail paths
only -- deliberately does NOT exercise a real scheduled run, since that
would call the real HF Hub and Anthropic APIs from the regular test suite.
No DB or network needed, so unlike test_api_endpoints.py this always runs.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from modelscout.api.main import app

client = TestClient(app)


def test_pipeline_run_404_for_unknown_profile_without_scheduling_anything():
    resp = client.post("/pipeline/run", json={"profile_name": "definitely-not-a-real-profile", "limit": 5})

    assert resp.status_code == 404


def test_pipeline_run_status_404_for_unknown_run_id():
    resp = client.get("/pipeline/runs/not-a-real-run-id")

    assert resp.status_code == 404
