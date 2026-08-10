"""Integration tests for ingestion/persist.py -- the only test file that
actually touches Postgres, unlike test_filters.py/test_parser.py/
test_fact_checker_parser.py which exercise pure logic against synthetic
inputs. Filters/parsers already prove the transformation logic is right;
this proves the data actually lands in the DB correctly, and idempotently,
which nothing else in the suite checks.

Requires DATABASE_URL to point at a live Postgres matching db/schema.sql
(the models table specifically). Skips automatically rather than failing
if it isn't reachable -- so `pytest` still works for anyone running just
the unit tests locally without `docker compose up`. In CI, tests.yml runs
a real pgvector/pgvector:pg16 service container with the schema applied so
this suite actually runs there instead of always skipping.

Uses a model_id under a "test/" namespace and always cleans up in a
finally block, specifically so this is safe to run against the same
Postgres instance used for real demo data (verified: won't collide with
or leave behind rows among the real ingested models).
"""

from __future__ import annotations

import pytest

from modelscout.db import get_connection
from modelscout.ingestion.hf_client import CandidateModel
from modelscout.ingestion.persist import upsert_models

TEST_MODEL_ID = "test/idempotency-check-do-not-use-as-real-model-id"


def _db_available() -> bool:
    try:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable (DATABASE_URL)")


def _model(**overrides) -> CandidateModel:
    base = {
        "model_id": TEST_MODEL_ID,
        "pipeline_tag": "test-tag",
        "downloads": 1,
        "likes": 1,
        "last_modified": None,
        "created_at": None,
        "tags": ["test"],
        "card_data": {"license": "mit"},
        "params_billion": 1.0,
        "readme_text": "test readme for integration test",
    }
    base.update(overrides)
    return CandidateModel(**base)


@pytest.fixture(autouse=True)
def _cleanup_test_row():
    yield
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM models WHERE model_id = %s", (TEST_MODEL_ID,))


def _count_test_rows(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM models WHERE model_id = %s", (TEST_MODEL_ID,))
        return cur.fetchone()[0]


def test_upsert_inserts_a_new_row():
    with get_connection() as conn:
        upsert_models(conn, [_model()], "test_profile")
        assert _count_test_rows(conn) == 1


def test_upsert_is_idempotent_not_duplicated():
    model = _model()
    with get_connection() as conn:
        upsert_models(conn, [model], "test_profile")
        upsert_models(conn, [model], "test_profile")
        upsert_models(conn, [model], "test_profile")
        assert _count_test_rows(conn) == 1


def test_upsert_updates_existing_row_on_reingest():
    with get_connection() as conn:
        upsert_models(conn, [_model(downloads=100)], "test_profile")
        upsert_models(conn, [_model(downloads=999)], "test_profile")

        with conn.cursor() as cur:
            cur.execute("SELECT downloads FROM models WHERE model_id = %s", (TEST_MODEL_ID,))
            downloads = cur.fetchone()[0]

    assert downloads == 999


def test_upsert_handles_nested_non_primitive_card_data():
    # card_data can contain nested dataclass-like objects (e.g. EvalResult)
    # that json.dumps can't serialize directly -- _safe_jsonb's
    # default=str fallback is what's actually being verified here.
    class _NotJsonSerializable:
        def __str__(self):
            return "custom-repr"

    with get_connection() as conn:
        upsert_models(
            conn,
            [_model(card_data={"nested": _NotJsonSerializable()})],
            "test_profile",
        )

        with conn.cursor() as cur:
            cur.execute("SELECT card_data FROM models WHERE model_id = %s", (TEST_MODEL_ID,))
            card_data = cur.fetchone()[0]

    assert card_data["nested"] == "custom-repr"
