"""Regression test for a real bug found via live testing: index_model_readmes
treated `model_ids=[]` (0 candidates passed triage/filters -- a normal
outcome) the same as `model_ids=None` ("index everything"), because
`if model_ids:` is falsy for both. In a real run against a live dashboard
this silently re-embedded every model already in the table on every 0-result
run instead of doing nothing, observed as the "Run Pipeline" tab hanging for
minutes. Fixed with an explicit `is not None` check -- see index.py.

Same skip-if-unreachable pattern as test_persist_integration.py.
"""

from __future__ import annotations

import pytest

from modelscout.db import get_connection
from modelscout.rag.index import index_model_readmes

TEST_MODEL_ID = "test/rag-index-empty-list-regression-do-not-use-as-real-model-id"


def _db_available() -> bool:
    try:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable (DATABASE_URL)")


@pytest.fixture(autouse=True)
def _cleanup_test_row():
    yield
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM models WHERE model_id = %s", (TEST_MODEL_ID,))


def _seed_model_with_readme():
    from psycopg.types.json import Jsonb

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO models (model_id, pipeline_tag, downloads, likes, tags, card_data, readme_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (TEST_MODEL_ID, "test-tag", 1, 1, Jsonb(["test"]), Jsonb({}), "some readme content to chunk"),
        )


def _chunk_count(conn, model_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM model_card_chunks WHERE model_id = %s", (model_id,))
        return cur.fetchone()[0]


def test_empty_model_ids_list_indexes_nothing_not_everything():
    _seed_model_with_readme()

    with get_connection() as conn:
        total = index_model_readmes(conn, [])
        assert total == 0
        assert _chunk_count(conn, TEST_MODEL_ID) == 0
