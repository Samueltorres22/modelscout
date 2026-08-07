"""Upsert ingested candidate models into the `models` table. Idempotent --
re-running the same profile updates existing rows instead of duplicating them.
"""

from __future__ import annotations

import json

from psycopg.types.json import Jsonb

from modelscout.ingestion.hf_client import CandidateModel

_UPSERT_SQL = """
INSERT INTO models (
    model_id, pipeline_tag, downloads, likes, last_modified, tags,
    card_data, readme_text, matched_profile
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (model_id) DO UPDATE SET
    pipeline_tag = EXCLUDED.pipeline_tag,
    downloads = EXCLUDED.downloads,
    likes = EXCLUDED.likes,
    last_modified = EXCLUDED.last_modified,
    tags = EXCLUDED.tags,
    card_data = EXCLUDED.card_data,
    readme_text = EXCLUDED.readme_text,
    matched_profile = EXCLUDED.matched_profile,
    ingested_at = now()
"""


def _safe_jsonb(value: dict) -> Jsonb:
    """card_data can contain nested non-primitive objects (e.g. EvalResult
    dataclasses) that psycopg's default JSON encoder can't serialize --
    round-trip through json.dumps(default=str) first so this never raises.
    """
    return Jsonb(json.loads(json.dumps(value, default=str)))


def upsert_models(conn, models: list[CandidateModel], profile_name: str) -> int:
    with conn.cursor() as cur:
        for m in models:
            cur.execute(
                _UPSERT_SQL,
                (
                    m.model_id,
                    m.pipeline_tag,
                    m.downloads,
                    m.likes,
                    m.last_modified,
                    _safe_jsonb(m.tags),
                    _safe_jsonb(m.card_data),
                    m.readme_text,
                    profile_name,
                ),
            )
    return len(models)
