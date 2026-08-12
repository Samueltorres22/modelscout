#!/usr/bin/env python
"""One-time copy of local demo data into a Supabase Postgres instance (see
README's Deploy section, step 2). Python + psycopg rather than `pg_dump`/
`psql` deliberately -- those binaries aren't guaranteed to be on a dev
machine's PATH (they weren't on the one this project was built on), while
psycopg is already a project dependency.

Copies models -> {model_card_chunks, triage_results, extracted_specs,
fact_checks} -> {llm_calls, digest_runs}, in that order for the tables with
a models(model_id) foreign key. Auto-increment `id` columns are NOT copied
-- the target regenerates its own, since nothing downstream depends on the
id values matching between source and target, only on the model_id FKs
(which model_card_chunks/triage_results/extracted_specs/fact_checks all use
instead of id).

    python scripts/migrate_to_supabase.py --target-url "postgresql://postgres.xxx:PASSWORD@aws-...pooler.supabase.com:6543/postgres"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from modelscout.config import settings

# (table, columns to copy, JSONB column names within that list, conflict
# target for ON CONFLICT DO NOTHING or None). JSONB columns need explicit
# wrapping on insert -- psycopg happily decodes jsonb into a plain dict/list
# on SELECT, but won't re-encode a bare dict/list back to jsonb on its own.
_TABLES: list[tuple[str, list[str], set[str], str | None]] = [
    (
        "models",
        [
            "model_id",
            "pipeline_tag",
            "downloads",
            "likes",
            "last_modified",
            "tags",
            "card_data",
            "readme_text",
            "matched_profile",
            "ingested_at",
        ],
        {"tags", "card_data"},
        "model_id",
    ),
    (
        "model_card_chunks",
        ["model_id", "chunk_index", "chunk_text", "embedding", "created_at"],
        set(),
        "model_id, chunk_index",
    ),
    (
        "triage_results",
        ["model_id", "profile_name", "is_relevant", "confidence", "raw_labels", "created_at"],
        {"raw_labels"},
        None,
    ),
    (
        "extracted_specs",
        [
            "model_id",
            "params_billion",
            "license",
            "architecture_family",
            "hardware_requirements",
            "quantization_available",
            "declared_benchmarks",
            "parse_error",
            "parse_error_detail",
            "raw_model_response",
            "extracted_at",
        ],
        {"quantization_available", "declared_benchmarks", "raw_model_response"},
        None,
    ),
    (
        "fact_checks",
        [
            "model_id",
            "verdict",
            "confidence",
            "flags",
            "consistency_issues",
            "reasoning",
            "parse_error",
            "parse_error_detail",
            "raw_model_response",
            "checked_at",
        ],
        {"flags", "consistency_issues", "raw_model_response"},
        None,
    ),
    (
        "llm_calls",
        [
            "agent_name",
            "model_id",
            "model",
            "input_tokens",
            "output_tokens",
            "latency_ms",
            "estimated_cost_usd",
            "prompt_version",
            "called_at",
        ],
        set(),
        None,
    ),
    (
        "digest_runs",
        [
            "profile_name",
            "run_at",
            "n_ingested",
            "n_triage_pass",
            "n_extracted",
            "n_parse_errors",
            "n_fact_checked",
            "n_implausible",
            "digest_markdown_path",
        ],
        set(),
        None,
    ),
]


def _connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=True, connect_timeout=10)
    register_vector(conn)
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy local demo data into a Supabase Postgres instance")
    parser.add_argument("--target-url", required=True, help="Supabase connection string (Transaction pooler)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if the target's models table already has rows (default: abort to avoid duplicates)",
    )
    args = parser.parse_args()

    source = _connect(settings.database_url)
    target = _connect(args.target_url)

    with target.cursor() as cur:
        cur.execute("SELECT count(*) FROM models")
        existing = cur.fetchone()[0]
    if existing and not args.force:
        print(f"[ABORT] Target already has {existing} row(s) in models -- pass --force to copy anyway.")
        print("        (Tables without a unique constraint on the copied columns, e.g. triage_results,")
        print("         would get duplicated on a second run.)")
        return 1

    total = 0
    for table, columns, json_columns, conflict_target in _TABLES:
        col_list = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        on_conflict = f" ON CONFLICT ({conflict_target}) DO NOTHING" if conflict_target else ""
        insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}){on_conflict}"
        json_indices = {i for i, c in enumerate(columns) if c in json_columns}

        with source.cursor() as cur:
            cur.execute(f"SELECT {col_list} FROM {table}")
            rows = cur.fetchall()

        with target.cursor() as cur:
            for row in rows:
                row = tuple(Jsonb(v) if i in json_indices and v is not None else v for i, v in enumerate(row))
                cur.execute(insert_sql, row)

        print(f"[OK] {table}: copied {len(rows)} row(s)")
        total += len(rows)

    source.close()
    target.close()
    print(f"\nDone -- {total} row(s) copied across {len(_TABLES)} tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
