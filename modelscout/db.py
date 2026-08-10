"""Postgres connection helper with pgvector adapter registration."""

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from modelscout.config import settings


@contextmanager
def get_connection():
    # connect_timeout matters more than it looks: without it, psycopg/libpq
    # falls back to the OS-level TCP timeout (30-60+ seconds, verified
    # empirically against an unreachable host during development) rather
    # than failing fast. That's bad in two places specifically -- Postgres
    # being unreachable mid-pipeline-run means EVERY observability write in
    # record_llm_call() would each hang that long before its own try/except
    # gives up (an "always swallowed, never raised" failure mode that isn't
    # actually graceful if it takes a minute to trigger), and a test-suite
    # DB-availability check that hangs 30+ seconds before skipping is a bad
    # experience for anyone running `pytest` without Docker up.
    conn = psycopg.connect(settings.database_url, autocommit=True, connect_timeout=5)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()
