"""Postgres connection helper with pgvector adapter registration."""

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from modelscout.config import settings


@contextmanager
def get_connection():
    conn = psycopg.connect(settings.database_url, autocommit=True)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()
