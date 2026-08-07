"""Markdown-header-aware chunking for model card READMEs.

Splits on markdown headers first (so a chunk doesn't straddle unrelated
sections like "License" and "Benchmarks"), then further splits any section
that's still too long by paragraph, accumulating paragraphs up to
max_chars per chunk.
"""

from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def _split_by_headers(text: str) -> list[str]:
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [text]

    sections = []
    if matches[0].start() > 0:
        sections.append(text[: matches[0].start()])

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[m.start():end])

    return sections


def _split_long_section(section: str, max_chars: int) -> list[str]:
    if len(section) <= max_chars:
        return [section]

    paragraphs = [p for p in re.split(r"\n\s*\n", section) if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > max_chars:
            chunks.append(current.strip())
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current.strip():
        chunks.append(current.strip())

    # A single huge paragraph with no blank lines still needs a hard split.
    final: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i : i + max_chars])
    return final


def chunk_readme(readme_text: str, max_chars: int = 1200, min_chars: int = 30) -> list[str]:
    if not readme_text or not readme_text.strip():
        return []

    chunks: list[str] = []
    for section in _split_by_headers(readme_text):
        chunks.extend(_split_long_section(section, max_chars))

    return [c.strip() for c in chunks if len(c.strip()) >= min_chars]
