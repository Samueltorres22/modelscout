"""Notifier: pure Python, no LLM call. Ranks accumulated per-model results and
writes a ranked Markdown digest. Runs once, only after every parallel
triage/extract branch has converged (see graph.py) -- LangGraph's superstep
execution is what guarantees `results` is fully populated by the time this
node fires.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from modelscout.agents.state import PipelineState
from modelscout.config import PROJECT_ROOT


def _rank_key(result: dict) -> tuple:
    # Primary: triage confidence (how well it matches the interest profile).
    # Tiebreak: HF downloads (a cheap popularity/maturity signal).
    return (-result.get("triage_confidence", 0.0), -result.get("downloads", 0))


def _format_entry(result: dict) -> str:
    model_id = result["model_id"]
    lines = [f"### [{model_id}](https://huggingface.co/{model_id})"]
    lines.append(
        f"Triage confidence: {result.get('triage_confidence', 0):.2f} · "
        f"Downloads: {result.get('downloads', 0):,}"
    )

    extracted = result.get("extracted")
    if extracted is None:
        lines.append("_Did not pass triage relevance threshold._")
        return "\n".join(lines)

    if extracted.get("parse_error"):
        lines.append(
            f"⚠️ **Extraction failed** ({extracted.get('parse_error_detail', 'unknown reason')}) "
            "-- needs manual review."
        )
        return "\n".join(lines)

    params = extracted.get("params_billion")
    lines.append(
        f"- Params: {params}B" if params is not None else "- Params: unknown"
    )
    lines.append(f"- License: {extracted.get('license') or 'unknown'}")
    lines.append(f"- Architecture: {extracted.get('architecture_family') or 'unknown'}")
    lines.append(f"- Hardware: {extracted.get('hardware_requirements') or 'unknown'}")

    quant = extracted.get("quantization_available") or []
    if quant:
        lines.append(f"- Quantization: {', '.join(quant)}")

    benchmarks = extracted.get("declared_benchmarks") or []
    if benchmarks:
        bench_str = "; ".join(
            f"{b['name']}: {b.get('score', '?')}" + (f" ({b['metric']})" if b.get("metric") else "")
            for b in benchmarks
        )
        lines.append(f"- Declared benchmarks: {bench_str}")

    return "\n".join(lines)


def build_digest(state: PipelineState) -> str:
    profile = state["interest_profile"]
    results = sorted(state["results"], key=_rank_key)
    top_n = profile.get("notify", {}).get("top_n", 10)
    top_results = results[:top_n]

    n_ingested = len(state["candidates"])
    n_triage_pass = sum(1 for r in results if r.get("is_relevant"))
    n_extracted = sum(1 for r in results if r.get("extracted") is not None)
    n_parse_errors = sum(
        1 for r in results if r.get("extracted") and r["extracted"].get("parse_error")
    )

    lines = [
        f"# ModelScout Digest — {profile['name']}",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        f"**{n_ingested}** candidates ingested → **{n_triage_pass}** passed triage "
        f"(screened locally, $0 API cost) → **{n_extracted}** sent to Claude for extraction "
        f"(**{n_parse_errors}** parse error(s)).",
        "",
        "---",
        "",
    ]

    for result in top_results:
        lines.append(_format_entry(result))
        lines.append("")

    return "\n".join(lines)


def notifier_node(state: PipelineState) -> dict:
    digest_markdown = build_digest(state)

    profile = state["interest_profile"]
    output_dir = PROJECT_ROOT / profile.get("notify", {}).get("output_dir", "digests")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = output_dir / f"{date_str}-{profile['name']}.md"
    path.write_text(digest_markdown, encoding="utf-8")

    return {"digest_markdown": digest_markdown}
