"""Unauthenticated Hugging Face Hub API client: list candidate models for an
interest profile and fetch their README (model card) text.

Deliberately unauthenticated -- no HF_TOKEN required. Rate-limited but
sufficient for a 20-30 model demo run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from modelscout.agents.schemas import InterestProfile

logger = logging.getLogger(__name__)

_api = HfApi()

# Best-effort size-hint pattern for model ids/tags like "Qwen2-VL-2B-Instruct"
# or "phi-3.5-mini-3.8b". Used only when HF's safetensors metadata is absent --
# see extract_param_count().
_SIZE_HINT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:illion)?(?:[-_]|$)")


@dataclass
class CandidateModel:
    model_id: str
    pipeline_tag: str | None
    downloads: int
    likes: int
    last_modified: datetime | None
    created_at: datetime | None
    tags: list[str]
    card_data: dict
    params_billion: float | None
    readme_text: str


def extract_param_count(model_id: str, tags: list[str], safetensors_total: int | None) -> float | None:
    """Best-effort param count: safetensors metadata -> regex size hint -> None.

    Never hard-fails -- a model with no discoverable size just gets
    params_billion=None and is left to triage rather than silently dropped.
    """
    if safetensors_total:
        return round(safetensors_total / 1e9, 3)

    for candidate in [model_id, *tags]:
        m = _SIZE_HINT_RE.search(candidate)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def fetch_readme(model_id: str) -> str:
    try:
        path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (EntryNotFoundError, RepositoryNotFoundError):
        return ""
    except Exception as exc:  # noqa: BLE001 -- ingestion must never crash on one bad repo
        logger.warning("Could not fetch README for %s: %s", model_id, exc)
        return ""


def search_candidate_models(profile: InterestProfile, limit_per_tag: int = 20) -> list[CandidateModel]:
    """Pull candidate models for every pipeline_tag in the profile, deduped by
    model_id, filtered by lookback window and min downloads/likes. Does NOT
    apply keyword or param-size filtering -- see ingestion/filters.py for that
    (kept separate so filtering logic is unit-testable without network calls).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=profile.lookback_days)
    seen: dict[str, CandidateModel] = {}

    tags = profile.pipeline_tags or [None]
    for tag in tags:
        try:
            results = _api.list_models(
                pipeline_tag=tag,
                sort="trending_score",
                limit=limit_per_tag,
                cardData=True,
                full=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("HF list_models failed for pipeline_tag=%s: %s", tag, exc)
            continue

        for info in results:
            if info.id in seen:
                continue
            if info.downloads is not None and info.downloads < profile.min_downloads:
                continue
            if info.likes is not None and info.likes < profile.min_likes:
                continue

            created_at = getattr(info, "created_at", None)
            last_modified = getattr(info, "last_modified", None)
            reference_date = created_at or last_modified
            if reference_date and reference_date < cutoff:
                continue

            safetensors_total = None
            if getattr(info, "safetensors", None) is not None:
                safetensors_total = getattr(info.safetensors, "total", None)

            params_billion = extract_param_count(info.id, info.tags or [], safetensors_total)
            readme_text = fetch_readme(info.id)
            card_data = info.card_data.to_dict() if info.card_data else {}

            seen[info.id] = CandidateModel(
                model_id=info.id,
                pipeline_tag=info.pipeline_tag,
                downloads=info.downloads or 0,
                likes=info.likes or 0,
                last_modified=last_modified,
                created_at=created_at,
                tags=info.tags or [],
                card_data=card_data,
                params_billion=params_billion,
                readme_text=readme_text,
            )

    return list(seen.values())
