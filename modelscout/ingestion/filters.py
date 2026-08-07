"""Keyword and param-size filtering, kept separate from hf_client.py's network
calls so this logic is unit-testable without hitting the HF Hub API.
"""

from __future__ import annotations

from modelscout.agents.schemas import InterestProfile
from modelscout.ingestion.hf_client import CandidateModel


def _searchable_text(model: CandidateModel) -> str:
    return " ".join([model.model_id, model.readme_text[:2000], " ".join(model.tags)]).lower()


def passes_filters(model: CandidateModel, profile: InterestProfile) -> bool:
    """True if the model should proceed to triage. Missing param-count data is
    NOT treated as a rejection -- see extract_param_count()'s docstring for why.
    """
    text = _searchable_text(model)

    if profile.keywords_exclude and any(kw.lower() in text for kw in profile.keywords_exclude):
        return False

    if profile.keywords_include and not any(kw.lower() in text for kw in profile.keywords_include):
        return False

    if profile.max_params_billion is not None and model.params_billion is not None:
        if model.params_billion > profile.max_params_billion:
            return False

    return True


def filter_candidates(models: list[CandidateModel], profile: InterestProfile) -> list[CandidateModel]:
    return [m for m in models if passes_filters(m, profile)]
