"""Unit tests for interest-profile filtering. No network calls."""

from modelscout.agents.schemas import InterestProfile, TriageConfig
from modelscout.ingestion.filters import passes_filters
from modelscout.ingestion.hf_client import CandidateModel


def _profile(**overrides) -> InterestProfile:
    base = dict(
        name="test",
        pipeline_tags=["image-text-to-text"],
        keywords_include=["ocr"],
        keywords_exclude=["audio"],
        max_params_billion=5.0,
        triage=TriageConfig(candidate_labels=["a", "b"]),
    )
    base.update(overrides)
    return InterestProfile.model_validate(base)


def _model(**overrides) -> CandidateModel:
    base = dict(
        model_id="org/some-ocr-model-2b",
        pipeline_tag="image-text-to-text",
        downloads=1000,
        likes=10,
        last_modified=None,
        created_at=None,
        tags=["vision"],
        card_data={},
        params_billion=2.0,
        readme_text="This model does OCR on documents.",
    )
    base.update(overrides)
    return CandidateModel(**base)


def test_passes_when_keyword_present_and_under_size_cap():
    assert passes_filters(_model(), _profile()) is True


def test_rejects_when_include_keyword_missing():
    m = _model(
        model_id="org/generic-vision-model",
        readme_text="A generic language model with no relevant capability.",
    )
    assert passes_filters(m, _profile()) is False


def test_rejects_when_exclude_keyword_present():
    m = _model(readme_text="This model does OCR and also audio transcription.")
    assert passes_filters(m, _profile()) is False


def test_rejects_when_over_param_cap():
    m = _model(params_billion=12.0)
    assert passes_filters(m, _profile()) is False


def test_passes_when_param_count_unknown():
    # Missing size data must NOT be treated as a rejection -- see filters.py docstring.
    m = _model(params_billion=None)
    assert passes_filters(m, _profile()) is True
