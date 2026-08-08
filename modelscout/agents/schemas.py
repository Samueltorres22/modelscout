"""Pydantic schemas shared across the ingestion/RAG/agent pipeline."""

from pydantic import BaseModel, Field


class TriageConfig(BaseModel):
    candidate_labels: list[str]
    relevance_threshold: float = 0.6


class NotifyConfig(BaseModel):
    top_n: int = 10
    output_dir: str = "digests"


class InterestProfile(BaseModel):
    """A configured area of interest that ingestion/triage filter models against."""

    name: str
    description: str = ""

    pipeline_tags: list[str] = Field(default_factory=list)
    keywords_include: list[str] = Field(default_factory=list)
    keywords_exclude: list[str] = Field(default_factory=list)
    max_params_billion: float | None = None
    min_downloads: int = 0
    min_likes: int = 0
    lookback_days: int = 365

    triage: TriageConfig
    notify: NotifyConfig = Field(default_factory=NotifyConfig)


class BenchmarkResult(BaseModel):
    name: str
    metric: str | None = None
    score: float | str | None = None  # some cards report "82.3%" as a string


class ExtractedModelSpecs(BaseModel):
    """Structured output of the Extractor Agent's forced tool-use call."""

    model_id: str
    params_billion: float | None = None
    license: str | None = None
    architecture_family: str | None = None
    hardware_requirements: str | None = None
    quantization_available: list[str] = Field(default_factory=list)
    declared_benchmarks: list[BenchmarkResult] = Field(default_factory=list)

    parse_error: bool = False
    parse_error_detail: str | None = None
    raw_model_response: dict | str | None = None


class TriageResult(BaseModel):
    model_id: str
    profile_name: str
    is_relevant: bool
    confidence: float
    raw_labels: dict


class FactCheckResult(BaseModel):
    """Structured output of the Fact-Checker Agent's forced tool-use call.

    This is an LLM-as-judge plausibility/consistency read of the Extractor's
    declared_benchmarks against the card's own stated params/architecture --
    NOT a re-run of the actual benchmarks (infeasible for arbitrary
    multi-billion-parameter HF models in this project's scope). Same kind of
    judgment a human reviewer makes skimming a paper's claims for red flags
    without independently reproducing every experiment.
    """

    model_id: str
    verdict: str = "plausible"  # plausible | questionable | implausible
    confidence: float = 0.0
    flags: list[str] = Field(default_factory=list)
    consistency_issues: list[str] = Field(default_factory=list)
    reasoning: str = ""

    parse_error: bool = False
    parse_error_detail: str | None = None
    raw_model_response: dict | str | None = None
