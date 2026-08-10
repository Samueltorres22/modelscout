from datetime import datetime

from pydantic import BaseModel


class RunPipelineRequest(BaseModel):
    profile_name: str
    limit: int = 20


class PipelineRunAccepted(BaseModel):
    """202 response for POST /pipeline/run -- the run has been scheduled,
    not completed. Poll GET /pipeline/runs/{run_id} for its outcome.
    """

    run_id: str
    status: str


class PipelineRunStatus(BaseModel):
    """GET /pipeline/runs/{run_id}. result is populated once status is
    "completed" (same shape pipeline.run() has always returned); error is
    populated once status is "failed".
    """

    run_id: str
    status: str
    result: dict | None = None
    error: str | None = None


class SearchResultItem(BaseModel):
    model_id: str
    chunk_text: str
    score: float
    pipeline_tag: str | None
    downloads: int | None
    hf_url: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class ModelSummary(BaseModel):
    """One row of GET /models -- the catalog table. Deliberately excludes
    readme_text and the full benchmark/flag lists (see ModelDetail) so the
    list payload stays small regardless of how many models are ingested.
    """

    model_id: str
    pipeline_tag: str | None
    downloads: int | None
    likes: int | None
    matched_profile: str | None
    ingested_at: datetime
    is_relevant: bool | None
    triage_confidence: float | None
    params_billion: float | None
    license: str | None
    architecture_family: str | None
    extraction_parse_error: bool | None
    n_benchmarks: int
    fact_check_verdict: str | None
    fact_check_confidence: float | None


class ModelsResponse(BaseModel):
    models: list[ModelSummary]


class BenchmarkItem(BaseModel):
    name: str
    metric: str | None = None
    score: float | str | None = None


class ModelDetail(BaseModel):
    """GET /models/{model_id} -- everything ModelSummary omits: the full
    benchmark list and the Fact-Checker's actual reasoning/flags.
    """

    model_id: str
    pipeline_tag: str | None
    downloads: int | None
    likes: int | None
    tags: list[str] = []
    matched_profile: str | None
    ingested_at: datetime
    hf_url: str

    is_relevant: bool | None
    triage_confidence: float | None

    params_billion: float | None
    license: str | None
    architecture_family: str | None
    hardware_requirements: str | None
    quantization_available: list[str] = []
    declared_benchmarks: list[BenchmarkItem] = []
    extraction_parse_error: bool | None

    fact_check_verdict: str | None
    fact_check_confidence: float | None
    fact_check_flags: list[str] = []
    fact_check_consistency_issues: list[str] = []
    fact_check_reasoning: str | None
    fact_check_parse_error: bool | None


class DigestRun(BaseModel):
    id: int
    profile_name: str
    run_at: datetime
    n_ingested: int | None
    n_triage_pass: int | None
    n_extracted: int | None
    n_parse_errors: int | None
    n_fact_checked: int | None
    n_implausible: int | None
    digest_markdown_path: str | None


class RunsResponse(BaseModel):
    runs: list[DigestRun]


class AgentCallSummary(BaseModel):
    agent_name: str
    n_calls: int
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_ms: float
    total_cost_usd: float | None


class ObservabilitySummary(BaseModel):
    agents: list[AgentCallSummary]
