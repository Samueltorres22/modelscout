// Typed client for the ModelScout API. One function per endpoint, mirroring
// modelscout/api/schemas.py field-for-field -- kept in sync by hand since
// this project doesn't generate an OpenAPI client (small enough surface
// that codegen tooling would be more machinery than the problem needs).

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface ModelSummary {
  model_id: string;
  pipeline_tag: string | null;
  downloads: number | null;
  likes: number | null;
  matched_profile: string | null;
  ingested_at: string;
  is_relevant: boolean | null;
  triage_confidence: number | null;
  params_billion: number | null;
  license: string | null;
  architecture_family: string | null;
  extraction_parse_error: boolean | null;
  n_benchmarks: number;
  fact_check_verdict: string | null;
  fact_check_confidence: number | null;
}

export interface BenchmarkItem {
  name: string;
  metric: string | null;
  score: number | string | null;
}

export interface ModelDetail {
  model_id: string;
  pipeline_tag: string | null;
  downloads: number | null;
  likes: number | null;
  tags: string[];
  matched_profile: string | null;
  ingested_at: string;
  hf_url: string;
  is_relevant: boolean | null;
  triage_confidence: number | null;
  params_billion: number | null;
  license: string | null;
  architecture_family: string | null;
  hardware_requirements: string | null;
  quantization_available: string[];
  declared_benchmarks: BenchmarkItem[];
  extraction_parse_error: boolean | null;
  fact_check_verdict: string | null;
  fact_check_confidence: number | null;
  fact_check_flags: string[];
  fact_check_consistency_issues: string[];
  fact_check_reasoning: string | null;
  fact_check_parse_error: boolean | null;
}

export interface DigestRun {
  id: number;
  profile_name: string;
  run_at: string;
  n_ingested: number | null;
  n_triage_pass: number | null;
  n_extracted: number | null;
  n_parse_errors: number | null;
  n_fact_checked: number | null;
  n_implausible: number | null;
  digest_markdown_path: string | null;
}

export interface AgentCallSummary {
  agent_name: string;
  n_calls: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_latency_ms: number;
  total_cost_usd: number | null;
}

export interface SearchResultItem {
  model_id: string;
  chunk_text: string;
  score: number;
  pipeline_tag: string | null;
  downloads: number | null;
  hf_url: string;
}

export interface PipelineRunCounts {
  ingested: number;
  triage_pass: number;
  extracted: number;
  parse_errors: number;
  fact_checked: number;
  implausible: number;
}

export interface PipelineRunResult {
  profile: string;
  counts: PipelineRunCounts;
  digest_markdown_path: string;
  top_models: unknown[];
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => apiFetch<{ db_ok: boolean; embedding_model_ok: boolean }>("/health"),

  listModels: () => apiFetch<{ models: ModelSummary[] }>("/models"),

  getModel: (modelId: string) => apiFetch<ModelDetail>(`/models/${modelId}`),

  listRuns: () => apiFetch<{ runs: DigestRun[] }>("/runs"),

  observabilitySummary: () =>
    apiFetch<{ agents: AgentCallSummary[] }>("/observability/summary"),

  search: (query: string, k = 5) =>
    apiFetch<{ query: string; results: SearchResultItem[] }>(
      `/search?q=${encodeURIComponent(query)}&k=${k}`,
    ),

  runPipeline: (profileName: string, limit: number) =>
    apiFetch<PipelineRunResult>("/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile_name: profileName, limit }),
    }),
};
