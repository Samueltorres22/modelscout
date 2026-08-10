import { Fragment, useEffect, useState } from "react";
import { api, type ModelDetail, type ModelSummary } from "../api";
import { VerdictPill } from "../components/VerdictPill";

function formatNumber(n: number | null): string {
  if (n === null) return "—";
  return n.toLocaleString();
}

function BenchmarkTable({ benchmarks }: { benchmarks: ModelDetail["declared_benchmarks"] }) {
  if (benchmarks.length === 0) {
    return <p className="text-muted">No declared benchmarks.</p>;
  }
  return (
    <table style={{ marginTop: 8 }}>
      <thead>
        <tr>
          <th>Benchmark</th>
          <th>Metric</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        {benchmarks.map((b, i) => (
          <tr key={i}>
            <td>{b.name}</td>
            <td className="text-muted">{b.metric ?? "—"}</td>
            <td>{b.score ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ModelDetailPanel({ modelId }: { modelId: string }) {
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError(null);
    api
      .getModel(modelId)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  if (error) return <div className="error-box">{error}</div>;
  if (!detail) return <p className="text-muted">Loading…</p>;

  return (
    <div style={{ padding: "16px 12px" }}>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 16 }}>
        <div>
          <div className="text-muted" style={{ fontSize: 12 }}>
            License
          </div>
          <div>{detail.license ?? "unknown"}</div>
        </div>
        <div>
          <div className="text-muted" style={{ fontSize: 12 }}>
            Architecture
          </div>
          <div>{detail.architecture_family ?? "unknown"}</div>
        </div>
        <div>
          <div className="text-muted" style={{ fontSize: 12 }}>
            Hardware
          </div>
          <div>{detail.hardware_requirements ?? "unknown"}</div>
        </div>
        <div>
          <div className="text-muted" style={{ fontSize: 12 }}>
            Quantization
          </div>
          <div>{detail.quantization_available.join(", ") || "none stated"}</div>
        </div>
      </div>

      <h4 style={{ marginBottom: 4 }}>Declared benchmarks</h4>
      {detail.extraction_parse_error && (
        <p style={{ color: "var(--verdict-implausible-text)" }}>
          ⚠ Extraction failed to parse cleanly for this model.
        </p>
      )}
      <BenchmarkTable benchmarks={detail.declared_benchmarks} />

      {detail.fact_check_verdict && (
        <>
          <h4 style={{ marginTop: 20, marginBottom: 4 }}>
            Fact-check: <VerdictPill verdict={detail.fact_check_verdict} /> (confidence{" "}
            {detail.fact_check_confidence?.toFixed(2)})
          </h4>
          {detail.fact_check_parse_error ? (
            <p style={{ color: "var(--verdict-implausible-text)" }}>
              ⚠ Fact-check could not be completed automatically for this model.
            </p>
          ) : (
            <>
              <p>{detail.fact_check_reasoning}</p>
              {detail.fact_check_flags.length > 0 && (
                <>
                  <div className="text-muted" style={{ fontSize: 12, marginTop: 8 }}>
                    Flags
                  </div>
                  <ul>
                    {detail.fact_check_flags.map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </>
              )}
              {detail.fact_check_consistency_issues.length > 0 && (
                <>
                  <div className="text-muted" style={{ fontSize: 12, marginTop: 8 }}>
                    Consistency issues
                  </div>
                  <ul>
                    {detail.fact_check_consistency_issues.map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </>
      )}

      <div style={{ marginTop: 16 }}>
        <a href={detail.hf_url} target="_blank" rel="noreferrer">
          View on Hugging Face ↗
        </a>
      </div>
    </div>
  );
}

export function ModelsView() {
  const [models, setModels] = useState<ModelSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    api
      .listModels()
      .then((r) => setModels(r.models))
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error-box">Failed to load models: {error}</div>;
  if (!models) return <p className="text-muted">Loading…</p>;
  if (models.length === 0) {
    return (
      <p className="text-muted">
        No models ingested yet. Run the pipeline from the "Run Pipeline" tab, or via{" "}
        <code>python cli.py run --profile vlm_ocr</code>.
      </p>
    );
  }

  return (
    <div className="card">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Downloads</th>
            <th>Triage</th>
            <th>Params</th>
            <th>License</th>
            <th>Benchmarks</th>
            <th>Fact-check</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <Fragment key={m.model_id}>
              <tr
                onClick={() => setExpanded(expanded === m.model_id ? null : m.model_id)}
                style={{ cursor: "pointer" }}
              >
                <td>
                  <strong>{m.model_id}</strong>
                  {m.matched_profile && (
                    <div className="text-muted" style={{ fontSize: 12 }}>
                      profile: {m.matched_profile}
                    </div>
                  )}
                </td>
                <td>{formatNumber(m.downloads)}</td>
                <td>
                  {m.is_relevant === null
                    ? "—"
                    : `${m.is_relevant ? "✓ relevant" : "skipped"} (${m.triage_confidence?.toFixed(2)})`}
                </td>
                <td>{m.params_billion !== null ? `${m.params_billion}B` : "unknown"}</td>
                <td>{m.license ?? "unknown"}</td>
                <td>
                  {m.n_benchmarks}
                  {m.extraction_parse_error && (
                    <span title="Extraction parse error" style={{ marginLeft: 4 }}>
                      ⚠
                    </span>
                  )}
                </td>
                <td>
                  <VerdictPill verdict={m.fact_check_verdict} />
                </td>
              </tr>
              {expanded === m.model_id && (
                <tr>
                  <td colSpan={7} style={{ background: "var(--bg)" }}>
                    <ModelDetailPanel modelId={m.model_id} />
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
