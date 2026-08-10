import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, type PipelineRunResult, type PipelineRunState } from "../api";

const POLL_INTERVAL_MS = 2000;

export function RunPipelineView() {
  const [profileName, setProfileName] = useState("vlm_ocr");
  const [limit, setLimit] = useState(20);
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<PipelineRunState | null>(null);
  const [result, setResult] = useState<PipelineRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollHandle = useRef<number | undefined>(undefined);

  useEffect(() => {
    return () => window.clearInterval(pollHandle.current);
  }, []);

  function stopPolling() {
    window.clearInterval(pollHandle.current);
    pollHandle.current = undefined;
  }

  async function pollStatus(id: string) {
    try {
      const s = await api.getPipelineRunStatus(id);
      setStatus(s.status);
      if (s.status === "completed") {
        setResult(s.result);
        stopPolling();
      } else if (s.status === "failed") {
        setError(s.error ?? "Pipeline run failed");
        stopPolling();
      }
    } catch (err) {
      setError(String(err));
      stopPolling();
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    stopPolling();
    setError(null);
    setResult(null);
    setRunId(null);
    try {
      // POST /pipeline/run returns as soon as the run is scheduled (202) --
      // ingestion + triage + extraction + fact-checking can take minutes,
      // so the actual outcome comes from polling GET /pipeline/runs/{id}
      // rather than blocking this request.
      const { run_id } = await api.runPipeline(profileName, limit);
      setRunId(run_id);
      setStatus("pending");
      pollHandle.current = window.setInterval(() => pollStatus(run_id), POLL_INTERVAL_MS);
    } catch (err) {
      setError(String(err));
    }
  }

  const running = status === "pending" || status === "running";

  return (
    <div>
      <form onSubmit={handleSubmit} className="card" style={{ padding: 16, maxWidth: 480 }}>
        <div style={{ marginBottom: 12 }}>
          <label style={{ display: "block", marginBottom: 4, fontSize: 12 }} className="text-muted">
            Profile name
          </label>
          <input
            type="text"
            value={profileName}
            onChange={(e) => setProfileName(e.target.value)}
            style={{ width: "100%" }}
          />
        </div>
        <div style={{ marginBottom: 16 }}>
          <label style={{ display: "block", marginBottom: 4, fontSize: 12 }} className="text-muted">
            Limit per pipeline_tag
          </label>
          <input
            type="number"
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </div>
        <button type="submit" disabled={running}>
          {running
            ? `${status === "pending" ? "Scheduled" : "Running"}… (ingests, triages, extracts, and fact-checks — can take a while)`
            : "Run pipeline"}
        </button>
      </form>

      {runId && <p className="text-muted" style={{ fontSize: 12 }}>run_id: {runId}</p>}

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="card" style={{ padding: 16, marginTop: 16, maxWidth: 480 }}>
          <h4 style={{ marginTop: 0 }}>Run complete: {result.profile}</h4>
          <table>
            <tbody>
              <tr>
                <td>Ingested</td>
                <td>{result.counts.ingested}</td>
              </tr>
              <tr>
                <td>Passed triage (local, $0)</td>
                <td>{result.counts.triage_pass}</td>
              </tr>
              <tr>
                <td>Extracted via Claude</td>
                <td>{result.counts.extracted}</td>
              </tr>
              <tr>
                <td>Parse errors</td>
                <td>{result.counts.parse_errors}</td>
              </tr>
              <tr>
                <td>Fact-checked</td>
                <td>{result.counts.fact_checked}</td>
              </tr>
              <tr>
                <td>Flagged implausible</td>
                <td>{result.counts.implausible}</td>
              </tr>
            </tbody>
          </table>
          <p className="text-muted" style={{ marginBottom: 0 }}>
            Digest: {result.digest_markdown_path} — see the Models tab for the updated catalog.
          </p>
        </div>
      )}
    </div>
  );
}
