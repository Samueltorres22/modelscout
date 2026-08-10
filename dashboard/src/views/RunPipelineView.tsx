import { useState, type FormEvent } from "react";
import { api, type PipelineRunResult } from "../api";

export function RunPipelineView() {
  const [profileName, setProfileName] = useState("vlm_ocr");
  const [limit, setLimit] = useState(20);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<PipelineRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.runPipeline(profileName, limit);
      setResult(r);
    } catch (err) {
      setError(String(err));
    } finally {
      setRunning(false);
    }
  }

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
          {running ? "Running… (this ingests, triages, extracts, and fact-checks — can take a while)" : "Run pipeline"}
        </button>
      </form>

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
