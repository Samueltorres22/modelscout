import { useEffect, useState } from "react";
import { api, type AgentCallSummary } from "../api";

export function ObservabilityView() {
  const [agents, setAgents] = useState<AgentCallSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .observabilitySummary()
      .then((r) => setAgents(r.agents))
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error-box">{error}</div>;
  if (!agents) return <p className="text-muted">Loading…</p>;
  if (agents.length === 0) {
    return <p className="text-muted">No LLM calls recorded yet. Run the pipeline first.</p>;
  }

  const totalCalls = agents.reduce((s, a) => s + a.n_calls, 0);
  const totalIn = agents.reduce((s, a) => s + a.total_input_tokens, 0);
  const totalOut = agents.reduce((s, a) => s + a.total_output_tokens, 0);
  const anyCost = agents.some((a) => a.total_cost_usd !== null);
  const totalCost = agents.reduce((s, a) => s + (a.total_cost_usd ?? 0), 0);

  return (
    <div className="card">
      <table>
        <thead>
          <tr>
            <th>Agent</th>
            <th>Calls</th>
            <th>Input tokens</th>
            <th>Output tokens</th>
            <th>Avg latency</th>
            <th>Est. cost</th>
          </tr>
        </thead>
        <tbody>
          {agents.map((a) => (
            <tr key={a.agent_name}>
              <td>{a.agent_name}</td>
              <td>{a.n_calls}</td>
              <td>{a.total_input_tokens.toLocaleString()}</td>
              <td>{a.total_output_tokens.toLocaleString()}</td>
              <td>{(a.avg_latency_ms / 1000).toFixed(1)}s</td>
              <td>{a.total_cost_usd !== null ? `$${a.total_cost_usd.toFixed(4)}` : "n/a"}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td>
              <strong>Total</strong>
            </td>
            <td>
              <strong>{totalCalls}</strong>
            </td>
            <td>
              <strong>{totalIn.toLocaleString()}</strong>
            </td>
            <td>
              <strong>{totalOut.toLocaleString()}</strong>
            </td>
            <td></td>
            <td>
              <strong>{anyCost ? `$${totalCost.toFixed(4)}` : "n/a"}</strong>
            </td>
          </tr>
        </tfoot>
      </table>
      {!anyCost && (
        <p className="text-muted" style={{ padding: 12 }}>
          Set PRICE_PER_MTOK_INPUT / PRICE_PER_MTOK_OUTPUT in .env to see cost estimates.
        </p>
      )}
    </div>
  );
}
