import { useState, type FormEvent } from "react";
import { api, type SearchResultItem } from "../api";

export function SearchView() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResultItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await api.search(query, 5);
      setResults(r.results);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. lightweight vision model for document OCR"
          style={{ flex: 1 }}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {error && <div className="error-box">{error}</div>}

      {results && results.length === 0 && (
        <p className="text-muted">No results. Ingest some models first.</p>
      )}

      {results && results.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {results.map((r, i) => (
            <div key={i} className="card" style={{ padding: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <strong>{r.model_id}</strong>
                <span className="text-muted">score {r.score.toFixed(3)}</span>
              </div>
              <p style={{ margin: "8px 0", whiteSpace: "pre-wrap" }}>{r.chunk_text}</p>
              <a href={r.hf_url} target="_blank" rel="noreferrer">
                View on Hugging Face ↗
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
