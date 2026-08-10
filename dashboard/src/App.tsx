import { useState } from "react";
import { ModelsView } from "./views/ModelsView";
import { SearchView } from "./views/SearchView";
import { RunPipelineView } from "./views/RunPipelineView";
import { ObservabilityView } from "./views/ObservabilityView";

type Tab = "models" | "search" | "run" | "observability";

const TABS: { id: Tab; label: string }[] = [
  { id: "models", label: "Models" },
  { id: "search", label: "Search" },
  { id: "run", label: "Run Pipeline" },
  { id: "observability", label: "Observability" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("models");

  return (
    <div style={{ paddingTop: 24, paddingBottom: 48 }}>
      <header style={{ marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>ModelScout</h1>
        <p className="text-muted" style={{ margin: "4px 0 0" }}>
          Multi-agent radar for open-source models
        </p>
      </header>

      <nav
        style={{
          display: "flex",
          gap: 4,
          borderBottom: "1px solid var(--border)",
          marginBottom: 20,
        }}
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              background: tab === t.id ? "var(--surface)" : "transparent",
              color: tab === t.id ? "var(--accent)" : "var(--text-muted)",
              border: "none",
              borderBottom: tab === t.id ? "2px solid var(--accent)" : "2px solid transparent",
              borderRadius: 0,
              padding: "10px 16px",
              fontWeight: tab === t.id ? 600 : 400,
            }}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "models" && <ModelsView />}
        {tab === "search" && <SearchView />}
        {tab === "run" && <RunPipelineView />}
        {tab === "observability" && <ObservabilityView />}
      </main>
    </div>
  );
}
