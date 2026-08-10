const LABELS: Record<string, string> = {
  plausible: "Plausible",
  questionable: "Questionable",
  implausible: "Implausible",
};

export function VerdictPill({ verdict }: { verdict: string | null }) {
  if (!verdict) {
    return <span className="pill pill-unknown">—</span>;
  }
  const className = `pill pill-${verdict}`;
  return <span className={className}>{LABELS[verdict] ?? verdict}</span>;
}
