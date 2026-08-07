# ModelScout — Multi-Agent Radar for Open-Source Models (v1 Vertical Slice)

ModelScout watches Hugging Face for new/trending models, filters them against a configurable **interest profile**, extracts real specs from the model card, and produces a ranked digest — built to demonstrate real agentic-engineering practice (LangGraph orchestration, RAG, cost-aware model routing), not a thin API wrapper.

This is the **v1 vertical slice**: one complete, working path through the core architecture. Deliberately out of scope for now (see [Phase 2](#phase-2-not-built-yet)): the Fact-Checker (LLM-as-judge) agent, a dashboard, AWS deployment, and the eval-suite/LLMOps layer.

## Architecture

```
HF Hub API (unauthenticated)
        │
        ▼
  Ingestion + filters ──▶ Postgres (models table, upsert/idempotent)
        │                        │
        │                        ▼
        │              RAG: chunk + embed READMEs (BAAI/bge-small-en-v1.5)
        │                        │
        │                        ▼
        │              pgvector (model_card_chunks) ──▶ GET /search (semantic search)
        ▼
  LangGraph: fan-out one Triage call per candidate (Send)
        │
        ▼
  Triage Agent (local zero-shot classification, facebook/bart-large-mnli, CPU, $0)
        │
   is_relevant? ──── no ──▶ skip_node (no LLM call)
        │ yes                      │
        ▼                          │
  Extractor Agent (Claude, forced tool-use, 3-level tolerant fallback parser)
        │                          │
        └──────────┬───────────────┘
                    ▼
         Notifier (ranks + writes digests/<date>-<profile>.md)
                    │
                    ▼
         cli.py / POST /pipeline/run (same orchestrator function)
```

**The core claim this architecture makes verifiable, not just assertable:** the Extractor Agent (the only node that calls the paid Anthropic API) is reachable *only* via the `"extract"` branch of the triage routing decision — there is no other edge into it. `pipeline.py` asserts `n_extracted == n_triage_pass` after every run, so "N models screened locally for $0, only M sent to Claude" isn't a docstring promise, it's a checked invariant.

## Setup

### Prerequisites
- Python 3.14 (this project was built and tested on it — torch/transformers CPU wheels are available; see note below if you're on an older/newer version)
- Docker Desktop
- An Anthropic API key
- No Hugging Face token required (ingestion runs unauthenticated, rate-limited but sufficient for a 20-30 model demo)

### Steps

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If `pip install torch` can't find a wheel for your Python version, install Python 3.12 as a sibling interpreter and recreate the venv with it — don't fight the newest Python for ML wheel availability.

**Windows note:** if `pip`/`huggingface_hub` requests fail with `CERTIFICATE_VERIFY_FAILED` (common when antivirus/corporate software does TLS inspection), `pip install pip-system-certs` fixes it by making Python use the Windows certificate store instead of the bundled `certifi` CAs.

```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY

docker compose up -d
# wait for the container to report healthy, then confirm:
docker exec modelscout-postgres psql -U modelscout -d modelscout -c "\dt"

python scripts/check_env.py   # sanity-checks ML imports, DB connection, and the API key before you run anything real
```

### Run the demo

```bash
python cli.py run --profile vlm_ocr --limit 20 -v
```

This ingests candidates for the example `vlm_ocr` profile (vision-language models under 5B params for on-device OCR — edit/add profiles under `config/interest_profiles/`), runs them through the full graph, and writes a ranked digest to `digests/<date>-vlm_ocr.md`.

Then try the API:

```bash
uvicorn modelscout.api.main:app --reload
curl "http://localhost:8000/search?q=lightweight vision model for document OCR&k=5"
curl -X POST http://localhost:8000/pipeline/run -H "Content-Type: application/json" -d "{\"profile_name\": \"vlm_ocr\", \"limit\": 20}"
```

## Tuning

- **Model / cost**: `ANTHROPIC_EXTRACTOR_MODEL` in `.env` — extraction from a README isn't reasoning-heavy, so a cheaper model is a reasonable choice; no code change needed.
- **Triage threshold / candidate labels / which pipeline_tags to watch**: edit the profile YAML (e.g. `config/interest_profiles/vlm_ocr.yaml`). `triage.candidate_labels[0]` is the "relevant" hypothesis the classifier scores independently (multi_label scoring — see `modelscout/agents/triage_node.py` for why a forced two-way softmax was tried and rejected).
- **Diff/README size cap**: the Extractor call truncates README input at 12,000 characters (`extractor_node.py`).

## Design notes worth knowing before extending this

- **Triage input cleaning matters more than it looks.** Raw HF READMEs open with YAML frontmatter, badge/logo HTML, and changelog lists before any descriptive prose — feeding that directly to zero-shot classification produced backwards results (a real vision-language model scored *lower* than unrelated text-only LLMs) during development. `triage_node.py`'s `_clean_for_classification` strips all of that first. If you swap the triage model or approach, re-verify against a few real model cards before trusting the scores — this class of bug is silent, not an exception.
- **Idempotent by design.** Re-running the same profile upserts on `model_id` rather than duplicating rows — safe to schedule on a cron without deduping logic elsewhere.
- **`pipeline.py` is the only orchestrator.** Both `cli.py` and `POST /pipeline/run` call the exact same function — if you add a third entrypoint (a scheduled job, a Slack slash command), call `modelscout.pipeline.run()`, don't reimplement the flow.

## Phase 2 (not built yet)

- **Fact-Checker agent**: LLM-as-judge evaluation of the Extractor's declared-benchmark claims against a golden dataset, with regression tests that catch quality drops when a prompt or model changes.
- **LLMOps**: prompt versioning, per-agent tracing/cost observability (Langfuse), CI that runs the eval suite on any PR touching prompts or agents.
- **Serving**: React dashboard over the API; move `POST /pipeline/run` to `BackgroundTasks` + a status-polling endpoint instead of running synchronously.
- **Deploy**: Docker Compose → AWS (ECS Fargate or Lambda, RDS with pgvector).
