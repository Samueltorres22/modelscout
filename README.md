# ModelScout — Multi-Agent Radar for Open-Source Models (v1 Vertical Slice)

[![Tests](https://github.com/Samueltorres22/modelscout/actions/workflows/tests.yml/badge.svg)](https://github.com/Samueltorres22/modelscout/actions/workflows/tests.yml)

ModelScout watches Hugging Face for new/trending models, filters them against a configurable **interest profile**, extracts real specs from the model card, fact-checks the declared benchmark claims, and produces a ranked digest — built to demonstrate real agentic-engineering practice (LangGraph orchestration, RAG, cost-aware model routing, eval-driven development), not a thin API wrapper.

This is the **v1 vertical slice**: one complete, working path through the core architecture, including a Fact-Checker agent with its own golden regression suite, lightweight per-agent cost/latency observability, and CI. Deliberately out of scope for now (see [Phase 2](#phase-2-not-built-yet)): a dashboard, AWS deployment, and prompt versioning.

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
  LangGraph: fan-out one process_model_node call per candidate (Send)
        │
        ▼
  Triage Agent (local zero-shot classification, facebook/bart-large-mnli, CPU, $0)
        │
   is_relevant? ──── no ──▶ skip (no LLM call)
        │ yes
        ▼
  Extractor Agent (Claude, forced tool-use, 3-level tolerant fallback parser)
        │
   declared_benchmarks non-empty & extraction succeeded? ──── no ──▶ skip (no LLM call)
        │ yes
        ▼
  Fact-Checker Agent (Claude, LLM-as-judge plausibility/consistency read, same forced
  tool-use + tolerant-fallback pattern, own golden regression suite)
        │
        ▼
         Notifier (ranks + writes digests/<date>-<profile>.md)
                    │
                    ▼
         cli.py / POST /pipeline/run (same orchestrator function)
```

**The core claim this architecture makes verifiable, not just assertable:** the Extractor Agent (the only path that calls the paid Anthropic API for extraction) only runs for models Triage marked relevant, and the Fact-Checker only runs for models with something to check. `pipeline.py` asserts `n_extracted == n_triage_pass` after every run — "N models screened locally for $0, only M sent to Claude" isn't a docstring promise, it's a checked invariant. Fact-Checker's gate (extraction succeeded AND has declared benchmarks) is the same cost-aware philosophy applied a second time.

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

### Run the Fact-Checker's golden eval

```bash
python scripts/run_golden_eval.py
```

Makes real Anthropic API calls (one per example in `config/golden/fact_check_examples.yaml`) and reports whether each hand-labeled example's verdict still lands in the expected bucket. **Not** a pytest test — it costs real money, so run it deliberately when you change the judge's prompt or model, not on every `pytest` invocation. This is the actual eval-driven-development loop: change the prompt, run this, see if the verdict distribution held before you ship. Exits non-zero only if the pass rate drops below 50% (configurable via `--min-pass-rate`) — an LLM judge legitimately disagrees with hand labels on fuzzy cases run to run (verified baseline: 5/8), so this catches a real collapse rather than flagging normal variance.

## Observability

Every Extractor/Fact-Checker Claude call is recorded to the `llm_calls` table (agent name, model, input/output tokens, latency, and an optional estimated cost) by `modelscout/observability.py` — self-built rather than a hosted product like Langfuse, since the project already runs entirely local/self-hosted and the point is having real numbers to check the cost-aware-routing story against, not a second service to stand up. Recording a call can never break the pipeline: a DB write failure there is logged and swallowed, not raised.

```bash
python scripts/cost_report.py            # all-time summary by agent
python scripts/cost_report.py --since 24h
```

Cost estimation is opt-in: set `PRICE_PER_MTOK_INPUT` / `PRICE_PER_MTOK_OUTPUT` in `.env` (both required to enable it) if you want `$` figures. Deliberately unset by default rather than shipping a hardcoded per-model price table that goes stale.

## Dashboard

A React + TypeScript UI (`dashboard/`) over the API — the models catalog with fact-check verdicts, semantic search, a pipeline-run trigger, and the observability summary, browsable instead of requiring `curl`/`psql`. Its own Vite dev server, no build/deploy pipeline yet, no auth (local dev only, same as the API itself right now).

```bash
# terminal 1 -- the API (from the project root, with .venv active)
uvicorn modelscout.api.main:app --reload

# terminal 2 -- the dashboard
cd dashboard
npm install
npm run dev
```

Open http://localhost:5173. The API's `CORSMiddleware` in `modelscout/api/main.py` explicitly allows `http://localhost:5173` — if you change the dashboard's dev port, update that allowlist too. To point the dashboard at a different API URL, copy `dashboard/.env.example` to `dashboard/.env.local` and set `VITE_API_BASE_URL`.

No new backend concept here beyond four read-only endpoints (`GET /models`, `GET /models/{id}`, `GET /runs`, `GET /observability/summary`) that all do the same thing: join `models` against the *latest* row per model in `triage_results`/`extracted_specs`/`fact_checks` via `LEFT JOIN LATERAL ... ORDER BY ... LIMIT 1` (see `modelscout/api/queries.py`) — those three tables can have multiple rows per model from re-ingestion, the read-side of the same idempotency `pipeline.py` already handles on writes.

## CI

- **`.github/workflows/tests.yml`** — runs `pytest tests/` on every push/PR to `main`. No API key or database needed: every test exercises pure logic (filters, the tolerant parsers) against synthetic inputs.
- **`.github/workflows/golden-eval.yml`** — manually triggered only (`workflow_dispatch`), never on push, because it costs real money. Needs an `ANTHROPIC_API_KEY` repository secret. Runs `scripts/run_golden_eval.py` deliberately, e.g. after changing the Fact-Checker's prompt.

## Tuning

- **Model / cost**: `ANTHROPIC_EXTRACTOR_MODEL` in `.env` — used by both Extractor and Fact-Checker. Extraction and fact-checking from a README aren't reasoning-heavy tasks, so a cheaper model is a reasonable choice; no code change needed.
- **Triage threshold / candidate labels / which pipeline_tags to watch**: edit the profile YAML (e.g. `config/interest_profiles/vlm_ocr.yaml`). `triage.candidate_labels[0]` is the "relevant" hypothesis the classifier scores independently (multi_label scoring — see `modelscout/agents/triage_node.py`).
- **Fact-Checker rubric / golden set**: `modelscout/agents/fact_checker_node.py`'s `_SYSTEM_PROMPT`, and `config/golden/fact_check_examples.yaml` for the regression cases.
- **README size cap**: the Extractor call truncates README input at 12,000 characters (`extractor_node.py`).
- **Cost estimates in `cost_report.py`**: `PRICE_PER_MTOK_INPUT` / `PRICE_PER_MTOK_OUTPUT` in `.env`.

## Design notes worth knowing before extending this

Several of these were found by running the real pipeline against real models, not by reasoning about the code in the abstract — worth reading before you change the agents or the graph.

- **Triage classifies on the model's name + HF tags, not README prose.** The first approach — classifying cleaned README text — produced backwards results (a real vision-language model scored *lower* than unrelated text-only LLMs) even after stripping YAML frontmatter, badges, changelogs, and code blocks: `facebook/bart-large-mnli` is trained on short sentence pairs and degrades on long, heterogeneous documents no matter how much noise you strip. HF's own `tags` field is short, curated, and a much better match for the classifier's training distribution. Known tradeoff: a low-quality community upload can carry misleading tags copied from a base model (verified case: a GGUF text-only "uncensored" merge tagged `vision, multimodal` scored as relevant) — a real data-quality limit of the free tag signal, not a bug in the classifier call. See `triage_node.py`'s docstring.
- **LangGraph's `Send`-based fan-out does not give each parallel branch isolated state.** An earlier version modeled Triage and Extractor/Skip as separate graph nodes joined by a conditional edge, on the theory that "only the extract branch can reach the Extractor node" should be a structural graph property. It isn't possible that way: every `Send`-spawned branch shares the same graph-level state channels, and any key without an `Annotated` reducer raises `InvalidUpdateError: Can receive only one value per step` the moment more than one candidate runs in the same superstep. Fix: collapse the whole per-item chain (triage → extract/skip → fact-check) into one node (`process_model_node`) that calls the underlying functions directly and writes only to the one `Annotated[list, operator.add]` field (`results`). The cost-aware-routing guarantee is now an explicit `if`, verified by the `n_extracted == n_triage_pass` assertion in `pipeline.py` rather than by graph topology. See `graph.py`'s docstring.
- **Forced tool-use reliably enforces `required`/`enum`/`type` — not `minLength` or numeric `minimum`/`maximum`.** Verified repeatedly and empirically on the Fact-Checker: even with `"minLength": 1` on `reasoning` and `"minimum": 0.05` on `confidence`, both fields sometimes came back empty/zero anyway (more often on models with very large `declared_benchmarks` lists). `fact_checker_node.py`'s `_normalize_result` patches both rather than surfacing a misleading blank/zero — same tolerant-fallback philosophy as the parser itself.
- **Field ORDER in a forced tool-use schema controls generation order, and that matters for consistency.** Early on, `verdict` was the first field in the schema. The model would commit to a verdict, then write `reasoning`/`flags` that flatly contradicted it — one case had `reasoning` conclude "looks like fabricated or mislabeled benchmark reporting" while `verdict` said `"plausible"`. Moving `verdict` to be the LAST field (decided only after `flags`, `consistency_issues`, `reasoning`, `confidence` are written) fixed it: the conclusion now has to follow from the analysis instead of being picked before the analysis exists. If you add a new judge/extractor tool schema, put "decide" fields after "analyze" fields.
- **Idempotent by design.** Re-running the same profile upserts on `model_id` rather than duplicating rows — safe to schedule on a cron without deduping logic elsewhere.
- **`pipeline.py` is the only orchestrator.** Both `cli.py` and `POST /pipeline/run` call the exact same function — if you add a third entrypoint (a scheduled job, a Slack slash command), call `modelscout.pipeline.run()`, don't reimplement the flow.
- **`db/schema.sql` only applies on first container init.** Postgres runs `docker-entrypoint-initdb.d` scripts once, on an empty volume. If you edit the schema after the volume already has data, apply the DDL by hand against the running container (`docker exec modelscout-postgres psql ...`) as well as updating `schema.sql` for fresh installs — there's no migration tool in this v1 scope.
- **Anthropic calls already retry transient failures — verified from the SDK source, not assumed.** `_get_client()` sets `max_retries=3` explicitly; `anthropic._base_client`'s retry predicate retries on 408/409/429 and any 5xx with exponential backoff before a response ever reaches the tolerant-parser fallback path. Worth stating outright since it's easy to assume a raw API client has no retry behavior unless you check.

## Phase 2 (not built yet)

- **Prompt versioning**: agent prompts live in code, not in a versioned/diffable store; no automated way yet to correlate a golden-eval run with the exact prompt version it tested.
- **Serving**: React dashboard over the API; move `POST /pipeline/run` to `BackgroundTasks` + a status-polling endpoint instead of running synchronously.
- **Deploy**: Docker Compose → AWS (ECS Fargate or Lambda, RDS with pgvector).
