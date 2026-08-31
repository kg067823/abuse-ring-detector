# AbuseRing Command Center

## Frontend decision

The repository already had a Streamlit frontend and no JavaScript package,
React build, or frontend deployment pipeline. The Command Center therefore
uses Streamlit with a small HTTP API client. This keeps demo startup fast,
reuses the Python 3.11 environment, and avoids introducing a second frontend
runtime while preserving a clean backend/frontend boundary.

## Architecture

```text
Streamlit Command Center
        │ app/api_client.py
        ▼
FastAPI R1 + Investigator APIs ── Redis state
        │
        ├── frozen Model F-R1 /v1/predict
        ├── non-causal /v1/explain
        └── alerts, cases, evidence, graph, timeline, status, notes
```

The UI renders backend responses and sends mutations through the API. It does
not duplicate case consolidation, severity, evidence, or status logic.

## Pages

- **Overview** — open/critical cases, high-risk alerts, exposure, average risk,
  severity/status distributions, and model provenance.
- **Alert Queue** — risk-sorted masked alerts with search, severity, and minimum
  risk filtering.
- **Investigation Cases** — status/severity/search filtering and case summaries.
- **Case Workspace** — observed evidence, timeline, graph summary, technical
  history, and authenticated mutation guidance.
- **Network Explorer** — graph nodes, typed relationships, legend-like type
  metadata, and a Graphviz view with a tabular fallback.
- **System Health** — health/readiness/liveness/metrics and prominent shadow-only
  state.
- **Demo Mode** — instructions for replaying actual R1 API scenarios.

## Backend integration

The UI consumes `/v1/alerts`, `/v1/cases`, case detail, graph, timeline,
evidence, status, notes, `/health`, `/readiness`, `/liveness`, and `/metrics`.
`app/api_client.py` uses `ABUSERING_API_URL` and an optional
`ABUSERING_ADMIN_TOKEN`; no token is embedded in frontend source.

## Privacy and safety

Only masked/pseudonymous IDs returned by the backend are displayed. The UI uses
“Observed evidence associated with elevated risk” and never claims causality.
It visibly labels synthetic data as `DEMO DATA / SYNTHETIC`. It does not expose
raw entity identifiers, ground-truth ring IDs, or enforcement actions.

The R1 model remains frozen, shadow-only, and non-enforcing:

```text
SHADOW_MODE=true
ENFORCE_DECISIONS=false
LIVE_PRODUCTION_OBSERVATION=NOT_STARTED
QUALIFYING_DAYS=0/7
CANARY_STAGE_1=BLOCKED
```

## Startup

Terminal 1:

```bash
ADMIN_KILL_SWITCH_TOKEN=demo-secret docker compose up --build
```

Terminal 2:

```bash
uv pip install --python .venv/bin/python -e '.[frontend]'
ABUSERING_API_URL=http://localhost:8000 \
ABUSERING_ADMIN_TOKEN=demo-secret \
.venv/bin/streamlit run app/command_center.py
```

The UI is a local/demo investigator console. Production use requires proper
identity, RBAC, durable case storage, and authenticated deployment controls.

## Tests

Pure frontend helper tests cover aggregate calculations, filtering/sorting,
timeline ordering, graph counts, and no-invented-metrics behavior. Backend
regression tests remain the authoritative R1 compatibility gate.

## Limitations

- Case storage is currently process-local in-memory storage.
- The UI is desktop-first and Streamlit-based rather than a production React
  application.
- Graph interaction is limited by Streamlit/Graphviz capabilities; a richer
  interactive canvas can be added later.
- Demo scenarios require the separate replay helper and do not represent live
  production evidence.
- Status and note mutations require a configured admin token; production RBAC
  is a follow-up requirement.
