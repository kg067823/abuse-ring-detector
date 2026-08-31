# Demo recovery guide

## Docker does not start

Verify Docker Desktop is running and the R1 artifact exists:

```bash
.venv/bin/python scripts/start_demo.py --no-ui
```

On Windows PowerShell use `.venv\Scripts\python.exe`. Run `docker compose
config` to surface missing `ADMIN_KILL_SWITCH_TOKEN` or port errors.

## Redis/API is not ready

Inspect logs:

```bash
docker compose logs --tail=100 redis api
```

Wait for Redis health and `/readiness`. The API fails closed when the R1
artifact, manifest, contract, or Redis dependency is unavailable. Do not replace
or retrain the model.

## Streamlit does not start

Install the optional frontend dependencies in the Python 3.11 environment:

```bash
uv pip install --python .venv/bin/python -e '.[frontend]'
```

Then run `.venv/bin/streamlit run app/command_center.py` or the Windows
`.venv\Scripts\streamlit.exe` equivalent.

## Graph rendering fails

The case workspace still shows evidence, timeline, node/edge counts, and a
relationship table. Install Graphviz (`dot`) if a rendered graph is required;
the UI does not fabricate graph data when rendering is unavailable.

## Scenario replay fails

Confirm the API URL and use a fresh deterministic replay ID in Demo Mode. A
repeated ID may return cached/idempotent responses. Check `/readiness` and API
logs. Only `/v1/predict` responses are used; a missing alert/case is reported,
not invented.

## Stale demo state

The current investigator case repository is process-local and the inference
state is Redis-backed. For a disposable demo, stop the stack and remove only
the demo Redis volume after confirming that no shared environment is using it:

```bash
docker compose down
# Only for a disposable local demo volume:
docker volume rm abuse-ring-detector_redis_data
```

Never run destructive volume removal against a production or shared environment.

## Safety

All recovery paths preserve `SHADOW_MODE=true`, `ENFORCE_DECISIONS=false`, the
frozen R1 artifact, and the seven-day live gate. Never present synthetic replay
as production evidence.
