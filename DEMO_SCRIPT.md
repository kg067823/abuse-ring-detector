# AbuseRing three-minute demo script

## Setup

From the repository root:

```bash
# macOS/Linux
ADMIN_KILL_SWITCH_TOKEN=demo-secret .venv/bin/python scripts/start_demo.py --docker

# Windows PowerShell
$env:ADMIN_KILL_SWITCH_TOKEN="demo-secret"
.venv\Scripts\python.exe scripts\start_demo.py --docker
```

Open `http://localhost:8501`. Confirm the banner says `DEMO DATA / SYNTHETIC`,
`Model F-R1`, `SHADOW MODE`, and `ENFORCEMENT OFF`.

## 0:00–0:20 — Problem

Open **Overview**.

Say: “Individual transactions can look ordinary while the same infrastructure
quietly connects many accounts. AbuseRing looks for the network forming behind
the transaction.”

## 0:20–0:45 — Plausible events

Open **Demo Mode** and choose **Mixed multi-entity**. Explain that the replay
uses ordinary-looking amounts and customers, then progressively shares device,
address, IP, and payment infrastructure.

## 0:45–1:20 — Network forms

Click **RUN SCENARIO**. Point to the `Event n of 8` progression, score, alert,
and case counts. The UI is sending every event through the real `/v1/predict`
endpoint and waiting for the backend case API; it is not fabricating cases.

## 1:20–1:50 — Case appears

Open **Case Workspace** and select the returned case. Highlight severity, risk,
alert count, related customers, and **observed exposure**.

Say: “The score is a shadow signal for analyst review. No customer transaction
is blocked or modified.”

## 1:50–2:20 — Evidence and graph

Show **Observed evidence associated with elevated risk**, then open **Network
Explorer**. Point out customer/order/entity node types and shared relationship
edges. Hover/zoom if Graphviz is available; use the relationship table as the
fallback.

## 2:20–2:40 — Timeline and analyst workflow

Return to the workspace and show the timeline. Open technical details only if a
judge asks. Status/note mutations require the configured local admin token.

## 2:40–2:55 — Legitimate control

Open Demo Mode, choose **Legitimate high-connectivity**, and run it with a new
replay ID. Say: “Shared infrastructure alone is not enough. This control is
reported from the real frozen R1 output; we do not force a negative result.”

## 2:55–3:00 — Close

Open **System Health** briefly and point to R1 checksum verification, 137
features, Redis health, shadow mode, and enforcement disabled.

Close with: “AbuseRing turns coordinated risk signals into an explainable analyst
case, while keeping the production decision path safely in shadow mode.”

## Guardrails

All displayed cases are `DEMO / SYNTHETIC`. They are not live-production
observation evidence. The seven-day gate remains not started and Canary Stage 1
remains blocked.
