# AbuseRing three-minute demo script

## Presenter headline

**Fraud systems inspect transactions. AbuseRing investigates networks.**

The UI is a single monitoring screen: transactions flow in, AbuseRing scores
every one automatically, and connected risk becomes one investigation case.
There is no pattern selection and no manual refresh — the presenter only starts
the traffic.

## Setup

From the repository root:

```bash
# macOS/Linux
ADMIN_KILL_SWITCH_TOKEN=demo-secret .venv/bin/python scripts/start_demo.py --docker

# Windows PowerShell
$env:ADMIN_KILL_SWITCH_TOKEN="demo-secret"
.venv\Scripts\python.exe scripts\start_demo.py --docker
```

Open `http://localhost:8501`. Confirm the header shows `DEMO / SYNTHETIC`,
`SHADOW MODE`, `ENFORCEMENT OFF` and `○ STREAM IDLE` with an empty feed.

## 0:00–0:20 — Problem

On the quiet Live Monitor, say: “Individual transactions can look ordinary while
the same infrastructure quietly connects many accounts. AbuseRing watches every
transaction and looks for the network forming behind them — automatically, with
no pattern selection.”

## 0:20–0:35 — Start the stream

Expand **Demo Traffic Generator** (collapsed by default — it is a traffic
source, not a detection mode), leave **Mixed multi-entity** selected, click
**START DEMO TRAFFIC**, and collapse the generator again. Say: “A stream of
transactions is now arriving. Each one is scored by the real frozen Model F-R1
through the live API.”

## 0:35–1:10 — Scores appear

Point to the Transaction Stream filling row by row (~0.9 s apart). Early events
score low — `NORMAL`, then `WATCHING`. Say: “No single order is obviously
fraudulent. The score reflects what each event is connected to.”

## 1:10–1:30 — The risk signal

When the first `ALERT` row appears, point to the **NETWORK RISK SIGNAL**
banner: “The calibrated score crossed the 0.50 review threshold. This is a
review signal for an analyst — not confirmed fraud, and nothing is enforced.”
Statuses derive directly from backend scores; fallback responses never present
as alerts.

## 1:30–2:10 — The case builds itself

Point to **Active Investigation** appearing below the feed without any click:
risk, connected accounts, alerts, shared entities, observed exposure. Say:
“Related alerts consolidate automatically into one investigation case.” As new
alerts arrive, watch the numbers and the **Network** graph grow: customers
(circles) connecting through shared devices, addresses, IPs, and payments
(diamonds) — supposedly separate customers, one network.

## 2:10–2:30 — Evidence and timeline

Read one or two evidence lines under **Observed evidence associated with
elevated risk** (observed signals, never causal claims). Point to the
**Timeline**: threshold crossing and case creation, from backend data.

## 2:30–2:50 — Legitimate control

Expand the generator, choose **Legitimate high-connectivity**, **START DEMO
TRAFFIC**, collapse. Scores stay low and no case appears. Say: “Shared
infrastructure alone is not enough. This is the real frozen model’s answer — we
never force a result.”

## 2:50–3:00 — Close

Point to the bottom status strip (API / Redis / Model F-R1 indicators, Shadow
Mode ON, Enforcement OFF). Close with: “AbuseRing turns connected risk signals
into one explainable analyst case, while the production decision path stays
safely in shadow mode.”

## Guardrails

All displayed data is `DEMO / SYNTHETIC` — not live-production observation
evidence. The seven-day gate remains not started and Canary Stage 1 remains
blocked. If the backend returns no case for a run, the UI says so; never
fabricate a case or an alert.
