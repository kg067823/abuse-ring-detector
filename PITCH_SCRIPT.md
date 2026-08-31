# AbuseRing — three-minute pitch

## 0:00–0:20 — Hook and problem

“Fraud systems inspect transactions. AbuseRing investigates networks.

Imagine several customers placing ordinary-looking orders. No single order
is obviously fraudulent. But behind them, devices, addresses, IPs, payment relationships, and timing may connect them.

If we inspect only one order at a time, the signal fragments. AbuseRing helps an
investigator inspect the coordinated relationships behind those events.”

## 0:20–0:40 — The blind spot

“Transaction scoring is valuable, but coordinated abuse is relational. The signal
may be weak on each individual event and strong in the relationships between
events, customers, and shared infrastructure.

That creates two operational problems: fragmented alerts for the analyst, and a
network pattern that is hard to reconstruct after the fact.”

## 0:40–1:00 — What AbuseRing does

“AbuseRing adds a network intelligence layer alongside existing transaction risk
controls. It combines temporal behavior, graph relationships, customer-relative
activity, two-hop connections, and suspicious subgraph signals in a frozen
Model F-R1 scorer.

When a shadow alert crosses the locked review threshold, the backend consolidates
related alerts into one masked investigation case with observed evidence, a
timeline, and a network view.”

## 1:00–2:10 — Live demo

“Here is the Command Center. This is explicitly demo data, and enforcement is
off.

I’ll choose Mixed multi-entity and run the scenario. Each event is sent through
the real `/v1/predict` API, not a prebuilt case. Watch the progression: plausible orders arrive, and the UI reports scores,
alerts, and cases returned so far from the actual backend. If this run produces a
qualifying alert, I’ll open the returned case workspace; otherwise I’ll show the
alert/evidence state and say so explicitly. The important phrase is ‘Observed evidence
associated with elevated risk.’ We show the score, related alerts, observed
exposure, and evidence such as shared entities and prior activity windows. We do
not claim that any single feature caused abuse.

This network is the moment: customers and orders connect through shared
infrastructure. The graph turns a list of isolated alerts into one investigation
story. The timeline shows when the observable relationships and risk signal
appeared.

Now I’ll run the legitimate high-connectivity control. Shared infrastructure
alone is not enough. We show whatever the frozen R1 pipeline actually returns;
we never force a negative result or fabricate a case.”

## 2:10–2:35 — Technical differentiation

“Technically, R1 has 137 ordered features: baseline behavior, graph history,
temporal velocity, customer-relative behavior, two-hop relationships, and
subgraph structure. Features are strictly as-of: future events cannot influence
an earlier score. A new validation-only isotonic calibrator is persisted with the
artifact.

That workspace is available when the backend returns a case; otherwise the
alert queue remains the honest fallback.

The product layer is equally important: deterministic observable-entity case
consolidation, masked identifiers, non-causal evidence, and a timeline an
analyst can actually work.”

## 2:35–2:50 — Measured results and business value

“On the reconstructed chronological held-out synthetic test split of 8,180
orders, R1 measured PR-AUC 0.798, precision 0.919, recall 0.640, and F1 0.754 at
the locked 0.50 threshold. The artifact rebuilt byte-for-byte with the same
SHA-256, and 141 Python 3.11 tests passed.

These are not Razorpay production numbers. The business promise is workflow:
instead of asking an investigator to chase disconnected transactions, AbuseRing
presents one evidence-backed network case and its observed exposure.”

## 2:50–3:00 — Close

“AbuseRing does not replace transaction risk. It complements it by asking one
more question: what is this transaction connected to?

Today it is safely shadow-only. Live observation is not started, Canary Stage 1
is blocked, and enforcement is disabled. We built the evidence and the case
workflow first, so customer-impacting decisions can be earned—not assumed.”
