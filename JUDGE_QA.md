# Judge Q&A

## 1. What is AbuseRing?

Fraud systems inspect transactions. AbuseRing investigates networks. It adds
network-level intelligence and analyst case consolidation to transaction risk.

## 2. Why not only score each transaction?

Coordinated abuse can be weak on each individual event but obvious in shared
entities, timing, velocity, and graph structure. Transaction-level scores
fragment that pattern.

## 3. Why graph features?

A graph turns customers, orders, devices, addresses, IPs, and payments into
relationships. It exposes coordinated activity that an isolated row cannot show.

## 4. Why machine learning instead of rules?

Rules are excellent transparent evidence, such as repeated device sharing. R1
combines many weak signals: temporal velocity, customer-relative behavior,
entity relationships, two-hop structure, and subgraph features. Rules and ML are
complementary.

## 5. Why not a graph neural network?

R1 deliberately uses a deterministic, inspectable feature design and
HistGradientBoosting model. A GNN may be a future experiment, but it is not
necessary to demonstrate the product workflow and would change the frozen model
contract.

## 6. Why not an LLM?

The scoring path needs deterministic, bounded, auditable outputs. R1 handles
risk scoring. The case layer generates deterministic observed evidence; no LLM
is required to make or explain a customer-impacting decision.

## 7. What exactly is frozen?

Model F-R1: `model_f_r1`, identity
`graph_temporal_custrel_subgraph`, 137 ordered features, seed 42, threshold
0.50, isotonic calibration, and artifact SHA-256
`3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff`.

## 8. What does 137 features mean?

19 baseline, 18 graph, 30 temporal, 30 customer-relative, 20 two-hop, and 20
subgraph features, assembled in a fixed order.

## 9. How did you prevent leakage?

Features are computed strictly as-of. Only events with timestamps earlier than
the scored event are eligible; future events cannot alter an earlier feature
vector.

## 10. What are the measured results?

On the reconstructed chronological held-out synthetic test split of 8,180
orders: PR-AUC 0.79812, ROC-AUC 0.93815, precision 0.91874, recall 0.63994, and
F1 0.75440 at threshold 0.50. These are not production results.

## 11. What does precision mean here?

At the locked threshold, about 92% of R1 alerts corresponded to positive labels
in this held-out synthetic test split. It is not a promise about live traffic.

## 12. What about false positives?

Shared device or address alone is not fraud. The control scenario demonstrates
that shared infrastructure must be considered alongside velocity, customer
history, multiple relationships, and graph structure. The control outcome is
reported honestly from R1, not forced.

## 13. What does calibration do?

The new isotonic calibrator maps raw model scores to better-behaved probabilities
using validation data only. Test calibrated Brier score was 0.02840 and ECE was
0.00752 in the R1 manifest.

## 14. Are explanations causal?

No. The UI says “Observed evidence associated with elevated risk.” Evidence is
supporting observed association with provenance and time windows, not proof that
an attribute caused abuse.

## 15. Does the system block customers?

No. `SHADOW_MODE=true`, `ENFORCE_DECISIONS=false`, and
`enforcement_applied=false`. No customer transaction is blocked or modified.

## 16. How does a case form?

A qualifying shadow alert is deduplicated by order ID. Alerts consolidate using
observable shared customer/device/address/IP/payment relationships. Synthetic
`ring_id` and labels are never used for production grouping.

## 17. How does this help investigators?

Instead of chasing disconnected alerts, an analyst receives one case with risk,
observed exposure, evidence cards, related entities, graph context, a timeline,
and analyst history.

## 18. How does it fit Razorpay?

Conceptually, it complements an existing payment/risk stack: payment/order
events enter existing controls, AbuseRing adds network intelligence, and
investigator cases return to the risk operations workflow. We have no access to
Razorpay internal systems or data.

## 19. Does fraud detection already exist?

Yes, and AbuseRing is not claiming to replace it. The differentiation is the
integrated workflow: temporal multi-entity graph signals, alert consolidation,
masked evidence, timeline, and a case an analyst can work.

## 20. Is this production ready?

The model and engineering stack are validated for pre-production/demo use, with
138 automated tests passing. Enforcement is intentionally disabled. A real
deployment requires live shadow observation, durable case persistence, proper
identity/RBAC, and authorization before customer-impacting decisions.

## 21. What is Redis doing?

Redis supports streaming inference feature state and readiness dependency checks.
It is not currently the long-term case system of record; the demo case
repository is process-local.

## 22. Can it scale?

The API is containerized and designed around shared feature state, but aggregate
case persistence and multi-worker investigator consistency need a durable case
store before production scale claims. We do not present unsupported throughput
numbers.

## 23. Why is the model called R1?

The historical Model F binary was unrecoverable. Rather than fabricate recovery
or reuse its checksum, we reconstructed the documented architecture from source,
regenerated it deterministically, measured it afresh, and froze it as Model F-R1
with a new checksum.

## 24. Is the synthetic data realistic?

It is useful for deterministic engineering and held-out evaluation and includes
shared legitimate infrastructure as hard negatives. It is not a substitute for
merchant production traffic, delayed labels, or live drift validation.

## 25. What is the next step?

Pitch and present the demo now. Afterward, add durable case persistence and
production identity/RBAC, then attach real shadow traffic and earn the seven-day
gate before any canary decision.
