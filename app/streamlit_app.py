"""Friendly artifact viewer and runner for the AbuseRing POC.

Run from the repository root with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install optional dependencies with: pip install streamlit pandas") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifacts" / "full-run"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

st.set_page_config(page_title="AbuseRing · risk lab", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }
    .hero { padding: 1.35rem 1.5rem; border: 1px solid #d8e1e8; border-radius: 14px;
            background: linear-gradient(120deg, #f4f8f8 0%, #eef4f7 60%, #e8f0f4 100%); }
    .eyebrow { color: #087f8c; font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { margin: .25rem 0 .4rem; color: #112d3d; font-size: 2.35rem; letter-spacing: -.045em; }
    .hero p { color: #496171; max-width: 760px; margin: 0; font-size: 1rem; }
    .callout { border-left: 4px solid #087f8c; background: #f1f8f8; padding: .8rem 1rem; border-radius: 4px; color: #294655; }
    [data-testid="stMetricValue"] { color: #112d3d; }
    </style>
    """,
    unsafe_allow_html=True,
)


def absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@st.cache_data(show_spinner=False)
def load_run(run_path: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    run_dir = Path(run_path)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    tables: dict[str, pd.DataFrame] = {}
    for mode in manifest.get("models", {}):
        table_path = run_dir / f"{mode}_thresholds.csv"
        if table_path.exists():
            tables[mode] = pd.read_csv(table_path)
    return manifest, tables


def run_pipeline(config_path: Path, output_dir: Path) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "abuse_ring_detector.cli", "run-poc",
               "--config", str(config_path), "--output-dir", str(output_dir)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=900)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0, output


# Sidebar is intentionally explicit: the Streamlit browser refresh is not a model rerun.
st.sidebar.markdown("### Run controls")
run_dir_text = st.sidebar.text_input("Results folder", str(DEFAULT_RUN_DIR.relative_to(PROJECT_ROOT)))
run_dir = absolute_path(run_dir_text)
config_text = st.sidebar.text_input("Config file", str(DEFAULT_CONFIG.relative_to(PROJECT_ROOT)))
config_path = absolute_path(config_text)
st.sidebar.caption("The dashboard reads the manifest and threshold tables in the results folder.")

if st.sidebar.button("Run full POC", type="primary", use_container_width=True):
    if not config_path.exists():
        st.sidebar.error(f"Config not found: {config_path}")
    else:
        with st.spinner("Generating data, training both models, and writing the report…"):
            try:
                succeeded, output = run_pipeline(config_path, run_dir)
            except subprocess.TimeoutExpired:
                succeeded, output = False, "The run exceeded the 15-minute UI timeout."
        if succeeded:
            load_run.clear()
            st.sidebar.success("Run complete. Dashboard refreshed.")
            st.rerun()
        else:
            st.sidebar.error("Run failed. See the details below.")
            st.code(output[-5000:] if output else "No process output.")

if not (run_dir / "run_manifest.json").exists():
    st.markdown('<div class="hero"><div class="eyebrow">Defensive risk lab · synthetic data</div><h1>AbuseRing Detector</h1><p>No completed run is loaded yet. Choose a results folder, then use <b>Run full POC</b> in the sidebar. The browser refresh button only redraws the page; it does not retrain the models.</p></div>', unsafe_allow_html=True)
    st.markdown("### Quick start")
    st.code("PYTHONPATH=src python3 -m abuse_ring_detector.cli run-poc --config configs/default.yaml --output-dir artifacts/full-run", language="bash")
    st.info("After the command finishes, set Results folder to artifacts/full-run and click Refresh page, or use the Run full POC button above.")
    st.stop()

if st.sidebar.button("Refresh page", use_container_width=True):
    load_run.clear()
    st.rerun()

summary, threshold_tables = load_run(str(run_dir))
dataset = summary.get("dataset", {})
models = summary.get("models", {})

st.markdown('<div class="hero"><div class="eyebrow">Merchant risk lab · held-out test results</div><h1>Find the network, not just the account</h1><p>Compare a behavioural baseline with graph-enhanced risk signals. Every number below comes from the selected synthetic run and its chronological test split.</p></div>', unsafe_allow_html=True)

st.markdown("### Run at a glance")
metrics = st.columns(5)
for column, label, key in zip(metrics, ["Customers", "Orders", "Returns", "Abuse orders", "Rings"], ["customers", "orders", "returns", "abuse_orders", "rings"]):
    column.metric(label, f"{dataset.get(key, 0):,}")
st.caption(f"Results: `{run_dir}`  ·  Train ends {summary.get('split', {}).get('train_end', '—')}  ·  Validation ends {summary.get('split', {}).get('validation_end', '—')}")

if not models:
    st.warning("The manifest has no model results yet. Run the POC again from the sidebar.")
    st.stop()

# Normalize model results for compact comparison tables.
model_labels = {
    "baseline": "Behavioural baseline",
    "graph": "Graph-enhanced",
    "graph_temporal": "Graph + Temporal",
    "graph_temporal_custrel": "Graph + Temp + CustRel",
    "graph_temporal_custrel_2hop": "Graph + Temp + CustRel + 2Hop",
}
comparison_rows = []
for mode, result in models.items():
    test = result.get("test", {})
    comparison_rows.append({
        "Model": model_labels.get(mode, mode.replace("_", " ").title()),
        "PR-AUC": test.get("pr_auc", 0), "ROC-AUC": test.get("roc_auc", 0),
        "Precision": test.get("precision", 0), "Recall": test.get("recall", 0),
        "F1": test.get("f1", 0), "Alerts": test.get("alerts", 0),
        "Expected loss": test.get("expected_loss", next((float(row["expected_loss"]) for row in threshold_tables.get(mode, []).to_dict("records") if float(row["threshold"]) == float(result.get("threshold", 0.5))), None)),
    })
comparison = pd.DataFrame(comparison_rows).set_index("Model")

st.markdown("### What should I look at?")
st.markdown('<div class="callout"><b>Start with PR-AUC, precision, recall, and expected loss.</b> PR-AUC is more useful than ROC-AUC for this imbalanced alerting problem. Then open Thresholds to decide how many alerts the merchant can review. A graph win is meaningful only if it survives the held-out test split without an unacceptable false-positive rate.</div>', unsafe_allow_html=True)

summary_tab, thresholds_tab, evidence_tab, guide_tab = st.tabs(["Model comparison", "Thresholds & cost", "Run details", "How to use this UI"])
with summary_tab:
    st.subheader("Held-out test comparison")
    st.dataframe(comparison.style.format({"PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Expected loss": "₹{:,.0f}"}), use_container_width=True)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Higher is better · ranking and alert quality")
        st.bar_chart(comparison[["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"]], y_label="score", height=340)
    with chart_cols[1]:
        st.caption("Lower is better · configured expected financial loss")
        if comparison["Expected loss"].notna().any():
            st.bar_chart(comparison[["Expected loss"]], y_label="₹", height=340)
        else:
            st.info("Expected loss was not present in this older run manifest.")
    for mode, result in models.items():
        test = result.get("test", {})
        label = model_labels.get(mode, mode.replace("_", " ").title())
        st.markdown(f"**{label}** — {test.get('true_positives', 0)} true positives, {test.get('false_positives', 0)} false positives, {test.get('false_negatives', 0)} missed abuse orders at threshold `{result.get('threshold', '—')}`.")

with thresholds_tab:
    st.subheader("Choose the operating point")
    st.caption("Threshold selection is based on validation in the pipeline. This table shows how each locked model behaves on the held-out test set under the configured cost assumptions.")
    for mode, table in threshold_tables.items():
        label = model_labels.get(mode, mode.replace("_", " ").title())
        st.markdown(f"#### {label}")
        display = table.set_index("threshold")
        st.dataframe(display.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "expected_loss": "₹{:,.0f}"}), use_container_width=True)
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.line_chart(display[["precision", "recall", "f1"]], y_label="score", height=260)
        with chart_cols[1]:
            st.line_chart(display[["expected_loss"]], y_label="₹ expected loss", height=260)

with evidence_tab:
    st.subheader("Run details")
    st.json({"dataset": dataset, "split": summary.get("split", {}), "limitations": summary.get("limitations", [])})
    report_path = run_dir / "report.md"
    if report_path.exists():
        with st.expander("Open generated Markdown report"):
            st.markdown(report_path.read_text())

with guide_tab:
    st.subheader("A simple path through the dashboard")
    st.markdown("""
    1. **Run controls:** use **Run full POC** when you want new synthetic data and fresh models. This can take several minutes at the default 20k/50k scale.
    2. **Model comparison:** compare the two rows on the held-out test split. Focus first on PR-AUC and expected loss; use ROC-AUC as a secondary diagnostic.
    3. **Thresholds & cost:** inspect how precision and recall change as the alert threshold moves. Lower thresholds catch more abuse but create more review work.
    4. **Run details:** confirm the dataset size, chronological cutoffs, and limitations before sharing results.

    **Do you need to run the command again?** No, not if `artifacts/full-run/run_manifest.json` already exists. The dashboard reads that completed run. Run it again only when you want a fresh dataset/model run, or when you change `configs/default.yaml`.
    """)
    st.code("PYTHONPATH=src python3 -m abuse_ring_detector.cli run-poc --config configs/default.yaml --output-dir artifacts/full-run", language="bash")
    st.info("The Streamlit browser's ↻ / Rerun action only reloads artifacts. It does not execute the CLI. Use the sidebar's Run full POC button for that.")
