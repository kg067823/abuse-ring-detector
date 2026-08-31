"""AbuseRing Command Center — API-backed investigator console.

Run with: streamlit run app/command_center.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
from api_client import ApiError, CommandCenterClient
from command_center_helpers import aggregate_cases, filter_items, graph_counts, timeline_sorted

st.set_page_config(page_title="AbuseRing Command Center", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#e8edf2; --muted:#91a0af; --panel:#151d27; --panel2:#1b2633; --line:#2b3948; --cyan:#4fd1c5; --red:#f87171; --amber:#fbbf24; --green:#34d399; }
.stApp { background: #0c1219; color: var(--ink); }
.block-container { max-width: 1480px; padding: 2rem 3rem 4rem; }
.hero { background: linear-gradient(135deg,#172431,#101820); border:1px solid #304050; border-radius:18px; padding:28px 32px; margin-bottom:22px; }
.eyebrow { color:var(--cyan); font-size:11px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { color:#f7fafc; letter-spacing:-.045em; margin:.3rem 0 .45rem; font-size:2.5rem; }
.hero p,.muted { color:var(--muted); }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; }
.badge { border-radius:999px; padding:3px 9px; font-size:11px; font-weight:700; }
.demo { color:#07141b; background:var(--cyan); border-radius:999px; padding:5px 10px; font-size:11px; font-weight:800; }
[data-testid="stMetricValue"] { color:#f7fafc; }
</style>
""", unsafe_allow_html=True)

client = CommandCenterClient()


def money(value: float) -> str:
    return f"₹{value:,.0f}"


def badge(value: str) -> str:
    colors = {"CRITICAL": "#f87171", "HIGH": "#fb923c", "MEDIUM": "#fbbf24", "LOW": "#34d399", "NEW": "#4fd1c5", "IN_REVIEW": "#60a5fa", "ESCALATED": "#f87171"}
    color = colors.get(value, "#91a0af")
    return f'<span class="badge" style="background:{color}22;color:{color}">{value.replace("_", " ")}</span>'


def fetch(path: str, fallback):
    try:
        return getattr(client, path)()
    except ApiError as exc:
        st.error(str(exc))
        return fallback


st.sidebar.markdown("## ◈ AbuseRing")
st.sidebar.caption("Command Center · investigator console")
page = st.sidebar.radio("Navigate", ["Overview", "Alert Queue", "Investigation Cases", "Case Workspace", "Network Explorer", "System Health", "Demo Mode"])
st.sidebar.markdown('<span class="demo">DEMO DATA / SYNTHETIC</span>', unsafe_allow_html=True)
st.sidebar.caption("R1 shadow mode · no customer enforcement")

if page == "System Health":
    st.markdown('<div class="hero"><div class="eyebrow">Operations / model integrity</div><h1>System Health</h1><p>Runtime signals for the frozen Model F-R1 shadow service.</p></div>', unsafe_allow_html=True)
    try:
        health, ready, live, metrics = client.health(), client.readiness(), client.liveness(), client.metrics()
        cols = st.columns(4)
        cols[0].metric("API", health.get("status", "unknown").upper())
        cols[1].metric("Readiness", ready.get("status", "unknown").upper())
        cols[2].metric("Features", ready.get("feature_count", "—"))
        cols[3].metric("Threshold", ready.get("threshold", "—"))
        st.success("SHADOW MODE · NO CUSTOMER ENFORCEMENT")
        st.json({"health": health, "readiness": ready, "liveness": live})
        with st.expander("Prometheus metrics"):
            st.code(metrics if isinstance(metrics, str) else json.dumps(metrics, indent=2))
    except ApiError as exc:
        st.error(str(exc)); st.button("Retry")
    st.stop()

if page == "Demo Mode":
    st.markdown('<div class="hero"><div class="eyebrow">Demo / synthetic replay</div><h1>Build the network</h1><p>Replay coordinated scenarios through the real R1 API. No case objects are hard-coded in the UI.</p></div>', unsafe_allow_html=True)
    st.warning("DEMO / SYNTHETIC DATA — these records are not live production evidence.")
    scenario = st.selectbox("Scenario", ["Shared-device ring", "Shared-address ring", "Mixed multi-entity", "Behavioral coordination", "Legitimate high-connectivity"])
    if st.button("Run Abuse Scenario", type="primary"):
        st.info("Use `scratch/run_investigator_demo.py` to replay this scenario through /v1/predict, then refresh this page. The frontend never bypasses Model F-R1.")
        st.code(f".venv/bin/python scratch/run_investigator_demo.py --scenario '{scenario}'", language="bash")
    st.stop()

try:
    cases_payload = client.cases()
    alerts_payload = client.alerts()
    cases = cases_payload.get("items", [])
    alerts = alerts_payload.get("items", [])
except ApiError as exc:
    st.markdown('<div class="hero"><div class="eyebrow">Connection required</div><h1>Command Center unavailable</h1><p>Start the API and Redis stack, then retry.</p></div>', unsafe_allow_html=True)
    st.error(str(exc)); st.code("ADMIN_KILL_SWITCH_TOKEN=demo-secret docker compose up --build", language="bash"); st.stop()

summary = aggregate_cases(cases)

if page == "Overview":
    st.markdown('<div class="hero"><div class="eyebrow">Coordinated abuse intelligence</div><h1>Detect the network, not just the account.</h1><p>Turn shadow alerts into explainable investigation cases while the frozen R1 model remains non-enforcing.</p></div>', unsafe_allow_html=True)
    st.markdown('<span class="demo">DEMO DATA / SYNTHETIC</span> &nbsp; <span class="muted">Model F-R1 · isotonic calibration · threshold 0.50</span>', unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("Open cases", summary["open"])
    cols[1].metric("Critical", summary["critical"])
    cols[2].metric("High-risk alerts", len(alerts))
    cols[3].metric("Estimated exposure", money(summary["exposure"]))
    cols[4].metric("Average case risk", f'{summary["avg_risk"]:.2f}')
    left, right = st.columns(2)
    with left:
        st.markdown("### Cases by severity")
        st.bar_chart(summary["by_severity"])
    with right:
        st.markdown("### Cases by status")
        st.bar_chart(summary["by_status"])
    st.markdown("### What is active now")
    st.dataframe(cases[:10], use_container_width=True, hide_index=True)

elif page == "Alert Queue":
    st.markdown('<div class="hero"><div class="eyebrow">Triage / alert queue</div><h1>High-risk signals</h1><p>Every row is a shadow alert from the R1 backend. Select a case to investigate; no customer action is taken.</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,1,1])
    query = c1.text_input("Search masked ID or evidence")
    severity = c2.selectbox("Severity", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    min_risk = c3.slider("Minimum risk", 0.0, 1.0, 0.5, 0.01)
    filtered = filter_items(alerts, query, severity, "ALL")
    filtered = [row for row in filtered if row.get("risk_score", 0) >= min_risk]
    st.caption(f"{len(filtered)} alerts · DEMO / SYNTHETIC")
    for alert in filtered:
        with st.container(border=True):
            cols = st.columns([1,1,1,2,1])
            cols[0].markdown(badge("HIGH" if alert.get("risk_score",0) < .9 else "CRITICAL"), unsafe_allow_html=True)
            cols[1].metric("Risk", f'{alert.get("risk_score",0):.3f}')
            cols[2].caption("Masked order"); cols[2].write(alert.get("order_id", "—"))
            cols[3].write(alert.get("evidence", [{}])[0].get("description", "Observed evidence associated with elevated risk.") if alert.get("evidence") else "Observed evidence associated with elevated risk.")
            cols[4].caption(alert.get("created_at", "—"))

elif page == "Investigation Cases":
    st.markdown('<div class="hero"><div class="eyebrow">Investigation queue</div><h1>Cases</h1><p>Deterministic cases consolidated from observable graph relationships.</p></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,1,2]); status_filter=c1.selectbox("Status", ["ALL","NEW","IN_REVIEW","ESCALATED","CONFIRMED_ABUSE","LEGITIMATE","CLOSED"]); severity_filter=c2.selectbox("Severity", ["ALL","MEDIUM","HIGH","CRITICAL"]); search=c3.text_input("Search cases")
    rows=filter_items(cases, search, severity_filter, status_filter)
    for case in rows:
        with st.container(border=True):
            cols=st.columns([1.2,1,1,1,1,2]); cols[0].markdown(badge(case.get("severity","—")), unsafe_allow_html=True); cols[1].markdown(badge(case.get("status","—")), unsafe_allow_html=True); cols[2].metric("Risk",f'{case.get("risk_score",0):.3f}'); cols[3].metric("Alerts",case.get("alert_count",0)); cols[4].metric("Exposure",money(case.get("estimated_exposure",0))); cols[5].write(case.get("case_id","—"))

elif page in {"Case Workspace", "Network Explorer"}:
    st.markdown('<div class="hero"><div class="eyebrow">Investigator workspace</div><h1>Understand the ring.</h1><p>Observed evidence associated with elevated risk — never causal attribution.</p></div>', unsafe_allow_html=True)
    if not cases: st.info("No cases found. Run the deterministic demo replay to create cases."); st.stop()
    selected = st.selectbox("Case", [case.get("case_id") for case in cases])
    try:
        case = client.case(selected); evidence = client.evidence(selected); timeline = client.timeline(selected); graph = client.graph(selected)
    except ApiError as exc: st.error(str(exc)); st.stop()
    top=st.columns(5); top[0].markdown(badge(case.get("severity","—")),unsafe_allow_html=True); top[1].markdown(badge(case.get("status","—")),unsafe_allow_html=True); top[2].metric("Risk",f'{case.get("risk_score",0):.3f}'); top[3].metric("Exposure",money(case.get("estimated_exposure",0))); top[4].caption("Model"); top[4].write(case.get("model_version","model_f_r1"))
    st.info("Observed evidence associated with elevated risk. These signals are not causal proof.")
    if page == "Network Explorer":
        st.subheader("Network explorer"); st.caption(f'{graph_counts(graph)["nodes"]} nodes · {graph_counts(graph)["edges"]} edges')
        st.graphviz_chart("graph {\n" + "\n".join(f'"{edge["source"]}" -- "{edge["target"]}" [label="{edge["relationship"]}"];' for edge in graph.get("edges", [])) + "\n}")
        st.dataframe(graph.get("nodes", []), use_container_width=True, hide_index=True)
    else:
        left,right=st.columns([1.2,1])
        with left:
            st.subheader("Why this case was surfaced")
            for item in evidence.get("items", []):
                with st.container(border=True): st.markdown(f"**{item.get('description','Observed signal')}**"); st.caption(f"Value: {item.get('value','—')} · Window: {item.get('window','—')} · Provenance: {item.get('provenance','observed_signal')}")
            st.subheader("Timeline")
            for event in timeline_sorted(timeline.get("items", [])): st.write(f'**{event.get("timestamp","—")}** · {event.get("description","—")}')
        with right:
            st.subheader("Network context"); st.metric("Nodes",graph_counts(graph)["nodes"]); st.metric("Edges",graph_counts(graph)["edges"]); st.dataframe(graph.get("edges", []), use_container_width=True, hide_index=True)
            st.subheader("Analyst action")
            st.caption("Mutations require ABUSERING_ADMIN_TOKEN; no customer enforcement is enabled.")
            st.write("Status mutation is available through the authenticated backend API.")
            with st.expander("Technical details"): st.json(case.get("history", []))
