"""AbuseRing Live Monitor — monitoring console, not a scenario selector.

Real-time-feel transaction stream: events are scored one at a time by the real
frozen Model F-R1 via /v1/predict; alerts and the investigation case update
automatically as the stream flows. Demo scenarios are only a traffic generator
(hidden in a collapsed expander), never detection modes.

Run with: streamlit run app/command_center.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
from api_client import ApiError, CommandCenterClient
from demo_scenarios import scenario_payloads
from monitor import (
    REVIEW_THRESHOLD,
    SCENARIOS,
    graph_dot,
    mask_customer,
    masked_orders_for_run,
    pick_active_case,
    risk_status,
    row_from_event,
    shared_entity_count,
    status_color,
    timeline_view,
)

st.set_page_config(page_title="AbuseRing Live Monitor", page_icon="◈", layout="wide")
st.markdown("""
<style>
:root { --ink:#e8edf2; --muted:#93a3b3; --panel:#141d27; --line:#2b3948; --cyan:#4fd1c5; --red:#f87171; --amber:#fbbf24; --green:#34d399; }
.stApp { background:#0c1219; color:var(--ink); }
.block-container { max-width:1240px; padding:1.2rem 2.2rem 2rem; }
h1, h2, h3 { letter-spacing:-.02em; }
.badge { border-radius:999px; padding:3px 11px; font-size:11.5px; font-weight:700; margin-right:6px; }
.badge.demo  { color:#07141b; background:var(--cyan); }
.badge.shadow{ color:#fbbf24; background:#fbbf241a; }
.badge.off   { color:#f87171; background:#f871711a; }
.streamdot { font-size:13px; font-weight:700; }
.alertbanner { background:linear-gradient(90deg,#3b1216,#221014); border:1px solid var(--red); border-radius:10px; padding:10px 18px; margin:8px 0; }
.alertbanner .t { color:var(--red); font-weight:800; font-size:15px; letter-spacing:.05em; }
.alertbanner .s { color:#f3c1c1; font-size:13px; margin-top:2px; }
.legend-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
[data-testid="stMetricValue"] { color:#f7fafc; font-size:1.5rem; }
[data-testid="stMetricLabel"] { color:var(--muted); font-size:.82rem; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:10px; }
.stButton > button { border-radius:9px; font-weight:700; }
/* Hide Streamlit chrome: hamburger menu, Deploy button, "made with Streamlit", running-man */
#MainMenu, header[data-testid="stHeader"] button[kind="header"], footer, [data-testid="stStatusWidget"],
header[data-testid="stHeader"] [data-testid="stLogo"], header[data-testid="stHeader"] a[title*="streamlit" i],
header[data-testid="stHeader"] div[data-testid="stDecoration"] { visibility:visible; opacity:0; height:0; pointer-events:none; }
header[data-testid="stHeader"] { background:transparent; }
/* Brand mark in header */
.brandmark { display:inline-grid; place-items:center; width:44px; height:44px; border-radius:12px;
  background:radial-gradient(circle at 32% 28%, #1d3a44, #0e1b24 70%); border:1px solid #2b4a56; margin-right:14px; }
.brandmark .ring { width:22px; height:22px; border:3px solid var(--cyan); border-radius:50%;
  box-shadow:0 0 0 3px #4fd1c526, inset 0 0 0 1px #4fd1c55c; }
.brandmark .node { position:absolute; }
footer { visibility:hidden; }
h3 { margin-top:.6rem; margin-bottom:.35rem; }
</style>
""", unsafe_allow_html=True)

client = CommandCenterClient()

TICK_PAUSE = 0.9          # seconds between demo events (0.7–1.5s per spec)
MAX_FEED_ROWS = 9         # compact feed for recording


def money(v: float) -> str:
    return f"₹{v:,.0f}"


# ------------------------------------------------------------------ header
st.markdown('<div style="display:flex;align-items:baseline;gap:16px;flex-wrap:wrap">'
            '<h1 style="margin:0;font-size:1.9rem">AbuseRing</h1>'
            '<span style="color:#93a3b3;font-size:1.02rem">Fraud systems inspect transactions. '
            '<b style="color:#e8edf2">AbuseRing investigates networks.</b></span>'
            '<span style="margin-left:auto">'
            '<span class="badge demo">DEMO / SYNTHETIC</span>'
            '<span class="badge shadow">SHADOW MODE</span>'
            '<span class="badge off">ENFORCEMENT OFF</span>'
            '</span></div>', unsafe_allow_html=True)
st.caption("AbuseRing watches every transaction automatically — no pattern selection, no manual triage. "
           "Risk below 0.30 is Normal, 0.30–0.50 is Watching, and a score at or above the 0.50 review threshold raises a network risk signal.")

running = st.session_state.get("running", False)
st.markdown(f'<div class="streamdot" style="color:{"#34d399" if running else "#93a3b3"}">'
            f'{"● DEMO STREAM RUNNING" if running else "○ STREAM IDLE"}</div>', unsafe_allow_html=True)

# ------------------------------------------------- demo traffic generator
with st.expander("Demo Traffic Generator (synthetic replay — not a detection mode)", expanded=running):
    gc1, gc2, gc3, gc4 = st.columns([2, 1, 1, 1])
    with gc1:
        scenario = st.selectbox("Traffic scenario", SCENARIOS, index=0,
                                disabled=running, label_visibility="collapsed")
    with gc2:
        start_clicked = st.button("▶ START DEMO TRAFFIC", type="primary", disabled=running, use_container_width=True)
    with gc3:
        stop_clicked = st.button("■ STOP", disabled=not running, use_container_width=True)
    with gc4:
        reset_clicked = st.button("⟲ RESET", use_container_width=True)

    st.caption("Generates synthetic transactions through the real /v1/predict pipeline. "
               "Detection is fully automatic — the scenario only decides what traffic looks like.")

if stop_clicked:
    st.session_state["running"] = False
    st.rerun()

if reset_clicked:
    st.session_state.clear()
    st.session_state["run_n"] = 0
    st.rerun()

if start_clicked:
    st.session_state["running"] = True
    st.session_state["pending"] = scenario_payloads(scenario, f"run{st.session_state.get('run_n', 0) + 1}")
    st.session_state["run_orders"] = {p["order_id"] for p in st.session_state["pending"]}
    st.session_state["previous"] = []
    st.session_state["masked_orders"] = set()
    st.rerun()

# ------------------------------------------------------ tick: one event per rerun
if st.session_state.get("running") and st.session_state.get("pending"):
    payload = st.session_state["pending"].pop(0)
    index = st.session_state.get("sent", 0) + 1
    try:
        response = client.predict(payload)
    except ApiError as exc:
        st.session_state["running"] = False
        st.error(f"Stream stopped safely: {exc}")
    else:
        st.session_state["previous"].append(payload)
        st.session_state.setdefault("events", []).append(
            row_from_event(index, payload, response, st.session_state["previous"][:-1])
        )
        st.session_state["sent"] = index
        # Refresh the raw→masked order mapping from live alerts after each event.
        try:
            st.session_state["masked_orders"] = masked_orders_for_run(
                client.alerts().get("items", []), st.session_state.get("run_orders", set())
            )
        except ApiError:
            pass
        if not st.session_state["pending"]:
            st.session_state["running"] = False
            st.session_state["stream_done"] = True
    time.sleep(TICK_PAUSE)
    st.rerun()

# ------------------------------------------------------ transaction stream
st.markdown("### Transaction Stream")
events = st.session_state.get("events", [])

if not events:
    st.info("Waiting for transactions… start the demo traffic generator to feed the live pipeline.")
else:
    # Alert banner before the first alert row.
    shown_banner = st.session_state.get("banner_shown", False)
    header = ("Time | Customer | Amount | Risk | Status | Observation")
    feed_html = ['<div style="font-size:12px;color:#93a3b3;letter-spacing:.06em;'
                 'display:grid;grid-template-columns:90px 130px 90px 70px 110px 1fr;gap:0 12px;'
                 'padding:4px 14px;">' +
                 "".join(f"<span>{h}</span>" for h in header.split(" | ")) + "</div>"]
    for row in events[-MAX_FEED_ROWS:]:
        color = status_color(row["status"])
        weight = "800" if row["status"] == "ALERT" else "600"
        bg = "#2a1518" if row["status"] == "ALERT" else ("#241f10" if row["status"] == "WATCHING" else "var(--panel)")
        feed_html.append(
            f'<div style="display:grid;grid-template-columns:90px 130px 90px 70px 110px 1fr;gap:0 12px;'
            f'align-items:center;background:{bg};border:1px solid {"#f8717155" if row["status"] == "ALERT" else "var(--line)"};'
            f'border-radius:9px;padding:7px 14px;margin-bottom:5px;font-size:14.5px;">'
            f'<span style="color:#93a3b3">{row["time"]}</span>'
            f'<span>{row["customer"]}</span>'
            f'<span>{money(row["amount"])}</span>'
            f'<span style="font-weight:800;color:{color}">{row["risk"]:.2f}</span>'
            f'<span style="font-weight:{weight};color:{color}">{row["status"]}</span>'
            f'<span style="color:#93a3b3;font-size:13px">{row["observation"]}</span>'
            f'</div>'
        )
    st.markdown("".join(feed_html), unsafe_allow_html=True)
    if len(events) > MAX_FEED_ROWS:
        st.caption(f"Showing latest {MAX_FEED_ROWS} of {len(events)} scored events.")

    first_alert = next((r for r in events if r["status"] == "ALERT"), None)
    if first_alert and not shown_banner:
        st.session_state["banner_shown"] = True
    if first_alert:
        st.markdown(
            f'<div class="alertbanner"><div class="t">⚠ NETWORK RISK SIGNAL</div>'
            f'<div class="s">Risk score {first_alert["risk"]:.2f} reached the review threshold {REVIEW_THRESHOLD:.2f}. '
            f'Review signal generated — not confirmed fraud. Shadow mode: no customer action.</div></div>',
            unsafe_allow_html=True,
        )

# ------------------------------------------------------ active investigation
st.markdown("### Active Investigation")

active_case = None
case_bundle = None
try:
    cases = client.cases().get("items", [])
    active_case = pick_active_case(cases, st.session_state.get("masked_orders", set()) or st.session_state.get("run_orders", set()))
    if active_case:
        case_id = active_case["case_id"]
        case_bundle = {
            "case": client.case(case_id),
            "graph": client.graph(case_id),
            "timeline": client.timeline(case_id),
            "evidence": client.evidence(case_id),
        }
except ApiError as exc:
    st.error(f"Backend unavailable: {exc}")

if not case_bundle:
    if st.session_state.get("stream_done"):
        st.info("Stream complete. The frozen R1 pipeline returned no qualifying case for this traffic — reported as returned.")
    else:
        st.caption("No active investigation case yet.")
else:
    case = case_bundle["case"]
    graph = case_bundle["graph"]
    cols = st.columns(5)
    cols[0].metric("Risk Score", f'{case.get("risk_score", 0):.2f}')
    cols[1].metric("Connected Accounts", len(case.get("related_customers", []) or []))
    cols[2].metric("Alerts", case.get("alert_count", 0))
    cols[3].metric("Shared Entities", shared_entity_count(graph))
    cols[4].metric("Observed Exposure", money(case.get("estimated_exposure", 0)))

    st.markdown("**Observed evidence associated with elevated risk**")
    seen, shown = set(), 0
    for item in case_bundle["evidence"].get("items", []):
        desc = item.get("description", "")
        if desc in seen or shown >= 5:
            continue
        seen.add(desc)
        shown += 1
        st.markdown(f'- {desc} `({item.get("value", "")})`')

    with st.expander("Technical details"):
        st.json({"case_id": case.get("case_id"), "status": case.get("status"),
                 "evidence_count": len(case_bundle["evidence"].get("items", [])),
                 "history": case.get("history", [])[-6:]})

    # ------------------------------------------------------ network graph
    st.markdown("### Network")
    gl, gr = st.columns([0.62, 0.38])
    with gl:
        try:
            st.graphviz_chart(graph_dot(graph), use_container_width=True)
        except Exception as exc:
            st.warning(f"Graph rendering unavailable ({exc}).")
    with gr:
        st.caption("● Customer &nbsp;·&nbsp; ◆ Shared device / address / IP / payment")
        st.caption("Supposedly separate customers, one network. Orders hidden for clarity; full relationship data is in the backend.")

    # ------------------------------------------------------ timeline
    st.markdown("### Timeline")
    for row in timeline_view(case_bundle["timeline"].get("items", [])):
        st.markdown(f'`{row["time"]}`  {row["event"]}')

# ------------------------------------------------------ compact alerts panel
try:
    alerts = client.alerts().get("items", [])[:5]
except ApiError:
    alerts = []
if alerts:
    with st.expander(f"Active Alerts ({len(alerts)} shown)"):
        for alert in alerts:
            risk = float(alert.get("risk_score", 0) or 0)
            color = status_color(risk_status(risk))
            ts = str(alert.get("created_at", ""))[11:19]
            st.markdown(f'`{ts}`  {mask_customer(str(alert.get("customer_id", "")))} '
                        f'<span style="color:{color};font-weight:800">{risk:.2f}</span> '
                        f'<span style="color:#93a3b3">{str(alert.get("order_id", ""))[-6:]}</span>', unsafe_allow_html=True)

# ------------------------------------------------------ system status
st.divider()
cols = st.columns([1, 2])
with cols[0]:
    st.caption("Shadow Mode: **ON** · Enforcement: **OFF** · Threshold 0.50 locked · Model F-R1")
with cols[1]:
    try:
        ready = client.readiness()
        dots = "".join(
            f'<span class="legend-dot" style="background:{"#34d399" if ok else "#f87171"}"></span>{label} &nbsp; '
            for label, ok in (
                ("API", ready.get("status") == "ready"),
                ("Redis", bool(ready.get("state_backend_healthy"))),
                ("Model F-R1", bool(ready.get("model_loaded"))),
            )
        )
        st.markdown(dots, unsafe_allow_html=True)
    except ApiError:
        st.markdown('<span class="legend-dot" style="background:#f87171"></span>API offline', unsafe_allow_html=True)
