import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import streamlit as st
import pandas as pd
import joblib
import shap

from explain import explain_session, FEATURE_EXPLANATIONS
from fallback_rules import score_offline, CachedUserBaseline, reconciliation_note, Action
import db as db_module

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(REPO_ROOT, "data", "ato.db")
db_module.init_db(DB_PATH)


# ---------------------------------------------------------------------------
# Page config — force LIGHT theme so Streamlit never injects white text
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PulseGuard — ICSC 2026",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    st._config.set_option("theme.base", "light")
    st._config.set_option("theme.backgroundColor", "#FBFCFE")
    st._config.set_option("theme.secondaryBackgroundColor", "#F1F5F9")
    st._config.set_option("theme.textColor", "#0B1F33")
    st._config.set_option("theme.primaryColor", "#0E4E7A")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------

LOGO_SVG = (
    '<svg width="52" height="56" viewBox="0 0 100 112" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="PulseGuard">'
    '<defs><linearGradient id="pgFill" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0%" stop-color="#0B2A4A"/><stop offset="100%" stop-color="#0E4E7A"/>'
    '</linearGradient></defs>'
    '<path d="M50 3 L93 19 C95 20 96 22 96 24 L96 54 C96 84 76 102 51 109 C50 109.3 49 109.3 48 109 C23 102 4 84 4 54 L4 24 C4 22 5 20 7 19 Z" fill="url(#pgFill)"/>'
    '<path d="M50 12 L86 25 L86 54 C86 79 69 95 50 101 C31 95 14 79 14 54 L14 25 Z" fill="none" stroke="#10B981" stroke-width="2.2" opacity="0.85"/>'
    '<polyline points="20,58 34,58 40,50 46,58 56,58 62,32 68,80 74,58 82,58" fill="none" stroke="#FFFFFF" stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="62" cy="32" r="4" fill="#10B981"/>'
    '</svg>'
)

LOGO_SVG_SMALL = LOGO_SVG.replace('width="52" height="56"', 'width="30" height="32"')


# ---------------------------------------------------------------------------
# Tier styles
# ---------------------------------------------------------------------------

TIER_STYLE = {
    "allow":   dict(color="#059669", bg="#ECFDF5", icon="✓", label="Allowed"),
    "monitor": dict(color="#D97706", bg="#FFFBEB", icon="◐", label="Monitoring"),
    "block":   dict(color="#DC2626", bg="#FEF2F2", icon="✕", label="Blocked"),
}


# ---------------------------------------------------------------------------
# Design system — aggressive color enforcement so nothing stays hidden
# ---------------------------------------------------------------------------

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

/* ---------- Global resets: force visible text on light surfaces ---------- */
.stApp, .stApp * { color: #0B1F33; }
.stApp { background: #FBFCFE !important; font-family: 'Inter', system-ui, sans-serif; font-feature-settings: 'cv02','cv03','cv04','ss01'; }
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #FBFCFE !important; }
[data-testid="stHeader"] { display: none !important; }

/* Markdown/text elements — never white */
.stApp p, .stApp span, .stApp div, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp small, .stApp strong, .stApp em, .stApp caption { color: #0B1F33; }

/* Widget labels (radio, selectbox, checkbox, slider) */
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp .stRadio label, .stApp .stSelectbox label,
.stApp .stCheckbox label, .stApp .stSlider label { color: #334155 !important; font-weight: 500; }

/* Radio / selectbox selected values */
.stApp [data-baseweb="select"] * { color: #0B1F33 !important; }
.stApp [data-baseweb="radio"] * { color: #0B1F33 !important; }

/* Captions */
.stApp [data-testid="stCaptionContainer"], .stApp small, .stApp .stCaption { color: #94A3B8 !important; }

/* Expander */
.stApp [data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #E8EDF3; border-radius: 12px; }
.stApp [data-testid="stExpander"] summary { color: #0B1F33 !important; font-weight: 600; }

/* Divider */
.stApp hr { border-color: #E8EDF3 !important; }

/* Toggle */
.stApp [data-baseweb="checkbox"] div { color: #0B1F33 !important; }

/* Sidebar — WHITE surface, dark text */
section[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E8EDF3; }
section[data-testid="stSidebar"] * { color: #0B1F33; }
section[data-testid="stSidebar"] h4 { color: #0B1F33 !important; font-weight: 700; letter-spacing: -0.01em; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] small { color: #64748B !important; }

/* ---------- Design tokens ---------- */
:root {
  --pg-ink:#0B1F33; --pg-navy:#0B2A4A; --pg-teal:#0E4E7A;
  --pg-emerald:#10B981; --pg-sky:#0EA5E9;
  --pg-safe:#059669; --pg-safe-bg:#ECFDF5;
  --pg-watch:#D97706; --pg-watch-bg:#FFFBEB;
  --pg-threat:#DC2626; --pg-threat-bg:#FEF2F2;
  --pg-bg:#F8FAFC; --pg-surface:#FFFFFF; --pg-muted:#64748B;
  --pg-border:#E8EDF3; --pg-shadow:0 1px 2px rgba(11,31,51,.04), 0 8px 24px rgba(11,31,51,.06);
}

/* ---------- Component classes ---------- */
.pg-header { display:flex; align-items:center; gap:18px; padding:10px 0 16px; }
.pg-title { font-size:2.1rem; font-weight:800; margin:0; letter-spacing:-0.028em; color:var(--pg-ink) !important; }
.pg-tagline { color:var(--pg-muted) !important; font-size:.98rem; margin:3px 0 0; font-weight:500; letter-spacing:-0.005em; }

.pg-trust { display:flex; gap:8px; flex-wrap:wrap; margin:2px 0 8px; }
.pg-chip { display:inline-flex; align-items:center; gap:7px; font-size:.76rem; font-weight:600; letter-spacing:.01em; padding:5px 11px; border-radius:9999px; background:#FFFFFF; border:1px solid var(--pg-border); color:#475569 !important; }
.pg-chip .dot { width:7px; height:7px; border-radius:50%; background:var(--pg-emerald); box-shadow:0 0 0 3px rgba(16,185,129,.15); }

.pg-card { border-radius:16px; padding:22px 24px; margin-top:8px; background:#FFFFFF; border:1px solid var(--pg-border); box-shadow:var(--pg-shadow); }
.pg-card * { color: #0B1F33; }
.pg-card--result { padding:26px 28px; }

.pg-tier-badge { display:inline-flex; align-items:center; gap:9px; font-size:.8rem; font-weight:700; letter-spacing:.06em; padding:6px 14px; border-radius:9999px; text-transform:uppercase; }

.pg-risk-number { font-size:3.2rem; font-weight:800; margin:12px 0 0; letter-spacing:-0.035em; line-height:1; font-variant-numeric:tabular-nums; }
.pg-subtitle { color:var(--pg-muted) !important; font-size:.9rem; margin:8px 0 0; font-weight:500; }

.pg-mono { font-family:'JetBrains Mono',monospace; font-size:.82rem; font-weight:500; color:#475569 !important; background:var(--pg-bg); padding:3px 9px; border-radius:6px; border:1px solid var(--pg-border); }

.pg-stat-row { display:flex; gap:12px; margin-top:20px; }
.pg-stat { flex:1; background:#F8FAFC; border-radius:12px; padding:12px 14px; border:1px solid var(--pg-border); }
.pg-stat-label { font-size:.68rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:#94A3B8 !important; margin-bottom:5px; }
.pg-stat-value { font-size:1.1rem; font-weight:700; color:#0B1F33 !important; font-variant-numeric:tabular-nums; }

.pg-reason { display:flex; gap:10px; padding:11px 0; border-bottom:1px solid var(--pg-border); font-size:.94rem; color:#0B1F33 !important; line-height:1.5; }
.pg-reason:last-child { border-bottom:none; }
.pg-reason-marker { flex:0 0 20px; height:20px; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:.7rem; font-weight:700; background:var(--pg-watch-bg); color:var(--pg-watch) !important; }

.pg-section-label { font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#94A3B8 !important; margin:0 0 12px 0; }

.pg-kv-row { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--pg-border); }
.pg-kv-row:last-child { border-bottom:none; }
.pg-kv-k { color:#64748B !important; font-size:.85rem; font-weight:500; }
.pg-kv-v { color:#0B1F33 !important; font-size:.85rem; font-weight:600; font-variant-numeric:tabular-nums; }

.pg-sidebar-brand { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.pg-sidebar-brand span { font-weight:800; font-size:1.1rem; color:#0B1F33 !important; letter-spacing:-0.02em; }

/* Streamlit metric widget */
div[data-testid="stMetric"] { background:#F8FAFC; border-radius:12px; padding:12px 14px; border:1px solid var(--pg-border); }
div[data-testid="stMetric"] label p { font-size:.68rem !important; font-weight:600 !important; letter-spacing:.06em !important; text-transform:uppercase !important; color:#94A3B8 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; font-weight:800; letter-spacing:-0.02em; color:#0B1F33 !important; }

/* Alerts (st.success/warning/info/error) — readable on light */
.stApp [data-testid="stAlert"] { border-radius:12px; }
.stApp [data-testid="stAlert"] * { color: #0B1F33 !important; }

/* Buttons */
.stApp button { color: #0B1F33; }
.stApp button[kind="primary"] { color: #FFFFFF !important; }
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model + data
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    saved = joblib.load(os.path.join(REPO_ROOT, "models", "ato_model.pkl"))
    thresholds = dict(block_threshold=saved["block_threshold"],
                       monitor_threshold=saved["monitor_threshold"])
    return saved["model"], saved["feature_cols"], thresholds

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(REPO_ROOT, "data", "processed", "features.csv"),
                        parse_dates=["timestamp"])

@st.cache_data
def load_raw():
    return pd.read_csv(os.path.join(REPO_ROOT, "data", "raw", "sessions.csv"),
                        parse_dates=["timestamp"])

model, feature_cols, thresholds = load_model()
feat_df = load_data()
raw_df = load_raw()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

header_html = (
    '<div class="pg-header">'
    f'{LOGO_SVG}'
    '<div>'
    '<p class="pg-title">PulseGuard</p>'
    "<p class=\"pg-tagline\">Learns your rhythm. Notices when it's not you.</p>"
    '</div>'
    '</div>'
    '<div class="pg-trust">'
    '<span class="pg-chip"><span class="dot"></span>ML engine online</span>'
    '<span class="pg-chip">⚡ p99 latency · 1.86 ms</span>'
    '<span class="pg-chip">🔒 256-bit encrypted</span>'
    '<span class="pg-chip">🧪 Synthetic data only</span>'
    '</div>'
)
st.markdown(header_html, unsafe_allow_html=True)
st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

sidebar_brand = (
    '<div class="pg-sidebar-brand">'
    f'<div style="width:30px;">{LOGO_SVG_SMALL}</div>'
    '<span>PulseGuard</span>'
    '</div>'
)
st.sidebar.markdown(sidebar_brand, unsafe_allow_html=True)

st.sidebar.markdown("#### System status")
offline_mode = st.sidebar.toggle("Simulate system offline (network/power cut)", value=False)

if offline_mode:
    offline_card = (
        '<div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:12px; padding:14px 16px; margin-top:8px;">'
        '<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
        '<span style="width:8px; height:8px; border-radius:50%; background:#D97706;"></span>'
        '<span style="font-weight:700; font-size:.8rem; letter-spacing:.05em; text-transform:uppercase; color:#92400E !important;">Resilient Mode</span>'
        '</div>'
        '<p style="font-size:.85rem; line-height:1.5; color:#78350F !important; margin:0;">ML engine offline. Running on-device rule fallback — decisions continue, audit trail preserved, reconciliation queued.</p>'
        '</div>'
    )
    st.sidebar.markdown(offline_card, unsafe_allow_html=True)
else:
    online_card = (
        '<div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:12px; padding:14px 16px; margin-top:8px;">'
        '<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
        '<span style="width:8px; height:8px; border-radius:50%; background:#10B981;"></span>'
        '<span style="font-weight:700; font-size:.8rem; letter-spacing:.05em; text-transform:uppercase; color:#065F46 !important;">Full Engine Online</span>'
        '</div>'
        '<p style="font-size:.85rem; line-height:1.5; color:#064E3B !important; margin:0;">XGBoost + SHAP. Sub-2 ms p99 scoring latency.</p>'
        '</div>'
    )
    st.sidebar.markdown(online_card, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("#### Proof, not promises")

proof_card = (
    '<div style="background:#F8FAFC; border-radius:12px; padding:14px; border:1px solid #E8EDF3;">'
    '<div style="font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#94A3B8 !important; margin-bottom:10px;">On time-based holdout</div>'
    '<div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">'
    '<div><div style="font-size:1.35rem; font-weight:800; color:#059669 !important; font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">100%</div><div style="font-size:.7rem; color:#64748B !important; font-weight:500;">Attacks tracked</div></div>'
    '<div><div style="font-size:1.35rem; font-weight:800; color:#0B1F33 !important; font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">0.98%</div><div style="font-size:.7rem; color:#64748B !important; font-weight:500;">False stops</div></div>'
    '<div><div style="font-size:1.35rem; font-weight:800; color:#0B1F33 !important; font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">1.86<span style="font-size:.8rem; font-weight:600; color:#64748B !important;">ms</span></div><div style="font-size:.7rem; color:#64748B !important; font-weight:500;">p99 latency</div></div>'
    '<div><div style="font-size:1.35rem; font-weight:800; color:#0B1F33 !important; font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">4</div><div style="font-size:.7rem; color:#64748B !important; font-weight:500;">Attack types</div></div>'
    '</div>'
    '</div>'
)
st.sidebar.markdown(proof_card, unsafe_allow_html=True)
st.sidebar.caption("Time-based split · full methodology in technical write-up")


# ---------------------------------------------------------------------------
# Session picker
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<p class="pg-section-label">01 — Choose a session to score</p>', unsafe_allow_html=True)

    mode = st.radio("Source", ["Pick an example session", "Enter a custom session"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "Pick an example session":
        attack_type_filter = st.selectbox(
            "Filter by type",
            ["none", "credential_theft", "sim_swap", "patient_low_and_slow", "social_engineering"],
        )
        subset = feat_df[feat_df["attack_type"] == attack_type_filter]
        session_id = st.selectbox("Session ID", subset["session_id"].head(20).tolist())
        feature_row = feat_df[feat_df["session_id"] == session_id].iloc[0]
        raw_row = raw_df[raw_df["session_id"] == session_id].iloc[0]

        details = {
            "Channel": str(raw_row["channel"]).title(),
            "Device": "New / unrecognized" if bool(raw_row["is_new_device"]) else "Recognized",
            "Amount": f"₦{float(raw_row['amount']):,.2f}",
            "Recipient": "Known" if bool(raw_row["recipient_known"]) else "Unknown",
            "PIN entry": "Pasted" if bool(raw_row["pin_pasted"]) else "Typed",
            "Ground truth": (
                f"⚠ {str(raw_row['attack_type']).replace('_', ' ').title()}"
                if raw_row["label"] == 1 else "✓ Genuine"
            ),
        }
        rows = "".join(
            f'<div class="pg-kv-row"><span class="pg-kv-k">{k}</span><span class="pg-kv-v">{v}</span></div>'
            for k, v in details.items()
        )
        st.markdown(f'<div class="pg-card" style="padding:16px 20px;">{rows}</div>', unsafe_allow_html=True)
    else:
        st.caption("Enter feature values directly (deviation-from-baseline units):")
        hour_dev = st.slider("Hour deviation (std devs from usual hour)", 0.0, 5.0, 1.0)
        amount_dev = st.slider("Amount deviation (std devs from usual amount)", -2.0, 6.0, 0.0)
        typing_dev = st.slider("Typing deviation (std devs, app only)", -2.0, 6.0, 0.0)
        is_new_device = st.checkbox("New/unrecognized device")
        recipient_known = st.checkbox("Recipient known", value=True)
        pin_pasted = st.checkbox("PIN/OTP pasted rather than typed")

        feature_row = pd.Series({
            "hour_deviation": hour_dev, "amount_deviation": amount_dev,
            "typing_deviation": typing_dev, "is_new_device": int(is_new_device),
            "device_seen_in_history": int(not is_new_device),
            "recipient_known": int(recipient_known),
            "pin_pasted": int(pin_pasted), "prior_session_count": 20, "cold_start": 0,
        })
        raw_row = dict(
            device_id="custom_device" if is_new_device else "usual_device",
            amount=0, recipient="custom", hour=int(12 + hour_dev),
        )

with col2:
    st.markdown('<p class="pg-section-label">02 — Decision</p>', unsafe_allow_html=True)

    log_user_id = int(feature_row["user_id"]) if "user_id" in feature_row else 0
    log_session_id = int(feature_row["session_id"]) if "session_id" in feature_row else None

    if offline_mode:
        cache = CachedUserBaseline(
            user_id=log_user_id, last_known_device="usual_device",
            typical_amount_band_low=1000, typical_amount_band_high=15000,
            known_recipients={"known_acct"}, typical_hour_range=(7, 22),
            synced_at="2026-01-01",
        )
        offline_session = dict(
            device_id=(raw_row["device_id"] if isinstance(raw_row, dict) else raw_row["device_id"]),
            amount=(raw_row["amount"] if isinstance(raw_row, dict) else float(raw_row["amount"])),
            recipient="known_acct" if feature_row.get("recipient_known", 1) else "unknown_acct",
            hour=(raw_row["hour"] if isinstance(raw_row, dict) else 12),
        )

        t0 = time.perf_counter()
        result = score_offline(offline_session, cache, cache_age_days=3)
        latency_ms = (time.perf_counter() - t0) * 1000

        tier_map = {Action.ALLOW: "allow", Action.SOFT_CHALLENGE: "monitor",
                    Action.HOLD_FOR_RECONCILIATION: "block"}
        tier = tier_map[result["action"]]
        style = TIER_STYLE[tier]

        with db_module.get_conn(DB_PATH) as conn:
            decision_id = db_module.log_decision(
                conn, user_id=log_user_id, engine="offline_fallback", tier=tier,
                reasons=result["reasons"], session_id=log_session_id,
                risk_score=None, latency_ms=latency_ms,
            )

        result_card = (
            f'<div class="pg-card pg-card--result" style="background:linear-gradient(180deg,{style["bg"]} 0%,#FFFFFF 100%);">'
            f'<span class="pg-tier-badge" style="background:{style["color"]}1A; color:{style["color"]}; border:1px solid {style["color"]}33;">{style["icon"]} {style["label"]}</span>'
            f'<p class="pg-risk-number" style="color:{style["color"]};">—</p>'
            f'<p class="pg-subtitle">Rule-based score · decision preserved for reconciliation</p>'
            f'<div class="pg-stat-row">'
            f'<div class="pg-stat"><div class="pg-stat-label">Latency</div><div class="pg-stat-value">{latency_ms:.2f} ms</div></div>'
            f'<div class="pg-stat"><div class="pg-stat-label">Engine</div><div class="pg-stat-value">Rule fallback</div></div>'
            f'<div class="pg-stat"><div class="pg-stat-label">Audit ID</div><div class="pg-stat-value"><span class="pg-mono">#{decision_id}</span></div></div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(result_card, unsafe_allow_html=True)

        reasons = result["reasons"] or ["No risk factors triggered."]
        reason_rows = "".join(
            f'<div class="pg-reason"><span class="pg-reason-marker">!</span><span>{r}</span></div>'
            for r in reasons
        )
        reasons_card = (
            '<div class="pg-card">'
            '<p class="pg-section-label">Signals detected</p>'
            f'{reason_rows}'
            '</div>'
        )
        st.markdown(reasons_card, unsafe_allow_html=True)

        recon_card = (
            '<div class="pg-card" style="border-left:3px solid #D97706; background:#FFFBEB;">'
            '<p class="pg-section-label" style="color:#92400E !important;">Reconciliation</p>'
            f'<p style="font-size:.96rem; line-height:1.6; color:#78350F !important; margin:0;">{reconciliation_note(result)}</p>'
            '</div>'
        )
        st.markdown(recon_card, unsafe_allow_html=True)

    else:
        t0 = time.perf_counter()
        result = explain_session(model, feature_row, feature_cols, thresholds)
        latency_ms = (time.perf_counter() - t0) * 1000

        with db_module.get_conn(DB_PATH) as conn:
            decision_id = db_module.log_decision(
                conn, user_id=log_user_id, engine="ml_model", tier=result["tier"],
                reasons=result["reasons"], session_id=log_session_id,
                risk_score=result["risk_score"], latency_ms=latency_ms,
            )

        risk_pct = result["risk_score"]
        tier = result["tier"]
        style = TIER_STYLE[tier]

        subtitle = {
            "block": "High confidence — transaction stopped.",
            "monitor": "Not blocked — flagged for analyst review / step-up check.",
            "allow": "No significant behavioral risk detected.",
        }[tier]

        result_card = (
            f'<div class="pg-card pg-card--result" style="background:linear-gradient(180deg,{style["bg"]} 0%,#FFFFFF 100%);">'
            f'<span class="pg-tier-badge" style="background:{style["color"]}1A; color:{style["color"]}; border:1px solid {style["color"]}33;">{style["icon"]} {style["label"]}</span>'
            f'<p class="pg-risk-number" style="color:{style["color"]};">{risk_pct:.1%}</p>'
            f'<p class="pg-subtitle">Risk score · {subtitle}</p>'
            f'<div class="pg-stat-row">'
            f'<div class="pg-stat"><div class="pg-stat-label">Latency</div><div class="pg-stat-value">{latency_ms:.2f} ms</div></div>'
            f'<div class="pg-stat"><div class="pg-stat-label">Engine</div><div class="pg-stat-value">XGBoost + SHAP</div></div>'
            f'<div class="pg-stat"><div class="pg-stat-label">Audit ID</div><div class="pg-stat-value"><span class="pg-mono">#{decision_id}</span></div></div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(result_card, unsafe_allow_html=True)

        explain_card = (
            '<div class="pg-card" style="border-left:3px solid #10B981;">'
            '<p class="pg-section-label" style="color:#059669 !important;">Plain-English explanation</p>'
            f'<p style="font-size:1.0rem; line-height:1.6; color:#0B1F33 !important; margin:0;">{result["explanation"]}</p>'
            '</div>'
        )
        st.markdown(explain_card, unsafe_allow_html=True)

        with st.expander("Top contributing features (technical view)"):
            st.write(result["top_contributing_features"])

st.divider()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

footer_html = (
    '<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; padding-top:8px;">'
    '<div style="display:flex; align-items:center; gap:10px;">'
    '<span style="font-weight:700; color:#0B1F33 !important; letter-spacing:-0.02em;">PulseGuard</span>'
    '<span style="color:#E8EDF3;">·</span>'
    '<span style="color:#64748B !important; font-size:.85rem;">ICSC 2026 · Track A</span>'
    '</div>'
    '<div style="display:flex; gap:8px;">'
    '<span class="pg-chip">Synthetic data only</span>'
    '<span class="pg-chip">No PII stored</span>'
    '</div>'
    '</div>'
    '<p style="color:#94A3B8 !important; font-size:.78rem; line-height:1.6; margin-top:12px;">'
    'Known limitation: "patient/low-and-slow" pattern recall is 79% at the single-threshold level — full honesty on limitations in the technical write-up.'
    '</p>'
)
st.markdown(footer_html, unsafe_allow_html=True)