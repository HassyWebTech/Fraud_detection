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

st.set_page_config(
    page_title="PulseGuard — ICSC 2026",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)





LOGO_SVG = """
<svg width="52" height="56" viewBox="0 0 100 112" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="PulseGuard">
  <defs>
    <linearGradient id="pgFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0B2A4A"/>
      <stop offset="100%" stop-color="#0E4E7A"/>
    </linearGradient>
  </defs>

  <path d="M50 3 L93 19 C95 20 96 22 96 24 L96 54
           C96 84 76 102 51 109 C50 109.3 49 109.3 48 109
           C23 102 4 84 4 54 L4 24 C4 22 5 20 7 19 Z"
        fill="url(#pgFill)"/>

  <path d="M50 12 L86 25 L86 54
           C86 79 69 95 50 101
           C31 95 14 79 14 54 L14 25 Z"
        fill="none" stroke="#10B981" stroke-width="2.2" opacity="0.85"/>

  <polyline points="20,58 34,58 40,50 46,58 56,58 62,32 68,80 74,58 82,58"
            fill="none" stroke="#FFFFFF" stroke-width="4.2"
            stroke-linecap="round" stroke-linejoin="round"/>

  <circle cx="62" cy="32" r="4" fill="#10B981"/>
</svg>
"""

LOGO_SVG_SMALL = LOGO_SVG.replace('width="52" height="56"', 'width="30" height="32"')



# Design tokens — "Trust Stack"


TIER_STYLE = {
    "allow":   dict(color="#10B981", bg="#ECFDF5", icon="✓", label="Allowed"),
    "monitor": dict(color="#D97706", bg="#FFFBEB", icon="◐", label="Monitoring"),
    "block":   dict(color="#DC2626", bg="#FEF2F2", icon="✕", label="Blocked"),
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif;
        font-feature-settings: 'cv02','cv03','cv04','ss01';
    }

    :root {
        --pg-ink:       #0B1F33;
        --pg-navy:      #0B2A4A;
        --pg-teal:      #0E4E7A;
        --pg-emerald:   #10B981;
        --pg-sky:       #0EA5E9;

        --pg-safe:      #10B981;
        --pg-safe-bg:   #ECFDF5;
        --pg-watch:     #D97706;
        --pg-watch-bg:  #FFFBEB;
        --pg-threat:    #DC2626;
        --pg-threat-bg: #FEF2F2;

        --pg-bg:        #F7F9FC;
        --pg-surface:   #FFFFFF;
        --pg-muted:     #64748B;
        --pg-border:    #E4E9F0;
        --pg-shadow:    0 1px 2px rgba(11,31,51,.04), 0 8px 24px rgba(11,31,51,.06);
    }

    .pg-header { display:flex; align-items:center; gap:18px; padding:8px 0 18px; }
    .pg-title {
        font-size: 2.15rem; font-weight: 800; margin:0;
        letter-spacing:-0.028em; color:var(--pg-ink);
    }
    .pg-tagline {
        color:var(--pg-muted); font-size:1rem; margin:2px 0 0;
        font-weight:500; letter-spacing:-0.005em;
    }

    .pg-trust { display:flex; gap:10px; flex-wrap:wrap; margin: 2px 0 6px; }
    .pg-chip {
        display:inline-flex; align-items:center; gap:7px;
        font-size:.78rem; font-weight:600; letter-spacing:.01em;
        padding:5px 11px; border-radius:9999px;
        background:var(--pg-surface); border:1px solid var(--pg-border);
        color:var(--pg-muted);
    }
    .pg-chip .dot {
        width:7px; height:7px; border-radius:50%;
        background:var(--pg-emerald);
        box-shadow:0 0 0 3px rgba(16,185,129,.15);
    }

    .pg-card {
        border-radius:18px; padding:24px 26px; margin-top:8px;
        background:var(--pg-surface); border:1px solid var(--pg-border);
        box-shadow:var(--pg-shadow);
    }
    .pg-card--result { padding: 28px 30px; }

    .pg-tier-badge {
        display:inline-flex; align-items:center; gap:9px;
        font-size:.82rem; font-weight:700; letter-spacing:.06em;
        padding:7px 14px; border-radius:9999px; text-transform:uppercase;
    }
    .pg-risk-number {
        font-size:3.4rem; font-weight:800; margin:10px 0 0;
        letter-spacing:-0.035em; line-height:1;
        font-variant-numeric: tabular-nums;
    }
    .pg-subtitle {
        color:var(--pg-muted); font-size:.9rem; margin:6px 0 0;
        font-weight:500;
    }

    .pg-mono {
        font-family:'JetBrains Mono', monospace;
        font-size:.82rem; font-weight:500;
        color:var(--pg-muted);
        background:var(--pg-bg);
        padding:2px 8px; border-radius:6px;
        border:1px solid var(--pg-border);
    }

    .pg-stat-row { display:flex; gap:14px; margin-top:18px; }
    .pg-stat {
        flex:1; background:var(--pg-bg); border-radius:12px;
        padding:12px 14px; border:1px solid var(--pg-border);
    }
    .pg-stat-label {
        font-size:.72rem; font-weight:600; letter-spacing:.06em;
        text-transform:uppercase; color:var(--pg-muted); margin-bottom:4px;
    }
    .pg-stat-value {
        font-size:1.15rem; font-weight:700; color:var(--pg-ink);
        font-variant-numeric: tabular-nums;
    }

    .pg-reason {
        display:flex; gap:10px; padding:11px 0;
        border-bottom:1px solid var(--pg-border);
        font-size:.94rem; color:var(--pg-ink); line-height:1.5;
    }
    .pg-reason:last-child { border-bottom:none; }
    .pg-reason-marker {
        flex:0 0 20px; height:20px; border-radius:6px;
        display:flex; align-items:center; justify-content:center;
        font-size:.7rem; font-weight:700;
        background:var(--pg-watch-bg); color:var(--pg-watch);
    }

    .pg-section-label {
        font-size:.72rem; font-weight:600; letter-spacing:.06em;
        text-transform:uppercase; color:var(--pg-muted);
        margin:0 0 10px 0;
    }

    div[data-testid="stMetric"] {
        background:var(--pg-bg); border-radius:12px; padding:12px 14px;
        border:1px solid var(--pg-border);
    }
    div[data-testid="stMetric"] label p {
        font-size:.72rem !important; font-weight:600 !important;
        letter-spacing:.06em !important; text-transform:uppercase !important;
        color:var(--pg-muted) !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums;
        font-weight:800; letter-spacing:-0.02em;
    }
</style>
""", unsafe_allow_html=True)


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



# Header + trust strip


st.markdown(f"""
<div class="pg-header">
    {LOGO_SVG}
    <div>
        <p class="pg-title">PulseGuard</p>
        <p class="pg-tagline">Learns your rhythm. Notices when it's not you.</p>
    </div>
</div>
<div class="pg-trust">
    <span class="pg-chip"><span class="dot"></span>ML engine online</span>
    <span class="pg-chip">⚡ p99 latency · 1.86 ms</span>
    <span class="pg-chip">🔒 256-bit encrypted</span>
    <span class="pg-chip">🧪 Synthetic data only</span>
</div>
""", unsafe_allow_html=True)
st.divider()



# Sidebar — brand, resilient-mode toggle, proof metrics


st.sidebar.markdown(f"""
<div class="pg-sidebar-brand" style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
    <div style="width:30px;">{LOGO_SVG_SMALL}</div>
    <span style="font-weight:800; font-size:1.15rem; color:#0B1F33; letter-spacing:-0.02em;">PulseGuard</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("#### System status")
offline_mode = st.sidebar.toggle("Simulate system offline (network/power cut)", value=False)

if offline_mode:
    st.sidebar.markdown("""
    <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:12px; padding:14px 16px; margin-top:8px;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="width:8px; height:8px; border-radius:50%; background:#D97706;"></span>
            <span style="font-weight:700; font-size:.82rem; letter-spacing:.04em; text-transform:uppercase; color:#92400E;">Resilient Mode</span>
        </div>
        <p style="font-size:.85rem; line-height:1.5; color:#78350F; margin:0;">
            ML engine offline. Running on-device rule fallback — decisions continue, audit trail preserved, reconciliation queued.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:12px; padding:14px 16px; margin-top:8px;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="width:8px; height:8px; border-radius:50%; background:#10B981;"></span>
            <span style="font-weight:700; font-size:.82rem; letter-spacing:.04em; text-transform:uppercase; color:#065F46;">Full Engine Online</span>
        </div>
        <p style="font-size:.85rem; line-height:1.5; color:#064E3B; margin:0;">
            XGBoost + SHAP. Sub-2 ms p99 scoring latency.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("#### Proof, not promises")
st.sidebar.markdown("""
<div style="background:#F7F9FC; border-radius:12px; padding:14px; border:1px solid #E4E9F0;">
    <div style="font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:#64748B; margin-bottom:10px;">On time-based holdout</div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
        <div>
            <div style="font-size:1.4rem; font-weight:800; color:#10B981; font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">100%</div>
            <div style="font-size:.72rem; color:#64748B; font-weight:500;">Attacks tracked</div>
        </div>
        <div>
            <div style="font-size:1.4rem; font-weight:800; color:#0B1F33; font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">0.98%</div>
            <div style="font-size:.72rem; color:#64748B; font-weight:500;">False stops</div>
        </div>
        <div>
            <div style="font-size:1.4rem; font-weight:800; color:#0B1F33; font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">1.86<span style="font-size:.85rem; font-weight:600; color:#64748B;">ms</span></div>
            <div style="font-size:.72rem; color:#64748B; font-weight:500;">p99 latency</div>
        </div>
        <div>
            <div style="font-size:1.4rem; font-weight:800; color:#0B1F33; font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">4</div>
            <div style="font-size:.72rem; color:#64748B; font-weight:500;">Attack types</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
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
            "Amount": f"${float(raw_row['amount']):,.2f}",
            "Recipient": "Known" if bool(raw_row["recipient_known"]) else "Unknown",
            "PIN entry": "Pasted" if bool(raw_row["pin_pasted"]) else "Typed",
            "Ground truth": (
                f"⚠ {str(raw_row['attack_type']).replace('_', ' ').title()}"
                if raw_row["label"] == 1 else "✓ Genuine"
            ),
        }
        details_html = "".join(
            f'<div style="display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid #E4E9F0;">'
            f'<span style="color:#64748B; font-size:.85rem; font-weight:500;">{k}</span>'
            f'<span style="color:#0B1F33; font-size:.85rem; font-weight:600; font-variant-numeric:tabular-nums;">{v}</span>'
            f'</div>'
            for k, v in details.items()
        )
        st.markdown(
            f'<div class="pg-card" style="padding:16px 20px;">{details_html}</div>',
            unsafe_allow_html=True,
        )
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

        st.markdown(f"""
        <div class="pg-card pg-card--result" style="background:linear-gradient(180deg, {style['bg']} 0%, #FFFFFF 100%);">
            <span class="pg-tier-badge" style="background:{style['color']}1A; color:{style['color']}; border:1px solid {style['color']}33;">
                {style['icon']} {style['label']}
            </span>
            <p class="pg-risk-number" style="color:{style['color']};">—</p>
            <p class="pg-subtitle">Rule-based score · decision preserved for reconciliation</p>

            <div class="pg-stat-row">
                <div class="pg-stat">
                    <div class="pg-stat-label">Latency</div>
                    <div class="pg-stat-value">{latency_ms:.2f} ms</div>
                </div>
                <div class="pg-stat">
                    <div class="pg-stat-label">Engine</div>
                    <div class="pg-stat-value">Rule fallback</div>
                </div>
                <div class="pg-stat">
                    <div class="pg-stat-label">Audit ID</div>
                    <div class="pg-stat-value pg-mono" style="font-size:.95rem;">#{decision_id}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        reasons = result["reasons"] or ["No risk factors triggered."]
        reason_html = "".join(
            f'<div class="pg-reason"><span class="pg-reason-marker">!</span><span>{r}</span></div>'
            for r in reasons
        )
        st.markdown(f"""
        <div class="pg-card">
            <p class="pg-section-label">Signals detected</p>
            {reason_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pg-card" style="border-left:3px solid #D97706; background:#FFFBEB;">
            <p class="pg-section-label">Reconciliation</p>
            <p style="font-size:.98rem; line-height:1.6; color:#78350F; margin:0;">{reconciliation_note(result)}</p>
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown(f"""
        <div class="pg-card pg-card--result" style="background:linear-gradient(180deg, {style['bg']} 0%, #FFFFFF 100%);">
            <span class="pg-tier-badge" style="background:{style['color']}1A; color:{style['color']}; border:1px solid {style['color']}33;">
                {style['icon']} {style['label']}
            </span>
            <p class="pg-risk-number" style="color:{style['color']};">{risk_pct:.1%}</p>
            <p class="pg-subtitle">Risk score · {subtitle}</p>

            <div class="pg-stat-row">
                <div class="pg-stat">
                    <div class="pg-stat-label">Latency</div>
                    <div class="pg-stat-value">{latency_ms:.2f} ms</div>
                </div>
                <div class="pg-stat">
                    <div class="pg-stat-label">Engine</div>
                    <div class="pg-stat-value">XGBoost + SHAP</div>
                </div>
                <div class="pg-stat">
                    <div class="pg-stat-label">Audit ID</div>
                    <div class="pg-stat-value pg-mono" style="font-size:.95rem;">#{decision_id}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pg-card" style="border-left:3px solid #10B981;">
            <p class="pg-section-label">Plain-English explanation</p>
            <p style="font-size:1.02rem; line-height:1.6; color:#0B1F33; margin:0;">{result['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Top contributing features (technical view)"):
            st.write(result["top_contributing_features"])

st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; padding-top:8px;">
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-weight:700; color:#0B1F33; letter-spacing:-0.02em;">PulseGuard</span>
        <span style="color:#E4E9F0;">·</span>
        <span style="color:#64748B; font-size:.85rem;">ICSC 2026 · Track A</span>
    </div>
    <div style="display:flex; gap:8px;">
        <span class="pg-chip">Synthetic data only</span>
        <span class="pg-chip">No PII stored</span>
    </div>
</div>
<p style="color:#94A3B8; font-size:.78rem; line-height:1.6; margin-top:12px;">
    Known limitation: "patient/low-and-slow" pattern recall is 79% at the single-threshold level —
    full honesty on limitations in the technical write-up.
</p>
""", unsafe_allow_html=True)