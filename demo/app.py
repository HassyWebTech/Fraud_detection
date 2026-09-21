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

st.set_page_config(page_title="ATO Detection — ICSC 2026", layout="wide")


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

st.title("Behavioral Account Takeover Detection")
st.caption("ICSC 2026 Hackathon — Track A · Prototype demo (synthetic data only)")


st.sidebar.header("System status")
offline_mode = st.sidebar.toggle("Simulate system offline (network/power cut)", value=False)
if offline_mode:
    st.sidebar.warning("Running in DEGRADED MODE: no model access, rule-based fallback only.")
else:
    st.sidebar.success("Full ML risk engine online.")

st.sidebar.divider()
st.sidebar.header("Model performance (test set)")
st.sidebar.metric("Recall (attacks caught)", "93.7%")
st.sidebar.metric("Precision (flags that are real)", "55.1%")
st.sidebar.metric("False positive rate", "0.98%")
st.sidebar.caption("Measured on a time-based holdout split — see technical write-up for full methodology and honest limitations.")


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Choose a session to score")
    mode = st.radio("Source", ["Pick an example session", "Enter a custom session"], horizontal=True)

    if mode == "Pick an example session":
        attack_type_filter = st.selectbox(
            "Filter by type",
            ["none", "credential_theft", "sim_swap", "patient_low_and_slow", "social_engineering"],
        )
        subset = feat_df[feat_df["attack_type"] == attack_type_filter]
        session_id = st.selectbox("Session ID", subset["session_id"].head(20).tolist())
        feature_row = feat_df[feat_df["session_id"] == session_id].iloc[0]
        raw_row = raw_df[raw_df["session_id"] == session_id].iloc[0]

        st.write("**Raw session details:**")
        st.json(dict(
            channel=raw_row["channel"], device_id=raw_row["device_id"],
            is_new_device=bool(raw_row["is_new_device"]), amount=float(raw_row["amount"]),
            recipient_known=bool(raw_row["recipient_known"]),
            pin_pasted=bool(raw_row["pin_pasted"]),
            true_label=("ATTACK: " + raw_row["attack_type"]) if raw_row["label"] == 1 else "genuine",
        ))
    else:
        st.write("Enter feature values directly (already 'deviation from baseline' units):")
        hour_dev = st.slider("Hour deviation (std devs from user's usual hour)", 0.0, 5.0, 1.0)
        amount_dev = st.slider("Amount deviation (std devs from user's usual amount)", -2.0, 6.0, 0.0)
        typing_dev = st.slider("Typing deviation (std devs, app only)", -2.0, 6.0, 0.0)
        is_new_device = st.checkbox("New/unrecognized device")
        recipient_known = st.checkbox("Recipient known", value=True)
        pin_pasted = st.checkbox("PIN/OTP pasted rather than typed")

        feature_row = pd.Series({
            "hour_deviation": hour_dev, "amount_deviation": amount_dev,
            "typing_deviation": typing_dev, "is_new_device": int(is_new_device),
            "device_seen_in_history": int(not is_new_device), "recipient_known": int(recipient_known),
            "pin_pasted": int(pin_pasted), "prior_session_count": 20, "cold_start": 0,
        })
        raw_row = dict(device_id="custom_device" if is_new_device else "usual_device",
                        amount=0, recipient="custom", hour=int(12 + hour_dev))

with col2:
    st.subheader("2. Decision")

    
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

        with db_module.get_conn(DB_PATH) as conn:
            decision_id = db_module.log_decision(
                conn, user_id=log_user_id, engine="offline_fallback", tier=tier,
                reasons=result["reasons"], session_id=log_session_id,
                risk_score=None, latency_ms=latency_ms,
            )

        st.error("⚠️ DEGRADED MODE — scored by local rule-based fallback, not the ML model")
        st.metric("Action", result["action"].value.upper())
        st.write("**Reasons:**")
        for r in result["reasons"] or ["No risk factors triggered."]:
            st.write(f"- {r}")
        st.caption(reconciliation_note(result))
        st.success(f"✅ Logged to audit trail (decision_id={decision_id}, {latency_ms:.3f}ms) "
                   f"— queued for reconciliation once back online.")

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
        if result["tier"] == "block":
            st.error(f"🚫 BLOCKED — risk score {risk_pct:.1%}")
        elif result["tier"] == "monitor":
            st.warning(f"🔍 MONITORING — risk score {risk_pct:.1%} (not blocked, flagged for review)")
        else:
            st.success(f"✅ ALLOWED — risk score {risk_pct:.1%}")

        st.progress(min(risk_pct, 1.0))
        st.write("**Explanation (customer-facing):**")
        st.info(result["explanation"])
        st.caption(f"Logged to audit trail (decision_id={decision_id}, "
                   f"scoring latency {latency_ms:.3f}ms)")

        with st.expander("Top contributing features (technical view)"):
            st.write(result["top_contributing_features"])

st.divider()
st.caption(
    "Synthetic data only — no real customer data used. See technical write-up for data generation "
    "methodology, full evaluation, and known limitations (notably: recall on the 'patient/low-and-slow' "
    "attack pattern is lower than other attack types, and reported AUC is likely optimistic relative to "
    "real-world fraud due to synthetic data characteristics)."
)