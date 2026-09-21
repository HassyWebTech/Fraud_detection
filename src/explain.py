import time
import numpy as np
import pandas as pd
import joblib
import shap
import builtins

import db as db_module


def _tree_explainer_safe(booster):
    
    def _patched_float(value, _orig=builtins.float):
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("[") and s.endswith("]"):
                return _orig(s.strip("[]"))
        return _orig(value)

    orig_float = builtins.float
    builtins.float = _patched_float
    try:
        return shap.TreeExplainer(booster)
    finally:
        builtins.float = orig_float


FEATURE_EXPLANATIONS = {
    "hour_deviation": "the time of this transaction was unusual compared to your normal activity",
    "amount_deviation": "the transfer amount was significantly higher than your typical transfers",
    "typing_deviation": "the way the PIN/OTP was entered didn't match your usual typing pattern",
    "is_new_device": "this transaction came from a device we haven't seen on your account before",
    "device_seen_in_history": "this device is not one we recognize from your account history",
    "recipient_known": "the recipient is not someone you have sent money to before",
    "pin_pasted": "the PIN or OTP appeared to be pasted rather than typed, which is unusual",
    "prior_session_count": "there is limited transaction history to compare this session against",
    "cold_start": "there is limited transaction history to compare this session against",
}


def explain_session(model, feature_row: pd.Series, feature_cols: list,
                     thresholds: dict, top_k: int = 2) -> dict:
   
    # See _tree_explainer_safe() docstring for why this isn't a plain shap.TreeExplainer(...) call.
    explainer = _tree_explainer_safe(model.get_booster())
    X = feature_row[feature_cols].to_frame().T.astype(float)
    shap_values = explainer.shap_values(X)

    contributions = pd.Series(shap_values[0], index=feature_cols)
    proba = model.predict_proba(X)[0, 1]

    if proba >= thresholds["block_threshold"]:
        tier = "block"
    elif proba >= thresholds["monitor_threshold"]:
        tier = "monitor"
    else:
        tier = "allow"

    # only positive contributions (pushed risk UP) are relevant to "why flagged"
    top_features = contributions[contributions > 0].sort_values(ascending=False).head(top_k)
    reasons = [FEATURE_EXPLANATIONS[f] for f in top_features.index if f in FEATURE_EXPLANATIONS]

    if tier == "allow":
        sentence = "This transaction was reviewed and processed normally with no significant behavioral risk factors detected."
    elif not reasons:
        sentence = "This transaction was flagged for review based on a combination of minor risk factors."
    elif len(reasons) == 1:
        sentence = f"This transaction was flagged because {reasons[0]}."
    else:
        sentence = f"This transaction was flagged because {reasons[0]}, and {reasons[1]}."

    return dict(risk_score=round(float(proba), 4), tier=tier, flagged=(tier != "allow"),
                explanation=sentence, top_contributing_features=list(top_features.index),
                reasons=reasons)


def demo():
    saved = joblib.load("../models/ato_model.pkl")
    model, feature_cols = saved["model"], saved["feature_cols"]
    thresholds = dict(block_threshold=saved["block_threshold"],
                       monitor_threshold=saved["monitor_threshold"])

    df = pd.read_csv("../data/processed/features.csv", parse_dates=["timestamp"])

    db_module.init_db("../data/ato.db")

    print(f"BLOCK threshold: {thresholds['block_threshold']:.3f} | "
          f"MONITOR threshold: {thresholds['monitor_threshold']:.3f}\n")
    print("=" * 70)

    with db_module.get_conn("../data/ato.db") as conn:
        for attack_type in ["none", "credential_theft", "sim_swap",
                             "patient_low_and_slow", "social_engineering"]:
            subset = df[df["attack_type"] == attack_type]
            if len(subset) == 0:
                continue
            row = subset.iloc[0]

            t0 = time.perf_counter()
            result = explain_session(model, row, feature_cols, thresholds)
            latency_ms = (time.perf_counter() - t0) * 1000

            decision_id = db_module.log_decision(
                conn, user_id=int(row["user_id"]), engine="ml_model",
                tier=result["tier"], reasons=result["reasons"],
                session_id=int(row["session_id"]), risk_score=result["risk_score"],
                latency_ms=latency_ms, true_label=int(row["label"]),
            )

            print(f"\nSession {int(row['session_id'])} | true label: {attack_type} | "
                  f"tier: {result['tier'].upper()} | logged as decision_id={decision_id}")
            print(f"Risk score: {result['risk_score']:.1%} | latency: {latency_ms:.3f}ms")
            print(f"Customer-facing explanation: \"{result['explanation']}\"")
            print("-" * 70)


if __name__ == "__main__":
    demo()