import numpy as np
import pandas as pd
import joblib
import shap


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
                     threshold: float, top_k: int = 2) -> dict:
    
    explainer = shap.TreeExplainer(model)
    X = feature_row[feature_cols].to_frame().T.astype(float)
    shap_values = explainer.shap_values(X)

    # shap_values shape: (1, n_features) for binary XGBoost classifier
    contributions = pd.Series(shap_values[0], index=feature_cols)
    proba = model.predict_proba(X)[0, 1]
    is_flagged = proba >= threshold

    # only positive contributions (pushed risk UP) are relevant to "why flagged"
    top_features = contributions[contributions > 0].sort_values(ascending=False).head(top_k)
    reasons = [FEATURE_EXPLANATIONS[f] for f in top_features.index if f in FEATURE_EXPLANATIONS]

    if not is_flagged:
        sentence = "This transaction was reviewed and processed normally with no significant behavioral risk factors detected."
    elif not reasons:
        sentence = "This transaction was flagged for manual review based on a combination of minor risk factors."
    elif len(reasons) == 1:
        sentence = f"This transaction was flagged because {reasons[0]}."
    else:
        sentence = f"This transaction was flagged because {reasons[0]}, and {reasons[1]}."

    return dict(risk_score=round(float(proba), 4), flagged=bool(is_flagged),
                explanation=sentence, top_contributing_features=list(top_features.index))


def demo():
    saved = joblib.load("../models/ato_model.pkl")
    model, feature_cols, threshold = saved["model"], saved["feature_cols"], saved["threshold"]

    df = pd.read_csv("../data/processed/features.csv", parse_dates=["timestamp"])

    print(f"Model decision threshold: {threshold:.3f}\n")
    print("=" * 70)

    # Show one example from each attack type, plus one normal session
    for attack_type in ["none", "credential_theft", "sim_swap",
                         "patient_low_and_slow", "social_engineering"]:
        subset = df[df["attack_type"] == attack_type]
        if len(subset) == 0:
            continue
        row = subset.iloc[0]
        result = explain_session(model, row, feature_cols, threshold)
        flagged = "FLAGGED" if result["flagged"] else "ALLOWED"

        print(f"\nSession {int(row['session_id'])} | true label: {attack_type} | decision: {flagged}")
        print(f"Risk score: {result['risk_score']:.1%}")
        print(f"Customer-facing explanation: \"{result['explanation']}\"")
        print("-" * 70)


if __name__ == "__main__":
    demo()