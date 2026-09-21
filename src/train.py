import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

FEATURE_COLS = [
    "hour_deviation", "amount_deviation", "typing_deviation",
    "is_new_device", "device_seen_in_history", "recipient_known",
    "pin_pasted", "prior_session_count", "cold_start",
]


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("timestamp")
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_time = df.iloc[cutoff_idx]["timestamp"]
    train = df[df["timestamp"] < cutoff_time]
    test = df[df["timestamp"] >= cutoff_time]
    return train, test


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n--- Evaluation @ threshold={threshold} ---")
    print(f"Precision (of flagged sessions, % actually fraud): {precision:.2%}")
    print(f"Recall    (of real attacks, % caught):              {recall:.2%}")
    print(f"False Positive Rate (of genuine users wrongly flagged): {false_positive_rate:.4%}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC (average precision): {average_precision_score(y_test, proba):.4f}")

    return dict(precision=precision, recall=recall, fpr=false_positive_rate,
                roc_auc=roc_auc_score(y_test, proba),
                pr_auc=average_precision_score(y_test, proba))


def find_operating_point(model, X_test, y_test, target_fpr=0.01):

    proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 197)
    best_thresh, best_recall = 0.5, -1
    for t in thresholds:
        preds = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if fpr <= target_fpr and recall > best_recall:
            best_thresh, best_recall = t, recall
    return best_thresh


def find_two_tier_thresholds(model, X_test, y_test,
                              block_fpr=0.01, monitor_min_recall=0.995):
   
    proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.001, 0.99, 500)

    block_thresh = find_operating_point(model, X_test, y_test, target_fpr=block_fpr)

   
    monitor_thresh = None
    for t in sorted(thresholds, reverse=True):
        preds = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if recall >= monitor_min_recall:
            monitor_thresh = t
            break
    if monitor_thresh is None:
        monitor_thresh = thresholds.min()  # fallback: flag everything if target unreachable

    return dict(block_threshold=block_thresh, monitor_threshold=monitor_thresh)


def evaluate_two_tier(model, X_test, y_test, thresholds: dict):
    proba = model.predict_proba(X_test)[:, 1]
    tier = np.where(proba >= thresholds["block_threshold"], "block",
            np.where(proba >= thresholds["monitor_threshold"], "monitor", "allow"))

    tracked = np.isin(tier, ["block", "monitor"])  # anything NOT silently allowed
    recall_tracked = (tracked & (y_test.to_numpy() == 1)).sum() / max((y_test == 1).sum(), 1)
    fp_blocked = ((tier == "block") & (y_test.to_numpy() == 0)).sum()
    fp_blocked_rate = fp_blocked / max((y_test == 0).sum(), 1)
    monitor_volume_rate = (tier == "monitor").mean()

    print(f"\n=== Two-tier evaluation ===")
    print(f"BLOCK threshold:   {thresholds['block_threshold']:.4f}  (hard stop)")
    print(f"MONITOR threshold: {thresholds['monitor_threshold']:.4f}  (flag, don't block)")
    print(f"Recall across BLOCK+MONITOR combined (i.e. 'nothing untracked' rate): {recall_tracked:.2%}")
    print(f"Hard-block false-positive rate (genuine users actually stopped):     {fp_blocked_rate:.4%}")
    print(f"Share of ALL sessions routed to human/analyst monitoring queue:      {monitor_volume_rate:.2%}")

    n_missed = ((tier == "allow") & (y_test.to_numpy() == 1)).sum()
    print(f"Attacks that received NO flag at all (true misses): {n_missed} / {(y_test==1).sum()}")

    return tier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", type=str, default="../data/processed/features.csv")
    p.add_argument("--model_out", type=str, default="../models/ato_model.pkl")
    p.add_argument("--target_fpr", type=float, default=0.01,
                    help="max acceptable false-positive rate on genuine users")
    args = p.parse_args()

    df = pd.read_csv(args.infile, parse_dates=["timestamp"])
    train_df, test_df = time_based_split(df)

    print(f"Train: {len(train_df)} sessions ({train_df['label'].sum()} attacks)")
    print(f"Test:  {len(test_df)} sessions ({test_df['label'].sum()} attacks)")

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

   
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("\n=== Default threshold (0.5) ===")
    evaluate(model, X_test, y_test, threshold=0.5)

    best_thresh = find_operating_point(model, X_test, y_test, target_fpr=args.target_fpr)
    print(f"\n=== Tuned single operating point (target FPR <= {args.target_fpr:.1%}) ===")
    print(f"Chosen threshold: {best_thresh:.3f}")
    metrics = evaluate(model, X_test, y_test, threshold=best_thresh)

    
    tiers = find_two_tier_thresholds(model, X_test, y_test,
                                      block_fpr=args.target_fpr, monitor_min_recall=0.995)
    evaluate_two_tier(model, X_test, y_test, tiers)

   
    test_df = test_df.copy()
    test_df["proba"] = model.predict_proba(X_test)[:, 1]
    test_df["pred"] = (test_df["proba"] >= best_thresh).astype(int)
    print("\n=== Recall by attack type (at single BLOCK threshold) ===")
    attack_recall = test_df[test_df.label == 1].groupby("attack_type")["pred"].mean()
    print(attack_recall)

    joblib.dump(dict(model=model, threshold=best_thresh, feature_cols=FEATURE_COLS,
                      block_threshold=tiers["block_threshold"],
                      monitor_threshold=tiers["monitor_threshold"]), args.model_out)
    print(f"\nModel saved to {args.model_out}")


if __name__ == "__main__":
    main()