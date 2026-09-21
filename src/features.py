import argparse
import numpy as np
import pandas as pd



COLD_START_MIN_SESSIONS = 3


def circular_hour_distance(h1: float, h2: float) -> float:
    
    diff = abs(h1 - h2) % 24
    return min(diff, 24 - diff)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

  
    pop_hour_mean = df["hour"].mean()
    pop_hour_std = max(df["hour"].std(), 0.5) if pd.notna(df["hour"].std()) else 0.5
    pop_amount_log = np.log(df["amount"])
    pop_amount_log_mean = pop_amount_log.mean()
    pop_amount_log_std = max(pop_amount_log.std(), 0.3) if pd.notna(pop_amount_log.std()) else 0.3
    app_typing = df.loc[df.channel == "app", "typing_ms"]
    pop_typing_mean = app_typing.mean() if len(app_typing) else 3000.0
    pop_typing_std = max(app_typing.std(), 100) if len(app_typing) and pd.notna(app_typing.std()) else 100.0

    feature_rows = []

    for user_id, g in df.groupby("user_id", sort=False):
        g = g.sort_values("timestamp")
        # running (expanding, shifted by 1) stats — i.e. "history strictly before this row"
        hours = g["hour"].to_numpy()
        amounts_log = np.log(g["amount"].to_numpy())
        typing = g["typing_ms"].to_numpy()
        devices = g["device_id"].to_numpy() if "device_id" in g else None

        n = len(g)
        for i in range(n):
            history_n = i  # number of PRIOR sessions available
            if history_n >= COLD_START_MIN_SESSIONS:
                h_hist = hours[:i]
                a_hist = amounts_log[:i]
                t_hist = typing[:i]
                user_hour_mean, user_hour_std = h_hist.mean(), max(h_hist.std(), 0.5)
                user_amount_mean, user_amount_std = a_hist.mean(), max(a_hist.std(), 0.3)
                valid_typing = t_hist[~np.isnan(t_hist)] if len(t_hist) else np.array([])
                if len(valid_typing) >= COLD_START_MIN_SESSIONS:
                    user_typing_mean, user_typing_std = valid_typing.mean(), max(valid_typing.std(), 100)
                else:
                    user_typing_mean, user_typing_std = pop_typing_mean, pop_typing_std
                cold_start = False
            else:
                user_hour_mean, user_hour_std = pop_hour_mean, pop_hour_std
                user_amount_mean, user_amount_std = pop_amount_log_mean, pop_amount_log_std
                user_typing_mean, user_typing_std = pop_typing_mean, pop_typing_std
                cold_start = True

            row = g.iloc[i]
            hour_dev = circular_hour_distance(row["hour"], user_hour_mean) / max(user_hour_std, 0.5)
            amount_dev = (np.log(row["amount"]) - user_amount_mean) / user_amount_std

            if row["channel"] == "app" and not np.isnan(row["typing_ms"]):
                typing_dev = (row["typing_ms"] - user_typing_mean) / user_typing_std
            else:
                typing_dev = np.nan

            device_seen_before = (
                history_n > 0 and row["device_id"] in set(devices[:i])
            ) if devices is not None else np.nan

            feature_rows.append(dict(
                session_id=row["session_id"],
                user_id=user_id,
                timestamp=row["timestamp"],
                channel=row["channel"],
                hour_deviation=round(hour_dev, 3),
                amount_deviation=round(amount_dev, 3),
                typing_deviation=round(typing_dev, 3) if not np.isnan(typing_dev) else np.nan,
                is_new_device=int(row["is_new_device"]),
                device_seen_in_history=int(device_seen_before) if not pd.isna(device_seen_before) else np.nan,
                recipient_known=int(row["recipient_known"]),
                pin_pasted=int(row["pin_pasted"]),
                prior_session_count=history_n,
                cold_start=int(cold_start),
                raw_amount=row["amount"],
                label=row["label"],
                attack_type=row["attack_type"],
            ))

    feat_df = pd.DataFrame(feature_rows).sort_values("session_id").reset_index(drop=True)
    return feat_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", type=str, default="../data/raw/sessions.csv")
    p.add_argument("--out", dest="outfile", type=str, default="../data/processed/features.csv")
    args = p.parse_args()

    raw = pd.read_csv(args.infile)
    feat = build_features(raw)
    feat.to_csv(args.outfile, index=False)

    print(f"Built features for {len(feat)} sessions")
    print(f"Cold-start sessions: {feat['cold_start'].mean():.2%}")
    print("\nFeature summary by label:")
    print(feat.groupby("label")[["hour_deviation", "amount_deviation", "typing_deviation", "is_new_device", "pin_pasted"]].mean())
    print(f"\nSaved to {args.outfile}")


if __name__ == "__main__":
    main()