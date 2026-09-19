import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta



@dataclass
class UserProfile:
    user_id: int
    channel: str                     # "app" or "ussd"
    typical_hour_mean: float         # 0-23, hour of day they usually transact
    typical_hour_std: float
    typing_ms_mean: float            # PIN/OTP entry duration, app users only
    typing_ms_std: float
    device_id: str                   # their one usual device
    txn_amount_mean: float           # naira
    txn_amount_std: float
    known_recipients: list = field(default_factory=list)
    sessions_per_week: float = 3.0


def make_users(n_users: int, rng: np.random.Generator) -> list[UserProfile]:
    users = []
    # ~35% of Nigerian mobile money / low-end banking users rely on USSD
    channels = rng.choice(["app", "ussd"], size=n_users, p=[0.65, 0.35])

    for uid in range(n_users):
        channel = channels[uid]
        known_recipients = [f"acct_{rng.integers(0, 100000)}" for _ in range(rng.integers(1, 8))]
        users.append(UserProfile(
            user_id=uid,
            channel=channel,
            typical_hour_mean=rng.normal(14, 4) % 24,       # most activity midday-ish
            typical_hour_std=abs(rng.normal(2.5, 1.0)) + 0.5,
            typing_ms_mean=rng.normal(3200, 700) if channel == "app" else np.nan,
            typing_ms_std=abs(rng.normal(400, 100)) if channel == "app" else np.nan,
            device_id=f"dev_{uid}_{rng.integers(0, 999999)}",
            txn_amount_mean=max(500, rng.lognormal(mean=8.5, sigma=1.0)),
            txn_amount_std=abs(rng.normal(0.4, 0.15)),       # multiplicative spread factor
            known_recipients=known_recipients,
            sessions_per_week=max(0.5, rng.normal(3, 1.5)),
        ))
    return users


# --------------------------------------------------------------------------
# Session generation
# --------------------------------------------------------------------------

def gen_normal_session(user: UserProfile, sess_id: int, ts: datetime, rng) -> dict:
    hour = np.clip(rng.normal(user.typical_hour_mean, user.typical_hour_std), 0, 23.99)
    amount = max(200, rng.lognormal(
        mean=np.log(user.txn_amount_mean), sigma=user.txn_amount_std
    ))
    recipient_known = rng.random() < 0.85
    recipient = (
        rng.choice(user.known_recipients) if (recipient_known and user.known_recipients)
        else f"acct_{rng.integers(0, 100000)}"
    )

    row = dict(
        session_id=sess_id,
        user_id=user.user_id,
        channel=user.channel,
        timestamp=ts.replace(hour=int(hour), minute=int((hour % 1) * 60)),
        device_id=user.device_id,          # normal: their own device
        is_new_device=False,
        amount=round(amount, 2),
        recipient=recipient,
        recipient_known=recipient_known,
        label=0,
        attack_type="none",
    )
    if user.channel == "app":
        typing_ms = max(400, rng.normal(user.typing_ms_mean, user.typing_ms_std))
        row["typing_ms"] = round(typing_ms, 1)
        row["pin_pasted"] = rng.random() < 0.03  # rare legit paste (password manager etc.)
    else:
        row["typing_ms"] = np.nan
        row["pin_pasted"] = False
    return row


def gen_attack_session(user: UserProfile, sess_id: int, ts: datetime, rng) -> dict:
    """Attacker acting on this user's account. Attack subtype chosen
    based on channel (USSD attacks are necessarily metadata-only)."""
    if user.channel == "ussd":
        attack_type = rng.choice(["sim_swap", "patient_low_and_slow"])
    else:
        attack_type = rng.choice(
            ["credential_theft", "sim_swap", "patient_low_and_slow", "social_engineering"],
            p=[0.35, 0.25, 0.20, 0.20],
        )

    row = dict(
        session_id=sess_id, user_id=user.user_id, channel=user.channel,
        timestamp=ts, label=1, attack_type=attack_type,
    )

    if attack_type == "credential_theft":
        row["device_id"] = f"dev_unknown_{rng.integers(0, 999999)}"
        row["is_new_device"] = True
        off_hour = (user.typical_hour_mean + rng.choice([-1, 1]) * rng.uniform(6, 10)) % 24
        row["timestamp"] = ts.replace(hour=int(off_hour), minute=int(rng.integers(0, 60)))
        row["amount"] = round(user.txn_amount_mean * rng.uniform(2, 6), 2)
        row["recipient"] = f"acct_{rng.integers(0, 100000)}"
        row["recipient_known"] = False
        row["typing_ms"] = round(rng.uniform(600, 1200), 1)   # fast, robotic paste
        row["pin_pasted"] = True

    elif attack_type == "sim_swap":
        row["device_id"] = f"dev_unknown_{rng.integers(0, 999999)}"
        row["is_new_device"] = True
        row["timestamp"] = ts.replace(hour=int(rng.integers(0, 24)), minute=int(rng.integers(0, 60)))
        row["amount"] = round(user.txn_amount_mean * rng.uniform(4, 10), 2)  # large, immediate drain
        row["recipient"] = f"acct_{rng.integers(0, 100000)}"
        row["recipient_known"] = False
        if user.channel == "app":
            row["typing_ms"] = round(rng.uniform(500, 1000), 1)
            row["pin_pasted"] = True
        else:
            row["typing_ms"] = np.nan
            row["pin_pasted"] = False

    elif attack_type == "patient_low_and_slow":
        # stays close-ish to habits but amount creeps just under obvious limits,
        # slightly unfamiliar recipient
        row["device_id"] = user.device_id if rng.random() < 0.4 else f"dev_unknown_{rng.integers(0, 999999)}"
        row["is_new_device"] = row["device_id"] != user.device_id
        row["timestamp"] = ts.replace(hour=int(np.clip(rng.normal(user.typical_hour_mean, user.typical_hour_std * 2), 0, 23)))
        row["amount"] = round(user.txn_amount_mean * rng.uniform(1.3, 1.9), 2)  # elevated but not extreme
        row["recipient"] = f"acct_{rng.integers(0, 100000)}"
        row["recipient_known"] = False
        if user.channel == "app":
            row["typing_ms"] = round(rng.normal(user.typing_ms_mean * 0.9, user.typing_ms_std), 1)
            row["pin_pasted"] = rng.random() < 0.3
        else:
            row["typing_ms"] = np.nan
            row["pin_pasted"] = False

    else:  # social_engineering — the hard case, deliberately subtle
        row["device_id"] = user.device_id       # genuine device
        row["is_new_device"] = False
        row["timestamp"] = ts.replace(hour=int(np.clip(rng.normal(user.typical_hour_mean, user.typical_hour_std), 0, 23)))
        row["amount"] = round(user.txn_amount_mean * rng.uniform(3, 8), 2)  # unusually large for THIS user
        row["recipient"] = f"acct_{rng.integers(0, 100000)}"
        row["recipient_known"] = False
        if user.channel == "app":
            row["typing_ms"] = round(rng.normal(user.typing_ms_mean * 1.4, user.typing_ms_std), 1)  # hesitant
            row["pin_pasted"] = False
        else:
            row["typing_ms"] = np.nan
            row["pin_pasted"] = False

    return row


# --------------------------------------------------------------------------
# Main generation loop
# --------------------------------------------------------------------------

def generate_dataset(n_users: int, n_sessions: int, attack_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = make_users(n_users, rng)
    start = datetime(2026, 1, 1)

    rows = []
    sess_id = 0
    for _ in range(n_sessions):
        user = users[rng.integers(0, n_users)]
        ts = start + timedelta(
            days=int(rng.integers(0, 180)),
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
        )
        if rng.random() < attack_rate:
            rows.append(gen_attack_session(user, sess_id, ts, rng))
        else:
            rows.append(gen_normal_session(user, sess_id, ts, rng))
        sess_id += 1

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_users", type=int, default=2000)
    p.add_argument("--n_sessions", type=int, default=50000)
    p.add_argument("--attack_rate", type=float, default=0.012,  # ~1.2%, realistic severe imbalance
                    help="fraction of sessions that are attacks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="../data/sessions.csv")
    args = p.parse_args()

    df = generate_dataset(args.n_users, args.n_sessions, args.attack_rate, args.seed)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} sessions for {args.n_users} users")
    print(f"Attack rate: {df['label'].mean():.4%}")
    print(f"Channel split:\n{df['channel'].value_counts(normalize=True)}")
    print(f"Attack type breakdown:\n{df[df.label==1]['attack_type'].value_counts()}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()