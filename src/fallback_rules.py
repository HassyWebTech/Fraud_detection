from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    ALLOW = "allow"
    SOFT_CHALLENGE = "soft_challenge"  
    HOLD_FOR_RECONCILIATION = "hold"     # queue, notify customer, resolve once back online


@dataclass
class CachedUserBaseline:
    user_id: int
    last_known_device: str
    typical_amount_band_low: float
    typical_amount_band_high: float
    known_recipients: set
    typical_hour_range: tuple  # (start, end)
    synced_at: str  # ISO date of last successful sync, for staleness checks


RULE_WEIGHTS = {
    "device_mismatch": 3,
    "amount_far_above_band": 3,
    "amount_moderately_above_band": 1,
    "unknown_recipient": 2,
    "unusual_hour": 1,
    "stale_cache": 1,  
}

# Tier thresholds
TIER_SOFT_CHALLENGE = 3
TIER_HOLD = 6


def score_offline(session: dict, cache: CachedUserBaseline, cache_age_days: int = 0) -> dict:
    triggered = []
    score = 0

    if cache is None:
        return dict(action=Action.SOFT_CHALLENGE,
                    reasons=["no cached profile available for this account — extra verification required"],
                    score=None, degraded_mode=True)

    if session["device_id"] != cache.last_known_device:
        score += RULE_WEIGHTS["device_mismatch"]
        triggered.append("transaction from an unrecognized device")

    if session["amount"] > cache.typical_amount_band_high * 3:
        score += RULE_WEIGHTS["amount_far_above_band"]
        triggered.append("amount far above your typical range")
    elif session["amount"] > cache.typical_amount_band_high * 1.5:
        score += RULE_WEIGHTS["amount_moderately_above_band"]
        triggered.append("amount somewhat above your typical range")

    if session["recipient"] not in cache.known_recipients:
        score += RULE_WEIGHTS["unknown_recipient"]
        triggered.append("recipient not in your known contacts")

    h_start, h_end = cache.typical_hour_range
    if not (h_start <= session["hour"] <= h_end):
        score += RULE_WEIGHTS["unusual_hour"]
        triggered.append("outside your usual transaction hours")

    if cache_age_days > 14:
        score += RULE_WEIGHTS["stale_cache"]
        triggered.append("account profile has not synced recently, so this check is more cautious than usual")

    if score >= TIER_HOLD:
        action = Action.HOLD_FOR_RECONCILIATION
    elif score >= TIER_SOFT_CHALLENGE:
        action = Action.SOFT_CHALLENGE
    else:
        action = Action.ALLOW

    return dict(action=action, reasons=triggered, score=score, degraded_mode=True)


def reconciliation_note(offline_decision: dict) -> str:
    return (
        f"Session scored in DEGRADED MODE (action={offline_decision['action'].value}). "
        f"Queued for automatic rescoring by full model on reconnect; "
        f"customer notified if final decision differs from the offline one."
    )


def demo():
    cache = CachedUserBaseline(
        user_id=1, last_known_device="dev_1_500123",
        typical_amount_band_low=1000, typical_amount_band_high=15000,
        known_recipients={"acct_222", "acct_888"},
        typical_hour_range=(7, 22), synced_at="2026-01-10",
    )

    scenarios = [
        dict(name="Normal transfer", session=dict(
            device_id="dev_1_500123", amount=5000, recipient="acct_222", hour=14)),
        dict(name="New device + large amount + unknown recipient (likely ATO)", session=dict(
            device_id="dev_unknown_999", amount=80000, recipient="acct_555", hour=3)),
        dict(name="Same device, slightly high amount, new recipient", session=dict(
            device_id="dev_1_500123", amount=22000, recipient="acct_777", hour=15)),
    ]

    print("=== Offline/Degraded-Mode Fallback Demo (no model, no network) ===\n")
    for s in scenarios:
        result = score_offline(s["session"], cache, cache_age_days=3)
        print(f"{s['name']}")
        print(f"  -> Action: {result['action'].value} (score={result['score']})")
        print(f"  -> Reasons: {result['reasons']}")
        print(f"  -> {reconciliation_note(result)}\n")


if __name__ == "__main__":
    demo()