import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest  # type: ignore[import-not-found]

from features import circular_hour_distance, build_features
from fallback_rules import score_offline, CachedUserBaseline, Action



def test_circular_hour_distance_same_hour():
    assert circular_hour_distance(14, 14) == 0

def test_circular_hour_distance_wraparound():
    # 23:00 and 01:00 are only 2 hours apart on a clock, not 22
    assert circular_hour_distance(23, 1) == 2

def test_circular_hour_distance_opposite():
    assert circular_hour_distance(0, 12) == 12


def test_features_cold_start_flagged():
    """A user's first few sessions must be marked cold_start — there's
    no real history yet to compute a trustworthy personal baseline."""
    raw = pd.DataFrame([
        dict(session_id=i, user_id=1, channel="app",
             timestamp=f"2026-01-{i+1:02d} 12:00:00",
             device_id="dev_1", is_new_device=False, amount=5000,
             recipient="acct_1", recipient_known=True,
             typing_ms=3000, pin_pasted=False, label=0, attack_type="none")
        for i in range(5)
    ])
    feat = build_features(raw)
    assert feat.iloc[0]["cold_start"] == 1
    assert feat.iloc[-1]["cold_start"] == 0  # by session 5, enough history exists

def test_features_no_future_leakage():
    """A session's deviation features must not change based on sessions
    that happen AFTER it — only prior history should matter."""
    raw = pd.DataFrame([
        dict(session_id=0, user_id=1, channel="app", timestamp="2026-01-01 12:00:00",
             device_id="dev_1", is_new_device=False, amount=5000, recipient="acct_1",
             recipient_known=True, typing_ms=3000, pin_pasted=False, label=0, attack_type="none"),
        dict(session_id=1, user_id=1, channel="app", timestamp="2026-01-02 12:00:00",
             device_id="dev_1", is_new_device=False, amount=5000, recipient="acct_1",
             recipient_known=True, typing_ms=3000, pin_pasted=False, label=0, attack_type="none"),
        dict(session_id=2, user_id=1, channel="app", timestamp="2026-01-03 12:00:00",
             device_id="dev_1", is_new_device=False, amount=5000, recipient="acct_1",
             recipient_known=True, typing_ms=3000, pin_pasted=False, label=0, attack_type="none"),
        dict(session_id=3, user_id=1, channel="app", timestamp="2026-01-04 12:00:00",
             device_id="dev_1", is_new_device=False, amount=5000, recipient="acct_1",
             recipient_known=True, typing_ms=3000, pin_pasted=False, label=0, attack_type="none"),
    ])
    feat_full = build_features(raw)

    
    raw_with_future = pd.concat([raw, pd.DataFrame([
        dict(session_id=4, user_id=1, channel="app", timestamp="2026-01-05 12:00:00",
             device_id="dev_1", is_new_device=False, amount=9_000_000, recipient="acct_1",
             recipient_known=True, typing_ms=3000, pin_pasted=False, label=0, attack_type="none"),
    ])], ignore_index=True)
    feat_with_future = build_features(raw_with_future)

    # session 3's features must be IDENTICAL whether or not a future
    # session 4 exists — if they differ, the future is leaking backward.
    row_before = feat_full[feat_full.session_id == 3].iloc[0]
    row_after = feat_with_future[feat_with_future.session_id == 3].iloc[0]
    assert row_before["amount_deviation"] == row_after["amount_deviation"]




@pytest.fixture
def cache():
    return CachedUserBaseline(
        user_id=1, last_known_device="dev_known",
        typical_amount_band_low=1000, typical_amount_band_high=15000,
        known_recipients={"acct_known"}, typical_hour_range=(7, 22),
        synced_at="2026-01-01",
    )

def test_offline_normal_session_allowed(cache):
    session = dict(device_id="dev_known", amount=5000, recipient="acct_known", hour=14)
    result = score_offline(session, cache, cache_age_days=1)
    assert result["action"] == Action.ALLOW

def test_offline_obvious_takeover_held(cache):
    session = dict(device_id="dev_unknown", amount=100000, recipient="acct_unknown", hour=3)
    result = score_offline(session, cache, cache_age_days=1)
    assert result["action"] == Action.HOLD_FOR_RECONCILIATION

def test_offline_no_cache_defaults_to_challenge():
    session = dict(device_id="dev_x", amount=5000, recipient="acct_x", hour=14)
    result = score_offline(session, cache=None, cache_age_days=0)
    assert result["action"] == Action.SOFT_CHALLENGE
    assert result["degraded_mode"] is True

def test_offline_stale_cache_increases_caution(cache):
    session = dict(device_id="dev_known", amount=20000, recipient="acct_unknown", hour=14)
    fresh = score_offline(session, cache, cache_age_days=1)
    stale = score_offline(session, cache, cache_age_days=30)
    assert stale["score"] > fresh["score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])