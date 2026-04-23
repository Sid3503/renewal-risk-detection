"""Tests for deterministic signal computation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from pipeline.signals.usage_signals import compute_usage_signals
from pipeline.signals.support_signals import compute_support_signals
from pipeline.signals.nps_signals import compute_nps_signals, _nps_category, _detect_competitor
from pipeline.signals.changelog_signals import compute_changelog_signals, _is_deprecated
from models.signals import UsageSignal


def _make_usage_df(account_id: str, monthly_users: list[int], sdk: str = "v4.2.0") -> pd.DataFrame:
    """Build usage DataFrame with specified monthly active_users."""
    months = pd.date_range("2025-10-01", periods=len(monthly_users), freq="MS")
    return pd.DataFrame(
        {
            "account_id": account_id,
            "month": months,
            "active_users": monthly_users,
            "api_calls": 1000,
            "content_entries_created": 100,
            "workflows_triggered": 10,
            "sdk_version": sdk,
        }
    )


# --- Usage signals ---

def test_usage_drop_flag_triggered_on_big_mom_decline():
    df = _make_usage_df("1", [100, 100, 100, 100, 100, 60])
    signals = compute_usage_signals(df)
    assert signals["1"].usage_drop_flag is True


def test_usage_drop_flag_not_triggered_on_small_decline():
    df = _make_usage_df("1", [100, 100, 100, 100, 100, 85])
    signals = compute_usage_signals(df)
    assert signals["1"].usage_drop_flag is False


def test_near_zero_usage_flag():
    df = _make_usage_df("1", [10, 8, 5, 3, 2, 1])
    signals = compute_usage_signals(df)
    assert signals["1"].near_zero_usage is True


def test_negative_slope_detected():
    df = _make_usage_df("1", [50, 40, 30, 20, 10, 5])
    signals = compute_usage_signals(df)
    assert signals["1"].three_month_slope < 0


def test_sdk_version_captured():
    df = _make_usage_df("1", [10, 10, 10, 10, 10, 10], sdk="v3.2.0")
    signals = compute_usage_signals(df)
    assert signals["1"].sdk_version == "v3.2.0"


# --- Support signals ---

def _make_tickets_df(account_id: str, priorities: list[str], statuses: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": account_id,
            "ticket_id": [f"T{i}" for i in range(len(priorities))],
            "created_date": "2026-01-01",
            "subject": "test",
            "description": "test",
            "priority": priorities,
            "resolution_time_hours": [10.0] * len(priorities),
            "status": statuses,
        }
    )


def test_p1_count():
    df = _make_tickets_df("1", ["P1", "P1", "P2"], ["resolved", "resolved", "open"])
    signals = compute_support_signals(df)
    assert signals["1"].p1_count == 2


def test_unresolved_p1_detection():
    df = _make_tickets_df("1", ["P1", "P2"], ["open", "resolved"])
    signals = compute_support_signals(df)
    assert signals["1"].has_unresolved_p1 is True


def test_escalated_tickets():
    df = _make_tickets_df("1", ["P2", "P3"], ["escalated", "open"])
    signals = compute_support_signals(df)
    assert signals["1"].escalated_tickets == 1


# --- NPS signals ---

def test_nps_category_detractor():
    assert _nps_category(5) == "detractor"
    assert _nps_category(0) == "detractor"
    assert _nps_category(6) == "detractor"


def test_nps_category_passive():
    assert _nps_category(7) == "passive"
    assert _nps_category(8) == "passive"


def test_nps_category_promoter():
    assert _nps_category(9) == "promoter"
    assert _nps_category(10) == "promoter"


def test_competitor_detection_contentful():
    assert _detect_competitor("we are moving to Contentful next quarter") is True


def test_competitor_detection_case_insensitive():
    assert _detect_competitor("STRAPI looks promising") is True


def test_no_competitor_detected():
    assert _detect_competitor("great product, very happy") is False


# --- Changelog signals ---

def test_deprecated_sdk_v3_detected():
    assert _is_deprecated("v3.2.0") is True
    assert _is_deprecated("v3.1.0") is True


def test_current_sdk_not_deprecated():
    assert _is_deprecated("v4.2.0") is False
    assert _is_deprecated("v4.3.1") is False


def test_changelog_signals_cross_reference():
    usage_signal = UsageSignal(
        account_id="1",
        latest_active_users=10,
        mom_change_pct=0.0,
        three_month_slope=0.0,
        usage_drop_flag=False,
        near_zero_usage=False,
        sdk_version="v3.2.0",
    )
    signals = compute_changelog_signals({"1": usage_signal})
    assert signals["1"].is_deprecated is True
    assert signals["1"].deprecation_deadline == "2026-04-30"


def test_current_sdk_not_flagged_in_changelog():
    usage_signal = UsageSignal(
        account_id="2",
        latest_active_users=10,
        mom_change_pct=0.0,
        three_month_slope=0.0,
        usage_drop_flag=False,
        near_zero_usage=False,
        sdk_version="v4.3.1",
    )
    signals = compute_changelog_signals({"2": usage_signal})
    assert signals["2"].is_deprecated is False
