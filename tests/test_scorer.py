"""Tests for deterministic risk scorer."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models.account import AccountRecord, RiskTier
from models.signals import UsageSignal, SupportSignal, NpsSignal, ChangelogSignal, CsmSignal
from pipeline.scoring.risk_scorer import score_account, _tier_from_score
from pipeline.scoring.weight_config import SIGNAL_WEIGHTS, TIER_THRESHOLDS


def _base_account(**overrides) -> AccountRecord:
    """Build a minimal AccountRecord with no signals attached."""
    defaults = dict(
        account_id="9999",
        account_name="Test Account",
        arr=50000.0,
        contract_end_date="2026-06-01",
        plan_tier="Growth",
        industry="Technology",
        csm_name="Test CSM",
        region="NA-West",
        days_to_renewal=45,
        in_90_day_window=True,
    )
    defaults.update(overrides)
    return AccountRecord(**defaults)


def _usage(drop=False, near_zero=False, slope=0.0, sdk="v4.2.0") -> UsageSignal:
    return UsageSignal(
        account_id="9999",
        latest_active_users=1 if near_zero else 50,
        mom_change_pct=-40.0 if drop else 0.0,
        three_month_slope=slope,
        usage_drop_flag=drop,
        near_zero_usage=near_zero,
        sdk_version=sdk,
    )


def _support(p1=0, unresolved_p1=False, escalated=0) -> SupportSignal:
    return SupportSignal(
        account_id="9999",
        total_tickets=p1 + escalated,
        p1_count=p1,
        open_tickets=0,
        escalated_tickets=escalated,
        avg_resolution_hours=24.0,
        has_unresolved_p1=unresolved_p1,
    )


def _nps(score: int, competitor: bool = False) -> NpsSignal:
    return NpsSignal(
        account_id="9999",
        score=score,
        category="detractor" if score <= 6 else "passive",
        verbatim="test",
        competitor_mentioned=competitor,
    )


def _changelog(deprecated: bool = False, days_left: int = 60) -> ChangelogSignal:
    return ChangelogSignal(
        account_id="9999",
        sdk_version="v3.2.0" if deprecated else "v4.2.0",
        is_deprecated=deprecated,
        deprecation_deadline="2026-04-30" if deprecated else None,
        days_to_deadline=days_left if deprecated else None,
        affected_features=[] if not deprecated else ["REST API v2"],
    )


def _csm(
    competitor=False,
    budget_cut=False,
    exec_esc=False,
    migration=False,
    threatened=False,
    renewal_threatened=False,   # alias so both call styles work
    missed=0,
    confidence=0.9,
) -> CsmSignal:
    return CsmSignal(
        account_id="9999",
        account_name_in_notes="Test Account",
        competitor_mentioned=competitor,
        budget_cut_mentioned=budget_cut,
        exec_escalation=exec_esc,
        migration_risk=migration,
        renewal_threatened=threatened or renewal_threatened,
        missed_meetings=missed,
        positive_signal=False,
        raw_summary="Test summary.",
        confidence=confidence,
    )


# --- Tier thresholds ---

def test_tier_high():
    assert _tier_from_score(TIER_THRESHOLDS["High"]) == RiskTier.HIGH
    assert _tier_from_score(10.0) == RiskTier.HIGH


def test_tier_medium():
    assert _tier_from_score(TIER_THRESHOLDS["Medium"]) == RiskTier.MEDIUM
    assert _tier_from_score(5.0) == RiskTier.MEDIUM


def test_tier_low():
    assert _tier_from_score(0.0) == RiskTier.LOW
    assert _tier_from_score(3.4) == RiskTier.LOW


# --- Individual signal contributions ---

def test_usage_drop_adds_weight():
    account = _base_account(usage=_usage(drop=True))
    result = score_account(account)
    assert result.raw_score == SIGNAL_WEIGHTS["usage_drop_flag"]


def test_near_zero_usage_adds_weight():
    account = _base_account(usage=_usage(near_zero=True))
    result = score_account(account)
    assert result.raw_score == SIGNAL_WEIGHTS["near_zero_usage"]


def test_renewal_threatened_is_highest_single_csm_signal():
    account = _base_account(csm=_csm(threatened=True))
    result = score_account(account)
    assert result.raw_score == SIGNAL_WEIGHTS["renewal_threatened"]
    assert "churn" in result.top_signal.lower() or "threat" in result.top_signal.lower() or "ultimatum" in result.top_signal.lower()


def test_p1_tickets_capped_at_3():
    # 5 P1 tickets should contribute same as 3
    s5 = _support(p1=5)
    s3 = _support(p1=3)
    a5 = _base_account(support=s5)
    a3 = _base_account(support=s3)
    assert score_account(a5).raw_score == score_account(a3).raw_score


def test_missed_meetings_capped_at_2():
    c2 = _csm(missed=2)
    c5 = _csm(missed=5)
    a2 = _base_account(csm=c2)
    a5 = _base_account(csm=c5)
    assert score_account(a2).raw_score == score_account(a5).raw_score


def test_low_confidence_csm_halves_weight():
    high_conf = _base_account(csm=_csm(budget_cut=True, confidence=0.9))
    low_conf = _base_account(csm=_csm(budget_cut=True, confidence=0.4))
    assert score_account(high_conf).raw_score == pytest.approx(
        score_account(low_conf).raw_score * 2, rel=1e-3
    )


def test_deprecated_sdk_adds_weight():
    account = _base_account(
        usage=_usage(sdk="v3.2.0"),
        changelog=_changelog(deprecated=True, days_left=60),
    )
    result = score_account(account)
    assert SIGNAL_WEIGHTS["deprecated_sdk"] in [result.raw_score] or result.raw_score >= SIGNAL_WEIGHTS["deprecated_sdk"]


def test_sdk_deadline_30d_adds_extra_weight():
    account_far = _base_account(
        usage=_usage(sdk="v3.2.0"),
        changelog=_changelog(deprecated=True, days_left=60),
    )
    account_near = _base_account(
        usage=_usage(sdk="v3.2.0"),
        changelog=_changelog(deprecated=True, days_left=15),
    )
    assert score_account(account_near).raw_score > score_account(account_far).raw_score


def test_multiple_signals_compound():
    account = _base_account(
        usage=_usage(drop=True),
        support=_support(p1=2, unresolved_p1=True),
        nps=_nps(4),
        changelog=_changelog(deprecated=True),
        csm=_csm(renewal_threatened=True),
    )
    result = score_account(account)
    assert result.tier == RiskTier.HIGH
    assert result.raw_score > TIER_THRESHOLDS["High"]
    assert len(result.contributing_signals) >= 4


def test_zero_signals_is_low_risk():
    account = _base_account()
    result = score_account(account)
    assert result.tier == RiskTier.LOW
    assert result.raw_score == 0.0


def test_top_signal_is_highest_weighted():
    """Top signal should be the one with the highest weight contribution."""
    account = _base_account(
        csm=_csm(renewal_threatened=True, budget_cut=True),
    )
    result = score_account(account)
    # renewal_threatened (4.0) > budget_cut (2.0), so top signal should reflect that
    assert "churn" in result.top_signal.lower() or "threat" in result.top_signal.lower() or "ultimatum" in result.top_signal.lower()


def test_competitor_mentioned_not_double_counted():
    """Competitor in both NPS and CSM notes should only fire once per source."""
    account = _base_account(
        nps=_nps(5, competitor=True),
        csm=_csm(competitor=True),
    )
    result = score_account(account)
    competitor_signals = [s for s in result.contributing_signals if "competitor" in s.lower()]
    # NPS fires once, CSM should not double-fire since NPS already captured it
    assert len(competitor_signals) == 1
