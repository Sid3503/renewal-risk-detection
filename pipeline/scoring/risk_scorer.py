"""
Deterministic risk scoring — combines all signals into RiskScore.

Rules:
  - Each fired signal adds its weight from SIGNAL_WEIGHTS
  - CSM signals below CSM_CONFIDENCE_THRESHOLD are weighted at 50%
  - P1 tickets capped at 3 (contribution capped at 3 × 2.0 = 6.0)
  - Missed meetings capped at 2
  - Tier assigned from TIER_THRESHOLDS
"""
from models.account import AccountRecord, RiskScore, RiskTier
from pipeline.scoring.weight_config import (
    SIGNAL_WEIGHTS,
    TIER_THRESHOLDS,
    CSM_CONFIDENCE_THRESHOLD,
)


def _tier_from_score(score: float) -> RiskTier:
    """Map raw score to RiskTier using TIER_THRESHOLDS."""
    tier_map = [
        (TIER_THRESHOLDS["High"], RiskTier.HIGH),
        (TIER_THRESHOLDS["Medium"], RiskTier.MEDIUM),
    ]
    return next(
        (tier for threshold, tier in tier_map if score >= threshold), RiskTier.LOW
    )


def score_account(account: AccountRecord) -> RiskScore:
    """Compute deterministic RiskScore for a single AccountRecord."""
    score = 0.0
    fired: list[tuple[str, float]] = []

    # --- Usage signals ---
    if account.usage:
        u = account.usage
        if u.usage_drop_flag:
            w = SIGNAL_WEIGHTS["usage_drop_flag"]
            score += w
            fired.append((f"Usage dropped {abs(u.mom_change_pct):.0f}% MoM", w))
        if u.near_zero_usage:
            w = SIGNAL_WEIGHTS["near_zero_usage"]
            score += w
            fired.append((f"Near-zero active users ({u.latest_active_users})", w))
        if u.three_month_slope < -0.5:
            w = SIGNAL_WEIGHTS["negative_mom_trend"]
            score += w
            fired.append(("Declining 3-month usage trend", w))

    # --- Support signals ---
    if account.support:
        s = account.support
        if s.p1_count > 0:
            p1_contrib = min(s.p1_count, 3) * SIGNAL_WEIGHTS["p1_tickets"]
            score += p1_contrib
            fired.append((f"{s.p1_count} P1 support ticket(s)", p1_contrib))
        if s.has_unresolved_p1:
            w = SIGNAL_WEIGHTS["unresolved_p1"]
            score += w
            fired.append(("Unresolved P1 ticket", w))
        if s.escalated_tickets > 0:
            w = SIGNAL_WEIGHTS["escalated_tickets"]
            score += w
            fired.append((f"{s.escalated_tickets} escalated ticket(s)", w))

    # --- NPS signals ---
    if account.nps:
        n = account.nps
        if n.score <= 6:
            w = SIGNAL_WEIGHTS["low_nps"]
            score += w
            fired.append((f"Low NPS score ({n.score})", w))
        if n.score <= 3:
            w = SIGNAL_WEIGHTS["detractor_nps"]
            score += w
            fired.append(("Strong detractor NPS (score ≤ 3)", w))
        if n.competitor_mentioned:
            w = SIGNAL_WEIGHTS["competitor_mentioned"]
            score += w
            fired.append(("Competitor mentioned in NPS verbatim", w))

    # --- Changelog / SDK signals ---
    if account.changelog and account.changelog.is_deprecated:
        w = SIGNAL_WEIGHTS["deprecated_sdk"]
        score += w
        fired.append((f"On deprecated SDK {account.changelog.sdk_version}", w))
        if (
            account.changelog.days_to_deadline is not None
            and account.changelog.days_to_deadline <= 30
        ):
            w2 = SIGNAL_WEIGHTS["sdk_deadline_30d"]
            score += w2
            fired.append(
                (f"SDK deadline in {account.changelog.days_to_deadline} days", w2)
            )

    # --- CSM signals (confidence-discounted) ---
    if account.csm:
        c = account.csm
        conf_mult = 1.0 if c.confidence >= CSM_CONFIDENCE_THRESHOLD else 0.5

        nps_competitor_already_fired = bool(
            account.nps and account.nps.competitor_mentioned
        )

        csm_checks: list[tuple[bool, str, str]] = [
            (c.budget_cut_mentioned, "budget_cut", "Budget cut mentioned in CSM notes"),
            (c.exec_escalation, "exec_escalation", "Executive escalation (CTO/VP/CRO on call)"),
            (c.migration_risk, "migration_risk", "Evaluating competitors or building internal solution"),
            (c.renewal_threatened, "renewal_threatened", "Explicit churn threat or discount ultimatum"),
            (
                c.competitor_mentioned and not nps_competitor_already_fired,
                "competitor_mentioned",
                "Competitor mentioned in CSM notes",
            ),
        ]
        for condition, weight_key, label in csm_checks:
            if condition:
                w = SIGNAL_WEIGHTS[weight_key] * conf_mult
                score += w
                fired.append((label, w))

        if c.missed_meetings > 0:
            miss_contrib = (
                min(c.missed_meetings, 2) * SIGNAL_WEIGHTS["missed_meetings"] * conf_mult
            )
            score += miss_contrib
            fired.append((f"{c.missed_meetings} missed meeting(s)", miss_contrib))

    fired_sorted = sorted(fired, key=lambda x: x[1], reverse=True)
    top_signal = fired_sorted[0][0] if fired_sorted else "No significant signals"
    contributing = [label for label, _ in fired_sorted]

    return RiskScore(
        account_id=account.account_id,
        raw_score=round(score, 2),
        tier=_tier_from_score(score),
        contributing_signals=contributing,
        top_signal=top_signal,
    )
