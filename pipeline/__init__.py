"""
Main pipeline runner — orchestrates all stages end to end.
Call run_pipeline() to get a list of RiskReport for accounts in renewal window.
"""
import logging
import pandas as pd
from datetime import date, timedelta

from pipeline.ingestion.loader import (
    load_accounts, load_usage_metrics, load_support_tickets,
    load_nps_responses, load_csm_notes,
)
from pipeline.signals.usage_signals import compute_usage_signals
from pipeline.signals.support_signals import compute_support_signals
from pipeline.signals.nps_signals import compute_nps_signals
from pipeline.signals.changelog_signals import compute_changelog_signals
from pipeline.llm.csm_extractor import extract_csm_signals
from pipeline.scoring.risk_scorer import score_account
from pipeline.scoring.weight_config import RENEWAL_WINDOW_DAYS
from pipeline.llm.risk_explainer import generate_risk_report
from models.account import AccountRecord, RiskReport, RiskTier

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


from typing import Callable

def run_pipeline(
    explain_all: bool = False,
    stage_callback: Callable[[str, float], None] | None = None,
) -> list[RiskReport]:
    """
    Run full renewal risk pipeline.

    Args:
        explain_all: If True, generate LLM explanations for all tiers.
                     If False (default), only High and Medium get explanations.
    Returns:
        List of RiskReport sorted by risk score descending.
    """
    def _cb(msg: str, pct: float) -> None:
        if stage_callback:
            stage_callback(msg, pct)

    logger.info("=== STAGE 1: Loading data ===")
    _cb("📥 Loading 5 data sources…", 0.05)
    accounts_df = load_accounts()
    usage_df = load_usage_metrics()
    tickets_df = load_support_tickets()
    nps_df = load_nps_responses()
    csm_notes = load_csm_notes()

    today = date.today()
    cutoff = today + timedelta(days=RENEWAL_WINDOW_DAYS)

    today_ts = pd.Timestamp(today)
    cutoff_ts = pd.Timestamp(cutoff)

    accounts_df["days_to_renewal"] = (
        accounts_df["contract_end_date"] - today_ts
    ).dt.days

    in_window = accounts_df[
        (accounts_df["contract_end_date"] >= today_ts)
        & (accounts_df["contract_end_date"] <= cutoff_ts)
    ].copy()
    logger.info(
        "%d accounts renewing in next %d days", len(in_window), RENEWAL_WINDOW_DAYS
    )

    logger.info("=== STAGE 2: Computing deterministic signals ===")
    _cb("⚡ Computing usage trend signals…", 0.20)
    usage_signals = compute_usage_signals(usage_df)
    _cb("🎫 Computing support ticket signals…", 0.30)
    support_signals = compute_support_signals(tickets_df)
    _cb("⭐ Computing NPS & changelog signals…", 0.40)
    nps_signals = compute_nps_signals(nps_df)
    changelog_signals = compute_changelog_signals(usage_signals)

    logger.info("=== STAGE 3: LLM CSM extraction ===")
    _cb(f"🤖 Extracting CSM note signals with LLM ({len(in_window)} accounts)…", 0.55)
    csm_signals = extract_csm_signals(csm_notes, in_window)

    logger.info("=== STAGE 4: Scoring + LLM explanation ===")
    _cb("📊 Scoring all accounts…", 0.75)
    reports: list[RiskReport] = []

    for _, row in in_window.iterrows():
        aid = str(row["account_id"])
        account = AccountRecord(
            account_id=aid,
            account_name=str(row["account_name"]),
            arr=float(row["arr"]),
            contract_end_date=str(row["contract_end_date"].date()),
            plan_tier=str(row["plan_tier"]),
            industry=str(row["industry"]),
            csm_name=str(row["csm_name"]),
            region=str(row["region"]),
            days_to_renewal=int(row["days_to_renewal"]),
            in_90_day_window=True,
            usage=usage_signals.get(aid),
            support=support_signals.get(aid),
            nps=nps_signals.get(aid),
            changelog=changelog_signals.get(aid),
            csm=csm_signals.get(aid),
        )
        score = score_account(account)

        if explain_all or score.tier in (RiskTier.HIGH, RiskTier.MEDIUM):
            logger.info(
                "Generating explanation for %s (%s)",
                account.account_name,
                score.tier.value,
            )
            _cb(f"✍️ Writing briefing for {account.account_name} ({score.tier.value})…", 0.80)
            report = generate_risk_report(account, score)
        else:
            report = RiskReport(
                account_id=account.account_id,
                account_name=account.account_name,
                arr=account.arr,
                days_to_renewal=account.days_to_renewal,
                tier=score.tier,
                raw_score=score.raw_score,
                top_signal=score.top_signal,
                contributing_signals=score.contributing_signals,
                contributing_signal_weights=score.contributing_signal_weights,
                explanation="Low risk — no significant signals detected.",
                recommended_action="Maintain regular check-in cadence.",
                csm_name=account.csm_name,
                plan_tier=account.plan_tier,
                industry=account.industry,
                nps_verbatim_translated=account.nps.verbatim_translated if account.nps else None,
                csm_confidence=account.csm.confidence if account.csm else None,
            )
        reports.append(report)

    reports.sort(key=lambda r: r.raw_score, reverse=True)
    logger.info("=== PIPELINE COMPLETE: %d reports generated ===", len(reports))
    _cb(f"✅ Done — {len(reports)} accounts analysed", 1.0)
    return reports
