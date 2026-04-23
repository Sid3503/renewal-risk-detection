"""Compute NPS-based risk signals from nps_responses.csv."""
import pandas as pd
from models.signals import NpsSignal
from pipeline.scoring.weight_config import COMPETITOR_SIGNALS


def _detect_competitor(text: str) -> bool:
    """Check if any competitor name appears in verbatim text."""
    text_lower = text.lower()
    return any(c in text_lower for c in COMPETITOR_SIGNALS)


def _nps_category(score: int) -> str:
    """Map NPS score to standard category."""
    category_map = [
        (range(0, 7), "detractor"),
        (range(7, 9), "passive"),
        (range(9, 11), "promoter"),
    ]
    return next((cat for r, cat in category_map if score in r), "unknown")


def compute_nps_signals(nps_df: pd.DataFrame) -> dict[str, NpsSignal]:
    """Compute NpsSignal for every account_id with an NPS response."""
    signals: dict[str, NpsSignal] = {}

    for _, row in nps_df.iterrows():
        account_id = str(row["account_id"])
        score = int(row["score"]) if not pd.isna(row["score"]) else 7
        verbatim = str(row.get("verbatim_comment", ""))

        signals[account_id] = NpsSignal(
            account_id=account_id,
            score=score,
            category=_nps_category(score),
            verbatim=verbatim if verbatim else None,
            competitor_mentioned=_detect_competitor(verbatim),
        )

    return signals
