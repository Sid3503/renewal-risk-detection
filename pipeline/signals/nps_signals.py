"""Compute NPS-based risk signals from nps_responses.csv."""
import logging
import pandas as pd
from pydantic import BaseModel
from models.signals import NpsSignal
from pipeline.scoring.weight_config import COMPETITOR_SIGNALS

logger = logging.getLogger(__name__)


class _Translation(BaseModel):
    translated_text: str


def _has_non_ascii(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def _translate_to_english(text: str) -> str:
    """Translate non-English verbatim to English using the LLM. Returns original on failure."""
    try:
        from pipeline.llm.llm_client import call_llm
    except ImportError:
        return text

    result = call_llm(
        user_prompt=f"Translate the following customer feedback to English. Return only the translation, nothing else.\n\nText: {text}",
        output_schema=_Translation,
        system_prompt="You are a translator. Translate the input to English and return ONLY valid JSON with a single field 'translated_text'.",
        max_retries=2,
        temperature=0.0,
    )
    if result is not None:
        logger.info("Translated non-English NPS verbatim (%d chars)", len(text))
        return result.translated_text
    logger.warning("Translation failed — using original non-English verbatim for competitor detection")
    return text


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

        # Translate non-English verbatims before competitor keyword scan
        text_for_detection = verbatim
        translated: str | None = None
        if verbatim and _has_non_ascii(verbatim):
            translated = _translate_to_english(verbatim)
            text_for_detection = translated

        signals[account_id] = NpsSignal(
            account_id=account_id,
            score=score,
            category=_nps_category(score),
            verbatim=verbatim if verbatim else None,
            verbatim_translated=translated,
            competitor_mentioned=_detect_competitor(text_for_detection),
        )

    return signals
