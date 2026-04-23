"""
Account name reconciliation using fuzzy matching.

Problem: csm_notes.txt references accounts by name with typos.
  'BritePath Solutions (sic)' -> account_id for 'BrightPath Solutions'
  'Pinacle Media' -> account_id for 'Pinnacle Media Group'
  'Thunderbolt Moters' -> account_id for 'Thunderbolt Motors'

Strategy:
  Build a lookup dict: normalized_name -> account_id
  For each unmatched name, use rapidfuzz to find best match above threshold
  Return matched account_id or None if confidence too low
"""
import re
import logging
import pandas as pd
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 75


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, remove noise words like (sic)."""
    name = name.lower()
    name = re.sub(r"\(sic\)", "", name)
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def build_name_lookup(accounts_df: pd.DataFrame) -> dict[str, str]:
    """Build normalized_name -> account_id lookup from accounts master."""
    lookup: dict[str, str] = {}
    for _, row in accounts_df.iterrows():
        normalized = _normalize(str(row["account_name"]))
        lookup[normalized] = str(row["account_id"])
    return lookup


def reconcile_name(
    name: str,
    lookup: dict[str, str],
    threshold: int = MATCH_THRESHOLD,
) -> tuple[str | None, float]:
    """
    Return (account_id, confidence) for a given account name string.
    Returns (None, 0.0) if no match above threshold found.
    """
    normalized = _normalize(name)

    if normalized in lookup:
        return lookup[normalized], 1.0

    result = process.extractOne(
        normalized,
        list(lookup.keys()),
        scorer=fuzz.token_sort_ratio,
    )
    if result and result[1] >= threshold:
        matched_name, score, _ = result
        account_id = lookup[matched_name]
        logger.debug("Fuzzy match: '%s' -> '%s' (score=%d)", name, matched_name, score)
        return account_id, score / 100.0

    logger.warning("No match for account name: '%s'", name)
    return None, 0.0
