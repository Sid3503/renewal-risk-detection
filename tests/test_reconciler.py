"""Tests for account name reconciliation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from pipeline.ingestion.reconciler import build_name_lookup, reconcile_name, _normalize


def _make_accounts(names: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a minimal accounts DataFrame for testing."""
    return pd.DataFrame(
        [{"account_id": aid, "account_name": name} for aid, name in names]
    )


def test_normalize_strips_sic():
    assert "(sic)" not in _normalize("BritePath Solutions (sic)")


def test_normalize_lowercases_and_strips_punctuation():
    result = _normalize("NovaTech Industries!")
    assert result == "novatech industries"


def test_exact_match():
    df = _make_accounts([("1001", "BrightPath Solutions")])
    lookup = build_name_lookup(df)
    aid, conf = reconcile_name("BrightPath Solutions", lookup)
    assert aid == "1001"
    assert conf == 1.0


def test_fuzzy_match_typo():
    """BritePath Solutions (sic) should match BrightPath Solutions."""
    df = _make_accounts([("1001", "BrightPath Solutions")])
    lookup = build_name_lookup(df)
    aid, conf = reconcile_name("BritePath Solutions (sic)", lookup)
    assert aid == "1001"
    assert conf > 0.7


def test_fuzzy_match_pinacle():
    """Pinacle Media should match Pinnacle Media Group."""
    df = _make_accounts([("1004", "Pinnacle Media Group")])
    lookup = build_name_lookup(df)
    aid, conf = reconcile_name("Pinacle Media", lookup)
    assert aid == "1004"
    assert conf > 0.7


def test_fuzzy_match_thunderbolt_moters():
    """Thunderbolt Moters should match Thunderbolt Motors."""
    df = _make_accounts([("1099", "Thunderbolt Motors")])
    lookup = build_name_lookup(df)
    aid, conf = reconcile_name("Thunderbolt Moters", lookup)
    assert aid == "1099"
    assert conf > 0.7


def test_no_match_returns_none():
    df = _make_accounts([("1001", "BrightPath Solutions")])
    lookup = build_name_lookup(df)
    aid, conf = reconcile_name("Completely Unrelated Corp XYZ", lookup)
    assert aid is None
    assert conf == 0.0


def test_build_name_lookup_normalizes_keys():
    df = _make_accounts([("42", "ACME Corp.")])
    lookup = build_name_lookup(df)
    assert "acme corp" in lookup
    assert lookup["acme corp"] == "42"
