"""Load all 5 raw data files into typed DataFrames with basic validation."""
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def load_accounts() -> pd.DataFrame:
    """Load accounts.csv — spine of the entire pipeline."""
    df = pd.read_csv(RAW_DIR / "accounts.csv")
    df["account_id"] = df["account_id"].astype(str).str.strip()
    df["contract_end_date"] = pd.to_datetime(df["contract_end_date"], errors="coerce")
    df["arr"] = pd.to_numeric(df["arr"], errors="coerce").fillna(0)
    logger.info("Loaded %d accounts", len(df))
    return df


def load_usage_metrics() -> pd.DataFrame:
    """Load usage_metrics.csv — 6 monthly rows per account."""
    df = pd.read_csv(RAW_DIR / "usage_metrics.csv")
    df["account_id"] = df["account_id"].astype(str).str.strip()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    for col in ["api_calls", "content_entries_created", "active_users", "workflows_triggered"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["sdk_version"] = df["sdk_version"].astype(str).str.strip()
    logger.info("Loaded %d usage rows", len(df))
    return df


def load_support_tickets() -> pd.DataFrame:
    """Load support_tickets.csv — not all accounts have tickets."""
    df = pd.read_csv(RAW_DIR / "support_tickets.csv")
    df["account_id"] = df["account_id"].astype(str).str.strip()
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["priority"] = df["priority"].astype(str).str.upper().str.strip()
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    df["resolution_time_hours"] = pd.to_numeric(
        df["resolution_time_hours"], errors="coerce"
    ).fillna(0)
    logger.info("Loaded %d tickets", len(df))
    return df


def load_nps_responses() -> pd.DataFrame:
    """Load nps_responses.csv — not all accounts have NPS responses."""
    df = pd.read_csv(RAW_DIR / "nps_responses.csv")
    df["account_id"] = df["account_id"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["verbatim_comment"] = df["verbatim_comment"].fillna("").astype(str)
    logger.info("Loaded %d NPS responses", len(df))
    return df


def load_csm_notes() -> str:
    """Load csm_notes.txt as raw string — LLM handles parsing."""
    text = (RAW_DIR / "csm_notes.txt").read_text(encoding="utf-8")
    logger.info("Loaded CSM notes (%d chars)", len(text))
    return text


def load_changelog() -> str:
    """Load changelog.md as raw string."""
    text = (RAW_DIR / "changelog.md").read_text(encoding="utf-8")
    logger.info("Loaded changelog (%d chars)", len(text))
    return text
