"""Account-level risk models — final output of the pipeline."""
from pydantic import BaseModel
from typing import Optional
from enum import Enum
from models.signals import UsageSignal, SupportSignal, NpsSignal, ChangelogSignal, CsmSignal


class RiskTier(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class AccountRecord(BaseModel):
    """Enriched account with all signals attached."""
    account_id: str
    account_name: str
    arr: float
    contract_end_date: str
    plan_tier: str
    industry: str
    csm_name: str
    region: str
    days_to_renewal: int
    in_90_day_window: bool
    usage: Optional[UsageSignal] = None
    support: Optional[SupportSignal] = None
    nps: Optional[NpsSignal] = None
    changelog: Optional[ChangelogSignal] = None
    csm: Optional[CsmSignal] = None


class RiskScore(BaseModel):
    """Weighted risk score with contributing signal breakdown."""
    account_id: str
    raw_score: float
    tier: RiskTier
    contributing_signals: list[str]
    top_signal: str


class RiskReport(BaseModel):
    """Final deliverable per account — score + LLM explanation + action."""
    account_id: str
    account_name: str
    arr: float
    days_to_renewal: int
    tier: RiskTier
    raw_score: float
    top_signal: str
    contributing_signals: list[str]
    explanation: str
    recommended_action: str
    csm_name: str
    plan_tier: str
    industry: str
