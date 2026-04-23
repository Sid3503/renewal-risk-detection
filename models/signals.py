"""All signal models — typed data passed between pipeline stages."""
from pydantic import BaseModel, Field
from typing import Optional


class UsageSignal(BaseModel):
    """Computed from usage_metrics.csv — trend and drop detection."""
    account_id: str
    latest_active_users: int
    mom_change_pct: float
    three_month_slope: float
    usage_drop_flag: bool
    near_zero_usage: bool
    sdk_version: str


class SupportSignal(BaseModel):
    """Computed from support_tickets.csv."""
    account_id: str
    total_tickets: int
    p1_count: int
    open_tickets: int
    escalated_tickets: int
    avg_resolution_hours: float
    has_unresolved_p1: bool


class NpsSignal(BaseModel):
    """Computed from nps_responses.csv."""
    account_id: str
    score: int
    category: str
    verbatim: Optional[str] = None
    competitor_mentioned: bool


class ChangelogSignal(BaseModel):
    """Cross-reference of account SDK version against changelog deprecations."""
    account_id: str
    sdk_version: str
    is_deprecated: bool
    deprecation_deadline: Optional[str] = None
    days_to_deadline: Optional[int] = None
    affected_features: list[str] = Field(default_factory=list)


class CsmSignal(BaseModel):
    """LLM-extracted structured signal from csm_notes.txt per account."""
    account_id: str
    account_name_in_notes: str
    competitor_mentioned: bool
    budget_cut_mentioned: bool
    exec_escalation: bool
    migration_risk: bool
    renewal_threatened: bool
    missed_meetings: int
    positive_signal: bool
    raw_summary: str
    confidence: float
